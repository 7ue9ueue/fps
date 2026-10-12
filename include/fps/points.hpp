// Values at points, interpolation and the Newton basis modulo P = 998244353 (DESIGN.md, Functions:
// Evaluation and interpolation), on any span of points.
//
//   fps::evaluate(f, points)              // f(points[i]) for each i: points.size() values
//   fps::interpolate<P>(points, values)   // the f with f.size() = m and f(points[i]) = values[i]
//   fps::to_newton_basis(f, points)       // c with f = sum_k c[k] prod_(i < k) (x - points[i])
//
// Points and values are canonical, in [0, P); one >= P throws std::domain_error. 2^25 or more
// points (to_newton_basis: f.size() >= 2^25) throw std::length_error, before any work: the tree's
// root would pass the NTT's 2^25 coefficients. evaluate takes
// any points, repeats too. interpolate needs as many values as points, all points distinct (else
// std::domain_error: a repeat makes some M'(x_i) zero, below), and returns m = points.size()
// coefficients (an empty polynomial for m = 0). to_newton_basis returns f.size() = n coefficients
// and reads the first n - 1 points (the last basis polynomial's); fewer throws std::domain_error.
//
// Paths: product trees, plainly. A tree over the points: node [l, r) holds
// T = prod_(l <= i < r) (x - x_i) and splits at (l + r) / 2; a node of at most 32 points
// (kLeafPoints) is a leaf, its T multiplied out by a loop. The other nodes' T are the products of
// their children's (f * g: small::multiply or ntt::multiply, fps/polynomial.hpp).
// - evaluate: down the tree, f mod T at each node (fps/division.hpp's remainder; skipped while f
//   is shorter than T), then at a leaf Horner's rule on the remainder, its points side by side.
// - interpolate: c_i = values[i] / M'(x_i) for M the root's T (M' evaluated as above, one
//   inversion for all), then up the tree: a leaf's sum of c_i T / (x - x_i) by a loop, a node's
//   f_left T_right + f_right T_left (two products).
// - to_newton_basis: down the tree, g = A + T_left B with A = g mod T_left and B = g / T_left
//   (one divmod); A goes to the left child, B to the right; at a leaf, synthetic division by
//   x - x_i, point by point.
// O(M(m) log m) for m points (M a product's cost), as lc-opt's, but with a Newton inverse in each
// division and every node's product kept. lc-opt's tuned versions (transposed evaluation, kept
// transforms) are backend issue #104, on backend/product_tree (whose product() returns only the
// root, not the kept nodes these need). At 2^17 points on the timed targets, ns per point: evaluate
// 300-720, interpolate 440-990, to_newton_basis 190-460; lc-opt's Newton basis is about 3.7x
// faster on Zen 3, and the descent's divisions take about 80% of evaluate (frontend/notes.md).
//
// Memory: the tree keeps every node's T, about m log2(m / 32) words, for the call; the results are
// new polynomials and vectors.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fps/division.hpp"
#include "fps/modular.hpp"
#include "fps/polynomial.hpp"
#include "fps/series.hpp"

namespace fps {

namespace detail::points {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Span = std::span<const u32>;
using Out = std::span<u32>;

inline constexpr std::size_t kLeafPoints = 32;

inline void check_canonical(Span a, u32 p, const char* what) {
    if (std::ranges::any_of(a, [p](u32 v) { return v >= p; })) throw std::domain_error(what);
}

// The product tree over the points x: node id covers [l, r), its children 2 id over [l, mid) and
// 2 id + 1 over [mid, r), mid = (l + r) / 2; the root is node 1 over [0, x.size()).
template <u32 P>
class Tree {
public:
    explicit Tree(Span x) : x_(x) {
        if (x.size() >= std::size_t(1) << ntt::kMaxLog)
            throw std::length_error("fps: 2^25 or more points (the tree's root passes 2^25 coefficients)");
        node_.resize(capacity(x.size()));
        if (!x.empty()) build(1, 0, x.size());
    }

    static bool leaf(std::size_t l, std::size_t r) { return r - l <= kLeafPoints; }
    static std::size_t middle(std::size_t l, std::size_t r) { return (l + r) / 2; }

    Span points() const { return x_; }
    const Polynomial<P>& operator[](std::size_t id) const { return node_[id]; }

private:
    // Node ids at depth t are below 2^(t + 1); a segment at depth t has at most ceil(m / 2^t) points.
    static std::size_t capacity(std::size_t m) {
        std::size_t c = 2;
        for (std::size_t s = m; s > kLeafPoints; s = (s + 1) / 2) c *= 2;
        return c;
    }

    void build(std::size_t id, std::size_t l, std::size_t r) {
        if (!leaf(l, r)) {
            const std::size_t mid = middle(l, r);
            build(2 * id, l, mid), build(2 * id + 1, mid, r);
            node_[id] = node_[2 * id] * node_[2 * id + 1];
            return;
        }
        // T = prod (x - x_i), one factor at a time: T[j] = T[j - 1] - x_i T[j].
        Polynomial<P>& T = node_[id];
        T = Polynomial<P>(r - l + 1);
        T[0] = 1;
        for (std::size_t i = l; i < r; ++i) {
            const u32 a = x_[i] ? P - x_[i] : 0;
            const std::size_t d = i - l + 1;
            T[d] = T[d - 1];
            for (std::size_t j = d - 1; j > 0; --j) T[j] = u32((u64(a) * T[j] + T[j - 1]) % P);
            T[0] = mod_mul<P>(a, T[0]);
        }
    }

    Span x_;
    std::vector<Polynomial<P>> node_;
};

// out[i] = f(x[i]), the points side by side so the products overlap.
template <u32 P>
void horner(Span f, Span x, Out out) {
    std::ranges::fill(out, 0u);
    for (std::size_t i = f.size(); i-- > 0;)
        for (std::size_t j = 0; j < x.size(); ++j) out[j] = u32((u64(out[j]) * x[j] + f[i]) % P);
}

// out[i] = f(x_i) for the points of node id over [l, r); f of any size.
template <u32 P>
void evaluate(const Tree<P>& tree, std::size_t id, std::size_t l, std::size_t r, Span f, Out out) {
    Polynomial<P> reduced;
    if (f.size() > r - l) {
        reduced = division::modulo<P>(f, division::divisor(tree[id]));
        f = reduced.span();
    }
    if (Tree<P>::leaf(l, r)) return horner<P>(f, tree.points().subspan(l, r - l), out.subspan(l, r - l));
    const std::size_t mid = Tree<P>::middle(l, r);
    evaluate(tree, 2 * id, l, mid, f, out);
    evaluate(tree, 2 * id + 1, mid, r, f, out);
}

// sum over the points x_i of node id of c[i] T / (x - x_i): r - l coefficients.
template <u32 P>
Polynomial<P> combine(const Tree<P>& tree, std::size_t id, std::size_t l, std::size_t r, Span c) {
    if (!Tree<P>::leaf(l, r)) {
        const std::size_t mid = Tree<P>::middle(l, r);
        Polynomial<P> f = combine(tree, 2 * id, l, mid, c) * tree[2 * id + 1];
        f += combine(tree, 2 * id + 1, mid, r, c) * tree[2 * id];
        return f;
    }
    // T / (x - a) by synthetic division: q[k - 1] = T[k], q[j - 1] = T[j] + a q[j].
    const std::size_t k = r - l;
    const Polynomial<P>& T = tree[id];
    Polynomial<P> f(k);
    for (std::size_t i = l; i < r; ++i) {
        const u32 a = tree.points()[i];
        u32 q = 1;  // T[k]
        for (std::size_t j = k; j-- > 0;) {
            f[j] = u32((u64(c[i]) * q + f[j]) % P);
            if (j) q = u32((u64(a) * q + T[j]) % P);
        }
    }
    return f;
}

// c[l, r) for g of r - l coefficients: g = sum_(l <= k < r) c[k] prod_(l <= i < k) (x - x_i).
template <u32 P>
void newton(const Tree<P>& tree, std::size_t id, std::size_t l, std::size_t r, Span g, Out c) {
    if (Tree<P>::leaf(l, r)) {
        // g = c[k] + (x - x_k) g', g' the quotient: synthetic division, in place.
        u32 buffer[kLeafPoints];
        const Out h(buffer, g.size());
        std::ranges::copy(g, h.begin());
        for (std::size_t k = l; k < r; ++k) {
            const u32 a = tree.points()[k];
            u32 b = 0;
            for (std::size_t i = r - k; i-- > 0;) {
                const u32 t = u32((u64(a) * b + h[i]) % P);
                h[i] = b, b = t;
            }
            c[k] = b;
        }
        return;
    }
    const std::size_t mid = Tree<P>::middle(l, r);
    const division::Divisor D = division::divisor(tree[2 * id]);
    const Polynomial<P> q = division::quotient<P>(g, D);
    newton(tree, 2 * id, l, mid, division::remainder<P>(g, D, q.span()).span(), c);
    newton(tree, 2 * id + 1, mid, r, q.span(), c);
}

}  // namespace detail::points

template <std::uint32_t P>
std::vector<std::uint32_t> evaluate(const Polynomial<P>& f, std::span<const std::uint32_t> points) {
    detail::points::check_canonical(points, P, "fps::evaluate: point >= P");
    std::vector<std::uint32_t> values(points.size());
    if (points.empty()) return values;
    const detail::points::Tree<P> tree(points);
    detail::points::evaluate(tree, 1, 0, points.size(), f.span(), std::span(values));
    return values;
}

template <std::uint32_t P>
Polynomial<P> interpolate(std::span<const std::uint32_t> points, std::span<const std::uint32_t> values) {
    using detail::points::u32;
    const std::size_t m = points.size();
    if (values.size() != m) throw std::domain_error("fps::interpolate: points and values differ in size");
    detail::points::check_canonical(points, P, "fps::interpolate: point >= P");
    detail::points::check_canonical(values, P, "fps::interpolate: value >= P");
    if (m == 0) return {};
    const detail::points::Tree<P> tree(points);
    // c_i = values[i] / M'(x_i): one inversion for all, from the prefix products of M'(x_i).
    std::vector<u32> c(m), prefix(m);
    const Polynomial<P> d = derivative(tree[1]);
    detail::points::evaluate(tree, 1, 0, m, d.span(), std::span(c));
    u32 product = 1;
    for (std::size_t i = 0; i < m; ++i) {
        if (c[i] == 0) throw std::domain_error("fps::interpolate: points not distinct");
        prefix[i] = product, product = mod_mul<P>(product, c[i]);
    }
    u32 inverse = mod_inverse<P>(product);  // 1 / (M'(x_0) ... M'(x_i)) for i from m - 1 down
    for (std::size_t i = m; i-- > 0;) {
        const u32 w = c[i];
        c[i] = mod_mul<P>(values[i], mod_mul<P>(inverse, prefix[i]));
        inverse = mod_mul<P>(inverse, w);
    }
    return detail::points::combine(tree, 1, 0, m, std::span<const u32>(c));
}

template <std::uint32_t P>
Polynomial<P> to_newton_basis(const Polynomial<P>& f, std::span<const std::uint32_t> points) {
    const std::size_t n = f.size();
    if (n > 1 && points.size() < n - 1) throw std::domain_error("fps::to_newton_basis: needs f.size() - 1 points");
    if (n == 0) return {};
    // The tree over n points, the last (0) unread: no node holds it but the right edge's leaf.
    std::vector<std::uint32_t> x(n);
    std::copy_n(points.begin(), n - 1, x.begin());
    detail::points::check_canonical(std::span(x), P, "fps::to_newton_basis: point >= P");
    const detail::points::Tree<P> tree(x);
    Polynomial<P> c = detail::unwritten_polynomial<P>(n);
    detail::points::newton(tree, 1, 0, n, f.span(), c.span());
    return c;
}

}  // namespace fps
