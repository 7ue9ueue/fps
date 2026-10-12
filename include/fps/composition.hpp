// Composition of power series modulo P = 998244353 (DESIGN.md, Functions: Series).
//
//   fps::compose(f, g, n)              // f(g) mod x^n; n defaults to max(f.size(), g.size())
//   fps::compositional_inverse(f, n)   // g with f(g) = g(f) = x mod x^n; n defaults to f.size()
//
// compose takes any f and g; only g mod x^n counts. With a = g[0]:
// - g mod x^n constant: f(a), then zeros. O(f.size()).
// - g mod x^n = a + b x, b != 0: h[k] = F[k] b^k, F(y) = f(y + a) (for a = 0, F = f). The Taylor
//   shift is one middle product: k! F[k] = sum_j (f[k + j] (k + j)!) (a^j / j!), for k < min(n,
//   f.size()). O(n log n).
// - else f(g) = F(g - a), F mod x^n by the Taylor shift above, by Kinoshita and Li's algorithm
//   (below). O(n log^2 n).
// compositional_inverse needs f[0] = 0 for n >= 1 and f[1] != 0 for n >= 2 (only f mod x^n is
// read), and n <= P; else std::domain_error. It runs Kinoshita and Li's power projection,
// [x^(n-1)] f^k for k < n, then Lagrange's inversion: [x^j] (x / g)^(n-1) = (n-1) / (n-1-j)
// [x^(n-1)] f^(n-1-j), so g / x = (1 / f[1]) (h / h[0])^(-1 / (n-1)) for h = (x / g)^(n-1),
// by the series' pow.
//
// Kinoshita and Li (arXiv 2404.05177). Let Q(x, y) = 1 - y G(x), G[0] = 0, mod x^n. Then
// f(G) = [y^(m-1)] f_rev(y) / Q, f_rev(y) = y^(m-1) f(1 / y), and 1 / Q = Q(-x, y) / V(x^2, y)
// with V(x^2, y) = Q(x, y) Q(-x, y): x halves, y doubles. Going down, the levels keep
// Q_l (ceil(n / 2^l) rows in x, y-degree 2^l); Q_l(0, y) = 1. Going up, level l needs only
// the y-window [m - 2^l, m) of f_rev / Q_l: the window of level l + 1 times Q_l(-x, y), a
// product truncated in x and a middle product in y. At the bottom the window is f reversed.
// Power projection is Bostan and Mori's [x^N] P / Q with y carried along, P = 1. Each step is a
// product of Kronecker-packed bivariates (x^i y^j at word i Y + j) on raw::mul, cut to its rows:
// for n a power of two, 2 n to 3 n words in compose's levels and 3 n in the power projection's
// first step (fewer after), so transforms of 8 n. A product past 2^25 words throws
// std::length_error; both functions check every level's size before any work: up to about 6 n
// words for compose (3 n for n a power of two, so 2^23 passes and 2^23 + 1 throws), 3 n for
// compositional_inverse (2^23 passes, 2^24 throws).
// The tuned versions are backend issues: #105 (Kinoshita and Li, lc-opt's), #124 (Taylor
// shift). Times against ntt::multiply: frontend/notes/26.md.
//
// Memory: compose keeps every level's Q_l, about (log2 n + 2) n words (2^20: 92 MB), and
// compositional_inverse O(n) words; the products use the backend's per-thread memory.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fps/factorials.hpp"
#include "fps/modular.hpp"
#include "fps/polynomial.hpp"
#include "fps/products.hpp"
#include "fps/series.hpp"

namespace fps {

namespace detail::composition {

using u32 = std::uint32_t;
using Span = std::span<const u32>;
using Out = std::span<u32>;

inline constexpr std::size_t kMaxProduct = std::size_t(1) << ntt::kMaxLog;

inline u32 at(Span f, std::size_t i) { return i < f.size() ? f[i] : 0; }

template <u32 P>
u32 negate(u32 c) {
    return c ? P - c : 0;
}

// A bivariate polynomial: word i * width + j is the coefficient of x^i y^j.
struct Grid {
    std::size_t rows = 0, width = 0;
    std::vector<u32> c;

    Grid(std::size_t rows, std::size_t width) : rows(rows), width(width), c(rows * width) {}
    u32* row(std::size_t i) { return c.data() + i * width; }
    const u32* row(std::size_t i) const { return c.data() + i * width; }
};

// q in rows of Y >= q.width words, and q(-x, y) the same way if negate.
template <u32 P>
std::vector<u32> pack(const Grid& q, std::size_t Y, bool negate) {
    std::vector<u32> a(q.rows * Y);
    for (std::size_t i = 0; i < q.rows; ++i)
        for (std::size_t j = 0; j < q.width; ++j) {
            const u32 c = q.row(i)[j];
            a[i * Y + j] = negate && i % 2 ? composition::negate<P>(c) : c;
        }
    return a;
}

// a b mod z^(a.size()), a.size() = b.size().
template <u32 P>
std::vector<u32> product(const std::vector<u32>& a, const std::vector<u32>& b) {
    std::vector<u32> c(a.size());
    raw::mul<P>(a, b, c);
    return c;
}

// Q(x, y) = 1 - y g(x) mod x^n, g[0] taken as 0.
template <u32 P>
Grid denominator(Span g, std::size_t n) {
    Grid q(n, 2);
    q.row(0)[0] = 1;
    for (std::size_t i = 1; i < n; ++i) q.row(i)[1] = negate<P>(at(g, i));
    return q;
}

// V with V(x^2, y) = q(x, y) q(-x, y) mod x^q.rows: ceil(rows / 2) rows, y-degree 2 k.
template <u32 P>
Grid graeffe(const Grid& q) {
    const std::size_t k = q.width - 1, Y = 2 * k + 1;
    const std::vector<u32> c = product<P>(pack<P>(q, Y, false), pack<P>(q, Y, true));
    Grid v((q.rows + 1) / 2, Y);
    for (std::size_t s = 0; s < v.rows; ++s) std::copy_n(c.data() + 2 * s * Y, Y, v.row(s));
    return v;
}

// The y-window [k, b.width) of q(-x, y) b(x^2, y) mod x^q.rows, for q of y-degree k: a product in
// x, a middle product in y. Rows q.rows, width b.width - k.
template <u32 P>
Grid lift(const Grid& q, const Grid& b) {
    const std::size_t n = q.rows, k = q.width - 1, w = b.width - k, Y = b.width + k;
    std::vector<u32> e(n * Y);
    for (std::size_t s = 0; 2 * s < n; ++s) std::copy_n(b.row(s), b.width, e.data() + 2 * s * Y);
    const std::vector<u32> c = product<P>(pack<P>(q, Y, true), e);
    Grid r(n, w);
    for (std::size_t i = 0; i < n; ++i) std::copy_n(c.data() + i * Y + k, w, r.row(i));
    return r;
}

// The longest product of compose's levels (graeffe and lift), in words.
inline std::size_t compose_longest(std::size_t n) {
    std::size_t longest = 0;
    for (std::size_t rows = n, k = 1; rows > 1; rows = (rows + 1) / 2, k *= 2)
        longest = std::max({longest, rows * (2 * k + 1), rows * 3 * k});
    return longest;
}

// out = f(g) mod x^n, n = out.size() >= 1, for g[0] = 0 (not read), f.size() <= n.
template <u32 P>
void kinoshita_li(Span f, Span g, Out out) {
    const std::size_t n = out.size();
    std::vector<Grid> levels;
    Grid q = denominator<P>(g, n);
    while (q.rows > 1) {
        Grid next = graeffe<P>(q);
        levels.push_back(std::move(q));
        q = std::move(next);
    }
    const std::size_t top = std::size_t(1) << levels.size();  // >= n >= f.size()
    Grid b(1, top);
    for (std::size_t j = 0; j < top; ++j) b.row(0)[j] = at(f, top - 1 - j);
    for (std::size_t l = levels.size(); l-- > 0;) b = lift<P>(levels[l], b);
    for (std::size_t i = 0; i < n; ++i) out[i] = b.row(i)[0];
}

// The longest product of the power projection's steps, in words.
inline std::size_t projection_longest(std::size_t N) {
    std::size_t longest = 0;
    for (std::size_t k = 1; N > 0; N /= 2, k *= 2) longest = std::max(longest, (N + 1) * (2 * k + 1));
    return longest;
}

// out[k] = [x^N] f^k for k <= N = out.size() - 1, for f[0] = 0 (not read). Bostan and Mori on
// p / q, p = 1, q = 1 - y f(x): p(x, y) q(-x, y) and q(x, y) q(-x, y), then the half of x's
// parity of N; [x^0] p / q = p(0, y), as q(0, y) = 1.
template <u32 P>
void power_projection(Span f, Out out) {
    std::size_t N = out.size() - 1;
    Grid p(N + 1, 1), q = denominator<P>(f, N + 1);
    p.row(0)[0] = 1;
    while (N > 0) {
        const std::size_t k = q.width - 1, Y = 2 * k + 1, half = N / 2, parity = N % 2;
        const std::vector<u32> negated = pack<P>(q, Y, true);
        const std::vector<u32> u = product<P>(pack<P>(p, Y, false), negated);
        const std::vector<u32> v = product<P>(pack<P>(q, Y, false), negated);
        Grid p2(half + 1, 2 * k), q2(half + 1, Y);
        for (std::size_t s = 0; s <= half; ++s) {
            std::copy_n(u.data() + (2 * s + parity) * Y, 2 * k, p2.row(s));
            std::copy_n(v.data() + 2 * s * Y, Y, q2.row(s));
        }
        p = std::move(p2), q = std::move(q2), N = half;
    }
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = p.row(0)[i];  // p.width > N
}

// out[k] = [y^k] f(y + a) for k < out.size() <= f.size(): k! out[k] = sum_j A[k + j] E[j] with
// A[i] = f[i] i!, E[j] = a^j / j!, a middle product of A (zeros past f) and E reversed.
template <u32 P>
void taylor_shift(Span f, u32 a, Out out) {
    const std::size_t m = f.size(), t = out.size();
    if (t == 0) return;
    if (m - 1 + t > kMaxProduct) throw std::length_error("fps::compose: Taylor shift past 2^25 words");
    const Factorials<P> fact(m);
    Polynomial<P> weighted(m + t - 1), e(m);
    for (std::size_t i = 0; i < m; ++i) weighted[i] = mod_mul<P>(f[i], fact[i]);
    u32 power = 1;
    for (std::size_t j = 0; j < m; ++j, power = mod_mul<P>(power, a)) e[m - 1 - j] = mod_mul<P>(power, fact.inverse(j));
    const Polynomial<P> h = middle_product(weighted, e);
    for (std::size_t k = 0; k < t; ++k) out[k] = mod_mul<P>(h[k], fact.inverse(k));
}

}  // namespace detail::composition

template <std::uint32_t P>
Polynomial<P> compose(const Polynomial<P>& f, const Polynomial<P>& g, std::size_t n) {
    namespace c = detail::composition;
    static_assert(P == ntt::kModulus, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");
    const std::span<const std::uint32_t> gs = g.span().first(std::min(g.size(), n));
    const std::uint32_t a = c::at(gs, 0);
    std::size_t terms = gs.size();  // g mod x^n has degree terms - 1
    while (terms > 0 && gs[terms - 1] == 0) --terms;
    if (terms > 2 && c::compose_longest(n) > c::kMaxProduct)
        throw std::length_error("fps::compose: a product past 2^25 words");
    Polynomial<P> h(n);
    if (n == 0 || f.size() == 0) return h;
    if (terms <= 1) {
        h[0] = f(a);
        return h;
    }
    // F = f(y + a) mod y^t.
    const std::size_t t = std::min(n, f.size());
    Polynomial<P> shifted;
    std::span<const std::uint32_t> F = f.span().first(t);
    if (a != 0) {
        shifted = Polynomial<P>(t);
        c::taylor_shift<P>(f.span(), a, shifted.span());
        F = shifted.span();
    }
    if (terms == 2) {
        std::uint32_t power = 1;
        for (std::size_t k = 0; k < t; ++k, power = mod_mul<P>(power, gs[1])) h[k] = mod_mul<P>(F[k], power);
        return h;
    }
    c::kinoshita_li<P>(F, gs, h.span());
    return h;
}

template <std::uint32_t P>
Polynomial<P> compose(const Polynomial<P>& f, const Polynomial<P>& g) {
    return compose(f, g, std::max(f.size(), g.size()));
}

template <std::uint32_t P>
Polynomial<P> compositional_inverse(const Polynomial<P>& f, std::size_t n) {
    namespace c = detail::composition;
    static_assert(P == ntt::kModulus, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");
    if (n >= 1 && f.coeff(0) != 0) throw std::domain_error("fps::compositional_inverse: f[0] != 0");
    if (n >= 2 && f.coeff(1) == 0) throw std::domain_error("fps::compositional_inverse: f[1] = 0");
    if (n > P) throw std::domain_error("fps::compositional_inverse: n > P");
    if (n >= 3 && c::projection_longest(n - 1) > c::kMaxProduct)
        throw std::length_error("fps::compositional_inverse: a product past 2^25 words");
    Polynomial<P> g(n);
    if (n <= 1) return g;
    const std::uint32_t f1 = f[1], inverse1 = mod_inverse<P>(f1);
    if (n == 2) {
        g[1] = inverse1;
        return g;
    }
    // h[j] = [x^j] (x / g)^N / N = [x^N] f^(N - j) / (N - j) for j < N = n - 1 (the factor N
    // cancels below).
    const std::size_t N = n - 1;
    std::vector<std::uint32_t> projection(n), r(n), tmp(n), h(N);
    c::power_projection<P>(f.span().first(std::min(f.size(), n)), projection);
    detail::series::reciprocals<P>(r, tmp);
    for (std::size_t j = 0; j < N; ++j) h[j] = mod_mul<P>(r[N - j], projection[N - j]);
    // g / x = (1 / f[1]) (h / h[0])^(-1 / N) mod x^N.
    const std::uint32_t e = c::negate<P>(r[N]);
    detail::series::power<P>(h, e, inverse1, g.span().subspan(1), detail::series::scratch(3 * N));
    g[0] = 0;
    return g;
}

template <std::uint32_t P>
Polynomial<P> compositional_inverse(const Polynomial<P>& f) {
    return compositional_inverse(f, f.size());
}

}  // namespace fps
