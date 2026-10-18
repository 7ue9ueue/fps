// Multipoint evaluation, interpolation and the Newton basis modulo P = 998244353 (backend kernel).
//
//   fps::evaluation::evaluate(f, points, values);     // values[i] = f(points[i]), values.size() = points.size()
//   fps::evaluation::interpolate(points, values, f);  // the f of f.size() = m = points.size() coefficients
//                                                     // with f(points[i]) = values[i]; distinct points
//   fps::evaluation::to_newton_basis(f, points, c);   // c.size() = f.size() = n: f = sum_k c[k] prod_(i < k)
//                                                     // (x - points[i]); reads points[0, n - 1)
//
// Spans of words < P (not checked), any alignment; the output apart from the inputs. Throws
// std::invalid_argument on sizes that do not match (interpolate: values.size() and f.size() must
// be m; to_newton_basis: points.size() >= n - 1), std::domain_error from interpolate if two points
// are equal (some M'(points[i]) is 0; f is then unspecified), and std::length_error, before any
// work, past 2^25 points (interpolate) or coefficients (to_newton_basis). evaluate takes any sizes,
// repeated points too.
//
// From lc-opt's lib/poly/{evaluation,interpolation,newton}.hpp (issue #104), on this repo's product
// trees (backend/product_tree, with kept transforms) and ntt::Transform, so every target runs it:
// - Points in 8 lanes (point 8 k + l in slot k of lane l), padded with zeros (factors 1) to m8 = 8
//   C points, C >= kMinSlots. The lane tree multiplies Q_v = prod (1 - a_i x) of the 8 lanes side by
//   side, a top tree the 8 lane products; each node's children's transforms are kept (tree.hpp).
// - evaluate: the root's window (rev_K(f) / Q mod x^K)[K - m8, K), K = max(n, m8), by Karp and
//   Markstein's division (fps::series::inverse, fused cyclic products); then the transposed descent (Bostan, Lecerf and Schost): one product with
//   the sibling's kept transform per child, then halving (inverse_upper and forward of half the
//   length), schoolbook middle products below kLaneBase slots. Horner's rule where a cost model
//   per target says it is cheaper (select.hpp: up to n = m = 256-512, or a short side).
// - interpolate: M'(a_i) by evaluate's descent (M = prod (x - a_i), from Q), the weights w_i = y_i
//   / M'(a_i) (one inversion per base), then the transpose of the descent, in the same pass: R_v =
//   R_l Q_r + R_r Q_l up the tree in the kept transforms; f = rev_m(R). Up to kDirectInterpolate
//   points (about 100), O(m^2) in vectors of points instead.
// - to_newton_basis: with F = rev(f), c_k = [x^(N-1-k)] F / Q_(k+1); a node's right child's window
//   is the low part of its own, its left child's a middle product with the right sibling's Q (one
//   product per node); the top tree's rightmost windows are prefixes of the root's. Points in
//   blocks per lane (lane l holds points [l C, (l + 1) C)). Up to kDirectNewton points (about
//   200), the synthetic divisions in a systolic array of vector lanes instead.
// Memory: one region of this thread (ntt's, kept across calls): the kept transforms, about 2 m8
// words per level of the lane tree (14.7 MB at m = 2^17), and scratch. Design and measurements:
// notes.md.
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "backend/evaluation/select.hpp"
#include "backend/evaluation/tree.hpp"
#include "backend/product_tree/product_tree.hpp"

namespace fps::evaluation {

namespace detail {

using product_tree::detail::RegionGuard;

inline constexpr int kMaxLog = ntt::kMaxLog;

// Points with padding: a multiple of 8, at least 8 kMinSlots.
inline std::size_t padded(std::size_t m) { return std::max((m + 7) / 8 * 8, 8 * kMinSlots); }

// Stack words of a PointTree of m8 points and its kept transforms, at most.
inline std::size_t tree_words(std::size_t m8, bool left) {
    const std::size_t slots = m8 / 8, L = lane_length(slots), lanes = 8 * L + 8;
    const std::size_t top = Stack::footprint(16 * L) + (left ? 4 : 2) * Stack::footprint(4 * L) + (left ? 2 : 1) * Stack::footprint(8 * L) +
                            Stack::footprint(8 * L) + Stack::footprint(8 * L + 8);
    return Stack::footprint(slots + 1) + Stack::footprint(2 * slots + 2) + lane_kept_words(slots, left) + top + 6 * Stack::footprint(lanes) +
           1024;
}

// For tests: with workspace_exact, a Workspace's stack has exactly the words asked for, and
// workspace_spilled tells whether the last one of this thread went past them (a size below is
// too small). Otherwise the stack has the whole region.
inline thread_local bool workspace_exact = false, workspace_spilled = false;

// A stack on this thread's region, of at least `words` words.
struct Workspace {
    RegionGuard guard;
    Stack stack;
    explicit Workspace(std::size_t words)
        : guard{ntt::detail::take_region(words * sizeof(u32))},
          stack(reinterpret_cast<u32*>(guard.region.start), workspace_exact ? words : guard.region.bytes / sizeof(u32)) {}
    ~Workspace() { workspace_spilled = stack.ever_spilled(); }
};

// q = Q / 2^32 (Q in Montgomery form), n words.
inline void normal_form(const u32* from, u32* to, std::size_t n) { scale<Wide>(from, 1, to, n); }

// values[i] = f(points[i]) by the tree, m <= 2^24, n <= 2^24.
inline void evaluate_tree(std::span<const u32> f, std::span<const u32> points, std::span<u32> values) {
    const std::size_t n = f.size(), m = points.size(), m8 = padded(m), big = std::max(n, m8);
    Workspace work(Stack::footprint(m8) + tree_words(m8, true) + Stack::footprint(m8 + 1) + Stack::footprint(big) +
                   std::max(division_words(big), PointTree::pass_words(m8)) + Stack::footprint(m8));
    Stack& stack = work.stack;
    u32* const x = stack.take(m8);
    std::fill(std::copy(points.begin(), points.end(), x), x + m8, 0u);
    const PointTree tree(x, m8 / 8, stack, true);
    u32* const q = stack.take(m8 + 1);
    normal_form(tree.product(), q, m8 + 1);
    u32* const d = stack.take(big);
    root_window(f.data(), n, q, m8 + 1, big, d, stack, tree.product_transform(), top_length(m8));
    u32* const all = stack.take(m8);
    tree.descend(d + big - m8, all, stack);
    std::copy_n(all, m, values.begin());
}

// evaluate(), with f and points in chunks of at most `chunk` (2^24: the tree's transforms stay
// within 2^25 words): points chunk by chunk; f = sum_j x^(j chunk) f_j, Horner's rule over j on
// the chunks' values.
inline void evaluate_chunks(std::span<const u32> f, std::span<const u32> points, std::span<u32> values, std::size_t chunk) {
    const std::size_t n = f.size(), m = points.size();
    if (m == 0) return;
    if (n == 0) return std::fill(values.begin(), values.end(), 0u);
    if (evaluate_directly(n, m)) return horner<Wide>(f.data(), n, points.data(), m, values.data());
    if (m > chunk) {
        for (std::size_t i = 0; i < m; i += chunk) {
            const std::size_t here = std::min(chunk, m - i);
            evaluate_chunks(f, points.subspan(i, here), values.subspan(i, here), chunk);
        }
        return;
    }
    if (n > chunk) {
        std::vector<u32> part(m), step(m);
        for (std::size_t i = 0; i < m; ++i) step[i] = power(points[i], chunk);
        std::fill(values.begin(), values.end(), 0u);
        for (std::size_t j = (n - 1) / chunk + 1; j-- > 0;) {
            evaluate_chunks(f.subspan(j * chunk, std::min(chunk, n - j * chunk)), points, part, chunk);
            for (std::size_t i = 0; i < m; ++i) values[i] = add_mod(multiply_mod(values[i], step[i]), part[i]);
        }
        return;
    }
    evaluate_tree(f, points, values);
}

// interpolate() for m <= kDirectInterpolate, in O(m^2): M = prod (x - a_i), w_i = y_i / M'(a_i), f =
// sum w_i M / (x - a_i), in vectors of points (interpolate_small).
inline void interpolate_direct(std::span<const u32> a, std::span<const u32> y, std::span<u32> f) {
    const std::size_t m = a.size();
    std::vector<u32> big(m + 1 + 16), scratch((Wide::lanes + 2) * m + 16);
    product_of_linear<Wide>(a.data(), m, big.data());
    if (!interpolate_small<Wide>(big.data(), m, a.data(), y.data(), f.data(), scratch.data()))
        throw std::domain_error("fps::evaluation::interpolate: repeated points");
}

// interpolate() by the tree, m >= 1.
inline void interpolate_tree(std::span<const u32> points, std::span<const u32> values, std::span<u32> f) {
    const std::size_t m = points.size(), m8 = padded(m), length = top_length(m8);
    Workspace work(Stack::footprint(m8) + tree_words(m8, true) + Stack::footprint(m8 + 1) + Stack::footprint(m) + 2 * Stack::footprint(m8) +
                   Stack::footprint(length) + std::max(division_words(m8), PointTree::pass_words(m8)));
    Stack& stack = work.stack;
    u32* const x = stack.take(m8);
    std::fill(std::copy(points.begin(), points.end(), x), x + m8, 0u);
    const PointTree tree(x, m8 / 8, stack, true);
    u32* const q = stack.take(m8 + 1);
    normal_form(tree.product(), q, m8 + 1);
    // M[j] = q[m - j] (padding factors are 1), M'[j] = (j + 1) M[j + 1] = (j + 1) q[m - 1 - j].
    u32* const derivative = stack.take(m);
    for (std::size_t j = 0; j < m; ++j) derivative[j] = multiply_mod(u32(j + 1), q[m - 1 - j]);
    u32* const d = stack.take(m8);
    root_window(derivative, m, q, m8 + 1, m8, d, stack, tree.product_transform(), top_length(m8));
    u32* const y = stack.take(m8);
    std::fill(std::copy(values.begin(), values.end(), y), y + m8, 0u);
    u32* const r = stack.take(length);
    tree.interpolate(d, y, m, r, stack);
    for (std::size_t j = 0; j < m; ++j) f[j] = r[m - 1 - j];
}

// Slots per lane for the Newton basis: C, or the power of two above it past kNewtonPad of it. A
// right child halves its parent's state when its left sibling fills half the parent's length:
// with C a power of two, every one does (else an inverse and a forward); padding points cost
// base work and a longer division only (lc-opt's notes: 7% at 101656 points, 0.65).
inline std::size_t newton_slots(std::size_t slots) {
    const std::size_t up = std::bit_ceil(slots);
    return double(slots) > kNewtonPad * double(up) ? up : slots;
}

// to_newton_basis() by the tree, n >= 1.
inline void newton_tree(std::span<const u32> f, std::span<const u32> points, std::span<u32> c) {
    const std::size_t n = f.size(), slots = newton_slots(padded(n) / 8), m8 = 8 * slots;
    Workspace work(Stack::footprint(m8) + tree_words(m8, false) + Stack::footprint(m8 + 1) + 2 * Stack::footprint(m8) +
                   std::max(division_words(m8), PointTree::pass_words(m8)));
    Stack& stack = work.stack;
    // Lane l holds points [l C, (l + 1) C): x[8 k + l] = points[l C + k]; points from n - 1 on are 0.
    u32* const x = stack.take(m8);
    for (std::size_t l = 0; l < 8; ++l)
        for (std::size_t k = 0; k < slots; ++k) x[8 * k + l] = l * slots + k + 1 < n ? points[l * slots + k] : 0;
    const PointTree tree(x, slots, stack, false);
    u32* const q = stack.take(m8 + 1);
    normal_form(tree.product(), q, m8 + 1);
    u32* const d = stack.take(m8);
    root_window(f.data(), n, q, m8 + 1, m8, d, stack, tree.product_transform(), top_length(m8));
    u32* const all = stack.take(m8);
    tree.descend_newton(d, all, stack);
    for (std::size_t l = 0; l < 8; ++l)
        for (std::size_t k = 0; k < slots && l * slots + k < n; ++k) c[l * slots + k] = all[8 * k + l];
}

}  // namespace detail

// values[i] = f(points[i]).
inline void evaluate(std::span<const std::uint32_t> f, std::span<const std::uint32_t> points, std::span<std::uint32_t> values) {
    if (values.size() != points.size()) throw std::invalid_argument("fps::evaluation::evaluate: values.size() != points.size()");
    detail::evaluate_chunks(f, points, values, std::size_t(1) << (detail::kMaxLog - 1));
}

// f of m = points.size() coefficients with f(points[i]) = values[i], for distinct points.
inline void interpolate(std::span<const std::uint32_t> points, std::span<const std::uint32_t> values, std::span<std::uint32_t> f) {
    const std::size_t m = points.size();
    if (values.size() != m || f.size() != m) throw std::invalid_argument("fps::evaluation::interpolate: sizes differ");
    if (m > (std::size_t(1) << detail::kMaxLog)) throw std::length_error("fps::evaluation::interpolate: more than 2^25 points");
    if (m <= kDirectInterpolate) return detail::interpolate_direct(points, values, f);
    detail::interpolate_tree(points, values, f);
}

// c of n = f.size() coefficients with f = sum_k c[k] prod_(i < k) (x - points[i]).
inline void to_newton_basis(std::span<const std::uint32_t> f, std::span<const std::uint32_t> points, std::span<std::uint32_t> c) {
    const std::size_t n = f.size();
    if (c.size() != n || points.size() + 1 < n) throw std::invalid_argument("fps::evaluation::to_newton_basis: sizes");
    if (n > (std::size_t(1) << detail::kMaxLog)) throw std::length_error("fps::evaluation::to_newton_basis: more than 2^25 coefficients");
    if (n == 0) return;
    if (n <= kDirectNewton) {
        std::copy(f.begin(), f.end(), c.begin());
        std::vector<std::uint32_t> scratch(n + 8 * detail::Wide::lanes);
        detail::newton_small<detail::Wide, 4>(c.data(), n, points.data(), scratch.data());
        return;
    }
    detail::newton_tree(f, points, c);
}

}  // namespace fps::evaluation
