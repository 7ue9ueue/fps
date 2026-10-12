// Evaluation and interpolation on a geometric sequence modulo P = 998244353 (backend kernel).
//
//   fps::chirp::evaluate_geometric(f, a, r, out);     // out[i] = f(a r^i), i < out.size()
//   fps::chirp::interpolate_geometric(a, r, y, out);  // out: the f of degree < n = y.size() with
//                                                     // f(a r^i) = y[i]; out.size() == n
//
// Words < P; spans of any size and alignment; out must not overlap f or y. interpolate_geometric
// throws std::invalid_argument, before writing out, if the points a r^i (i < n) are not distinct
// or out has another size. Long products throw std::length_error (ntt, past 2^25 words).
//
// evaluate_geometric (the chirp z-transform, Bluestein): with t(k) = k (k - 1) / 2,
// i j = t(i) + t(-j) - t(i - j), so for r != 0 and f of n coefficients
//   f(a r^i) = r^t(i) sum_j A_j K_(i - j + n - 1),  A_j = f_j (a r)^j r^t(j),  K_v = r^-t(v - n + 1),
// one ntt::middle_product of K (n + m - 1 words) and A (n words), and three chirp passes (K, A, and
// the output's r^t(i)). Small n m: Horner at each point. r = 0: f(a), then f(0).
//
// interpolate_geometric, Newton's form on x_i = a r^i (lc-opt's geometric interpolation): with
// A_k = prod_(s < k) (r - r^-s) = (-1)^k r^-t(k) (1 - r) .. (1 - r^k) and B = 1 / A,
//   E' = (y_i B_i (-1)^i r^-t(i)) B mod x^n,   H_l = sum_k A_k E'_k B_(k - l),
//   f_l = (-1/a)^l r^-t(l) B_l H_l.
// Two ntt::multiply of n words mod x^n (H: of (A E') reversed and B), A and B in lanes
// (scan.hpp), two chirp passes. n <= 32 or r = 0: divided differences in O(n^2). Derivation:
// notes.md.
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "backend/chirp/chirp.hpp"
#include "backend/chirp/scan.hpp"
#include "backend/ntt/ntt.hpp"

namespace fps::chirp {

namespace detail {

inline u32 horner(std::span<const u32> f, u32 x) {
    u64 r = 0;
    for (std::size_t i = f.size(); i-- > 0;) r = (r * x + f[i]) % kP;
    return u32(r);
}

// Below this n m, evaluate_geometric runs Horner at each point.
inline constexpr std::size_t kHornerWork = 256;
// Up to this n, interpolate_geometric runs divided differences in O(n^2).
inline constexpr std::size_t kNewtonSize = 32;

// The coefficients of the f of degree < n through (x_i, y_i), x_i distinct, in O(n^2).
inline void newton(const u32* x, const u32* y, u32* out, std::size_t n) {
    std::vector<u32> d(y, y + n);
    for (std::size_t k = 1; k < n; ++k)
        for (std::size_t i = n - 1; i >= k; --i)
            d[i] = mul((d[i] + kP - d[i - 1]) % kP, inverse((x[i] + kP - x[i - k]) % kP));
    std::fill(out, out + n, 0u);
    for (std::size_t k = n; k-- > 0;) {  // out = out (x - x_k) + d_k
        for (std::size_t l = n - 1; l > 0; --l) out[l] = u32((out[l - 1] + u64(kP - x[k]) * out[l]) % kP);
        out[0] = u32((d[k] + u64(kP - x[k]) * out[0]) % kP);
    }
}

}  // namespace detail

inline void evaluate_geometric(std::span<const u32> f, u32 a, u32 r, std::span<u32> out) {
    using namespace detail;
    const std::size_t n = f.size(), m = out.size();
    if (m == 0) return;
    if (n == 0) return std::fill(out.begin(), out.end(), 0u);
    if (r == 0 || a == 0 || n == 1) {
        out[0] = horner(f, a);
        std::fill(out.begin() + 1, out.end(), a == 0 ? out[0] : f[0]);
        return;
    }
    if (n * m <= kHornerWork) {
        for (std::size_t i = 0; i < m; ++i, a = mul(a, r)) out[i] = horner(f, a);
        return;
    }
    const std::size_t g = n - 1, total = n + m - 1;
    u32* k = Scratch::get(total + n);
    u32* weighted = k + total;
    const u32 r_inverse = inverse(r);
    // K_v = r^-t(v - g) = r^-t(-g) (r^g)^v (1/r)^t(v), t(-g) = g (g + 1) / 2.
    chirp(power_of(r_inverse, triangle(g + 1)), power_of(r, g), r_inverse, {k, total});
    multiply_chirp(1, mul(a, r), r, f, {weighted, n});
    ntt::middle_product({k, total}, {weighted, n}, out);
    multiply_chirp(1, 1, r, out);
}

template <Products kProducts = kInterpolationProducts>
void interpolate_geometric(u32 a, u32 r, std::span<const u32> y, std::span<u32> out) {
    using namespace detail;
    const std::size_t n = y.size();
    if (out.size() != n) throw std::invalid_argument("fps::chirp::interpolate_geometric: out and y differ in size");
    if (n == 0) return;
    if (n == 1) {
        out[0] = y[0];
        return;
    }
    const auto not_distinct = [] {
        throw std::invalid_argument("fps::chirp::interpolate_geometric: the points are not distinct");
    };
    if (a == 0 || (r == 0 && n > 2)) not_distinct();
    if (n <= kNewtonSize || r == 0) {
        std::vector<u32> x(n);
        x[0] = a;
        for (std::size_t i = 1; i < n; ++i) x[i] = mul(x[i - 1], r);
        for (std::size_t i = 1; i < n; ++i)
            if (x[i] == a) not_distinct();  // r^i = 1
        return newton(x.data(), y.data(), out.data(), n);
    }
    const u32 r_inverse = inverse(r);
    const int lg = std::max(ntt::kMinLog, int(std::bit_width(2 * n - 2)));  // 2^lg >= 2n - 1
    const bool transform = kProducts == Products::transform && lg <= ntt::kValuesMaxLog;
    const std::size_t len = std::size_t(1) << lg, words = (n + 16) / 16 * 16, span = len + ntt::kPadding;
    // A and B = 1 / A: products of u(s) = r - r^-s. Then work (n words, or a transform's span).
    u32* A = Scratch::get(2 * words + (transform ? 2 * span : 2 * words));
    u32 *B = A + words, *work = B + words, *e = work + (transform ? span : words);
    products(Geometric(r, kP - 1, r_inverse), n - 1, A, 1, B, 1);
    if (A[n - 1] == 0) not_distinct();
    // work_i = y_i B_i (-1)^i r^-t(i); E' = work B mod x^n.
    multiply_chirp(1, kP - 1, r_inverse, y, {work, n});
    multiply(work, B, work, n);
    if (transform) {
        // Products in the values form, with B's transform (in e) shared: 5 transforms, not 6.
        const ntt::Transform t(lg);
        std::copy_n(B, n, e);
        std::fill(e + n, e + span, 0u);
        t.forward(e, ntt::Form::values);
        const auto product = [&] {  // work[0, n) = work[0, n) B mod x^n
            std::fill(work + n, work + span, 0u);
            t.forward(work, ntt::Form::values);
            multiply(work, e, work, len);
            t.inverse(work, ntt::Form::values);
        };
        product();
        multiply(A, work, work, n);
        std::reverse(work, work + n);
        product();
        e = work;
    } else {
        ntt::multiply({work, n}, {B, n}, {e, n});
        // work = (A E') reversed; H reversed = work B mod x^n.
        multiply(A, e, work, n);
        std::reverse(work, work + n);
        ntt::multiply({work, n}, {B, n}, {e, n});
    }
    // f_l = (-1/a)^l r^-t(l) B_l H_l.
    std::reverse(e, e + n);
    multiply(e, B, e, n);
    multiply_chirp(1, kP - inverse(a), r_inverse, {e, n}, out);
}

}  // namespace fps::chirp
