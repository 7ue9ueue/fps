// Shift of sampling points modulo P = 998244353 (backend kernel).
//
//   fps::chirp::extrapolate(values, c, out);   // out[k] = f(c + k), k < out.size(), for the f of
//                                              // degree < n with f(i) = values[i], i < n = values.size()
//
// Words < P, n <= P; spans of any size and alignment; out must not overlap values.
//
// Lagrange on 0 .. n - 1: f(x) = W(x) sum_i g_i / (x - i) with W(x) = x (x - 1) .. (x - n + 1) and
// g_i = values[i] (-1)^(n - 1 - i) / (i! (n - 1 - i)!). Points c + k that are samples (mod P) are
// copied; the others form runs x0, x0 + 1, .. inside [n, P), each one ntt::middle_product: with
// u_t = x0 - n + 1 + t (all in [1, P)), sum_i g_i / (x0 + k - i) = sum_j h_(k + j) g_(n - 1 - j),
// h_t = 1 / u_t, and W(x0 + k) = u_k .. u_(k + n - 1). Prefix products of u and their inverses
// (scan.hpp) give both. From lc-opt's shift_of_sampling_points_of_polynomial (one run per call
// there too, at most two in practice: before and after the samples).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

#include "backend/chirp/scan.hpp"
#include "backend/factorials/factorials.hpp"
#include "backend/ntt/ntt.hpp"

namespace fps::chirp {

inline void extrapolate(std::span<const u32> values, u32 c, std::span<u32> out) {
    using namespace detail;
    const std::size_t n = values.size(), m = out.size();
    if (m == 0) return;
    if (n <= 1) return std::fill(out.begin(), out.end(), n ? values[0] : 0u);
    u32* g = Scratch::get(3 * n + 3 * (m + n));
    u32 *fact = g + n, *inverse_fact = fact + n, *prefix = inverse_fact + n;
    u32 *inverse_prefix = prefix + (m + n), *h = inverse_prefix + (m + n);
    factorials::tables<kP>({fact, n}, {inverse_fact, n});
    std::reverse_copy(inverse_fact, inverse_fact + n, fact);  // 1 / (n - 1 - i)!
    multiply(values.data(), inverse_fact, g, n);
    multiply(g, fact, g, n);
    for (std::size_t i = n % 2; i < n; i += 2) g[i] = g[i] ? kP - g[i] : 0;  // (-1)^(n - 1 - i)
    for (std::size_t k = 0; k < m;) {
        const std::size_t x = (c + k) % kP;
        if (x < n) {
            const std::size_t count = std::min(m - k, n - x);
            std::copy_n(values.data() + x, count, out.data() + k);
            k += count;
            continue;
        }
        const std::size_t count = std::min<std::size_t>(m - k, kP - x), t = count + n - 1;
        products(Arithmetic{u32(x - n + 1)}, t, prefix, 1, inverse_prefix, 1);
        multiply(prefix, inverse_prefix + 1, h, t);  // h_t = 1 / u_t
        ntt::middle_product({h, t}, {g, n}, out.subspan(k, count));
        multiply(out.data() + k, prefix + n, out.data() + k, count);
        multiply(out.data() + k, inverse_prefix, out.data() + k, count);
        k += count;
    }
}

}  // namespace fps::chirp
