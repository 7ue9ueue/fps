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
// - else f(g) = F(g - a), F mod x^n by the Taylor shift above (a != 0), then one call into the
//   backend's fps::composition::compose (Kinoshita and Li with lc-opt's layout), which reads g[0]
//   as 0. O(n log^2 n).
// compositional_inverse needs f[0] = 0 for n >= 1 and f[1] != 0 for n >= 2 (only f mod x^n is
// read), and n <= P; else std::domain_error. Then it is one call into the backend's
// fps::composition::compositional_inverse (power projection, then Lagrange's inversion).
//
// Limits: compose's general path and compositional_inverse run for n up to 2^23, the kernel's
// kMaxSize (its transforms have 4 bit_ceil(n) words), and throw std::length_error past it before
// any work. The Taylor shift is a cyclic product of f.size() + t - 1 words, t = min(n, f.size());
// it throws past 2^25. The tuned Taylor shift is backend issue #124.
//
// Memory: the kernel keeps its work per thread, the largest so far, m = bit_ceil(n): compose about
// 16 m (log2 m + 3) bytes (n = 2^17: 40 MiB; 2^23: 3.3 GiB), compositional_inverse about 150 m.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "backend/composition/composition.hpp"
#include "fps/factorials.hpp"
#include "fps/modular.hpp"
#include "fps/polynomial.hpp"
#include "fps/products.hpp"

namespace fps {

namespace detail::composition {

using u32 = std::uint32_t;
using Span = std::span<const u32>;
using Out = std::span<u32>;

// The NTT's longest transform, 2^25 words.
inline constexpr std::size_t kMaxTransform = std::size_t(1) << ntt::kMaxLog;

inline u32 at(Span f, std::size_t i) { return i < f.size() ? f[i] : 0; }

// out[k] = [y^k] f(y + a) for k < out.size() <= f.size(): k! out[k] = sum_j A[k + j] E[j] with
// A[i] = f[i] i!, E[j] = a^j / j!, a middle product of A (zeros past f) and E reversed.
template <u32 P>
void taylor_shift(Span f, u32 a, Out out) {
    const std::size_t m = f.size(), t = out.size();
    if (t == 0) return;
    // ntt::middle_product: a cyclic product of length >= m - 1 + t.
    if (m - 1 + t > kMaxTransform) throw std::length_error("fps::compose: Taylor shift past 2^25 words");
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
    if (n == 0 || f.size() == 0) return Polynomial<P>(n);
    if (terms > 2 && n > composition::kMaxSize)
        throw std::length_error("fps::compose: n past 2^23 (the composition kernel's kMaxSize)");
    if (terms <= 1) {
        Polynomial<P> h(n);
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
        Polynomial<P> h(n);
        std::uint32_t power = 1;
        for (std::size_t k = 0; k < t; ++k, power = mod_mul<P>(power, gs[1])) h[k] = mod_mul<P>(F[k], power);
        return h;
    }
    Polynomial<P> h = detail::unwritten_polynomial<P>(n);
    composition::compose(F, gs, h.span());
    return h;
}

template <std::uint32_t P>
Polynomial<P> compose(const Polynomial<P>& f, const Polynomial<P>& g) {
    return compose(f, g, std::max(f.size(), g.size()));
}

template <std::uint32_t P>
Polynomial<P> compositional_inverse(const Polynomial<P>& f, std::size_t n) {
    static_assert(P == ntt::kModulus, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");
    if (n >= 1 && f.coeff(0) != 0) throw std::domain_error("fps::compositional_inverse: f[0] != 0");
    if (n >= 2 && f.coeff(1) == 0) throw std::domain_error("fps::compositional_inverse: f[1] = 0");
    if (n > P) throw std::domain_error("fps::compositional_inverse: n > P");
    if (n > composition::kMaxSize)
        throw std::length_error("fps::compositional_inverse: n past 2^23 (the composition kernel's kMaxSize)");
    Polynomial<P> g = detail::unwritten_polynomial<P>(n);
    composition::compositional_inverse(f.span(), g.span());
    return g;
}

template <std::uint32_t P>
Polynomial<P> compositional_inverse(const Polynomial<P>& f) {
    return compositional_inverse(f, f.size());
}

}  // namespace fps
