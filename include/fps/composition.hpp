// Composition of power series modulo P = 998244353 (DESIGN.md, Functions: Series).
//
//   fps::compose(f, g, n)              // f(g) mod x^n; n defaults to max(f.size(), g.size())
//   fps::compositional_inverse(f, n)   // g with f(g) = g(f) = x mod x^n; n defaults to f.size()
//
// compose takes any f and g; only g mod x^n counts. With a = g[0] and F(y) = f(y + a) mod y^t,
// t = min(n, f.size()) (for a = 0, F = f; else one call into the backend's Taylor shift,
// fps::taylor_shift::shift, on f without its trailing zeros):
// - g mod x^n constant: f(a), then zeros. O(f.size()).
// - g mod x^n = a + b x, b != 0: h[k] = F[k] b^k (b = 1: F itself). O(n log n). The scaling is a
//   serial loop here, 3-4 ns a coefficient, up to the shift's own time (backend issue #186).
// - else f(g) = F(g - a): one call into the backend's fps::composition::compose (Kinoshita and Li
//   with lc-opt's layout), which reads g[0] as 0. O(n log^2 n).
// compositional_inverse needs f[0] = 0 for n >= 1 and f[1] != 0 for n >= 2 (only f mod x^n is
// read), and n <= P; else std::domain_error. Then it is one call into the backend's
// fps::composition::compositional_inverse (power projection, then Lagrange's inversion).
//
// Limits: compose's general path and compositional_inverse run for n up to 2^23, the kernel's
// kMaxSize (its transforms have 4 bit_ceil(n) words), and throw std::length_error past it before
// any work. The Taylor shift (a != 0, g not constant) takes f of degree below 2^24, the shift
// kernel's kMaxSize (a cyclic product of 2^25 words), and throws std::length_error past it before
// any work; trailing zeros do not count.
//
// Memory: the kernels keep their work per thread, the largest so far, m = bit_ceil(n): compose about
// 16 m (log2 m + 3) bytes (n = 2^17: 40 MiB; 2^23: 3.3 GiB), compositional_inverse about 150 m; the
// Taylor shift 2 L + d words, d = deg f + 1 and L = bit_ceil(2 d) (d = 2^24: 320 MiB).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "backend/composition/composition.hpp"
#include "backend/taylor_shift/taylor_shift.hpp"
#include "fps/modular.hpp"
#include "fps/polynomial.hpp"

namespace fps {

namespace detail::composition {

using u32 = std::uint32_t;

inline u32 at(std::span<const u32> f, std::size_t i) { return i < f.size() ? f[i] : 0; }

// f without its trailing zeros.
inline std::span<const u32> trimmed(std::span<const u32> f) {
    std::size_t d = f.size();
    while (d > 0 && f[d - 1] == 0) --d;
    return f.first(d);
}

// out[k] = [y^k] f(y + a) for k < out.size(), any out.size(): the kernel's shift, zeros past f.
// The caller checks f.size() <= the kernel's kMaxSize.
inline void shift(std::span<const u32> f, u32 a, std::span<u32> out) {
    const std::size_t m = std::min(f.size(), out.size());
    fps::taylor_shift::shift(f, a, out.first(m));
    std::fill(out.begin() + std::ptrdiff_t(m), out.end(), 0);
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
    // F = f(y + a) mod y^t, t = min(n, f.size()); the shift reads f without its trailing zeros.
    const std::size_t t = std::min(n, f.size());
    const std::span<const std::uint32_t> fs = a != 0 ? c::trimmed(f.span()) : f.span().first(t);
    if (a != 0 && fs.size() > fps::taylor_shift::kMaxSize)
        throw std::length_error("fps::compose: Taylor shift of f of degree 2^24 or more (the shift kernel's kMaxSize)");
    if (terms == 2) {  // h = F(b y): F written into h, then scaled by b^k
        Polynomial<P> h = detail::unwritten_polynomial<P>(n);
        if (a != 0) c::shift(fs, a, h.span().first(t));
        else std::copy(fs.begin(), fs.end(), h.span().begin());
        std::fill(h.span().begin() + std::ptrdiff_t(t), h.span().end(), 0);
        if (const std::uint32_t b = gs[1]; b != 1) {
            std::uint32_t power = b;
            for (std::size_t k = 1; k < t; ++k, power = mod_mul<P>(power, b)) h[k] = mod_mul<P>(h[k], power);
        }
        return h;
    }
    Polynomial<P> shifted;
    std::span<const std::uint32_t> F = fs;
    if (a != 0) {
        shifted = detail::unwritten_polynomial<P>(t);
        c::shift(fs, a, shifted.span());
        F = shifted.span();
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
