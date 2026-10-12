// Division with remainder of polynomials modulo P = 998244353 (DESIGN.md, Polynomial: arithmetic;
// Functions: Division and calculus).
//
//   auto [q, r] = fps::divmod(f, g);   // f = q g + r with deg r < deg g; g != 0
//   f / g, f % g                       // q alone (no remainder is computed), r alone
//   f /= g, f %= g
//
// Sizes, with n = f.size() and d = deg g: q has max(n - d, 0) coefficients and r has d (f's, then
// zeros, when n <= d). Nothing is trimmed: q and r may end in zeros. g with no nonzero coefficient
// (of any size) throws std::domain_error.
//
// Paths. g = c x^d (one nonzero coefficient) is a shift and a scaling, O(n): f / x^k drops the
// lowest k coefficients, f % x^k is f.truncate(k). Otherwise, with k = n - d, F = x^(n-1) f(1/x)
// and G = x^d g(1/x) (G[0] = g[d] != 0), the reversed quotient is F / G mod x^k:
// - G mod x^k with at most 16 nonzero terms past G[0] (fps/series.hpp's kSparseTerms: divisors of
//   degree <= 16, quotients of <= 17 coefficients, divisors such as x^d - x - 1): the recurrence
//   G[0] Q[i] = F[i] - sum_j G[j] Q[i - j], O(k t) for t terms;
// - else F (1 / G mod x^k) mod x^k: fps/series.hpp's inverse (backend/series), then one product;
// and the remainder is f - (q g mod x^d), one product. The products are fps/products.hpp's
// (raw::mul: shift, small::multiply or ntt::multiply, as small::beats_ntt picks).
// The backend has no division kernel yet: lc-opt's blocked division (quotient blocks against one
// inverse of G, the remainder from a cyclic product of length >= d) is backend issue #102. f of 2n
// by g of n coefficients from 2^10 on, in units of one ntt::multiply of two factors of n into n
// words: divmod and f % g 6.9-8.5, f / g 5.9-7.4, on the timed targets (frontend/notes.md). The
// recurrence: 2.9-4.2 ns per coefficient of f for 2 terms.
//
// A product or inverse past 2^25 coefficients throws std::length_error from the backend
// (fps/series.hpp, fps/products.hpp); the recurrence and the shift have no such limit.
//
// Memory: q and r are new polynomials; the quotient works in fps/series.hpp's per-thread scratch
// (3 k words on the Newton path, min(k, d + 1) on the recurrence).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>

#include "backend/modarith/modarith.hpp"
#include "backend/simd/select.hpp"
#include "fps/modular.hpp"
#include "fps/polynomial.hpp"
#include "fps/products.hpp"
#include "fps/series.hpp"

namespace fps {

namespace detail::division {

using u32 = std::uint32_t;
using Span = std::span<const u32>;
using Out = std::span<u32>;

// g, d = deg g, and whether g = c x^d (g[0, d) is zero). Throws for g = 0.
struct Divisor {
    Span g;
    std::size_t d;
    bool monomial;
};

template <u32 P>
Divisor divisor(const Polynomial<P>& g) {
    const int d = g.degree();
    if (d < 0) throw std::domain_error("fps: division by zero");
    const Span low = g.span().first(std::size_t(d));
    return {g.span(), std::size_t(d), std::ranges::all_of(low, [](u32 c) { return c == 0; })};
}

// out[i] = c a[i] (Shoup's product); a copy for c = 1.
template <u32 P>
void scale(Span a, u32 c, Out out) {
    if (c == 1) return void(std::ranges::copy(a, out.begin()));
    const u32 q = modarith::Ops<simd::Scalar, P>::shoup_quotient(c);
    lanewise(a.data(), a.data(), out.data(), a.size(), [c, q]<class S>(typename S::V x, typename S::V) {
        using Ops = modarith::Ops<S, P>;
        return Ops::template reduce<P>(Ops::shoup(x, S::broadcast(c), S::broadcast(q)));
    });
}

// out[i] = a[i] - out[i].
template <u32 P>
void subtract_from(Span a, Out out) {
    lanewise(a.data(), out.data(), out.data(), a.size(),
             []<class S>(typename S::V x, typename S::V y) { return modarith::Ops<S, P>::sub(x, y); });
}

// q = the quotient of f (n words) by g (deg g = d < n, not a monomial), k = n - d = q.size()
// words. scratch: 3 k words.
template <u32 P>
void quotient(Span f, Span g, std::size_t d, Out q, Out scratch) {
    const std::size_t n = f.size(), k = q.size();
    // G mod x^k, then Q = F / G mod x^k into q, then q = Q reversed.
    const Out G = scratch.first(std::min(k, d + 1));
    for (std::size_t i = 0; i < G.size(); ++i) G[i] = g[d - i];
    if (auto s = series::sparse(G)) {
        // Q[i] = c F[i] - sum over terms (j, b) of (c b) Q[i - j], c = 1 / G[0].
        const u32 c = mod_inverse<P>(G[0]);
        for (std::size_t t = 0; t < s->count; ++t) s->terms[t].a = mod_mul<P>(s->terms[t].a, c);
        for (std::size_t i = 0; i < k; ++i) {
            const u32 a = mod_mul<P>(c, f[n - 1 - i]), sum = series::tap_sum<P>(*s, q, i);
            q[i] = a >= sum ? a - sum : a + P - sum;
        }
    } else {
        const Out h = scratch.subspan(k, k), F = scratch.first(k);
        series::inverse<P>(G, h, scratch.subspan(2 * k, k));
        std::reverse_copy(f.end() - std::ptrdiff_t(k), f.end(), F.begin());  // G is no longer needed
        Products<P>::multiply_into(F, h, q);
    }
    std::ranges::reverse(q);
}

// The quotient of f by D.g.
template <u32 P>
Polynomial<P> quotient(Span f, const Divisor& D) {
    const std::size_t n = f.size(), d = D.d, k = n > d ? n - d : 0;
    Polynomial<P> q = unwritten_polynomial<P>(k);
    if (k == 0) return q;
    if (D.monomial) scale<P>(f.subspan(d), mod_inverse<P>(D.g[d]), q.span());
    else quotient<P>(f, D.g, d, q.span(), series::scratch(3 * k));
    return q;
}

// The remainder of f by D.g from the quotient q (unread for a monomial D.g or f.size() <= D.d).
template <u32 P>
Polynomial<P> remainder(Span f, const Divisor& D, Span q) {
    const std::size_t d = D.d;
    Polynomial<P> r = unwritten_polynomial<P>(d);
    if (f.size() <= d || D.monomial) {
        const std::size_t m = std::min(f.size(), d);
        std::copy_n(f.begin(), m, r.span().begin());
        std::fill(r.span().begin() + std::ptrdiff_t(m), r.span().end(), 0u);
        return r;
    }
    // r = f - q g mod x^d: the coefficients of q g past d are f's.
    Products<P>::multiply_into(q, D.g.first(d), r.span());
    subtract_from<P>(f.first(d), r.span());
    return r;
}

// The remainder of f by D.g, with the quotient only where the remainder reads it.
template <u32 P>
Polynomial<P> modulo(Span f, const Divisor& D) {
    if (f.size() <= D.d || D.monomial) return remainder<P>(f, D, {});
    const Polynomial<P> q = quotient<P>(f, D);
    return remainder<P>(f, D, q.span());
}

}  // namespace detail::division

template <std::uint32_t P>
std::pair<Polynomial<P>, Polynomial<P>> divmod(const Polynomial<P>& f, const Polynomial<P>& g) {
    const detail::division::Divisor D = detail::division::divisor(g);
    Polynomial<P> q = detail::division::quotient<P>(f.span(), D);
    Polynomial<P> r = detail::division::remainder<P>(f.span(), D, q.span());
    return {std::move(q), std::move(r)};
}

template <std::uint32_t P>
Polynomial<P> operator/(const Polynomial<P>& f, const Polynomial<P>& g) {
    return detail::division::quotient<P>(f.span(), detail::division::divisor(g));
}

template <std::uint32_t P>
Polynomial<P> operator%(const Polynomial<P>& f, const Polynomial<P>& g) {
    return detail::division::modulo<P>(f.span(), detail::division::divisor(g));
}

template <std::uint32_t P>
Polynomial<P>& operator/=(Polynomial<P>& f, const Polynomial<P>& g) {
    return f = f / g;
}

template <std::uint32_t P>
Polynomial<P>& operator%=(Polynomial<P>& f, const Polynomial<P>& g) {
    return f = f % g;
}

}  // namespace fps
