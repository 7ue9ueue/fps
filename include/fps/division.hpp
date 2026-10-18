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
// Paths, by one scan of g from the top (it stops at the 17th nonzero coefficient below g[d]):
// - g = c x^d: a shift and a scaling, O(n): f / x^k drops the lowest k coefficients, f % x^k is
//   f.truncate(k).
// - g with t <= 16 nonzero coefficients below g[d], all at least 8 (t + 1) below it (x^d - 1 for
//   d >= 16, x^d - x - 1 for d >= 25): long division by those terms, in blocks of quotient words
//   that do not reach each other, O(n t) in vector passes.
// - Else g with t <= 2 such terms (x^3 + 2 x + 5, x^d - x^(d-1) - 1): the recurrence q[i] =
//   (f[d + i] - sum g[j] q[i + d - j]) / g[d] word by word, then r likewise, O(n t).
// - Every other g: one call into the backend's kernel, fps::division::divide
//   (backend/division/division.hpp: long division for k d <= 2^12, else lc-opt's blocked division
//   on ntt::Transform with a tuned plan per target), on g trimmed to its degree (the kernel needs
//   g[m - 1] != 0). f / g asks it for no remainder; f % g gives it a quotient in this thread's
//   scratch (k words).
// Every path writes q and r into the returned memory. Times: backend/division/notes.md (the kernel
// against lc-opt) and frontend/notes/147.md (API against kernel, and against the frontend's own
// Newton division and few-term recurrence before #147).
//
// Long inputs: the kernel splits the quotient into blocks and the remainder into products of at
// most 2^25 words (f of 2^25 + 2^10 by a dense g of 2^25 + 1 runs, checked by hand); it throws
// std::length_error for the shapes no such plan covers (longer divisors). The other paths have no
// limit.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>

#include "backend/division/division.hpp"
#include "backend/modarith/modarith.hpp"
#include "backend/simd/align.hpp"
#include "backend/simd/select.hpp"
#include "fps/modular.hpp"
#include "fps/polynomial.hpp"
#include "fps/series.hpp"

namespace fps {

namespace detail::division {

using u32 = std::uint32_t;
using Span = std::span<const u32>;
using Out = std::span<u32>;

// Few terms below g[d], t of them, the highest gap below d (frontend/notes/147.md):
// - blocks: t <= kSparseTerms and gap >= kSparseGap (t + 1). About t + 1 vector passes per quotient
//   word plus (t + 1) / gap calls; the kernel's is a few products' worth.
// - recurrence: else t <= kRecurrenceTerms. Word by word, t products and one reduction each; the
//   kernel takes 1.2-2.1 times as long for x^3 + 2x + 5, but 0.75-0.85 on Zen 5. Moving both into
//   the kernel's plan, picked per target by its cost model: backend issue #158.
inline constexpr std::size_t kSparseTerms = 16;
inline constexpr std::size_t kSparseGap = 8;
inline constexpr std::size_t kRecurrenceTerms = 2;
// arm: the nops after the 32-byte alignment before the recurrence's loop (divide_recurrence). With
// GCC 15.2's and 16.2's setup code in between, it starts at 4 mod 32: the fastest start measured,
// 0.1-0.9% ahead of 8 and 12, 1-3% of 0 and 16 (frontend/notes/230-r1.md).
inline constexpr int kRecurrenceLoopNops = 2;

// g trimmed to d + 1 words, d = deg g, and its nonzero coefficients below g[d] from the top down,
// if at most kSparseTerms (count = kSparseTerms + 1: more). Throws for g = 0.
struct Divisor {
    Span g;
    std::size_t d;
    std::array<std::size_t, kSparseTerms> terms;
    std::size_t count;

    bool monomial() const { return count == 0; }
    bool blocks() const { return count > 0 && count <= kSparseTerms && d - terms[0] >= kSparseGap * (count + 1); }
    bool recurrence() const { return count > 0 && count <= kRecurrenceTerms; }  // when not blocks()
};

template <u32 P>
Divisor divisor(const Polynomial<P>& g) {
    static_assert(P == ntt::kModulus, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");
    const int deg = g.degree();
    if (deg < 0) throw std::domain_error("fps: division by zero");
    const std::size_t d = std::size_t(deg);
    Divisor D{g.span().first(d + 1), d, {}, 0};
    for (std::size_t hi = d; hi > 0;) {  // chunks of 64 words, zero ones skipped by one vector OR
        const std::size_t lo = hi > 64 ? hi - 64 : 0;
        u32 any = 0;
        for (std::size_t j = lo; j < hi; ++j) any |= D.g[j];
        for (std::size_t j = hi; any && j-- > lo;) {
            if (D.g[j] == 0) continue;
            if (D.count == kSparseTerms) {
                D.count = kSparseTerms + 1;
                return D;
            }
            D.terms[D.count++] = j;
        }
        hi = lo;
    }
    return D;
}

inline std::size_t quotient_size(std::size_t n, const Divisor& D) { return n > D.d ? n - D.d : 0; }

// out[i] = c a[i] (Shoup's product); out may be a.
template <u32 P>
void scale(Span a, u32 c, Out out) {
    const u32 s = modarith::Ops<simd::Scalar, P>::shoup_quotient(c);
    lanewise(a.data(), a.data(), out.data(), a.size(), [c, s]<class S>(typename S::V x, typename S::V) {
        using Ops = modarith::Ops<S, P>;
        return Ops::template reduce<P>(Ops::shoup(x, S::broadcast(c), S::broadcast(s)));
    });
}

// out[i] -= c a[i], i < a.size(); out apart from a.
template <u32 P>
void subtract_multiple(Span a, u32 c, Out out) {
    const u32 s = modarith::Ops<simd::Scalar, P>::shoup_quotient(c);
    lanewise(out.data(), a.data(), out.data(), a.size(), [c, s]<class S>(typename S::V x, typename S::V y) {
        using Ops = modarith::Ops<S, P>;
        return Ops::sub(x, Ops::template reduce<P>(Ops::shoup(y, S::broadcast(c), S::broadcast(s))));
    });
}

// r = f mod x^d for g = c x^d (or n <= d): f's words, then zeros.
inline void copy_remainder(Span f, Out r) {
    const std::size_t m = std::min(f.size(), r.size());
    std::copy_n(f.begin(), m, r.begin());
    std::fill(r.begin() + std::ptrdiff_t(m), r.end(), 0u);
}

// g = c x^d: q = f[d, n) / c (a copy for c = 1), r = f mod x^d.
template <u32 P>
void divide_monomial(Span f, const Divisor& D, Out q, Out r) {
    const u32 c = mod_inverse<P>(D.g[D.d]);
    copy_remainder(f, r);
    if (q.empty()) return;
    if (c == 1) std::ranges::copy(f.subspan(D.d), q.begin());
    else scale<P>(f.subspan(D.d), c, q);
}

// D.blocks(). The rest of f, f - g (q's words found so far), lives in q (its word d + i in q[i])
// and r (its words below d). Quotient words i in [lo, hi) with hi - lo <= gap = d - terms[0] do
// not reach each other: word i + j of the rest, j <= terms[0], is below lo + d. So each block is
// final once the blocks above it are subtracted: scale it by 1 / g[d], then subtract it times each
// term from the rest below it.
template <u32 P>
void divide_blocks(Span f, const Divisor& D, Out q, Out r) {
    const std::size_t d = D.d, k = q.size();
    const u32 c = mod_inverse<P>(D.g[d]);
    copy_remainder(f, r);
    if (k == 0) return;
    std::ranges::copy(f.subspan(d), q.begin());
    const std::size_t gap = d - D.terms[0];
    for (std::size_t hi = k; hi > 0;) {
        const std::size_t lo = hi > gap ? hi - gap : 0;
        const Out block = q.subspan(lo, hi - lo);
        scale<P>(block, c, block);
        for (std::size_t t = 0; t < D.count; ++t) {
            const std::size_t j = D.terms[t];
            const u32 a = D.g[j];
            // Rest words [lo + j, hi + j): those >= d in q, those below d in r.
            const std::size_t split = std::clamp(d, lo + j, hi + j);
            if (split < hi + j) subtract_multiple<P>(Span(block).subspan(split - lo - j), a, q.subspan(split - d));
            if (!r.empty() && lo + j < split) subtract_multiple<P>(Span(block).first(split - lo - j), a, r.subspan(lo + j));
        }
        hi = lo;
    }
}

// D.recurrence(): q[i] = (f[d + i] - sum over terms j of g[j] q[i + d - j]) / g[d] from the top
// down, as Q[p] = c F[p] - sum (c g[j]) Q[p - (d - j)] for Q = rev(q), F = rev(f), c = 1 / g[d];
// then q = rev(Q) and r[m] = f[m] - sum g[j] q[m - j]. Each word is one reduction of a 64-bit
// sum: 2 P^2 plus one product minus at most 2, below 3 P^2 < 2^62 (no compare: GCC turned one
// into a mispredicted branch, 2 times as slow).
template <u32 P>
void divide_recurrence(Span f, const Divisor& D, Out q, Out r) {
    using u64 = std::uint64_t;
    const std::size_t d = D.d, k = q.size(), t = D.count, n = f.size();
    if (k == 0) return copy_remainder(f, r);
    constexpr u64 kBias = 2 * u64(P) * P;
    const u32 c = mod_inverse<P>(D.g[d]);
    std::array<u32, kRecurrenceTerms> a{}, ca{};
    std::array<std::size_t, kRecurrenceTerms> e{};  // distances d - j
    for (std::size_t u = 0; u < t; ++u) a[u] = D.g[D.terms[u]], ca[u] = mod_mul<P>(c, a[u]), e[u] = d - D.terms[u];
#if defined(__aarch64__)
    // arm: every term each word (an absent one has e = k: never reached), so no inner loop, and the
    // loop pinned (simd/align.hpp; frontend/notes/230-r1.md). The inner loop over t (10
    // instructions) ran 8% slower where it took 3 blocks of 32 bytes, as with GCC 15.2; without it
    // x^13 + a x + b takes 16-19% less time.
    for (std::size_t u = t; u < kRecurrenceTerms; ++u) e[u] = k;
    const u32* fp = f.data() + n - 1;
    u32* qp = q.data();
    std::size_t p = 0;
    simd::align_loop<kRecurrenceLoopNops>(fp, qp, p);
    do {
        u64 s = kBias + u64(c) * *fp--;
        for (std::size_t u = 0; u < kRecurrenceTerms; ++u)
            if (e[u] <= p) s -= u64(ca[u]) * qp[-std::ptrdiff_t(e[u])];
        *qp++ = u32(s % P);
    } while (++p != k);
#else
    for (std::size_t p = 0; p < k; ++p) {
        u64 s = kBias + u64(c) * f[n - 1 - p];
        for (std::size_t u = 0; u < t; ++u)
            if (e[u] <= p) s -= u64(ca[u]) * q[p - e[u]];
        q[p] = u32(s % P);
    }
#endif
    std::ranges::reverse(q);
    for (std::size_t m = 0; m < r.size(); ++m) {
        u64 s = kBias + f[m];
        for (std::size_t u = 0; u < t; ++u)
            if (m >= D.terms[u] && m - D.terms[u] < k) s -= u64(a[u]) * q[m - D.terms[u]];
        r[m] = u32(s % P);
    }
}

// q of quotient_size(n, D) words and r of D.d words or none (no remainder).
template <u32 P>
void divide(Span f, const Divisor& D, Out q, Out r) {
    if (D.monomial()) return divide_monomial<P>(f, D, q, r);
    if (D.blocks()) return divide_blocks<P>(f, D, q, r);
    if (D.recurrence()) return divide_recurrence<P>(f, D, q, r);
    fps::division::divide(f, D.g, q, r);
}

}  // namespace detail::division

template <std::uint32_t P>
std::pair<Polynomial<P>, Polynomial<P>> divmod(const Polynomial<P>& f, const Polynomial<P>& g) {
    const detail::division::Divisor D = detail::division::divisor(g);
    Polynomial<P> q = detail::unwritten_polynomial<P>(detail::division::quotient_size(f.size(), D));
    Polynomial<P> r = detail::unwritten_polynomial<P>(D.d);
    detail::division::divide<P>(f.span(), D, q.span(), r.span());
    return {std::move(q), std::move(r)};
}

template <std::uint32_t P>
Polynomial<P> operator/(const Polynomial<P>& f, const Polynomial<P>& g) {
    const detail::division::Divisor D = detail::division::divisor(g);
    Polynomial<P> q = detail::unwritten_polynomial<P>(detail::division::quotient_size(f.size(), D));
    detail::division::divide<P>(f.span(), D, q.span(), {});
    return q;
}

template <std::uint32_t P>
Polynomial<P> operator%(const Polynomial<P>& f, const Polynomial<P>& g) {
    const detail::division::Divisor D = detail::division::divisor(g);
    Polynomial<P> r = detail::unwritten_polynomial<P>(D.d);
    if (D.monomial()) {
        detail::division::copy_remainder(f.span(), r.span());
        return r;
    }
    const std::span<std::uint32_t> q = detail::series::scratch(detail::division::quotient_size(f.size(), D));
    detail::division::divide<P>(f.span(), D, q, r.span());
    return r;
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
