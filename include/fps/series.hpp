// Calculus and power series modulo P = 998244353 (DESIGN.md, Functions: Division and calculus,
// Series; Raw functions).
//
//   fps::derivative(f)              // f': f.size() - 1 coefficients (0 if f is empty)
//   fps::integral(f)                // constant term 0: f.size() + 1 coefficients; f.size() < P
//   fps::inverse(f, n)              // 1 / f mod x^n; f[0] != 0
//   fps::log(f, n)                  // log f mod x^n; f[0] = 1, n <= P
//   fps::exp(f, n)                  // exp f mod x^n; f[0] = 0, n <= P
//   fps::pow(f, e, n)               // f^e mod x^n; any f, any e < 2^64 (0^0 = 1), n <= P
//   fps::sqrt(f, n)                 // std::optional: g with g^2 = f mod x^n (below), or nullopt
//   fps::raw::inverse<P>(f, out)    // the same on spans, n = out.size(): out apart from f
//   fps::raw::log<P>(f, out)        // out may overlap f
//   fps::raw::exp<P>(f, out)        // out may be f (the same first word), else apart from it
//   fps::raw::pow<P>(f, e, out)     // out apart from f
//   fps::raw::sqrt<P>(f, out)       // out apart from f; false (out not written) if no root
//
// n defaults to f.size(). Each function reads only f mod x^n, so the precondition on f[0] holds
// for n > 0 only (an empty f has f[0] = 0). A broken precondition, or out overlapping f where it
// may not, throws std::domain_error. log, exp and pow divide by 1 .. n - 1, which needs n <= P
// (pow: only e mod P counts below x^P).
//
// pow: f mod x^n = x^z u with u[0] != 0 gives f^e = x^(z e) u[0]^e (u / u[0])^e; 0 if z e >= n.
// sqrt: the root of the polynomial f mod x^n = x^(2k) u, u[0] != 0: x^k s with s^2 = u mod
// x^(n - k), s[0] the smaller of the two square roots of u[0] in [0, P); 0 if f mod x^n = 0;
// nullopt if the lowest nonzero term has an odd index or a non-square coefficient (no root).
// On the backend's path, n past 2^25 (pow: n - z e, sqrt: n - k) throws std::length_error before
// out is written. The recurrence has no such limit (n <= P).
//
// Paths. f mod x^n with at most kSparseTerms = 16 nonzero coefficients past f[0] (past u[0] for
// pow and sqrt) runs a linear recurrence with one tap per such term, O(n t) for t terms:
// inverse(1 - x, n) is O(n); pow and sqrt only where it beats the backend (see kPowTerms below).
// Others run the backend (backend/series: lc-opt's Newton steps on the NTT's leaf form):
// inverse, log, exp of f; pow as c (u / u[0])^e with c = u[0]^e, e mod P (lc-opt's power: x g' =
// q g, q = e x u' / u); sqrt of u with g[0] = c, the smaller root. The frontend keeps the
// leading zeros, u[0]^e, e mod P, the choice of root, and the errors.
// From 2^13 coefficients on, in units of one ntt::multiply of two factors of n into n words, on
// the four timed targets: inverse 1.6-2.1, log 2.0-2.9, exp 2.7-3.7 (frontend/notes/87.md), pow
// 4.3-5.1 and sqrt 1.5-1.8 (backend/series/notes/107-r1.md; 0.95-1.02 and 0.94-0.99 of lc-opt's
// time on Zen 3); the API adds nothing measurable (frontend/notes/167.md).
// The recurrences are plain scalar loops: 7.5-21 ns per coefficient for 1 or 2 terms, 15-32 for 16
// (pow, sqrt; frontend/notes/167.md). The ones for pow and sqrt solve u g' = e u' g (e = 1/2 for
// sqrt). Divisions by 1 .. n - 1 take their reciprocals from the backend's factorials::tables:
// 1 / i = (i - 1)! / i!.
//
// Memory: the results are new polynomials; the raw functions write only out. The recurrences
// work in a per-thread scratch kept across calls, of n words for inverse, 2 n for log, pow and
// sqrt, 3 n for exp, f.size() + 1 for integral; the backend uses its own per-thread memory.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>

#include "backend/series/series.hpp"
#include "fps/factorials.hpp"
#include "fps/modular.hpp"
#include "fps/polynomial.hpp"
#include "fps/words.hpp"

namespace fps {

namespace detail::series {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Span = std::span<const u32>;
using Out = std::span<u32>;

inline constexpr std::size_t kSparseTerms = 16;

// This thread's scratch of at least n words, kept and reused. The raw:: functions, integral, power
// and sqrt take it (power and sqrt on their recurrence path only); the other detail functions get
// their part of it as an argument and never call it.
inline Out scratch(std::size_t n) {
    thread_local Words words;
    if (words.size() < n) words = Words(n, Words::Unwritten{});
    return {words.data(), n};
}

inline bool overlaps(Span a, Span b) {
    return !a.empty() && !b.empty() && a.data() < b.data() + b.size() && b.data() < a.data() + a.size();
}

// r[i] = 1 / i for 0 < i < n = r.size() <= P (r[0] = 1), with tmp of n words:
// (i - 1)! / i! from the backend's tables.
template <u32 P>
void reciprocals(Out r, Out tmp) {
    factorial_tables<P>(r, tmp);
    for (std::size_t i = r.size(); i-- > 1;) r[i] = mod_mul<P>(r[i - 1], tmp[i]);
}

// The nonzero terms of f past f[0], if there are at most kSparseTerms of them.
struct Term {
    std::size_t d;
    u32 a;
};

struct Sparse {
    std::array<Term, kSparseTerms> terms;
    std::size_t count = 0;
};

inline std::optional<Sparse> sparse(Span f) {
    Sparse s;
    for (std::size_t i = 1; i < f.size(); ++i) {
        if (f[i] == 0) continue;
        if (s.count == kSparseTerms) return std::nullopt;
        s.terms[s.count++] = {i, f[i]};
    }
    return s;
}

// sum over terms (d, b) with d <= i of b g[i - d], mod P: at most 16 products < P^2 < 2^60.
template <u32 P>
u32 tap_sum(const Sparse& s, Span g, std::size_t i) {
    u64 sum = 0;
    for (std::size_t t = 0; t < s.count && s.terms[t].d <= i; ++t) sum += u64(s.terms[t].a) * g[i - s.terms[t].d];
    return u32(sum % P);
}

// g = 1 / f by f[0] g[i] = [i = 0] - sum_d f[d] g[i - d].
template <u32 P>
void sparse_inverse(u32 f0, Sparse s, Out g) {
    const u32 c = mod_inverse<P>(f0);
    for (std::size_t t = 0; t < s.count; ++t) s.terms[t].a = mod_mul<P>(s.terms[t].a, c);
    g[0] = c;
    for (std::size_t i = 1; i < g.size(); ++i) {
        const u32 sum = tap_sum<P>(s, g, i);
        g[i] = sum ? P - sum : 0;
    }
}

// g = log f, f[0] = 1: f g' = f' gives G = i g[i] by G[i] = i f[i] - sum_d f[d] G[i - d].
// tmp: 2 n words.
template <u32 P>
void sparse_log(const Sparse& s, Out g, Out tmp) {
    const std::size_t n = g.size();
    g[0] = 0;
    for (std::size_t i = 1, t = 0; i < n; ++i) {
        u32 rhs = 0;
        if (t < s.count && s.terms[t].d == i) rhs = mod_mul<P>(u32(i), s.terms[t++].a);
        const u32 sum = tap_sum<P>(s, g, i);
        g[i] = rhs >= sum ? rhs - sum : rhs + P - sum;
    }
    const Out r = tmp.first(n);
    reciprocals<P>(r, tmp.subspan(n, n));
    for (std::size_t i = 1; i < n; ++i) g[i] = mod_mul<P>(g[i], r[i]);
}

// g = exp f, f[0] = 0: g' = f' g gives i g[i] = sum_d d f[d] g[i - d]. tmp: 2 n words.
template <u32 P>
void sparse_exp(Sparse s, Out g, Out tmp) {
    const std::size_t n = g.size();
    for (std::size_t t = 0; t < s.count; ++t) s.terms[t].a = mod_mul<P>(u32(s.terms[t].d % P), s.terms[t].a);
    const Out r = tmp.first(n);
    reciprocals<P>(r, tmp.subspan(n, n));
    g[0] = 1;
    for (std::size_t i = 1; i < n; ++i) g[i] = mod_mul<P>(tap_sum<P>(s, g, i), r[i]);
}

// The functions below: f is f mod x^n (at most n words), out has n >= 1 words, f's precondition
// holds, and scratch has the words each states.

// out = 1 / f mod x^n; out apart from f. scratch: n words (the recurrence's; the Newton path
// runs in the backend's own memory).
template <u32 P>
void inverse(Span f, Out out, Out scratch) {
    (void)scratch;
    if (const auto s = sparse(f)) return sparse_inverse<P>(f[0], *s, out);
    fps::series::inverse(f, out);
}

// out = log f mod x^n; out may overlap f, which is read before out is written. scratch: 2 n words.
template <u32 P>
void log(Span f, Out out, Out scratch) {
    if (const auto s = sparse(f)) return sparse_log<P>(*s, out, scratch);
    fps::series::log(f, out);
}

// out = exp f mod x^n; out may overlap f, which is read before out is written. scratch: 3 n
// words.
template <u32 P>
void exp(Span f, Out out, Out scratch) {
    if (const auto s = sparse(f)) return sparse_exp<P>(*s, out, scratch);
    fps::series::exp(f, out);
}

// The index of f's first nonzero coefficient, f.size() if none.
inline std::size_t leading_zeros(Span f) {
    return std::size_t(std::ranges::find_if(f, [](u32 c) { return c != 0; }) - f.begin());
}

// g = c (u / u[0])^e mod x^n, n = g.size() <= P, for the terms s of u past u[0]. u g' = e u' g gives
// i g[i] = sum_d a_d ((e + 1) d - i) g[i - d] for a_d = u[d] / u[0], so
// g[i] = (sum_d (e + 1) d a_d g[i - d]) / i - sum_d a_d g[i - d]. tmp: 2 n words.
template <u32 P>
void sparse_power(u32 u0, Sparse a, u32 e, u32 c, Out g, Out tmp) {
    const std::size_t n = g.size();
    g[0] = c;
    if (a.count == 0) return std::fill(g.begin() + 1, g.end(), 0u);  // u = u[0]: a monomial
    const u32 inverse0 = mod_inverse<P>(u0), e1 = u32((u64(e) + 1) % P);
    Sparse b = a;
    for (std::size_t t = 0; t < a.count; ++t) {
        a.terms[t].a = mod_mul<P>(a.terms[t].a, inverse0);
        b.terms[t].a = mod_mul<P>(mod_mul<P>(e1, u32(a.terms[t].d % P)), a.terms[t].a);
    }
    const Out r = tmp.first(n);
    reciprocals<P>(r, tmp.subspan(n, n));
    for (std::size_t i = 1; i < n; ++i) {
        const u32 x = mod_mul<P>(tap_sum<P>(b, g, i), r[i]), y = tap_sum<P>(a, g, i);
        g[i] = x >= y ? x - y : x + P - y;
    }
}

// Where the recurrence beats the backend for t <= kSparseTerms terms past u[0] (frontend/notes/
// 167.md): pow for t <= kPowTerms at any n, and any t from kPowLong coefficients on (all four
// timed targets); sqrt for t <= kRootTerms up to kRootShort coefficients (all four) and from
// kRootLong on (0.54-0.88 of the backend's time, but 1.26-1.40 on Zen 5, whose backend sqrt is
// fastest; backend issue #171). Past the backend's kBackendMax both run it (n <= P), where the
// backend has no plan. Speed choices only: the paths give the same results.
inline constexpr std::size_t kPowTerms = 8, kRootTerms = 4;
inline constexpr std::size_t kPowLong = std::size_t(1) << 17;
inline constexpr std::size_t kRootShort = std::size_t(1) << 8, kRootLong = std::size_t(1) << 18;
inline constexpr std::size_t kBackendMax = std::size_t(1) << ntt::kMaxLog;

// out = c (u / u[0])^e mod x^n, n = out.size() <= P, for u[0] != 0, u of at most n words and
// e < P (only e mod P counts below x^P); out apart from u. The recurrence works in this thread's
// scratch (2 n words); the backend in its own memory, and throws std::length_error for n past
// 2^25 before out is written.
template <u32 P>
void power(Span u, u32 e, u32 c, Out out) {
    const std::size_t n = out.size();
    if (const auto s = sparse(u); s && (s->count <= kPowTerms || n >= kPowLong))
        return sparse_power<P>(u[0], *s, e, c, out, scratch(2 * n));
    fps::series::pow(u, e, c, out);
}

// out = the root g of u mod x^n with g[0] = c, c^2 = u[0] != 0, for u of at most n words; out
// apart from u. As power; the recurrence divides by 1 .. n - 1, so it runs for n <= P only.
template <u32 P>
void sqrt(Span u, u32 c, Out out) {
    const std::size_t n = out.size();
    const auto s = sparse(u);
    if (s && n <= P && ((s->count <= kRootTerms && (n <= kRootShort || n >= kRootLong)) || n > kBackendMax))
        return sparse_power<P>(u[0], *s, (P + 1) / 2, c, out, scratch(2 * n));
    fps::series::sqrt(u, c, out);
}

}  // namespace detail::series

namespace raw {

template <std::uint32_t P>
void inverse(std::span<const std::uint32_t> f, std::span<std::uint32_t> out) {
    static_assert(P == ntt::kModulus, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");
    const std::size_t n = out.size();
    if (n == 0) return;
    f = f.first(std::min(f.size(), n));
    if (f.empty() || f[0] == 0) throw std::domain_error("fps::inverse: f[0] = 0");
    if (detail::series::overlaps(f, out)) throw std::domain_error("fps::raw::inverse: out overlaps f");
    detail::series::inverse<P>(f, out, detail::series::scratch(n));
}

template <std::uint32_t P>
void log(std::span<const std::uint32_t> f, std::span<std::uint32_t> out) {
    static_assert(P == ntt::kModulus, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");
    const std::size_t n = out.size();
    if (n == 0) return;
    f = f.first(std::min(f.size(), n));
    if (f.empty() || f[0] != 1) throw std::domain_error("fps::log: f[0] != 1");
    if (n > P) throw std::domain_error("fps::log: n > P");
    detail::series::log<P>(f, out, detail::series::scratch(2 * n));
}

template <std::uint32_t P>
void exp(std::span<const std::uint32_t> f, std::span<std::uint32_t> out) {
    static_assert(P == ntt::kModulus, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");
    const std::size_t n = out.size();
    if (n == 0) return;
    f = f.first(std::min(f.size(), n));
    if (!f.empty() && f[0] != 0) throw std::domain_error("fps::exp: f[0] != 0");
    if (n > P) throw std::domain_error("fps::exp: n > P");
    if (f.data() != out.data() && detail::series::overlaps(f, out))
        throw std::domain_error("fps::raw::exp: out overlaps f but is not f");
    detail::series::exp<P>(f, out, detail::series::scratch(3 * n));
}

template <std::uint32_t P>
void pow(std::span<const std::uint32_t> f, std::uint64_t e, std::span<std::uint32_t> out) {
    static_assert(P == ntt::kModulus, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");
    const std::size_t n = out.size();
    if (n == 0) return;
    if (n > P) throw std::domain_error("fps::pow: n > P");
    f = f.first(std::min(f.size(), n));
    if (detail::series::overlaps(f, out)) throw std::domain_error("fps::raw::pow: out overlaps f");
    const std::size_t z = detail::series::leading_zeros(f);
    if (e == 0 || z == f.size() || z > (n - 1) / e) {  // 1, or 0 when z e >= n
        std::ranges::fill(out, 0u);
        out[0] = e == 0;
        return;
    }
    const std::size_t shift = z * std::size_t(e), m = n - shift;
    const std::span<const std::uint32_t> u = f.subspan(z, std::min(f.size() - z, m));
    detail::series::power<P>(u, std::uint32_t(e % P), mod_pow<P>(u[0], e), out.subspan(shift));
    std::fill_n(out.begin(), shift, 0u);
}

template <std::uint32_t P>
bool sqrt(std::span<const std::uint32_t> f, std::span<std::uint32_t> out) {
    static_assert(P == ntt::kModulus, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");
    const std::size_t n = out.size();
    if (n == 0) return true;
    f = f.first(std::min(f.size(), n));
    if (detail::series::overlaps(f, out)) throw std::domain_error("fps::raw::sqrt: out overlaps f");
    const std::size_t z = detail::series::leading_zeros(f);
    if (z == f.size()) return std::ranges::fill(out, 0u), true;
    const std::optional<std::uint32_t> c = detail::mod_sqrt<P>(f[z]);
    if (z % 2 || !c) return false;
    detail::series::sqrt<P>(f.subspan(z), *c, out.subspan(z / 2));
    std::fill_n(out.begin(), z / 2, 0u);
    return true;
}

}  // namespace raw

template <std::uint32_t P>
Polynomial<P> derivative(const Polynomial<P>& f) {
    const std::size_t n = f.size() ? f.size() - 1 : 0;
    Polynomial<P> d = detail::unwritten_polynomial<P>(n);
    for (std::size_t i = 0; i < n; ++i) d[i] = mod_mul<P>(std::uint32_t((i + 1) % P), f[i + 1]);
    return d;
}

template <std::uint32_t P>
Polynomial<P> integral(const Polynomial<P>& f) {
    const std::size_t n = f.size() + 1;
    if (f.size() >= P) throw std::domain_error("fps::integral: f.size() >= P");
    Polynomial<P> q = detail::unwritten_polynomial<P>(n);
    detail::series::reciprocals<P>(q.span(), detail::series::scratch(n));
    q[0] = 0;
    for (std::size_t i = 1; i < n; ++i) q[i] = mod_mul<P>(q[i], f[i - 1]);
    return q;
}

template <std::uint32_t P>
Polynomial<P> inverse(const Polynomial<P>& f, std::size_t n) {
    Polynomial<P> g = detail::unwritten_polynomial<P>(n);
    raw::inverse<P>(f.span(), g.span());
    return g;
}

template <std::uint32_t P>
Polynomial<P> inverse(const Polynomial<P>& f) {
    return inverse(f, f.size());
}

template <std::uint32_t P>
Polynomial<P> log(const Polynomial<P>& f, std::size_t n) {
    Polynomial<P> g = detail::unwritten_polynomial<P>(n);
    raw::log<P>(f.span(), g.span());
    return g;
}

template <std::uint32_t P>
Polynomial<P> log(const Polynomial<P>& f) {
    return log(f, f.size());
}

template <std::uint32_t P>
Polynomial<P> exp(const Polynomial<P>& f, std::size_t n) {
    Polynomial<P> g = detail::unwritten_polynomial<P>(n);
    raw::exp<P>(f.span(), g.span());
    return g;
}

template <std::uint32_t P>
Polynomial<P> exp(const Polynomial<P>& f) {
    return exp(f, f.size());
}

template <std::uint32_t P>
Polynomial<P> pow(const Polynomial<P>& f, std::uint64_t e, std::size_t n) {
    Polynomial<P> g = detail::unwritten_polynomial<P>(n);
    raw::pow<P>(f.span(), e, g.span());
    return g;
}

template <std::uint32_t P>
Polynomial<P> pow(const Polynomial<P>& f, std::uint64_t e) {
    return pow(f, e, f.size());
}

template <std::uint32_t P>
std::optional<Polynomial<P>> sqrt(const Polynomial<P>& f, std::size_t n) {
    Polynomial<P> g = detail::unwritten_polynomial<P>(n);
    if (!raw::sqrt<P>(f.span(), g.span())) return std::nullopt;
    return g;
}

template <std::uint32_t P>
std::optional<Polynomial<P>> sqrt(const Polynomial<P>& f) {
    return sqrt(f, f.size());
}

}  // namespace fps
