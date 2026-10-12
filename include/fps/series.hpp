// Calculus and power series modulo P = 998244353 (DESIGN.md, Functions: Division and calculus,
// Series; Raw functions).
//
//   fps::derivative(f)              // f': f.size() - 1 coefficients (0 if f is empty)
//   fps::integral(f)                // constant term 0: f.size() + 1 coefficients; f.size() < P
//   fps::inverse(f, n)              // 1 / f mod x^n; f[0] != 0
//   fps::log(f, n)                  // log f mod x^n; f[0] = 1, n <= P
//   fps::exp(f, n)                  // exp f mod x^n; f[0] = 0, n <= P
//   fps::raw::inverse<P>(f, out)    // the same on spans, n = out.size(): out apart from f
//   fps::raw::log<P>(f, out)        // out may overlap f
//   fps::raw::exp<P>(f, out)        // out may be f (the same first word), else apart from it
//
// n defaults to f.size(). Each function reads only f mod x^n, so the precondition on f[0] holds
// for n > 0 only (an empty f has f[0] = 0). A broken precondition, or out overlapping f where it
// may not, throws std::domain_error. log and exp divide by 1 .. n - 1, which needs n <= P.
// On the Newton path, n past 2^25 throws std::length_error from ntt::multiply in the first step
// whose precision passes 2^25, after the earlier steps ran (out is then partly written); the
// recurrence has no such limit.
//
// Paths. f mod x^n with at most kSparseTerms = 16 nonzero coefficients past f[0] runs a linear
// recurrence with one tap per such term, O(n t) for t terms: inverse(1 - x, n) is O(n). Others run
// Newton iteration on the products of fps/products.hpp (raw::mul: shift, small::multiply or
// ntt::multiply), doubling the precision k each step:
// - inverse: g[k, 2k) = -(g (f g mod x^2k)[k, 2k)) mod x^k, two products a step;
// - log f = integral of f' (1 / f) mod x^(n-1): one inverse and one product;
// - exp: g[k, 2k) = (g (f - log g)[k, 2k)) mod x^k: one log of g and one product a step.
// The backend has no series kernels yet. From 2^13 coefficients on, in units of one ntt::multiply
// of two factors of n into n words, on the four timed targets: inverse 2.6-2.9, log 4.1-4.8, exp
// 7.6-9.8 (frontend/notes.md); lc-opt's, with transforms of length 2k only and halves reused
// across steps, need about 1.7, 2 and 2.6 (a guess from its transform counts): backend issue #87.
// The recurrences are plain scalar loops: 3.7 to 12.5 ns per coefficient for 2 or 3 terms (#87).
// Divisions by 1 .. n - 1 take their reciprocals from the backend's factorials::tables:
// 1 / i = (i - 1)! / i!.
//
// Memory: the results are new polynomials; the raw functions write only out. Both work in a
// per-thread scratch kept across calls, of n words for inverse, 2 n for log, 3 n for exp and
// f.size() + 1 for integral; the products use the backend's own per-thread memory.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>

#include "fps/factorials.hpp"
#include "fps/modular.hpp"
#include "fps/polynomial.hpp"
#include "fps/products.hpp"
#include "fps/words.hpp"

namespace fps {

namespace detail::series {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Span = std::span<const u32>;
using Out = std::span<u32>;

inline constexpr std::size_t kSparseTerms = 16;

// This thread's scratch of at least n words, kept and reused. The raw:: functions and integral
// take it; the detail functions below get their part of it as an argument and never call it.
inline Out scratch(std::size_t n) {
    thread_local Words words;
    if (words.size() < n) words = Words(n, Words::Unwritten{});
    return {words.data(), n};
}

inline bool overlaps(Span a, Span b) {
    return !a.empty() && !b.empty() && a.data() < b.data() + b.size() && b.data() < a.data() + a.size();
}

// f[i], 0 past the end.
inline u32 at(Span f, std::size_t i) { return i < f.size() ? f[i] : 0; }

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

// out = 1 / f mod x^n; out apart from f. scratch: n words.
template <u32 P>
void inverse(Span f, Out out, Out scratch) {
    const std::size_t n = out.size();
    if (const auto s = sparse(f)) return sparse_inverse<P>(f[0], *s, out);
    out[0] = mod_inverse<P>(f[0]);
    for (std::size_t k = 1; k < n;) {
        const std::size_t m = std::min(2 * k, n);
        // e = f g mod x^m = 1 + x^k e[k, m); then g[k, m) = -(g e[k, m) mod x^(m - k)).
        const Out e = scratch.first(m), high = out.subspan(k, m - k);
        raw::mul<P>(f.first(std::min(f.size(), m)), out.first(k), e);
        raw::mul<P>(out.first(k), e.subspan(k), high);
        for (u32& c : high) c = c ? P - c : 0;
        k = m;
    }
}

// out = log f mod x^n; out may overlap f, which is read before out is written. scratch: 2 n words.
template <u32 P>
void log(Span f, Out out, Out scratch) {
    const std::size_t n = out.size();
    if (const auto s = sparse(f)) return sparse_log<P>(*s, out, scratch);
    // f' / f mod x^(n - 1) into out[1, n), then divided by the index.
    const Out h = scratch.first(n - 1), d = scratch.subspan(n, n - 1);
    inverse<P>(f.first(std::min(f.size(), n - 1)), h, d);
    for (std::size_t i = 0; i + 1 < n; ++i) d[i] = mod_mul<P>(u32((i + 1) % P), at(f, i + 1));
    raw::mul<P>(d, h, out.subspan(1));
    const Out r = scratch.first(n);
    reciprocals<P>(r, scratch.subspan(n, n));
    out[0] = 0;
    for (std::size_t i = 1; i < n; ++i) out[i] = mod_mul<P>(out[i], r[i]);
}

// out = exp f mod x^n; out may be f (the same first word: step k reads f[k, 2k) before it writes
// there), else apart from it. scratch: 3 n words.
template <u32 P>
void exp(Span f, Out out, Out scratch) {
    const std::size_t n = out.size();
    if (const auto s = sparse(f)) return sparse_exp<P>(*s, out, scratch);
    out[0] = 1;
    for (std::size_t k = 1; k < n;) {
        const std::size_t m = std::min(2 * k, n);
        // h = f - log g mod x^m = x^k h[k, m); then g[k, m) = g h[k, m) mod x^(m - k).
        const Out h = scratch.first(m);
        log<P>(out.first(k), h, scratch.subspan(n, 2 * n));
        for (std::size_t i = k; i < m; ++i) h[i] = at(f, i) >= h[i] ? at(f, i) - h[i] : at(f, i) + P - h[i];
        raw::mul<P>(out.first(k), h.subspan(k), out.subspan(k, m - k));
        k = m;
    }
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

}  // namespace fps
