// Geometric points, the shift of sampling points, and the DFT modulo P = 998244353 (DESIGN.md,
// Functions: Evaluation and interpolation; Values).
//
//   fps::geometric(a, r, m)                   // the m points a r^i, i < m
//   fps::evaluate(f, fps::geometric(a, r, m)) // f(a r^i), i < m: m values (chirp z-transform)
//   fps::interpolate<P>(fps::geometric(a, r, m), values)
//                                             // the f with f.size() = m and f(a r^i) = values[i]
//   fps::extrapolate<P>(values, c, m)         // f(c + i), i < m, for the f of degree < n with
//                                             // f(j) = values[j], j < n = values.size()
//   fps::dft(f, L)                            // f(w^i), i < L: L values
//   fps::idft<P>(values)                      // the f with f.size() = L and f(w^i) = values[i]
//   fps::raw::dft<P>(f, out)                  // dft(f, L) into out, L = out.size(); out may overlap f
//   fps::raw::idft<P>(values, out)            // idft(values) into out, out.size() = values.size();
//                                             // out may overlap values; values unchecked (< P)
//
// w = 3^((P - 1) / L), a primitive L-th root of unity (3 generates the units mod P). The order is
// natural: dft(f, L)[i] = f(w^i), for any f.size(), so it is the DFT of f mod (x^L - 1), and
// idft(dft(f, L)) is f mod (x^L - 1). L is 0 (no values) or divides P - 1 = 2^23 7 17: powers of
// two and others alike. So among powers of two they stop at 2^23, while to_values and the
// products go on to 2^25: the NTT stops at leaves x^8 - w, whose w need only order L / 8; true
// values at 2^24 or 2^25 points would need roots of unity that do not exist mod P.
// Lengths: for L = 2^6 .. 2^23, any f.size(); for other L, dft needs f.size() + L - 1 <= 2^25 (or
// f.size() <= 1), idft L <= 2^24; past that, std::length_error before anything is allocated (e.g.
// dft at L = 17 2^21, idft at 7 2^22).
//
// Points, values, a, r and c are canonical, in [0, P); one >= P throws std::domain_error. evaluate
// takes any a and r (0 included; repeated points too). interpolate needs as many values as points,
// and distinct points (a != 0 for m >= 2, r != 0 for m >= 3, r^k != 1 for 0 < k < m); else
// std::domain_error. extrapolate needs n <= P (std::domain_error); any c, and c + i wraps mod P.
// idft needs L = values.size() to be 0 or a divisor of P - 1, as dft its L (std::domain_error:
// L = 2^24 and 2^25 too). raw::dft and raw::idft throw the same errors, and raw::idft
// std::domain_error if out.size() != values.size(), all before writing; raw::idft does not check
// that the values are below P (as raw::to_values), a check that costs 4-8% of idft at 2^20.
//
// Paths: one backend call each, into the result's memory (raw::dft, raw::idft: out).
// - dft, idft for L = 2^6 .. 2^23: ntt::dft, ntt::idft (backend/ntt/dft.hpp, issue #138): the
//   NTT's forward transform of length L (of f mod (x^L - 1)) and a permutation into natural order;
//   idft the reverse, its 1 / L in the inverse's scale.
// - evaluate, other dft: chirp::evaluate_geometric (backend/chirp), one ntt::middle_product of
//   n + m - 1 by n words.
// - interpolate: chirp::interpolate_geometric, two ntt::multiply of m words mod x^m, about twice
//   evaluate's time (Newton's form on the points, then back to coefficients).
// - extrapolate: chirp::extrapolate, one ntt::middle_product of (count + n - 1) by n words per run
//   of points c + i outside 0 .. n - 1; points inside are copied from values.
// - other idft: evaluate_geometric at w^-1, then times 1 / L (chirp::multiply_chirp).
// The middle products and products throw std::length_error past 2^25 words (interpolate: m > 2^25,
// before any work); small sizes run Horner or divided differences and have no limit.
//
// Memory: the results are new vectors and polynomials; the raw functions write only out. The
// kernels work in a per-thread scratch, grown to the longest call and kept; so does the chirp's
// copy of the input of raw::dft and raw::idft when out overlaps it (the chirp needs them apart).
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "backend/chirp/chirp.hpp"
#include "backend/chirp/geometric.hpp"
#include "backend/chirp/shift.hpp"
#include "backend/ntt/dft.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/simd/select.hpp"
#include "fps/modular.hpp"
#include "fps/polynomial.hpp"

namespace fps {

// The m points a r^i, i < m. Built by geometric(a, r, m); the constructor is explicit so that a
// braced list of points ({1, 2, 3}) never turns into one.
struct Geometric {
    explicit constexpr Geometric(std::uint32_t a, std::uint32_t r, std::size_t m) : a(a), r(r), m(m) {}
    std::uint32_t a, r;
    std::size_t m;
};

constexpr Geometric geometric(std::uint32_t a, std::uint32_t r, std::size_t m) { return Geometric(a, r, m); }

namespace detail::geometric {

inline constexpr std::uint32_t kGenerator = 3;  // of the units mod 998244353

template <std::uint32_t P>
void check(const Geometric& points, const char* what) {
    if (points.a >= P || points.r >= P) throw std::domain_error(what);
}

// Throws unless every word is < P: v - P wraps to at least 2^32 - P for v < P and stays below it
// for v >= P, so the least v - P decides. In vectors, 4 chains: a scalar loop with an early exit
// (std::ranges::any_of) ran 2 cycles per word on Neoverse V2, 2-4% of extrapolate and idft.
template <std::uint32_t P>
void check_canonical(std::span<const std::uint32_t> a, const char* what) {
    using S = simd::Native;
    constexpr int kChains = 4;
    const typename S::V p = S::broadcast(P);
    typename S::V low[kChains];
    for (auto& v : low) v = S::broadcast(~0u);
    std::size_t i = 0;
    for (; i + kChains * S::lanes <= a.size(); i += kChains * S::lanes)
        for (int k = 0; k < kChains; ++k) low[k] = S::min(low[k], S::sub(S::load(a.data() + i + k * S::lanes), p));
    std::uint32_t lanes[kChains * S::lanes];
    for (int k = 0; k < kChains; ++k) S::store(lanes + k * S::lanes, low[k]);
    std::uint32_t least = *std::min_element(lanes, lanes + kChains * S::lanes);
    for (; i < a.size(); ++i) least = std::min(least, a[i] - P);
    if (least < 0u - P) throw std::domain_error(what);
}

// w = 3^((P - 1) / L) (inverse: w^-1) for each divisor L = 2^i 7^j 17^k of P - 1 = 2^23 7 17,
// computed at compile time: mod_pow at run time cost 2-5% of dft and idft at L = 2^10.
template <std::uint32_t P>
struct Roots {
    static_assert(P == 998244353, "fps: v1 supports P = 998244353 only");
    std::uint32_t w[2][24][2][2];  // [inverse][i][j][k]

    constexpr Roots() : w{} {
        for (int i = 0; i < 24; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k) {
                    const std::uint32_t e = (P - 1) >> i, d = e / (j ? 7 : 1) / (k ? 17 : 1);
                    w[0][i][j][k] = mod_pow<P>(kGenerator, d);
                    w[1][i][j][k] = mod_pow<P>(kGenerator, P - 1 - d);
                }
    }
};

template <std::uint32_t P>
inline constexpr Roots<P> kRoots{};

// w, or w^-1 if inverse, for L = 0 (w unused) or a divisor of P - 1; else std::domain_error.
template <std::uint32_t P>
std::uint32_t root(std::size_t L, bool inverse, const char* what) {
    if (L == 0) return 1;
    if ((P - 1) % L) throw std::domain_error(what);
    const int i = std::countr_zero(L);
    const std::size_t odd = L >> i;
    return kRoots<P>.w[inverse][i][odd % 7 == 0][odd % 17 == 0];
}

// The chirp's middle product of n + m - 1 by n words (n coefficients, m points; Horner for
// n <= 1), checked against the NTT's limit before the result is allocated. For a, r != 0.
inline void check_chirp(std::size_t n, std::size_t m, const char* what) {
    if (n > 1 && m > 0 && n + m - 1 > std::size_t(1) << ntt::kMaxLog) throw std::length_error(what);
}

// dft and idft of length L run ntt::dft and ntt::idft: L = 2^6 .. 2^23.
inline bool transform_length(std::size_t L) {
    return std::has_single_bit(L) && L >= (std::size_t(1) << ntt::kMinLog) && L <= (std::size_t(1) << ntt::kValuesMaxLog);
}

// The path of a dft or idft of length L and its root (w; w^-1 for idft), the errors checked.
struct Plan {
    std::uint32_t w;
    bool transform;
};

template <std::uint32_t P>
Plan dft_plan(std::size_t n, std::size_t L) {
    const Plan plan{root<P>(L, false, "fps::dft: L does not divide P - 1"), transform_length(L)};
    if (!plan.transform) check_chirp(n, L, "fps::dft: f.size() + L - 1 > 2^25");
    return plan;
}

template <std::uint32_t P>
Plan idft_plan(std::size_t L) {
    const Plan plan{root<P>(L, true, "fps::idft: values.size() does not divide P - 1"), transform_length(L)};
    if (!plan.transform) check_chirp(L, L, "fps::idft: L > 2^24");
    return plan;
}

// a, or a copy of it in this thread's scratch (kept and reused) if it overlaps out.
inline std::span<const std::uint32_t> apart(std::span<const std::uint32_t> a, std::span<const std::uint32_t> out) {
    if (a.empty() || out.empty() || a.data() >= out.data() + out.size() || out.data() >= a.data() + a.size()) return a;
    thread_local std::vector<std::uint32_t> scratch;
    if (scratch.size() < a.size()) scratch.resize(a.size());
    std::ranges::copy(a, scratch.begin());
    return {scratch.data(), a.size()};
}

// Into out of L words. ntt::dft and ntt::idft read their input before writing out; the chirp does not.
template <std::uint32_t P>
void dft(std::span<const std::uint32_t> f, std::span<std::uint32_t> out, Plan plan) {
    if (plan.transform) ntt::dft(f, out);
    else if (!out.empty()) chirp::evaluate_geometric(apart(f, out), 1, plan.w, out);
}

template <std::uint32_t P>
void idft(std::span<const std::uint32_t> values, std::span<std::uint32_t> out, Plan plan) {
    const std::size_t L = values.size();
    if (plan.transform) {
        ntt::idft(values, out);
        return;
    }
    if (L == 0) return;
    // f_j = (1 / L) sum_i values[i] w^(-i j): the values as coefficients, at the points w^-j.
    chirp::evaluate_geometric(apart(values, out), 1, plan.w, out);
    // 1 / L = P - (P - 1) / L, as L (P - 1) / L = P - 1 = -1.
    chirp::multiply_chirp(P - std::uint32_t((P - 1) / L), 1, 1, out);
}

}  // namespace detail::geometric

template <std::uint32_t P>
std::vector<std::uint32_t> evaluate(const Polynomial<P>& f, const Geometric& points) {
    detail::geometric::check<P>(points, "fps::evaluate: a or r >= P");
    std::vector<std::uint32_t> values(points.m);
    chirp::evaluate_geometric(f.span(), points.a, points.r, values);
    return values;
}

template <std::uint32_t P>
Polynomial<P> interpolate(const Geometric& points, std::span<const std::uint32_t> values) {
    const std::size_t m = points.m;
    if (values.size() != m) throw std::domain_error("fps::interpolate: points and values differ in size");
    if (m > std::size_t(1) << ntt::kMaxLog)
        throw std::length_error("fps::interpolate: more than 2^25 geometric points");
    detail::geometric::check<P>(points, "fps::interpolate: a or r >= P");
    detail::geometric::check_canonical<P>(values, "fps::interpolate: value >= P");
    Polynomial<P> f = detail::unwritten_polynomial<P>(m);
    try {
        chirp::interpolate_geometric(points.a, points.r, values, f.span());
    } catch (const std::invalid_argument&) {  // the sizes match, so the points repeat
        throw std::domain_error("fps::interpolate: points not distinct");
    }
    return f;
}

template <std::uint32_t P>
std::vector<std::uint32_t> extrapolate(std::span<const std::uint32_t> values, std::uint32_t c, std::size_t m) {
    if (values.size() > P) throw std::domain_error("fps::extrapolate: more than P values");
    if (c >= P) throw std::domain_error("fps::extrapolate: c >= P");
    detail::geometric::check_canonical<P>(values, "fps::extrapolate: value >= P");
    std::vector<std::uint32_t> out(m);
    chirp::extrapolate(values, c, out);
    return out;
}

template <std::uint32_t P>
std::vector<std::uint32_t> dft(const Polynomial<P>& f, std::size_t L) {
    const auto plan = detail::geometric::dft_plan<P>(f.size(), L);
    std::vector<std::uint32_t> values(L);
    detail::geometric::dft<P>(f.span(), values, plan);
    return values;
}

template <std::uint32_t P>
Polynomial<P> idft(std::span<const std::uint32_t> values) {
    const auto plan = detail::geometric::idft_plan<P>(values.size());
    detail::geometric::check_canonical<P>(values, "fps::idft: value >= P");
    Polynomial<P> f = detail::unwritten_polynomial<P>(values.size());
    detail::geometric::idft<P>(values, f.span(), plan);
    return f;
}

namespace raw {

template <std::uint32_t P>
void dft(std::span<const std::uint32_t> f, std::span<std::uint32_t> out) {
    detail::geometric::dft<P>(f, out, detail::geometric::dft_plan<P>(f.size(), out.size()));
}

template <std::uint32_t P>
void idft(std::span<const std::uint32_t> values, std::span<std::uint32_t> out) {
    if (out.size() != values.size()) throw std::domain_error("fps::raw::idft: out.size() != values.size()");
    detail::geometric::idft<P>(values, out, detail::geometric::idft_plan<P>(values.size()));
}

}  // namespace raw

}  // namespace fps
