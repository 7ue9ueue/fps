// Arithmetic modulo P = 998244353 in Montgomery form (x stored as x 2^32 mod P), and the dense
// kernels of the factorization on short coefficient arrays, as templates on a backend/simd struct S:
//
//   combine<S>(out, len, init, coef, count, base, stride)
//       out[t] = init[t] + sum_{s < count} coef[s] (base + s stride)[t]; init may be null (zero)
//   combine_scaled<S>(out, len, init, scale, coef, count, base, stride)   the same with scale init[t]
//   lincomb<S>(out, x, a, y, b, len)          out[t] = a x[t] + b y[t]
//   multiply<S>(a, na, b, nb, out, scratch)   out = a b, na + nb - 1 words
//   transpose_rows<S>(dst, ld, rows, j0, src, stride)
//       dst[i ld + j0 + q] = src[q stride + i], q < 8, i < rows
//
// Words are Montgomery residues in [0, P) in and out. Kernels work on whole vectors of W = S::lanes
// words: t runs to len rounded up to W, so arrays are readable (and out writable) that far, and
// the extra words of the inputs must be zero where the extra outputs are to be zero. out may be
// init or a column.
//
// Sums: products of two residues (< (P - 1)^2) add up in 64-bit lanes; fold(s) = s - (s >> 32) kQ
// = (s >> 32) R1 + (s mod 2^32), R1 = 2^32 mod P, kQ = 2^32 - R1 (a multiple of P), keeps a sum
// below kFolded = 2^32 (R1 + 1); kChunk = 17 products fit on top of that, or on zero, below 2^64.
// The last step folds, then Montgomery: (s + m P) / 2^32 < 2P for s < 2^32 P, and one min.
//   x86 (mul_even): two 64-bit-lane sums per vector, even and odd words (EvenOdd).
//   Neon: words 0-1 and 2-3, umlal / umlal2 (Halves).
//   Scalar: one 64-bit sum per word (Single).
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "backend/modarith/modarith.hpp"
#include "backend/simd/simd.hpp"

namespace fps::factorization {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u32 kP = 998244353;

namespace detail {

using u128 = unsigned __int128;

// ---- Scalar arithmetic (Montgomery form) --------------------------------------------------------

inline constexpr u32 kPinv = u32(modarith::inverse_mod_power_of_two(kP, 32));  // P^-1 mod 2^32
inline constexpr u32 kNegPinv = 0u - kPinv;
inline constexpr u32 kOne = u32((u64(1) << 32) % kP);  // 1 in Montgomery form; R1
inline constexpr u32 kR2 = u32(u64(kOne) * kOne % kP);  // 2^64 mod P
inline constexpr u32 kQ = u32((u64(1) << 32) - kOne);   // 2^32 - R1, a multiple of P

inline constexpr u128 kProduct = u128(kP - 1) * (kP - 1);
inline constexpr u128 kFolded = (u128(1) << 32) * (kOne + 1);
inline constexpr int kChunk = int(((u128(1) << 64) - kFolded) / kProduct);  // 17
static_assert(kChunk >= 2 && kFolded < (u128(kP) << 32));

// x 2^-32 mod P for x < 2^32 P, canonical.
constexpr u32 reduce(u64 x) {
    const u32 m = u32(x) * kNegPinv;
    const u32 t = u32((x + u64(m) * kP) >> 32);
    return t >= kP ? t - kP : t;
}
constexpr u64 fold(u64 x) { return x - (x >> 32) * kQ; }
constexpr u32 mul(u32 a, u32 b) { return reduce(u64(a) * b); }
constexpr u32 add(u32 a, u32 b) { return a + b >= kP ? a + b - kP : a + b; }
constexpr u32 sub(u32 a, u32 b) { return a >= b ? a - b : a + kP - b; }
constexpr u32 neg(u32 a) { return a ? kP - a : 0; }
constexpr u32 to_montgomery(u32 a) { return mul(a, kR2); }  // a < P
constexpr u32 from_montgomery(u32 a) { return reduce(a); }
constexpr u32 power(u32 a, u64 e) {
    u32 r = kOne;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
constexpr u32 inverse(u32 a) { return power(a, kP - 2); }

constexpr int round_up(int n, int w) { return (n + w - 1) / w * w; }

// ---- Sums per family of simd struct -------------------------------------------------------------

template <class S>
concept HasMulEven = requires(typename S::V a) { S::mul_even(a, a); };
template <class S>
concept HasMulWide = requires(typename S::V a) { S::mul_wide_lo(a, a); };

// zero(), mac(s, c, x): s += c x lane-wise for c a broadcast word; op(x): x as mac's operand, to
// share among several sums (mac(s, c, op)); fold(s); reduce(s): the folded sums times 2^-32,
// canonical.
template <class S>
struct EvenOdd {  // x86: 64-bit lanes by mul_even
    using V = typename S::V;
    struct Acc {
        V even, odd;
    };
    struct Op {
        V even, odd;  // the odd words moved to the even ones
    };
    [[gnu::always_inline]] static Acc zero() { return {S::broadcast(0), S::broadcast(0)}; }
    [[gnu::always_inline]] static Op op(V x) { return {x, S::template shr64<32>(x)}; }
    [[gnu::always_inline]] static void mac(Acc& s, V c, const Op& x) {
        s.even = S::add64(s.even, S::mul_even(c, x.even));
        s.odd = S::add64(s.odd, S::mul_even(c, x.odd));
    }
    [[gnu::always_inline]] static void mac(Acc& s, V c, V x) { mac(s, c, op(x)); }
    // Sums x 2^32 (x < P), so that reduce() returns x plus the products: a shift and a mask.
    [[gnu::always_inline]] static Acc shifted(V x) {
        return {S::template shl64<32>(x), S::template blend<0x5555>(x, S::broadcast(0))};
    }
    [[gnu::always_inline]] static V fold(V s) {
        return S::sub64(s, S::mul_even(S::template shr64<32>(s), S::broadcast(kQ)));
    }
    [[gnu::always_inline]] static void fold(Acc& s) { s.even = fold(s.even), s.odd = fold(s.odd); }
    // Montgomery's step without the last reduction: below P + s / 2^32.
    [[gnu::always_inline]] static V reduce_lazy(const Acc& s) {
        const V ninv = S::broadcast(kNegPinv), p = S::broadcast(kP);
        const V even = S::add64(s.even, S::mul_even(S::mul_even(s.even, ninv), p));
        const V odd = S::add64(s.odd, S::mul_even(S::mul_even(s.odd, ninv), p));
        return S::join_high(even, odd);
    }
    [[gnu::always_inline]] static V reduce(const Acc& s) { return modarith::Ops<S, kP>::template reduce<kP>(reduce_lazy(s)); }
};

template <class S>
struct Halves {  // Neon: words 0-1 and 2-3 in 64-bit lanes
    using V = typename S::V;
    using W = typename S::W;
    struct Acc {
        W lo, hi;
    };
    using Op = V;
    [[gnu::always_inline]] static Acc zero() { return {W{}, W{}}; }
    [[gnu::always_inline]] static Op op(V x) { return x; }
    [[gnu::always_inline]] static Acc shifted(V x) {  // x 2^32 mod P: one umull each
        const V one = S::broadcast(kOne);
        return {S::mul_wide_lo(x, one), S::mul_wide_hi(x, one)};
    }
    [[gnu::always_inline]] static void mac(Acc& s, V c, V x) {
        s.lo = S::mla_wide_lo(s.lo, x, c);
        s.hi = S::mla_wide_hi(s.hi, x, c);
    }
    [[gnu::always_inline]] static void fold(Acc& s) {
        const V high = S::hi_halves(s.lo, s.hi), q = S::broadcast(kQ);
        s.lo = S::mls_wide_lo(s.lo, high, q);
        s.hi = S::mls_wide_hi(s.hi, high, q);
    }
    [[gnu::always_inline]] static V reduce_lazy(const Acc& s) {
        const V m = S::mul_lo(S::lo_halves(s.lo, s.hi), S::broadcast(kNegPinv)), p = S::broadcast(kP);
        return S::hi_halves(S::mla_wide_lo(s.lo, m, p), S::mla_wide_hi(s.hi, m, p));
    }
    [[gnu::always_inline]] static V reduce(const Acc& s) { return modarith::Ops<S, kP>::template reduce<kP>(reduce_lazy(s)); }
};

template <class S>
struct Single {  // one word per vector
    using V = u32;
    using Acc = u64;
    using Op = u32;
    [[gnu::always_inline]] static Acc zero() { return 0; }
    [[gnu::always_inline]] static Op op(V x) { return x; }
    [[gnu::always_inline]] static Acc shifted(V x) { return u64(x) << 32; }
    [[gnu::always_inline]] static void mac(Acc& s, V c, V x) { s += u64(c) * x; }
    [[gnu::always_inline]] static void fold(Acc& s) { s = detail::fold(s); }
    [[gnu::always_inline]] static V reduce_lazy(Acc s) { return u32((s + u64(u32(s) * kNegPinv) * kP) >> 32); }
    [[gnu::always_inline]] static V reduce(Acc s) { return detail::reduce(s); }
};

template <class S>
using Sums = std::conditional_t<HasMulEven<S>, EvenOdd<S>, std::conditional_t<HasMulWide<S>, Halves<S>, Single<S>>>;

template <class S>
inline constexpr bool kSingle = !HasMulEven<S> && !HasMulWide<S>;

// Vectors per block of combine: the sums of a block stay in registers (2 per vector).
template <class S>
inline constexpr int kBlock = S::registers >= 32 ? 8 : 4;

alignas(64) inline constexpr u32 kZeros[16 * 8] = {};  // init of combine when null: one block

// out[t] = scale init[t] + sum_s coef[s] column_s[t], t < V W, column_s = base + s stride.
template <class S, int V>
[[gnu::always_inline]] inline void combine_block(u32* out, const u32* init, u32 scale, const u32* coef, int count,
                                                 const u32* base, std::ptrdiff_t stride) {
    using A = Sums<S>;
    constexpr int W = S::lanes;
    typename A::Acc acc[V];
    const typename S::V vscale = S::broadcast(scale);
#pragma GCC unroll 8
    for (int v = 0; v < V; ++v) {
        acc[v] = A::zero();
        A::mac(acc[v], vscale, S::load(init + W * v));
    }
    // One product (init) and kChunk - 1 columns, then folds every kChunk columns.
    for (int s0 = 0, s1 = std::min(count, kChunk - 1);; s0 = s1, s1 = std::min(count, s1 + kChunk)) {
        const auto step = [&](int s) {
            const typename S::V c = S::broadcast(coef[s]);
            const u32* column = base + s * stride;
#pragma GCC unroll 8
            for (int v = 0; v < V; ++v) A::mac(acc[v], c, S::load(column + W * v));
        };
        if constexpr (kSingle<S>) {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC novector  // GCC 15.2 vectorizes Single's sums wrongly at -O2 -march=x86-64-v3 (#183)
#endif
            for (int s = s0; s < s1; ++s) step(s);
        } else {
            for (int s = s0; s < s1; ++s) step(s);
        }
#pragma GCC unroll 8
        for (int v = 0; v < V; ++v) A::fold(acc[v]);
        if (s1 == count) break;
    }
#pragma GCC unroll 8
    for (int v = 0; v < V; ++v) S::store(out + W * v, A::reduce(acc[v]));
}

template <class S, int V = kBlock<S> - 1>
inline void combine_rest(int vectors, u32* out, const u32* init, u32 scale, const u32* coef, int count,
                         const u32* base, std::ptrdiff_t stride) {
    if constexpr (V > 0) {
        if (vectors == V) return combine_block<S, V>(out, init, scale, coef, count, base, stride);
        combine_rest<S, V - 1>(vectors, out, init, scale, coef, count, base, stride);
    }
}

// Rows and vectors per block of combine_rows: R rows x V vectors of sums in registers. 8 rows x m
// x m products per ns (notes/123-r1.md): Zen 3 R4V1 18.1 at m = 100 (R2V2 17.5, R8V1 16.1, R4V2
// 15.9); Emerald Rapids (zmm) R8V1 23.2 (R4V2 23.0, R4V1 21.9); Zen 5 R8V1 44.4 (R4V2 44.0, R4V1
// 43.7); Neoverse V2 R4V1 10.3 (R2V2 10.2, R4V2 9.1, R8V1 8.7).
template <class S>
inline constexpr int kRows = S::registers >= 32 && HasMulEven<S> ? 8 : 4;
template <class S>
inline constexpr int kRowVectors = 1;

// Row r < R: out_r[t] = scale init_r[t] + sum_s coef_r[s] column_s[t], t < V W, with out_r =
// out + r ostride, init_r = init + r istride, coef_r[s] = coef[r cstride + s sstride], column_s =
// base + s stride. Each column vector loaded serves the R rows.
template <class S, int R, int V>
[[gnu::always_inline]] inline void rows_block(u32* out, std::ptrdiff_t ostride, const u32* init, std::ptrdiff_t istride,
                                              u32 scale, const u32* coef, std::ptrdiff_t cstride, std::ptrdiff_t sstride,
                                              int count, const u32* base, std::ptrdiff_t stride) {
    using A = Sums<S>;
    constexpr int W = S::lanes;
    typename A::Acc acc[R][V];
    const typename S::V vscale = S::broadcast(scale);
#pragma GCC unroll 8
    for (int r = 0; r < R; ++r)
#pragma GCC unroll 4
        for (int v = 0; v < V; ++v) {
            acc[r][v] = A::zero();
            A::mac(acc[r][v], vscale, S::load(init + r * istride + W * v));
        }
    for (int s0 = 0, s1 = std::min(count, kChunk - 1);; s0 = s1, s1 = std::min(count, s1 + kChunk)) {
        const auto step = [&](int s) {
            typename A::Op x[V];
#pragma GCC unroll 4
            for (int v = 0; v < V; ++v) x[v] = A::op(S::load(base + s * stride + W * v));
#pragma GCC unroll 8
            for (int r = 0; r < R; ++r) {
                const typename S::V c = S::broadcast(coef[r * cstride + s * sstride]);
#pragma GCC unroll 4
                for (int v = 0; v < V; ++v) A::mac(acc[r][v], c, x[v]);
            }
        };
        if constexpr (kSingle<S>) {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC novector  // #183
#endif
            for (int s = s0; s < s1; ++s) step(s);
        } else {
            for (int s = s0; s < s1; ++s) step(s);
        }
#pragma GCC unroll 8
        for (int r = 0; r < R; ++r)
#pragma GCC unroll 4
            for (int v = 0; v < V; ++v) A::fold(acc[r][v]);
        if (s1 == count) break;
    }
#pragma GCC unroll 8
    for (int r = 0; r < R; ++r)
#pragma GCC unroll 4
        for (int v = 0; v < V; ++v) S::store(out + r * ostride + W * v, A::reduce(acc[r][v]));
}

// R rows over len words.
template <class S, int R>
inline void rows_group(u32* out, std::ptrdiff_t ostride, int len, const u32* init, std::ptrdiff_t istride, u32 scale,
                       const u32* coef, std::ptrdiff_t cstride, std::ptrdiff_t sstride, int count, const u32* base,
                       std::ptrdiff_t stride) {
    constexpr int W = S::lanes, V = kRowVectors<S>;
    const int vectors = (len + W - 1) / W, step = init == kZeros ? 0 : W;  // kZeros: one block's words
    int v = 0;
    for (; v + V <= vectors; v += V)
        rows_block<S, R, V>(out + W * v, ostride, init + step * v, istride, scale, coef, cstride, sstride, count,
                            base + W * v, stride);
    if constexpr (V > 1)
        for (; v < vectors; ++v)
            rows_block<S, R, 1>(out + W * v, ostride, init + step * v, istride, scale, coef, cstride, sstride, count,
                                base + W * v, stride);
}

template <class S, int R = kRows<S> - 1>
inline void rows_rest(int rows, u32* out, std::ptrdiff_t ostride, int len, const u32* init, std::ptrdiff_t istride,
                      u32 scale, const u32* coef, std::ptrdiff_t cstride, std::ptrdiff_t sstride, int count,
                      const u32* base, std::ptrdiff_t stride) {
    if constexpr (R > 0) {
        if (rows == R)
            return rows_group<S, R>(out, ostride, len, init, istride, scale, coef, cstride, sstride, count, base, stride);
        rows_rest<S, R - 1>(rows, out, ostride, len, init, istride, scale, coef, cstride, sstride, count, base, stride);
    }
}

}  // namespace detail

// Row r < rows: out_r[t] = scale init_r[t] + sum_{s < count} coef_r[s] (base + s stride)[t] for t
// < len rounded up to W, with out_r = out + r ostride, init_r = init + r istride (init may be null:
// zero), coef_r[s] = coef[r cstride + s sstride]. As combine_scaled for each row, but a column's
// words, loaded once, serve kRows rows (their sums held in registers): a matrix product, rows times
// columns. out_r may be init_r; not another row's init or coefficients.
template <class S>
inline void combine_rows(u32* out, std::ptrdiff_t ostride, int rows, int len, const u32* init, std::ptrdiff_t istride,
                         u32 scale, const u32* coef, std::ptrdiff_t cstride, std::ptrdiff_t sstride, int count,
                         const u32* base, std::ptrdiff_t stride) {
    using namespace detail;
    constexpr int R = kRows<S>;
    if (!init) init = kZeros, istride = 0;
    int r = 0;
    for (; r + R <= rows; r += R)
        rows_group<S, R>(out + r * ostride, ostride, len, init + r * istride, istride, scale, coef + r * cstride, cstride,
                         sstride, count, base, stride);
    if (r < rows)
        rows_rest<S>(rows - r, out + r * ostride, ostride, len, init + r * istride, istride, scale, coef + r * cstride,
                     cstride, sstride, count, base, stride);
}

// out[t] = scale init[t] + sum_{s < count} coef[s] (base + s stride)[t] for t < len rounded up to
// W; init may be null (zero). Columns and init are read over the same rounded range.
template <class S>
inline void combine_scaled(u32* out, int len, const u32* init, u32 scale, const u32* coef, int count, const u32* base,
                           std::ptrdiff_t stride) {
    using namespace detail;
    constexpr int W = S::lanes, B = kBlock<S>;
    const int vectors = (len + W - 1) / W;
    int v = 0;
    for (; v + B <= vectors; v += B)
        combine_block<S, B>(out + W * v, init ? init + W * v : kZeros, scale, coef, count, base + W * v, stride);
    if (v < vectors)
        combine_rest<S>(vectors - v, out + W * v, init ? init + W * v : kZeros, scale, coef, count, base + W * v,
                        stride);
}

template <class S>
inline void combine(u32* out, int len, const u32* init, const u32* coef, int count, const u32* base,
                    std::ptrdiff_t stride) {
    combine_scaled<S>(out, len, init, detail::kOne, coef, count, base, stride);
}

// out[t] = a x[t] + b y[t] for t < len rounded up to W; out may alias x or y.
template <class S>
inline void lincomb(u32* out, const u32* x, u32 a, const u32* y, u32 b, int len) {
    using A = detail::Sums<S>;
    constexpr int W = S::lanes;
    const typename S::V va = S::broadcast(a), vb = S::broadcast(b);
    for (int t = 0; t < len; t += W) {
        typename A::Acc s = A::zero();
        A::mac(s, va, S::load(x + t));
        A::mac(s, vb, S::load(y + t));  // 2 products < 2^32 P: no fold
        S::store(out + t, A::reduce(s));
    }
}

// out[t] = x[t] + c y[t] for t < len rounded up to W; out may alias x or y. Sums x 2^32 + c y <
// P 2^32 + P^2, so Montgomery's step gives less than 2.24 P: two min-reductions.
template <class S>
inline void axpy(u32* out, const u32* x, u32 c, const u32* y, int len) {
    using A = detail::Sums<S>;
    using M = modarith::Ops<S, kP>;
    constexpr int W = S::lanes;
    const typename S::V vc = S::broadcast(c);
    for (int t = 0; t < len; t += W) {
        typename A::Acc s = A::shifted(S::load(x + t));
        A::mac(s, vc, S::load(y + t));
        S::store(out + t, M::template reduce<kP>(M::template reduce<2 * kP>(A::reduce_lazy(s))));
    }
}

namespace detail {

// [p[W-1], c[0], .., c[W-2]]: c one lane up, the lane in from p (one or two shuffles).
inline u32 shift_in(u32 p, u32) { return p; }
#if defined(__SSE4_1__)
inline __m128i shift_in(__m128i p, __m128i c) { return _mm_alignr_epi8(c, p, 12); }
#endif
#if defined(__AVX2__)
inline __m256i shift_in(__m256i p, __m256i c) {
    return _mm256_alignr_epi8(c, _mm256_permute2x128_si256(p, c, 0x21), 12);
}
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__)
inline __m512i shift_in(__m512i p, __m512i c) { return _mm512_alignr_epi32(c, p, 15); }
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
inline uint32x4_t shift_in(uint32x4_t p, uint32x4_t c) { return vextq_u32(p, c, 3); }
#endif

}  // namespace detail

namespace detail {

// y = x a mod g given c = -a[m-1] in every lane; returns y's vector at word `last` (the one with
// word m - 1), computed first, so that the next step of a chain can start before this one ends.
template <class S>
[[gnu::always_inline]] inline typename S::V times_x_step(u32* y, const u32* a, const u32* g, int last,
                                                         typename S::V c) {
    using A = Sums<S>;
    using M = modarith::Ops<S, kP>;
    using V = typename S::V;
    constexpr int W = S::lanes;
    const auto word = [&](V previous, V current, int t) {
        typename A::Acc sum = A::shifted(shift_in(previous, current));
        A::mac(sum, c, S::load(g + t));
        const V r = M::template reduce<kP>(M::template reduce<2 * kP>(A::reduce_lazy(sum)));
        S::store(y + t, r);
        return r;
    };
    const V top = word(last ? S::load(a + last - W) : S::broadcast(0), S::load(a + last), last);
    V previous = S::broadcast(0);
    for (int t = 0; t < last; t += W) {
        const V current = S::load(a + t);
        word(previous, current, t);
        previous = current;
    }
    return top;
}

}  // namespace detail

// y = x a mod g for g monic of degree m, a of m words (zero from m to m rounded up to W), y not
// overlapping a: y[t] = a[t-1] - a[m-1] g[t], written to m rounded up to W (zeros past m). The
// shift is done in registers from aligned loads of a.
template <class S>
inline void times_x(u32* y, const u32* a, const u32* g, int m) {
    detail::times_x_step<S>(y, a, g, (m - 1) / S::lanes * S::lanes, S::broadcast(detail::neg(a[m - 1])));
}

// Column s = x column_(s-1) mod g at first + s stride for s = 1 .. count - 1, from column 0
// (times_x down a chain). Each column's loads of the one before match its stores (store
// forwarding), and its top word comes from the register that held it, broadcast: a word load from
// inside a vector store was not forwarded on Zen 3 (40 ns per column at m = 100 against about 10).
template <class S>
inline void times_x_chain(u32* first, int count, std::ptrdiff_t stride, const u32* g, int m) {
    using M = modarith::Ops<S, kP>;
    using V = typename S::V;
    constexpr int W = S::lanes;
    const int last = (m - 1) / W * W;
    std::array<int, W> lane;
    lane.fill((m - 1) % W);
    const auto index = S::index(lane);
    V top = S::load(first + last);
    for (int s = 1; s < count; ++s) {
        const V c = M::sub(S::broadcast(0), S::permute(top, index));  // -column_(s-1)[m-1]
        top = detail::times_x_step<S>(first + s * stride, first + (s - 1) * stride, g, last, c);
    }
}

// Words of scratch multiply() needs for a factor b of nb words.
template <class S>
constexpr int multiply_scratch(int nb) {
    return nb + 2 * detail::kBlock<S> * S::lanes;
}

// out[0, na + nb - 1) = a b, written to that length rounded up to W (zeros past the product); na,
// nb >= 1. Column s of a block from t0 is b shifted by lo + s: a zero-padded copy of b read
// backwards (stride -1).
template <class S>
inline void multiply(const u32* a, int na, const u32* b, int nb, u32* out, u32* scratch) {
    using namespace detail;
    constexpr int W = S::lanes, B = kBlock<S> * W;
    u32* padded = scratch + B;
    std::fill(scratch, padded, 0);
    std::copy(b, b + nb, padded);
    std::fill(padded + nb, padded + nb + B, 0);
    const int len = na + nb - 1;
    for (int t0 = 0; t0 < len; t0 += B) {
        const int lo = std::max(0, t0 - nb + 1), hi = std::min(na, t0 + B);
        const int vectors = std::min(B, round_up(len - t0, W)) / W;
        if (vectors == kBlock<S>) combine_block<S, kBlock<S>>(out + t0, kZeros, kOne, a + lo, hi - lo, padded + t0 - lo, -1);
        else combine_rest<S>(vectors, out + t0, kZeros, kOne, a + lo, hi - lo, padded + t0 - lo, -1);
    }
}

// dst[i ld + j0 + q] = src[q stride + i] for q < 8, i < rows. src: 8 rows of round_up(rows, 8)
// readable words.
template <class S>
inline void transpose_rows(u32* dst, std::ptrdiff_t ld, int rows, int j0, const u32* src, std::ptrdiff_t stride) {
#if defined(__AVX2__)
    if constexpr (S::lanes >= 8) {
        for (int i0 = 0; i0 < rows; i0 += 8) {
            __m256i r[8], t[8];
            for (int q = 0; q < 8; ++q) r[q] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + q * stride + i0));
            for (int q = 0; q < 8; q += 2) {
                t[q] = _mm256_unpacklo_epi32(r[q], r[q + 1]);
                t[q + 1] = _mm256_unpackhi_epi32(r[q], r[q + 1]);
            }
            for (int q = 0; q < 8; q += 4) {
                r[q] = _mm256_unpacklo_epi64(t[q], t[q + 2]);
                r[q + 1] = _mm256_unpackhi_epi64(t[q], t[q + 2]);
                r[q + 2] = _mm256_unpacklo_epi64(t[q + 1], t[q + 3]);
                r[q + 3] = _mm256_unpackhi_epi64(t[q + 1], t[q + 3]);
            }
            for (int q = 0; q < 4; ++q) {
                t[q] = _mm256_permute2x128_si256(r[q], r[q + 4], 0x20);
                t[q + 4] = _mm256_permute2x128_si256(r[q], r[q + 4], 0x31);
            }
            for (int q = 0; q < 8 && i0 + q < rows; ++q)
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + (i0 + q) * ld + j0), t[q]);
        }
        return;
    }
#endif
    for (int i = 0; i < rows; ++i)
        for (int q = 0; q < 8; ++q) dst[i * ld + j0 + q] = src[q * stride + i];
}

}  // namespace fps::factorization
