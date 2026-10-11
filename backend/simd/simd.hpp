// Vector wrappers, one struct per ISA. Kernels are templates on the struct (S::add(a, b), ...), so
// one source runs on every target. Every function is static, inline and maps to the instruction
// named next to it; "composed" ones are short fixed sequences, with their instruction count.
//
//   Struct     Vector                       Lanes  Registers  Available when
//   Scalar     uint32_t                     1      16         always
//   Sse        __m128i                      4      16         __SSE4_1__
//   Avx2       __m256i                      8      16         __AVX2__
//   Avx512Vl   __m256i, EVEX                8      32         __AVX512VL__ (and __AVX2__)
//   Avx512     __m512i                      16     32         __AVX512F__ && __AVX512BW__
//   Neon       uint32x4_t                   4      32         __ARM_NEON && __aarch64__
//
// Lanes are unsigned 32-bit words; arithmetic wraps mod 2^32. "64-bit lanes" view lanes 2i, 2i + 1
// as one word, 2i the low half.
//
// Every struct:
//   V, lanes, registers, name
//   load(p), store(p, a)       lanes words, any alignment
//   broadcast(x)               every lane x
//   add, sub, min              lane-wise; min unsigned
//   mul_lo(a, b)               low 32 bits of a b
//   shr<k>(a), shl<k>(a)       logical shifts, 1 <= k <= 31
//   mul_sub(c, a, b)           c - a b (low 32 bits); Neon: one mls, x86: mul_lo + sub
//   mulhi(a, b)                composed: high 32 bits of a b. x86: 6 instructions (2 mul_even;
//                              5 with AVX-512 off Zen),
//                              Neon: 3 (umull, umull2, uzp2); 1 with SVE2 (umulh)
//   mul_shr<k>(a, b)           composed: (a b) >> k, 1 <= k <= 32, valid if it is < 2^32.
//                              x86: 7 instructions (6 for k = 32), Neon: 4 (3 for k = 32)
//   blend<mask>(a, b)          lane i from b if bit i of mask is set, else from a
//   Index, index(lanes[])      permute(a, idx): lane i = a[lanes[i]]; index() builds idx, off the
//                              hot path (Sse, Neon: byte shuffles, so idx is a byte table)
// x86 only (Sse, Avx2, Avx512Vl, Avx512):
//   mul_even(a, b)             64-bit lanes: product of the low halves (vpmuludq)
//   broadcast64(x), add64, sub64
//                              64-bit lanes
//   shr64<k>, shl64<k>         64-bit lanes, logical
//   join_high(e, o)            composed: word 2i = high word of 64-bit lane i of e, word 2i + 1 =
//                              high word of lane i of o. Sse, Avx2: shr64 + blend; AVX-512: vpermt2d
// Avx512Vl, Avx512:
//   permute2(a, idx, b)        lane i = idx[i] < lanes ? a[idx[i]] : b[idx[i] - lanes] (vpermt2d)
// Avx512Vl, Avx512 with __AVX512IFMA__:
//   madd52lo(c, a, b), madd52hi(c, a, b)
//                              64-bit lanes: c + low / high 52 bits of the 104-bit product of
//                              the low 52 bits of a and b (vpmadd52luq, vpmadd52huq)
// Neon only:
//   mul_wide_lo(a, b), mul_wide_hi(a, b)
//                              uint64x2_t products of lanes 0, 1 / 2, 3 (umull, umull2)
//   mla_wide_lo(c, a, b), mla_wide_hi(c, a, b)
//                              c + those products (umlal, umlal2)
//   lo_halves(x, y), hi_halves(x, y)
//                              low / high 32 bits of the 64-bit lanes of x then y (uzp1, uzp2)
//   qdmulh(a, b)               signed: high 32 bits of 2 a b, saturated (sqdmulh)
//   hsub(a, b)                 signed: (a - b) >> 1 without overflow (shsub)
#pragma once

#include <array>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__ARM_NEON) && defined(__aarch64__) && defined(__ARM_FEATURE_SVE2)
#include <arm_neon_sve_bridge.h>
#include <arm_sve.h>
#endif

namespace fps::simd {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

struct Scalar {
    using V = u32;
    static constexpr int lanes = 1;
    static constexpr int registers = 16;  // general registers: 16 on x86-64, 31 on AArch64
    static constexpr const char* name = "scalar";

    static V load(const u32* p) { return *p; }
    static void store(u32* p, V a) { *p = a; }
    static V broadcast(u32 x) { return x; }
    static V add(V a, V b) { return a + b; }
    static V sub(V a, V b) { return a - b; }
    static V min(V a, V b) { return a < b ? a : b; }
    static V mul_lo(V a, V b) { return a * b; }
    template <int k>
    static V shr(V a) { return a >> k; }
    template <int k>
    static V shl(V a) { return a << k; }
    static V mul_sub(V c, V a, V b) { return c - a * b; }
    static V mulhi(V a, V b) { return V(u64(a) * b >> 32); }
    template <int k>
    static V mul_shr(V a, V b) {
        static_assert(1 <= k && k <= 32);
        return V(u64(a) * b >> k);
    }
    template <unsigned mask>
    static V blend(V a, V b) { return mask & 1 ? b : a; }

    using Index = int;
    static Index index(const std::array<int, lanes>&) { return 0; }
    static V permute(V a, Index) { return a; }
};

#if defined(__SSE4_1__)
struct Sse {
    using V = __m128i;
    static constexpr int lanes = 4;
    static constexpr int registers = 16;
    static constexpr const char* name = "sse";

    static V load(const u32* p) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)); }
    static void store(u32* p, V a) { _mm_storeu_si128(reinterpret_cast<__m128i*>(p), a); }
    static V broadcast(u32 x) { return _mm_set1_epi32(int(x)); }
    static V broadcast64(u64 x) { return _mm_set1_epi64x(std::int64_t(x)); }
    static V add(V a, V b) { return _mm_add_epi32(a, b); }
    static V sub(V a, V b) { return _mm_sub_epi32(a, b); }
    static V min(V a, V b) { return _mm_min_epu32(a, b); }
    static V mul_lo(V a, V b) { return _mm_mullo_epi32(a, b); }
    template <int k>
    static V shr(V a) { return _mm_srli_epi32(a, k); }
    template <int k>
    static V shl(V a) { return _mm_slli_epi32(a, k); }
    static V mul_sub(V c, V a, V b) { return sub(c, mul_lo(a, b)); }
    static V mul_even(V a, V b) { return _mm_mul_epu32(a, b); }
    static V add64(V a, V b) { return _mm_add_epi64(a, b); }
    static V sub64(V a, V b) { return _mm_sub_epi64(a, b); }
    template <int k>
    static V shr64(V a) { return _mm_srli_epi64(a, k); }
    template <int k>
    static V shl64(V a) { return _mm_slli_epi64(a, k); }
    template <unsigned mask>
    static V blend(V a, V b) {  // pblendw: one mask bit per 16-bit word
        constexpr int words = (mask & 1 ? 0x03 : 0) | (mask & 2 ? 0x0c : 0) | (mask & 4 ? 0x30 : 0) |
                              (mask & 8 ? 0xc0 : 0);
        return _mm_blend_epi16(a, b, words);
    }
    static V join_high(V e, V o) { return blend<0xa>(shr64<32>(e), o); }
    static V mulhi(V a, V b);
    template <int k>
    static V mul_shr(V a, V b);

    using Index = __m128i;  // pshufb byte indices
    static Index index(const std::array<int, lanes>& from) {
        alignas(16) std::uint8_t bytes[16];
        for (int i = 0; i < 16; ++i) bytes[i] = std::uint8_t(4 * from[i / 4] + i % 4);
        return _mm_load_si128(reinterpret_cast<const __m128i*>(bytes));
    }
    static V permute(V a, Index idx) { return _mm_shuffle_epi8(a, idx); }
};
#endif

#if defined(__AVX2__)
struct Avx2 {
    using V = __m256i;
    static constexpr int lanes = 8;
    static constexpr int registers = 16;
    static constexpr const char* name = "avx2";

    static V load(const u32* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }
    static void store(u32* p, V a) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), a); }
    static V broadcast(u32 x) { return _mm256_set1_epi32(int(x)); }
    static V broadcast64(u64 x) { return _mm256_set1_epi64x(std::int64_t(x)); }
    static V add(V a, V b) { return _mm256_add_epi32(a, b); }
    static V sub(V a, V b) { return _mm256_sub_epi32(a, b); }
    static V min(V a, V b) { return _mm256_min_epu32(a, b); }
    static V mul_lo(V a, V b) { return _mm256_mullo_epi32(a, b); }
    template <int k>
    static V shr(V a) { return _mm256_srli_epi32(a, k); }
    template <int k>
    static V shl(V a) { return _mm256_slli_epi32(a, k); }
    static V mul_sub(V c, V a, V b) { return sub(c, mul_lo(a, b)); }
    static V mul_even(V a, V b) { return _mm256_mul_epu32(a, b); }
    static V add64(V a, V b) { return _mm256_add_epi64(a, b); }
    static V sub64(V a, V b) { return _mm256_sub_epi64(a, b); }
    template <int k>
    static V shr64(V a) { return _mm256_srli_epi64(a, k); }
    template <int k>
    static V shl64(V a) { return _mm256_slli_epi64(a, k); }
    template <unsigned mask>
    static V blend(V a, V b) { return _mm256_blend_epi32(a, b, int(mask & 0xff)); }
    static V join_high(V e, V o) { return blend<0xaa>(shr64<32>(e), o); }
    static V mulhi(V a, V b);
    template <int k>
    static V mul_shr(V a, V b);

    using Index = __m256i;
    static Index index(const std::array<int, lanes>& from) {
        alignas(32) u32 words[8];
        for (int i = 0; i < 8; ++i) words[i] = u32(from[i]);
        return load(words);
    }
    static V permute(V a, Index idx) { return _mm256_permutevar8x32_epi32(a, idx); }
};
#endif

#if defined(__AVX2__) && defined(__AVX512VL__)
// Avx2's instructions with 32 registers (EVEX encodings) and AVX-512 extensions on ymm.
struct Avx512Vl : Avx2 {
    static constexpr int registers = 32;
    static constexpr const char* name = "avx512vl";
    static V permute2(V a, Index idx, V b) { return _mm256_permutex2var_epi32(a, idx, b); }
    static V join_high(V e, V o) { return permute2(e, _mm256_setr_epi32(1, 9, 3, 11, 5, 13, 7, 15), o); }
    static V mulhi(V a, V b);
    template <int k>
    static V mul_shr(V a, V b);
#if defined(__AVX512IFMA__)
    static V madd52lo(V c, V a, V b) { return _mm256_madd52lo_epu64(c, a, b); }
    static V madd52hi(V c, V a, V b) { return _mm256_madd52hi_epu64(c, a, b); }
#endif
};
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
struct Avx512 {
    using V = __m512i;
    static constexpr int lanes = 16;
    static constexpr int registers = 32;
    static constexpr const char* name = "avx512";

    static V load(const u32* p) { return _mm512_loadu_si512(p); }
    static void store(u32* p, V a) { _mm512_storeu_si512(p, a); }
    static V broadcast(u32 x) { return _mm512_set1_epi32(int(x)); }
    static V broadcast64(u64 x) { return _mm512_set1_epi64(std::int64_t(x)); }
    static V add(V a, V b) { return _mm512_add_epi32(a, b); }
    static V sub(V a, V b) { return _mm512_sub_epi32(a, b); }
    static V min(V a, V b) { return _mm512_min_epu32(a, b); }
    static V mul_lo(V a, V b) { return _mm512_mullo_epi32(a, b); }
    template <int k>
    static V shr(V a) { return _mm512_srli_epi32(a, k); }
    template <int k>
    static V shl(V a) { return _mm512_slli_epi32(a, k); }
    static V mul_sub(V c, V a, V b) { return sub(c, mul_lo(a, b)); }
    static V mul_even(V a, V b) { return _mm512_mul_epu32(a, b); }
    static V add64(V a, V b) { return _mm512_add_epi64(a, b); }
    static V sub64(V a, V b) { return _mm512_sub_epi64(a, b); }
    template <int k>
    static V shr64(V a) { return _mm512_srli_epi64(a, k); }
    template <int k>
    static V shl64(V a) { return _mm512_slli_epi64(a, k); }
    template <unsigned mask>
    static V blend(V a, V b) { return _mm512_mask_blend_epi32(__mmask16(mask & 0xffff), a, b); }
    static V permute2(V a, __m512i idx, V b) { return _mm512_permutex2var_epi32(a, idx, b); }
    static V join_high(V e, V o) {
        return permute2(e, _mm512_setr_epi32(1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31), o);
    }
    static V mulhi(V a, V b);
    template <int k>
    static V mul_shr(V a, V b);
#if defined(__AVX512IFMA__)
    static V madd52lo(V c, V a, V b) { return _mm512_madd52lo_epu64(c, a, b); }
    static V madd52hi(V c, V a, V b) { return _mm512_madd52hi_epu64(c, a, b); }
#endif

    using Index = __m512i;
    static Index index(const std::array<int, lanes>& from) {
        alignas(64) u32 words[16];
        for (int i = 0; i < 16; ++i) words[i] = u32(from[i]);
        return load(words);
    }
    static V permute(V a, Index idx) { return _mm512_permutexvar_epi32(idx, a); }
};
#endif

// x86 composed products, shared by the x86 structs: even lanes by mul_even, odd lanes by shifting
// them down first. Shifts of a broadcast constant fold at compile time.
namespace detail {

// mulhi joins its halves by join_high (vpermt2d on AVX-512) except on Zen 4 and 5, where shoup
// ran 5% slower with it (notes.md, 2026-10-10). -DFPS_SIMD_MULHI_PERMUTE=0 or 1 forces it.
#if defined(FPS_SIMD_MULHI_PERMUTE)
inline constexpr bool kMulhiPermute = FPS_SIMD_MULHI_PERMUTE;
#elif defined(__znver4__) || defined(__znver5__)
inline constexpr bool kMulhiPermute = false;
#else
inline constexpr bool kMulhiPermute = true;
#endif

template <class S>
typename S::V x86_mulhi(typename S::V a, typename S::V b) {
    const auto even = S::mul_even(a, b);
    const auto odd = S::mul_even(S::template shr64<32>(a), S::template shr64<32>(b));
    if constexpr (kMulhiPermute) return S::join_high(even, odd);
    else return S::template blend<0xaaaa>(S::template shr64<32>(even), odd);
}

template <class S, int k>
typename S::V x86_mul_shr(typename S::V a, typename S::V b) {
    static_assert(1 <= k && k <= 32);
    if constexpr (k == 32) {
        return x86_mulhi<S>(a, b);
    } else {
        const auto even = S::template shr64<k>(S::mul_even(a, b));
        const auto odd = S::mul_even(S::template shr64<32>(a), S::template shr64<32>(b));
        return S::template blend<0xaaaa>(even, S::template shl64<32 - k>(odd));
    }
}

}  // namespace detail

#if defined(__SSE4_1__)
inline Sse::V Sse::mulhi(V a, V b) { return detail::x86_mulhi<Sse>(a, b); }
template <int k>
Sse::V Sse::mul_shr(V a, V b) { return detail::x86_mul_shr<Sse, k>(a, b); }
#endif
#if defined(__AVX2__)
inline Avx2::V Avx2::mulhi(V a, V b) { return detail::x86_mulhi<Avx2>(a, b); }
template <int k>
Avx2::V Avx2::mul_shr(V a, V b) { return detail::x86_mul_shr<Avx2, k>(a, b); }
#endif
#if defined(__AVX2__) && defined(__AVX512VL__)
inline Avx512Vl::V Avx512Vl::mulhi(V a, V b) { return detail::x86_mulhi<Avx512Vl>(a, b); }
template <int k>
Avx512Vl::V Avx512Vl::mul_shr(V a, V b) { return detail::x86_mul_shr<Avx512Vl, k>(a, b); }
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__)
inline Avx512::V Avx512::mulhi(V a, V b) { return detail::x86_mulhi<Avx512>(a, b); }
template <int k>
Avx512::V Avx512::mul_shr(V a, V b) { return detail::x86_mul_shr<Avx512, k>(a, b); }
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)
struct Neon {
    using V = uint32x4_t;
    using W = uint64x2_t;
    static constexpr int lanes = 4;
    static constexpr int registers = 32;
    static constexpr const char* name = "neon";

    static V load(const u32* p) { return vld1q_u32(p); }
    static void store(u32* p, V a) { vst1q_u32(p, a); }
    static V broadcast(u32 x) { return vdupq_n_u32(x); }
    static V add(V a, V b) { return vaddq_u32(a, b); }
    static V sub(V a, V b) { return vsubq_u32(a, b); }
    static V min(V a, V b) { return vminq_u32(a, b); }
    static V mul_lo(V a, V b) { return vmulq_u32(a, b); }
    template <int k>
    static V shr(V a) { return vshrq_n_u32(a, k); }
    template <int k>
    static V shl(V a) { return vshlq_n_u32(a, k); }
    static V mul_sub(V c, V a, V b) { return vmlsq_u32(c, a, b); }
    template <unsigned mask>
    static V blend(V a, V b) {  // bsl with a constant mask
        const V select = {mask & 1 ? ~0u : 0u, mask & 2 ? ~0u : 0u, mask & 4 ? ~0u : 0u, mask & 8 ? ~0u : 0u};
        return vbslq_u32(select, b, a);
    }

    static W mul_wide_lo(V a, V b) { return vmull_u32(vget_low_u32(a), vget_low_u32(b)); }
    static W mul_wide_hi(V a, V b) { return vmull_high_u32(a, b); }
    static W mla_wide_lo(W c, V a, V b) { return vmlal_u32(c, vget_low_u32(a), vget_low_u32(b)); }
    static W mla_wide_hi(W c, V a, V b) { return vmlal_high_u32(c, a, b); }
    static V lo_halves(W x, W y) { return vuzp1q_u32(vreinterpretq_u32_u64(x), vreinterpretq_u32_u64(y)); }
    static V hi_halves(W x, W y) { return vuzp2q_u32(vreinterpretq_u32_u64(x), vreinterpretq_u32_u64(y)); }
    static V qdmulh(V a, V b) {
        return vreinterpretq_u32_s32(vqdmulhq_s32(vreinterpretq_s32_u32(a), vreinterpretq_s32_u32(b)));
    }
    static V hsub(V a, V b) {
        return vreinterpretq_u32_s32(vhsubq_s32(vreinterpretq_s32_u32(a), vreinterpretq_s32_u32(b)));
    }

#if defined(__ARM_FEATURE_SVE2)
    // SVE2 umulh on the low 128 bits of the z registers (any vector length).
    static V mulhi(V a, V b) {
        const svuint32_t x = svset_neonq_u32(svundef_u32(), a), y = svset_neonq_u32(svundef_u32(), b);
        return svget_neonq_u32(svmulh_u32_x(svptrue_b32(), x, y));
    }
#else
    static V mulhi(V a, V b) { return hi_halves(mul_wide_lo(a, b), mul_wide_hi(a, b)); }
#endif
    template <int k>
    static V mul_shr(V a, V b) {
        static_assert(1 <= k && k <= 32);
        if constexpr (k == 32) return mulhi(a, b);
        else return vshrn_high_n_u64(vshrn_n_u64(mul_wide_lo(a, b), k), mul_wide_hi(a, b), k);
    }

    using Index = uint8x16_t;  // tbl byte indices
    static Index index(const std::array<int, lanes>& from) {
        std::uint8_t bytes[16];
        for (int i = 0; i < 16; ++i) bytes[i] = std::uint8_t(4 * from[i / 4] + i % 4);
        return vld1q_u8(bytes);
    }
    static V permute(V a, Index idx) { return vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(a), idx)); }
};
#endif

}  // namespace fps::simd
