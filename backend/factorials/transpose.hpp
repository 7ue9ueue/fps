// In-register transposes of lanes x lanes words, one per simd vector type: on entry r[t] holds
// step t of every lane, on exit r[l] holds every step of lane l (lane l of r[t] moves to lane t
// of r[l]). Shuffle counts: 4 x 4 8, 8 x 8 24, 16 x 16 64.
#pragma once

#include <cstdint>

#include "backend/simd/simd.hpp"

namespace fps::factorials {

inline void transpose(std::uint32_t (&)[1]) {}

#if defined(__SSE4_1__)
inline void transpose(__m128i (&r)[4]) {
    const __m128i t0 = _mm_unpacklo_epi32(r[0], r[1]), t1 = _mm_unpackhi_epi32(r[0], r[1]);
    const __m128i t2 = _mm_unpacklo_epi32(r[2], r[3]), t3 = _mm_unpackhi_epi32(r[2], r[3]);
    r[0] = _mm_unpacklo_epi64(t0, t2), r[1] = _mm_unpackhi_epi64(t0, t2);
    r[2] = _mm_unpacklo_epi64(t1, t3), r[3] = _mm_unpackhi_epi64(t1, t3);
}
#endif

#if defined(__AVX2__)
// Within 128-bit halves, then the halves (vperm2i128).
inline void transpose(__m256i (&r)[8]) {
    __m256i t[8], u[8];
#pragma GCC unroll 4
    for (int i = 0; i < 8; i += 2) {
        t[i] = _mm256_unpacklo_epi32(r[i], r[i + 1]);
        t[i + 1] = _mm256_unpackhi_epi32(r[i], r[i + 1]);
    }
#pragma GCC unroll 2
    for (int i = 0; i < 8; i += 4) {  // u[i + m]: column m (low half), 4 + m (high half)
        u[i] = _mm256_unpacklo_epi64(t[i], t[i + 2]);
        u[i + 1] = _mm256_unpackhi_epi64(t[i], t[i + 2]);
        u[i + 2] = _mm256_unpacklo_epi64(t[i + 1], t[i + 3]);
        u[i + 3] = _mm256_unpackhi_epi64(t[i + 1], t[i + 3]);
    }
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        r[i] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x20);
        r[i + 4] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x31);
    }
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
// Within 128-bit chunks, then the chunks in two vshufi32x4 levels.
inline void transpose(__m512i (&r)[16]) {
    __m512i t[16], u[16];
#pragma GCC unroll 8
    for (int i = 0; i < 16; i += 2) {
        t[i] = _mm512_unpacklo_epi32(r[i], r[i + 1]);
        t[i + 1] = _mm512_unpackhi_epi32(r[i], r[i + 1]);
    }
#pragma GCC unroll 4
    for (int i = 0; i < 16; i += 4) {  // u[4g + m], chunk c: rows 4g .. 4g + 3 of column 4c + m
        u[i] = _mm512_unpacklo_epi64(t[i], t[i + 2]);
        u[i + 1] = _mm512_unpackhi_epi64(t[i], t[i + 2]);
        u[i + 2] = _mm512_unpacklo_epi64(t[i + 1], t[i + 3]);
        u[i + 3] = _mm512_unpackhi_epi64(t[i + 1], t[i + 3]);
    }
#pragma GCC unroll 4
    for (int m = 0; m < 4; ++m) {
        const __m512i low0 = _mm512_shuffle_i32x4(u[m], u[4 + m], 0x44);  // chunks 0, 1 of each
        const __m512i high0 = _mm512_shuffle_i32x4(u[m], u[4 + m], 0xee);  // chunks 2, 3
        const __m512i low1 = _mm512_shuffle_i32x4(u[8 + m], u[12 + m], 0x44);
        const __m512i high1 = _mm512_shuffle_i32x4(u[8 + m], u[12 + m], 0xee);
        r[m] = _mm512_shuffle_i32x4(low0, low1, 0x88);  // chunks 0, 2 of each
        r[4 + m] = _mm512_shuffle_i32x4(low0, low1, 0xdd);  // chunks 1, 3
        r[8 + m] = _mm512_shuffle_i32x4(high0, high1, 0x88);
        r[12 + m] = _mm512_shuffle_i32x4(high0, high1, 0xdd);
    }
}
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)
inline void transpose(uint32x4_t (&r)[4]) {
    const uint32x4_t t0 = vtrn1q_u32(r[0], r[1]), t1 = vtrn2q_u32(r[0], r[1]);
    const uint32x4_t t2 = vtrn1q_u32(r[2], r[3]), t3 = vtrn2q_u32(r[2], r[3]);
    const auto pairs = [](uint32x4_t x) { return vreinterpretq_u64_u32(x); };
    r[0] = vreinterpretq_u32_u64(vtrn1q_u64(pairs(t0), pairs(t2)));
    r[2] = vreinterpretq_u32_u64(vtrn2q_u64(pairs(t0), pairs(t2)));
    r[1] = vreinterpretq_u32_u64(vtrn1q_u64(pairs(t1), pairs(t3)));
    r[3] = vreinterpretq_u32_u64(vtrn2q_u64(pairs(t1), pairs(t3)));
}
#endif

}  // namespace fps::factorials
