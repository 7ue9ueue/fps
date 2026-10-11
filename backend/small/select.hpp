// The small kernel's variant per target: the simd struct and the vectors per block of the
// schoolbook, and the costs behind beats_ntt (small.hpp). -DFPS_SMALL_ISA=<struct> and
// -DFPS_SMALL_VECTORS=<R> force the variant. Measurements: notes.md.
#pragma once

#include "backend/simd/select.hpp"

namespace fps::small {

#if defined(FPS_SMALL_ISA)
using Isa = simd::FPS_SMALL_ISA;
#else
using Isa = simd::Native;
#endif

// Geomean over the bench shapes against the best R: 16 registers (Zen 3) R = 2, 6, 8 1.12, 1.33,
// 1.40 of R = 4; 32 registers R = 4, 8 1.008-1.039 and 1.010-1.084 of R = 6.
#if defined(FPS_SMALL_VECTORS)
inline constexpr int kVectors = FPS_SMALL_VECTORS;
#else
inline constexpr int kVectors = Isa::registers == 32 ? 6 : 4;
#endif

// Time of small::multiply, about term n m + word (n + m) + call + square min(n, m)^2 ns, fitted on
// shapes within 0.4x-2.5x of the NTT's time; ntt[lg - 6]: ns of an ntt::Convolution of length
// 2^lg with the product in the caller's memory (construct, copy the factors in, multiply), for
// lg = 6 .. 25. Measured on the target's timing machine against main at e754d90 (notes.md).
struct Costs {
    double term, word, call, square;
    double ntt[20];
};

#if defined(__znver5__)  // Zen 5, lc-zen5
inline constexpr Costs kCosts = {0.0186, 0.1476, 36.2, 0.00675,
                                 {98, 176, 343, 713, 1315, 2792, 5794, 12422, 27063, 56213, 120376, 257521,
                                  550780, 1144384, 2428935, 5000803, 10985436, 24174429, 53491309, 112682724}};
#elif defined(__AVX512F__)  // Emerald Rapids, lc-intel
inline constexpr Costs kCosts = {0.0386, 0.2088, 43.3, 0.01401,
                                 {150, 233, 513, 1062, 2370, 5120, 10653, 23000, 49945, 105183, 225889, 502828,
                                  1109883, 2458242, 5357638, 11013981, 23868184, 48989648, 110801071, 235823031}};
#elif defined(__AVX2__)  // Zen 3, lc-bench
inline constexpr Costs kCosts = {0.0501, 0.2653, 60.1, 0.02697,
                                 {122, 282, 536, 1080, 2295, 4727, 10399, 22352, 47949, 102584, 221662, 478702,
                                  1009379, 2165323, 4430625, 9428182, 19517016, 42731171, 87740104, 196073152}};
#elif defined(__ARM_NEON)  // Neoverse V2, lc-arm (also the Mac's guess)
inline constexpr Costs kCosts = {0.1066, 0.5149, 58.7, 0.04264,
                                 {241, 503, 1048, 2285, 4836, 10466, 22206, 47396, 102171, 216091, 461490, 975359,
                                  2103790, 4417528, 9453275, 19741878, 42058095, 86710299, 183069591, 377584494}};
#elif defined(__SSE4_1__)  // sse: not a timed target; lc-bench with -march=x86-64-v2
inline constexpr Costs kCosts = {0.1061, 0.4981, 83.7, 0.05892,
                                 {350, 655, 1672, 3516, 5886, 12678, 25833, 55191, 119749, 251473, 528093, 1108995,
                                  2328891, 4890670, 10270407, 21567855, 45292496, 95114241, 199739907, 419453804}};
#else  // scalar: not a timed target; lc-bench with no flags
inline constexpr Costs kCosts = {0.3110, 2.3043, 149.3, 0.15745,
                                 {913, 2022, 4521, 9984, 21859, 47621, 103382, 221994, 479905, 1007800, 2116381, 4444400,
                                  9333240, 19599805, 41159590, 86435140, 181513793, 381178966, 800475828, 1680999239}};
#endif

}  // namespace fps::small
