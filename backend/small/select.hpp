// The small kernel's variant per target: the simd struct, the vectors per block, the unrolling and
// the vector-at-a-time edges of the schoolbook, and the costs behind beats_ntt (small.hpp).
// -DFPS_SMALL_ISA=<struct>, -DFPS_SMALL_VECTORS=<R>, -DFPS_SMALL_UNROLL=<U> and
// -DFPS_SMALL_SOLO=<words> force the variant. Measurements: notes.md.
#pragma once

#include <array>
#include <cstddef>

#include "backend/simd/select.hpp"
#include "backend/small/tuned.hpp"
#include "backend/tune/tune.hpp"

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

// Unrolling of the loop over b with all vectors of a block, on long products: Zen 3 U = 2 0.95 of
// U = 1, Neoverse V2 U = 4 0.95; no gain on Emerald Rapids, Zen 5 1.00-1.10 (notes.md, Log).
#if defined(FPS_SMALL_UNROLL)
inline constexpr int kUnroll = FPS_SMALL_UNROLL;
#elif defined(__ARM_NEON) && defined(__aarch64__)
inline constexpr int kUnroll = 4;
#elif defined(__AVX2__) && !defined(__AVX512F__)
inline constexpr int kUnroll = 2;
#else
inline constexpr int kUnroll = 1;
#endif

// Edge blocks one vector after the other, without phases, for b of at most kSolo words: Neoverse
// V2 0.92-0.97 of the phases for b of 8-16 words; x86 up to 1.27 on unbalanced shapes (notes.md).
#if defined(FPS_SMALL_SOLO)
inline constexpr std::size_t kSolo = FPS_SMALL_SOLO;
#elif defined(__ARM_NEON) && defined(__aarch64__)
inline constexpr std::size_t kSolo = 16;
#else
inline constexpr std::size_t kSolo = 0;
#endif

// Time of multiply(), about term n m + word (n + m) + call + square min(n, m)^2 ns: fitted per
// target by tools/tune.py (tune.cpp) on the target's machine (tuned.hpp).
struct Model {
    double term, word, call, square;
};

inline constexpr int kTuned = int(tune::kTarget);
inline constexpr Model kModel = {tuned::term[kTuned], tuned::word[kTuned], tuned::call[kTuned], tuned::square[kTuned]};

// The NTT's time per transform length 2^lg, lg = 6 .. 25, in the model's ns: an ntt::Convolution
// with the product in the caller's memory (construct, copy the factors in, multiply), times the
// model's over multiply()'s measured time where the two cross. The compiled defaults are
// tools/tune.py's; small::measure_ntt (tune.hpp) measures them on this machine.
inline constexpr std::array<double, 20> kNttDefaults = std::to_array(tuned::ntt[kTuned]);
inline constinit tune::Table<kNttDefaults> ntt_times;

}  // namespace fps::small
