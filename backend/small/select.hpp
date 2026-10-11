// The small kernel's variant per target: the simd struct and the vectors per block of the
// schoolbook, and the costs behind beats_ntt (small.hpp). -DFPS_SMALL_ISA=<struct> and
// -DFPS_SMALL_VECTORS=<R> force the variant. Measurements: notes.md.
#pragma once

#include <array>

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
