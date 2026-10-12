// The small kernel's variant per target and shape, a tuned choice (backend/tune): Schoolbook<S, R,
// P, U, Solo, T> of one of the target's candidates, per shape class; and the costs behind beats_ntt
// (small.hpp).
//
//   shape_class(n, m)    n >= m >= 1: the octave of m, (2^(k-1), 2^k] for k <= 9 and m > 512 the
//                        last, times 4 ratios: n < 2m, n < 4m, n < 8m, the rest
//   variants.get()[c]    the variant of class c: an index into Selected
//   kDefaults            compiled defaults: tools/tune.py small (tune.cpp, tuned.hpp)
//
// backend/small/tune.hpp measures the table on this machine. -DFPS_SMALL_ISA=<struct> forces one
// variant, with -DFPS_SMALL_VECTORS=<R> (default 4), -DFPS_SMALL_UNROLL=<U> (1),
// -DFPS_SMALL_SOLO=<0 or 1> (0) and -DFPS_SMALL_TINY=<T> (2). Measurements: notes.md.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "backend/simd/select.hpp"
#include "backend/small/tuned.hpp"
#include "backend/tune/tune.hpp"

namespace fps::small {

// Schoolbook<S, R, P, U, Solo, T> without P: R vectors per block, the loop with all R unrolled U
// times, Solo the edges one vector at a time, products of at most T vectors on the tiny path.
template <class S, int R, int U, bool Solo, int T = 2>
struct Variant {
    using Isa = S;
    static constexpr int vectors = R, unroll = U, tiny = T;
    static constexpr bool solo = Solo;
};
template <class... V>
struct Candidates {};

// Per target, the variants that are fastest in some shape class on its machine; the first is the
// variant used for every shape before the table (notes.md, Log).
#if defined(FPS_SMALL_ISA)
#if !defined(FPS_SMALL_VECTORS)
#define FPS_SMALL_VECTORS 4
#endif
#if !defined(FPS_SMALL_UNROLL)
#define FPS_SMALL_UNROLL 1
#endif
#if !defined(FPS_SMALL_SOLO)
#define FPS_SMALL_SOLO 0
#endif
#if !defined(FPS_SMALL_TINY)
#define FPS_SMALL_TINY 2
#endif
using Selected = Candidates<
    Variant<simd::FPS_SMALL_ISA, FPS_SMALL_VECTORS, FPS_SMALL_UNROLL, bool(FPS_SMALL_SOLO), FPS_SMALL_TINY>>;
#elif defined(__znver5__)  // zen5 (tune::kTarget)
using Selected = Candidates<Variant<simd::Avx512, 6, 1, false>, Variant<simd::Avx512, 4, 1, false>,
                            Variant<simd::Avx512, 8, 2, false>, Variant<simd::Avx512, 6, 1, true>,
                            Variant<simd::Avx512, 4, 1, true>, Variant<simd::Avx512Vl, 4, 1, true>,
                            Variant<simd::Avx512, 6, 1, false, 3>, Variant<simd::Avx512Vl, 4, 1, true, 4>>;
#elif defined(__AVX512F__) && defined(__AVX512BW__)  // avx512
using Selected = Candidates<Variant<simd::Avx512, 6, 1, false>, Variant<simd::Avx512, 4, 1, false>,
                            Variant<simd::Avx512, 6, 2, false>, Variant<simd::Avx512, 8, 1, false>,
                            Variant<simd::Avx512, 8, 2, false>, Variant<simd::Avx512Vl, 6, 2, false>,
                            Variant<simd::Avx512Vl, 4, 1, true>, Variant<simd::Avx512, 4, 1, false, 3>,
                            Variant<simd::Avx512Vl, 4, 1, true, 4>, Variant<simd::Avx512, 6, 1, false, 3>>;
#elif defined(__AVX2__)
using Selected = Candidates<Variant<simd::Avx2, 4, 2, false>, Variant<simd::Avx2, 4, 4, false>,
                            Variant<simd::Avx2, 4, 1, false>, Variant<simd::Avx2, 5, 2, false>,
                            Variant<simd::Avx2, 4, 2, true>, Variant<simd::Sse, 4, 1, true>,
                            Variant<simd::Avx2, 4, 2, true, 4>, Variant<simd::Sse, 4, 1, true, 4>,
                            Variant<simd::Avx2, 4, 2, false, 4>>;
#elif defined(__ARM_NEON) && defined(__aarch64__)
using Selected = Candidates<Variant<simd::Neon, 6, 4, false>, Variant<simd::Neon, 6, 4, true>,
                            Variant<simd::Neon, 4, 1, false>, Variant<simd::Neon, 8, 4, false>,
                            Variant<simd::Neon, 8, 1, false>, Variant<simd::Neon, 6, 4, true, 4>,
                            Variant<simd::Neon, 8, 1, false, 4>>;
#elif defined(__SSE4_1__)
using Selected = Candidates<Variant<simd::Sse, 4, 1, false>, Variant<simd::Sse, 4, 2, false>,
                            Variant<simd::Sse, 4, 1, true>, Variant<simd::Sse, 4, 1, true, 4>>;
#else
using Selected = Candidates<Variant<simd::Scalar, 4, 1, false>, Variant<simd::Scalar, 6, 1, false>,
                            Variant<simd::Scalar, 8, 1, false>, Variant<simd::Scalar, 4, 2, false>,
                            Variant<simd::Scalar, 4, 1, true>>;
#endif

inline constexpr int kCandidates = []<class... V>(Candidates<V...>) { return int(sizeof...(V)); }(Selected{});

// Shape classes. The octave of m sets the steps per block and the edges' length; the ratio n / m
// the share of interior blocks.
inline constexpr int kOctaves = 11, kRatios = 4, kClasses = kOctaves * kRatios;
constexpr int shape_class(std::size_t n, std::size_t m) {  // n >= m >= 1
    return std::min(int(std::bit_width(m - 1)), kOctaves - 1) * kRatios + (n >= 2 * m) + (n >= 4 * m) + (n >= 8 * m);
}

using Choices = std::array<std::uint8_t, kClasses>;

// "<isa>.r<R>.u<U>", ".solo" with Solo and ".t<T>" for T other than 2, e.g. "avx512.r4.u1.solo.t4";
// and a hash of the list that tuned.hpp records, so a table measured for another list does not
// compile.
template <class V>
std::string variant_name() {
    return std::string(V::Isa::name) + ".r" + std::to_string(V::vectors) + ".u" + std::to_string(V::unroll) +
           (V::solo ? ".solo" : "") + (V::tiny != 2 ? ".t" + std::to_string(V::tiny) : "");
}
template <class... V>
constexpr std::uint32_t signature(Candidates<V...>) {
    constexpr std::string_view digits = "0123456789";
    tune::Hash h;
    (h.add(V::Isa::name)
         .add(".r")
         .add(digits.substr(V::vectors, 1))
         .add(".u")
         .add(digits.substr(V::unroll, 1))
         .add(V::solo ? ".solo" : "")
         .add(V::tiny != 2 ? ".t" : "")
         .add(V::tiny != 2 ? digits.substr(V::tiny, 1) : "")
         .add(";"),
     ...);
    return h.value >> 1;  // fits an int
}
inline constexpr std::uint32_t kSignature = signature(Selected{});

constexpr bool valid(const Choices& choices) {
    return std::ranges::all_of(choices, [](std::uint8_t v) { return v < kCandidates; });
}

inline constexpr int kTuned = int(tune::kTarget);

#if defined(FPS_SMALL_ISA)
inline constexpr Choices kDefaults{};
#else
static_assert(tuned::signature[kTuned] == kSignature, "tuned.hpp was measured for other candidates: run tools/tune.py small");
inline constexpr Choices kDefaults = [] {
    Choices choices{};
    for (int c = 0; c < kClasses; ++c) choices[c] = std::uint8_t(tuned::variant[kTuned][c]);
    return choices;
}();
#endif
static_assert(valid(kDefaults));

inline constinit tune::Table<kDefaults> variants;

// Time of multiply(), about term n m + word (n + m) + call + square min(n, m)^2 ns: fitted per
// target by tools/tune.py (tune.cpp) on the target's machine (tuned.hpp).
struct Model {
    double term, word, call, square;
};

inline constexpr Model kModel = {tuned::term[kTuned], tuned::word[kTuned], tuned::call[kTuned], tuned::square[kTuned]};

// The NTT's time per transform length 2^lg, lg = 6 .. 25, in the model's ns: an ntt::Convolution
// with the product in the caller's memory (construct, copy the factors in, multiply), times the
// model's over multiply()'s measured time where the two cross. The compiled defaults are
// tools/tune.py's; small::measure_ntt (tune.hpp) measures them on this machine.
inline constexpr std::array<double, 20> kNttDefaults = std::to_array(tuned::ntt[kTuned]);
inline constinit tune::Table<kNttDefaults> ntt_times;

}  // namespace fps::small
