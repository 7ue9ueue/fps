// The evaluation kernel's choices per target: where the direct O(n m) paths give way to the trees.
// Measurements: notes.md, notes/104-r1.md and notes/136-r1.md (grids on each timing machine).
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>

namespace fps::evaluation {

// The trees need more than kLaneBase slots of 8 points (backend/product_tree): fewer points are
// padded with zeros (factors 1) up to kMinSlots slots.
inline constexpr std::size_t kMinSlots = 17;

// evaluate(): Horner's rule costs about kHorner ns per coefficient and point; the tree about
// kFixed + kPerPoint m8 lg(m8) + kPerCoefficient K lg(K) ns (m8 points with padding, K = max(n,
// m8)), fitted (relative least squares) to a grid of n, m in 8 .. 16384 where the two are within
// 4x of each other (notes/136-r1.md). Horner runs when it is the cheaper.
struct EvaluateCosts {
    double horner, fixed, point, coefficient;
};
#if defined(__znver5__)
inline constexpr EvaluateCosts kEvaluateCosts{0.064, 3739, 1.59, 0.391};  // lc-zen5
#elif defined(__AVX512F__)
inline constexpr EvaluateCosts kEvaluateCosts{0.1065, 3621, 2.353, 0.746};  // lc-intel(-test)
#elif defined(__AVX2__)
inline constexpr EvaluateCosts kEvaluateCosts{0.21, 4840, 2.90, 0.860};  // lc-bench
#elif defined(__ARM_NEON) && defined(__aarch64__)
inline constexpr EvaluateCosts kEvaluateCosts{0.2286, 4247, 4.338, 1.300};  // lc-arm(-test)
#elif defined(__SSE4_1__)
inline constexpr EvaluateCosts kEvaluateCosts{0.42, 6450, 3.87, 1.15};  // a guess: AVX2's tree costs times 4/3, Horner at half the width
#else
inline constexpr EvaluateCosts kEvaluateCosts{1.5, 40000, 20.0, 5.0};  // a guess
#endif

inline bool evaluate_directly(std::size_t n, std::size_t m) {
    const std::size_t m8 = std::max<std::size_t>((m + 7) / 8 * 8, 8 * kMinSlots), big = std::max(n, m8);
    const auto lg = [](std::size_t x) { return double(std::bit_width(x)); };
    const EvaluateCosts& c = kEvaluateCosts;
    return c.horner * double(n) * double(m) <= c.fixed + c.point * double(m8) * lg(m8) + c.coefficient * double(big) * lg(big);
}

// interpolate() and to_newton_basis(): their O(m^2) paths in vectors (interpolate_small,
// newton_small) up to this many points, about where the trees take over on each timing machine
// (notes/136-r1.md; the scalar loops before crossed at about 36 and 70).
#if defined(__znver5__)
inline constexpr std::size_t kDirectInterpolate = 128, kDirectNewton = 400;  // lc-zen5
#elif defined(__AVX512F__)
inline constexpr std::size_t kDirectInterpolate = 128, kDirectNewton = 400;  // lc-intel(-test)
#elif defined(__AVX2__)
inline constexpr std::size_t kDirectInterpolate = 96, kDirectNewton = 352;  // lc-bench
#elif defined(__ARM_NEON) && defined(__aarch64__)
inline constexpr std::size_t kDirectInterpolate = 96, kDirectNewton = 192;  // lc-arm(-test)
#elif defined(__SSE4_1__)
inline constexpr std::size_t kDirectInterpolate = 80, kDirectNewton = 288;  // a guess: AVX2's times 0.8 (half the width, trees 1.3x slower)
#else
inline constexpr std::size_t kDirectInterpolate = 32, kDirectNewton = 64;  // a guess: the scalar loops' crossovers
#endif

// to_newton_basis(): slots per lane C are padded to the power of two above when C is past this
// fraction of it (lc-opt's choice; newton_slots).
inline constexpr double kNewtonPad = 0.65;

// interpolate(): lane nodes of at most this many slots (4 bases of 16) run as one group: the
// descent to all their bases' values, one inversion for them all (one power), then the sums.
// Against 32 slots (two bases), interpolate at 2^12-2^20: Zen 5 0.978-0.991 of the time, Zen 3,
// Neoverse V2 0.993-1.002, Emerald Rapids 0.99-1.00; 128 and 256 no better (notes/136-r3.md).
// -DFPS_EVALUATION_GROUP=s sets another (at least 32).
#if defined(FPS_EVALUATION_GROUP)
inline constexpr std::size_t kInterpolateGroup = FPS_EVALUATION_GROUP;
#else
inline constexpr std::size_t kInterpolateGroup = 64;
#endif

// The lanes' halvings leave their combine (P_lo -+ T) to the child's products, which then read
// both halves (pointwise_twice): the descent at 2^17 0.978 of the time on Zen 3 and 0.985 on
// Emerald Rapids, 0.976-1.003 on Zen 5 (two runs), 1.005 on Neoverse V2 (notes/136-r2.md). The
// Newton basis' too (combine_product, #136 r3): newton 0.995-1.000 on Zen 3, about 1 on Emerald
// Rapids and Zen 5.
#if defined(__ARM_NEON) && defined(__aarch64__)
inline constexpr bool kDeferredCombine = false;
#else
inline constexpr bool kDeferredCombine = true;
#endif

}  // namespace fps::evaluation
