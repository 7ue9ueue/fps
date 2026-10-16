// The evaluation kernel's choices per target: where the direct O(n m) paths give way to the trees.
// Measurements: notes.md (issue #104, crossover grids on each timing machine).
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
// m8)), fitted to n or m = 4096 against 8 (notes.md). Horner runs when it is the cheaper.
struct EvaluateCosts {
    double horner, fixed, point, coefficient;
};
#if defined(__znver5__)
inline constexpr EvaluateCosts kEvaluateCosts{0.063, 7000, 2.4, 0.59};  // lc-zen5
#elif defined(__AVX512F__)
inline constexpr EvaluateCosts kEvaluateCosts{0.105, 9400, 3.5, 0.9};  // lc-intel
#elif defined(__AVX2__)
inline constexpr EvaluateCosts kEvaluateCosts{0.21, 11000, 4.5, 1.08};  // lc-bench
#elif defined(__ARM_NEON) && defined(__aarch64__)
inline constexpr EvaluateCosts kEvaluateCosts{0.227, 15000, 6.9, 1.66};  // lc-arm
#elif defined(__SSE4_1__)
inline constexpr EvaluateCosts kEvaluateCosts{0.42, 15000, 6.0, 1.5};  // a guess: AVX2's, Horner at half the width
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
inline constexpr std::size_t kDirectInterpolate = 104, kDirectNewton = 240;  // lc-intel
#elif defined(__AVX2__)
inline constexpr std::size_t kDirectInterpolate = 96, kDirectNewton = 352;  // lc-bench
#elif defined(__ARM_NEON) && defined(__aarch64__)
inline constexpr std::size_t kDirectInterpolate = 104, kDirectNewton = 184;  // lc-arm
#elif defined(__SSE4_1__)
inline constexpr std::size_t kDirectInterpolate = 80, kDirectNewton = 160;  // a guess: AVX2's at half the width
#else
inline constexpr std::size_t kDirectInterpolate = 32, kDirectNewton = 64;  // a guess: the scalar loops' crossovers
#endif

// to_newton_basis(): slots per lane C are padded to the power of two above when C is past this
// fraction of it (lc-opt's choice; newton_slots).
inline constexpr double kNewtonPad = 0.65;

}  // namespace fps::evaluation
