// The factorials variant per target and size n, a tuned choice (backend/tune): the scalar loop
// (model.hpp) or Chains<S, K, P, step> of one of the target's candidates, per size class.
//
//   size_class(n)            n >= kFirstClass: quarter octaves (e, e + 2^(k-2)], e = 2^k (1 + q / 4)
//   variants.get()[c]        the variant of class c: 0 the loop, i >= 1 the candidate Selected[i - 1]
//   kDefaults                compiled defaults: tools/tune.py factorials (tune.cpp, tuned.hpp)
//
// Below kFirstClass the loop. A candidate's lanes cover W C(n) >= n positions (Chains::chunk), a
// step function of n whose steps fall on multiples of its lane grid: a class ends at a round size
// (a best case) and the other sizes tune.hpp times sit just past round sizes (worst cases).
// backend/factorials/tune.hpp measures the table on this machine.
// -DFPS_FACTORIALS_ISA=<struct>, -DFPS_FACTORIALS_VECTORS=<K>, -DFPS_FACTORIALS_STEP=<step> and
// -DFPS_FACTORIALS_MIN_CHAINS=<n> force one variant from the class of n on (defaults 2, shoup, 64).
// Measurements: notes.md.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "backend/factorials/tuned.hpp"
#include "backend/simd/select.hpp"
#include "backend/tune/tune.hpp"

namespace fps::factorials {

// The scan's chain product (Chains): Montgomery with the multiplier in Montgomery form, or Shoup
// with the multiplier plain and its quotient stepped along.
enum class Step { montgomery, shoup };

// prefetch: the scan prefetches each lane's next line (Chains).
template <class S, int K, Step step, bool prefetching = false>
struct Variant {
    using Isa = S;
    static constexpr int vectors = K;
    static constexpr Step scan = step;
    static constexpr bool prefetch = prefetching;
};
template <class... V>
struct Candidates {};

#if defined(FPS_FACTORIALS_ISA)
#if !defined(FPS_FACTORIALS_VECTORS)
#define FPS_FACTORIALS_VECTORS 2
#endif
#if !defined(FPS_FACTORIALS_STEP)
#define FPS_FACTORIALS_STEP shoup
#endif
#if !defined(FPS_FACTORIALS_MIN_CHAINS)
#define FPS_FACTORIALS_MIN_CHAINS 64
#endif
using Selected = Candidates<Variant<simd::FPS_FACTORIALS_ISA, FPS_FACTORIALS_VECTORS, Step::FPS_FACTORIALS_STEP>>;
#elif defined(__znver5__)  // zen5 (tune::kTarget): no prefetching variant yet (notes.md, issue #70)
using Selected = Candidates<Variant<simd::Sse, 1, Step::shoup>, Variant<simd::Avx512Vl, 1, Step::shoup>,
                            Variant<simd::Avx512Vl, 1, Step::montgomery>, Variant<simd::Avx512Vl, 2, Step::montgomery>,
                            Variant<simd::Avx512, 1, Step::shoup>, Variant<simd::Avx512, 1, Step::montgomery>,
                            Variant<simd::Avx512, 2, Step::shoup>, Variant<simd::Avx512, 2, Step::montgomery>>;
#elif defined(__AVX512F__) && defined(__AVX512BW__)  // avx512
using Selected = Candidates<Variant<simd::Sse, 1, Step::shoup>, Variant<simd::Avx512Vl, 1, Step::shoup>,
                            Variant<simd::Avx512Vl, 1, Step::montgomery>, Variant<simd::Avx512Vl, 2, Step::montgomery>,
                            Variant<simd::Avx512, 1, Step::shoup>, Variant<simd::Avx512, 1, Step::montgomery>,
                            Variant<simd::Avx512, 2, Step::shoup, true>, Variant<simd::Avx512, 2, Step::montgomery>,
                            Variant<simd::Avx512, 2, Step::montgomery, true>>;
#elif defined(__AVX2__)
using Selected = Candidates<Variant<simd::Sse, 1, Step::shoup>, Variant<simd::Avx2, 1, Step::shoup>,
                            Variant<simd::Avx2, 1, Step::montgomery>, Variant<simd::Avx2, 2, Step::shoup>,
                            Variant<simd::Avx2, 2, Step::montgomery>>;
#elif defined(__ARM_NEON) && defined(__aarch64__)
using Selected = Candidates<Variant<simd::Neon, 1, Step::shoup>, Variant<simd::Neon, 1, Step::montgomery>,
                            Variant<simd::Neon, 2, Step::shoup>, Variant<simd::Neon, 2, Step::montgomery>,
                            Variant<simd::Neon, 4, Step::shoup>, Variant<simd::Neon, 4, Step::montgomery>>;
#elif defined(__SSE4_1__)
using Selected = Candidates<Variant<simd::Sse, 1, Step::shoup>, Variant<simd::Sse, 1, Step::montgomery>,
                            Variant<simd::Sse, 2, Step::shoup>, Variant<simd::Sse, 2, Step::montgomery>>;
#else
using Selected = Candidates<Variant<simd::Scalar, 2, Step::shoup>, Variant<simd::Scalar, 2, Step::montgomery>,
                            Variant<simd::Scalar, 4, Step::shoup>, Variant<simd::Scalar, 4, Step::montgomery>>;
#endif

// Size classes: (e, e + 2^(k-2)] for e = 2^k (1 + q / 4), k >= 4, q < 4; the last also takes every
// larger n. Class c is (class_end(c - 1), class_end(c)].
inline constexpr std::size_t kFirstClass = 17;  // below: the loop
inline constexpr int kClasses = 88;             // up to 2^26
constexpr int size_class(std::size_t n) {  // n >= kFirstClass
    const std::size_t m = n - 1;
    const int k = std::bit_width(m) - 1;
    return std::min(4 * (k - 4) + int(m >> (k - 2) & 3), kClasses - 1);
}
constexpr std::size_t class_end(int c) { return std::size_t(5 + c % 4) << (c / 4 + 2); }
constexpr std::size_t class_start(int c) { return c == 0 ? kFirstClass : class_end(c - 1) + 1; }

using Choices = std::array<std::uint8_t, kClasses>;

// The candidates' names, e.g. "avx512.1.shoup" or "avx512.2.montgomery.prefetch", and a hash of
// the list that tuned.hpp records, so a table measured for another list does not compile.
template <class V>
constexpr std::string_view step_name = V::scan == Step::shoup ? "shoup" : "montgomery";
template <class V>
constexpr std::string_view prefetch_name = V::prefetch ? ".prefetch" : "";
template <class... V>
constexpr std::uint32_t signature(Candidates<V...>) {
    std::uint32_t h = 2166136261u;  // FNV-1a over "isa.K.step[.prefetch];" per candidate
    const auto add = [&h](std::string_view s) {
        for (const char c : s) h = (h ^ std::uint8_t(c)) * 16777619u;
    };
    ((add(V::Isa::name), add("."), add(std::string_view("0123456789").substr(V::vectors, 1)), add("."),
      add(step_name<V>), add(prefetch_name<V>), add(";")),
     ...);
    return h >> 1;  // fits an int
}
inline constexpr std::uint32_t kSignature = signature(Selected{});

// The least n each variant runs: 0 for the loop, Chains::kMinSize = 4 S::lanes for the rest.
inline constexpr auto kMinSizes = []<class... V>(Candidates<V...>) {
    return std::array<std::size_t, sizeof...(V) + 1>{0, 4 * std::size_t(V::Isa::lanes)...};
}(Selected{});

// A table is valid if every class's variant exists and runs at the class's least n.
constexpr bool valid(const Choices& choices) {
    for (int c = 0; c < kClasses; ++c)
        if (choices[c] >= kMinSizes.size() || kMinSizes[choices[c]] > class_start(c)) return false;
    return true;
}

#if defined(FPS_FACTORIALS_ISA)
inline constexpr Choices kDefaults = [] {
    Choices choices{};
    for (int c = 0; c < kClasses; ++c)
        choices[c] = class_start(c) >= std::max<std::size_t>(FPS_FACTORIALS_MIN_CHAINS, kMinSizes[1]);
    return choices;
}();
#else
inline constexpr int kTuned = int(tune::kTarget);
static_assert(tuned::signature[kTuned] == kSignature,
              "tuned.hpp was measured for other candidates: run tools/tune.py factorials");
inline constexpr Choices kDefaults = [] {
    Choices choices{};
    for (int c = 0; c < kClasses; ++c) choices[c] = std::uint8_t(tuned::variant[kTuned][c]);
    return choices;
}();
#endif
static_assert(valid(kDefaults));

inline constinit tune::Table<kDefaults> variants;

}  // namespace fps::factorials
