// The factorials variant per target and size n: the scalar loop (model.hpp) below the first tier,
// then Chains<S, K, P, step> of the last tier whose min_size <= n. Fewer lanes (W = K S::lanes)
// win at small n: less duplicate work and fewer lanes to start. Measurements: notes.md.
// -DFPS_FACTORIALS_ISA=<struct>, -DFPS_FACTORIALS_VECTORS=<K>, -DFPS_FACTORIALS_STEP=<step> and
// -DFPS_FACTORIALS_MIN_CHAINS=<n> force one tier from n on (defaults 2, shoup, 64).
#pragma once

#include <cstddef>

#include "backend/simd/select.hpp"

namespace fps::factorials {

// The scan's chain product (Chains): Montgomery with the multiplier in Montgomery form, or Shoup
// with the multiplier plain and its quotient stepped along.
enum class Step { montgomery, shoup };

template <class S, int K, Step step, std::size_t from>
struct Tier {
    using Isa = S;
    static constexpr int vectors = K;
    static constexpr Step scan = step;
    static constexpr std::size_t min_size = from;
};
template <class... T>  // ascending min_size
struct Tiers {};

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
using Selected = Tiers<Tier<simd::FPS_FACTORIALS_ISA, FPS_FACTORIALS_VECTORS, Step::FPS_FACTORIALS_STEP,
                            FPS_FACTORIALS_MIN_CHAINS>>;
#elif defined(__znver5__)
// Zen 5. ymm: Shoup on Avx512Vl (Avx2's K = 2 ran 2.4x slower in one build); zmm: Montgomery
// (Shoup 5-11% slower from 2^14). zmm's 32 lanes stall on memory from 5 2^20 on (round 1).
using Selected = Tiers<Tier<simd::Sse, 1, Step::shoup, 16>, Tier<simd::Avx512Vl, 1, Step::shoup, 48>,
                       Tier<simd::Avx512Vl, 2, Step::shoup, 256>, Tier<simd::Avx512, 2, Step::montgomery, 2048>,
                       Tier<simd::Avx512Vl, 2, Step::shoup, std::size_t(5) << 20>>;
#elif defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
// Emerald Rapids: Montgomery (Shoup 4-6% slower: vpmulld is 2 uops). zmm's 32 lanes stall on
// memory from 2^19 on, by the tables' placement (rounds 1 and 2).
using Selected = Tiers<Tier<simd::Sse, 1, Step::shoup, 16>, Tier<simd::Avx512Vl, 1, Step::montgomery, 48>,
                       Tier<simd::Avx512Vl, 2, Step::montgomery, 256>, Tier<simd::Avx512, 2, Step::montgomery, 1024>,
                       Tier<simd::Avx512Vl, 2, Step::montgomery, std::size_t(1) << 19>>;
#elif defined(__AVX2__)
// Zen 3: Shoup.
using Selected = Tiers<Tier<simd::Sse, 1, Step::shoup, 16>, Tier<simd::Avx2, 1, Step::shoup, 48>,
                       Tier<simd::Avx2, 2, Step::shoup, 2048>>;
#elif defined(__SSE4_1__)
using Selected = Tiers<Tier<simd::Sse, 1, Step::shoup, 16>, Tier<simd::Sse, 2, Step::shoup, 256>>;
#elif defined(__ARM_NEON) && defined(__aarch64__)
// Neoverse V2: Shoup (umulh, mul, mls with SVE2).
using Selected = Tiers<Tier<simd::Neon, 1, Step::shoup, 16>, Tier<simd::Neon, 2, Step::shoup, 64>,
                       Tier<simd::Neon, 4, Step::shoup, 1280>>;
#else
using Selected = Tiers<Tier<simd::Scalar, 4, Step::montgomery, 64>>;
#endif

}  // namespace fps::factorials
