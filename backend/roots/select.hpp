// The root finder's structs per target: the Graeffe steps on product_tree::Lanes (8 words of 8
// lanes); the cosets' folds on simd::Avx512 where the target has it (16 cosets or tracked points
// per group, zmm products: twice ymm's per instruction at the same throughput on Emerald Rapids and
// Zen 5), else on Lanes. -DFPS_ROOTS_FOLDS=<struct> forces one, e.g. -DFPS_ROOTS_FOLDS=
// fps::simd::Avx512Vl. Measurements: notes.md.
#pragma once

#include "backend/product_tree/select.hpp"
#include "backend/simd/select.hpp"

namespace fps::root_finding {

using Steps = product_tree::Lanes;

#if defined(FPS_ROOTS_FOLDS)
using Folds = FPS_ROOTS_FOLDS;
#elif defined(__AVX512F__) && defined(__AVX512BW__)
using Folds = simd::Avx512;
#else
using Folds = product_tree::Lanes;
#endif

// The Graeffe step on x86 through a buffer of columns, a pass per lane parity
// (kernels.hpp, even_parts_columns), or with all operands in registers (even_parts): with the
// columns the step's pair pass took 0.65 of the time on Zen 5, 1.07 on Emerald Rapids and 1.01 on
// Zen 3, where whole runs were 2-5% slower at 2^6-2^8 (notes/122-r1.md). -DFPS_ROOTS_PAIR_COLUMNS=0
// or 1 forces it.
#if defined(FPS_ROOTS_PAIR_COLUMNS)
inline constexpr bool kPairColumns = FPS_ROOTS_PAIR_COLUMNS;
#elif (defined(__znver4__) || defined(__znver5__)) && defined(__AVX512VL__)
inline constexpr bool kPairColumns = true;
#else
inline constexpr bool kPairColumns = false;
#endif

}  // namespace fps::root_finding
