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

}  // namespace fps::root_finding
