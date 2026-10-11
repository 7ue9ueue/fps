// The NTT variant per op and target. -DFPS_NTT_FORWARD=<variant> (and _INVERSE, _MULTIPLY)
// forces one. Variants: scalar (every target), avx2 (__AVX2__).
#pragma once

#include "backend/ntt/scalar.hpp"
#ifdef __AVX2__
#include "backend/ntt/avx2.hpp"
#endif

namespace fps::ntt {

// The fastest variant this target has; it also builds the tables of ntt::Transform.
#ifdef __AVX2__
using Native = variant::avx2;
#else
using Native = variant::scalar;
#endif

#ifdef FPS_NTT_FORWARD
using SelectedForward = variant::FPS_NTT_FORWARD;
#else
using SelectedForward = Native;
#endif

#ifdef FPS_NTT_INVERSE
using SelectedInverse = variant::FPS_NTT_INVERSE;
#else
using SelectedInverse = Native;
#endif

#ifdef FPS_NTT_MULTIPLY
using SelectedMultiply = variant::FPS_NTT_MULTIPLY;
#else
using SelectedMultiply = Native;
#endif

}  // namespace fps::ntt
