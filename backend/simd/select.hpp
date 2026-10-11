// The ISA struct kernels use on this target: simd::Native. -DFPS_SIMD_ISA=<struct> forces one,
// e.g. -DFPS_SIMD_ISA=Avx512Vl. Measurements behind the choice: notes.md.
#pragma once

#include "backend/simd/simd.hpp"

namespace fps::simd {

#if defined(FPS_SIMD_ISA)
using Native = FPS_SIMD_ISA;
#elif defined(__AVX512F__) && defined(__AVX512BW__)
using Native = Avx512;
#elif defined(__AVX2__)
using Native = Avx2;
#elif defined(__SSE4_1__)
using Native = Sse;
#elif defined(__ARM_NEON) && defined(__aarch64__)
using Native = Neon;
#else
using Native = Scalar;
#endif

// Every struct this target can run, for tests and benches:
// for_each(All{}, []<class S> { ... }).
template <class... S>
struct List {};
using All = List<Scalar
#if defined(__SSE4_1__)
                 , Sse
#endif
#if defined(__AVX2__)
                 , Avx2
#endif
#if defined(__AVX2__) && defined(__AVX512VL__)
                 , Avx512Vl
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__)
                 , Avx512
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
                 , Neon
#endif
                 >;

template <class... S, class F>
void for_each(List<S...>, F f) {
    (f.template operator()<S>(), ...);
}

}  // namespace fps::simd
