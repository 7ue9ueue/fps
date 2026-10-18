// The product tree's choices per target: the struct for 8-word vectors (one coefficient of 8
// lanes), the struct for passes over whole arrays, and the trees' cutoffs. Measurements: notes.md.
#pragma once

#include <cstddef>
#include <cstdint>

#include "backend/simd/select.hpp"

namespace fps::product_tree {

// 8 words of 8 lanes: one ymm with AVX2 (32 registers with AVX-512VL), two vectors of 4 words on
// Sse and Neon, 8 words on Scalar.
#if defined(__AVX2__) && defined(__AVX512VL__)
using Lanes = simd::Avx512Vl;
#elif defined(__AVX2__)
using Lanes = simd::Avx2;
#elif defined(__SSE4_1__)
using Lanes = simd::Sse;
#elif defined(__ARM_NEON) && defined(__aarch64__)
using Lanes = simd::Neon;
#else
using Lanes = simd::Scalar;
#endif

// Pointwise products and conversions: the target's widest struct.
using Wide = simd::Native;

// Slots of 8 linear factors loaded in pairs and transposed in registers (load_linear): 1-5%
// faster on Emerald Rapids, Zen 5 and Neoverse V2; on Zen 3 the generic path was 1-2% faster
// (notes.md, issue #91). Sse and Scalar run the generic path either way.
#if defined(__AVX2__) && !defined(__AVX512VL__)
inline constexpr bool kLinearSlots = false;
#else
inline constexpr bool kLinearSlots = true;
#endif

// Leaf products two leaves per zmm (leaf_products_k): on Zen 5 0.80-0.83 of the time of one leaf
// per ymm, two products of one a 0.63-0.65 per product; on Emerald Rapids 1.16-1.24
// (notes/136-r3.md). Zen 4 as Zen 5 (a guess, as ntt's WideBottom).
#if defined(__AVX512F__) && defined(__AVX512BW__) && (defined(__znver4__) || defined(__znver5__))
inline constexpr bool kWideLeaves = true;
#else
inline constexpr bool kWideLeaves = false;
#endif

// Nodes of the lane tree up to this degree are multiplied out by schoolbook (at most 9 terms per
// sum: Montgomery's step needs no fold); the standard tree's by small::multiply.
inline constexpr std::uint32_t kLaneBase = 16;
inline constexpr std::uint32_t kStandardBase = 32;

// The standard tree's nodes over two lane products come from the lane root's transform (lane_pairs)
// when the lanes have at least this length (the nodes' length is twice it, >= 64).
inline constexpr std::size_t kPairsLength = 32;

// Lanes run when at least this many factors are small (degree at most 1/64 of the sum of all
// degrees): from 16 linear factors on Zen 3 (notes.md).
inline constexpr std::size_t kMinLaneFactors = 16;

}  // namespace fps::product_tree
