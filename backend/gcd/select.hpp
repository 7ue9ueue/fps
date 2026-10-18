// The gcd kernel's choices per target: Euclid's cutoffs and the form of its passes.
// -DFPS_GCD_DIRECT=<k> forces kDirect. Measurements behind the choices: notes.md.
#pragma once

#include <cstddef>

#include "backend/tune/tune.hpp"

namespace fps::half_gcd::detail {

// Jumps of at most kDirect degrees run Euclid's algorithm. Jumps come in powers of two, so 64 and
// 96 run the same leaves (as 32 and 48). inverse_mod's time against the cutoff, 2^6 to 2^20
// (notes/157-r1.md): 64 on Zen 3, Emerald Rapids and Zen 5 (128 within 1% there); 32 on Neoverse V2
// from 2^8 (1-4% faster than 64). The same with #157 r2's faster steps (128: 1-11% slower on x86).
#if defined(FPS_GCD_DIRECT)
inline constexpr std::ptrdiff_t kDirect = FPS_GCD_DIRECT;
#else
inline constexpr std::ptrdiff_t kDirect = tune::kTarget == tune::Target::arm ? 32 : 64;
#endif

// Jumps of k on fewer than 2k + 1 words (inverse_mod's top: the remainders shrink from the start)
// and the gcd's tail run Euclid up to this many degrees: at 2^6 a cutoff of 32 ran 1.3x slower on
// Neoverse V2.
inline constexpr std::ptrdiff_t kDirectWhole = 64;

// Euclid's passes leave the rows' words (kLazyRows) and the remainders' (kLazyRemainders) below
// 3.31 P, not reduced (gcd.hpp; targets with 64-bit lane products), and take v[j - 1] as a lane
// shift of aligned vectors (kAlignedShifts for the remainders, kAlignedRows for the rows) instead of
// an unaligned load. -DFPS_GCD_LAZY_ROWS=0/1, -DFPS_GCD_LAZY_REMAINDERS=0/1, -DFPS_GCD_ALIGNED=0/1,
// -DFPS_GCD_ALIGNED_ROWS=0/1 force them. notes/157-r2.md: both lazy on every machine; the shifts on
// Emerald Rapids (gcd and inverse_mod 0.96 of the time with unaligned loads) and Zen 5, not on Zen
// 3 or Neoverse V2 (there 1.02-1.23x and 1.004-1.03x, 2^6 to 2^20).
#if defined(FPS_GCD_LAZY_ROWS)
inline constexpr bool kLazyRows = FPS_GCD_LAZY_ROWS;
#else
inline constexpr bool kLazyRows = true;
#endif
#if defined(FPS_GCD_LAZY_REMAINDERS)
inline constexpr bool kLazyRemainders = FPS_GCD_LAZY_REMAINDERS;
#else
inline constexpr bool kLazyRemainders = true;
#endif
#if defined(FPS_GCD_ALIGNED)
inline constexpr bool kAlignedShifts = FPS_GCD_ALIGNED;
#else
inline constexpr bool kAlignedShifts = tune::kTarget == tune::Target::zen5 || tune::kTarget == tune::Target::avx512;
#endif
#if defined(FPS_GCD_ALIGNED_ROWS)
inline constexpr bool kAlignedRows = FPS_GCD_ALIGNED_ROWS;
#else
inline constexpr bool kAlignedRows = kAlignedShifts;
#endif

}  // namespace fps::half_gcd::detail
