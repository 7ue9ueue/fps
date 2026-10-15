// The gcd kernel's choices per target: Euclid's cutoffs. -DFPS_GCD_DIRECT=<k> forces kDirect.
// Measurements behind the choices: notes.md.
#pragma once

#include <cstddef>

#include "backend/tune/tune.hpp"

namespace fps::half_gcd::detail {

// Jumps of at most kDirect degrees run Euclid's algorithm. Jumps come in powers of two, so 64 and
// 96 run the same leaves (as 32 and 48). inverse_mod's time against the cutoff, 2^6 to 2^20
// (notes/157-r1.md): 64 on Zen 3, Emerald Rapids and Zen 5 (128 within 1% there); 32 on Neoverse V2
// from 2^8 (1-4% faster than 64).
#if defined(FPS_GCD_DIRECT)
inline constexpr std::ptrdiff_t kDirect = FPS_GCD_DIRECT;
#else
inline constexpr std::ptrdiff_t kDirect = tune::kTarget == tune::Target::arm ? 32 : 64;
#endif

// Jumps of k on fewer than 2k + 1 words (inverse_mod's top: the remainders shrink from the start)
// and the gcd's tail run Euclid up to this many degrees: at 2^6 a cutoff of 32 ran 1.3x slower on
// Neoverse V2.
inline constexpr std::ptrdiff_t kDirectWhole = 64;

}  // namespace fps::half_gcd::detail
