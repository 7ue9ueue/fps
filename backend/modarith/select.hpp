// The variant of each modarith operation per target. -DFPS_MODARITH_MONTGOMERY=<variant> and
// -DFPS_MODARITH_SHOUP=<variant> force one. If a struct lacks the target's variant (e.g. Scalar
// on an x86 target), it runs mulhi. Measurements behind the choices: notes.md.
#pragma once

namespace fps::modarith {

enum class Montgomery { wide, mulhi, doubling };
enum class Shoup { mulhi, wide };

#if defined(FPS_MODARITH_MONTGOMERY)
inline constexpr Montgomery kMontgomery = Montgomery::FPS_MODARITH_MONTGOMERY;
#elif defined(__ARM_NEON) && defined(__aarch64__)
inline constexpr Montgomery kMontgomery = Montgomery::doubling;  // Neoverse V2: 17% faster than wide
#else
inline constexpr Montgomery kMontgomery = Montgomery::wide;  // x86: 1.13-1.37x faster than mulhi
#endif

#if defined(FPS_MODARITH_SHOUP)
inline constexpr Shoup kShoup = Shoup::FPS_MODARITH_SHOUP;
#else
inline constexpr Shoup kShoup = Shoup::mulhi;
#endif

template <class S>
constexpr bool has(Montgomery variant) {
    using V = typename S::V;
    if (variant == Montgomery::wide)
        return requires(V a) { S::mul_even(a, a); } || requires(V a) { S::mul_wide_lo(a, a); };
    if (variant == Montgomery::doubling) return requires(V a) { S::qdmulh(a, a); };
    return true;
}

template <class S>
constexpr bool has(Shoup variant) {
    using V = typename S::V;
    if (variant == Shoup::wide) return requires(V a) { S::mul_even(a, a); };
    return true;
}

template <class S>
constexpr Montgomery selected_montgomery() { return has<S>(kMontgomery) ? kMontgomery : Montgomery::mulhi; }

template <class S>
constexpr Shoup selected_shoup() { return has<S>(kShoup) ? kShoup : Shoup::mulhi; }

}  // namespace fps::modarith
