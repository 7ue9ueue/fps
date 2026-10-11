// Scalars modulo a prime P < 2^32. Values are canonical: in [0, P), in and out.
//
//   fps::mod_mul<P>(a, b)      a b
//   fps::mod_pow<P>(a, e)      a^e for any e < 2^64; 0^0 = 1
//   fps::mod_inverse<P>(a)     1 / a; throws std::domain_error for a = 0
#pragma once

#include <cstdint>
#include <stdexcept>

namespace fps {

template <std::uint32_t P>
constexpr std::uint32_t mod_mul(std::uint32_t a, std::uint32_t b) {
    return std::uint32_t(std::uint64_t(a) * b % P);
}

template <std::uint32_t P>
constexpr std::uint32_t mod_pow(std::uint32_t a, std::uint64_t e) {
    std::uint32_t r = 1 % P;
    for (; e; e >>= 1, a = mod_mul<P>(a, a))
        if (e & 1) r = mod_mul<P>(r, a);
    return r;
}

template <std::uint32_t P>
constexpr std::uint32_t mod_inverse(std::uint32_t a) {
    if (a == 0) throw std::domain_error("fps::mod_inverse: 0 has no inverse");
    return mod_pow<P>(a, P - 2);
}

}  // namespace fps
