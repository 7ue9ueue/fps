// Scalar model of the factorials kernel: the frontend's loop. Two chains of n - 1 dependent
// products and one exponentiation; any prime P < 2^32, n <= P. The kernel runs it below its
// threshold (select.hpp), and the tests check every variant against it.
#pragma once

#include <cstddef>
#include <cstdint>

namespace fps::factorials::model {

template <std::uint32_t P>
constexpr std::uint32_t multiply(std::uint32_t a, std::uint32_t b) {
    return std::uint32_t(std::uint64_t(a) * b % P);
}

// a^(P - 2) = 1 / a for a in [1, P).
template <std::uint32_t P>
constexpr std::uint32_t inverse(std::uint32_t a) {
    std::uint32_t r = 1 % P;
    for (std::uint32_t e = P - 2; e; e >>= 1, a = multiply<P>(a, a))
        if (e & 1) r = multiply<P>(r, a);
    return r;
}

// fact[i] = i! and inverse[i] = 1 / i! for i < n.
template <std::uint32_t P>
void tables(std::uint32_t* fact, std::uint32_t* inverse, std::size_t n) {
    if (n == 0) return;
    fact[0] = 1 % P;
    for (std::size_t i = 1; i < n; ++i) fact[i] = multiply<P>(fact[i - 1], std::uint32_t(i));
    inverse[n - 1] = model::inverse<P>(fact[n - 1]);
    for (std::size_t i = n - 1; i > 0; --i) inverse[i - 1] = multiply<P>(inverse[i], std::uint32_t(i));
}

}  // namespace fps::factorials::model
