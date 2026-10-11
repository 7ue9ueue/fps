// Scalar model of the small kernel: the O(n m) product, one reduction per term. Any P < 2^32.
// The tests check every variant against it.
#pragma once

#include <cstddef>
#include <cstdint>

namespace fps::small::model {

// out[k] = sum over i + j = k of a[i] b[j] mod P, for k < n + m - 1; nothing if n or m is 0.
template <std::uint32_t P>
void multiply(const std::uint32_t* a, std::size_t n, const std::uint32_t* b, std::size_t m, std::uint32_t* out) {
    if (n == 0 || m == 0) return;
    for (std::size_t k = 0; k < n + m - 1; ++k) out[k] = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j) out[i + j] = std::uint32_t((out[i + j] + std::uint64_t(a[i]) * b[j]) % P);
}

}  // namespace fps::small::model
