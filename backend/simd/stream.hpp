// Bench helper for element-wise vector operations: out[i] = op(a[i], b[i]) over arrays of n words.
// a and b are random below `bound`; the three arrays are 64-byte aligned (no load splits a cache
// line) and the loop reads them through local pointers (no reloads after the aliasing stores).
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "backend/bench.hpp"

namespace fps::simd {

template <class S, class Op>
void add_stream(bench::Suite& suite, const std::string& name, std::size_t n, std::uint32_t bound, Op op) {
    suite.add(name + "/" + std::to_string(n), [n, bound, op] {
        std::vector<std::uint32_t> storage(3 * n + 16);
        const auto misaligned = reinterpret_cast<std::uintptr_t>(storage.data()) % 64 / 4;
        std::uint32_t* const a = storage.data() + (16 - misaligned) % 16;
        std::mt19937 rng(n);
        for (std::size_t i = 0; i < 2 * n; ++i) a[i] = std::uint32_t(rng() % bound);
        return [storage = std::move(storage), a, n, op] {
            const std::uint32_t* const b = a + n;
            std::uint32_t* const out = a + 2 * n;
            for (std::size_t i = 0; i < n; i += S::lanes) S::store(out + i, op(S::load(a + i), S::load(b + i)));
            bench::keep(out[0]);
        };
    }, n);
}

}  // namespace fps::simd
