// Shared by every variant of the NTT: the modulus, its roots of unity and the twiddle tables.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace fps::ntt {

inline constexpr std::uint32_t kModulus = 998244353;  // 119 * 2^23 + 1
inline constexpr int kMinLog = 6;                     // transform lengths 2^6 .. 2^25
inline constexpr int kMaxLog = 25;
inline constexpr std::size_t kPadding = 16;  // words after each array: the AVX2 kernels read 4 bytes past

namespace detail {

inline constexpr std::uint32_t kP = kModulus;

constexpr std::uint32_t multiply_mod(std::uint32_t x, std::uint32_t y) {
    return std::uint32_t(std::uint64_t(x) * y % kP);
}

constexpr std::uint32_t power(std::uint32_t x, std::uint32_t e) {
    std::uint32_t r = 1;
    for (; e; e >>= 1, x = multiply_mod(x, x))
        if (e & 1) r = multiply_mod(r, x);
    return r;
}

// Shoup quotient floor(w 2^32 / P) of a constant w < P.
constexpr std::uint32_t quotient(std::uint32_t w) { return std::uint32_t((std::uint64_t(w) << 32) / kP); }

inline constexpr std::uint32_t kR = std::uint32_t((std::uint64_t(1) << 32) % kP);  // 2^32 mod P

// kRoots[0][j] has order 2^(j + 2); 3 generates the multiplicative group. kRoots[1]: inverses.
inline constexpr int kRootCount = kMaxLog - 4;
using Roots = std::array<std::uint32_t, kRootCount>;
inline constexpr auto kRoots = [] {
    std::array<Roots, 2> t{};
    for (int j = 0; j < kRootCount; ++j) {
        t[0][j] = power(3, (kP - 1) >> (j + 2));
        t[1][j] = power(t[0][j], kP - 2);
    }
    return t;
}();

// Twiddle tables: entry k, r(k), is the product of roots[j] over the set bits j of k, so r(1) is
// sqrt(-1), r(2k) is a square root of r(k) and r(2k + 1) = sqrt(-1) r(2k). Memory layout: blocks
// of 8 values, then their 8 Shoup quotients. A transform of length 2^lg uses entries < 2^lg / 16.
constexpr std::size_t slot(std::size_t k) { return (k >> 3 << 4) | (k & 7); }

constexpr std::size_t table_words(int lg) { return (std::size_t(1) << lg) / 8 + 16; }

// Entries [count / 2, count) of a table whose entries [0, count / 2) are built; count >= 2.
constexpr void extend_table(std::uint32_t* t, std::size_t count, const Roots& roots) {
    const std::size_t h = count / 2;
    const std::uint32_t root = roots[std::countr_zero(h)];
    for (std::size_t j = 0; j < h; ++j) {
        const std::uint32_t v = multiply_mod(t[slot(j)], root);
        t[slot(h + j)] = v;
        t[slot(h + j) + 8] = quotient(v);
    }
}

constexpr void start_table(std::uint32_t* t) {
    t[slot(0)] = 1;
    t[slot(0) + 8] = quotient(1);
}

// Both tables for lengths up to 2^kStaticLog, built at compile time (2 x 32 KiB). Shorter
// transforms read them in place; longer ones copy them and extend. Building them per call cost
// 75 cycles of a 493-cycle product at 2^6 on Zen 3 (notes.md).
inline constexpr int kStaticLog = 16;
inline constexpr std::size_t kStaticCount = (std::size_t(1) << kStaticLog) / 16;  // entries per table

struct alignas(64) StaticTables {
    std::uint32_t words[2][table_words(kStaticLog)];
};

inline constexpr StaticTables kStaticTables = [] {
    StaticTables s{};
    for (int inverse = 0; inverse < 2; ++inverse) {
        std::uint32_t* t = s.words[inverse];
        start_table(t);
        for (std::size_t count = 2; count <= kStaticCount; count *= 2) extend_table(t, count, kRoots[inverse]);
    }
    return s;
}();

// Entries [0, min(count, kStaticCount)) of a table, from kStaticTables.
inline std::size_t copy_static_table(std::uint32_t* t, std::size_t count, bool inverse) {
    const std::size_t n = count < kStaticCount ? count : kStaticCount;
    const std::uint32_t* s = kStaticTables.words[inverse];
    std::copy(s, s + std::max<std::size_t>(16, 2 * n), t);  // blocks of 8 values and 8 quotients
    return n;
}

// nv^-1 mod P for nv = 2^lg / 8 leaves, the gain of a forward and an inverse transform; and
// nv^-1 2^32 mod P, which also undoes the Montgomery factor 2^-32 of the AVX2 leaf products.
// Tables: a modular power costs about 100 ns per call.
inline constexpr auto kLeafScales = [] {
    std::array<std::array<std::uint32_t, kMaxLog + 1>, 2> t{};
    for (int lg = kMinLog; lg <= kMaxLog; ++lg) {
        t[0][lg] = power(std::uint32_t(1) << (lg - 3), kP - 2);
        t[1][lg] = multiply_mod(t[0][lg], kR);
    }
    return t;
}();
constexpr std::uint32_t leaf_scale(int lg) { return kLeafScales[0][lg]; }
constexpr std::uint32_t product_scale(int lg) { return kLeafScales[1][lg]; }

// Radix-4 levels run on groups of 4 quarters. nv = 4^j leaves start with one group; nv = 2 * 4^j
// start with a radix-2 level and two groups.
constexpr bool has_radix2(int lg) { return (lg - 3) % 2 == 1; }

}  // namespace detail
}  // namespace fps::ntt
