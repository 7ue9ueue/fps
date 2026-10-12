// Shared by every variant of the NTT: the modulus, its roots of unity and the twiddle tables.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace fps::ntt {

inline constexpr std::uint32_t kModulus = 998244353;  // 119 * 2^23 + 1
inline constexpr int kMinLog = 6;                     // transform lengths 2^6 .. 2^25
inline constexpr int kMaxLog = 25;
inline constexpr std::size_t kPadding = 16;  // words after each array: the AVX2 kernels read 4 bytes past
inline constexpr int kValuesMaxLog = 23;     // the values form: 2^lg divides P - 1 = 119 2^23

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

// The values form (ntt.hpp): leaf v = 8g + t, in block g of 64 words, taken on to its values at
// rho_v z^j (j < 8), with rho_v = r(4v), an 8th root of w_v, and z = r(2), of order 8; word
// 64g + 8i + t holds the value at j = bitrev3(i). Per leaf: the twist c_k -> c_k rho_v^k, then an
// 8-point transform with twiddles z^k = r(eighth_root(k)). rho_v = A(g / 8) B(g % 8, t) with
// A(h) = r(256 h) and B(l, t) = r(32 l + 4 t) (disjoint bits): the B powers are a table of 8 rows
// per l, the A powers a table per transform (value_rows).

// The table index of z^k, k < 4: z = r(2), z^2 = r(1) = sqrt(-1), z^3 = r(3).
constexpr std::size_t eighth_root(int k) { return std::size_t((k & 1) << 1 | (k >> 1 & 1)); }

// Rows of 16 words: 8 values w, then their Shoup quotients. forward[l][k]: B(l, t)^k for lanes
// t < 8 (k = 0 unused); inverse[l][k]: B(l, t)^-k / 8 (1/8: the 8-point transform's gain).
struct alignas(64) ValueLanes {
    std::uint32_t forward[8][8][16];
    std::uint32_t inverse[8][8][16];
};
inline constexpr ValueLanes kValueLanes = [] {
    ValueLanes v{};
    const std::uint32_t eighth = power(8, kP - 2);
    for (int l = 0; l < 8; ++l) {
        for (int t = 0; t < 8; ++t) {
            std::uint32_t b = 1, bi = 1;  // B(l, t) and its inverse: bits j of 32 l + 4 t
            for (int j = 0; j < 8; ++j)
                if ((32 * l + 4 * t) >> j & 1) b = multiply_mod(b, kRoots[0][j]), bi = multiply_mod(bi, kRoots[1][j]);
            std::uint32_t x = 1, y = eighth;
            for (int k = 0; k < 8; ++k, x = multiply_mod(x, b), y = multiply_mod(y, bi)) {
                v.forward[l][k][t] = x, v.forward[l][k][8 + t] = quotient(x);
                v.inverse[l][k][t] = y, v.inverse[l][k][8 + t] = quotient(y);
            }
        }
    }
    return v;
}();

// Rows of A(h) per h < value_row_count(lg), 32 words each: A(h)^k for k < 8, their quotients,
// A(h)^-k, their quotients.
constexpr std::size_t value_row_count(int lg) { return lg > 9 ? (std::size_t(1) << lg) / 512 : 1; }
inline void build_value_rows(std::uint32_t* rows, int lg) {
    std::uint32_t root[26], inverse_root[26];  // orders 2^(j + 2): bit j of 256 h is bit j - 8 of h
    for (int j = 0; j < 26; ++j) root[j] = j < 22 ? power(3, (kP - 1) >> (j + 2)) : 1, inverse_root[j] = power(root[j], kP - 2);
    std::uint32_t a = 1, ai = 1;
    for (std::size_t h = 0; h < value_row_count(lg); ++h) {
        if (h) {
            const std::uint32_t* prior = rows + 32 * (h & (h - 1));  // A(h) = A(h without its low bit) r(256 lowbit)
            const int j = std::countr_zero(h) + 8;
            a = multiply_mod(prior[1], root[j]), ai = multiply_mod(prior[17], inverse_root[j]);
        }
        std::uint32_t* row = rows + 32 * h;
        std::uint32_t x = 1, y = 1;
        for (int k = 0; k < 8; ++k, x = multiply_mod(x, a), y = multiply_mod(y, ai)) {
            row[k] = x, row[8 + k] = quotient(x);
            row[16 + k] = y, row[24 + k] = quotient(y);
        }
    }
}

// A product from spans for the variants (ntt::multiply): words [0, count) of f g into out. f: n
// words, g: m words, values < P, in caller memory of any alignment, read only there; n, m >= 1,
// n + m - 1 <= 2^lg, 1 <= count <= 2^lg. out: count words of caller memory, any alignment, written
// only there; it may overlap f or g, as the variants read f and g before they write out. a, b: the
// variants' scratch, 2^lg + kPadding words each, 64-byte aligned. g may be b itself, with m = 2^lg
// (read in place: ntt::middle_product). The variants' square(p) ignores g, m and b: f^2.
struct Product {
    const std::uint32_t* f;
    std::size_t n;
    const std::uint32_t* g;
    std::size_t m;
    std::uint32_t* a;
    std::uint32_t* b;
    std::uint32_t* out;
    std::size_t count;
    int lg;
};

// f(std::integral_constant<int, t>{}) for t in [0, K]: a block count known at run time as a
// template argument.
template <int K, class F>
[[gnu::always_inline]] inline void with_blocks(int t, F f) {
    [&]<int... I> [[gnu::always_inline]] (std::integer_sequence<int, I...>) {
        (void)((t == I && (f(std::integral_constant<int, I>{}), true)) || ...);
    }(std::make_integer_sequence<int, K + 1>{});
}

// The first and last levels of a product move its factors in from the caller and the result out,
// so nothing is copied whole. Their columns span K blocks of `block` words; `step` words of columns
// per call; block and step powers of 2, step <= block. run(std::integral_constant<int, T>{}, from,
// to, words) runs `words` words of columns of the level from `from` to `to` (both at the first
// column), with T blocks read or written; the levels pick T with an inlined switch, so run should
// call one function per T. Without a boundary inside a block (words a multiple of block) that is
// one call, with no division (notes.md, Design: one function for every T and two divisions cost a
// 2^7 product 72 cycles on Emerald Rapids).

// run with the block count t as a template argument, on columns [begin, end).
template <int K, class Run>
[[gnu::always_inline]] inline void run_blocks(int t, Run& run, const std::uint32_t* from, std::uint32_t* to,
                                              std::size_t begin, std::size_t end) {
    with_blocks<K>(t, [&] [[gnu::always_inline]] (auto blocks) {
        run(blocks, from + begin, to + begin, end - begin);
    });
}

// First level, from a source of n words (zero past them): run reads blocks t < T of `from` (the
// rest are zero, not read) and writes all K to `to`. With F whole blocks and c words of block F:
// columns [0, c0) read F + 1 blocks, [c1, block) F, where c0 and c1 are c rounded down and up to
// step. Between them, the source's last words are staged in `to`, zero past n, and run in place.
// from and to: equal, or apart.
template <int K, class Run>
[[gnu::noinline]] void first_level_boundary(const std::uint32_t* from, std::size_t n, std::uint32_t* to,
                                            std::size_t block, std::size_t step, Run run) {
    const std::size_t whole = n / block, c = n % block, c0 = c / step * step, c1 = c0 == c ? c : c0 + step;
    if (c0) run_blocks<K>(int(whole) + 1, run, from, to, 0, c0);
    if (c1 > c0) {
        for (std::size_t t = 0; t <= whole; ++t) {
            const std::size_t words = t < whole ? step : c - c0;
            std::uint32_t* column = to + t * block + c0;
            if (from != to) std::copy_n(from + t * block + c0, words, column);
            std::fill(column + words, column + step, 0u);
        }
        run_blocks<K>(int(whole) + 1, run, to, to, c0, c1);
    }
    if (c1 < block) run_blocks<K>(int(whole), run, from, to, c1, block);
}

template <int K, class Run>
[[gnu::always_inline]] inline void first_level(const std::uint32_t* from, std::size_t n, std::uint32_t* to,
                                               std::size_t block, std::size_t step, Run run) {
    if (n & (block - 1)) return first_level_boundary<K>(from, n, to, block, step, run);
    run_blocks<K>(int(n >> std::countr_zero(block)), run, from, to, 0, block);
}

// Last level, writing count words: run writes blocks t < T of its output to `to`. Columns [0, c0)
// write F + 1 blocks and [c0, block) F, for count = F block + c. Then run writes all K blocks of
// columns [c0, c1) in place in `from`, and block F's words below count go to out.
template <int K, class Run>
[[gnu::noinline]] void last_level_boundary(std::uint32_t* from, std::size_t block, std::uint32_t* out,
                                           std::size_t count, std::size_t step, Run run) {
    const std::size_t whole = count / block, c = count % block, c0 = c / step * step, c1 = c0 == c ? c : c0 + step;
    if (c0) run_blocks<K>(int(whole) + 1, run, from, out, 0, c0);
    if (whole) run_blocks<K>(int(whole), run, from, out, c0, block);
    if (c1 > c0) {
        run(std::integral_constant<int, K>{}, from + c0, from + c0, c1 - c0);
        std::copy_n(from + whole * block + c0, c - c0, out + whole * block + c0);
    }
}

template <int K, class Run>
[[gnu::always_inline]] inline void last_level(std::uint32_t* from, std::size_t block, std::uint32_t* out,
                                              std::size_t count, std::size_t step, Run run) {
    if (count & (block - 1)) return last_level_boundary<K>(from, block, out, count, step, run);
    run_blocks<K>(int(count >> std::countr_zero(block)), run, from, out, 0, block);
}

}  // namespace detail
}  // namespace fps::ntt
