// The DFT of power-of-two length in natural order, and its inverse, modulo P = 998244353 (issue
// #138), on the NTT's own transform.
//
//   fps::ntt::dft(f, out);        // out[i] = f(w^i) for i < L = out.size(), any f.size()
//   fps::ntt::idft(values, out);  // out = the f of L = values.size() words with f(w^i) = values[i]
//
// w = 3^((P - 1) / L), a primitive L-th root of unity; L = 2^lg, kMinLog <= lg <= kValuesMaxLog
// (2^6 .. 2^23: past it P - 1 = 119 2^23 has no roots of order L). L not a power of two:
// std::invalid_argument; out of that range: std::length_error; idft with out.size() !=
// values.size(): std::invalid_argument. All before anything is written. Words < P in, canonical
// out; spans of any alignment; out may overlap f or values (they are read before out is written).
//
// dft: f mod (x^L - 1) to the values form (Transform::forward(f, Form::values): the forward
// transform reads f from its first level, and each subtree's leaves go on to their values while in
// cache), then one pass of the values form's order into natural order (permute, below). idft: the
// pass back, then the values form's inverse, whose scale 2^-lg is the 1 / L, its top level writing
// out. The transform works in this thread's region (ntt.hpp); its tables are the products' (the
// compile-time ones to 2^16, then this thread's), the values form's rows are built once per length
// (L / 16 words each), and the permutation's buffers take 40 KiB of stack.
//
// Cost (bench_dft.cpp, notes/138-r1.md), 2^10-2^22 on Zen 3, Emerald Rapids, Zen 5 and Neoverse V2:
// 1.4-2.2x the time of the frontend's to_values (a leaf-form forward of L / 2 coefficients; at 2^20
// 1.5-1.7x), of which the values form's forward alone is 1.2-1.6x; 4.6-7.8x faster than the chirp
// z-transform (backend/chirp: a middle product on transforms of 2L).
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>

#include "backend/factorials/transpose.hpp"
#include "backend/ntt/ntt.hpp"

namespace fps::ntt {

namespace detail::dft {

using u32 = std::uint32_t;

// rev6(x) for x < 64: its 6 bits reversed.
inline constexpr auto kReverse6 = [] {
    std::array<std::uint8_t, 64> r{};
    for (int x = 0; x < 64; ++x)
        for (int i = 0; i < 6; ++i) r[std::size_t(x)] |= std::uint8_t((x >> i & 1) << (5 - i));
    return r;
}();

// x's low `bits` bits reversed, bits <= 18.
constexpr std::size_t reverse(std::size_t x, int bits) {
    const std::size_t r = std::size_t(kReverse6[x & 63]) << 12 | std::size_t(kReverse6[x >> 6 & 63]) << 6 | kReverse6[x >> 12 & 63];
    return r >> (18 - bits);
}

// lg of a length L = 2^lg in [2^kMinLog, 2^kValuesMaxLog]; else the errors above.
inline int checked_log(std::size_t L, const char* what) {
    if (!std::has_single_bit(L)) throw std::invalid_argument(what);
    const int lg = std::countr_zero(L);
    if (lg < kMinLog || lg > kValuesMaxLog) throw std::length_error(what);
    return lg;
}

// The values form's rows (build_value_rows) of each length, built on first use and kept.
inline const u32* value_rows(int lg) {
    static std::array<std::once_flag, kValuesMaxLog + 1> once;
    static std::array<std::unique_ptr<u32[]>, kValuesMaxLog + 1> rows;
    std::call_once(once[std::size_t(lg)], [lg] {
        rows[std::size_t(lg)].reset(new u32[32 * value_row_count(lg)]);
        build_value_rows(rows[std::size_t(lg)].get(), lg);
    });
    return rows[std::size_t(lg)].get();
}

// to[i] = from[i] for i < count, on S (any count).
template <class S>
[[gnu::always_inline]] inline void copy_words(u32* to, const u32* from, std::size_t count) {
    std::size_t i = 0;
    for (; i + S::lanes <= count; i += S::lanes) S::store(to + i, S::load(from + i));
    for (; i < count; ++i) to[i] = from[i];
}

inline constexpr std::size_t kLine = 16;  // words per cache line
inline constexpr int kNever = 99;         // a permute buffer's lg for never

// The order. Word p = 64 g + 8 i + t of the values form is leaf v = 8 g + t at rho_v z^j, j =
// bitrev3(i) (common.hpp), and with r(k) = c^rev22(k) for c of order 2^23, rho_v = r(4 v) =
// w^rev_(lg-3)(v) and z = r(2) = w^(L/8): word p is f(w^e), e = rev_lg(64 g + 8 t + i)
// = rev6(8 t + i) 2^n + rev_n(g), n = lg - 6. So natural order is a bit reversal of the blocks g
// and, within each block, of the 64 words transposed: 64 runs of 2^n words, run k = 8 i + t at
// rev6(8 t + i) 2^n, word rev_n(g) of run k from word k of block g.
//
// permute (kInverse: its inverse) moves the words in tiles of R = 2^rho blocks: g = rev_rho(r)
// 2^(n - rho) + m for r < R, m = rev_(n - rho)(c) for tile c, whose words are words [R c, R c + R)
// of each run. Each W x W piece is W row loads (W words of W blocks), a transpose (simd's,
// backend/factorials), and W stores (W words of W runs); the h loop runs the pieces of one line of
// W blocks back to back. R = 16 blocks below 2^kWideLog, 2^kRho from it. From 2^kValuesLog the
// tile's blocks go through a buffer, copied whole (in, or kInverse out). From 2^kNaturalLog, when
// the natural side's lines are not aligned to the runs' chunks, the runs go through a buffer too,
// so their lines are written (read) whole and in order: the chunk of a run, R words, goes out
// shifted to the line boundary, its last a words (a: the side's words past a line) kept for the
// next tile. Without the buffers, partial lines and rows of a block 2^lg / R words apart ran up to
// 3.6x slower (Zen 3, 2^18; 1.4-1.8x on Zen 5 for a natural side 16 bytes past a line; notes/138-
// r1.md). Lengths with R < W run it on Scalar.
template <class S, bool kInverse, int kRho = 6, int kWideLog = 14, int kValuesLog = 14, int kNaturalLog = 14>
void permute(const u32* from, u32* to, int lg) {
    using V = typename S::V;
    constexpr int W = S::lanes;
    const int n = lg - 6, rho = std::min(n, lg < kWideLog ? 4 : kRho);
    if constexpr (W > 1) {
        if ((1 << rho) < W) return permute<simd::Scalar, kInverse, kRho, kWideLog, kNever, kNever>(from, to, lg);
    }
    constexpr std::size_t kMaxRows = std::size_t(1) << std::max(kRho, 4);
    const std::size_t R = std::size_t(1) << rho, L = std::size_t(1) << lg, tiles = std::size_t(1) << (n - rho);
    std::size_t run[64], row[kMaxRows];  // offsets of the runs (natural side) and of the tile's blocks
    for (std::size_t k = 0; k < 64; ++k) run[k] = std::size_t(kReverse6[8 * (k % 8) + k / 8]) << n;
    for (std::size_t r = 0; r < R; ++r) row[r] = reverse(r, rho) * (L / R);
    const u32* natural = kInverse ? from : to;
    const std::size_t a = reinterpret_cast<std::uintptr_t>(natural) / sizeof(u32) % kLine, s = (kLine - a) % kLine;
    constexpr bool kBlocks = kValuesLog != kNever, kRuns = kNaturalLog != kNever;
    const bool values_staged = kBlocks && lg >= kValuesLog, staged = kRuns && lg >= kNaturalLog && R >= kLine && a != 0;
    constexpr std::size_t kStride = kMaxRows + 2 * kLine;
    alignas(64) u32 blocks[kBlocks ? 64 * kMaxRows : 1];  // block r of the tile at blocks + 64 r
    alignas(64) u32 runs[kRuns ? 64 * kStride : 1];       // word j of run k's chunk at runs + k kStride + kLine + j
    for (std::size_t c = 0; c < tiles; ++c) {
        const std::size_t block = 64 * reverse(c, n - rho), at = R * c;
        if (kInverse && staged) {  // words [0, R + s) of each chunk: [0, s) kept, [s, R + s) whole lines
            for (std::size_t k = 0; k < 64; ++k) {
                u32* q = runs + k * kStride + kLine;
                const u32* p = from + run[k] + at;
                if (c == 0) copy_words<simd::Scalar>(q, p, s);
                else copy_words<S>(q, q + R, kLine);
                copy_words<S>(q + s, p + s, c + 1 < tiles ? R : R - s);
            }
        }
        if (!kInverse && values_staged)
            for (std::size_t r = 0; r < R; ++r) copy_words<S>(blocks + 64 * r, from + block + row[r], 64);
        for (std::size_t u = 0; u < 64; u += kLine) {
            for (std::size_t b = 0; b < R; b += W) {
                #pragma GCC unroll 16
                for (std::size_t h = 0; h < kLine; h += W) {
                    V x[W];
                    #pragma GCC unroll 16
                    for (int j = 0; j < W; ++j) {
                        const std::size_t k = u + h + j, r = b + j;
                        if constexpr (kInverse) x[j] = S::load(staged ? runs + k * kStride + kLine + b : from + run[k] + at + b);
                        else x[j] = S::load(values_staged ? blocks + 64 * r + u + h : from + block + row[r] + u + h);
                    }
                    factorials::transpose(x);
                    #pragma GCC unroll 16
                    for (int j = 0; j < W; ++j) {
                        const std::size_t k = u + h + j, r = b + j;
                        if constexpr (kInverse) S::store(values_staged ? blocks + 64 * r + u + h : to + block + row[r] + u + h, x[j]);
                        else S::store(staged ? runs + k * kStride + kLine + b : to + run[k] + at + b, x[j]);
                    }
                }
            }
        }
        if (kInverse && values_staged)
            for (std::size_t r = 0; r < R; ++r) copy_words<S>(to + block + row[r], blocks + 64 * r, 64);
        if (!kInverse && staged) {  // words [-a, R - a): the last tile's [R - a, R), then this one's
            for (std::size_t k = 0; k < 64; ++k) {
                u32* q = runs + k * kStride + kLine;
                u32* p = to + run[k] + at;
                if (c == 0) copy_words<simd::Scalar>(p, q, R - a);
                else copy_words<S>(p - a, q - a, R);
                copy_words<S>(q - kLine, q + R - kLine, kLine);
                if (c + 1 == tiles) copy_words<simd::Scalar>(p + R - a, q - a, a);
            }
        }
    }
}

// A permutation's configuration (permute's): the default, or another for bench_dft.cpp.
template <int kRho = 6, int kWideLog = 14, int kValuesLog = 14, int kNaturalLog = 14>
struct Order {
    template <bool kInverse>
    static void run(const u32* from, u32* to, int lg) {
        permute<simd::Native, kInverse, kRho, kWideLog, kValuesLog, kNaturalLog>(from, to, lg);
    }
};

// f mod (x^L - 1) to w (L words), f longer than L.
inline void fold(std::span<const u32> f, std::size_t L, u32* w) {
    ragged::fold(f.data(), std::min(f.size(), 2 * L), L, w);
    for (std::size_t at = 2 * L; at < f.size(); at += L) add_words(w, f.data() + at, std::min(L, f.size() - at));
}

// w (L + kPadding words, 64-byte aligned, apart from f) = the values form of f mod (x^L - 1): for
// f.size() <= L Variant's forward_hooked, which reads f from its first level, with the values
// form's split as its hook; else f folded into w and the values form in place. (f copied into w,
// then the values form in place, ran 0.98-1.04 of its time.)
template <class Variant>
void forward(std::span<const u32> f, u32* w, int lg, const u32* roots, const u32* rows) {
    const std::size_t L = std::size_t(1) << lg, n = f.size();
    if (n > L) {
        fold(f, L, w);
        return Variant::forward_values(w, lg, roots, rows);
    }
    Variant::forward_hooked(n ? f.data() : w, n, w, lg, roots, 0, [rows](u32* t, std::size_t first, std::size_t count) {
        intrinsics::ValuesFor<simd::Native>::split(t, t, count / 8, first / 8, rows);
    });
}

// The work array of a transform of 2^lg words in the region, far from x's and y's page offsets.
inline u32* work(const Region& region, const void* x, const void* y, int lg) {
    return reinterpret_cast<u32*>(region.start + place(x, x, y, 0, place_period(lg)));
}

inline std::size_t region_bytes(int lg) { return sizeof(u32) * ((std::size_t(1) << lg) + kPadding) + 4096; }

template <class Variant, class O = Order<>>
void run_dft(std::span<const u32> f, std::span<u32> out) {
    const int lg = checked_log(out.size(), "fps::ntt::dft: out.size() not a power of two in [2^6, 2^23]");
    const u32* roots = tables<Variant>(lg).first;
    const u32* rows = value_rows(lg);
    const Region region = take_region(region_bytes(lg));
    u32* w = work(region, f.data(), out.data(), lg);
    forward<Variant>(f, w, lg, roots, rows);
    O::template run<false>(w, out.data(), lg);
    give_region(region);
}

template <class Variant, class O = Order<>>
void run_idft(std::span<const u32> values, std::span<u32> out) {
    if (out.size() != values.size()) throw std::invalid_argument("fps::ntt::idft: out.size() != values.size()");
    const int lg = checked_log(values.size(), "fps::ntt::idft: values.size() not a power of two in [2^6, 2^23]");
    const u32* inverse_roots = tables<Variant>(lg).second;
    const u32* rows = value_rows(lg);
    const Region region = take_region(region_bytes(lg));
    u32* w = work(region, values.data(), out.data(), lg);
    O::template run<true>(values.data(), w, lg);
    Variant::inverse(w, w, out.data(), out.size(), lg, inverse_roots, rows);
    give_region(region);
}

}  // namespace detail::dft

// out[i] = f(w^i), i < L = out.size() (above).
template <class Variant = SelectedForward>
void dft(std::span<const std::uint32_t> f, std::span<std::uint32_t> out) {
    detail::dft::run_dft<Variant>(f, out);
}

// out = the f of L = values.size() coefficients with f(w^i) = values[i] (above).
template <class Variant = SelectedInverse>
void idft(std::span<const std::uint32_t> values, std::span<std::uint32_t> out) {
    detail::dft::run_idft<Variant>(values, out);
}

}  // namespace fps::ntt
