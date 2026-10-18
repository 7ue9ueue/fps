// Products of ragged length (issue #155): N = n + m - 1 words with L < N <= 2L, L = 2^lg, at about
// the cost of N words, not of the transform of 2L. ntt.hpp runs them (multiply, square, and
// middle_product as the transpose, below); this header has the plans and the passes around the
// power-of-two products. A second form, the truncated transform of 2L (truncated.hpp, round 2),
// serves multiply and square, and middle products (round 3); it is a plan kind here.
//
//   plan(N, kind), options(N, plans)   the plan of a kind; the distinct plans of N
//   choose_product(N)                  the tuned plan of multiply and square ("ntt.ragged")
//   choose_middle(N)                   the tuned plan of middle products of N words read (round 3)
//   choose(N)                          the tuned plan of ragged_into's callers outside ntt: never
//                                      truncated
//   truncates(N), rounded(size, n)     whether N has a truncated plan; the words it computes
//   twist_fold, fold, combine          the passes of a split product
//   split_transposed, stage_turned, unfold_add   those of a middle product
//
// c = f g, N = L + r, 0 < r <= L. With A = c mod (x^L - 1) (a cyclic product of length L) and q =
// c div x^L (the top r words of c, deg q < r): c = A + (x^L - 1) q, so c[k] = A[k] - q[k] for k < r,
// c[k] = A[k] for r <= k < L and c[L + k] = q[k]. q comes from at most two more products:
//   - the twisted piece, of s words (a power of two, s <= L/2): X = q mod (x^s - z^s), z = kZeta,
//     from C = c mod (x^s - z^s): X = (C - A mod (x^s - z^s)) / (z^L - 1), as x^L = z^L there. The
//     substitution x = z y turns x^s - z^s into y^s - 1, so C is a cyclic product of length s of
//     f(z y) and g(z y) mod (y^s - 1) (twist_fold), followed by the untwist;
//   - the top product: T = the top t = r - s words of c (t <= s), which depend only on the top t
//     words of f and g: T is the top of their product, of at most 2t - 1 words, itself a product
//     of ragged length (planned again).
// Then q = X + (x^s - z^s) T. With s = 0, q = T; with t = 0, q = X. The plan (choose) is a tuned
// choice per length and r / L (kinds); the plain product at 2L stays one of its options. Sources:
// notes/155-r1.md.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

#include "backend/modarith/modarith.hpp"
#include "backend/ntt/common.hpp"
#include "backend/ntt/tuned.hpp"
#include "backend/simd/select.hpp"
#include "backend/tune/tune.hpp"

namespace fps::ntt::detail::ragged {

using u32 = std::uint32_t;

// The twist: z = 3 generates the multiplicative group, so z^L != 1 for every L < P - 1.
inline constexpr u32 kZeta = 3;

// Twisted pieces have s >= max(2^kMinLog, L / kMaxChunks): the passes fold at most kMaxChunks
// blocks of s words of A (and 2 kMaxChunks of a factor).
inline constexpr std::size_t kMaxChunks = 32;

// A product of N words. split: c mod (x^L - 1) at 2^lg, the twisted piece of s words (0: none), the
// top t words of c from the top product; N = L + r, t = r - s if r > s else 0. truncated: c mod (x^L
// - 1) at 2^lg and a truncated transform of x^L + 1 (truncated.hpp; tables of 2^(lg + 1)). Else a
// plain product at 2^lg >= N.
struct Plan {
    int lg = 0;
    std::size_t s = 0, t = 0;
    bool split = false, truncated = false;
};

// Truncated plans: x^L + 1's leaf form to a prefix of words, in nodes of at least kMinNode words
// (group products of 16 leaves and more; truncated.hpp).
inline constexpr std::size_t kMinNode = 128;
constexpr bool power_of_4(std::size_t x) { return std::has_single_bit(x) && std::countr_zero(x) % 2 == 0; }
// The children of a node of `size` words: its quarters where they are groups (size / 32 leaves a
// power of 4), else its halves.
constexpr int radix(std::size_t size) { return power_of_4(size / 32) ? 4 : 2; }
// The words of a node of `size` words its first n words take in nodes of kMinNode words and more:
// n rounded up, a child shorter than kMinNode taken whole.
constexpr std::size_t rounded(std::size_t size, std::size_t n) {
    const std::size_t c = size / std::size_t(radix(size));
    if (c < kMinNode) return size;
    return n % c ? n / c * c + rounded(c, n % c) : n;
}

using Vec = simd::Native;
using Mod = modarith::Ops<Vec, kP>;
using V = Vec::V;
inline constexpr std::size_t kLanes = Vec::lanes;

inline u32 inverse(u32 x) { return power(x, kP - 2); }

// Passes run kChains vectors of kLanes words per step, with as many independent chains of powers
// (one chain was latency-bound: a Shoup product and a reduction per vector, notes/155-r1.md).
inline constexpr int kChains = 4;
inline constexpr std::size_t kBlock = kChains * kLanes;  // divides every s and L here (>= 2^kMinLog)

// The Montgomery forms of scale z^k (times 2^32 mod P), lazy (< 2P), for k in a block of kBlock
// words: (*this)[j] holds lanes [j kLanes, (j + 1) kLanes) of the block; advance() moves to the
// next block. montgomery(x, y) for x < 2P is then x scale z^k, in [0, 2P).
class Powers {
public:
    Powers(u32 z, u32 scale) {
        alignas(64) u32 lanes[kBlock];
        u32 x = multiply_mod(scale, kR);
        for (std::size_t i = 0; i < kBlock; ++i) lanes[i] = x, x = multiply_mod(x, z);
        for (int j = 0; j < kChains; ++j) y_[j] = Vec::load(lanes + j * kLanes);
        const u32 step = power(z, u32(kBlock));
        w_ = Vec::broadcast(step), q_ = Vec::broadcast(quotient(step));
    }
    [[gnu::always_inline]] V operator[](int j) const { return y_[j]; }
    [[gnu::always_inline]] void advance() {
        for (int j = 0; j < kChains; ++j) y_[j] = Mod::shoup(y_[j], w_, q_);
    }

private:
    V y_[kChains], w_, q_;
};

// A constant w < P and its Shoup quotient: times(x) = x w in [0, 2P), x < 2^32.
struct Constant {
    V w, q;
    Constant() = default;  // unset
    explicit Constant(u32 x) : w(Vec::broadcast(x)), q(Vec::broadcast(quotient(x))) {}
    Constant(u32 x, u32 shoup) : w(Vec::broadcast(x)), q(Vec::broadcast(shoup)) {}  // shoup = quotient(x)
    // Entry k of a twiddle table (common.hpp: values, then their quotients).
    static Constant entry(const u32* table, std::size_t k) { return {table[slot(k)], table[slot(k) + 8]}; }
    [[gnu::always_inline]] V times(V x) const { return Mod::shoup(x, w, q); }
};

// a + b for a, b < 2P, in [0, 2P).
[[gnu::always_inline]] inline V add2(V a, V b) { return Mod::reduce<2 * kP>(Vec::add(a, b)); }
// a - b for a, b < 2P, in [0, 2P).
[[gnu::always_inline]] inline V sub2(V a, V b) { return Mod::reduce<2 * kP>(Mod::sub_lazy<2 * kP>(a, b)); }

// lanes [k, k + kLanes) of src, zero from len on.
[[gnu::always_inline]] inline V load_upto(const u32* src, std::size_t k, std::size_t len) {
    if (k + kLanes <= len) return Vec::load(src + k);
    alignas(64) u32 lanes[kLanes] = {};
    for (std::size_t i = k; i < len; ++i) lanes[i - k] = src[i];
    return Vec::load(lanes);
}

// out[i] = z^i sum_{c < chunks} w[c] src[i + s c] for i in [k, end) (w[0] = 1), canonical: blocks
// whose chunks are all whole (kLoad = false: full vectors) or one block where the last ends (kLoad).
template <bool kLoad>
[[gnu::always_inline]] inline void twist_blocks(const u32* src, std::size_t len, std::size_t s, const Constant* w,
                                                std::size_t chunks, std::size_t k, std::size_t end, Powers& z, u32* out) {
    for (; k < end; k += kBlock, z.advance()) {
        for (int j = 0; j < kChains; ++j) {
            const std::size_t i = k + j * kLanes;
            const auto load = [&](std::size_t at) { return kLoad ? load_upto(src, at, len) : Vec::load(src + at); };
            V acc = chunks ? load(i) : Vec::broadcast(0);
            for (std::size_t c = 1; c < chunks; ++c) acc = add2(acc, w[c].times(load(i + c * s)));
            Vec::store(out + i, Mod::reduce<kP>(Mod::montgomery(acc, z[j])));
        }
    }
}

// out[k] = scale z^k sum_c z^(s c) src[k + s c] for k < s (src: len words, zero past them): scale
// src(z y) mod (y^s - 1), a factor of the twisted piece, canonical. len <= 2 kMaxChunks s. The
// blocks before the end of the last chunk read every chunk, those after it one fewer.
inline void twist_fold(const u32* src, std::size_t len, std::size_t s, u32* out, u32 zeta = kZeta, u32 scale = 1) {
    const std::size_t chunks = (len + s - 1) / s, last = len - (chunks - 1) * s;  // words of the last chunk
    std::array<Constant, 2 * kMaxChunks> w;
    const u32 x = power(zeta, u32(s));
    u32 wc = x;
    for (std::size_t c = 1; c < chunks; ++c, wc = multiply_mod(wc, x)) w[c] = Constant(wc);
    Powers z(zeta, scale);
    const std::size_t whole = last / kBlock * kBlock, past = std::min(s, whole + kBlock);
    twist_blocks<false>(src, len, s, w.data(), chunks, 0, whole, z, out);
    if (whole < last) twist_blocks<true>(src, len, s, w.data(), chunks, whole, past, z, out);
    twist_blocks<false>(src, len, s, w.data(), chunks - 1, whole < last ? past : whole, s, z, out);
}

// out[k] = sum_c src[k + L c] for k < L (src: len <= 2L words): src mod (x^L - 1), for a factor
// longer than the main product. L a multiple of kLanes.
inline void fold(const u32* src, std::size_t len, std::size_t L, u32* out) {
    for (std::size_t k = 0; k < L; k += kLanes) {
        V x = load_upto(src, k, len);
        if (k + L < len) x = Mod::add(x, load_upto(src, k + L, len));
        Vec::store(out + k, x);
    }
}

// The recombination: c from A = out[0, L), C (the twisted piece's cyclic product, s words; unread
// if s = 0) and T = out[L + s, L + r) (t <= s words). Writes out[0, L + r).
inline void combine(u32* out, std::size_t L, std::size_t r, std::size_t s, std::size_t t, const u32* C) {
    if (s) {
        // X[k] = z^-k C[k] / (z^L - 1) - sum_c w^c A[k + s c] / (z^L - 1) (w = z^s), then q[k] = X[k]
        // - w T[k] for k < s; for t = 0, only k < r (X = q has r words).
        const std::size_t chunks = L / s, end = t ? s : std::min(s, (r + kBlock - 1) / kBlock * kBlock);
        const u32 kinv = inverse((power(kZeta, u32(L)) + kP - 1) % kP), w = power(kZeta, u32(s));
        std::array<Constant, kMaxChunks> e;  // w^c / (z^L - 1)
        u32 x = kinv;
        for (std::size_t c = 0; c < chunks; ++c, x = multiply_mod(x, w)) e[c] = Constant(x);
        const Constant minus_w(kP - w);
        Powers y(inverse(kZeta), kinv);  // z^-k / (z^L - 1)
        const u32* T = out + L + s;
        for (std::size_t k = 0; k < end; k += kBlock, y.advance()) {
            for (int j = 0; j < kChains; ++j) {
                const std::size_t i = k + j * kLanes;
                V sum = e[0].times(Vec::load(out + i));
                for (std::size_t c = 1; c < chunks; ++c) sum = add2(sum, e[c].times(Vec::load(out + i + c * s)));
                V q = sub2(Mod::montgomery(Vec::load(C + i), y[j]), sum);
                if (i < t) q = add2(q, minus_w.times(load_upto(T, i, t)));
                q = Mod::reduce<kP>(q);
                const V a = Mod::sub(Vec::load(out + i), q);
                if (i + kLanes <= r) {
                    Vec::store(out + L + i, q);
                    Vec::store(out + i, a);
                } else if (i < r) {  // the last words of q
                    Vec::store_first(out + L + i, q, int(r - i));
                    Vec::store_first(out + i, a, int(r - i));
                }
            }
        }
    }
    // c[k] = A[k] - q[k] where q[k] = T[k - s]: k in [s, r).
    std::size_t k = std::min(s, r);
    for (; k + kLanes <= r; k += kLanes) Vec::store(out + k, Mod::sub(Vec::load(out + k), Vec::load(out + L + k)));
    for (; k < r; ++k) {
        const u32 a = out[k], b = out[L + k];
        out[k] = a >= b ? a - b : a + kP - b;
    }
}

// Middle products (ntt::middle_product) run the transpose of the split: out = MP(v, g), v = f[0, L
// + r), is the transpose of u -> u G, G = g reversed, whose split is above. Transposed pass by
// pass: split_transposed (of combine) gives Abar, Cbar, Tbar from v; then out[i] = (Abar g
// x^-(m - 1) mod (x^L - 1))[i mod L] (the main product, as the plain middle product), plus z^i
// Z[(i + m - 1) mod s] with Z = Cbar (z^(m - 1) g(y / z) mod (y^s - 1)) (the twisted piece: G(z y)
// read backwards), plus the middle product of [0..0, Tbar] and g[0, mt) into the top words (the top
// product).

// From v (L + r words): Abar (L words), Cbar (s words, if s) and Tbar (t words) with qbar[k] = v[L
// + k] - v[k] (k < r): Tbar[k] = qbar[s + k] - z^s qbar[k], Cbar[k] = z^-k qbar[k] / (z^L - 1)
// and Abar[k + s c] = v[k + s c] - z^(s c) qbar[k] / (z^L - 1) for k < min(r, s); with s = 0,
// Tbar = qbar and Abar = v[0, L).
inline void split_transposed(const u32* v, std::size_t L, std::size_t r, std::size_t s, std::size_t t, u32* Abar, u32* Cbar,
                             u32* Tbar) {
    const u32* top = v + L;
    if (!s) {
        std::copy_n(v, L, Abar);
        std::size_t k = 0;
        for (; k + kLanes <= r; k += kLanes) Vec::store(Tbar + k, Mod::sub(Vec::load(top + k), Vec::load(v + k)));
        for (; k < r; ++k) Tbar[k] = top[k] >= v[k] ? top[k] - v[k] : top[k] + kP - v[k];
        return;
    }
    const std::size_t chunks = L / s;
    const u32 kinv = inverse((power(kZeta, u32(L)) + kP - 1) % kP), w = power(kZeta, u32(s));
    std::array<Constant, kMaxChunks> e;  // w^c / (z^L - 1)
    u32 x = kinv;
    for (std::size_t c = 0; c < chunks; ++c, x = multiply_mod(x, w)) e[c] = Constant(x);
    const Constant minus_w(kP - w);
    Powers y(inverse(kZeta), kinv);  // z^-k / (z^L - 1)
    for (std::size_t k = 0; k < s; k += kBlock, y.advance()) {
        for (int j = 0; j < kChains; ++j) {
            const std::size_t i = k + j * kLanes;
            const V q = i < r ? Mod::sub(load_upto(top, i, r), load_upto(v, i, r)) : Vec::broadcast(0);
            Vec::store(Cbar + i, Mod::reduce<kP>(Mod::montgomery(q, y[j])));
            for (std::size_t c = 0; c < chunks; ++c)
                Vec::store(Abar + i + c * s, Mod::sub(Vec::load(v + i + c * s), Mod::reduce<kP>(e[c].times(q))));
            if (i < t) {
                const V hi = Mod::sub(load_upto(top + s, i, t), load_upto(v + s, i, t));
                const V b = Mod::reduce<kP>(Mod::add(hi, Mod::reduce<kP>(minus_w.times(q))));
                if (i + kLanes <= t) Vec::store(Tbar + i, b);
                else Vec::store_first(Tbar + i, b, int(t - i));
            }
        }
    }
}

// b = g x^-(m - 1) mod (x^L - 1) (m <= 2L): b[(i - (m - 1)) mod L] summed over i < m.
inline void stage_turned(const u32* g, std::size_t m, std::size_t L, u32* b) {
    const std::size_t o = (L - (m - 1) % L) % L;  // g[i] goes to b[(i + o) mod L]
    std::fill_n(b, L, 0u);
    for (std::size_t i = 0; i < m;) {
        const std::size_t at = (i + o) % L, run = std::min(m - i, L - at);
        if (i < L) std::copy_n(g + i, std::min(run, L - i), b + at);
        if (i + run > L) {  // the words of g past L add to those before
            for (std::size_t k = std::max(i, L); k < i + run; ++k) {
                const u32 sum = b[at + (k - i)] + g[k];
                b[at + (k - i)] = sum >= kP ? sum - kP : sum;
            }
        }
        i += run;
    }
}

// out[i] += z^i Z[(i + rot) mod s] for i < count, out canonical.
inline void unfold_add(const u32* Z, std::size_t s, std::size_t rot, u32* out, std::size_t count) {
    Powers z(kZeta, 1);
    for (std::size_t k = 0; k < count; k += kBlock, z.advance()) {
        for (int j = 0; j < kChains; ++j) {
            const std::size_t i = k + j * kLanes, at = (i + rot) & (s - 1);
            if (i >= count) break;
            V x;
            if (at + kLanes <= s) {
                x = Vec::load(Z + at);
            } else {
                alignas(64) u32 lanes[kLanes];
                for (std::size_t l = 0; l < kLanes; ++l) lanes[l] = Z[(at + l) & (s - 1)];
                x = Vec::load(lanes);
            }
            x = Mod::reduce<kP>(Mod::montgomery(x, z[j]));
            if (i + kLanes <= count) {
                Vec::store(out + i, Mod::add(Vec::load(out + i), x));
            } else {
                alignas(64) u32 lanes[kLanes] = {};
                std::copy_n(out + i, count - i, lanes);
                Vec::store(lanes, Mod::add(Vec::load(lanes), x));
                std::copy_n(lanes, count - i, out + i);
            }
        }
    }
}

// Plan kinds for N = L + r: the plain product at 2L, the top product alone (s = 0, t = r), the
// twisted piece of bit_floor(r) words and the top product of the rest, the piece of bit_ceil(r)
// words alone (t = 0), and the truncated transform of x^L + 1 (truncated.hpp; products only).
enum Kind : std::uint8_t { kPlain, kTop, kFloor, kCeil, kTruncated };
inline constexpr int kKinds = 5;

// Whether a truncated plan exists for N words, L < N < 2L: x^L + 1's nodes to r words take fewer
// than its L (rounded() < L), and 2L <= 2^kMaxLog.
constexpr bool truncates(std::size_t N) {
    if (N <= (std::size_t(1) << kMinLog) || N > (std::size_t(1) << kMaxLog)) return false;
    const std::size_t L = std::bit_floor(N - 1);
    return rounded(L, N - L) < L;
}

// The plan of a kind below kTruncated for N words. A kind that is no plan here falls back to the
// next in the order ceil, floor, top, plain: N <= 2^kMinLog, a piece outside [max(2^kMinLog, L /
// kMaxChunks), L / 2] (r too short or above L / 2), or products not shorter than the plain one's (s
// + 2t - 1 >= L: the top product is about 2t words).
constexpr Plan split_plan(std::size_t N, int kind) {
    const int lg = std::max(kMinLog, int(std::bit_width(N - 1)));
    if (N <= (std::size_t(1) << kMinLog)) return {lg, 0, 0, false, false};
    const std::size_t L = std::size_t(1) << (lg - 1), r = N - L;
    const std::size_t least = std::max<std::size_t>(std::size_t(1) << kMinLog, L / kMaxChunks);
    for (; kind > kPlain; --kind) {
        const std::size_t s = kind == kTop ? 0 : kind == kFloor ? std::bit_floor(r) : std::bit_ceil(r), t = r > s ? r - s : 0;
        if (s + 2 * t <= L && (s == 0 || (s >= least && s <= L / 2))) return {lg - 1, s, t, true, false};
    }
    return {lg, 0, 0, false, false};
}

// The truncated plan of N words, or where it needs all of x^L + 1 (truncates) the next kind's, ceil.
constexpr Plan truncated_plan(std::size_t N) {
    if (truncates(N)) return {int(std::bit_width(N - 1)) - 1, 0, 0, false, true};
    return split_plan(N, kCeil);
}

// The plan of any kind for N words (kTruncated falls back to ceil, floor, top, plain).
constexpr Plan plan(std::size_t N, int kind) { return kind == kTruncated ? truncated_plan(N) : split_plan(N, kind); }

constexpr bool operator==(const Plan& a, const Plan& b) {
    return a.lg == b.lg && a.s == b.s && a.t == b.t && a.split == b.split && a.truncated == b.truncated;
}

// The distinct plans of N words, plans[0] the plain one; returns their number.
constexpr int options(std::size_t N, std::array<Plan, kKinds>& plans) {
    int count = 0;
    for (int kind = 0; kind < kKinds; ++kind) {
        const Plan p = plan(N, kind);
        if (std::find(plans.begin(), plans.begin() + count, p) == plans.begin() + count) plans[count++] = p;
    }
    return count;
}

// The tuned choice "ntt.ragged" (backend/tune): two tables, each an entry per main length L = 2^lg
// (kMinLog <= lg < kMaxLog) and bucket of r / L, (b - 1) / 16 < r / L <= b / 16. The first (entries
// [0, kTable)), for products: entry v < kTruncated is the kind of every product there; v >=
// kTruncated: products (multiply, square) truncated, the others (ragged_into's callers outside ntt,
// which have no truncated form) kind v - kTruncated. The second ([kTable, 2 kTable)), for middle
// products: the kind (middle_kind: v >= kTruncated is truncated). tune.hpp measures them: every kind
// at the middle of each bucket; compiled defaults from tools/tune.py ntt (tuned.hpp: ragged, middle).
inline constexpr int kBuckets = 16, kRows = kMaxLog - kMinLog, kEntries = 2 * kTruncated, kTable = kRows * kBuckets;
using Kinds = std::array<std::uint8_t, 2 * kTable>;

constexpr int product_kind(int v) { return v < kTruncated ? v : kTruncated; }
constexpr int other_kind(int v) { return v < kTruncated ? v : v - kTruncated; }
constexpr int middle_kind(int v) { return v < kTruncated ? v : kTruncated; }
// The entry for a product kind and the others' kind (< kTruncated).
constexpr std::uint8_t encode(int product, int other) { return std::uint8_t(product == kTruncated ? kTruncated + other : product); }

inline constexpr Kinds kKindDefaults = [] {
    Kinds kinds{};
    for (int i = 0; i < kTable; ++i) {
        kinds[i] = std::uint8_t(tuned::ragged[int(tune::kTarget)][i]);
        kinds[kTable + i] = std::uint8_t(tuned::middle[int(tune::kTarget)][i]);
    }
    return kinds;
}();
static_assert(std::ranges::all_of(kKindDefaults, [](std::uint8_t k) { return k < kEntries; }));
inline constinit tune::Table<kKindDefaults> kinds;

// The entry of N words in Kinds, N > 2^kMinLog.
constexpr int entry(std::size_t N) {
    const int lg = int(std::bit_width(N - 1)) - 1;
    const std::size_t r = N - (std::size_t(1) << lg);
    return (lg - kMinLog) * kBuckets + int((r * kBuckets - 1) >> lg);
}

// The plan for N words outside multiply, square and middle products: the table's kind for its
// length and bucket (one table read), never truncated (backend/series' ragged_into).
inline Plan choose(std::size_t N) {
    if (N <= (std::size_t(1) << kMinLog) || N > (std::size_t(1) << kMaxLog)) return split_plan(N, kPlain);
    return split_plan(N, other_kind(kinds.get()[entry(N)]));
}

// The truncated plan out of line (choose_product): multiply()'s inlined path stays as before.
[[gnu::noinline]] inline Plan truncated_choice(std::size_t N) { return truncated_plan(N); }

// The plan of multiply and square for N words: as choose, truncated where the table says.
inline Plan choose_product(std::size_t N) {
    if (N <= (std::size_t(1) << kMinLog) || N > (std::size_t(1) << kMaxLog)) return split_plan(N, kPlain);
    const int v = kinds.get()[entry(N)];
    if (v >= kTruncated) return truncated_choice(N);
    return split_plan(N, v);
}

// The plan of a middle product reading N words: the second table's kind.
inline Plan choose_middle(std::size_t N) {
    if (N <= (std::size_t(1) << kMinLog) || N > (std::size_t(1) << kMaxLog)) return split_plan(N, kPlain);
    const int v = kinds.get()[kTable + entry(N)];
    if (v >= kTruncated) return truncated_choice(N);
    return split_plan(N, v);
}

// Words of scratch a split plan with main length L needs, beyond out: the main product (a factor
// folded to L words, a and b), the twisted piece (its two factors, a and b: 4 (s + kPadding) <= 2
// (L + kPadding)) or the top product (under L words: at most this), each array kPadding words
// longer, and 4 KiB to place a (place() in ntt.hpp) and to align.
constexpr std::size_t split_words(std::size_t L) { return 3 * (L + kPadding) + 2048; }

// Words of scratch a truncated plan with main length L needs: the leaf forms of length 2L of both
// factors, and 4 KiB to place them. A top product of a split plan (under L words) fits in
// split_words(L).
constexpr std::size_t truncated_words(std::size_t L) { return 2 * (2 * L + kPadding) + 1024; }

// Words of work ntt.hpp's ragged_into needs for plan p (its tables: ntt.hpp's table_log(p)): a
// caller outside ntt (backend/series' low_product) sizes its work by it.
constexpr std::size_t work_words(const Plan& p) {
    const std::size_t len = std::size_t(1) << p.lg;
    return p.split ? split_words(len) : p.truncated ? truncated_words(len) : 2 * (len + kPadding) + 1024;
}

}  // namespace fps::ntt::detail::ragged
