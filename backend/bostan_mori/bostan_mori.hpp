// Coefficient k of p / q modulo P = 998244353, by Bostan and Mori (backend kernel), for
// fps::Fraction<P>::operator[].
//
//   fps::bostan_mori::coefficient(p, q, k)   // [x^k] p / q: spans of words < P, q[0] != 0, any k
//
// The variants, public as the fast path for callers that pick their own (same contract):
//   variant::schoolbook(p, q, k)          p and q cut to k + 1 words: at most kSchoolbookCapacity
//                                         (256), else std::length_error
//   variant::values(p, q, k, small)       transforms in the values form; schoolbook once p and q
//                                         fit in `small` words (0: never); any size
//   variant::products(p, q, k)            two ntt::multiply per step (the frontend's loop)
// coefficient() is values(p, q, k, kSchoolbookMax): schoolbook while p and q have at most
// kSchoolbookMax words (select.hpp: 76 to 96 by target), the values form beyond. Throws
// std::domain_error for an empty q or q[0] = 0. No allocation per call: each thread keeps two
// arrays of 2N words, its transforms and a table of N words, for its longest call so far.
//
// Bostan and Mori: with p(x) q(-x) = U_0(x^2) + x U_1(x^2) and q(x) q(-x) = V(x^2),
// [x^k] p / q = [x^(k / 2)] U_(k mod 2) / V. Only p and q mod x^(k + 1) matter: both are cut to
// k + 1 words first and after each step.
//
// values: p and q in the values form of length L = 2N (ntt.hpp; N > deg q, 2N > deg p + deg q).
// The words of w and -w are 8 apart (rows 2m and 2m + 1 of a block), and w^2 is a point of the
// values form of length N, so one pass gives U(w^2) = (A(w) +- A(-w)) / w^r, A = p(x) q(-x), and
// V(w^2) = q(w) q(-w) in the length-N layout: no transform of q(-x), no product transform. The
// next step needs U and V at the other N points: an inverse of N (into the upper half) and
// forward_upper of 2N (doubling), each. A step: four transforms of N words, where two products
// run six of 2N. The cut shortens the transforms when it lets them. Scale: U comes out times
// 2^-31 and V times 2^-32 (Montgomery's 2^-32, and A(w) + A(-w) = 2 U(w^2)), so the ratio picks
// up 1/2 per step. The last step needs only U(0) / V(0) = (sum of U's values) / (sum of V's).
//
// schoolbook: per step U_0 = qe pe - x qo po, U_1 = qe po - qo pe, V = qe qe - x qo qo, with
// pe, po, qe, qo the even and odd halves (p = pe(x^2) + x po(x^2)): vectors of U's and V's words,
// 64-bit sums of products (backend/small's lanes), one Montgomery reduction per word. At most
// kTinyMax (8) words: scalar sums by the definition on a fixed shape per size (a template), with
// no loop of variable length and no branch on k's bits. With q[0] = 1 (linear recurrences) both
// skip the inverse at the end.
//
// Design and measurements: notes.md.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "backend/bostan_mori/select.hpp"
#include "backend/modarith/modarith.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/simd/simd.hpp"
#include "backend/small/schoolbook.hpp"

namespace fps::bostan_mori {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
inline constexpr u32 kP = ntt::kModulus;

namespace detail {

constexpr u32 mul(u32 a, u32 b) { return u32(u64(a) * b % kP); }
constexpr u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
constexpr u32 inverse(u32 a) { return power(a, kP - 2); }
inline constexpr u32 kR = u32((u64(1) << 32) % kP);  // 2^32 mod P
// 2^-h mod P for h <= 64: the scale of the values path after h steps.
inline constexpr auto kHalves = [] {
    std::array<u32, 65> t{};
    t[0] = 1;
    for (int h = 1; h <= 64; ++h) t[h] = mul(t[h - 1], (kP + 1) / 2);
    return t;
}();
// 2^(32 (2^h - 1)) mod P for h <= 64: after h schoolbook steps from q[0] = 1, the inverse of q[0].
inline constexpr auto kUnscale = [] {
    std::array<u32, 65> t{};
    t[0] = 1;
    for (int h = 1; h <= 64; ++h) t[h] = mul(mul(t[h - 1], t[h - 1]), kR);
    return t;
}();

// p and q cut to k + 1 words, p without its zero top words.
struct Input {
    std::span<const u32> p, q;
};
inline Input trim(std::span<const u32> p, std::span<const u32> q, u64 k) {
    if (q.empty() || q[0] == 0) throw std::domain_error("fps::bostan_mori: q[0] = 0");
    const auto cut = [k](std::span<const u32> f) { return k < f.size() ? f.first(std::size_t(k) + 1) : f; };
    p = cut(p), q = cut(q);
    while (!p.empty() && p.back() == 0) p = p.first(p.size() - 1);
    while (q.back() == 0) q = q.first(q.size() - 1);
    return {p, q};
}

// ---- Memory ------------------------------------------------------------------------------------

// This thread's transform of length 2^lg, kept across calls.
inline const ntt::Transform& transform(int lg) {
    thread_local std::array<std::unique_ptr<ntt::Transform>, ntt::kMaxLog + 1> cache;
    if (!cache[lg]) cache[lg] = std::make_unique<ntt::Transform>(lg);
    return *cache[lg];
}

// This thread's words, kept across calls: the largest count asked for so far, 2 MiB aligned. Not
// the NTT's region (Transform::inverse into other memory takes that one).
struct Kept {
    ntt::detail::Region region;
    ~Kept() { ntt::detail::unmap_region(region); }
};
inline u32* thread_words(std::size_t words) {
    thread_local Kept kept;
    const std::size_t bytes = words * sizeof(u32);
    if (kept.region.bytes < bytes) {
        ntt::detail::unmap_region(std::exchange(kept.region, {}));
        kept.region = ntt::detail::map_region(bytes);
    }
    return reinterpret_cast<u32*>(kept.region.start);
}

// Words after an array of the values path: ntt::kPadding, and 80 so that two arrays of a power of
// two words sit 320 bytes apart modulo 4 KiB (4K aliasing, backend/ntt/notes.md).
inline constexpr std::size_t kGap = 80;

// ---- Inverse points ----------------------------------------------------------------------------

// The values form's points (ntt.hpp, common.hpp): word 64 g + 8 i + t is at rho_v z^bitrev3(i),
// v = 8 g + t, rho_v = r(4 v), z = r(2) of order 8. Rows 2m and 2m + 1 hold w and -w, with
// w = rho_v z^bitrev2(m). Entry 32 g + 8 m + t of the table: w^-1 2^32 mod P (a Montgomery
// factor). Entries of block g do not depend on the length, so the table for the longest length
// serves every shorter one. This thread's, for `words` = L / 2 entries at least.
inline const u32* inverse_points(std::size_t words) {
    thread_local std::vector<u32> table;
    if (table.size() >= words) return table.data();
    // x c mod P by Shoup's product with c's quotient, x < 2^32
    const auto times = [](u32 x, u32 c, u32 cq) {
        const u32 r = u32(x * c - u32((u64(x) * cq) >> 32) * kP);
        return r >= kP ? r - kP : r;
    };
    const auto quotient = [](u32 c) { return u32((u64(c) << 32) / kP); };
    const std::size_t leaves = words / 4;  // L / 8
    // r(4 v)^-1 2^32: the product over the bits j of v of the inverse root of order 2^(j + 4)
    std::vector<u32> rho(leaves);
    rho[0] = kR;
    for (std::size_t h = 1; h < leaves; h *= 2) {
        const u32 c = inverse(power(3, (kP - 1) >> (std::countr_zero(h) + 4))), cq = quotient(c);
        for (std::size_t i = 0; i < h; ++i) rho[h + i] = times(rho[i], c, cq);
    }
    const u32 zinv = inverse(power(3, (kP - 1) >> 3));  // z^-1, z = r(2) of order 8
    const u32 z2 = mul(zinv, zinv);
    const u32 rows[4] = {1, z2, zinv, mul(z2, zinv)};  // z^-bitrev2(m): z^0, z^-2, z^-1, z^-3
    std::vector<u32> t(words);
    for (int m = 0; m < 4; ++m) {
        const u32 c = rows[m], cq = quotient(c);
        for (std::size_t g = 0; g < words / 32; ++g)
            for (std::size_t x = 0; x < 8; ++x) t[32 * g + 8 * std::size_t(m) + x] = times(rho[8 * g + x], c, cq);
    }
    table = std::move(t);
    return table.data();
}

// (even, odd): the even and the odd lanes of the 2W lanes a then b (W = 1, 4 or 8).
template <class S>
[[gnu::always_inline]] inline std::pair<typename S::V, typename S::V> unzip(typename S::V a, typename S::V b) {
    if constexpr (S::lanes == 1) {
        return {a, b};
    } else if constexpr (S::lanes == 8) {
#if defined(__AVX2__)
        const __m256 x = _mm256_castsi256_ps(a), y = _mm256_castsi256_ps(b);
        const __m256i e = _mm256_castps_si256(_mm256_shuffle_ps(x, y, 0x88));  // a0 a2 b0 b2 | a4 a6 b4 b6
        const __m256i o = _mm256_castps_si256(_mm256_shuffle_ps(x, y, 0xdd));
        return {_mm256_permute4x64_epi64(e, 0xd8), _mm256_permute4x64_epi64(o, 0xd8)};
#endif
    } else {
#if defined(__ARM_NEON) && defined(__aarch64__)
        return {vuzp1q_u32(a, b), vuzp2q_u32(a, b)};
#elif defined(__SSE4_1__)
        const __m128 x = _mm_castsi128_ps(a), y = _mm_castsi128_ps(b);
        return {_mm_castps_si128(_mm_shuffle_ps(x, y, 0x88)), _mm_castps_si128(_mm_shuffle_ps(x, y, 0xdd))};
#endif
    }
}

// ---- The halving pass (values path) -------------------------------------------------------------

template <class S>
struct Pass {
    using V = typename S::V;
    using M = modarith::Ops<S, kP>;
    static constexpr int W = S::lanes;
    static_assert(W == 1 || W == 4 || W == 8);
    static constexpr int C = 8 / W;  // vectors per row of 8 words

    // (a b + c d) 2^-32 and (a b - c d + P^2) 2^-32 mod P, in [0, 2P), for a, b, c, d < P: 64-bit
    // sums below 2 P^2 < 2^32 P, one Montgomery reduction.
    template <bool Minus>
    [[gnu::always_inline]] static V reduce_pair(V a, V b, V c, V d) {
        if constexpr (small::detail::HasMulEven<S>) {
            const V p = S::broadcast(kP), ninv = S::broadcast(0u - M::kPinv);
            const V a1 = S::template shr64<32>(a), b1 = S::template shr64<32>(b);
            const V c1 = S::template shr64<32>(c), d1 = S::template shr64<32>(d);
            V e, o;
            if constexpr (Minus) {
                const V pp = S::broadcast64(u64(kP) * kP);
                e = S::sub64(S::add64(pp, S::mul_even(a, b)), S::mul_even(c, d));
                o = S::sub64(S::add64(pp, S::mul_even(a1, b1)), S::mul_even(c1, d1));
            } else {
                e = S::add64(S::mul_even(a, b), S::mul_even(c, d));
                o = S::add64(S::mul_even(a1, b1), S::mul_even(c1, d1));
            }
            e = S::add64(e, S::mul_even(S::mul_even(e, ninv), p));
            o = S::add64(o, S::mul_even(S::mul_even(o, ninv), p));
            return S::join_high(e, o);
        } else if constexpr (small::detail::HasMulWide<S>) {
            using Wide = typename S::W;
            Wide lo, hi;
            if constexpr (Minus) {
#if defined(__ARM_NEON) && defined(__aarch64__)
                const Wide pp = vdupq_n_u64(u64(kP) * kP);
                lo = S::mls_wide_lo(S::mla_wide_lo(pp, a, b), c, d);
                hi = S::mls_wide_hi(S::mla_wide_hi(pp, a, b), c, d);
#endif
            } else {
                lo = S::mla_wide_lo(S::mul_wide_lo(a, b), c, d);
                hi = S::mla_wide_hi(S::mul_wide_hi(a, b), c, d);
            }
            const V m = S::mul_lo(S::lo_halves(lo, hi), S::broadcast(0u - M::kPinv)), p = S::broadcast(kP);
            return S::hi_halves(S::mla_wide_lo(lo, m, p), S::mla_wide_hi(hi, m, p));
        } else {
            const u64 s = Minus ? u64(a) * b + u64(kP) * kP - u64(c) * d : u64(a) * b + u64(c) * d;
            const u32 m = u32(s) * (0u - M::kPinv);
            return u32((s + u64(m) * kP) >> 32);
        }
    }

    // Output block h of U and V (64 words each) from blocks 2h and 2h + 1 of p and q (length L).
    // Output row m gets the even lanes (leaves 2u) of pair m, row m + 4 the odd ones (leaves 2u + 1):
    // leaf u = 4 g + t / 2 of length N holds the squares of leaves 2u and 2u + 1 of length L.
    template <bool Odd>
    [[gnu::always_inline]] static void block(const u32* p, const u32* q, const u32* winv, std::size_t h, u32* u,
                                             u32* v) {
#pragma GCC unroll 4
        for (int m = 0; m < 4; ++m) {
            V us[2 * C], vs[2 * C];
#pragma GCC unroll 16
            for (int i = 0; i < 2 * C; ++i) {
                const std::size_t g = 2 * h + std::size_t(i / C), word = 64 * g + 16 * std::size_t(m) + std::size_t(i % C) * W;
                const V p0 = S::load(p + word), p1 = S::load(p + word + 8);
                const V q0 = S::load(q + word), q1 = S::load(q + word + 8);
                V x;
                if constexpr (Odd) {
                    const V w = S::load(winv + 32 * g + 8 * std::size_t(m) + std::size_t(i % C) * W);
                    x = M::montgomery(reduce_pair<true>(p0, q1, p1, q0), w);
                } else {
                    x = reduce_pair<false>(p0, q1, p1, q0);
                }
                us[i] = M::template reduce<kP>(x);
                vs[i] = M::template reduce<kP>(M::montgomery(q0, q1));
            }
#pragma GCC unroll 8
            for (int i = 0; i < C; ++i) {
                const auto [ue, uo] = unzip<S>(us[2 * i], us[2 * i + 1]);
                const auto [ve, vo] = unzip<S>(vs[2 * i], vs[2 * i + 1]);
                S::store(u + 8 * m + W * i, ue), S::store(u + 8 * (m + 4) + W * i, uo);
                S::store(v + 8 * m + W * i, ve), S::store(v + 8 * (m + 4) + W * i, vo);
            }
        }
    }

    // p, q: values forms of length L >= 128 -> p[0, L/2) = U, q[0, L/2) = V, values forms of length
    // L / 2, times 2^-31 (U) and 2^-32 (V). In place: output block h overwrites input block h,
    // read for output block h / 2; block 0 goes through a copy.
    template <bool Odd>
    static void run(u32* p, u32* q, std::size_t L, const u32* winv) {
        alignas(64) u32 u[64], v[64];
        block<Odd>(p, q, winv, 0, u, v);
        std::copy_n(u, 64, p), std::copy_n(v, 64, q);
        for (std::size_t h = 1; h < L / 128; ++h) block<Odd>(p, q, winv, h, p + 64 * h, q + 64 * h);
    }
};

// sum of n words < P, mod P
inline u32 sum(const u32* a, std::size_t n) {
    u64 s = 0;
    for (std::size_t i = 0; i < n; ++i) s += a[i];
    return u32(s % kP);
}

// ---- Schoolbook steps ---------------------------------------------------------------------------

template <class S>
struct Schoolbook {
    using V = typename S::V;
    using L = small::detail::Lanes<S, kP>;
    using Acc = typename L::Acc;
    using M = modarith::Ops<S, kP>;
    using Sums = small::Sums<kP>;
    static constexpr std::ptrdiff_t W = S::lanes;
    static constexpr std::size_t kMax = kSchoolbookCapacity;  // words of p and q at most
    // Halves (pe, po, qe, qo, -qo) with kFront zeros before (windows start at -W) and zeros after
    // (windows reach kMax + W).
    static constexpr std::size_t kFront = 16;
    static constexpr std::size_t kSpan = kFront + kMax + 2 * W + 16;
    static_assert(W <= 8);

    // Sums of a block of U and of V: words [j0, j0 + W).
    struct Block {
        Acc u, v;
        std::size_t count = 0, folds = 0;
        bool started = false;
    };

    // x += c[a] (window of bu at j0 - su - a, of bv at j0 - sv - a) for a in [lo, hi); wu, wv: the
    // windows at a = lo. One broadcast of c[a] for both.
    [[gnu::always_inline]] static void terms(Block& x, const u32* c, std::ptrdiff_t lo, std::ptrdiff_t hi,
                                             const u32* wu, const u32* wv) {
        if (lo >= hi) return;
        if (!x.started) {
            L::start(x.u, wu), L::start(x.v, wv), x.started = true;
        } else if constexpr (requires { x.u.next; }) {  // a new run of windows (EvenOdd)
            x.u.next = S::load(wu + 1), x.v.next = S::load(wv + 1);
        }
        for (std::ptrdiff_t a = lo;;) {
            const std::ptrdiff_t stop = std::min(hi, a + std::ptrdiff_t(Sums::kFold - x.count));
            x.count += std::size_t(stop - a);
            for (; a < stop; ++a, --wu, --wv) {
                const V b = S::broadcast(c[a]);
                L::step(x.u, wu, b), L::step(x.v, wv, b);
            }
            if (a == hi) return;
            L::fold(x.u), L::fold(x.v), x.count = 0, ++x.folds;
        }
    }

    [[gnu::always_inline]] static void finish(Block& x, u32* u, u32* v) {
        if (!x.started) {
            S::store(u, V{}), S::store(v, V{});
            return;
        }
        if (x.count > (x.folds ? Sums::kLastFromFold : Sums::kLastFromZero)) L::fold(x.u), L::fold(x.v);
        S::store(u, L::finish(x.u)), S::store(v, L::finish(x.v));
    }

    // [x^k] p / q times kept scale: p, q of at most kMax words (np, nq >= 1), k >= 1.
    static u32 run(std::span<const u32> p0, std::span<const u32> q0, u64 k) {
        alignas(64) u32 pe[kSpan], po[kSpan], qe[kSpan], qo[kSpan], nqo[kSpan];
        alignas(64) u32 out[2][kMax + 4 * W + 16];
        u32 *const e = pe + kFront, *const o = po + kFront, *const f = qe + kFront, *const g = qo + kFront,
                   *const ng = nqo + kFront;
        std::size_t np = p0.size(), nq = q0.size();
        const std::size_t width = std::max(np, nq);  // sizes stay at most this
        const auto halves = [](std::size_t n) { return ((n + 1) / 2 + W - 1) / W * W; };  // words of a half, rounded
        std::size_t written = halves(width);  // the halves may be nonzero below this
        const V zero = S::broadcast(0);
        // Zeros where windows reach, [-W - 1, written + W + 1) of each half, by vector stores. The
        // asm keeps these loops (and the ones below) from becoming calls of memset: GCC 16 makes
        // the per-step ones calls, with vector spills around them (notes/198-r1.md).
        for (std::size_t i = 0; i < kFront + written + 2 * W + 16; i += W) {
            asm("" : "+r"(i));
            S::store(pe + i, zero), S::store(po + i, zero), S::store(qe + i, zero), S::store(qo + i, zero);
            S::store(nqo + i, zero);
        }
        for (std::size_t i = 0; i < np; ++i) (i % 2 ? o : e)[i / 2] = p0[i];
        for (std::size_t i = 0; i < nq; ++i) (i % 2 ? g : f)[i / 2] = q0[i];
        for (std::size_t i = 0; i < nq / 2; ++i) ng[i] = g[i] ? kP - g[i] : 0;
        int steps = 0;
        for (; k > 0; k /= 2, ++steps) {
            const std::size_t r = k % 2, top = std::size_t(std::min<u64>(k / 2 + 1, width));
            const std::ptrdiff_t ne = std::ptrdiff_t(np + 1) / 2, no = std::ptrdiff_t(np / 2);
            const std::ptrdiff_t me = std::ptrdiff_t(nq + 1) / 2, mo = std::ptrdiff_t(nq / 2);
            const std::size_t nu = std::min(np + nq >= 2 + r ? (np + nq - 2 - r) / 2 + 1 : 0, top);
            const std::size_t nv = std::min(nq, top);
            if (nu == 0) return 0;
            // U_0 = qe pe + (-qo) x po, U_1 = qe po + (-qo) pe, V = qe qe + (-qo) x qo
            const u32 *const u1 = r ? o : e, *const u2 = r ? e : o;
            const std::ptrdiff_t nu1 = r ? no : ne, nu2 = r ? ne : no, su2 = r ? 0 : 1;
            for (std::ptrdiff_t j = 0; j < std::ptrdiff_t(std::max(nu, nv)); j += W) {
                Block x;
                // term 1: a < me, and some lane j' of the block with j' - a in [0, nu1) or [0, me)
                const std::ptrdiff_t lo1 = std::max<std::ptrdiff_t>(0, j + 1 - std::max(nu1, me));
                terms(x, f, lo1, std::min(me, j + W), u1 + j - lo1, f + j - lo1);
                // term 2: a < mo, and j' - su2 - a in [0, nu2) or j' - 1 - a in [0, mo)
                const std::ptrdiff_t lo2 = std::max<std::ptrdiff_t>(0, std::min(j - su2 + 1 - nu2, j - mo));
                terms(x, ng, lo2, std::min(mo, j + W - su2), u2 + j - su2 - lo2, g + j - 1 - lo2);
                finish(x, out[0] + j, out[1] + j);
            }
            // the halves of U and V, zeros from nu and nv on (the blocks wrote garbage past them)
            const std::size_t extent = halves(std::max(nu, nv));
            for (std::size_t j = nu; j < 2 * extent; j += W) {
                asm("" : "+r"(j));
                S::store(out[0] + j, zero);
            }
            for (std::size_t j = nv; j < 2 * extent; j += W) {
                asm("" : "+r"(j));
                S::store(out[1] + j, zero);
            }
            for (std::size_t i = 0; i < extent; i += W) {
                const auto [ue, uo] = unzip<S>(S::load(out[0] + 2 * i), S::load(out[0] + 2 * i + W));
                const auto [ve, vo] = unzip<S>(S::load(out[1] + 2 * i), S::load(out[1] + 2 * i + W));
                S::store(e + i, ue), S::store(o + i, uo), S::store(f + i, ve), S::store(g + i, vo);
                S::store(ng + i, M::sub(zero, vo));
            }
            for (std::size_t i = extent; i < written; i += W) {  // shorter than before: zeros
                asm("" : "+r"(i));
                S::store(e + i, zero), S::store(o + i, zero), S::store(f + i, zero), S::store(g + i, zero);
                S::store(ng + i, zero);
            }
            written = extent, np = nu, nq = nv;
        }
        // f[0] is q0[0]^(2^steps) 2^(-32 (2^steps - 1)) (each step squares it, times 2^-32)
        return q0[0] == 1 ? mul(e[0], kUnscale[steps]) : mul(e[0], inverse(f[0]));
    }
};

// ---- Tiny steps ---------------------------------------------------------------------------------

// f(integral_constant<int, i>) for i in [0, N), unrolled at compile time.
template <int N, class F>
[[gnu::always_inline]] inline void unroll(F&& f) {
    [&]<int... I>(std::integer_sequence<int, I...>) { (f(std::integral_constant<int, I>{}), ...); }(
        std::make_integer_sequence<int, N>{});
}

// Steps for p and q of at most kTinyMax words, in scalar code on a fixed shape: p of NP and q of NQ
// words, NP = NQ - 1 or NP = NQ (zero words padded), which a step keeps for either r. With
// s = q(-x): U[j] = sum over i of s_i p_(2j + r - i), V[j] = s_j q_j + 2 sum over i < j of
// s_i q_(2j - i), each a 64-bit sum of at most NQ + 1 products below P^2, reduced once: exact.
// p_(b + r) is picked by a mask of r: no loop of variable length, no branch on k's bits (GCC 16's
// exit tests on the loops of the old version cost Zen 3 1.5-2.8x, notes/198-r1.md). Faster than
// Schoolbook's vectors up to 8 words on every timed target.
inline constexpr std::size_t kTinyMax = 8;

template <int NP, int NQ>
[[gnu::noinline]] u32 tiny_steps(const u32* p0, const u32* q0, u64 k) {
    static_assert(NP == NQ || NP == NQ - 1);
    static_assert(NQ <= 16);  // sums of at most 17 products < P^2 stay below 2^64
    u32 p[NP], q[NQ];
    unroll<NP>([&](auto i) { p[i] = p0[i]; });
    unroll<NQ>([&](auto i) { q[i] = q0[i]; });
    for (; k > 0; k /= 2) {
        u32 s[NQ], ps[NP + 1], v[NQ];  // s = q(-x), odd words P - q_i; ps[b + 1] = p_(b + r)
        unroll<NQ>([&](auto i) { s[i] = i % 2 ? kP - q[i] : q[i]; });
        const u32 mask = 0u - u32(k % 2);
        unroll<NP + 1>([&](auto c) {
            constexpr int b = int(c) - 1;
            u32 lo = 0, hi = 0;
            if constexpr (b >= 0) lo = p[b];
            if constexpr (b + 1 < NP) hi = p[b + 1];
            ps[c] = lo ^ ((lo ^ hi) & mask);
        });
        unroll<NP>([&](auto j) {
            u64 sum = 0;
            unroll<NQ>([&](auto i) {
                constexpr int b = 2 * int(j) - int(i);
                if constexpr (b >= -1 && b < NP) sum += u64(s[i]) * ps[b + 1];
            });
            p[j] = u32(sum % kP);
        });
        unroll<NQ>([&](auto j) {
            u64 sum = u64(s[j]) * q[j];
            unroll<int(j)>([&](auto i) {
                constexpr int m = 2 * int(j) - int(i);
                if constexpr (m < NQ) sum += 2 * (u64(s[i]) * q[m]);
            });
            v[j] = u32(sum % kP);
        });
        unroll<NQ>([&](auto i) { q[i] = v[i]; });
    }
    // q[0] is q0[0]^(2^steps): 1 stays 1 (every linear recurrence)
    return q[0] == 1 ? p[0] : mul(p[0], inverse(q[0]));
}

inline u32 tiny(std::span<const u32> p, std::span<const u32> q, u64 k) {
    u32 a[kTinyMax] = {}, b[kTinyMax] = {};
    for (std::size_t i = 0; i < p.size(); ++i) a[i] = p[i];
    for (std::size_t i = 0; i < q.size(); ++i) b[i] = q[i];
    u32 result = 0;
    if (p.size() < q.size())
        unroll<int(kTinyMax) - 1>([&](auto i) {
            if (q.size() == i + 2) result = tiny_steps<i + 1, i + 2>(a, b, k);
        });
    else
        unroll<int(kTinyMax)>([&](auto i) {
            if (p.size() == i + 1) result = tiny_steps<i + 1, i + 1>(a, b, k);
        });
    return result;
}

// [x^k] p / q times kept scale for p, q of at most kSchoolbookCapacity words, k >= 1.
inline u32 small_steps(std::span<const u32> p, std::span<const u32> q, u64 k) {
    if (std::max(p.size(), q.size()) <= kTinyMax) return tiny(p, q, k);
    return Schoolbook<SchoolbookIsa>::run(p, q, k);
}

}  // namespace detail

namespace variant {

// [x^k] p / q by schoolbook steps; p and q cut to k + 1 words: at most kSchoolbookCapacity.
inline u32 schoolbook(std::span<const u32> p, std::span<const u32> q, u64 k) {
    const auto [a, b] = detail::trim(p, q, k);
    if (a.empty()) return 0;
    if (k == 0) return detail::mul(a[0], detail::inverse(b[0]));
    if (std::max(a.size(), b.size()) > kSchoolbookCapacity)
        throw std::length_error("fps::bostan_mori::variant::schoolbook: more than kSchoolbookCapacity words");
    return detail::small_steps(a, b, k);
}

// [x^k] p / q by two ntt::multiply per step, into vectors (the frontend's loop; any size).
inline u32 products(std::span<const u32> p, std::span<const u32> q, u64 k) {
    const auto [a, b] = detail::trim(p, q, k);
    if (a.empty()) return 0;
    std::vector<u32> f(a.begin(), a.end()), g(b.begin(), b.end()), minus, x, y;
    for (; k > 0; k /= 2) {
        minus = g;
        for (std::size_t i = 1; i < minus.size(); i += 2) minus[i] = minus[i] ? kP - minus[i] : 0;
        x.resize(f.size() + g.size() - 1), y.resize(2 * g.size() - 1);
        ntt::multiply(f, minus, x);
        ntt::multiply(g, minus, y);
        const u64 top = k / 2 + 1;
        f.assign(std::size_t(std::min<u64>((x.size() + 1 - k % 2) / 2, top)), 0);
        g.assign(std::size_t(std::min<u64>((y.size() + 1) / 2, top)), 0);
        for (std::size_t i = 0; i < f.size(); ++i) f[i] = x[2 * i + k % 2];
        for (std::size_t i = 0; i < g.size(); ++i) g[i] = y[2 * i];
        if (f.empty()) return 0;
    }
    return detail::mul(f[0], detail::inverse(g[0]));
}

// [x^k] p / q by steps in the values form (above); schoolbook once p and q fit in `small` words
// (at most kSchoolbookCapacity; 0: never). Past the values form's 2^23 words: products.
inline u32 values(std::span<const u32> p, std::span<const u32> q, u64 k, std::size_t small = kSchoolbookMax) {
    using namespace detail;
    const auto [a, b] = trim(p, q, k);
    if (a.empty()) return 0;
    if (k == 0) return mul(a[0], inverse(b[0]));
    small = std::min(small, kSchoolbookCapacity);
    std::size_t dp = a.size() - 1, dq = b.size() - 1;  // degree bounds
    if (std::max(dp, dq) < small) return small_steps(a, b, k);
    // N > dq, 2N > dp + dq, N >= 2^ntt::kMinLog
    const auto half_length = [](std::size_t dp, std::size_t dq) {
        return std::max<std::size_t>({std::bit_ceil(dq + 1), std::bit_ceil(dp + dq + 1) / 2, std::size_t(1) << ntt::kMinLog});
    };
    std::size_t n = half_length(dp, dq);
    if (std::countr_zero(n) + 1 > ntt::kValuesMaxLog) return products(a, b, k);
    const std::size_t span = 2 * n + ntt::kPadding + kGap;
    u32* const pv = thread_words(2 * span);
    u32* const qv = pv + span;
    const u32* const winv = inverse_points(n);
    std::copy(a.begin(), a.end(), pv), std::fill(pv + a.size(), pv + 2 * n + ntt::kPadding, 0u);
    std::copy(b.begin(), b.end(), qv), std::fill(qv + b.size(), qv + 2 * n + ntt::kPadding, 0u);
    int lg = std::countr_zero(n);  // of n; the forms have length 2n
    transform(lg + 1).forward(pv, ntt::Form::values);
    transform(lg + 1).forward(qv, ntt::Form::values);
    std::size_t halvings = 0;
    for (;;) {
        const bool odd = k % 2;
        if (odd) Pass<PassIsa>::run<true>(pv, qv, 2 * n, winv);
        else Pass<PassIsa>::run<false>(pv, qv, 2 * n, winv);
        ++halvings, k /= 2;
        const u32 scale = kHalves[halvings];
        if (k == 0) return mul(scale, mul(sum(pv, n), inverse(sum(qv, n))));
        if (dp + dq < (odd ? 1u : 0u)) return 0;
        dp = std::size_t(std::min<u64>((dp + dq - (odd ? 1 : 0)) / 2, k)), dq = std::size_t(std::min<u64>(dq, k));
        const ntt::Transform& t = transform(lg);
        if (std::max(dp, dq) < small) {
            t.inverse(pv, ntt::Form::values), t.inverse(qv, ntt::Form::values);
            return mul(scale, small_steps({pv, dp + 1}, {qv, dq + 1}, k));
        }
        const std::size_t next = half_length(dp, dq);
        if (next < n) {  // the cut shortens the transforms: from the coefficients, cut
            t.inverse(pv, ntt::Form::values), t.inverse(qv, ntt::Form::values);
            std::fill(pv + dp + 1, pv + n, 0u), std::fill(qv + dq + 1, qv + n, 0u);
            n = next, lg = std::countr_zero(n);
            transform(lg + 1).forward(pv, ntt::Form::values);
            transform(lg + 1).forward(qv, ntt::Form::values);
        } else {  // doubling: the coefficients into the upper half, then its values
            t.inverse(pv, {pv + n, n}, ntt::Form::values), t.inverse(qv, {qv + n, n}, ntt::Form::values);
            const ntt::Transform& t2 = transform(lg + 1);
            t2.forward_upper(pv + n, ntt::Form::values), t2.forward_upper(qv + n, ntt::Form::values);
        }
    }
}

}  // namespace variant

// [x^k] p / q (above).
inline u32 coefficient(std::span<const u32> p, std::span<const u32> q, u64 k) {
    return variant::values(p, q, k, kSchoolbookMax);
}

}  // namespace fps::bostan_mori
