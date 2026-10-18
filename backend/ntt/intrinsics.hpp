// Variant intrinsics<S>: the NTT of avx2.hpp (same splitting, same lazy ranges, same leaf
// products) written once as templates on a simd struct S (backend/simd) and modarith, so every
// target runs it. Design and measurements: notes.md.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "backend/modarith/modarith.hpp"
#include "backend/ntt/common.hpp"
#include "backend/simd/simd.hpp"

namespace fps::ntt::detail::intrinsics {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

// The struct for work inside 8-word leaves: S if it has at most 8 lanes, else its ymm form.
template <class S>
struct Narrow {
    using type = S;
};
#if defined(__AVX2__) && defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512BW__)
template <>
struct Narrow<simd::Avx512> {
    using type = simd::Avx512Vl;
};
#endif

// Generated kernels in place of some of the code below, for S (avx512.hpp: variant::avx512 runs
// this file on a struct of its own, with asm forward groups and product bottoms). None here.
template <class S>
struct Generated {
    static constexpr bool kOn = false;
};
// Optional members of Generated<S> beyond kOn's (neon.hpp, issue #151), each used where present:
//   forward_from<kIn>(from, f, q, words, k, r) -> bool    forward_group_part with from != f
//   inverse(f, q, k, ir) -> bool                           inverse_group
//   bottom_forward(f, count, first, r), bottom_inverse     Bottom::forward, inverse; count even
//   forward_groups(f, q, k, count, r) -> bool             forward_groups (avx512.hpp, issue #152)
//   kJoinBottoms                                           Ntt::kJoin (2^6, 2^7: one bottom call)
//   inverse_scaled(f, q, k, ir, scaled) -> bool            inverse_group_scaled
//   inverse_top<kStore, kUpper>(from, f, q, words, ir, s)  inverse_radix4, inverse_radix4_part; -> bool
//   inverse_radix2<kStore, kUpper>(from, f, half, words, s) inverse_radix2, inverse_radix2_part; -> bool
//   inverse_terms(out, terms, n, count, first, r, ir) -> bool  Bottom::inverse_terms; count even
//   kShareGroups                                           single groups in place (forward_group with kIn
//                                                          4, inverse_group; quarters of 64 words and
//                                                          more) run through forward_groups and
//                                                          inverse_groups: one inlined copy of the
//                                                          hooks' loops (avx512.hpp, issue #152)

// Lazy arithmetic mod P on S: values < 4P in forward transforms, < 2P in inverse ones.
template <class S>
struct Arith {
    using V = typename S::V;
    using M = modarith::Ops<S, kP>;

    struct Factor {  // a constant w < P and its Shoup quotient
        V w, q;
    };
    [[gnu::always_inline]] static Factor factor(u32 w, u32 q) { return {S::broadcast(w), S::broadcast(q)}; }
    static Factor factor(u32 w) { return factor(w, quotient(w)); }
    [[gnu::always_inline]] static Factor entry(const u32* table, std::size_t k) { return factor(table[slot(k)], table[slot(k) + 8]); }

    [[gnu::always_inline]] static V add(V x, V y) { return S::add(x, y); }
    [[gnu::always_inline]] static V diff(V x, V y) { return S::sub(S::add(x, S::broadcast(2 * kP)), y); }  // x - y + 2P, y <= 2P
    [[gnu::always_inline]] static V low(V x) { return M::template reduce<2 * kP>(x); }                       // [0, 4P) -> [0, 2P)
    [[gnu::always_inline]] static V canonical(V x) { return M::template reduce<kP>(x); }                     // [0, 2P) -> [0, P)
    // x - y mod 2P for x, y < 2P, in [0, 2P): x - y wraps above 2P exactly when it is negative.
    [[gnu::always_inline]] static V low_diff(V x, V y) {
        const V t = S::sub(x, y);
        return S::min(t, S::add(t, S::broadcast(2 * kP)));
    }
    // x f, any x -> [0, 2P): Shoup's product, as modarith's. Every factor has equal words 2i and
    // 2i + 1 (broadcasts, half broadcasts), so on x86 the odd words' product reads q unshifted: one
    // shift less where q is not loop-invariant (bottom levels, groups of two columns).
    [[gnu::always_inline]] static V mul(V x, const Factor& f) {
        if constexpr (requires { S::mul_even(x, x); }) {
            const V even = S::mul_even(x, f.q), odd = S::mul_even(S::template shr64<32>(x), f.q);
            V qx;  // floor(x q / 2^32), joined as simd's mulhi joins (simd/notes.md)
            if constexpr (simd::detail::kMulhiPermute) qx = S::join_high(even, odd);
            else qx = S::template blend<0xaaaa>(S::template shr64<32>(even), odd);
            return whole(S::mul_sub(S::mul_lo(x, f.w), qx, S::broadcast(kP)));
        } else {
            return whole(M::shoup(x, f.w, f.q));
        }
    }

    // x86: x, unchanged, through an empty asm statement. Without it GCC distributes a product
    // x w - q P over its uses (a + c -> (a - q P) + x w): 24 adds and subtracts per forward
    // radix-4 column instead of 19 (notes.md, issue #5). Not on Neon, where GCC fuses those adds
    // into mla/mls (products 1-2% slower with the statement).
    [[gnu::always_inline]] static V whole(V x) {
#if defined(__x86_64__)
        if constexpr (S::lanes > 1) asm("" : "+v"(x));
#endif
        return x;
    }
};

// Whether groups pipeline their columns (Butterflies::each2): groups k > 0, forward identity groups
// (k = 0), inverse identity groups. On L1 data (notes.md, issue #4 round 2), groups k > 0 ran
// faster on Neoverse V2 (forward 5%, inverse 9%), Zen 5 (11%, 7%) and Emerald Rapids (2%, 5%);
// identity groups on Neoverse V2 (1.5%, 8%) and Zen 5 (inverse 4%), while Emerald Rapids lost 1-3%.
struct Pipelined {
    bool groups, forward_identity, inverse_identity;
};
#if defined(__aarch64__)
inline constexpr Pipelined kPipelined = {true, true, true};
#elif defined(__znver4__) || defined(__znver5__)
inline constexpr Pipelined kPipelined = {true, false, true};
#else
inline constexpr Pipelined kPipelined = {true, false, false};
#endif

// Emerald Rapids ran 2^18 4-7% slower with the scale in the top groups (3 runs; 1 MiB arrays,
// its L2 is 2 MiB), and 2^19-2^25 within 1%: the scale stays at the top level there from 2^18.
#if defined(__AVX512F__) && !defined(__znver4__) && !defined(__znver5__)
inline constexpr int kFoldMaxLog = 17;
#else
inline constexpr int kFoldMaxLog = kMaxLog;
#endif

// Software prefetch in passes over data beyond the caches: per column, the line kPrefetch words
// ahead in each quarter (prefetchw where a pass writes lines it does not read), in groups on
// quarters of at least kPrefetchMin words and in radix-2 levels on halves of at least
// kPrefetchHalfMin words (0: off). Emerald Rapids (notes.md, issue #5 round 2): multiply 2^19-2^25
// 5-13% faster, forward 3-12%, inverse 2^21-2^25 1-7%; quarters of 128 KiB and more otherwise
// share their L2 sets in 2 MiB pages. Prefetching smaller quarters was slower (2^12-2^16 1-4%),
// and radix-2 halves of 2^19 words cost inverse/20 3%. Zen 5 ran multiply and forward 1-3% slower.
#if defined(__AVX512F__) && !defined(__znver4__) && !defined(__znver5__)
inline constexpr std::size_t kPrefetch = 512, kPrefetchMin = std::size_t(1) << 15, kPrefetchHalfMin = std::size_t(1) << 20;
#else
inline constexpr std::size_t kPrefetch = 0, kPrefetchMin = 0, kPrefetchHalfMin = 0;
#endif

// s r(i) for i < 8, r(i) the inverse twiddles, and their Shoup quotients: one table block. The
// top groups that take the scale (Butterflies::inverse_group_scaled) read their twiddles here.
struct alignas(64) ScaledBlock {
    u32 words[16];
};
constexpr ScaledBlock scaled_block(u32 s) {
    ScaledBlock b{};
    for (std::size_t i = 0; i < 8; ++i) {
        b.words[i] = multiply_mod(s, kStaticTables.words[1][slot(i)]);
        b.words[8 + i] = quotient(b.words[i]);
    }
    return b;
}
// Per length: [0] for leaf_scale (inverse), [1] for product_scale (products), [2] for values_scale
// (inverse from the values form).
inline constexpr auto kScaledRoots = [] {
    std::array<std::array<ScaledBlock, kMaxLog + 1>, 3> t{};
    for (int lg = kMinLog; lg <= kMaxLog; ++lg)
        t[0][lg] = scaled_block(leaf_scale(lg)), t[1][lg] = scaled_block(product_scale(lg)), t[2][lg] = scaled_block(values_scale(lg));
    return t;
}();

// Radix-4 butterflies on 4 quarters of q words, q a multiple of S::lanes. Group k at quarter q is
// a polynomial mod X^4 - r(k)^2, X = x^q; x = r(k), y = r(2k), z = r(2k + 1) (scalar.hpp).
// Each butterfly is two stages on a column of 4 vectors, first then second, so groups can
// pipeline them across columns (each2).
template <class S>
struct Butterflies : Arith<S> {
    using A = Arith<S>;
    using V = typename S::V;
    using F = typename A::Factor;
    using Column = V[4];
    using A::add, A::diff, A::low, A::low_diff, A::mul;

    // x < bound P -> < 2P.
    [[gnu::always_inline]] static V below2(V x, int bound) { return bound > 2 ? low(x) : x; }

    // Forward, in < kIn P (kIn = 1, 2 or 4), out < 4P. Inputs below 2P need no reduction (the first
    // level). Quarters t >= kLoad are zero (a first level from a short source), and their products
    // are not computed. First: a, b < 2P and c = x f2, d = x f3.
    template <int kIn = 4, int kLoad = 4>
    [[gnu::always_inline]] static void forward_first(Column& v, const F& x) {
        v[0] = below2(v[0], kIn), v[1] = below2(v[1], kIn);
        if constexpr (kLoad > 2) v[2] = mul(v[2], x);
        if constexpr (kLoad > 3) v[3] = mul(v[3], x);
    }
    [[gnu::always_inline]] static void forward_second(Column& v, const F& y, const F& z) {
        const V ac = low(add(v[0], v[2])), amc = low_diff(v[0], v[2]);
        const V yb = mul(add(v[1], v[3]), y), zd = mul(diff(v[1], v[3]), z);
        v[0] = add(ac, yb), v[1] = diff(ac, yb), v[2] = add(amc, zd), v[3] = diff(amc, zd);
    }
    template <int kIn = 4>
    [[gnu::always_inline]] static void forward4(Column& v, const F& x, const F& y, const F& z) {
        forward_first<kIn>(v, x), forward_second(v, y, z);
    }

    // x = y = 1. In < kIn P, out < 4P. First: a + c, a - c, b + d, z (b - d).
    template <int kIn = 4>
    [[gnu::always_inline]] static void forward_identity_first(Column& v, const F& z) {
        const V a = below2(v[0], kIn), b = below2(v[1], kIn), c = below2(v[2], kIn), d = below2(v[3], kIn);
        if constexpr (kIn == 1) v[0] = add(a, c), v[1] = S::sub(S::add(a, S::broadcast(kP)), c), v[2] = add(b, d);  // < 2P
        else v[0] = low(add(a, c)), v[1] = low_diff(a, c), v[2] = low(add(b, d));
        v[3] = mul(diff(b, d), z);
    }
    [[gnu::always_inline]] static void forward_identity_second(Column& v) {
        const V ac = v[0], amc = v[1], bd = v[2], zd = v[3];
        v[0] = add(ac, bd), v[1] = diff(ac, bd), v[2] = add(amc, zd), v[3] = diff(amc, zd);
    }

    // Inverse, in < 2P, out < 2P. First: a + b, c + d, y (a - b), z (c - d).
    [[gnu::always_inline]] static void inverse_first(Column& v, const F& y, const F& z) {
        const V ab = low(add(v[0], v[1])), cd = low(add(v[2], v[3]));
        const V amb = mul(diff(v[0], v[1]), y), cmd = mul(diff(v[2], v[3]), z);
        v[0] = ab, v[1] = cd, v[2] = amb, v[3] = cmd;
    }
    [[gnu::always_inline]] static void inverse_second(Column& v, const F& x) {
        const V ab = v[0], cd = v[1], amb = v[2], cmd = v[3];
        v[0] = low(add(ab, cd)), v[1] = low(add(amb, cmd)), v[2] = mul(diff(ab, cd), x), v[3] = mul(diff(amb, cmd), x);
    }
    [[gnu::always_inline]] static void inverse4(Column& v, const F& x, const F& y, const F& z) {
        inverse_first(v, y, z), inverse_second(v, x);
    }

    // x = y = 1.
    [[gnu::always_inline]] static void inverse_identity_first(Column& v, const F& z) {
        const V ab = low(add(v[0], v[1])), cd = low(add(v[2], v[3]));
        const V amb = low_diff(v[0], v[1]), cmd = mul(diff(v[2], v[3]), z);
        v[0] = ab, v[1] = cd, v[2] = amb, v[3] = cmd;
    }
    [[gnu::always_inline]] static void inverse_identity_second(Column& v) {
        const V ab = v[0], cd = v[1], amb = v[2], cmd = v[3];
        v[0] = low(add(ab, cd)), v[1] = low(add(amb, cmd)), v[2] = low_diff(ab, cd), v[3] = low_diff(amb, cmd);
    }

    // Quarters t < kLoad of a column; the others are zero.
    template <int kLoad = 4>
    [[gnu::always_inline]] static void load(const u32* s, std::size_t q, Column& v) {
        const V zero = S::broadcast(0);
        v[0] = kLoad > 0 ? S::load(s) : zero, v[1] = kLoad > 1 ? S::load(s + q) : zero;
        v[2] = kLoad > 2 ? S::load(s + 2 * q) : zero, v[3] = kLoad > 3 ? S::load(s + 3 * q) : zero;
    }
    // Quarters t < kStore of a column.
    template <int kStore = 4>
    [[gnu::always_inline]] static void store(u32* g, std::size_t q, const Column& v) {
        if constexpr (kStore > 0) S::store(g, v[0]);
        if constexpr (kStore > 1) S::store(g + q, v[1]);
        if constexpr (kStore > 2) S::store(g + 2 * q, v[2]);
        if constexpr (kStore > 3) S::store(g + 3 * q, v[3]);
    }
    // body(std::bool_constant<on>{}): on if a pass whose streams are q words apart prefetches
    // (min: kPrefetchMin or kPrefetchHalfMin).
    template <class Body>
    [[gnu::always_inline]] static void prefetching(std::size_t q, std::size_t min, Body body) {
        if (kPrefetch > 0 && q >= min) return body(std::bool_constant<kPrefetch != 0>{});
        body(std::false_type{});
    }
    template <bool kOn>
    [[gnu::always_inline]] static void ahead(const u32* s) {
        if constexpr (kOn) __builtin_prefetch(s + kPrefetch, 0, 3);
    }
    // Quarters t < kCount.
    template <bool kOn, int kCount = 4>
    [[gnu::always_inline]] static void ahead(const u32* s, std::size_t q) {
        if constexpr (kCount > 0) ahead<kOn>(s);
        if constexpr (kCount > 1) ahead<kOn>(s + q);
        if constexpr (kCount > 2) ahead<kOn>(s + 2 * q);
        if constexpr (kCount > 3) ahead<kOn>(s + 3 * q);
    }
    // For lines written, not read: prefetchw (x86; GCC emits it for __builtin_prefetch only with
    // -mprfchw, which the targets' flags lack).
    template <bool kOn>
    [[gnu::always_inline]] static void ahead_write(const u32* s) {
        if constexpr (kOn) {
#if defined(__x86_64__)
            asm volatile("prefetchw %0" : : "m"(*reinterpret_cast<const char*>(s + kPrefetch)));
#else
            __builtin_prefetch(s + kPrefetch, 1, 3);
#endif
        }
    }
    // The quarters t < kStore of f that a pass writes without reading: all of them if they are not
    // those of `from` (write: from != f), else those from kLoad on (`from` zero there).
    template <bool kOn, int kLoad = 4, int kStore = 4>
    [[gnu::always_inline]] static void ahead_write(bool write, const u32* f, std::size_t q) {
        if constexpr (kOn) {
            const auto quarter = [&] [[gnu::always_inline]] (int t) {
                if (t < kStore && (write || t >= kLoad)) ahead_write<kOn>(f + t * q);
            };
            quarter(0), quarter(1), quarter(2), quarter(3);
        }
    }

    // Applies op to the columns [0, words) of the 4 quarters of `from` (S::lanes words of each),
    // into those of f (may be from). Quarters t >= kLoad of `from` are zero, not read; quarters
    // t >= kStore of f are not written.
    template <int kLoad = 4, int kStore = 4, class Op>
    [[gnu::always_inline]] static void each(const u32* from, u32* f, std::size_t q, std::size_t words, Op op) {
        prefetching(q, kPrefetchMin, [&]<bool kPf> [[gnu::always_inline]] (std::bool_constant<kPf>) {
            const bool write = from != f;
            for (std::size_t j = 0; j < words; j += S::lanes) {
                Column v;
                ahead<kPf, kLoad>(from + j, q), ahead_write<kPf, kLoad, kStore>(write, f + j, q);
                load<kLoad>(from + j, q, v), op(v), store<kStore>(f + j, q, v);
            }
        });
    }
    template <class Op>
    [[gnu::always_inline]] static void each(const u32* from, u32* f, std::size_t q, Op op) {
        each(from, f, q, q, op);
    }

    // The same for an op in two stages, software pipelined if kPipe: the first stage of column
    // j + 1 runs before the second of column j, so a column's two halves are not in flight at once
    // (shorter chains per loop iteration). Unrolled by 2, so no register copies: words a
    // multiple of 2 S::lanes (every caller: quarters q >= 32, or q >= 16 at the top of the shortest
    // transforms; first and last levels in steps of 2 S::lanes).
    template <bool kPipe, int kLoad = 4, int kStore = 4, class First, class Second>
    [[gnu::always_inline]] static void each2(const u32* from, u32* f, std::size_t q, std::size_t words, First first,
                                             Second second) {
        constexpr std::size_t L = S::lanes;
        if constexpr (!kPipe) {
            each<kLoad, kStore>(from, f, q, words, [&] [[gnu::always_inline]] (Column& v) { first(v), second(v); });
        } else {
            prefetching(q, kPrefetchMin, [&]<bool kPf> [[gnu::always_inline]] (std::bool_constant<kPf>) {
                const bool write = from != f;
                Column h0, h1;
                load<kLoad>(from, q, h0), first(h0);
                std::size_t j = L;
                for (; j + L < words; j += 2 * L) {
                    ahead<kPf, kLoad>(from + j, q), ahead_write<kPf, kLoad, kStore>(write, f + j, q);
                    load<kLoad>(from + j, q, h1), first(h1);
                    second(h0), store<kStore>(f + j - L, q, h0);
                    ahead<kPf, kLoad>(from + j + L, q), ahead_write<kPf, kLoad, kStore>(write, f + j + L, q);
                    load<kLoad>(from + j + L, q, h0), first(h0);
                    second(h1), store<kStore>(f + j, q, h1);
                }
                load<kLoad>(from + j, q, h1), first(h1);
                second(h0), store<kStore>(f + j - L, q, h0);
                second(h1), store<kStore>(f + j, q, h1);
            });
        }
    }
    template <bool kPipe, class First, class Second>
    [[gnu::always_inline]] static void each2(const u32* from, u32* f, std::size_t q, First first, Second second) {
        each2<kPipe>(from, f, q, q, first, second);
    }

    // Group k of `from` into f (may be from). In < kIn P, out < 4P.
    template <int kIn = 4>
    static void forward_group(const u32* from, u32* f, std::size_t q, std::size_t k, const u32* r) {
        if constexpr (kIn == 4 && requires { Generated<S>::kShareGroups; })
            if (from == f && q % 64 == 0) return forward_groups(f, q, k, 1, r);
        forward_group_inline<kIn>(from, f, q, k, r);
    }
    template <int kIn = 4>
    static void forward_group(u32* f, std::size_t q, std::size_t k, const u32* r) { forward_group<kIn>(f, f, q, k, r); }
    template <int kIn>
    [[gnu::always_inline]] static void forward_group_inline(const u32* from, u32* f, std::size_t q, std::size_t k,
                                                            const u32* r) {
        forward_group_part<kIn, 4>(from, f, q, q, k, r);
    }
    // Columns [0, words) of group k, from the quarters t < kLoad of `from` (the others are zero, not
    // read). In < kIn P, out < 4P. With one quarter every output is its input.
    template <int kIn, int kLoad>
    [[gnu::always_inline]] static void forward_group_part(const u32* from, u32* f, std::size_t q, std::size_t words,
                                                          std::size_t k, const u32* r) {
        if constexpr (Generated<S>::kOn && kLoad == 4)
            if (from == f && Generated<S>::template forward<kIn>(f, q, words, k, r)) return;
        if constexpr (kLoad == 4 && requires { Generated<S>::template forward_from<kIn>(from, f, q, words, k, r); })
            if (from != f && Generated<S>::template forward_from<kIn>(from, f, q, words, k, r)) return;
        if constexpr (kLoad <= 1) {
            prefetching(q, kPrefetchMin, [&]<bool kPf> [[gnu::always_inline]] (std::bool_constant<kPf>) {
                const bool write = from != f;
                for (std::size_t j = 0; j < words; j += S::lanes) {
                    ahead<kPf, kLoad>(from + j, q), ahead_write<kPf, kLoad>(write, f + j, q);
                    const V a = kLoad ? S::load(from + j) : S::broadcast(0);
                    S::store(f + j, a), S::store(f + j + q, a), S::store(f + j + 2 * q, a), S::store(f + j + 3 * q, a);
                }
            });
        } else if (k == 0) {
            const F z = A::entry(r, 1);
            each2<kPipelined.forward_identity, kLoad>(
                from, f, q, words, [&] [[gnu::always_inline]] (Column& v) { forward_identity_first<kIn>(v, z); },
                [&] [[gnu::always_inline]] (Column& v) { forward_identity_second(v); });
        } else {
            const F x = A::entry(r, k), y = A::entry(r, 2 * k), z = A::entry(r, 2 * k + 1);
            each2<kPipelined.groups, kLoad>(
                from, f, q, words, [&] [[gnu::always_inline]] (Column& v) { forward_first<kIn, kLoad>(v, x); },
                [&] [[gnu::always_inline]] (Column& v) { forward_second(v, y, z); });
        }
    }

    static void inverse_group(u32* f, std::size_t q, std::size_t k, const u32* ir) {
        if constexpr (requires { Generated<S>::kShareGroups; })
            if (q % 64 == 0) return inverse_groups(f, q, k, 1, ir);
        inverse_group_inline(f, q, k, ir);
    }
    [[gnu::always_inline]] static void inverse_group_inline(u32* f, std::size_t q, std::size_t k, const u32* ir) {
        if constexpr (requires { Generated<S>::inverse(f, q, k, ir); })
            if (Generated<S>::inverse(f, q, k, ir)) return;
        if (k == 0) {
            const F z = A::entry(ir, 1);
            return each2<kPipelined.inverse_identity>(
                f, f, q, [&] [[gnu::always_inline]] (Column& v) { inverse_identity_first(v, z); },
                [&] [[gnu::always_inline]] (Column& v) { inverse_identity_second(v); });
        }
        const F x = A::entry(ir, k), y = A::entry(ir, 2 * k), z = A::entry(ir, 2 * k + 1);
        each2<kPipelined.groups>(f, f, q, [&] [[gnu::always_inline]] (Column& v) { inverse_first(v, y, z); },
                                 [&] [[gnu::always_inline]] (Column& v) { inverse_second(v, x); });
    }

    // Pruned groups (Transform's hooked ops, issue #149): group k on the columns j < q with (j &
    // width) == 0, chunks of width words. width a multiple of S::lanes, q of 2 width. Chunks of one
    // vector (issue #149 round 2) one column at a time.
    template <int kIn = 4>
    static void forward_group_columns(u32* f, std::size_t q, std::size_t width, std::size_t k, const u32* r) {
        if (width < 2 * S::lanes) {
            const F x = A::entry(r, k), y = A::entry(r, 2 * k), z = A::entry(r, 2 * k + 1);
            for (std::size_t c = 0; c < q; c += 2 * width)
                each(f + c, f + c, q, width, [&] [[gnu::always_inline]] (Column& v) { forward4<kIn>(v, x, y, z); });
            return;
        }
        for (std::size_t c = 0; c < q; c += 2 * width) forward_group_part<kIn, 4>(f + c, f + c, q, width, k, r);
    }
    static void inverse_group_columns(u32* f, std::size_t q, std::size_t width, std::size_t k, const u32* ir) {
        if (width < 2 * S::lanes) {
            const F x = A::entry(ir, k), y = A::entry(ir, 2 * k), z = A::entry(ir, 2 * k + 1);
            for (std::size_t c = 0; c < q; c += 2 * width)
                each(f + c, f + c, q, width, [&] [[gnu::always_inline]] (Column& v) { inverse4(v, x, y, z); });
            return;
        }
        if (k == 0) {
            const F z = A::entry(ir, 1);
            for (std::size_t c = 0; c < q; c += 2 * width)
                each2<kPipelined.inverse_identity>(
                    f + c, f + c, q, width, [&] [[gnu::always_inline]] (Column& v) { inverse_identity_first(v, z); },
                    [&] [[gnu::always_inline]] (Column& v) { inverse_identity_second(v); });
            return;
        }
        const F x = A::entry(ir, k), y = A::entry(ir, 2 * k), z = A::entry(ir, 2 * k + 1);
        for (std::size_t c = 0; c < q; c += 2 * width)
            each2<kPipelined.groups>(f + c, f + c, q, width, [&] [[gnu::always_inline]] (Column& v) { inverse_first(v, y, z); },
                                     [&] [[gnu::always_inline]] (Column& v) { inverse_second(v, x); });
    }

    // The group whose quarter index holds the bit (issue #149 round 2, as avx2.hpp's): quarters 2
    // and 3 (high) or 1 and 3 zero in the forward, not read; the inverse's outputs at quarters 0 and
    // 1 (high) or 0 and 2 only. In and out as forward_group and inverse_group.
    template <int kIn = 4>
    static void forward_group_half(u32* f, std::size_t q, std::size_t k, const u32* r, bool high) {
        if (high) return forward_group_part<kIn, 2>(f, f, q, q, k, r);
        const F x = A::entry(r, k);
        for (std::size_t j = 0; j < q; j += S::lanes) {  // A, A, C, C for A, C = f0 +- x f2
            const V a = below2(S::load(f + j), kIn), c = mul(S::load(f + j + 2 * q), x), s = add(a, c), d = diff(a, c);
            S::store(f + j, s), S::store(f + j + q, s), S::store(f + j + 2 * q, d), S::store(f + j + 3 * q, d);
        }
    }
    static void inverse_group_half(u32* f, std::size_t q, std::size_t k, const u32* ir, bool high) {
        const F x = A::entry(ir, k);
        if (high) {
            const F y = A::entry(ir, 2 * k), z = A::entry(ir, 2 * k + 1);
            return each<4, 2>(f, f, q, q, [&] [[gnu::always_inline]] (Column& v) { inverse_first(v, y, z), inverse_second(v, x); });
        }
        for (std::size_t j = 0; j < q; j += S::lanes) {
            const V ab = low(add(S::load(f + j), S::load(f + j + q))), cd = low(add(S::load(f + j + 2 * q), S::load(f + j + 3 * q)));
            S::store(f + j, low(add(ab, cd))), S::store(f + j + 2 * q, mul(diff(ab, cd), x));
        }
    }

    // Groups k, k + 1, ..., k + count - 1 of a level, group k + i at f + 4 q i, in one call: a call
    // per group cost the level of quarters of 32 words 18% on Emerald Rapids (notes.md). In < 4P
    // (forward), < 2P (inverse).
    static void forward_groups(u32* f, std::size_t q, std::size_t k, std::size_t count, const u32* r) {
        if constexpr (requires { Generated<S>::forward_groups(f, q, k, count, r); })
            if (Generated<S>::forward_groups(f, q, k, count, r)) return;
        const std::size_t end = k + count;
        if (q == 2 * S::lanes) {  // two columns per group
            for (; k < end; ++k, f += 4 * q) {
                if (k == 0) {
                    forward_group_inline<4>(f, f, q, k, r);
                    continue;
                }
                const F x = A::entry(r, k), y = A::entry(r, 2 * k), z = A::entry(r, 2 * k + 1);
                Column c0, c1;
                load(f, q, c0), load(f + S::lanes, q, c1);
                forward_first<4>(c0, x), forward_first<4>(c1, x);
                forward_second(c0, y, z), forward_second(c1, y, z);
                store(f, q, c0), store(f + S::lanes, q, c1);
            }
            return;
        }
        for (; k < end; ++k, f += 4 * q) forward_group_inline<4>(f, f, q, k, r);
    }
    static void inverse_groups(u32* f, std::size_t q, std::size_t k, std::size_t count, const u32* ir) {
        const std::size_t end = k + count;
        if (q == 2 * S::lanes) {
            for (; k < end; ++k, f += 4 * q) {
                if (k == 0) {
                    inverse_group_inline(f, q, k, ir);
                    continue;
                }
                const F x = A::entry(ir, k), y = A::entry(ir, 2 * k), z = A::entry(ir, 2 * k + 1);
                Column c0, c1;
                load(f, q, c0), load(f + S::lanes, q, c1);
                inverse_first(c0, y, z), inverse_first(c1, y, z);
                inverse_second(c0, x), inverse_second(c1, x);
                store(f, q, c0), store(f + S::lanes, q, c1);
            }
            return;
        }
        for (; k < end; ++k, f += 4 * q) inverse_group_inline(f, q, k, ir);
    }

    // inverse_group with every output times s, k < 4: twiddles s y, s z, s x and one product more
    // for the first output (5 products per column; 4 for k = 0, against 1). scaled: scaled_block(s).
    // In < 2P, out < 2P.
    static void inverse_group_scaled(u32* f, std::size_t q, std::size_t k, const u32* ir, const u32* scaled) {
        if constexpr (requires { Generated<S>::inverse_scaled(f, q, k, ir, scaled); })
            if (Generated<S>::inverse_scaled(f, q, k, ir, scaled)) return;
        const F scale = A::entry(scaled, 0), sx = A::entry(scaled, k), sy = A::entry(scaled, 2 * k),
                sz = A::entry(scaled, 2 * k + 1);
        const auto first = [&] [[gnu::always_inline]] (Column& v) { inverse_first(v, sy, sz); };
        if (k == 0) {
            return each2<kPipelined.groups>(f, f, q, first, [&] [[gnu::always_inline]] (Column& v) {
                const V ab = v[0], cd = v[1], amb = v[2], cmd = v[3];
                v[0] = mul(add(ab, cd), scale), v[1] = low(add(amb, cmd)), v[2] = mul(diff(ab, cd), scale);
                v[3] = low_diff(amb, cmd);
            });
        }
        const F x = A::entry(ir, k);
        each2<kPipelined.groups>(f, f, q, first, [&] [[gnu::always_inline]] (Column& v) {
            const V ab = v[0], cd = v[1], amb = v[2], cmd = v[3];
            v[0] = mul(add(ab, cd), scale), v[1] = low(add(amb, cmd)), v[2] = mul(diff(ab, cd), sx);
            v[3] = mul(diff(amb, cmd), x);
        });
    }
};

// The bottom level (quarters of 8 words, one leaf each) and the leaf products, on L (<= 8 lanes).
template <class L>
struct Bottom : Butterflies<L> {
    static_assert(8 % L::lanes == 0);
    using B = Butterflies<L>;
    using V = typename L::V;
    using F = typename B::Factor;
    using M = typename B::M;
    using B::canonical, B::low;

    static constexpr u32 kNegInverse = u32(0) - M::kPinv;  // -P^-1 mod 2^32

    // The bottom level runs kColumns columns of a group per step (both on Neon and Sse): bottom
    // forward 10% and inverse 8% faster on Neoverse V2 than one at a time (notes.md).
    static constexpr int kColumns = 8 / L::lanes >= 2 ? 2 : 1;
    using Column = typename B::Column;

    // Applies op to the columns of each group [first, first + count), 32 words each.
    template <class Op>
    [[gnu::always_inline]] static void each_group(u32* f, std::size_t count, std::size_t first, const u32* table, Op op) {
        for (std::size_t k = first; k < first + count; ++k, f += 32) {
            const F x = B::entry(table, k), y = B::entry(table, 2 * k), z = B::entry(table, 2 * k + 1);
            for (std::size_t j = 0; j < 8; j += kColumns * L::lanes) {
                Column v[kColumns];
                #pragma GCC unroll 8
                for (int c = 0; c < kColumns; ++c) B::load(f + j + c * L::lanes, 8, v[c]);
                #pragma GCC unroll 8
                for (int c = 0; c < kColumns; ++c) op(v[c], x, y, z);
                #pragma GCC unroll 8
                for (int c = 0; c < kColumns; ++c) B::store(f + j + c * L::lanes, 8, v[c]);
            }
        }
    }

    // Groups [first, first + count), in < 4P; canonical leaves out.
    static void forward(u32* f, std::size_t count, std::size_t first, const u32* r) {
        each_group(f, count, first, r, [] [[gnu::always_inline]] (Column& v, const F& x, const F& y, const F& z) {
            B::forward4(v, x, y, z);
            v[0] = canonical(low(v[0])), v[1] = canonical(low(v[1])), v[2] = canonical(low(v[2])), v[3] = canonical(low(v[3]));
        });
    }

    // In and out < 2P.
    static void inverse(u32* f, std::size_t count, std::size_t first, const u32* ir) {
        each_group(f, count, first, ir, [] [[gnu::always_inline]] (Column& v, const F& x, const F& y, const F& z) {
            B::inverse4(v, x, y, z);
        });
    }

    // One group's leaves for the products. Row t of window is [w_t A_t, A_t], canonical: x^i A_t
    // mod x^8 - w_t is the 8 words from 8 - i. The leaf products read 1 word past a row.
    struct alignas(64) Leaves {
        u32 window[4][16];
        u32 b[4][8];
    };

    // Forward butterflies of group k of a and b into leaves (of a alone if kSquare: its rows are
    // both factors). The moduli x^8 - w_t of leaves 4k + t: w_t = y, -y, z, -z (quotient(P - w) =
    // ~quotient(w)).
    template <bool kSquare>
    [[gnu::always_inline]] static void forward_leaves(const u32* a, const u32* b, Leaves& out, std::size_t k, const u32* r) {
        const F x = B::entry(r, k), y = B::entry(r, 2 * k), z = B::entry(r, 2 * k + 1);
        const u32* yz = r + slot(2 * k);
        const F w[4] = {B::factor(yz[0], yz[8]), B::factor(kP - yz[0], ~yz[8]), B::factor(yz[1], yz[9]),
                        B::factor(kP - yz[1], ~yz[9])};
        #pragma GCC unroll 8
        for (std::size_t j = 0; j < 8; j += L::lanes) {
            Column f;
            B::load(a + j, 8, f), B::forward4(f, x, y, z);
            #pragma GCC unroll 8
            for (int t = 0; t < 4; ++t) {
                const V v = canonical(low(f[t]));
                L::store(out.window[t] + 8 + j, v);
                L::store(out.window[t] + j, canonical(B::mul(v, w[t])));
            }
            if constexpr (kSquare) continue;
            Column g;
            B::load(b + j, 8, g), B::forward4(g, x, y, z);
            #pragma GCC unroll 8
            for (int t = 0; t < 4; ++t) L::store(out.b[t] + j, canonical(low(g[t])));
        }
    }

    // Words [j, j + lanes) of A B 2^-32 mod x^8 - w, < 2P: word l is the sum over i of
    // b_i window[8 - i + l] (< 8 P^2 < 2^63), then a Montgomery reduction (< 2^64 before the shift).
    [[gnu::always_inline]] static V leaf_product(const u32* window, const u32* b, std::size_t j) {
        const V p = L::broadcast(kP), ninv = L::broadcast(kNegInverse);
        if constexpr (requires(V v) { L::mul_even(v, v); }) {
            // 64-bit lanes: e gets the even words, o the odd ones (from loads one word on).
            V e = L::mul_even(L::load(window + 8 + j), L::broadcast(b[0]));
            V o = L::mul_even(L::load(window + 9 + j), L::broadcast(b[0]));
            #pragma GCC unroll 8
            for (int i = 1; i < 8; ++i) {
                const V bi = L::broadcast(b[i]);
                e = L::add64(e, L::mul_even(L::load(window + 8 - i + j), bi));
                o = L::add64(o, L::mul_even(L::load(window + 9 - i + j), bi));
            }
            e = L::add64(e, L::mul_even(L::mul_even(e, ninv), p));
            o = L::add64(o, L::mul_even(L::mul_even(o, ninv), p));
            return low(L::join_high(e, o));
        } else {
            static_assert(L::lanes == 1);
            u64 s = 0;
            #pragma GCC unroll 8
            for (int i = 0; i < 8; ++i) s += u64(window[8 - i + j]) * b[i];
            const u32 m = u32(s) * kNegInverse;
            return low(V((s + u64(m) * kP) >> 32));
        }
    }

    // c[u][t]: words [u lanes, (u + 1) lanes) of leaf t's product, as leaf_product. Neon: two
    // leaves at once, all their words (umull/umlal and their high forms), each b_i broadcast once
    // for both words, so 8 independent sums: bottom multiply 11% faster on Neoverse V2 than one
    // leaf and word at a time; 4 leaves at once spilled the 16 sums (notes.md).
    // window: the rows of Leaves::window; B_t at b + stride t (B's 32 words: stride 8; a square:
    // B_t is A_t, the second half of window row t, so b = window[0] + 8 and stride 16).
    static constexpr int kWords = 8 / L::lanes;
    [[gnu::always_inline]] static void leaf_products(const u32 (&window)[4][16], const u32* b, std::size_t stride,
                                                     V (&c)[kWords][4]) {
        if constexpr (requires(V v) { L::mul_wide_lo(v, v); }) {
            using W = decltype(L::mul_wide_lo(V{}, V{}));
            const V p = L::broadcast(kP), ninv = L::broadcast(kNegInverse);
            constexpr int kBlock = 2;
            #pragma GCC unroll 8
            for (int t0 = 0; t0 < 4; t0 += kBlock) {
                W lo[kWords][kBlock], hi[kWords][kBlock];
                #pragma GCC unroll 8
                for (int i = 0; i < 8; ++i) {
                    #pragma GCC unroll 8
                    for (int bb = 0; bb < kBlock; ++bb) {
                        const V bi = L::broadcast(b[stride * std::size_t(t0 + bb) + std::size_t(i)]);
                        #pragma GCC unroll 8
                        for (int u = 0; u < kWords; ++u) {
                            const V x = L::load(window[t0 + bb] + 8 - i + u * L::lanes);
                            if (i == 0) lo[u][bb] = L::mul_wide_lo(x, bi), hi[u][bb] = L::mul_wide_hi(x, bi);
                            else lo[u][bb] = L::mla_wide_lo(lo[u][bb], x, bi), hi[u][bb] = L::mla_wide_hi(hi[u][bb], x, bi);
                        }
                    }
                }
                #pragma GCC unroll 8
                for (int u = 0; u < kWords; ++u) {
                    #pragma GCC unroll 8
                    for (int bb = 0; bb < kBlock; ++bb) {
                        const V m = L::mul_lo(L::lo_halves(lo[u][bb], hi[u][bb]), ninv);
                        const W l = L::mla_wide_lo(lo[u][bb], m, p), h = L::mla_wide_hi(hi[u][bb], m, p);
                        c[u][t0 + bb] = low(L::hi_halves(l, h));
                    }
                }
            }
        } else {
            #pragma GCC unroll 8
            for (int u = 0; u < kWords; ++u) {
                #pragma GCC unroll 8
                for (int t = 0; t < 4; ++t) c[u][t] = leaf_product(window[t], b + stride * std::size_t(t), u * L::lanes);
            }
        }
    }

    // Leaf products of one group into a, then its inverse butterflies; out < 2P.
    template <bool kSquare>
    [[gnu::always_inline]] static void multiply_leaves(u32* a, const Leaves& in, std::size_t k, const u32* ir) {
        const F x = B::entry(ir, k), y = B::entry(ir, 2 * k), z = B::entry(ir, 2 * k + 1);
        V c[kWords][4];
        if constexpr (kSquare) leaf_products(in.window, in.window[0] + 8, 16, c);
        else leaf_products(in.window, in.b[0], 8, c);
        #pragma GCC unroll 8
        for (int u = 0; u < kWords; ++u) B::inverse4(c[u], x, y, z), B::store(a + u * L::lanes, 8, c[u]);
    }

    // Groups [first, first + count) of a and b, in < 4P: forward butterflies, leaf products into
    // a, inverse butterflies; out < 2P. Group g + 1's forward half runs before group g's
    // products, so the leaf rows are not loaded right after their stores. kSquare: a times a, b
    // not read.
    template <bool kSquare = false>
    static void multiply(u32* a, const u32* b, std::size_t count, std::size_t first, const u32* r, const u32* ir) {
        Leaves leaves[2];
        forward_leaves<kSquare>(a, b, leaves[0], first, r);
        for (std::size_t g = 0; g < count; ++g) {
            if (g + 1 < count)
                forward_leaves<kSquare>(a + 32 * (g + 1), b + 32 * (g + 1), leaves[(g + 1) % 2], first + g + 1, r);
            multiply_leaves<kSquare>(a + 32 * g, leaves[g % 2], first + g, ir);
        }
    }

    // Transform's fused ops (ntt.hpp): bottoms with a factor stored as leaves.

    // Forward butterflies of group k of a into rows [w_t A_t, A_t] (forward_leaves of a square).
    [[gnu::always_inline]] static void forward_rows(const u32* a, Leaves& out, std::size_t k, const u32* r) {
        const F x = B::entry(r, k), y = B::entry(r, 2 * k), z = B::entry(r, 2 * k + 1);
        const u32* yz = r + slot(2 * k);
        const F w[4] = {B::factor(yz[0], yz[8]), B::factor(kP - yz[0], ~yz[8]), B::factor(yz[1], yz[9]),
                        B::factor(kP - yz[1], ~yz[9])};
        #pragma GCC unroll 8
        for (std::size_t j = 0; j < 8; j += L::lanes) {
            Column f;
            B::load(a + j, 8, f), B::forward4(f, x, y, z);
            #pragma GCC unroll 8
            for (int t = 0; t < 4; ++t) {
                const V v = canonical(low(f[t]));
                L::store(out.window[t] + 8 + j, v);
                L::store(out.window[t] + j, canonical(B::mul(v, w[t])));
            }
        }
    }

    // Groups [first, first + count) of a, in < 4P, times b's leaves (canonical, at b + 32 (g -
    // first)): forward butterflies, leaf products, inverse butterflies into a, out < 2P, with the
    // factor 2^-32.
    static void multiply_stored(u32* a, const u32* b, std::size_t count, std::size_t first, const u32* r, const u32* ir) {
        Leaves leaves[2];
        forward_rows(a, leaves[0], first, r);
        for (std::size_t g = 0; g < count; ++g) {
            if (g + 1 < count) forward_rows(a + 32 * (g + 1), leaves[(g + 1) % 2], first + g + 1, r);
            const std::size_t k = first + g;
            const F x = B::entry(ir, k), y = B::entry(ir, 2 * k), z = B::entry(ir, 2 * k + 1);
            V c[kWords][4];
            leaf_products(leaves[g % 2].window, b + 32 * g, 8, c);
            #pragma GCC unroll 8
            for (int u = 0; u < kWords; ++u) B::inverse4(c[u], x, y, z), B::store(a + 32 * g + u * L::lanes, 8, c[u]);
        }
    }

    // The leaves of groups [first, first + count) into out (group g at out + 32 (g - first)): the
    // sum over the terms (common.hpp, LeafTerm; at a + 32 g, b + 32 g) of their leaf products, or of
    // a times 2^-32; then the inverse butterflies, out < 2P, with the factor 2^-32. Group g + 1's
    // rows are filled before group g's products.
    static void inverse_terms(u32* out, const LeafTerm* terms, int n, std::size_t count, std::size_t first, const u32* r,
                              const u32* ir) {
        static constexpr u32 kRinverse = power(kR, kP - 2);  // 2^-32 mod P
        const F to_montgomery = B::factor(kRinverse);
        Rows rows[2][kMaxTerms];
        const auto fill = [&](std::size_t k, Rows* to) {
            for (int i = 0; i < n; ++i)
                if (terms[i].b) fill_rows(terms[i].a + 32 * k, to[i], k, r);
        };
        fill(first, rows[0]);
        for (std::size_t g = 0; g < count; ++g) {
            const std::size_t k = first + g;
            if (g + 1 < count) fill(k + 1, rows[(g + 1) % 2]);
            V c[kWords][4], d[kWords][4];
            for (int i = 0; i < n; ++i) {
                if (terms[i].b) {
                    leaf_products(rows[g % 2][i].window, terms[i].b + 32 * k, 8, d);
                } else {
                    #pragma GCC unroll 8
                    for (int u = 0; u < kWords; ++u) {
                        #pragma GCC unroll 8
                        for (int t = 0; t < 4; ++t) d[u][t] = B::mul(L::load(terms[i].a + 32 * k + 8 * t + u * L::lanes), to_montgomery);
                    }
                }
                #pragma GCC unroll 8
                for (int u = 0; u < kWords; ++u) {
                    #pragma GCC unroll 8
                    for (int t = 0; t < 4; ++t) c[u][t] = i == 0 ? d[u][t] : low(B::add(c[u][t], d[u][t]));
                }
            }
            const F x = B::entry(ir, k), y = B::entry(ir, 2 * k), z = B::entry(ir, 2 * k + 1);
            #pragma GCC unroll 8
            for (int u = 0; u < kWords; ++u) B::inverse4(c[u], x, y, z), B::store(out + 32 * g + u * L::lanes, 8, c[u]);
        }
    }

    // Transform::toeplitz_product's bottom on groups [first, first + count) (out: group g at out + 32
    // (g - first); the leaf forms of z at + 32 g): rows of S = B0 + B1, D1 = A1 - A2 and D3 = A3 - A2,
    // m1 = S A2, m2 = D1 B1, m3 = D3 B0; m1 + m2 through the inverse butterflies (out < 2P, with the
    // factor 2^-32), m1 + m3 times 2^32 over A3, canonical.
    static void inverse_toeplitz(u32* out, const Toeplitz& z, std::size_t count, std::size_t first, const u32* r,
                                 const u32* ir) {
        using M = typename B::M;
        const F to_r = B::factor(kR);
        alignas(64) u32 sum[3][32];
        Rows rows[3];
        for (std::size_t g = 0; g < count; ++g) {
            const std::size_t k = first + g, at = 32 * k;
            #pragma GCC unroll 8
            for (std::size_t i = 0; i < 32; i += L::lanes) {
                const V a2 = L::load(z.a2 + at + i);
                L::store(sum[0] + i, M::add(L::load(z.b0 + at + i), L::load(z.b1 + at + i)));
                L::store(sum[1] + i, M::sub(L::load(z.a1 + at + i), a2));
                L::store(sum[2] + i, M::sub(L::load(z.a3 + at + i), a2));
            }
            for (int i = 0; i < 3; ++i) fill_rows(sum[i], rows[i], k, r);
            V m1[kWords][4], m2[kWords][4], m3[kWords][4];
            leaf_products(rows[0].window, z.a2 + at, 8, m1);
            leaf_products(rows[1].window, z.b1 + at, 8, m2);
            leaf_products(rows[2].window, z.b0 + at, 8, m3);
            const F x = B::entry(ir, k), y = B::entry(ir, 2 * k), w = B::entry(ir, 2 * k + 1);
            #pragma GCC unroll 8
            for (int u = 0; u < kWords; ++u) {
                #pragma GCC unroll 8
                for (int t = 0; t < 4; ++t) {
                    L::store(z.a3 + at + 8 * t + u * L::lanes, canonical(B::mul(B::add(m1[u][t], m3[u][t]), to_r)));
                    m1[u][t] = low(B::add(m1[u][t], m2[u][t]));
                }
                B::inverse4(m1[u], x, y, w), B::store(out + 32 * g + u * L::lanes, 8, m1[u]);
            }
        }
    }

    // Leaf products of leaf forms (Transform::multiply): rows [w_t A_t, A_t] of group k from
    // canonical a, as forward_leaves; the products read 1 word past the last row.
    struct alignas(64) Rows {
        u32 window[4][16];
        u32 past[16];
    };
    [[gnu::always_inline]] static void fill_rows(const u32* a, Rows& out, std::size_t k, const u32* r) {
        const u32* yz = r + slot(2 * k);
        const F w[4] = {B::factor(yz[0], yz[8]), B::factor(kP - yz[0], ~yz[8]), B::factor(yz[1], yz[9]),
                        B::factor(kP - yz[1], ~yz[9])};
        #pragma GCC unroll 8
        for (std::size_t j = 0; j < 8; j += L::lanes) {
            #pragma GCC unroll 8
            for (int t = 0; t < 4; ++t) {
                const V v = L::load(a + 8 * t + j);
                L::store(out.window[t] + 8 + j, v);
                L::store(out.window[t] + j, canonical(B::mul(v, w[t])));
            }
        }
    }

    // Groups [first, first + count) of canonical leaf forms a, b: out = a b leaf by leaf, or
    // out += a b if kAdd (out canonical); canonical out. out may be a or b: group g is written
    // after its products, and group g + 1's rows are filled before. The leaf products carry
    // 2^-32 (Montgomery), undone by a Shoup product by 2^32 mod P.
    // Words [j, j + lanes) of A B mod x^8 - w, exact, < 2P (x86): each 64-bit sum s < 8 P^2 is
    // hi 2^32 + lo, and s = hi (2^32 mod P) + lo mod P, one Shoup product: 4 multiplies a vector,
    // against 4 for Montgomery's reduction and 4 more to undo its 2^-32.
    [[gnu::always_inline]] static V leaf_product_exact(const u32* window, const u32* b, std::size_t j, const F& to_r) {
        V e = L::mul_even(L::load(window + 8 + j), L::broadcast(b[0]));
        V o = L::mul_even(L::load(window + 9 + j), L::broadcast(b[0]));
        #pragma GCC unroll 8
        for (int i = 1; i < 8; ++i) {
            const V bi = L::broadcast(b[i]);
            e = L::add64(e, L::mul_even(L::load(window + 8 - i + j), bi));
            o = L::add64(o, L::mul_even(L::load(window + 9 - i + j), bi));
        }
        const V hi = L::join_high(e, o), lo = L::template blend<0xaaaa>(e, L::template shl64<32>(o));
        const V lo2 = low(L::min(lo, L::sub(lo, L::broadcast(4 * kP))));  // lo < 2^32 < 8P -> [0, 2P)
        return low(B::add(lo2, B::mul(hi, to_r)));
    }

    template <bool kAdd>
    static void products(const u32* a, const u32* b, u32* out, std::size_t count, std::size_t first, const u32* r) {
        const F to_r = B::factor(kR);
        Rows rows[2];
        fill_rows(a, rows[0], first, r);
        for (std::size_t g = 0; g < count; ++g) {
            if (g + 1 < count) fill_rows(a + 32 * (g + 1), rows[(g + 1) % 2], first + g + 1, r);
            V c[kWords][4];
            if constexpr (requires(V v) { L::mul_even(v, v); }) {
                #pragma GCC unroll 8
                for (int u = 0; u < kWords; ++u) {
                    #pragma GCC unroll 8
                    for (int t = 0; t < 4; ++t) c[u][t] = leaf_product_exact(rows[g % 2].window[t], b + 32 * g + 8 * t, u * L::lanes, to_r);
                }
            } else {
                leaf_products(rows[g % 2].window, b + 32 * g, 8, c);
                #pragma GCC unroll 8
                for (int u = 0; u < kWords; ++u) {
                    #pragma GCC unroll 8
                    for (int t = 0; t < 4; ++t) c[u][t] = B::mul(c[u][t], to_r);
                }
            }
            #pragma GCC unroll 8
            for (int u = 0; u < kWords; ++u) {
                #pragma GCC unroll 8
                for (int t = 0; t < 4; ++t) {
                    u32* o = out + 32 * g + 8 * t + u * L::lanes;
                    L::store(o, kAdd ? canonical(low(B::add(L::load(o), c[u][t]))) : canonical(c[u][t]));
                }
            }
        }
    }
};

// The leaves of groups [first, first + count) at out (group g at out + 32 (g - first)) times b's
// (at b + 32 (g - first)), exactly, canonical: a tile hook of Transform::forward_product, on a run
// of subtrees in cache.
template <class L>
[[gnu::always_inline]] inline void leaf_products_at(u32* out, const u32* b, std::size_t count, std::size_t first,
                                                    const u32* r) {
    Bottom<L>::template products<false>(out, b, out, count, first, r);
}

// The bottom level on S with 16 lanes (zmm), two groups k, k + 1 (k even) at a time: vector t
// holds quarter t of both, [group k, group k + 1], so the butterflies are lane-wise; twiddles are
// [w_k x 8, w_(k+1) x 8].
template <class S>
struct WideBottom : Butterflies<S> {
    static_assert(S::lanes == 16);
    using B = Butterflies<S>;
    using V = typename S::V;
    using F = typename B::Factor;
    using B::canonical, B::low;

    static typename S::Index index(int a, int b) {
        std::array<int, 16> lanes;
        for (int i = 0; i < 16; ++i) lanes[i] = i < 8 ? a : b;
        return S::index(lanes);
    }

    // Lanes of a table block: x pair (0, 1), y pair (0, 2), z pair (1, 3); quotients 8 on.
    struct Indices {
        typename S::Index xw = index(0, 1), xq = index(8, 9), yw = index(0, 2), yq = index(8, 10),
                          zw = index(1, 3), zq = index(9, 11), low, high;
        Indices() {
            alignas(64) u32 l[16], h[16];
            for (int i = 0; i < 16; ++i) l[i] = i < 8 ? i : 8 + i, h[i] = l[i] + 8;
            low = S::load(l), high = S::load(h);
        }
    };

    struct Pair {
        F x, y, z;
    };
    // Twiddles of groups k, k + 1; the table blocks read are 16 words from entry k and 2k.
    [[gnu::always_inline]] static Pair twiddles(const u32* table, std::size_t k, const Indices& n) {
        const V xs = S::load(table + slot(k)), yz = S::load(table + slot(2 * k));
        return {{S::permute(xs, n.xw), S::permute(xs, n.xq)}, {S::permute(yz, n.yw), S::permute(yz, n.yq)},
                {S::permute(yz, n.zw), S::permute(yz, n.zq)}};
    }
    // The same by broadcast_halves: off Intel's p5, which the leaf products load (notes.md).
    [[gnu::always_inline]] static Pair twiddles(const u32* table, std::size_t k) {
        const u32 *x = table + slot(k), *y = table + slot(2 * k);
        const auto pair = [](const u32* p, int a, int b) { return S::broadcast_halves(p + a, p + b); };
        return {{pair(x, 0, 1), pair(x, 8, 9)}, {pair(y, 0, 2), pair(y, 8, 10)}, {pair(y, 1, 3), pair(y, 9, 11)}};
    }

    // 64 words of groups k, k + 1 -> the quarter vectors, and back.
    [[gnu::always_inline]] static void load(const u32* f, V q[4], const Indices& n) {
        const V z0 = S::load(f), z1 = S::load(f + 16), z2 = S::load(f + 32), z3 = S::load(f + 48);
        q[0] = S::permute2(z0, n.low, z2), q[1] = S::permute2(z0, n.high, z2);
        q[2] = S::permute2(z1, n.low, z3), q[3] = S::permute2(z1, n.high, z3);
    }
    [[gnu::always_inline]] static void store(u32* f, const V q[4], const Indices& n) {
        S::store(f, S::permute2(q[0], n.low, q[1])), S::store(f + 16, S::permute2(q[2], n.low, q[3]));
        S::store(f + 32, S::permute2(q[0], n.high, q[1])), S::store(f + 48, S::permute2(q[2], n.high, q[3]));
    }

    // Leaf::forward and Leaf::inverse on groups [first, first + count), first and count even.
    static void forward(u32* f, std::size_t count, std::size_t first, const u32* r) {
        const Indices n;
        for (std::size_t k = first; k < first + count; k += 2, f += 64) {
            const Pair w = twiddles(r, k, n);
            V q[4];
            load(f, q, n);
            B::forward4(q, w.x, w.y, w.z);
            q[0] = canonical(low(q[0])), q[1] = canonical(low(q[1])), q[2] = canonical(low(q[2])), q[3] = canonical(low(q[3]));
            store(f, q, n);
        }
    }

    static void inverse(u32* f, std::size_t count, std::size_t first, const u32* ir) {
        const Indices n;
        for (std::size_t k = first; k < first + count; k += 2, f += 64) {
            const Pair w = twiddles(ir, k, n);
            V q[4];
            load(f, q, n);
            B::inverse4(q, w.x, w.y, w.z);
            store(f, q, n);
        }
    }

    // The canonical leaves of groups k, k + 1 in memory, for leaf_products. rows[t][h] = [w a, a]
    // for quarter t of group k + h: x^(8-j) A mod x^8 - w is words j .. j + 7 of a row (j = 1 .. 9;
    // j = 9 reads 1 word past the row, a don't-care lane). a[t], b[t] = [quarter t of k, of k + 1].
    struct alignas(64) Leaves {
        u32 rows[4][2][16];
        u32 a[4][16], b[4][16];
    };

    // [A_k B_k, A_(k+1) B_(k+1)] 2^-32 mod x^8 - w per half, < 2P. Windows are load_halves of the
    // rows, b_i per half broadcast_halves of b: those run on p0 or p5 on Emerald Rapids, where the
    // vpermd of b ran on p5 only, as all shuffles do (notes.md).
    // kSquare: B is A.
    template <bool kSquare>
    [[gnu::always_inline]] static V leaf_products(const Leaves& in, int t) {
        using M = typename B::M;
        const u32* b = kSquare ? in.a[t] : in.b[t];
        const V p = S::broadcast(kP), ninv = S::broadcast(u32(0) - M::kPinv);
        const auto x = [&](int j) { return j == 8 ? S::load(in.a[t]) : S::load_halves(in.rows[t][0] + j, in.rows[t][1] + j); };
        V e = {}, o = {};
        #pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) {
            const V bi = S::broadcast_halves(b + i, b + 8 + i);
            const V pe = S::mul_even(x(8 - i), bi), po = S::mul_even(x(9 - i), bi);
            e = i == 0 ? pe : S::add64(e, pe), o = i == 0 ? po : S::add64(o, po);
        }
        e = S::add64(e, S::mul_even(S::mul_even(e, ninv), p));
        o = S::add64(o, S::mul_even(S::mul_even(o, ninv), p));
        return low(S::join_high(e, o));
    }

    template <bool kSquare>
    [[gnu::always_inline]] static void forward_leaves(const u32* a, const u32* b, std::size_t k, const u32* r,
                                                      const Indices& n, Leaves& out) {
        const V p = S::broadcast(kP), ones = S::broadcast(~u32(0));
        const Pair w = twiddles(r, k);
        const F minus_y = {S::sub(p, w.y.w), S::sub(ones, w.y.q)}, minus_z = {S::sub(p, w.z.w), S::sub(ones, w.z.q)};
        const F* weight[4] = {&w.y, &minus_y, &w.z, &minus_z};
        V f[4], g[4];
        load(a, f, n);
        B::forward4(f, w.x, w.y, w.z);
        if constexpr (!kSquare) load(b, g, n), B::forward4(g, w.x, w.y, w.z);
        #pragma GCC unroll 4
        for (int t = 0; t < 4; ++t) {
            const V at = canonical(low(f[t])), wat = canonical(B::mul(at, *weight[t]));
            S::template store_half<0>(out.rows[t][0], wat), S::template store_half<0>(out.rows[t][0] + 8, at);
            S::template store_half<1>(out.rows[t][1], wat), S::template store_half<1>(out.rows[t][1] + 8, at);
            S::store(out.a[t], at);
            if constexpr (!kSquare) S::store(out.b[t], canonical(low(g[t])));
        }
    }

    template <bool kSquare>
    [[gnu::always_inline]] static void multiply_leaves(u32* a, std::size_t k, const u32* ir, const Indices& n,
                                                       const Leaves& in) {
        V f[4];
        #pragma GCC unroll 4
        for (int t = 0; t < 4; ++t) f[t] = leaf_products<kSquare>(in, t);
        const Pair iw = twiddles(ir, k);
        B::inverse4(f, iw.x, iw.y, iw.z);
        store(a, f, n);
    }

    // Leaf::multiply on S; count and first even. Per chunk of kChunk pairs, one loop of forward
    // halves into a buffer, then one of leaf products and inverse halves: shorter dependency chains
    // per loop iteration than both in one loop (notes.md).
    static constexpr std::size_t kChunk = 4;
    template <bool kSquare = false>
    static void multiply(u32* a, const u32* b, std::size_t count, std::size_t first, const u32* r, const u32* ir) {
        const Indices n;
        Leaves leaves[kChunk];
        for (std::size_t c = 0; c < count / 2; c += kChunk, a += 64 * kChunk, b += 64 * kChunk) {
            const std::size_t m = std::min(kChunk, count / 2 - c), k = first + 2 * c;
            for (std::size_t g = 0; g < m; ++g) forward_leaves<kSquare>(a + 64 * g, b + 64 * g, k + 2 * g, r, n, leaves[g]);
            for (std::size_t g = 0; g < m; ++g) multiply_leaves<kSquare>(a + 64 * g, k + 2 * g, ir, n, leaves[g]);
        }
    }

    // Transform's fused ops (Bottom::multiply_stored, inverse_terms on S). leaf_products with b's
    // quarter t of groups k, k + 1 at b0, b1 (stored leaves).
    [[gnu::always_inline]] static V leaf_products_at(const Leaves& in, int t, const u32* b0, const u32* b1) {
        using M = typename B::M;
        const V p = S::broadcast(kP), ninv = S::broadcast(u32(0) - M::kPinv);
        const auto x = [&](int j) { return j == 8 ? S::load(in.a[t]) : S::load_halves(in.rows[t][0] + j, in.rows[t][1] + j); };
        V e = {}, o = {};
        #pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) {
            const V bi = S::broadcast_halves(b0 + i, b1 + i);
            const V pe = S::mul_even(x(8 - i), bi), po = S::mul_even(x(9 - i), bi);
            e = i == 0 ? pe : S::add64(e, pe), o = i == 0 ? po : S::add64(o, po);
        }
        e = S::add64(e, S::mul_even(S::mul_even(e, ninv), p));
        o = S::add64(o, S::mul_even(S::mul_even(o, ninv), p));
        return low(S::join_high(e, o));
    }

    // The rows of groups k, k + 1 from their canonical leaves at a (forward_leaves without the
    // butterflies).
    [[gnu::always_inline]] static void fill_rows(const u32* a, std::size_t k, const u32* r, const Indices& n, Leaves& out) {
        const V p = S::broadcast(kP), ones = S::broadcast(~u32(0));
        const Pair w = twiddles(r, k);
        const F minus_y = {S::sub(p, w.y.w), S::sub(ones, w.y.q)}, minus_z = {S::sub(p, w.z.w), S::sub(ones, w.z.q)};
        const F* weight[4] = {&w.y, &minus_y, &w.z, &minus_z};
        V f[4];
        load(a, f, n);
        #pragma GCC unroll 4
        for (int t = 0; t < 4; ++t) {
            const V at = f[t], wat = canonical(B::mul(at, *weight[t]));
            S::template store_half<0>(out.rows[t][0], wat), S::template store_half<0>(out.rows[t][0] + 8, at);
            S::template store_half<1>(out.rows[t][1], wat), S::template store_half<1>(out.rows[t][1] + 8, at);
            S::store(out.a[t], at);
        }
    }

    // The rows of Np pairs of groups k + 2j, k + 2j + 1 for Bottom::products, from the canonical
    // leaves of a (load_halves).
    template <int Np>
    [[gnu::always_inline]] static void fill_pairs(const u32* a, std::size_t k, const u32* r, Leaves (&out)[2]) {
        #pragma GCC unroll 2
        for (int j = 0; j < Np; ++j) {
            const V p = S::broadcast(kP), ones = S::broadcast(~u32(0));
            const Pair w = twiddles(r, k + 2 * j);
            const F minus_y = {S::sub(p, w.y.w), S::sub(ones, w.y.q)}, minus_z = {S::sub(p, w.z.w), S::sub(ones, w.z.q)};
            const F* weight[4] = {&w.y, &minus_y, &w.z, &minus_z};
            const u32* aj = a + 64 * j;
            #pragma GCC unroll 4
            for (int t = 0; t < 4; ++t) {
                const V at = S::load_halves(aj + 8 * t, aj + 32 + 8 * t), wat = canonical(B::mul(at, *weight[t]));
                S::template store_half<0>(out[j].rows[t][0], wat), S::template store_half<0>(out[j].rows[t][0] + 8, at);
                S::template store_half<1>(out[j].rows[t][1], wat), S::template store_half<1>(out[j].rows[t][1] + 8, at);
                S::store(out[j].a[t], at);
            }
        }
    }

    // The leaf products of those Np pairs, the pairs through each step together: times 2^-32
    // (leaf_products_at), then 2^32 by a Shoup product; out = them, or out += them (kAdd).
    template <bool kAdd, int Np>
    [[gnu::always_inline]] static void products_pairs(const Leaves (&in)[2], const u32* b, u32* out, const F& to_r) {
        V c[Np][4];
        #pragma GCC unroll 4
        for (int t = 0; t < 4; ++t) {
            #pragma GCC unroll 2
            for (int j = 0; j < Np; ++j) c[j][t] = leaf_products_at(in[j], t, b + 64 * j + 8 * t, b + 64 * j + 32 + 8 * t);
        }
        #pragma GCC unroll 4
        for (int t = 0; t < 4; ++t) {
            #pragma GCC unroll 2
            for (int j = 0; j < Np; ++j) {
                u32* o = out + 64 * j + 8 * t;
                V v = B::mul(c[j][t], to_r);
                v = kAdd ? canonical(low(B::add(S::load_halves(o, o + 32), v))) : canonical(v);
                S::template store_half<0>(o, v), S::template store_half<1>(o + 32, v);
            }
        }
    }

    // Bottom::products on S (Transform::multiply, multiply_add), count and first even: two pairs
    // of groups per step (one at an odd end), the next step's rows filled before this step's
    // products, as Bottom's (filled right before, their loads waited behind the stores of out
    // past the caches: Zen 5 at 2^23 2.5x the time of Bottom<Avx512Vl>). On L1 data, against
    // Bottom<Avx512Vl>::products: Zen 5 0.68 of its time (one pair per step 0.85), Emerald
    // Rapids 0.97-1.00 (not used there). out may be a or b.
    template <bool kAdd>
    static void products(const u32* a, const u32* b, u32* out, std::size_t count, std::size_t first, const u32* r) {
        const F to_r = B::factor(kR);
        const std::size_t steps = count / 4, last = count % 4;  // last: 0 or 2 groups
        Leaves leaves[2][2];
        if (steps) fill_pairs<2>(a, first, r, leaves[0]);
        else if (last) fill_pairs<1>(a, first, r, leaves[0]);
        for (std::size_t s = 0; s < steps; ++s) {
            const std::size_t g = 4 * s;
            if (s + 1 < steps) fill_pairs<2>(a + 32 * (g + 4), first + g + 4, r, leaves[(s + 1) % 2]);
            else if (last) fill_pairs<1>(a + 32 * (g + 4), first + g + 4, r, leaves[(s + 1) % 2]);
            products_pairs<kAdd, 2>(leaves[s % 2], b + 32 * g, out + 32 * g, to_r);
        }
        if (last) products_pairs<kAdd, 1>(leaves[steps % 2], b + 128 * steps, out + 128 * steps, to_r);
    }

    // Bottom::multiply_stored; count and first even. b: the stored leaves of group first.
    static void multiply_stored(u32* a, const u32* b, std::size_t count, std::size_t first, const u32* r, const u32* ir) {
        const Indices n;
        Leaves leaves[kChunk];
        for (std::size_t c = 0; c < count / 2; c += kChunk, a += 64 * kChunk, b += 64 * kChunk) {
            const std::size_t m = std::min(kChunk, count / 2 - c), k = first + 2 * c;
            for (std::size_t g = 0; g < m; ++g) forward_leaves<true>(a + 64 * g, a + 64 * g, k + 2 * g, r, n, leaves[g]);
            for (std::size_t g = 0; g < m; ++g) {
                const u32* bg = b + 64 * g;
                V f[4];
                #pragma GCC unroll 4
                for (int t = 0; t < 4; ++t) f[t] = leaf_products_at(leaves[g], t, bg + 8 * t, bg + 32 + 8 * t);
                const Pair iw = twiddles(ir, k + 2 * g);
                B::inverse4(f, iw.x, iw.y, iw.z);
                store(a + 64 * g, f, n);
            }
        }
    }

    // Bottom::inverse_terms; count and first even.
    static void inverse_terms(u32* out, const LeafTerm* terms, int nterms, std::size_t count, std::size_t first,
                              const u32* r, const u32* ir) {
        static constexpr u32 kRinverse = power(kR, kP - 2);  // 2^-32 mod P
        const F to_montgomery = B::factor(kRinverse);
        const Indices n;
        Leaves leaves;
        for (std::size_t k = first; k < first + count; k += 2, out += 64) {
            V f[4];
            for (int i = 0; i < nterms; ++i) {
                const u32 *a = terms[i].a + 32 * k, *b = terms[i].b;
                V d[4];
                if (b) {
                    fill_rows(a, k, r, n, leaves);
                    b += 32 * k;
                    #pragma GCC unroll 4
                    for (int t = 0; t < 4; ++t) d[t] = leaf_products_at(leaves, t, b + 8 * t, b + 32 + 8 * t);
                } else {
                    load(a, d, n);
                    #pragma GCC unroll 4
                    for (int t = 0; t < 4; ++t) d[t] = B::mul(d[t], to_montgomery);
                }
                #pragma GCC unroll 4
                for (int t = 0; t < 4; ++t) f[t] = i == 0 ? d[t] : low(B::add(f[t], d[t]));
            }
            const Pair iw = twiddles(ir, k);
            B::inverse4(f, iw.x, iw.y, iw.z);
            store(out, f, n);
        }
    }

    // Bottom::inverse_toeplitz; count and first even. S, D1, D3 word by word (any layout), their rows
    // from memory as fill_rows'.
    static void inverse_toeplitz(u32* out, const Toeplitz& z, std::size_t count, std::size_t first, const u32* r,
                                 const u32* ir) {
        using M = typename B::M;
        const F to_r = B::factor(kR);
        const Indices n;
        alignas(64) u32 sum[3][64];
        Leaves leaves[3];
        for (std::size_t k = first; k < first + count; k += 2, out += 64) {
            const std::size_t at = 32 * k;
            #pragma GCC unroll 4
            for (std::size_t i = 0; i < 64; i += 16) {
                const V a2 = S::load(z.a2 + at + i);
                S::store(sum[0] + i, M::add(S::load(z.b0 + at + i), S::load(z.b1 + at + i)));
                S::store(sum[1] + i, M::sub(S::load(z.a1 + at + i), a2));
                S::store(sum[2] + i, M::sub(S::load(z.a3 + at + i), a2));
            }
            for (int i = 0; i < 3; ++i) fill_rows(sum[i], k, r, n, leaves[i]);
            const u32 *a2 = z.a2 + at, *b1 = z.b1 + at, *b0 = z.b0 + at;
            V f[4], side[4];
            #pragma GCC unroll 4
            for (int t = 0; t < 4; ++t) {
                const V m1 = leaf_products_at(leaves[0], t, a2 + 8 * t, a2 + 32 + 8 * t);
                const V m2 = leaf_products_at(leaves[1], t, b1 + 8 * t, b1 + 32 + 8 * t);
                const V m3 = leaf_products_at(leaves[2], t, b0 + 8 * t, b0 + 32 + 8 * t);
                f[t] = low(B::add(m1, m2)), side[t] = canonical(B::mul(B::add(m1, m3), to_r));
            }
            store(z.a3 + at, side, n);
            const Pair iw = twiddles(ir, k);
            B::inverse4(f, iw.x, iw.y, iw.z);
            store(out, f, n);
        }
    }
};

// The values form (common.hpp) on L (1, 4 or 8 lanes), one block of 64 words (8 leaves) at a
// time: lane t of a vector is leaf t of a set of L::lanes leaves, so the twist and the 8-point
// transform are lane-wise; transposes move between leaves and lanes. Shoup products per set: the
// split 19, the join 19 (its 1/8 is in the inverse's scale), 7 fewer each for blocks g < 8.
template <class L>
struct Values : Arith<L> {
    using A = Arith<L>;
    using V = typename L::V;
    using F = typename A::Factor;
    using M = typename A::M;
    using A::add, A::diff, A::low, A::canonical, A::mul;
    static constexpr int W = L::lanes, kSets = 8 / W;
    using Set = V[8];

    // x[k] = word k of leaves [s W, s W + W) of the block at p (lane: leaf), and back.
    [[gnu::always_inline]] static void load_leaves(const u32* p, int s, Set& x) {
        if constexpr (W == 1) {
            #pragma GCC unroll 8
            for (int k = 0; k < 8; ++k) x[k] = p[8 * s + k];
        } else {
            #pragma GCC unroll 8
            for (int h = 0; h < 8; h += W) {
                V r[W];
                #pragma GCC unroll 8
                for (int u = 0; u < W; ++u) r[u] = L::load(p + 8 * (s * W + u) + h);
                transpose(r);
                #pragma GCC unroll 8
                for (int u = 0; u < W; ++u) x[h + u] = r[u];
            }
        }
    }
    [[gnu::always_inline]] static void store_leaves(u32* p, int s, Set& x) {
        if constexpr (W == 1) {
            #pragma GCC unroll 8
            for (int k = 0; k < 8; ++k) p[8 * s + k] = x[k];
        } else {
            #pragma GCC unroll 8
            for (int h = 0; h < 8; h += W) {
                V r[W];
                #pragma GCC unroll 8
                for (int u = 0; u < W; ++u) r[u] = x[h + u];
                transpose(r);
                #pragma GCC unroll 8
                for (int u = 0; u < W; ++u) L::store(p + 8 * (s * W + u) + h, r[u]);
            }
        }
    }
    // Values i of the set at p + 8 i + s W.
    [[gnu::always_inline]] static void load_values(const u32* p, int s, Set& x) {
        #pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) x[i] = L::load(p + 8 * i + s * W);
    }
    [[gnu::always_inline]] static void store_values(u32* p, int s, const Set& x) {
        #pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) L::store(p + 8 * i + s * W, x[i]);
    }

    // The square of W vectors transposed: lane u of vector i <-> lane i of vector u.
    [[gnu::always_inline]] static void transpose(V (&r)[W]) {
#if defined(__AVX2__)
        if constexpr (W == 8) {
            V t[8], u[8];
            #pragma GCC unroll 8
            for (int i = 0; i < 8; i += 2) t[i] = _mm256_unpacklo_epi32(r[i], r[i + 1]), t[i + 1] = _mm256_unpackhi_epi32(r[i], r[i + 1]);
            #pragma GCC unroll 8
            for (int i = 0; i < 8; i += 4) {
                u[i] = _mm256_unpacklo_epi64(t[i], t[i + 2]), u[i + 1] = _mm256_unpackhi_epi64(t[i], t[i + 2]);
                u[i + 2] = _mm256_unpacklo_epi64(t[i + 1], t[i + 3]), u[i + 3] = _mm256_unpackhi_epi64(t[i + 1], t[i + 3]);
            }
            #pragma GCC unroll 8
            for (int i = 0; i < 4; ++i) {
                r[i] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x20);
                r[i + 4] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x31);
            }
            return;
        }
#endif
#if defined(__SSE4_1__)
        if constexpr (W == 4 && std::is_same_v<V, __m128i>) {
            const V t0 = _mm_unpacklo_epi32(r[0], r[1]), t1 = _mm_unpackhi_epi32(r[0], r[1]);
            const V t2 = _mm_unpacklo_epi32(r[2], r[3]), t3 = _mm_unpackhi_epi32(r[2], r[3]);
            r[0] = _mm_unpacklo_epi64(t0, t2), r[1] = _mm_unpackhi_epi64(t0, t2);
            r[2] = _mm_unpacklo_epi64(t1, t3), r[3] = _mm_unpackhi_epi64(t1, t3);
            return;
        }
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
        if constexpr (W == 4 && std::is_same_v<V, uint32x4_t>) {
            const V t0 = vtrn1q_u32(r[0], r[1]), t1 = vtrn2q_u32(r[0], r[1]);
            const V t2 = vtrn1q_u32(r[2], r[3]), t3 = vtrn2q_u32(r[2], r[3]);
            const auto wide = [](V v) { return vreinterpretq_u64_u32(v); };
            r[0] = vreinterpretq_u32_u64(vtrn1q_u64(wide(t0), wide(t2)));
            r[1] = vreinterpretq_u32_u64(vtrn1q_u64(wide(t1), wide(t3)));
            r[2] = vreinterpretq_u32_u64(vtrn2q_u64(wide(t0), wide(t2)));
            r[3] = vreinterpretq_u32_u64(vtrn2q_u64(wide(t1), wide(t3)));
            return;
        }
#endif
        static_assert(W == 8 || W == 4, "Values: no transpose for this struct");
    }

    // x w per lane (any x -> [0, 2P)): q and w the quotients and values of a lane table row, qo
    // (x86) the odd lanes' quotients in the even lanes, so q needs no shift.
    [[gnu::always_inline]] static V mul_lanes(V x, V q, V qo, V w) {
        if constexpr (requires { L::mul_even(x, x); }) {
            const V even = L::mul_even(x, q), odd = L::mul_even(L::template shr64<32>(x), qo);
            V qx;  // floor(x q / 2^32), joined as simd's mulhi joins
            if constexpr (simd::detail::kMulhiPermute) qx = L::join_high(even, odd);
            else qx = L::template blend<0xaaaa>(L::template shr64<32>(even), odd);
            return A::whole(L::mul_sub(L::mul_lo(x, w), qx, L::broadcast(kP)));
        } else {
            return A::whole(M::shoup(x, w, q));
        }
    }
    // From a row of ValueLanes (quotients, then values): qo is the load 4 bytes on.
    [[gnu::always_inline]] static V mul_row(V x, const u32* row) {
        return mul_lanes(x, L::load(row), L::load(row + 1), L::load(row + 8));
    }

    // The 8-point transform of N twisted sets, step by step across the sets: x[k] < 2P in;
    // canonical values out, in bit-reversed order. z[k]: z^k. 84 adds, subtracts and mins per set
    // (100 before, with x[0] < P).
    template <int N = 1>
    [[gnu::always_inline]] static void forward8(Set* x, const F (&z)[4]) {
        #pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) {  // sums and x[4] < 4P, products < 2P
            #pragma GCC unroll 2
            for (int s = 0; s < N; ++s) {
                const V u = x[s][k], v = x[s][k + 4];
                x[s][k] = add(u, v), x[s][k + 4] = k == 0 ? diff(u, v) : mul(diff(u, v), z[k]);
            }
        }
        #pragma GCC unroll 5
        for (int k = 0; k < 5; ++k) {
            #pragma GCC unroll 2
            for (int s = 0; s < N; ++s) x[s][k] = low(x[s][k]);
        }
        #pragma GCC unroll 2
        for (int b = 0; b < 8; b += 4) {  // < 2P out
            #pragma GCC unroll 2
            for (int s = 0; s < N; ++s) {
                const V u = x[s][b], v = x[s][b + 2], c = x[s][b + 1], d = x[s][b + 3];
                x[s][b] = low(add(u, v)), x[s][b + 2] = A::low_diff(u, v);
                x[s][b + 1] = low(add(c, d)), x[s][b + 3] = mul(diff(c, d), z[2]);
            }
        }
        #pragma GCC unroll 4
        for (int b = 0; b < 8; b += 2) {
            #pragma GCC unroll 2
            for (int s = 0; s < N; ++s) {
                const V u = x[s][b], v = x[s][b + 1];
                x[s][b] = canonical(low(add(u, v))), x[s][b + 1] = canonical(A::low_diff(u, v));
            }
        }
    }

    // forward8 undone, times 8: canonical in, < 4P out. z[k]: z^-k.
    template <int N = 1>
    [[gnu::always_inline]] static void inverse8(Set* x, const F (&z)[4]) {
        const V p = L::broadcast(kP);
        #pragma GCC unroll 8
        for (int b = 0; b < 8; b += 2) {  // < 2P out
            #pragma GCC unroll 2
            for (int s = 0; s < N; ++s) {
                const V u = x[s][b], v = x[s][b + 1];
                x[s][b] = add(u, v), x[s][b + 1] = L::sub(L::add(u, p), v);
            }
        }
        #pragma GCC unroll 8
        for (int b = 0; b < 8; b += 4) {
            #pragma GCC unroll 2
            for (int k = 0; k < 2; ++k) {  // < 4P out
                #pragma GCC unroll 2
                for (int s = 0; s < N; ++s) {
                    const V u = x[s][b + k], v = k == 0 ? x[s][b + k + 2] : mul(x[s][b + k + 2], z[2]);
                    x[s][b + k] = add(u, v), x[s][b + k + 2] = diff(u, v);
                }
            }
        }
        #pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) {
            #pragma GCC unroll 2
            for (int s = 0; s < N; ++s) {
                const V u = low(x[s][k]), v = k == 0 ? low(x[s][k + 4]) : mul(x[s][k + 4], z[k]);
                x[s][k] = add(u, v), x[s][k + 4] = diff(u, v);
            }
        }
    }

    static void factors(F (&z)[4], int inverse) {
        #pragma GCC unroll 4
        for (int k = 0; k < 4; ++k) z[k] = A::entry(kStaticTables.words[inverse], eighth_root(k));
    }

    // The twist of x[1..7] of N sets by rho^k (forward) or rho^-k (inverse): lane(j, k, v) = v
    // times set j's lane factors B(g % 8, t)^k, then (kA) the broadcast A(g / 8)^k of row a (the
    // same for every set); A(0) = 1. < 2P out.
    template <bool kA, int N, class Lane>
    [[gnu::always_inline]] static void twist(Set* x, Lane lane, const u32* a) {
        #pragma GCC unroll 8
        for (int k = 1; k < 8; ++k) {
            #pragma GCC unroll 4
            for (int j = 0; j < N; ++j) {
                x[j][k] = lane(j, k, x[j][k]);
                if constexpr (kA) x[j][k] = mul(x[j][k], A::factor(a[k], a[8 + k]));
            }
        }
    }

    // Blocks g .. g + Nb - 1 (one A row: g a multiple of Nb), their sets through each step
    // together (Neon: both sets of a block, 0.80 of the time of one after the other on Neoverse
    // V2). Every set is loaded before a store: to may be from.
    template <bool kA, int Nb>
    [[gnu::always_inline]] static void split_blocks(const u32* from, u32* to, std::size_t g, const u32* rows,
                                                    const F (&z)[4]) {
        constexpr int N = Nb * kSets;
        Set x[N];
        #pragma GCC unroll 8
        for (int j = 0; j < N; ++j) load_leaves(from + 64 * (j / kSets), j % kSets, x[j]);
        twist<kA, N>(x, [g] [[gnu::always_inline]] (int j, int k, V v) {
            return mul_row(v, kValueLanes.forward[(g + j / kSets) % 8][k] + j % kSets * W);
        }, rows + 32 * (g / 8));
        forward8<N>(x, z);
        #pragma GCC unroll 8
        for (int j = 0; j < N; ++j) store_values(to + 64 * (j / kSets), j % kSets, x[j]);
    }

    template <bool kA, int Nb, bool kCanonical>
    [[gnu::always_inline]] static void join_blocks(const u32* from, u32* to, std::size_t g, const u32* rows,
                                                   const F (&z)[4]) {
        constexpr int N = Nb * kSets;
        Set x[N];
        #pragma GCC unroll 8
        for (int j = 0; j < N; ++j) load_values(from + 64 * (j / kSets), j % kSets, x[j]);
        inverse8<N>(x, z);
        #pragma GCC unroll 8
        for (int j = 0; j < N; ++j) x[j][0] = low(x[j][0]);
        twist<kA, N>(x, [g] [[gnu::always_inline]] (int j, int k, V v) {
            return mul_row(v, kValueLanes.inverse[(g + j / kSets) % 8][k] + j % kSets * W);
        }, rows + 32 * (g / 8) + 16);
        #pragma GCC unroll 8
        for (int j = 0; j < N; ++j) {
            if constexpr (kCanonical) {
                #pragma GCC unroll 8
                for (int k = 0; k < 8; ++k) x[j][k] = canonical(x[j][k]);
            }
            store_leaves(to + 64 * (j / kSets), j % kSets, x[j]);
        }
    }

    // Blocks per step (split_blocks' Nb): 2 on 8 lanes with 32 registers, else 1. Avx2: 2 ran the
    // split alone 0.93 of the time of 1 on Zen 3 (L1 data), but in the transforms 0.97-1.06
    // (bigger code); 4 spilled. Neon (2 sets per block): 2 blocks spilled, 1.19 the time of 1 on
    // Neoverse V2 (notes/98-r1.md).
#if defined(FPS_NTT_VALUES_STEP)
    static constexpr int kStep = FPS_NTT_VALUES_STEP;
#else
    static constexpr int kStep = W == 8 && L::registers >= 32 ? 2 : 1;
#endif

    // run(std::bool_constant<kA>{}, std::integral_constant<int, n>{}, g) over the blocks
    // [first, first + count): n = kStep blocks from g, a multiple of kStep, else n = 1; kA = g >= 8
    // (A(0) = 1: 7 products fewer per set).
    template <class Run>
    [[gnu::always_inline]] static void each_block(std::size_t count, std::size_t first, Run run) {
        using One = std::integral_constant<int, 1>;
        using Step = std::integral_constant<int, kStep>;
        std::size_t g = first;
        const std::size_t end = first + count;
        if constexpr (kStep > 1)
            for (; g < end && g % kStep; ++g) g < 8 ? run(std::false_type{}, One{}, g) : run(std::true_type{}, One{}, g);
        for (; g + kStep <= end && g < 8; g += kStep) run(std::false_type{}, Step{}, g);
        for (; g + kStep <= end; g += kStep) run(std::true_type{}, Step{}, g);
        if constexpr (kStep > 1)
            for (; g < end; ++g) g < 8 ? run(std::false_type{}, One{}, g) : run(std::true_type{}, One{}, g);
    }

    // Blocks [first, first + count) of the leaf form at `from` -> the values form at `to` (may be
    // from). Leaves < 2P in (the transforms': canonical); canonical out. rows: build_value_rows.
    static void split(const u32* from, u32* to, std::size_t count, std::size_t first, const u32* rows) {
        F z[4];
        factors(z, 0);
        each_block(count, first, [&]<class On, class Nb> [[gnu::always_inline]] (On, Nb, std::size_t g) {
            split_blocks<On::value, Nb::value>(from + 64 * (g - first), to + 64 * (g - first), g, rows, z);
        });
    }

    // split undone but for the 8-point transform's gain: the values form at `from` (canonical) ->
    // 8 times the leaf form at `to` (may be from), < 2P, or canonical (kCanonical). The inverses
    // after it scale by values_scale(lg), 1/8 of leaf_scale(lg) (common.hpp).
    template <bool kCanonical = true>
    static void join(const u32* from, u32* to, std::size_t count, std::size_t first, const u32* rows) {
        F z[4];
        factors(z, 1);
        each_block(count, first, [&]<class On, class Nb> [[gnu::always_inline]] (On, Nb, std::size_t g) {
            join_blocks<On::value, Nb::value, kCanonical>(from + 64 * (g - first), to + 64 * (g - first), g, rows, z);
        });
    }
};

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
// The values form on 16 lanes (issue #157): two blocks per step, g even, lane 8h + t of a vector
// is leaf t of block g + h, so each 256-bit half is Values' 8 lanes; the twist and the 8-point
// transform run on zmm, the transposes in both halves at once. A lone first or last block (odd
// first or count) on Values<Narrow>. The same words as Values. Emerald Rapids ran the split and
// join on ymm at 0.5-0.9 ns per word added to a transform of 0.4-1.0 (notes/157-r1.md).
template <class S>
struct WideValues : Values<S> {
    using Base = Values<S>;
    using Lone = Values<typename Narrow<S>::type>;
    using V = typename S::V;
    using F = typename Base::F;
    using M = typename Base::M;
    using Base::forward8, Base::inverse8, Base::factors, Base::canonical, Base::mul;
    using Set = V[8];
    static_assert(S::lanes == 16);

    // In each 256-bit half: lane u of r[i] <-> lane i of r[u] (Values::transpose on 8 lanes).
    [[gnu::always_inline]] static void transpose(Set& r) {
        V t[8], u[8];
        #pragma GCC unroll 8
        for (int i = 0; i < 8; i += 2) t[i] = _mm512_unpacklo_epi32(r[i], r[i + 1]), t[i + 1] = _mm512_unpackhi_epi32(r[i], r[i + 1]);
        #pragma GCC unroll 8
        for (int i = 0; i < 8; i += 4) {
            u[i] = _mm512_unpacklo_epi64(t[i], t[i + 2]), u[i + 1] = _mm512_unpackhi_epi64(t[i], t[i + 2]);
            u[i + 2] = _mm512_unpacklo_epi64(t[i + 1], t[i + 3]), u[i + 3] = _mm512_unpackhi_epi64(t[i + 1], t[i + 3]);
        }
        const __m512i low = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13), high = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
        #pragma GCC unroll 4
        for (int i = 0; i < 4; ++i) {
            r[i] = _mm512_permutex2var_epi64(u[i], low, u[i + 4]);
            r[i + 4] = _mm512_permutex2var_epi64(u[i], high, u[i + 4]);
        }
    }

    // The lane rows of blocks g, g + 1 (g even) as one 16-lane row: per pair p = g % 8 / 2 and
    // k, 48 words: the quotients, the odd lanes' quotients in the even lanes, the values. One zmm
    // load each (load_halves of ValueLanes' rows: a vinserti64x4 more per load).
    struct alignas(64) Pairs {
        u32 forward[4][8][48];
        u32 inverse[4][8][48];
    };
    static constexpr Pairs kPairs = [] {
        Pairs v{};
        for (int p = 0; p < 4; ++p)
            for (int k = 0; k < 8; ++k)
                for (int i = 0; i < 16; ++i) {
                    const int l = 2 * p + i / 8, t = i % 8, o = i % 2 ? i : i + 1;  // o: the odd lane
                    v.forward[p][k][i] = kValueLanes.forward[l][k][t], v.forward[p][k][32 + i] = kValueLanes.forward[l][k][8 + t];
                    v.inverse[p][k][i] = kValueLanes.inverse[l][k][t], v.inverse[p][k][32 + i] = kValueLanes.inverse[l][k][8 + t];
                    v.forward[p][k][16 + i] = kValueLanes.forward[2 * p + o / 8][k][o % 8];
                    v.inverse[p][k][16 + i] = kValueLanes.inverse[2 * p + o / 8][k][o % 8];
                }
        return v;
    }();

    // v times the lane factors of row k of pair p (kPairs.forward or inverse).
    [[gnu::always_inline]] static V mul_pair(V v, const u32 (&b)[8][48], int k) {
        return Base::mul_lanes(v, S::load(b[k]), S::load(b[k] + 16), S::load(b[k] + 32));
    }

    // The pairs g + 2j, g + 2j + 1 for j < Np (g a multiple of 2 Np: one A row) at from -> to,
    // through each step together; z: factors(z, 0).
    template <bool kA, int Np>
    [[gnu::always_inline]] static void split_pairs(const u32* from, u32* to, std::size_t g, const u32* rows,
                                                   const F (&z)[4]) {
        Set x[Np];
        #pragma GCC unroll 4
        for (int j = 0; j < Np; ++j) {
            #pragma GCC unroll 8
            for (int u = 0; u < 8; ++u) x[j][u] = S::load_halves(from + 128 * j + 8 * u, from + 128 * j + 64 + 8 * u);
            transpose(x[j]);
        }
        Base::template twist<kA, Np>(x, [g] [[gnu::always_inline]] (int j, int k, V v) {
            return mul_pair(v, kPairs.forward[(g + 2 * j) % 8 / 2], k);
        }, rows + 32 * (g / 8));
        Base::template forward8<Np>(x, z);
        #pragma GCC unroll 4
        for (int j = 0; j < Np; ++j) {
            #pragma GCC unroll 8
            for (int i = 0; i < 8; ++i)
                S::template store_half<0>(to + 128 * j + 8 * i, x[j][i]), S::template store_half<1>(to + 128 * j + 64 + 8 * i, x[j][i]);
        }
    }

    template <bool kA, int Np, bool kCanonical>
    [[gnu::always_inline]] static void join_pairs(const u32* from, u32* to, std::size_t g, const u32* rows,
                                                  const F (&z)[4]) {
        Set x[Np];
        #pragma GCC unroll 4
        for (int j = 0; j < Np; ++j) {
            #pragma GCC unroll 8
            for (int i = 0; i < 8; ++i) x[j][i] = S::load_halves(from + 128 * j + 8 * i, from + 128 * j + 64 + 8 * i);
        }
        Base::template inverse8<Np>(x, z);
        #pragma GCC unroll 4
        for (int j = 0; j < Np; ++j) x[j][0] = Base::low(x[j][0]);
        Base::template twist<kA, Np>(x, [g] [[gnu::always_inline]] (int j, int k, V v) {
            return mul_pair(v, kPairs.inverse[(g + 2 * j) % 8 / 2], k);
        }, rows + 32 * (g / 8) + 16);
        #pragma GCC unroll 4
        for (int j = 0; j < Np; ++j) {
            if constexpr (kCanonical) {
                #pragma GCC unroll 8
                for (int k = 0; k < 8; ++k) x[j][k] = canonical(x[j][k]);
            }
            transpose(x[j]);
            #pragma GCC unroll 8
            for (int u = 0; u < 8; ++u)
                S::template store_half<0>(to + 128 * j + 8 * u, x[j][u]), S::template store_half<1>(to + 128 * j + 64 + 8 * u, x[j][u]);
        }
    }

    // Pairs per step (split_pairs' Np). 2 against 1, in the transforms (2^10-2^18): Emerald Rapids
    // split 0.91-0.96, join 0.90-0.95; Zen 5 0.82-0.98, 0.87-0.93; 4 spilled (notes/98-r1.md).
#if defined(FPS_NTT_VALUES_PAIRS)
    static constexpr int kStep = FPS_NTT_VALUES_PAIRS;
#else
    static constexpr int kStep = 2;
#endif

    // run(std::bool_constant<kA>{}, std::integral_constant<int, n>{}, g) over the pairs in
    // [first, first + count), first and count even: n = kStep pairs from g, a multiple of 2 kStep,
    // else n = 1; kA = g >= 8.
    template <class Run>
    [[gnu::always_inline]] static void each_pair(std::size_t count, std::size_t first, Run run) {
        using One = std::integral_constant<int, 1>;
        using Step = std::integral_constant<int, kStep>;
        std::size_t g = first;
        const std::size_t end = first + count;
        if constexpr (kStep > 1)
            for (; g < end && g % (2 * kStep); g += 2) g < 8 ? run(std::false_type{}, One{}, g) : run(std::true_type{}, One{}, g);
        for (; g + 2 * kStep <= end && g < 8; g += 2 * kStep) run(std::false_type{}, Step{}, g);
        for (; g + 2 * kStep <= end; g += 2 * kStep) run(std::true_type{}, Step{}, g);
        if constexpr (kStep > 1)
            for (; g < end; g += 2) g < 8 ? run(std::false_type{}, One{}, g) : run(std::true_type{}, One{}, g);
    }

    // As Values::split and join.
    static void split(const u32* from, u32* to, std::size_t count, std::size_t first, const u32* rows) {
        if (count && first % 2) Lone::split(from, to, 1, first, rows), from += 64, to += 64, ++first, --count;
        F z[4];
        factors(z, 0);
        const std::size_t pairs = count / 2 * 2;
        each_pair(pairs, first, [&]<class On, class Np> [[gnu::always_inline]] (On, Np, std::size_t g) {
            split_pairs<On::value, Np::value>(from + 64 * (g - first), to + 64 * (g - first), g, rows, z);
        });
        if (count % 2) Lone::split(from + 64 * pairs, to + 64 * pairs, 1, first + pairs, rows);
    }
    template <bool kCanonical = true>
    static void join(const u32* from, u32* to, std::size_t count, std::size_t first, const u32* rows) {
        if (count && first % 2) Lone::template join<kCanonical>(from, to, 1, first, rows), from += 64, to += 64, ++first, --count;
        F z[4];
        factors(z, 1);
        const std::size_t pairs = count / 2 * 2;
        each_pair(pairs, first, [&]<class On, class Np> [[gnu::always_inline]] (On, Np, std::size_t g) {
            join_pairs<On::value, Np::value, kCanonical>(from + 64 * (g - first), to + 64 * (g - first), g, rows, z);
        });
        if (count % 2) Lone::template join<kCanonical>(from + 64 * pairs, to + 64 * pairs, 1, first + pairs, rows);
    }
};
#endif

// The values form's split and join for transforms on S: WideValues for 16 lanes (AVX-512).
template <class S>
struct ValuesOf {
    using type = Values<typename Narrow<S>::type>;
};
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
template <class S>
    requires(S::lanes == 16)
struct ValuesOf<S> {
    using type = WideValues<S>;
};
#endif
template <class S>
using ValuesFor = typename ValuesOf<S>::type;

template <class S>
struct Ntt : Butterflies<S> {
    using N = typename Narrow<S>::type;
    using Leaf = Bottom<N>;
    using B = Butterflies<S>;
    using V = typename S::V;
    using F = typename B::Factor;
    using Column = typename B::Column;
    using B::add, B::diff, B::canonical, B::forward_group, B::inverse_group;

    // Subtrees of at most this many leaves run level by level (8 KiB per array).
    static constexpr std::size_t kTile = 256;

    // variant::scalar::build_table, N::lanes entries at a time past the static prefix.
    static void build_table(u32* t, std::size_t count, bool inverse) {
        static_assert(kStaticCount >= 8);
        using A = Arith<N>;
        const Roots& roots = kRoots[inverse];
        for (std::size_t h = copy_static_table(t, count, inverse); h < count; h *= 2) {
            // Quotient of v: floor(v 2^32 / P) = -(v 2^32 mod P) P^-1 mod 2^32.
            const auto by = A::factor(roots[std::countr_zero(h)]), to_r = A::factor(kR);
            for (std::size_t j = 0; j < h; j += N::lanes) {
                const auto v = A::canonical(A::mul(N::load(t + slot(j)), by));
                const auto vr = A::canonical(A::mul(v, to_r));
                N::store(t + slot(h + j), v);
                N::store(t + slot(h + j) + 8, N::mul_lo(vr, N::broadcast(Leaf::kNegInverse)));
            }
        }
    }

    // The bottom level of groups [first, first + count): on S two groups at a time if S has 16
    // lanes and count is even, else on N.
    static constexpr bool kWideBottom = S::lanes == 16;
    static void bottom_forward(u32* f, std::size_t count, std::size_t first, const u32* r) {
        if constexpr (requires { Generated<S>::bottom_forward(f, count, first, r); })
            if (count % 2 == 0) return Generated<S>::bottom_forward(f, count, first, r);
        if constexpr (kWideBottom)
            if (count % 2 == 0) return WideBottom<S>::forward(f, count, first, r);
        Leaf::forward(f, count, first, r);
    }
    static void bottom_inverse(u32* f, std::size_t count, std::size_t first, const u32* ir) {
        if constexpr (requires { Generated<S>::bottom_inverse(f, count, first, ir); })
            if (count % 2 == 0) return Generated<S>::bottom_inverse(f, count, first, ir);
        if constexpr (kWideBottom)
            if (count % 2 == 0) return WideBottom<S>::inverse(f, count, first, ir);
        Leaf::inverse(f, count, first, ir);
    }
    // The bottom stages of Transform's fused ops (Bottom::multiply_stored, inverse_terms).
    static void bottom_multiply_stored(u32* a, const u32* b, std::size_t count, std::size_t first, const u32* r,
                                       const u32* ir) {
        if constexpr (Generated<S>::kOn)
            if (count % 2 == 0) return Generated<S>::multiply_stored(a, b, count, first, r, ir);
        if constexpr (kWideBottom)
            if (count % 2 == 0) return WideBottom<S>::multiply_stored(a, b, count, first, r, ir);
        Leaf::multiply_stored(a, b, count, first, r, ir);
    }
    static void bottom_inverse_terms(u32* out, const LeafTerm* terms, int n, std::size_t count, std::size_t first,
                                     const u32* r, const u32* ir) {
        if constexpr (requires { Generated<S>::inverse_terms(out, terms, n, count, first, r, ir); })
            if (count % 2 == 0 && Generated<S>::inverse_terms(out, terms, n, count, first, r, ir)) return;
        if constexpr (kWideBottom)
            if (count % 2 == 0) return WideBottom<S>::inverse_terms(out, terms, n, count, first, r, ir);
        Leaf::inverse_terms(out, terms, n, count, first, r, ir);
    }
    static void bottom_toeplitz(u32* out, const Toeplitz& z, std::size_t count, std::size_t first, const u32* r,
                                const u32* ir) {
        if constexpr (kWideBottom)
            if (count % 2 == 0) return WideBottom<S>::inverse_toeplitz(out, z, count, first, r, ir);
        Leaf::inverse_toeplitz(out, z, count, first, r, ir);
    }
    template <bool kSquare>
    static void bottom_multiply(u32* a, const u32* b, std::size_t count, std::size_t first, const u32* r,
                                const u32* ir) {
        if constexpr (Generated<S>::kOn)
            if (count % 2 == 0) return Generated<S>::template multiply<kSquare>(a, b, count, first, r, ir);
        if constexpr (kWideBottom)
            if (count % 2 == 0) return WideBottom<S>::template multiply<kSquare>(a, b, count, first, r, ir);
        Leaf::template multiply<kSquare>(a, b, count, first, r, ir);
    }

    // A subtree of 4 leaves (one group: 2^6 under the radix-2 level, 2^7 under group 0) is its
    // bottom stage alone. With kJoin (Generated<S>::kJoinBottoms) the top levels run the bottoms
    // of all their subtrees in one call (2 or 4 groups from group 0), so pair kernels take them.
    static constexpr bool kJoin = requires { Generated<S>::kJoinBottoms; };

    // A tile hook: tile(f, nv) runs on the nv leaves at f of a subtree of at most kTile leaves,
    // after its forward bottom level or before its inverse one (the values form, inverses from
    // other memory). NoTile: none.
    struct NoTile {
        [[gnu::always_inline]] void operator()(u32*, std::size_t) const {}
    };

    // Subtree of group k with nv leaves (a power of 4), depth first; its inputs < kIn P.
    template <int kIn = 4>
    static void forward_tree(u32* f, std::size_t nv, std::size_t k, const u32* r) {
        std::size_t h = nv / 4;
        if (nv > kTile) {
            B::template forward_group<kIn>(f, 8 * h, k, r);
            for (std::size_t t = 0; t < 4; ++t) forward_tree(f + 8 * h * t, h, 4 * k + t, r);
            return;
        }
        if (h >= 4) B::template forward_group<kIn>(f, 8 * h, k, r), h /= 4;  // the first level: group k alone
        for (; h >= 4; h /= 4) B::forward_groups(f, 8 * h, k * (nv / (4 * h)), nv / (4 * h), r);
        bottom_forward(f, nv / 4, k * (nv / 4), r);
    }

    // forward_tree and inverse_tree with a tile hook: copies, as avx2.hpp's forward_tiles.
    template <int kIn, class Tile>
    static void forward_tiles(u32* f, std::size_t nv, std::size_t k, const u32* r, Tile tile) {
        std::size_t h = nv / 4;
        if (nv > kTile) {
            B::template forward_group<kIn>(f, 8 * h, k, r);
            for (std::size_t t = 0; t < 4; ++t) forward_tiles<4>(f + 8 * h * t, h, 4 * k + t, r, tile);
            return;
        }
        if (h >= 4) B::template forward_group<kIn>(f, 8 * h, k, r), h /= 4;  // the first level: group k alone
        for (; h >= 4; h /= 4) B::forward_groups(f, 8 * h, k * (nv / (4 * h)), nv / (4 * h), r);
        bottom_forward(f, nv / 4, k * (nv / 4), r);
        tile(f, nv);
    }

    // The top group of a subtree, on quarters of q words: inverse_group, times s if scaled
    // (scaled_block(s)) is not null.
    static void inverse_top(u32* f, std::size_t q, std::size_t k, const u32* ir, const u32* scaled) {
        if (scaled) return B::inverse_group_scaled(f, q, k, ir, scaled);
        inverse_group(f, q, k, ir);
    }

    // Subtree of group k with nv leaves; its top group's outputs scaled if scaled is not null
    // (nv >= 16, k < 4).
    static void inverse_tree(u32* f, std::size_t nv, std::size_t k, const u32* ir, const u32* scaled = nullptr) {
        if (nv > kTile) {
            const std::size_t h = nv / 4;
            for (std::size_t t = 0; t < 4; ++t) inverse_tree(f + 8 * h * t, h, 4 * k + t, ir);
            return inverse_top(f, 8 * h, k, ir, scaled);
        }
        bottom_inverse(f, nv / 4, k * (nv / 4), ir);
        inverse_levels(f, nv, k, ir, scaled);
    }
    template <class Tile>
    static void inverse_tiles(u32* f, std::size_t nv, std::size_t k, const u32* ir, const u32* scaled, Tile tile) {
        if (nv > kTile) {
            const std::size_t h = nv / 4;
            for (std::size_t t = 0; t < 4; ++t) inverse_tiles(f + 8 * h * t, h, 4 * k + t, ir, nullptr, tile);
            return inverse_top(f, 8 * h, k, ir, scaled);
        }
        tile(f, nv);
        bottom_inverse(f, nv / 4, k * (nv / 4), ir);
        inverse_levels(f, nv, k, ir, scaled);
    }
    // The levels above the bottom of a subtree of at most kTile leaves.
    static void inverse_levels(u32* f, std::size_t nv, std::size_t k, const u32* ir, const u32* scaled) {
        for (std::size_t h = 4; h < nv / 4; h *= 4) B::inverse_groups(f, 8 * h, k * (nv / (4 * h)), nv / (4 * h), ir);
        if (nv >= 16) inverse_top(f, 2 * nv, k, ir, scaled);
    }

    // Columns per call of a product's first and last levels (detail::first_level, last_level):
    // each2 runs two at a time.
    static constexpr std::size_t kStep = 2 * S::lanes;

    // The words a subtree's first level reads: a pointer to all of them, or a Source: n words from
    // p, zero past them (a product from spans).
    struct Source {
        const u32* p;
        std::size_t n;
    };

    // Group k on quarters of q words from `from` into f; in < kIn P.
    template <int kIn>
    [[gnu::always_inline]] static void first_group(const u32* from, u32* f, std::size_t q, std::size_t k, const u32* r) {
        B::template forward_group<kIn>(from, f, q, k, r);
    }
    template <int kIn>
    static void first_group(Source s, u32* f, std::size_t q, std::size_t k, const u32* r) {
        if (s.n >= 4 * q) return B::template forward_group<kIn>(s.p, f, q, k, r);
        first_level<4>(s.p, s.n, f, q, kStep,
                       [&] [[gnu::always_inline]] (auto blocks, const u32* from, u32* to, std::size_t words) {
                           first_group_part<kIn, decltype(blocks)::value>(from, to, q, words, k, r);
                       });
    }
    template <int kIn, int kLoad>
    [[gnu::noinline]] static void first_group_part(const u32* from, u32* f, std::size_t q, std::size_t words,
                                                   std::size_t k, const u32* r) {
        B::template forward_group_part<kIn, kLoad>(from, f, q, words, k, r);
    }

    // Both transforms, the leaf products and the inverse transform of a subtree; inputs < kIn P.
    // The first level reads a_from and b_from (a and b, or other memory if nv >= 16): pointers or
    // Sources (first_group). kSquare: a times a; b and b_from not read.
    template <int kIn = 4, bool kSquare = false, class From>
    static void multiply_tree(From a_from, From b_from, u32* a, u32* b, std::size_t nv, std::size_t k, const u32* r,
                              const u32* ir, const u32* scaled = nullptr) {
        std::size_t h = nv / 4;
        if (nv > kTile) {
            first_group<kIn>(a_from, a, 8 * h, k, r);
            if constexpr (!kSquare) first_group<kIn>(b_from, b, 8 * h, k, r);
            for (std::size_t t = 0; t < 4; ++t)
                multiply_tree<4, kSquare>(a + 8 * h * t, b + 8 * h * t, h, 4 * k + t, r, ir);
            return inverse_top(a, 8 * h, k, ir, scaled);
        }
        if (h >= 4) {  // the first level: group k alone
            first_group<kIn>(a_from, a, 8 * h, k, r);
            if constexpr (!kSquare) first_group<kIn>(b_from, b, 8 * h, k, r);
            h /= 4;
        }
        for (; h >= 4; h /= 4) {
            const std::size_t g = k * (nv / (4 * h)), count = nv / (4 * h);
            B::forward_groups(a, 8 * h, g, count, r);
            if constexpr (!kSquare) B::forward_groups(b, 8 * h, g, count, r);
        }
        bottom_multiply<kSquare>(a, b, nv / 4, k * (nv / 4), r, ir);
        inverse_levels(a, nv, k, ir, scaled);
    }
    template <int kIn = 4, bool kSquare = false>
    static void multiply_tree(u32* a, u32* b, std::size_t nv, std::size_t k, const u32* r, const u32* ir,
                              const u32* scaled = nullptr) {
        multiply_tree<kIn, kSquare>(a, b, a, b, nv, k, r, ir, scaled);
    }

    // Top levels. Inputs canonical. If the upper half of f is zero ("sparse"), it is not read.

    // The radix-2 level on halves of `half` words: out < 2P.
    static void forward_radix2(u32* f, std::size_t half, bool sparse) {
        if (sparse) {
            std::memcpy(f + half, f, half * sizeof(u32));
            return;
        }
        const V p = S::broadcast(kP);
        B::prefetching(half, kPrefetchHalfMin, [&]<bool kPf> [[gnu::always_inline]] (std::bool_constant<kPf>) {
            for (std::size_t i = 0; i < half; i += S::lanes) {
                B::template ahead<kPf>(f + i), B::template ahead<kPf>(f + i + half);
                const V x = S::load(f + i), y = S::load(f + i + half);
                S::store(f + i, add(x, y)), S::store(f + i + half, S::sub(S::add(x, p), y));
            }
        });
    }

    // Group 0 on quarters of q words: out < 4P.
    static void forward_radix4(u32* f, std::size_t q, bool sparse, const u32* r) {
        if (!sparse) return B::template forward_group<1>(f, q, 0, r);
        const F z = B::entry(r, 1);
        B::prefetching(q, kPrefetchMin, [&]<bool kPf> [[gnu::always_inline]] (std::bool_constant<kPf>) {
            for (std::size_t j = 0; j < q; j += S::lanes) {
                B::template ahead<kPf>(f + j), B::template ahead<kPf>(f + j + q);
                B::template ahead_write<kPf>(f + j + 2 * q), B::template ahead_write<kPf>(f + j + 3 * q);
                const V a = S::load(f + j), b = S::load(f + j + q), zb = B::mul(b, z);
                S::store(f + j, add(a, b)), S::store(f + j + q, diff(a, b));
                S::store(f + j + 2 * q, add(a, zb)), S::store(f + j + 3 * q, diff(a, zb));
            }
        });
    }

    // The inverse radix-2 level, times s if s != 0: in < 2P, canonical out.
    static void inverse_radix2(u32* f, std::size_t half, u32 s) {
        if constexpr (requires { Generated<S>::template inverse_radix2<2, false>(f, f, half, half, s); })
            if (Generated<S>::template inverse_radix2<2, false>(f, f, half, half, s)) return;
        B::prefetching(half, kPrefetchHalfMin, [&]<bool kPf> [[gnu::always_inline]] (std::bool_constant<kPf>) {
            if (s == 0) {
                for (std::size_t i = 0; i < half; i += S::lanes) {
                    B::template ahead<kPf>(f + i), B::template ahead<kPf>(f + i + half);
                    const V x = S::load(f + i), y = S::load(f + i + half);
                    S::store(f + i, canonical(B::low(add(x, y)))), S::store(f + i + half, canonical(B::low(diff(x, y))));
                }
                return;
            }
            const F scale = B::factor(s);
            for (std::size_t i = 0; i < half; i += S::lanes) {
                B::template ahead<kPf>(f + i), B::template ahead<kPf>(f + i + half);
                const V x = S::load(f + i), y = S::load(f + i + half);
                S::store(f + i, canonical(B::mul(add(x, y), scale)));
                S::store(f + i + half, canonical(B::mul(diff(x, y), scale)));
            }
        });
    }

    // Inverse group 0 on quarters of q words, times s if s != 0: in < 2P, canonical out. With s, 4
    // products per column, not 5: s z (c - d) and s (a - b) serve two outputs each.
    static void inverse_radix4(u32* f, std::size_t q, const u32* ir, u32 s) {
        if constexpr (requires { Generated<S>::template inverse_top<4, false>(f, f, q, q, ir, s); })
            if (Generated<S>::template inverse_top<4, false>(f, f, q, q, ir, s)) return;
        if (s == 0) {
            const F z = B::entry(ir, 1);
            return B::template each2<kPipelined.inverse_identity>(
                f, f, q, [&] [[gnu::always_inline]] (Column& v) { B::inverse_identity_first(v, z); },
                [&] [[gnu::always_inline]] (Column& v) {
                    B::inverse_identity_second(v);
                    v[0] = canonical(v[0]), v[1] = canonical(v[1]), v[2] = canonical(v[2]), v[3] = canonical(v[3]);
                });
        }
        const F scale = B::factor(s), sz = B::factor(multiply_mod(s, ir[slot(1)]));
        B::template each2<kPipelined.groups>(
            f, f, q, [&] [[gnu::always_inline]] (Column& v) { B::inverse_first(v, scale, sz); },
            [&] [[gnu::always_inline]] (Column& v) {
                const V ab = v[0], cd = v[1], amb = v[2], cmd = v[3];
                v[0] = canonical(B::mul(add(ab, cd), scale)), v[1] = canonical(B::low(add(amb, cmd)));
                v[2] = canonical(B::mul(diff(ab, cd), scale)), v[3] = canonical(B::low(diff(amb, cmd)));
            });
    }

    // Products from spans (detail::Product). The first level reads a factor from a source of n
    // words in caller memory, zero past them (detail::first_level), and the last level writes the
    // product's first count words to caller memory (detail::last_level), any alignment. The *_part
    // functions run the columns [0, words) of a level from `from` to f: blocks t < kLoad read
    // (forward), t < kStore written (inverse); each is its own function, picked with an inlined
    // switch (avx2.hpp, Products from spans). Products in place keep the levels above.

    // The radix-2 level on halves of `half` words: out < 2P. One half read: copied to both.
    template <int kLoad>
    [[gnu::noinline]] static void forward_radix2_part(const u32* from, u32* f, std::size_t half, std::size_t words) {
        const V p = S::broadcast(kP);
        const bool write = from != f;
        B::prefetching(half, kPrefetchHalfMin, [&]<bool kPf> [[gnu::always_inline]] (std::bool_constant<kPf>) {
            for (std::size_t i = 0; i < words; i += S::lanes) {
                if constexpr (kLoad > 0) B::template ahead<kPf>(from + i);
                if constexpr (kLoad > 1) B::template ahead<kPf>(from + i + half);
                if (write || kLoad == 0) B::template ahead_write<kPf>(f + i);
                if (write || kLoad < 2) B::template ahead_write<kPf>(f + i + half);
                const V x = kLoad > 0 ? S::load(from + i) : S::broadcast(0);
                if constexpr (kLoad < 2) {
                    if (write || kLoad == 0) S::store(f + i, x);
                    S::store(f + i + half, x);
                } else {
                    const V y = S::load(from + i + half);
                    S::store(f + i, add(x, y)), S::store(f + i + half, S::sub(S::add(x, p), y));
                }
            }
        });
    }
    static void forward_radix2_from(const u32* from, std::size_t n, u32* f, std::size_t half) {
        first_level<2>(from, n, f, half, S::lanes,
                       [&] [[gnu::always_inline]] (auto blocks, const u32* s, u32* d, std::size_t words) {
                           forward_radix2_part<decltype(blocks)::value>(s, d, half, words);
                       });
    }

    // Group 0 on quarters of q words, from the quarters t < kLoad of `from`: out < 4P. Two quarters
    // read: a + b, a - b, a + z b, a - z b.
    template <int kLoad>
    [[gnu::noinline]] static void forward_radix4_part(const u32* from, u32* f, std::size_t q, std::size_t words,
                                                      const u32* r) {
        if constexpr (kLoad == 2) forward_radix4_sparse(from, f, q, words, r);
        else B::template forward_group_part<1, kLoad>(from, f, q, words, 0, r);
    }
    [[gnu::always_inline]] static void forward_radix4_sparse(const u32* from, u32* f, std::size_t q, std::size_t words,
                                                             const u32* r) {
        const F z = B::entry(r, 1);
        B::prefetching(q, kPrefetchMin, [&]<bool kPf> [[gnu::always_inline]] (std::bool_constant<kPf>) {
            const bool write = from != f;
            for (std::size_t j = 0; j < words; j += S::lanes) {
                B::template ahead<kPf, 2>(from + j, q), B::template ahead_write<kPf, 2>(write, f + j, q);
                const V a = S::load(from + j), b = S::load(from + j + q), zb = B::mul(b, z);
                S::store(f + j, add(a, b)), S::store(f + j + q, diff(a, b));
                S::store(f + j + 2 * q, add(a, zb)), S::store(f + j + 3 * q, diff(a, zb));
            }
        });
    }
    static void forward_radix4_from(const u32* from, std::size_t n, u32* f, std::size_t q, const u32* r) {
        first_level<4>(from, n, f, q, kStep,
                       [&] [[gnu::always_inline]] (auto blocks, const u32* s, u32* d, std::size_t words) {
                           forward_radix4_part<decltype(blocks)::value>(s, d, q, words, r);
                       });
    }

    // The inverse radix-2 level, times s if s != 0: in < 2P, canonical out.
    // kUpper: the differences (the upper half) to the first block (kStore = 1).
    template <int kStore, bool kUpper = false>
    [[gnu::noinline]] static void inverse_radix2_part(const u32* from, u32* f, std::size_t half, std::size_t words,
                                                      u32 s) {
        if constexpr (requires { Generated<S>::template inverse_radix2<kStore, kUpper>(from, f, half, words, s); })
            if (Generated<S>::template inverse_radix2<kStore, kUpper>(from, f, half, words, s)) return;
        if (backwards(from, f)) return inverse_radix2_backwards<kStore, kUpper>(from, f, half, words, s);
        B::prefetching(half, kPrefetchHalfMin, [&]<bool kPf> [[gnu::always_inline]] (std::bool_constant<kPf>) {
            if (s == 0) {
                for (std::size_t i = 0; i < words; i += S::lanes) {
                    B::template ahead<kPf>(from + i), B::template ahead<kPf>(from + i + half);
                    const V x = S::load(from + i), y = S::load(from + i + half);
                    if constexpr (kUpper) {
                        S::store(f + i, canonical(B::low(diff(x, y))));
                        continue;
                    }
                    S::store(f + i, canonical(B::low(add(x, y))));
                    if constexpr (kStore > 1) S::store(f + i + half, canonical(B::low(diff(x, y))));
                }
                return;
            }
            const F scale = B::factor(s);
            for (std::size_t i = 0; i < words; i += S::lanes) {
                B::template ahead<kPf>(from + i), B::template ahead<kPf>(from + i + half);
                const V x = S::load(from + i), y = S::load(from + i + half);
                if constexpr (kUpper) {
                    S::store(f + i, canonical(B::mul(diff(x, y), scale)));
                    continue;
                }
                S::store(f + i, canonical(B::mul(add(x, y), scale)));
                if constexpr (kStore > 1) S::store(f + i + half, canonical(B::mul(diff(x, y), scale)));
            }
        });
    }
    // inverse_radix2_part with the columns last to first (common.hpp's backwards), no prefetch.
    template <int kStore, bool kUpper>
    [[gnu::noinline, gnu::cold]] static void inverse_radix2_backwards(const u32* from, u32* f, std::size_t half,
                                                                      std::size_t words, u32 s) {
        const auto columns = [&] [[gnu::always_inline]] (auto product) {
            for (std::size_t i = words; i;) {
                i -= S::lanes;
                const V x = S::load(from + i), y = S::load(from + i + half);
                if constexpr (kUpper) {
                    S::store(f + i, canonical(product(diff(x, y))));
                    continue;
                }
                S::store(f + i, canonical(product(add(x, y))));
                if constexpr (kStore > 1) S::store(f + i + half, canonical(product(diff(x, y))));
            }
        };
        if (s == 0) return columns([] [[gnu::always_inline]] (V v) { return B::low(v); });
        const F scale = B::factor(s);
        columns([&] [[gnu::always_inline]] (V v) { return B::mul(v, scale); });
    }
    static void inverse_radix2_out(u32* f, std::size_t half, u32* out, std::size_t count, u32 s) {
        last_level<2>(f, half, out, count, S::lanes, [&] [[gnu::always_inline]] (auto blocks, const u32* from, u32* to,
                                                                 std::size_t words) {
            constexpr int T = decltype(blocks)::value;
            if constexpr (T > 0) inverse_radix2_part<T>(from, to, half, words, s);
        });
    }
    // The upper half of the result, count <= half words of it to out.
    static void inverse_radix2_upper_out(u32* f, std::size_t half, u32* out, std::size_t count, u32 s) {
        last_level<1>(f, half, out, count, S::lanes, [&] [[gnu::always_inline]] (auto blocks, const u32* from, u32* to,
                                                                 std::size_t words) {
            if constexpr (decltype(blocks)::value > 0) inverse_radix2_part<1, true>(from, to, half, words, s);
        });
    }

    // Inverse group 0 on quarters of q words, times s if s != 0: in < 2P, canonical out. With s, 4
    // products per column, not 5: s z (c - d) and s (a - b) serve two outputs each.
    // kUpper: quarters 2 + t (the upper half) to quarters t < kStore <= 2.
    template <int kStore, bool kUpper = false>
    [[gnu::noinline]] static void inverse_radix4_part(const u32* from, u32* f, std::size_t q, std::size_t words,
                                                      const u32* ir, u32 s) {
        if constexpr (requires { Generated<S>::template inverse_top<kStore, kUpper>(from, f, q, words, ir, s); })
            if (Generated<S>::template inverse_top<kStore, kUpper>(from, f, q, words, ir, s)) return;
        if (s == 0) {
            const F z = B::entry(ir, 1);
            return B::template each2<kPipelined.inverse_identity, 4, kStore>(
                from, f, q, words, [&] [[gnu::always_inline]] (Column& v) { B::inverse_identity_first(v, z); },
                [&] [[gnu::always_inline]] (Column& v) {
                    B::inverse_identity_second(v);
                    v[0] = canonical(v[0]), v[1] = canonical(v[1]), v[2] = canonical(v[2]), v[3] = canonical(v[3]);
                    if constexpr (kUpper) v[0] = v[2], v[1] = v[3];
                });
        }
        const F scale = B::factor(s), sz = B::factor(multiply_mod(s, ir[slot(1)]));
        B::template each2<kPipelined.groups, 4, kStore>(
            from, f, q, words, [&] [[gnu::always_inline]] (Column& v) { B::inverse_first(v, scale, sz); },
            [&] [[gnu::always_inline]] (Column& v) {
                const V ab = v[0], cd = v[1], amb = v[2], cmd = v[3];
                v[0] = canonical(B::mul(add(ab, cd), scale)), v[1] = canonical(B::low(add(amb, cmd)));
                v[2] = canonical(B::mul(diff(ab, cd), scale)), v[3] = canonical(B::low(diff(amb, cmd)));
                if constexpr (kUpper) v[0] = v[2], v[1] = v[3];
            });
    }
    // The upper half of the result, count <= 2q words of it to out.
    static void inverse_radix4_upper_out(u32* f, std::size_t q, u32* out, std::size_t count, const u32* ir, u32 s) {
        last_level<2>(f, q, out, count, kStep, [&] [[gnu::always_inline]] (auto blocks, const u32* from, u32* to,
                                                           std::size_t words) {
            constexpr int T = decltype(blocks)::value;
            if constexpr (T > 0) inverse_radix4_part<T, true>(from, to, q, words, ir, s);
        });
    }
    static void inverse_radix4_out(u32* f, std::size_t q, u32* out, std::size_t count, const u32* ir, u32 s) {
        last_level<4>(f, q, out, count, kStep, [&] [[gnu::always_inline]] (auto blocks, const u32* from, u32* to,
                                                           std::size_t words) {
            constexpr int T = decltype(blocks)::value;
            if constexpr (T > 0) inverse_radix4_part<T>(from, to, q, words, ir, s);
        });
    }

    // The scale goes into the top groups of the subtrees below the top level when those have one
    // (nv >= 16 leaves) and lg <= kFoldMaxLog: their scaled twiddles, else null (the top level
    // scales). Per vector: 0.5 products fewer and 1.25 other ops more under a radix-2 top, 0.375
    // products fewer under a radix-4 top (notes.md). kind: kScaledRoots' (0 inverse, 1 products,
    // 2 inverse from the values form).
    static const u32* fold_scale(std::size_t nv, int lg, int kind) {
        return nv >= 16 && lg <= kFoldMaxLog ? kScaledRoots[kind][lg].words : nullptr;
    }

    // Top levels by lg: nv = 4^j leaves start with group 0; nv = 2 * 4^j with the radix-2 level.
    static void forward(u32* f, int lg, const u32* r) {
        const std::size_t len = std::size_t(1) << lg, nv = len / 8;
        if (!has_radix2(lg)) return forward_tree<1>(f, nv, 0, r);
        forward_radix2(f, len / 2, false);
        if (kJoin && nv == 8) return bottom_forward(f, 2, 0, r);
        forward_tree<2>(f, nv / 2, 0, r);
        forward_tree<2>(f + len / 2, nv / 2, 1, r);
    }

    // The upper half of the leaf form of length 2L, L = 2^lg, from f = p mod (x^L + 1), canonical;
    // r: the table of length 2L. Below a radix-2 level that is group 1's subtree; below group 0
    // (radix 4) those of groups 2 and 3, from the residues mod x^(L/2) -+ r(1) (out < 3P).
    static void forward_upper(u32* f, int lg, const u32* r) {
        const std::size_t len = std::size_t(1) << lg, nv = len / 8, half = len / 2;
        if (has_radix2(lg + 1)) return forward_tree<1>(f, nv, 1, r);
        const F z = B::entry(r, 1);
        for (std::size_t i = 0; i < half; i += S::lanes) {
            const V x = S::load(f + i), y = B::mul(S::load(f + i + half), z);
            S::store(f + i, add(x, y)), S::store(f + i + half, diff(x, y));
        }
        if (kJoin && nv == 8) return bottom_forward(f, 2, 2, r);
        forward_tree<4>(f, nv / 2, 2, r);
        forward_tree<4>(f + half, nv / 2, 3, r);
    }

    // forward_upper undone, times s: the upper half of the leaf form of length 2L, L = 2^lg, to
    // p mod (x^L + 1), canonical; ir: the table of length 2L.
    static void inverse_upper(u32* f, int lg, const u32* ir, u32 s) {
        const std::size_t len = std::size_t(1) << lg, nv = len / 8, half = len / 2;
        const F scale = B::factor(s);
        if (has_radix2(lg + 1)) {
            inverse_tree(f, nv, 1, ir);
            for (std::size_t i = 0; i < len; i += S::lanes) S::store(f + i, canonical(B::mul(S::load(f + i), scale)));
            return;
        }
        if (kJoin && nv == 8) {
            bottom_inverse(f, 2, 2, ir);
        } else {
            inverse_tree(f, nv / 2, 2, ir);
            inverse_tree(f + half, nv / 2, 3, ir);
        }
        const F sz = B::factor(multiply_mod(s, ir[slot(1)]));
        for (std::size_t i = 0; i < half; i += S::lanes) {
            const V x = S::load(f + i), y = S::load(f + i + half);
            S::store(f + i, canonical(B::mul(add(x, y), scale)));
            S::store(f + i + half, canonical(B::mul(diff(x, y), sz)));
        }
    }

    static void inverse(u32* f, int lg, const u32* ir, u32 s) {
        const std::size_t len = std::size_t(1) << lg, nv = len / 8;
        if (!has_radix2(lg)) {
            const u32* fold = fold_scale(nv / 4, lg, false);
            if (kJoin && nv == 16) bottom_inverse(f, 4, 0, ir);
            else for (std::size_t t = 0; t < 4; ++t) inverse_tree(f + t * len / 4, nv / 4, t, ir, fold);
            return inverse_radix4(f, len / 4, ir, fold ? 0 : s);
        }
        const u32* fold = fold_scale(nv / 2, lg, false);
        if (kJoin && nv == 8) {
            bottom_inverse(f, 2, 0, ir);
        } else {
            inverse_tree(f, nv / 2, 0, ir, fold);
            inverse_tree(f + len / 2, nv / 2, 1, ir, fold);
        }
        inverse_radix2(f, len / 2, fold ? 0 : s);
    }

    static void multiply(u32* a, u32* b, std::size_t n, std::size_t m, int lg, const u32* r, const u32* ir) {
        const std::size_t len = std::size_t(1) << lg, nv = len / 8;
        const bool sparse_a = n <= len / 2, sparse_b = m <= len / 2;
        const u32 s = product_scale(lg);
        if (!has_radix2(lg)) {
            const std::size_t q = len / 4;
            const u32* fold = fold_scale(nv / 4, lg, true);
            forward_radix4(a, q, sparse_a, r);
            forward_radix4(b, q, sparse_b, r);
            if (kJoin && nv == 16) bottom_multiply<false>(a, b, 4, 0, r, ir);
            else for (std::size_t t = 0; t < 4; ++t) multiply_tree(a + t * q, b + t * q, nv / 4, t, r, ir, fold);
            return inverse_radix4(a, q, ir, fold ? 0 : s);
        }
        // After the radix-2 level a sparse factor's upper half equals its lower half, so the upper
        // subtree's first level reads the lower half instead of a copy, before the lower subtree runs
        // in place. Not for nv / 2 = 4, whose first level is the bottom (in place only).
        const std::size_t half = len / 2;
        const bool share_a = sparse_a && nv / 2 >= 16, share_b = sparse_b && nv / 2 >= 16;
        if (!share_a) forward_radix2(a, half, sparse_a);
        if (!share_b) forward_radix2(b, half, sparse_b);
        const u32* fold = fold_scale(nv / 2, lg, true);
        if (kJoin && nv == 8) {
            bottom_multiply<false>(a, b, 2, 0, r, ir);
        } else {
            multiply_tree<2>(share_a ? a : a + half, share_b ? b : b + half, a + half, b + half, nv / 2, 1, r, ir, fold);
            multiply_tree<2>(a, b, nv / 2, 0, r, ir, fold);
        }
        inverse_radix2(a, half, fold ? 0 : s);
    }

    // Leaf-form operations (ntt.hpp, Transform). Leaf products: on N, a group of 4 leaves at a
    // time; on 16 lanes on Zen 4 and 5 WideBottom's, two pairs of groups at a time (groups: even,
    // 2^(lg - 5)), up to 2^kWideProductsMaxLog. Against N's on Zen 5: 0.65 of its time to 2^14,
    // 0.84 at 2^20, 1.9x at 2^22 (memory; not understood); Emerald Rapids 0.97-1.00 on L1 data
    // (notes/98-r1.md).
#if defined(__znver4__) || defined(__znver5__)
    static constexpr bool kWideProducts = kWideBottom;
#else
    static constexpr bool kWideProducts = false;
#endif
    static constexpr int kWideProductsMaxLog = 20;
    static void leaf_products(const u32* a, const u32* b, u32* out, int lg, const u32* r, bool accumulate) {
        const std::size_t groups = (std::size_t(1) << lg) / 32;
        if constexpr (kWideProducts) {
            if (lg <= kWideProductsMaxLog) {
                if (accumulate) return WideBottom<S>::template products<true>(a, b, out, groups, 0, r);
                return WideBottom<S>::template products<false>(a, b, out, groups, 0, r);
            }
        }
        if (accumulate) return Leaf::template products<true>(a, b, out, groups, 0, r);
        Leaf::template products<false>(a, b, out, groups, 0, r);
    }

    // The values form: the split runs on each subtree of at least 8 leaves after its bottom level,
    // while its words are in cache. Below 2^8 a subtree has 4 leaves (half a block): split after.
    static void forward_values(u32* f, int lg, const u32* r, const u32* rows) {
        const std::size_t len = std::size_t(1) << lg, nv = len / 8;
        if (lg < 8) {
            forward(f, lg, r);
            return ValuesFor<S>::split(f, f, len / 64, 0, rows);
        }
        const auto tile = [f, rows] [[gnu::always_inline]] (u32* t, std::size_t leaves) {
            ValuesFor<S>::split(t, t, leaves / 8, std::size_t(t - f) / 64, rows);
        };
        if (!has_radix2(lg)) return forward_tiles<1>(f, nv, 0, r, tile);
        forward_radix2(f, len / 2, false);
        forward_tiles<2>(f, nv / 2, 0, r, tile);
        forward_tiles<2>(f + len / 2, nv / 2, 1, r, tile);
    }

    // The leaf form (rows null) or values form at `from` -> coefficients [0, count) to out, through
    // f (aligned, may be from; out may be f, then count = 2^lg). The subtrees read their leaves from
    // `from` (copied, or joined, into f before their bottom level), and the top level writes out.
    // s: leaf_scale(lg), or values_scale(lg) from the values form (fold_scale's tables are those).
    static void inverse(const u32* from, u32* f, u32* out, std::size_t count, int lg, const u32* ir, u32 s,
                        const u32* rows) {
        const std::size_t len = std::size_t(1) << lg;
        // The join's leaves are < 2P, as every bottom level here takes them.
        if (rows && lg < 8) {
            ValuesFor<S>::template join<false>(from, f, len / 64, 0, rows);
            return inverse_into(f, out, count, lg, ir, s, 2, NoTile{});
        }
        if (rows) {
            const auto join = [from, f, rows] [[gnu::always_inline]] (u32* t, std::size_t leaves) {
                const std::size_t at = std::size_t(t - f);
                ValuesFor<S>::template join<false>(from + at, t, leaves / 8, at / 64, rows);
            };
            return inverse_into(f, out, count, lg, ir, s, 2, join);
        }
        if (from != f) {
            const auto copy = [from, f] [[gnu::always_inline]] (u32* t, std::size_t leaves) {
                std::memcpy(t, from + (t - f), 8 * leaves * sizeof(u32));
            };
            return inverse_into(f, out, count, lg, ir, s, 0, copy);
        }
        inverse_into(f, out, count, lg, ir, s, 0, NoTile{});
    }

    // inverse_into at 2^6 and 2^7 with kJoin (the subtrees' bottoms in one call), in a function of
    // its own: with the joins inlined in inverse_into, its code for the other lengths changed and the
    // zmm variant's inverse of the values form at 2^9 ran 11% slower on Zen 5 (notes/176-r1.md).
    template <class Tile>
    [[gnu::noinline]] static void inverse_into_joined(u32* f, u32* out, std::size_t count, int lg, const u32* ir, u32 s,
                                                      int kind, Tile tile) {
        const std::size_t len = std::size_t(1) << lg, nv = len / 8;
        tile(f, nv), bottom_inverse(f, nv / 4, 0, ir);
        if (!has_radix2(lg)) {
            const u32* fold = fold_scale(nv / 4, lg, kind);
            if (out == f) return inverse_radix4(f, len / 4, ir, fold ? 0 : s);
            return inverse_radix4_out(f, len / 4, out, count, ir, fold ? 0 : s);
        }
        const u32* fold = fold_scale(nv / 2, lg, kind);
        if (out == f) return inverse_radix2(f, len / 2, fold ? 0 : s);
        inverse_radix2_out(f, len / 2, out, count, fold ? 0 : s);
    }
    // kind: fold_scale's, for s.
    template <class Tile>
    static void inverse_into(u32* f, u32* out, std::size_t count, int lg, const u32* ir, u32 s, int kind, Tile tile) {
        if constexpr (kJoin)
            if (lg <= 7) return inverse_into_joined(f, out, count, lg, ir, s, kind, tile);
        const std::size_t len = std::size_t(1) << lg, nv = len / 8;
        if (!has_radix2(lg)) {
            const u32* fold = fold_scale(nv / 4, lg, kind);
            for (std::size_t t = 0; t < 4; ++t) inverse_tiles(f + t * len / 4, nv / 4, t, ir, fold, tile);
            if (out == f) return inverse_radix4(f, len / 4, ir, fold ? 0 : s);
            return inverse_radix4_out(f, len / 4, out, count, ir, fold ? 0 : s);
        }
        const u32* fold = fold_scale(nv / 2, lg, kind);
        inverse_tiles(f, nv / 2, 0, ir, fold, tile);
        inverse_tiles(f + len / 2, nv / 2, 1, ir, fold, tile);
        if (out == f) return inverse_radix2(f, len / 2, fold ? 0 : s);
        inverse_radix2_out(f, len / 2, out, count, fold ? 0 : s);
    }

    // The product p (detail::Product).
    static void multiply(const Product& p, const u32* r, const u32* ir) {
        const std::size_t len = std::size_t(1) << p.lg, nv = len / 8;
        const u32 s = product_scale(p.lg);
        if (!has_radix2(p.lg)) {
            const std::size_t q = len / 4;
            const u32* fold = fold_scale(nv / 4, p.lg, true);
            forward_radix4_from(p.f, p.n, p.a, q, r);
            forward_radix4_from(p.g, p.m, p.b, q, r);
            if (kJoin && nv == 16) bottom_multiply<false>(p.a, p.b, 4, 0, r, ir);
            else for (std::size_t t = 0; t < 4; ++t) multiply_tree(p.a + t * q, p.b + t * q, nv / 4, t, r, ir, fold);
            return inverse_radix4_out(p.a, q, p.out, p.count, ir, fold ? 0 : s);
        }
        // A factor of at most half the length ("sparse") is its own residue mod x^(len/2) - 1 and
        // mod x^(len/2) + 1: no radix-2 level, both subtrees' first levels read it, the upper one
        // first. Not for nv / 2 = 4, whose first level is the bottom (in place only).
        const std::size_t half = len / 2;
        const bool share_a = p.n <= half && nv / 2 >= 16, share_b = p.m <= half && nv / 2 >= 16;
        if (!share_a) forward_radix2_from(p.f, p.n, p.a, half);
        if (!share_b) forward_radix2_from(p.g, p.m, p.b, half);
        const auto source = [half](bool share, const u32* f, std::size_t n, u32* a) {
            return share ? Source{f, n} : Source{a, half};
        };
        const u32* fold = fold_scale(nv / 2, p.lg, true);
        if (kJoin && nv == 8) {
            bottom_multiply<false>(p.a, p.b, 2, 0, r, ir);
        } else {
            multiply_tree<2>(source(share_a, p.f, p.n, p.a + half), source(share_b, p.g, p.m, p.b + half), p.a + half,
                             p.b + half, nv / 2, 1, r, ir, fold);
            multiply_tree<2>(source(share_a, p.f, p.n, p.a), source(share_b, p.g, p.m, p.b), p.a, p.b, nv / 2, 0, r, ir,
                             fold);
        }
        inverse_radix2_out(p.a, half, p.out, p.count, fold ? 0 : s);
    }

    // The square of p.f (g, m and b unused): multiply(p) with one forward transform.
    static void square(const Product& p, const u32* r, const u32* ir) {
        const std::size_t len = std::size_t(1) << p.lg, nv = len / 8;
        const u32 s = product_scale(p.lg);
        if (!has_radix2(p.lg)) {
            const std::size_t q = len / 4;
            const u32* fold = fold_scale(nv / 4, p.lg, true);
            forward_radix4_from(p.f, p.n, p.a, q, r);
            if (kJoin && nv == 16) bottom_multiply<true>(p.a, p.a, 4, 0, r, ir);
            else for (std::size_t t = 0; t < 4; ++t) multiply_tree<4, true>(p.a + t * q, p.a + t * q, nv / 4, t, r, ir, fold);
            return inverse_radix4_out(p.a, q, p.out, p.count, ir, fold ? 0 : s);
        }
        const std::size_t half = len / 2;
        const bool share = p.n <= half && nv / 2 >= 16;
        if (!share) forward_radix2_from(p.f, p.n, p.a, half);
        const auto source = [&](u32* a) { return share ? Source{p.f, p.n} : Source{a, half}; };
        const u32* fold = fold_scale(nv / 2, p.lg, true);
        if (kJoin && nv == 8) {
            bottom_multiply<true>(p.a, p.a, 2, 0, r, ir);
        } else {
            multiply_tree<2, true>(source(p.a + half), source(p.a + half), p.a + half, p.a + half, nv / 2, 1, r, ir, fold);
            multiply_tree<2, true>(source(p.a), source(p.a), p.a, p.a, nv / 2, 0, r, ir, fold);
        }
        inverse_radix2_out(p.a, half, p.out, p.count, fold ? 0 : s);
    }

    // Transform's fused ops (ntt.hpp): a forward transform from in (n <= 2^lg words of caller
    // memory, zero past them) into f, its leaves handed to a tile hook in cache, and for a cyclic
    // product the inverse transform after the hook, its top level writing count words to out.

    // Subtree of group k with nv leaves from `from` (a pointer, or a Source of the caller's words;
    // in < kIn P): forward levels, bottom, tile(f, nv).
    template <int kIn = 4, class From, class Tile>
    static void forward_tree_from(From from, u32* f, std::size_t nv, std::size_t k, const u32* r, Tile tile) {
        std::size_t h = nv / 4;
        if (nv > kTile) {
            first_group<kIn>(from, f, 8 * h, k, r);
            for (std::size_t t = 0; t < 4; ++t) forward_tree_from<4>(f + 8 * h * t, f + 8 * h * t, h, 4 * k + t, r, tile);
            return;
        }
        if (h >= 4) first_group<kIn>(from, f, 8 * h, k, r), h /= 4;
        for (; h >= 4; h /= 4) B::forward_groups(f, 8 * h, k * (nv / (4 * h)), nv / (4 * h), r);
        bottom_forward(f, nv / 4, k * (nv / 4), r);
        tile(f, nv);
    }

    // The same with the bottom stage stage(f, nv) (forward bottom level, leaf products, inverse
    // bottom level), then the inverse levels; the top group's outputs scaled if scaled is not null.
    template <int kIn = 4, class From, class Stage>
    static void cycle_tree(From from, u32* f, std::size_t nv, std::size_t k, const u32* r, const u32* ir,
                           const u32* scaled, Stage stage) {
        std::size_t h = nv / 4;
        if (nv > kTile) {
            first_group<kIn>(from, f, 8 * h, k, r);
            for (std::size_t t = 0; t < 4; ++t)
                cycle_tree<4>(f + 8 * h * t, f + 8 * h * t, h, 4 * k + t, r, ir, nullptr, stage);
            return inverse_top(f, 8 * h, k, ir, scaled);
        }
        if (h >= 4) first_group<kIn>(from, f, 8 * h, k, r), h /= 4;
        for (; h >= 4; h /= 4) B::forward_groups(f, 8 * h, k * (nv / (4 * h)), nv / (4 * h), r);
        stage(f, nv);
        inverse_levels(f, nv, k, ir, scaled);
    }

    // inverse_tree with the bottom level replaced by stage(f, nv), which writes the leaves and runs
    // the inverse bottom level.
    template <class Stage>
    static void inverse_tree_with(u32* f, std::size_t nv, std::size_t k, const u32* ir, const u32* scaled, Stage stage) {
        if (nv > kTile) {
            const std::size_t h = nv / 4;
            for (std::size_t t = 0; t < 4; ++t) inverse_tree_with(f + 8 * h * t, h, 4 * k + t, ir, nullptr, stage);
            return inverse_top(f, 8 * h, k, ir, scaled);
        }
        stage(f, nv);
        inverse_levels(f, nv, k, ir, scaled);
    }

    // The forward top level from in into f, then sub(kIn, from, g, nv, k) on each subtree of group k
    // (nv leaves at g, read from `from`), as multiply(p): a source of at most half the length is read
    // by both subtrees' first levels, with no radix-2 level (from 2^7 on).
    // With kJoin and a joined callable, joined(count) runs the subtrees of one group each at f
    // instead (count 2 or 4).
    struct NoJoin {};
    template <class Sub, class Joined = NoJoin>
    [[gnu::always_inline]] static void from_top(const u32* in, std::size_t n, u32* f, int lg, const u32* r, Sub sub,
                                                Joined joined = {}) {
        const std::size_t len = std::size_t(1) << lg, nv = len / 8;
        constexpr bool kJoined = kJoin && !std::is_same_v<Joined, NoJoin>;
        if (!has_radix2(lg)) {
            const std::size_t q = len / 4;
            forward_radix4_from(in, n, f, q, r);
            if constexpr (kJoined)
                if (nv == 16) return joined(4);
            for (std::size_t t = 0; t < 4; ++t) sub(std::integral_constant<int, 4>{}, f + t * q, f + t * q, nv / 4, t);
            return;
        }
        const std::size_t half = len / 2;
        if (n > half || nv / 2 < 16) {
            forward_radix2_from(in, n, f, half);
            if constexpr (kJoined)
                if (nv == 8) return joined(2);
            sub(std::integral_constant<int, 2>{}, f + half, f + half, nv / 2, 1);
            sub(std::integral_constant<int, 2>{}, f, f, nv / 2, 0);
            return;
        }
        sub(std::integral_constant<int, 2>{}, Source{in, n}, f + half, nv / 2, 1);
        sub(std::integral_constant<int, 2>{}, Source{in, n}, f, nv / 2, 0);
    }

    template <class Tile>
    static void forward_from(const u32* in, std::size_t n, u32* f, int lg, const u32* r, Tile tile) {
        from_top(
            in, n, f, lg, r,
            [&] [[gnu::always_inline]] (auto in_bound, auto from, u32* g, std::size_t nv, std::size_t k) {
                forward_tree_from<decltype(in_bound)::value>(from, g, nv, k, r, tile);
            },
            [&] [[gnu::always_inline]] (std::size_t groups) { bottom_forward(f, groups, 0, r), tile(f, 4 * groups); });
    }

    // f = T(in) leaf by leaf times b (forward_from with leaf products in each tile).
    static void forward_product(const u32* in, std::size_t n, const u32* b, u32* f, int lg, const u32* r) {
        forward_from(in, n, f, lg, r, [&] [[gnu::always_inline]] (u32* t, std::size_t leaves) {
            const std::size_t at = std::size_t(t - f);
            leaf_products_at<N>(t, b + at, leaves / 4, at / 32, r);
        });
    }

    // The upper half of the leaf form of length 2L, L = 2^lg, of in (n <= L words): forward_upper
    // with its first level reading in.
    static void forward_upper_from(const u32* in, std::size_t n, u32* f, int lg, const u32* r) {
        const std::size_t len = std::size_t(1) << lg, nv = len / 8, half = len / 2;
        if (has_radix2(lg + 1)) return forward_tree_from<1>(Source{in, n}, f, nv, 1, r, NoTile{});
        if (n < len) {  // staged: f = in, zero past n
            if (in != f) std::memmove(f, in, n * sizeof(u32));
            std::fill(f + n, f + len, 0u);
            in = f;
        }
        const F z = B::entry(r, 1);
        for (std::size_t i = 0; i < half; i += S::lanes) {
            const V x = S::load(in + i), y = B::mul(S::load(in + i + half), z);
            S::store(f + i, add(x, y)), S::store(f + i + half, diff(x, y));
        }
        if (kJoin && nv == 8) return bottom_forward(f, 2, 2, r);
        forward_tree<4>(f, nv / 2, 2, r);
        forward_tree<4>(f + half, nv / 2, 3, r);
    }

    // The top level of an inverse whose subtrees ran with leaf products (factor 2^-32): count words
    // of the lower or upper half (output & kUpperHalf) to out, negated if output & kNegated (then
    // no scale folded into the subtrees: fold null).
    static void product_top_out(u32* f, u32* out, std::size_t count, int lg, const u32* ir, const u32* fold,
                                unsigned output) {
        const std::size_t len = std::size_t(1) << lg;
        const u32 scale = product_scale(lg), s = fold ? 0 : output & kNegated ? kP - scale : scale;
        if (!has_radix2(lg)) {
            if (output & kUpperHalf) return inverse_radix4_upper_out(f, len / 4, out, count, ir, s);
            return inverse_radix4_out(f, len / 4, out, count, ir, s);
        }
        if (output & kUpperHalf) return inverse_radix2_upper_out(f, len / 2, out, count, s);
        inverse_radix2_out(f, len / 2, out, count, s);
    }

    // out = count words of in b mod (x^L - 1) as product_top_out picks them: one pass, each
    // subtree's bottom stage with the leaf products (bottom_multiply_stored).
    static void cyclic_product(const u32* in, std::size_t n, const u32* b, u32* f, u32* out, std::size_t count, int lg,
                               const u32* r, const u32* ir, unsigned output) {
        const std::size_t len = std::size_t(1) << lg, nv = len / 8;
        const auto stage = [&] [[gnu::always_inline]] (u32* t, std::size_t leaves) {
            const std::size_t at = std::size_t(t - f);
            bottom_multiply_stored(t, b + at, leaves / 4, at / 32, r, ir);
        };
        const auto fold = [&](std::size_t leaves) { return output & kNegated ? nullptr : fold_scale(leaves, lg, true); };
        if (!has_radix2(lg)) {
            const std::size_t q = len / 4;
            forward_radix4_from(in, n, f, q, r);
            if (kJoin && nv == 16) stage(f, 16);
            else for (std::size_t t = 0; t < 4; ++t) cycle_tree(f + t * q, f + t * q, nv / 4, t, r, ir, fold(nv / 4), stage);
            return product_top_out(f, out, count, lg, ir, fold(nv / 4), output);
        }
        const std::size_t h = len / 2;
        if (n > h || nv / 2 < 16) {
            forward_radix2_from(in, n, f, h);
            if (kJoin && nv == 8) {
                stage(f, 8);
            } else {
                cycle_tree<2>(f + h, f + h, nv / 2, 1, r, ir, fold(nv / 2), stage);
                cycle_tree<2>(f, f, nv / 2, 0, r, ir, fold(nv / 2), stage);
            }
        } else {
            cycle_tree<2>(Source{in, n}, f + h, nv / 2, 1, r, ir, fold(nv / 2), stage);
            cycle_tree<2>(Source{in, n}, f, nv / 2, 0, r, ir, fold(nv / 2), stage);
        }
        product_top_out(f, out, count, lg, ir, fold(nv / 2), output);
    }

    // out = count words of the inverse of the sum over the terms (bottom_inverse_terms).
    static void inverse_product(const LeafTerm* terms, int n, u32* f, u32* out, std::size_t count, int lg, const u32* r,
                                const u32* ir, unsigned output) {
        if (n == 1 && !terms[0].b && output == 0) return inverse(terms[0].a, f, out, count, lg, ir, leaf_scale(lg), nullptr);
        inverse_staged(f, out, count, lg, ir, output, [&] [[gnu::always_inline]] (u32* t, std::size_t leaves) {
            bottom_inverse_terms(t, terms, n, leaves / 4, std::size_t(t - f) / 32, r, ir);
        });
    }

    // Transform::toeplitz_product: inverse_product's levels with bottom_toeplitz.
    static void toeplitz_product(const Toeplitz& z, u32* f, u32* out, std::size_t count, int lg, const u32* r,
                                 const u32* ir, unsigned output) {
        inverse_staged(f, out, count, lg, ir, output, [&] [[gnu::always_inline]] (u32* t, std::size_t leaves) {
            bottom_toeplitz(t, z, leaves / 4, std::size_t(t - f) / 32, r, ir);
        });
    }

    // The inverse levels above stage(t, leaves), which writes each run of leaves at t and runs its
    // inverse bottom level, and the top level writing count words of the half output picks to out.
    template <class Stage>
    [[gnu::always_inline]] static void inverse_staged(u32* f, u32* out, std::size_t count, int lg, const u32* ir,
                                                      unsigned output, Stage stage) {
        const std::size_t len = std::size_t(1) << lg, nv = len / 8;
        const auto fold = [&](std::size_t leaves) { return output & kNegated ? nullptr : fold_scale(leaves, lg, true); };
        if (!has_radix2(lg)) {
            if (kJoin && nv == 16) stage(f, 16);
            else for (std::size_t t = 0; t < 4; ++t) inverse_tree_with(f + t * len / 4, nv / 4, t, ir, fold(nv / 4), stage);
            return product_top_out(f, out, count, lg, ir, fold(nv / 4), output);
        }
        if (kJoin && nv == 8) {
            stage(f, 8);
        } else {
            inverse_tree_with(f, nv / 2, 0, ir, fold(nv / 2), stage);
            inverse_tree_with(f + len / 2, nv / 2, 1, ir, fold(nv / 2), stage);
        }
        product_top_out(f, out, count, lg, ir, fold(nv / 2), output);
    }

    // Hooked ops (Transform, issue #149), as avx2.hpp's: width (words, a power of two; 0: none) is
    // a column bit zero in the forward's input or unused in the inverse's output; the groups above it
    // with quarters of at least kPruneForward (kPruneInverse) leaves run on the other columns only.
    // Pruned here: the groups of the depth-first levels and each subtree's top group in cache;
    // not the top level, nor a first group reading the caller's source.
    static constexpr std::size_t kPruneForward = 32, kPruneInverse = 8;  // leaves per quarter

    static bool prunes(std::size_t q, std::size_t width, std::size_t least) {
        return width >= S::lanes && q >= 2 * width && q >= 8 * least;
    }
    // The group whose quarter index holds the bit: half its inputs (forward) or outputs (inverse).
    static bool halves(std::size_t q, std::size_t width, std::size_t least) {
        return (q == width || 2 * q == width) && q >= 8 * least;
    }
    template <int kIn>
    static void first_group_pruned(const u32* from, u32* f, std::size_t q, std::size_t k, const u32* r, std::size_t width) {
        if (from == f && prunes(q, width, kPruneForward)) return B::template forward_group_columns<kIn>(f, q, width, k, r);
        if (from == f && halves(q, width, kPruneForward)) return B::template forward_group_half<kIn>(f, q, k, r, 2 * q == width);
        first_group<kIn>(from, f, q, k, r);
    }
    template <int kIn>
    static void first_group_pruned(Source s, u32* f, std::size_t q, std::size_t k, const u32* r, std::size_t) {
        first_group<kIn>(s, f, q, k, r);
    }
    static void inverse_group_pruned(u32* f, std::size_t q, std::size_t k, const u32* ir, std::size_t width) {
        if (prunes(q, width, kPruneInverse)) return B::inverse_group_columns(f, q, width, k, ir);
        if (halves(q, width, kPruneInverse)) return B::inverse_group_half(f, q, k, ir, 2 * q == width);
        inverse_group(f, q, k, ir);
    }

    // forward_tree_from with pruned groups.
    template <int kIn = 4, class From, class Tile>
    static void forward_pruned_tree(From from, u32* f, std::size_t nv, std::size_t k, const u32* r, std::size_t width,
                                    Tile tile) {
        std::size_t h = nv / 4;
        if (nv > kTile) {
            first_group_pruned<kIn>(from, f, 8 * h, k, r, width);
            for (std::size_t t = 0; t < 4; ++t)
                forward_pruned_tree<4>(f + 8 * h * t, f + 8 * h * t, h, 4 * k + t, r, width, tile);
            return;
        }
        if (h >= 4) first_group_pruned<kIn>(from, f, 8 * h, k, r, width), h /= 4;
        for (; h >= 4; h /= 4) B::forward_groups(f, 8 * h, k * (nv / (4 * h)), nv / (4 * h), r);
        bottom_forward(f, nv / 4, k * (nv / 4), r);
        tile(f, nv);
    }

    // inverse_tree_with (no scale folded) with pruned groups.
    template <class Stage>
    static void inverse_pruned_tree(u32* f, std::size_t nv, std::size_t k, const u32* ir, std::size_t width, Stage stage) {
        if (nv > kTile) {
            const std::size_t h = nv / 4;
            for (std::size_t t = 0; t < 4; ++t) inverse_pruned_tree(f + 8 * h * t, h, 4 * k + t, ir, width, stage);
            return inverse_group_pruned(f, 8 * h, k, ir, width);
        }
        stage(f, nv);
        for (std::size_t h = 4; h < nv / 4; h *= 4) B::inverse_groups(f, 8 * h, k * (nv / (4 * h)), nv / (4 * h), ir);
        if (nv >= 16) inverse_group_pruned(f, 2 * nv, k, ir, width);
    }

    // Below 2^kHookWholeLog the subtrees have 4 leaves: the hook runs once on all of them instead
    // (zen5, the composition's hooks at 2^7: 258 against 153 ns for inverse_product, notes/149-r1.md).
    static constexpr int kHookWholeLog = 8;

    // hook(leaves, first, count) on each subtree's leaves of the leaf form of in after its bottom level.
    // The top level pruned too (detail::PrunedLevel) when it reads in whole: not a source of at most
    // half the length under a radix-2 level, which the subtrees' first groups read, unless width is
    // at least kUnreadWidth (Transform::forward_hooked: the skipped words of in are not used then).
    template <class Hook>
    static void forward_hooked(const u32* in, std::size_t n, u32* f, int lg, const u32* r, std::size_t width, Hook hook) {
        if (lg < kHookWholeLog) {
            forward_from(in, n, f, lg, r, NoTile{});
            return hook(f, std::size_t(0), (std::size_t(1) << lg) / 8);
        }
        const auto tile = [&] [[gnu::always_inline]] (u32* t, std::size_t leaves) { hook(t, std::size_t(t - f) / 8, leaves); };
        const auto sub = [&] [[gnu::always_inline]] (auto in_bound, auto from, u32* g, std::size_t nv, std::size_t k) {
            forward_pruned_tree<decltype(in_bound)::value>(from, g, nv, k, r, width, tile);
        };
        const std::size_t len = std::size_t(1) << lg, nv = len / 8;
        if (!has_radix2(lg)) {
            const std::size_t q = len / 4;
            first_level_pruned<4>(in, n, f, q, kStep, width,
                                  [&] [[gnu::always_inline]] (auto blocks, const u32* s, u32* d, std::size_t words) {
                                      forward_radix4_part<decltype(blocks)::value>(s, d, q, words, r);
                                  });
            for (std::size_t t = 0; t < 4; ++t) sub(std::integral_constant<int, 4>{}, f + t * q, f + t * q, nv / 4, t);
            return;
        }
        // Runs in increasing order (Transform::forward_hooked): the lower subtree first, as in is
        // apart from f (from_top runs the upper one first, for a source in place).
        const std::size_t half = len / 2;
        if (n <= half && nv / 2 >= 16 && width < kUnreadWidth) {
            sub(std::integral_constant<int, 2>{}, Source{in, n}, f, nv / 2, 0);
            sub(std::integral_constant<int, 2>{}, Source{in, n}, f + half, nv / 2, 1);
            return;
        }
        first_level_pruned<2>(in, n, f, half, S::lanes, width,
                              [&] [[gnu::always_inline]] (auto blocks, const u32* s, u32* d, std::size_t words) {
                                  forward_radix2_part<decltype(blocks)::value>(s, d, half, words);
                              });
        sub(std::integral_constant<int, 2>{}, f, f, nv / 2, 0);
        sub(std::integral_constant<int, 2>{}, f + half, f + half, nv / 2, 1);
    }

    // product_top_out (no scale folded) with the columns (j & width) != 0 unwritten.
    static void product_top_out_pruned(u32* f, u32* out, std::size_t count, int lg, const u32* ir, unsigned output,
                                       std::size_t width) {
        const u32 scale = product_scale(lg);
        top_out_pruned(f, out, count, lg, ir, output & kNegated ? kP - scale : scale, output, width);
    }
    // The same times s (not 0).
    static void top_out_pruned(u32* f, u32* out, std::size_t count, int lg, const u32* ir, u32 s, unsigned output,
                               std::size_t width) {
        const std::size_t len = std::size_t(1) << lg, q = len / 4, half = len / 2;
        if (!has_radix2(lg)) {
            if (output & kUpperHalf) {
                return last_level_pruned<2>(f, q, out, count, kStep, width,
                                            [&] [[gnu::always_inline]] (auto blocks, const u32* from, u32* to, std::size_t words) {
                                                constexpr int T = decltype(blocks)::value;
                                                if constexpr (T > 0) inverse_radix4_part<T, true>(from, to, q, words, ir, s);
                                            });
            }
            return last_level_pruned<4>(f, q, out, count, kStep, width,
                                        [&] [[gnu::always_inline]] (auto blocks, const u32* from, u32* to, std::size_t words) {
                                            constexpr int T = decltype(blocks)::value;
                                            if constexpr (T > 0) inverse_radix4_part<T>(from, to, q, words, ir, s);
                                        });
        }
        if (output & kUpperHalf) {
            return last_level_pruned<1>(f, half, out, count, S::lanes, width,
                                        [&] [[gnu::always_inline]] (auto blocks, const u32* from, u32* to, std::size_t words) {
                                            if constexpr (decltype(blocks)::value > 0) inverse_radix2_part<1, true>(from, to, half, words, s);
                                        });
        }
        last_level_pruned<2>(f, half, out, count, S::lanes, width,
                             [&] [[gnu::always_inline]] (auto blocks, const u32* from, u32* to, std::size_t words) {
                                 constexpr int T = decltype(blocks)::value;
                                 if constexpr (T > 0) inverse_radix2_part<T>(from, to, half, words, s);
                             });
    }

    // hook(leaves, first, count) writes each subtree's leaves (canonical, times 2^-32) before its
    // bottom level; the top level writes count words of the half `output` picks to out.
    template <class Hook>
    static void inverse_hooked(Hook hook, u32* f, u32* out, std::size_t count, int lg, const u32* ir, unsigned output,
                               std::size_t width) {
        const std::size_t len = std::size_t(1) << lg, nv = len / 8;
        const bool whole = lg < kHookWholeLog;
        if (whole) hook(f, std::size_t(0), nv);
        const auto stage = [&] [[gnu::always_inline]] (u32* t, std::size_t leaves) {
            const std::size_t at = std::size_t(t - f);
            if (!whole) hook(t, at / 8, leaves);
            bottom_inverse(t, leaves / 4, at / 32, ir);
        };
        if (kJoin && whole) {  // 2^6, 2^7: subtrees of one group (Ntt::kJoin)
            stage(f, nv);
        } else if (!has_radix2(lg)) {
            for (std::size_t t = 0; t < 4; ++t) inverse_pruned_tree(f + t * len / 4, nv / 4, t, ir, width, stage);
        } else {
            inverse_pruned_tree(f, nv / 2, 0, ir, width, stage);
            inverse_pruned_tree(f + len / 2, nv / 2, 1, ir, width, stage);
        }
        product_top_out_pruned(f, out, count, lg, ir, output, width);
    }

    // ---- Doubling (Transform::truncate_blocks, join_blocks, forward_upper_hooked; issue #149 round 2)
    // A block of nb = block / 8 leaves (nb >= 4) is group j's subtree if nb is a power of 4
    // (aligned), else the subtrees of groups 2j and 2j + 1 (nb / 2 leaves each) under the second
    // stage of group j / 2 (a radix-2 level with r(j)). Blocks go in runs of a tile (kTile leaves).
    static bool aligned_block(std::size_t nb) { return std::countr_zero(nb) % 2 == 0; }

    // Subtrees [first, first + count) of nv leaves each (a power of 4) at f: forward (in < 4P) with
    // tile on runs of leaves, inverse (in, out < 2P, unscaled); level by level up to kTile leaves.
    template <class Tile>
    static void forward_runs(u32* f, std::size_t nv, std::size_t count, std::size_t first, const u32* r, Tile tile) {
        if (nv > kTile) {
            for (std::size_t i = 0; i < count; ++i) forward_tiles<4>(f + 8 * nv * i, nv, first + i, r, tile);
            return;
        }
        for (std::size_t h = nv / 4; h >= 4; h /= 4) B::forward_groups(f, 8 * h, first * (nv / (4 * h)), count * (nv / (4 * h)), r);
        bottom_forward(f, count * nv / 4, first * nv / 4, r);
        tile(f, count * nv);
    }
    static void inverse_runs(u32* f, std::size_t nv, std::size_t count, std::size_t first, const u32* ir) {
        if (nv > kTile) {
            for (std::size_t i = 0; i < count; ++i) inverse_tree(f + 8 * nv * i, nv, first + i, ir);
            return;
        }
        bottom_inverse(f, count * nv / 4, first * nv / 4, ir);
        for (std::size_t h = 4; h < nv; h *= 4) B::inverse_groups(f, 8 * h, first * (nv / (4 * h)), count * (nv / (4 * h)), ir);
    }

    template <class Hook>
    static void truncate_blocks(u32* f, std::size_t block, u32* copy, int lg, const u32* r, const u32* ir, Hook hook) {
        const std::size_t blocks = (std::size_t(1) << lg) / block, nb = block / 8, half = block / 2;
        const std::size_t run = std::max<std::size_t>(1, kTile / nb);
        const bool aligned = aligned_block(nb);
        const auto tile = [&] [[gnu::always_inline]] (u32* t, std::size_t leaves) { hook(t, std::size_t(t - f) / 8, leaves); };
        for (std::size_t j0 = 0; j0 < blocks; j0 += run) {
            const std::size_t n = std::min(run, blocks - j0);
            u32* const g = f + j0 * block;
            // The levels within the blocks undone: the lower halves, < 2P, times nb.
            if (aligned && nb == 4) {
                bottom_inverse(g, n, j0, ir);
            } else if (aligned) {
                inverse_runs(g, nb / 4, 4 * n, 4 * j0, ir);
                for (std::size_t j = 0; j < n; ++j) B::inverse_group_half(g + j * block, block / 4, j0 + j, ir, true);
            } else {  // the radix-2 level's lower output: u + v
                inverse_runs(g, nb / 2, 2 * n, 2 * j0, ir);
                for (std::size_t j = 0; j < n; ++j)
                    for (u32 *b = g + j * block, *e = b + half; b < e; b += S::lanes)
                        S::store(b, B::low(add(S::load(b), S::load(b + half))));
            }
            // The lower halves to copy (canonical); the upper halves zero (aligned) or a copy of the
            // lower ones (a radix-2 level with a zero upper input), then the levels within the blocks.
            for (std::size_t j = 0; j < n; ++j) {
                u32 *b = g + j * block, *c = copy + (j0 + j) * block;
                for (std::size_t i = 0; i < half; i += S::lanes) {
                    const V x = S::load(b + i);
                    S::store(c + i, canonical(x));
                    if (!aligned) S::store(b + half + i, x);
                }
            }
            if (aligned && nb == 4) {
                for (std::size_t j = 0; j < n; ++j) std::fill_n(g + j * block + half, half, 0u);
                bottom_forward(g, n, j0, r);
                tile(g, 4 * n);
            } else if (aligned) {
                for (std::size_t j = 0; j < n; ++j) B::template forward_group_half<4>(g + j * block, block / 4, j0 + j, r, true);
                forward_runs(g, nb / 4, 4 * n, 4 * j0, r, tile);
            } else {
                forward_runs(g, nb / 2, 2 * n, 2 * j0, r, tile);
            }
        }
    }

    // The first stage of inverse group k on quarters of q words, quarters 0 and 2 only: a + c,
    // x (a - c) (blocks 2k and 2k + 1's lower halves to the rows' lower halves). In, out < 2P.
    static void join_radix2(u32* f, std::size_t q, std::size_t k, const u32* ir) {
        const F x = B::entry(ir, k);
        for (std::size_t j = 0; j < q; j += S::lanes) {
            const V a = S::load(f + j), c = S::load(f + j + 2 * q);
            S::store(f + j, B::low(add(a, c))), S::store(f + j + 2 * q, B::mul(diff(a, c), x));
        }
    }
    // The inverse levels of group k's subtree (nv leaves) above the blocks of nb leaves, on the
    // columns with (j & width) == 0 from kTile leaves up, every column below.
    static void join_tree(u32* f, std::size_t nv, std::size_t k, std::size_t nb, const u32* ir, std::size_t width) {
        if (nv == nb) return;
        if (nv == 2 * nb) return join_radix2(f, 2 * nv, k, ir);
        if (nv > kTile) {
            const std::size_t h = nv / 4;
            for (std::size_t t = 0; t < 4; ++t) join_tree(f + 8 * h * t, h, 4 * k + t, nb, ir, width);
            return inverse_group_pruned(f, 8 * h, k, ir, width);
        }
        std::size_t h = nb;  // leaves per quarter of the lowest level above the blocks
        if (!aligned_block(nb)) {
            for (std::size_t g = 0; g < nv / (2 * nb); ++g) join_radix2(f + 16 * nb * g, 4 * nb, k * (nv / (2 * nb)) + g, ir);
            h = 2 * nb;
        }
        for (; h < nv; h *= 4) B::inverse_groups(f, 8 * h, k * (nv / (4 * h)), nv / (4 * h), ir);
    }

    static void join_blocks(u32* f, std::size_t block, u32 scale, int lg, const u32* ir) {
        const std::size_t len = std::size_t(1) << lg, nv = len / 8, nb = block / 8, width = block / 2;
        if (!has_radix2(lg)) {
            for (std::size_t t = 0; t < 4; ++t) join_tree(f + t * len / 4, nv / 4, t, nb, ir, width);
        } else {
            join_tree(f, nv / 2, 0, nb, ir, width);
            join_tree(f + len / 2, nv / 2, 1, nb, ir, width);
        }
        // The levels above the blocks multiply by nv / nb.
        top_out_pruned(f, f, len, lg, ir, multiply_mod(scale, power(u32(nv / nb), kP - 2)), 0, width);
    }

    // forward_upper with pruned groups and a hook (leaf indices from 2^lg / 8, f being the upper
    // half of a leaf form of 2^(lg + 1) words). Under a radix-2 top the first level is group 1's,
    // which leaves the skipped columns as they are: below 512 words a whole group at the column bit
    // reads them, so they are zeroed first. Under group 0 the radix-2 level with r(1) writes zeros.
    template <class Hook>
    static void forward_upper_hooked(u32* f, int lg, const u32* r, std::size_t width, Hook hook) {
        const std::size_t len = std::size_t(1) << lg, nv = len / 8, half = len / 2;
        const u32* const base = f - len;
        const auto tile = [&] [[gnu::always_inline]] (u32* t, std::size_t leaves) { hook(t, std::size_t(t - base) / 8, leaves); };
        if (has_radix2(lg + 1)) {
            if (width && width < 512)
                for (std::size_t j = width; j < len; j += 2 * width) std::fill_n(f + j, width, 0u);
            return forward_pruned_tree<1>(f, f, nv, 1, r, width, tile);
        }
        const F z = B::entry(r, 1);
        const V zero = S::broadcast(0);
        const bool pruned = width >= S::lanes && width < half;
        for (std::size_t c = 0; c < half; c += pruned ? 2 * width : half) {
            const std::size_t end = pruned ? c + width : half;
            for (std::size_t i = c; i < end; i += S::lanes) {
                const V x = S::load(f + i), y = B::mul(S::load(f + i + half), z);
                S::store(f + i, add(x, y)), S::store(f + i + half, diff(x, y));
            }
            if (pruned)
                for (std::size_t i = end; i < end + width; i += S::lanes) S::store(f + i, zero), S::store(f + i + half, zero);
        }
        forward_pruned_tree<4>(f, f, nv / 2, 2, r, width, tile);
        forward_pruned_tree<4>(f + half, f + half, nv / 2, 3, r, width, tile);
    }
};

}  // namespace fps::ntt::detail::intrinsics

namespace fps::ntt::variant {

// S: a struct of backend/simd/simd.hpp this target has.
template <class S>
struct intrinsics {
    using Ntt = detail::intrinsics::Ntt<S>;

    // Entries [0, count) of the forward or inverse table.
    static void build_table(std::uint32_t* t, std::size_t count, bool inverse) { Ntt::build_table(t, count, inverse); }

    static void forward(std::uint32_t* f, int lg, const std::uint32_t* roots) { Ntt::forward(f, lg, roots); }

    static void forward_upper(std::uint32_t* f, int lg, const std::uint32_t* roots) { Ntt::forward_upper(f, lg, roots); }

    static void inverse(std::uint32_t* f, int lg, const std::uint32_t* inverse_roots) {
        Ntt::inverse(f, lg, inverse_roots, detail::leaf_scale(lg));
    }

    static void inverse_upper(std::uint32_t* f, int lg, const std::uint32_t* inverse_roots) {
        Ntt::inverse_upper(f, lg, inverse_roots, detail::leaf_scale(lg));
    }

    // Does not read the upper half of a factor of at most half the transform length.
    static void multiply(std::uint32_t* a, std::uint32_t* b, std::size_t n, std::size_t m, int lg,
                         const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
        Ntt::multiply(a, b, n, m, lg, roots, inverse_roots);
    }

    static void multiply(const detail::Product& p, const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
        Ntt::multiply(p, roots, inverse_roots);
    }

    static void square(const detail::Product& p, const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
        Ntt::square(p, roots, inverse_roots);
    }

    // Leaf-form operations (ntt.hpp, Transform): their contracts are the Transform calls'.
    static void leaf_products(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t* out, int lg,
                              const std::uint32_t* roots, bool accumulate) {
        Ntt::leaf_products(a, b, out, lg, roots, accumulate);
    }

    static void forward_values(std::uint32_t* f, int lg, const std::uint32_t* roots, const std::uint32_t* rows) {
        Ntt::forward_values(f, lg, roots, rows);
    }

    static void inverse(const std::uint32_t* from, std::uint32_t* f, std::uint32_t* out, std::size_t count, int lg,
                        const std::uint32_t* inverse_roots, const std::uint32_t* rows) {
        Ntt::inverse(from, f, out, count, lg, inverse_roots, rows ? detail::values_scale(lg) : detail::leaf_scale(lg), rows);
    }

    // Fused leaf-form ops (ntt.hpp, Transform): forward from caller memory, forward and leaf
    // products, cyclic products with a stored leaf form, inverses of sums of leaf products.
    static void forward_from(const std::uint32_t* in, std::size_t n, std::uint32_t* f, int lg, const std::uint32_t* roots) {
        Ntt::forward_from(in, n, f, lg, roots, typename Ntt::NoTile{});
    }

    static void forward_product(const std::uint32_t* in, std::size_t n, const std::uint32_t* b, std::uint32_t* f, int lg,
                                const std::uint32_t* roots) {
        Ntt::forward_product(in, n, b, f, lg, roots);
    }

    static void forward_upper_from(const std::uint32_t* in, std::size_t n, std::uint32_t* f, int lg,
                                   const std::uint32_t* roots) {
        Ntt::forward_upper_from(in, n, f, lg, roots);
    }

    static void cyclic_product(const std::uint32_t* in, std::size_t n, const std::uint32_t* b, std::uint32_t* f,
                               std::uint32_t* out, std::size_t count, int lg, const std::uint32_t* roots,
                               const std::uint32_t* inverse_roots, unsigned output) {
        Ntt::cyclic_product(in, n, b, f, out, count, lg, roots, inverse_roots, output);
    }

    static void inverse_product(const detail::LeafTerm* terms, int n, std::uint32_t* f, std::uint32_t* out,
                                std::size_t count, int lg, const std::uint32_t* roots, const std::uint32_t* inverse_roots,
                                unsigned output) {
        Ntt::inverse_product(terms, n, f, out, count, lg, roots, inverse_roots, output);
    }

    static void toeplitz_product(const detail::Toeplitz& z, std::uint32_t* f, std::uint32_t* out, std::size_t count,
                                 int lg, const std::uint32_t* roots, const std::uint32_t* inverse_roots, unsigned output) {
        Ntt::toeplitz_product(z, f, out, count, lg, roots, inverse_roots, output);
    }

    // Hooked ops; width in words (Transform::forward(in, F, hook, ...)).
    template <class Hook>
    static void forward_hooked(const std::uint32_t* in, std::size_t n, std::uint32_t* f, int lg, const std::uint32_t* roots,
                               std::size_t width, Hook hook) {
        Ntt::forward_hooked(in, n, f, lg, roots, width, hook);
    }

    template <class Hook>
    static void inverse_hooked(Hook hook, std::uint32_t* f, std::uint32_t* out, std::size_t count, int lg,
                               const std::uint32_t*, const std::uint32_t* inverse_roots, unsigned output, std::size_t width) {
        Ntt::inverse_hooked(hook, f, out, count, lg, inverse_roots, output, width);
    }

    // Doubling ops (Transform::truncate_blocks, join_blocks, forward_upper_hooked).
    template <class Hook>
    static void truncate_blocks(std::uint32_t* f, std::size_t block, std::uint32_t* copy, int lg, const std::uint32_t* roots,
                                const std::uint32_t* inverse_roots, Hook hook) {
        Ntt::truncate_blocks(f, block, copy, lg, roots, inverse_roots, hook);
    }

    static void join_blocks(std::uint32_t* f, std::size_t block, std::uint32_t scale, int lg,
                            const std::uint32_t* inverse_roots) {
        Ntt::join_blocks(f, block, scale, lg, inverse_roots);
    }

    template <class Hook>
    static void forward_upper_hooked(std::uint32_t* f, int lg, const std::uint32_t* roots, std::size_t width, Hook hook) {
        Ntt::forward_upper_hooked(f, lg, roots, width, hook);
    }
};

}  // namespace fps::ntt::variant
