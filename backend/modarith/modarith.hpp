// Vector arithmetic modulo a prime P, 2^12 < P < 2^30, as templates on a simd struct S.
//
//   using M = fps::modarith::Ops<fps::simd::Native, 998244353>;
//   M::V c = M::montgomery(a, b);  // a b 2^-32 mod P, in [0, 2P)
//
// Lanes are uint32. Each operation states the ranges it needs and returns; values above P are
// "lazy" and reduce<>() brings them down. model.hpp restates every contract as scalar code.
//
//   add(a, b)                a, b < P -> [0, P)
//   sub(a, b)                a, b < P -> [0, P)
//   reduce<M>(x)             x < 2M, M <= 2^31 -> [0, M); same residue if P | M
//   sub_lazy<M>(a, b)        b <= M, a + M < 2^32 -> a - b + M
//   montgomery(a, b)         a, b < 2^31, a b < 2^32 P -> [0, 2P), = a b 2^-32
//   shoup(x, w, q)           x < 2^32, w < P, q = shoup_quotient(w) -> [0, 2P), = x w
//   barrett(a, b)            a, b < P -> [0, kBarrettBound), = a b; kBarrettBound is 2P or 3P
//
// montgomery and shoup run the variant select.hpp picks for this target and S; the variants are
// public too:
//   montgomery_as<Montgomery::wide>     x86: 6 mul_even; Neon: umull, umull2, umlal, umlal2, 1 mul
//   montgomery_as<Montgomery::mulhi>    any S: 2 mulhi, 2 mul_lo
//   montgomery_as<Montgomery::doubling> Neon: 2 sqdmulh, 2 mul, shsub; signed lanes
//   shoup_as<Shoup::mulhi>              any S: 1 mulhi, 2 mul_lo (Neon: mul + mls)
//   shoup_as<Shoup::wide>               x86: 6 mul_even
// Variants with their own contracts (not interchangeable with the above):
//   shoup_doubling(x, w, q31)  Neon. x < 2^31, q31 = shoup_quotient31(w) -> [0, 2P), = x w;
//                              3 instructions (sqdmulh, mul, mls)
//   montgomery52(a, b)         IFMA (Avx512, Avx512Vl). a, b < 2^32 -> [0, 2P), = a b 2^-52
//   shoup52(x, w, q52)         IFMA. x < 2^32, w < P, q52 = shoup_quotient52(w) -> [0, 2P), = x w.
//                              w and q52 are 64-bit lanes: words 2i and 2i + 1 of x use lane i.
// The IFMA variants work on 64-bit lanes: even and odd words are split and joined (4 extra ops).
#pragma once

#include <bit>
#include <cstdint>

#include "backend/modarith/select.hpp"
#include "backend/simd/simd.hpp"

namespace fps::modarith {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

// x^-1 mod 2^bits for odd x, bits <= 64, by Newton's iteration (each step doubles the bits).
constexpr u64 inverse_mod_power_of_two(u64 x, int bits) {
    u64 inv = x;  // correct to 3 bits for odd x
    for (int i = 0; i < 5; ++i) inv *= 2 - x * inv;
    return bits == 64 ? inv : inv & ((u64(1) << bits) - 1);
}

template <class S, u32 P>
struct Ops {
    static_assert(P > (1u << 12) && P < (1u << 30) && P % 2 == 1);
    using V = typename S::V;

    static constexpr u32 kPinv = u32(inverse_mod_power_of_two(P, 32));  // P^-1 mod 2^32
    static constexpr int kBits = std::bit_width(P);                     // 2^(kBits-1) < P < 2^kBits
    static constexpr u32 kBarrettM = u32((u64(1) << (kBits + 31)) / P);  // < 2^32
    // Barrett's quotient is short by less than E + 1, E = P^2 / 2^(kBits+31) + 2^(kBits-1) / P.
    static constexpr u32 kBarrettBound =
        (unsigned __int128)P * P * P + ((unsigned __int128)1 << (2 * kBits + 30)) <
                (unsigned __int128)P << (kBits + 31)
            ? 2 * P
            : 3 * P;

    static constexpr u32 shoup_quotient(u32 w) { return u32((u64(w) << 32) / P); }
    static constexpr u32 shoup_quotient31(u32 w) { return u32((u64(w) << 31) / P); }
    static constexpr u64 shoup_quotient52(u32 w) { return u64(((unsigned __int128)w << 52) / P); }

    static V add(V a, V b) {
        const V s = S::add(a, b);
        return S::min(s, S::sub(s, S::broadcast(P)));
    }
    static V sub(V a, V b) {
        const V d = S::sub(a, b);
        return S::min(d, S::add(d, S::broadcast(P)));
    }
    // min(x, x - M): for x < M, x - M wraps above x.
    template <u32 M>
    static V reduce(V x) {
        static_assert(M <= (1u << 31));
        return S::min(x, S::sub(x, S::broadcast(M)));
    }
    template <u32 M>
    static V sub_lazy(V a, V b) { return S::add(a, S::sub(S::broadcast(M), b)); }

    template <Montgomery variant>
    static V montgomery_as(V a, V b) {
        if constexpr (variant == Montgomery::wide) return montgomery_wide(a, b);
        else if constexpr (variant == Montgomery::doubling) return montgomery_doubling(a, b);
        else return montgomery_mulhi(a, b);
    }
    static V montgomery(V a, V b) { return montgomery_as<selected_montgomery<S>()>(a, b); }

    template <Shoup variant>
    static V shoup_as(V x, V w, V q) {
        if constexpr (variant == Shoup::wide) return shoup_wide(x, w, q);
        else return shoup_mulhi(x, w, q);
    }
    static V shoup(V x, V w, V q) { return shoup_as<selected_shoup<S>()>(x, w, q); }

    // q = floor(a b / 2^(kBits-1)) m / 2^32 <= a b / P.
    static V barrett(V a, V b) {
        const V high = S::template mul_shr<kBits - 1>(a, b);  // < 2P
        const V q = S::mulhi(high, S::broadcast(kBarrettM));
        return S::mul_sub(S::mul_lo(a, b), q, S::broadcast(P));
    }

    static V shoup_doubling(V x, V w, V q31)
        requires requires { S::qdmulh(x, x); }
    {
        return S::mul_sub(S::mul_lo(x, w), S::qdmulh(x, q31), S::broadcast(P));
    }

    static V montgomery52(V a, V b)
        requires requires { S::madd52lo(a, a, a); }
    {
        const auto half = [](V x, V y) {  // 64-bit lanes, x, y < 2^32
            const V zero = S::broadcast(0), p = S::broadcast64(P);
            const V low = S::madd52lo(zero, x, y);
            const V high = S::madd52hi(p, x, y);  // + P keeps the result positive
            const V m = S::madd52lo(zero, low, S::broadcast64(inverse_mod_power_of_two(P, 52)));
            return S::sub64(high, S::madd52hi(zero, m, p));  // (x y - m P) / 2^52 + P in (0, 2P)
        };
        return join(half(even(a), even(b)), half(odd(a), odd(b)));
    }

    static V shoup52(V x, V w, V q52)
        requires requires { S::madd52lo(x, x, x); }
    {
        // x w - floor(x q52 / 2^52) P is in [0, 2P); its low 32 bits come out of the 52-bit sums.
        const auto half = [](V x, V w, V q) {
            const V zero = S::broadcast(0);
            const V qx = S::madd52hi(zero, x, q);
            return S::madd52lo(S::madd52lo(zero, x, w), qx, S::broadcast64((u64(1) << 52) - P));
        };
        return join(half(even(x), w, q52), half(odd(x), w, q52));
    }

private:
    // a b < 2^32 P: (a b + m P) / 2^32 with m = -(a b) P^-1 mod 2^32 is in [0, 2P).
    static V montgomery_wide(V a, V b)
        requires requires { S::mul_even(a, b); }
    {
        const V ninv = S::broadcast(u32(0) - kPinv), p = S::broadcast(P);
        V even = S::mul_even(a, b);
        V odd = S::mul_even(S::template shr64<32>(a), S::template shr64<32>(b));
        even = S::add64(even, S::mul_even(S::mul_even(even, ninv), p));
        odd = S::add64(odd, S::mul_even(S::mul_even(odd, ninv), p));
        return S::template blend<0xaaaa>(S::template shr64<32>(even), odd);
    }
    static V montgomery_wide(V a, V b)
        requires requires { S::mul_wide_lo(a, b); }
    {
        const V ninv = S::broadcast(u32(0) - kPinv), p = S::broadcast(P);
        auto low = S::mul_wide_lo(a, b), high = S::mul_wide_hi(a, b);
        const V m = S::mul_lo(S::mul_lo(a, b), ninv);
        low = S::mla_wide_lo(low, m, p);
        high = S::mla_wide_hi(high, m, p);
        return S::hi_halves(low, high);
    }

    // With m = a b P^-1 mod 2^32, a b - m P = 2^32 r exactly, r in (-P, P).
    static V montgomery_mulhi(V a, V b) {
        const V m = S::mul_lo(S::mul_lo(a, b), S::broadcast(kPinv));
        const V r = S::sub(S::mulhi(a, b), S::mulhi(m, S::broadcast(P)));
        return S::add(r, S::broadcast(P));
    }

    // Signed m in [-2^31, 2^31): 2 a b - 2 m P = 2^33 r, and the high words of the doubled
    // products differ by exactly 2r, r in (-P/2, 3P/2). min(r, r + P) maps it into [0, 3P/2).
    static V montgomery_doubling(V a, V b)
        requires requires { S::qdmulh(a, b); }
    {
        const V p = S::broadcast(P);
        const V m = S::mul_lo(S::mul_lo(a, b), S::broadcast(kPinv));
        const V r = S::hsub(S::qdmulh(a, b), S::qdmulh(m, p));
        return S::min(r, S::add(r, p));
    }

    // floor(x q / 2^32) is floor(x w / P) or one less.
    static V shoup_mulhi(V x, V w, V q) { return S::mul_sub(S::mul_lo(x, w), S::mulhi(x, q), S::broadcast(P)); }

    static V shoup_wide(V x, V w, V q)
        requires requires { S::mul_even(x, w); }
    {
        const V p = S::broadcast(P);
        const auto half = [p](V x, V w, V q) {  // low 32 bits of each 64-bit lane
            const V qx = S::template shr64<32>(S::mul_even(x, q));
            return S::sub64(S::mul_even(x, w), S::mul_even(qx, p));
        };
        const V even = half(x, w, q);
        const V odd = half(S::template shr64<32>(x), S::template shr64<32>(w), S::template shr64<32>(q));
        return S::template blend<0xaaaa>(even, S::template shl64<32>(odd));
    }

    // 64-bit lanes holding the even words, the odd words; join takes the low word of each.
    static V even(V x) { return S::template blend<0xaaaa>(x, S::broadcast(0)); }
    static V odd(V x) { return S::template shr64<32>(x); }
    static V join(V even, V odd) { return S::template blend<0xaaaa>(even, S::template shl64<32>(odd)); }
};

}  // namespace fps::modarith
