// Polynomial gcd modulo P = 998244353 by half-gcd jumps, and the inverse modulo a polynomial
// (backend kernel). From lc-opt's lib/poly/gcd.hpp (its issues #82, #85), rewritten on
// ntt::Transform's fused ops and division::divide, so every target runs it. Design and
// measurements: notes.md.
//
//   std::size_t t = fps::half_gcd::gcd(f, g, out);           // out[0, t) = monic gcd(f, g); t = 0 if both 0
//   std::ptrdiff_t t = fps::half_gcd::inverse_mod(f, g, h);  // h = 1 / f mod g, deg h < deg g; -1 if
//                                                       // gcd(f, g) != 1
//   // The same in the caller's scratch (gcd_scratch(n, m), inverse_mod_scratch(n, m) words, any
//   // alignment) instead of this thread's NTT region:
//   fps::half_gcd::gcd(f, g, out, scratch);
//
// f, g: words < P, any sizes; zeros at their ends are ignored. gcd: out needs as many words as the
// gcd can have, min of the two sizes past trailing zeros if both are nonzero, else the nonzero one's
// (max(f.size(), g.size()) always suffices), else std::invalid_argument before anything is written.
// inverse_mod: g of m >= 1 words with g[m - 1] != 0, h of m - 1 words; returns the number t of
// coefficients of h past trailing zeros (h[t, m - 1) = 0; t = 0: h = 0, also for m = 1), or -1 if
// gcd(f, g) != 1 (h unspecified). Throws std::invalid_argument for other sizes, std::length_error
// past the transforms' 2^25 words.
//
// A jump of k on (a, b), deg a = n > deg b, is the matrix R = Q_h ... Q_1 of the Euclidean
// quotients q_i of (a, b) with deg r_h >= n - k > deg r_(h+1): R (a, b) = (r_h, r_(h+1)). It
// depends only on a and b divided by x^(n - 2k) (Thull and Yap). Remainders are scaled
// (fraction-free steps), so R (a, b) = (c r_h, c' r_(h+1)) for nonzero constants c, c'.
// - jump(k): jump(k1) on the top 2 k1 + 1 coefficients (k1 = k / 2 for a power of two, else the
//   largest power of two below k), apply R1 to (a, b) by cyclic products of length bit_ceil(n / 2),
//   jump(k - progress) on the top of the result, combine R2 R1 by cyclic products of length
//   bit_ceil(progress), keeping the product transforms (the parent doubles them by forward_upper).
//   Jumps of k <= kDirect: Euclid's algorithm with fraction-free steps on the top coefficients.
// - gcd: the spine of a full jump without its matrix: jump(k1) on the top, apply to the whole of
//   (a, b), repeat; Euclid (with inversions) below kDirect.
// - inverse_mod: b = f mod g, the full jump of deg g on (g, b) computing R[0][1] only (and every
//   entry's coefficient 0): h = R[0][1] / c for c = R (g, b)[0], a constant iff gcd = 1.
//
// Memory: one span (this thread's NTT region, kept across calls, or the caller's scratch) used as a
// stack; a division step inside a jump takes division's scratch from it (or from the heap if it is
// short). Transforms: series' per-thread ntt::Transform per length.
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "backend/division/division.hpp"
#include "backend/modarith/modarith.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/series/series.hpp"
#include "backend/simd/select.hpp"

namespace fps::half_gcd {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u32 kP = ntt::kModulus;

namespace detail {

using Span = std::span<const u32>;
using Out = std::span<u32>;
using S = simd::Native;
using M = modarith::Ops<S, kP>;
using V = S::V;
constexpr std::size_t kLanes = S::lanes;
using series::detail::footprint;

constexpr u32 mul(u32 a, u32 b) { return u32(u64(a) * b % kP); }
constexpr u32 sub1(u32 a, u32 b) { return a >= b ? a - b : a + kP - b; }
constexpr u32 negate(u32 a) { return a ? kP - a : 0; }
constexpr u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
constexpr u32 inverse(u32 a) { return power(a, kP - 2); }
constexpr u32 kR = u32((u64(1) << 32) % kP);  // 2^32 mod P
constexpr u32 kNegPinv = u32(0) - M::kPinv;  // -P^-1 mod 2^32

inline constexpr std::ptrdiff_t kDirect = 64;  // jumps of at most this many degrees: Euclid
inline constexpr std::size_t kMinLength = std::size_t(1) << ntt::kMinLog;

inline const ntt::Transform& transform(std::size_t len) { return series::detail::transform(std::countr_zero(len)); }

// ---- Memory -------------------------------------------------------------------------------------

// 64-byte aligned arrays (footprint: ntt::kPadding words after each) taken from one span and given
// back in reverse order (mark, release).
class Stack {
public:
    Stack(u32* start, std::size_t words) : next_(start), end_(start + words) {
        next_ += (64 - reinterpret_cast<std::uintptr_t>(start) % 64) % 64 / sizeof(u32);
    }
    u32* take(std::size_t words) {
        u32* p = next_;
        if (footprint(words) > std::size_t(end_ - next_)) throw std::logic_error("fps::half_gcd: scratch too small");
        next_ += footprint(words);
        return p;
    }
    std::span<u32> take_span(std::size_t words) { return {take(words), words}; }
    u32* mark() const { return next_; }
    void release(u32* mark) { next_ = mark; }
    std::size_t room() const { return std::size_t(end_ - next_); }

private:
    u32 *next_, *end_;
};

// ---- Euclid's steps -----------------------------------------------------------------------------

// Whether T has 64-bit products of lanes (x86 mul_even, Neon mul_wide_lo): sums of products then
// take one Montgomery reduction.
template <class T>
constexpr bool kWideOf = requires(typename T::V a) { T::mul_even(a, a); } || requires(typename T::V a) { T::mul_wide_lo(a, a); };
constexpr bool kWide = kWideOf<S>;

// The factors of out = x u + y v x + z v (one fraction-free step), as combine3 wants them: times
// 2^32 (Montgomery, kWide), else with their Shoup quotients.
struct Step {
    u32 x, y, z;  // kWide: times 2^32 mod P
    u32 xq = 0, yq = 0, zq = 0;
    Step(u32 x_, u32 y_, u32 z_) {
        if constexpr (kWide) {
            x = mul(x_, kR), y = mul(y_, kR), z = mul(z_, kR);
        } else {
            x = x_, y = y_, z = z_;
            xq = M::shoup_quotient(x_), yq = M::shoup_quotient(y_), zq = M::shoup_quotient(z_);
        }
    }
};

// out[j] = x u[j] + y v[j - 1] + z v[j] mod P for j in [from, to), rounded out to whole vectors:
// from down and to up to multiples of kLanes (callers keep kLanes words before u, v and out, and
// 2 kLanes after). Words of u, v < 2^32 give out < P; canonical ones give the right values.
// x86: 3 products in 64-bit sums (< 3 P 2^32 + 2^32 P < 2^64), one Montgomery reduction (lc-opt's).
// Neon: the same with umull / umlal. Otherwise Shoup products, reduced in pairs.
template <class T>
[[gnu::always_inline]] inline typename T::V combine3_vector(typename T::V a, typename T::V b, typename T::V c, const Step& s) {
    using W = typename T::V;
    using Ops = modarith::Ops<T, kP>;
    if constexpr (requires { T::mul_even(a, b); }) {
        const W xs = T::broadcast(s.x), ys = T::broadcast(s.y), zs = T::broadcast(s.z);
        const W ninv = T::broadcast(kNegPinv), p = T::broadcast(kP);
        W even = T::add64(T::add64(T::mul_even(a, xs), T::mul_even(b, ys)), T::mul_even(c, zs));
        W odd = T::add64(T::add64(T::mul_even(T::template shr64<32>(a), xs), T::mul_even(T::template shr64<32>(b), ys)),
                         T::mul_even(T::template shr64<32>(c), zs));
        even = T::add64(even, T::mul_even(T::mul_even(even, ninv), p));
        odd = T::add64(odd, T::mul_even(T::mul_even(odd, ninv), p));
        return Ops::template reduce<kP>(T::join_high(even, odd));
    } else if constexpr (requires { T::mul_wide_lo(a, b); }) {
        const W xs = T::broadcast(s.x), ys = T::broadcast(s.y), zs = T::broadcast(s.z), p = T::broadcast(kP);
        auto low = T::mla_wide_lo(T::mla_wide_lo(T::mul_wide_lo(a, xs), b, ys), c, zs);
        auto high = T::mla_wide_hi(T::mla_wide_hi(T::mul_wide_hi(a, xs), b, ys), c, zs);
        const W m = T::mul_lo(T::lo_halves(low, high), T::broadcast(kNegPinv));
        low = T::mla_wide_lo(low, m, p);
        high = T::mla_wide_hi(high, m, p);
        return Ops::template reduce<kP>(T::hi_halves(low, high));
    } else {
        const W t = Ops::template reduce<2 * kP>(T::add(Ops::shoup(a, T::broadcast(s.x), T::broadcast(s.xq)),
                                                         Ops::shoup(b, T::broadcast(s.y), T::broadcast(s.yq))));
        const W u = Ops::template reduce<2 * kP>(T::add(t, Ops::shoup(c, T::broadcast(s.z), T::broadcast(s.zq))));
        return Ops::template reduce<kP>(u);
    }
}

inline void combine3(const u32* u, const u32* v, u32* out, std::ptrdiff_t from, std::ptrdiff_t to, const Step& s) {
    constexpr std::ptrdiff_t L = std::ptrdiff_t(kLanes);
    for (std::ptrdiff_t j = from & ~(L - 1); j < to; j += L)
        S::store(out + j, combine3_vector<S>(S::load(u + j), S::load(v + j - 1), S::load(v + j), s));
}

// ---- Values-form passes ------------------------------------------------------------------------

constexpr u32 kR2 = mul(kR, kR);  // 2^64 mod P

// x[i] = x[i] 2^32 mod P for i < n (canonical).
inline void to_montgomery(u32* x, std::size_t n) {
    std::size_t i = 0;
    const V r2 = S::broadcast(kR2);
    for (; i + kLanes <= n; i += kLanes) S::store(x + i, M::template reduce<kP>(M::montgomery(S::load(x + i), r2)));
    for (; i < n; ++i) x[i] = mul(x[i], kR);
}

// (x0 y0 + x1 y1) 2^-32 mod P, canonical, for canonical words; twice: then times 2^-32 again.
// kWide: the two products in 64-bit sums (< 2 P^2 + 2^32 P < 2^64), one Montgomery reduction (and
// a second of the result); else Montgomery products, added.
template <class T>
[[gnu::always_inline]] inline typename T::V products_vector(typename T::V x0, typename T::V y0, typename T::V x1, typename T::V y1, bool twice) {
    using W = typename T::V;
    using Ops = modarith::Ops<T, kP>;
    if constexpr (requires { T::mul_even(x0, y0); }) {
        const W ninv = T::broadcast(kNegPinv), p = T::broadcast(kP);
        const auto reduce = [&](W s) { return T::add64(s, T::mul_even(T::mul_even(s, ninv), p)); };  // high words
        W even = reduce(T::add64(T::mul_even(x0, y0), T::mul_even(x1, y1)));
        W odd = reduce(T::add64(T::mul_even(T::template shr64<32>(x0), T::template shr64<32>(y0)),
                                T::mul_even(T::template shr64<32>(x1), T::template shr64<32>(y1))));
        if (twice) {  // (r + m P) / 2^32 again, r < 2P the high words
            even = reduce(T::template shr64<32>(even));
            odd = reduce(T::template shr64<32>(odd));
        }
        return Ops::template reduce<kP>(T::join_high(even, odd));
    } else if constexpr (requires { T::mul_wide_lo(x0, y0); }) {
        const W ninv = T::broadcast(kNegPinv), p = T::broadcast(kP);
        auto low = T::mla_wide_lo(T::mul_wide_lo(x0, y0), x1, y1);
        auto high = T::mla_wide_hi(T::mul_wide_hi(x0, y0), x1, y1);
        W m = T::mul_lo(T::lo_halves(low, high), ninv);
        W r = T::hi_halves(T::mla_wide_lo(low, m, p), T::mla_wide_hi(high, m, p));  // < 2P
        if (twice) {  // r 2^-32: (r + m P) / 2^32 with m = -r P^-1, r + m P < 2^32 (2P + P)
            m = T::mul_lo(r, ninv);
            r = T::hi_halves(T::mla_wide_lo(T::mul_wide_lo(r, T::broadcast(1)), m, p),
                             T::mla_wide_hi(T::mul_wide_hi(r, T::broadcast(1)), m, p));
        }
        return Ops::template reduce<kP>(r);
    } else {
        const W s = Ops::add(Ops::template reduce<kP>(Ops::montgomery(x0, y0)), Ops::template reduce<kP>(Ops::montgomery(x1, y1)));
        return twice ? Ops::template reduce<kP>(Ops::montgomery(s, T::broadcast(1))) : s;
    }
}

// m = (x0 y0 + x1 y1) 2^-32 mod P word by word (canonical words, n a multiple of kLanes); kept[i] =
// m if kept is not null; out[i] = m 2^-32 if twice, else m. A product of two forms times 2^32 is
// then kept as such and goes to the inverse as the plain values (twice).
inline void products(const u32* x0, const u32* y0, const u32* x1, const u32* y1, u32* kept, u32* out, std::size_t n,
                     bool twice) {
    if (!kept) {
        for (std::size_t i = 0; i < n; i += kLanes)
            S::store(out + i, products_vector<S>(S::load(x0 + i), S::load(y0 + i), S::load(x1 + i), S::load(y1 + i), twice));
        return;
    }
    const V one = S::broadcast(1);
    for (std::size_t i = 0; i < n; i += kLanes) {
        const V m = products_vector<S>(S::load(x0 + i), S::load(y0 + i), S::load(x1 + i), S::load(y1 + i), false);
        S::store(kept + i, m);
        S::store(out + i, twice ? M::template reduce<kP>(M::montgomery(m, one)) : m);
    }
}

// ---- Matrices -----------------------------------------------------------------------------------

// A 2 x 2 matrix of polynomials: entry[2 i + j] is row i, column j, with size[2 i + j]
// coefficients (0 for the zero polynomial). progress: the degree of entry[3], n - deg r_h for a
// jump. transform[i]: room for transforms of length up to room; length[i] > 0 when it holds the
// leaf form of entry i mod (x^length[i] - 1). at0[i]: coefficient 0 of entry i, also for the
// entries a jump leaves out if their row has an entry it computes.
struct Matrix {
    u32* entry[4];
    std::size_t size[4];
    u32 at0[4];
    u32* transform[4];
    std::size_t length[4];
    std::size_t room, progress;
};

// Jumps of (a, b) on a Stack.
class HalfGcd {
public:
    // direct: jumps of at most this many degrees run Euclid's algorithm (tests force others).
    explicit HalfGcd(Stack& stack, std::ptrdiff_t direct = kDirect) : stack_(stack), direct_(direct) {}

    // Room for a matrix of progress up to k, with transforms of length up to room.
    Matrix matrix(std::size_t k, std::size_t room) {
        Matrix r;
        for (int i = 0; i < 4; ++i) {
            r.entry[i] = take(k + 1);
            r.transform[i] = room ? take(room) : nullptr;
            r.length[i] = 0;
        }
        r.room = room;
        return r;
    }

    // out = the jump of k on (a, b): a has n + 1 coefficients, a[n] != 0; b has m + 1, m < n
    // (m = -1: b = 0), b[m] != 0; 0 <= k <= n. out: room for progress k. Only the entries in
    // the mask need (bit i: entry i) are computed; the others are left unspecified.
    void jump(const u32* a, std::ptrdiff_t n, const u32* b, std::ptrdiff_t m, std::ptrdiff_t k, Matrix& out,
              unsigned need = 0xF) {
        const std::ptrdiff_t target = n - k;
        if (m < target) return identity(out);
        if (n > 2 * k) {  // only a and b divided by x^(n - 2k) matter
            const std::ptrdiff_t s = n - 2 * k;
            return jump(a + s, n - s, b + s, m - s, k, out, need);
        }
        if (k <= direct_) return euclid(a, n, b, m, k, out);
        u32* const mark = stack_.mark();
        const std::size_t room = std::max(kMinLength, std::bit_ceil(std::size_t(k)));
        const std::ptrdiff_t k1 = first_jump(k), s1 = std::max<std::ptrdiff_t>(0, n - 2 * k1);
        Matrix r1 = matrix(std::size_t(k1), room);
        jump(a + s1, n - s1, b + s1, m - s1, k1, r1);
        std::ptrdiff_t c_deg = 0;
        const u32 *c = nullptr, *d = nullptr;
        if (r1.progress == 0) {
            r1 = matrix(std::size_t(n - m), room);
            u32* const rest = take(std::size_t(n + 1));
            divide_step(a, n, b, m, r1, rest);
            c = b, d = rest, c_deg = m;
        } else {
            c_deg = n - std::ptrdiff_t(r1.progress);
            const std::ptrdiff_t s2 = std::max<std::ptrdiff_t>(0, 2 * target - c_deg);
            u32 *const cs = take(std::size_t(c_deg + 1)), *const ds = take(std::size_t(c_deg + 1));
            apply(a, n, b, m, r1, c_deg, s2, cs, ds);
            c = cs, d = ds;
        }
        std::ptrdiff_t d_deg = c_deg - 1;
        while (d_deg >= target && d[d_deg] == 0) --d_deg;
        if (d_deg < target) {
            copy(r1, out);
            return stack_.release(mark);
        }
        const std::ptrdiff_t k2 = c_deg - target, s2 = std::max<std::ptrdiff_t>(0, c_deg - 2 * k2);
        Matrix r2 = matrix(std::size_t(k2), room);
        jump(c + s2, c_deg - s2, d + s2, d_deg - s2, k2, r2, (need & 3 ? 3u : 0u) | (need & 12 ? 12u : 0u));
        combine(r2, r1, out, need);
        stack_.release(mark);
    }

    // out[0, t) = the monic gcd of (a, b), deg a = n > deg b = m (b[m] != 0, or m = -1), t returned:
    // the remainders by jumps, then by Euclid's steps. a, b: n + 1 words each, overwritten.
    std::size_t spine(u32* a, std::ptrdiff_t n, u32* b, std::ptrdiff_t m, Out out) {
        u32* const top = stack_.mark();
        u32* c = take(std::size_t(n + 1));
        u32* d = take(std::size_t(n + 1));
        while (m >= 0 && n > direct_) {
            u32* const mark = stack_.mark();
            const std::ptrdiff_t k1 = first_jump(n), s1 = std::max<std::ptrdiff_t>(0, n - 2 * k1);
            const std::size_t room = std::max({kMinLength, std::bit_ceil(std::size_t(k1)), std::bit_ceil(std::size_t(n + 1) / 2)});
            Matrix r1 = matrix(std::size_t(k1), room);
            jump(a + s1, n - s1, b + s1, m - s1, k1, r1);
            std::ptrdiff_t c_deg;
            if (r1.progress == 0) {  // (c, d) = (b, a mod b)
                std::copy_n(b, m + 1, c);
                remainder(a, n, b, m, d);
                c_deg = m;
            } else {
                c_deg = n - std::ptrdiff_t(r1.progress);
                apply(a, n, b, m, r1, c_deg, 0, c, d);
            }
            stack_.release(mark);
            std::swap(a, c), std::swap(b, d);
            n = c_deg, m = c_deg - 1;
            while (m >= 0 && b[m] == 0) --m;
        }
        // Euclid's fraction-free steps on whole remainders, in padded copies.
        u32* p[3];
        for (auto& x : p) x = take_padded(std::size_t(n + 1));
        std::copy_n(a, n + 1, p[0]);
        std::copy_n(b, m + 1, p[1]);
        a = p[0], b = p[1];
        u32* e = p[2];
        while (m >= 0) {
            if (n - m == 1) {
                const u32 alpha = b[m], beta = a[n];
                const u32 gamma = sub1(mul(alpha, a[n - 1]), mul(beta, m ? b[m - 1] : 0));
                combine3(a, b, e, 0, m, Step(mul(alpha, alpha), negate(mul(alpha, beta)), negate(gamma)));
            } else {
                pseudo_remainder(a, n, b, m, e);
            }
            u32* const old = a;
            a = b, b = e, e = old;
            n = m, m = n - 1;
            while (m >= 0 && b[m] == 0) --m;
        }
        const u32 scale = inverse(a[n]);
        for (std::ptrdiff_t i = 0; i < n; ++i) out[std::size_t(i)] = mul(a[i], scale);
        out[std::size_t(n)] = 1;
        stack_.release(top);
        return std::size_t(n + 1);
    }

private:
    // The first part of a jump of k: half of a power of two, else its largest power of two
    // below k, so that the transform lengths fit (notes.md).
    static std::ptrdiff_t first_jump(std::ptrdiff_t k) {
        const std::size_t u = std::size_t(k);
        return std::ptrdiff_t(std::has_single_bit(u) ? u / 2 : std::bit_floor(u));
    }

    u32* take(std::size_t n) { return stack_.take(n); }
    std::span<u32> take_span(std::size_t n) { return stack_.take_span(n); }

    // n words, zero, with kLanes zero words before them and 2 kLanes after (for combine3).
    u32* take_padded(std::size_t n) {
        u32* const p = take(n + 3 * kLanes);
        std::fill_n(p, n + 3 * kLanes, 0u);
        return p + kLanes;
    }

    static void identity(Matrix& out) {
        out.entry[0][0] = out.entry[3][0] = 1;
        out.size[0] = out.size[3] = 1;
        out.size[1] = out.size[2] = 0;
        out.at0[0] = out.at0[3] = 1, out.at0[1] = out.at0[2] = 0;
        for (auto& l : out.length) l = 0;
        out.progress = 0;
    }

    static void copy(const Matrix& from, Matrix& to) {
        for (int i = 0; i < 4; ++i) {
            std::copy_n(from.entry[i], from.size[i], to.entry[i]);
            to.size[i] = from.size[i];
            to.at0[i] = from.at0[i];
            to.length[i] = 0;
        }
        to.progress = from.progress;
    }

    // Euclid's algorithm with fraction-free steps on the coefficients of degree >= the floor
    // n - 2k + (progress so far), which are all the remaining quotients read.
    void euclid(const u32* a, std::ptrdiff_t n, const u32* b, std::ptrdiff_t m, std::ptrdiff_t k, Matrix& out) {
        u32* const mark = stack_.mark();
        const std::ptrdiff_t target = n - k;
        std::ptrdiff_t floor = n - 2 * k, lo = std::max<std::ptrdiff_t>(0, floor);
        u32* p[3];
        for (auto& x : p) x = take_padded(std::size_t(n + 1));
        std::copy(a + lo, a + n + 1, p[0] + lo);
        std::copy(b + lo, b + m + 1, p[1] + lo);
        // Rows of the matrix: row[0] = (u, v) with u a + v b = c, row[1] for d, row[2] free. Zero
        // past their sizes.
        const std::size_t cap = std::size_t(k) + 1;
        u32* row[3][2];
        std::size_t row_size[3][2] = {{1, 0}, {0, 1}, {0, 0}};
        for (auto& r : row)
            for (auto& e : r) e = take_padded(cap);
        row[0][0][0] = 1, row[1][1][0] = 1;
        u32 *c = p[0], *d = p[1], *e = p[2];
        std::ptrdiff_t c_deg = n, d_deg = m;
        while (d_deg >= target) {
            const std::ptrdiff_t delta = c_deg - d_deg;
            const u32 alpha = d[d_deg];
            floor += delta;
            lo = std::max<std::ptrdiff_t>(0, floor);
            if (delta == 1) {
                // e = alpha^2 c - (alpha beta x + gamma) d kills the coefficients c_deg and d_deg.
                const u32 beta = c[c_deg];
                const u32 gamma = sub1(mul(alpha, c[c_deg - 1]), mul(beta, d_deg ? d[d_deg - 1] : 0));
                const Step step(mul(alpha, alpha), negate(mul(alpha, beta)), negate(gamma));
                combine3(c, d, e, lo, d_deg, step);
                for (int i = 0; i < 2; ++i) {
                    const std::size_t sc = row_size[0][i], sd = row_size[1][i];
                    const std::size_t se = std::max(sc, sd ? sd + 1 : 0);
                    combine3(row[0][i], row[1][i], row[2][i], 0, std::ptrdiff_t(se), step);
                    // combine3 wrote whole vectors past se: zeros, from zeros (sizes stay below cap).
                    if (row_size[2][i] > se) std::fill(row[2][i] + se, row[2][i] + row_size[2][i], 0u);
                    row_size[2][i] = se;
                }
            } else {
                pseudo_divide(c, c_deg, d, d_deg, e, lo, alpha, row, row_size);
            }
            u32* const old = c;
            c = d, d = e, e = old;
            for (int i = 0; i < 2; ++i) {
                std::swap(row[0][i], row[1][i]);
                std::swap(row[1][i], row[2][i]);
                std::swap(row_size[0][i], row_size[1][i]);
                std::swap(row_size[1][i], row_size[2][i]);
            }
            c_deg = d_deg;
            d_deg = c_deg - 1;
            while (d_deg >= target && d[d_deg] == 0) --d_deg;
        }
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                std::copy_n(row[i][j], row_size[i][j], out.entry[2 * i + j]);
                out.size[2 * i + j] = row_size[i][j];
                out.at0[2 * i + j] = row_size[i][j] ? row[i][j][0] : 0;
                out.length[2 * i + j] = 0;
            }
        out.progress = std::size_t(n - c_deg);
        stack_.release(mark);
    }

    // A quotient of degree delta > 1: e = c, then e = alpha e - e[d_deg + t] x^t d for t = delta
    // .. 0, on the coefficients from lo; the rows alike (row[2] from row[0] and row[1]).
    static void pseudo_divide(const u32* c, std::ptrdiff_t c_deg, const u32* d, std::ptrdiff_t d_deg, u32* e,
                              std::ptrdiff_t lo, u32 alpha, u32* (&row)[3][2], std::size_t (&row_size)[3][2]) {
        std::copy(c + lo, c + c_deg + 1, e + lo);
        for (int i = 0; i < 2; ++i) {
            std::copy_n(row[0][i], row_size[0][i], row[2][i]);
            if (row_size[2][i] > row_size[0][i]) std::fill(row[2][i] + row_size[0][i], row[2][i] + row_size[2][i], 0u);
            row_size[2][i] = row_size[0][i];
        }
        for (std::ptrdiff_t t = c_deg - d_deg; t >= 0; --t) {
            const u32 minus_tau = negate(e[d_deg + t]);
            for (std::ptrdiff_t j = lo; j <= d_deg + t; ++j) {
                const u32 dj = j - t >= 0 ? d[j - t] : 0;
                e[j] = u32((u64(alpha) * e[j] + u64(minus_tau) * dj) % kP);
            }
            for (int i = 0; i < 2; ++i) {
                u32* const re = row[2][i];
                const u32* const rd = row[1][i];
                const std::size_t sd = row_size[1][i], shift = std::size_t(t);
                const std::size_t se = std::max(row_size[2][i], sd ? sd + shift : 0);
                for (std::size_t j = 0; j < se; ++j) {
                    const u32 dj = j >= shift ? rd[j - shift] : 0;  // zero past sd
                    re[j] = u32((u64(alpha) * re[j] + u64(minus_tau) * dj) % kP);
                }
                row_size[2][i] = se;
            }
        }
    }

    // e = lc(d)^(delta + 1) c mod d (fraction-free, delta = c_deg - d_deg): e[0, d_deg).
    static void pseudo_remainder(const u32* c, std::ptrdiff_t c_deg, const u32* d, std::ptrdiff_t d_deg, u32* e) {
        const u32 alpha = d[d_deg];
        std::copy(c, c + c_deg + 1, e);
        for (std::ptrdiff_t t = c_deg - d_deg; t >= 0; --t) {
            const u32 minus_tau = negate(e[d_deg + t]);
            for (std::ptrdiff_t j = 0; j <= d_deg + t; ++j)
                e[j] = u32((u64(alpha) * e[j] + u64(minus_tau) * (j >= t ? d[j - t] : 0)) % kP);
        }
    }

    // rest = a mod b (m words; b of m + 1 coefficients, b[m] != 0), q (n - m + 1 words) = a quo b if
    // given. division::divide in scratch from the stack, else from the heap.
    void remainder(const u32* a, std::ptrdiff_t n, const u32* b, std::ptrdiff_t m, u32* rest, u32* q = nullptr) {
        const Span f(a, std::size_t(n + 1)), g(b, std::size_t(m + 1));
        u32* const mark = stack_.mark();
        const std::size_t qs = std::size_t(n - m + 1);
        const Out quotient = q ? Out(q, qs) : take_span(qs);
        const std::size_t words = division::scratch_words(f.size(), g.size(), true);
        if (stack_.room() >= footprint(words)) {
            division::divide(f, g, quotient, Out(rest, std::size_t(m)), take_span(words));
        } else {
            std::vector<u32> scratch(words);
            division::divide(f, g, quotient, Out(rest, std::size_t(m)), scratch);
        }
        stack_.release(mark);
    }

    // (c, d) = (b, a - q b) for q = a quo b, and r = [[0, 1], [1, -q]]. rest: n + 1 words.
    void divide_step(const u32* a, std::ptrdiff_t n, const u32* b, std::ptrdiff_t m, Matrix& r, u32* rest) {
        const std::size_t delta = std::size_t(n - m);
        u32* const q = r.entry[3];
        remainder(a, n, b, m, rest, q);
        std::fill(rest + m, rest + n + 1, 0u);
        for (std::size_t i = 0; i <= delta; ++i) q[i] = negate(q[i]);
        r.entry[1][0] = r.entry[2][0] = 1;
        r.size[0] = 0, r.size[1] = r.size[2] = 1, r.size[3] = delta + 1;
        r.at0[0] = 0, r.at0[1] = r.at0[2] = 1, r.at0[3] = q[0];
        for (auto& l : r.length) l = 0;
        r.progress = delta;
    }

    // out (len words) = the values form of length len of c mod (x^len - 1), c of size coefficients,
    // times 2^32 if montgomery (the matrices' form).
    static void values_of(const u32* c, std::size_t size, u32* out, std::size_t len, bool montgomery) {
        division::detail::fold(Span(c, size), Out(out, len));
        if (montgomery) to_montgomery(out, std::min(size, len));
        transform(len).forward(out, ntt::Form::values);
    }

    // r.transform[i] = the values form of length len of entry i, times 2^32 (room >= len): kept,
    // doubled from the half held (the upper half from the entry mod x^(len/2) + 1), or computed.
    static void extend(Matrix& r, int i, std::size_t len) {
        if (r.length[i] == len) return;
        u32* const t = r.transform[i];
        const std::size_t half = len / 2, size = r.size[i];
        if (r.length[i] == half && size <= half + 1) {
            u32* const upper = t + half;
            const u32* const e = r.entry[i];
            std::copy_n(e, std::min(size, half), upper);
            std::fill(upper + std::min(size, half), upper + half, 0u);
            if (size > half) upper[0] = sub1(upper[0], e[half]);
            to_montgomery(upper, std::min(size, half));
            transform(len).forward_upper(upper, ntt::Form::values);
        } else {
            values_of(r.entry[i], size, t, len, true);
        }
        r.length[i] = len;
    }

    // c, d = r1 (a, b) at degrees [s2, c_deg], c_deg = deg c. With L >= n / 2, the products are
    // cyclic of length L; when c_deg >= L, the coefficients [0, c_deg - L] of c and d (which the
    // wrap adds to [L, c_deg]) come from the products of r1 with a, b mod x^(c_deg - L + 1).
    void apply(const u32* a, std::ptrdiff_t n, const u32* b, std::ptrdiff_t m, Matrix& r1, std::ptrdiff_t c_deg,
               std::ptrdiff_t s2, u32* c, u32* d) {
        u32* const mark = stack_.mark();
        const std::size_t len = std::max(kMinLength, std::bit_ceil(std::size_t(n + 1) / 2));
        for (int i = 0; i < 4; ++i) extend(r1, i, len);
        u32 *const ta = take(len), *const tb = take(len);
        values_of(a, std::size_t(n + 1), ta, len, false);
        values_of(b, std::size_t(m + 1), tb, len, false);
        u32 *const pc = take(len), *const pd = take(len);
        apply_products(r1.transform, ta, tb, pc, pd, len);
        const std::ptrdiff_t l = std::ptrdiff_t(len), mixed = c_deg - l;  // c[j] + c[j + L] at j <= mixed
        if (mixed < 0) {
            std::copy(pc + s2, pc + c_deg + 1, c + s2);
            std::copy(pd + s2, pd + c_deg + 1, d + s2);
            return stack_.release(mark);
        }
        const std::size_t low = std::size_t(mixed + 1);
        u32 *const lc = take(len), *const ld = take(len);
        const std::size_t short_len = std::max(kMinLength, std::bit_ceil(2 * low - 1));
        const std::size_t mb = std::min(low, std::size_t(m + 1));
        if (short_len < len) {
            // Few wrapped coefficients: (r1 mod x^low) (a, b mod x^low) of degree < 2 low - 1.
            u32* rt[4];
            for (int i = 0; i < 4; ++i) {
                rt[i] = take(short_len);
                values_of(r1.entry[i], std::min(low, r1.size[i]), rt[i], short_len, true);
            }
            values_of(a, low, ta, short_len, false);
            values_of(b, mb, tb, short_len, false);
            apply_products(rt, ta, tb, lc, ld, short_len);
        } else {
            values_of(a, low, ta, len, false);
            values_of(b, mb, tb, len, false);
            apply_products(r1.transform, ta, tb, lc, ld, len);
            // Coefficient L of the low products (if n = 2L) wraps onto 0.
            lc[0] = u32((u64(r1.at0[0]) * a[0] + u64(r1.at0[1]) * b[0]) % kP);
            ld[0] = u32((u64(r1.at0[2]) * a[0] + u64(r1.at0[3]) * b[0]) % kP);
        }
        // [s2, mixed] from the low products, (mixed, L) as computed, [L, c_deg] unwrapped.
        const std::ptrdiff_t first = std::min(s2, mixed + 1);
        std::copy(lc + first, lc + mixed + 1, c + first);
        std::copy(ld + first, ld + mixed + 1, d + first);
        const std::ptrdiff_t from = std::max(s2, mixed + 1);
        std::copy(pc + from, pc + l, c + from);
        std::copy(pd + from, pd + l, d + from);
        division::detail::subtract(pc, lc, c + l, low);
        division::detail::subtract(pd, ld, d + l, low);
        stack_.release(mark);
    }

    // c, d (len words each) = the coefficients of r (a, b) mod (x^len - 1) from the values forms of
    // length len of r's entries (times 2^32) and of a, b.
    static void apply_products(u32* const (&r)[4], const u32* ta, const u32* tb, u32* c, u32* d, std::size_t len) {
        const ntt::Transform& t = transform(len);
        products(r[0], ta, r[1], tb, nullptr, c, len, false);
        products(r[2], ta, r[3], tb, nullptr, d, len, false);
        t.inverse(c, ntt::Form::values);
        t.inverse(d, ntt::Form::values);
    }

    // out = r2 r1 (the entries in need), progress p = r2.progress + r1.progress: cyclic products
    // of length L >= p (coefficient L, if p = L, wraps onto 0), whose values forms out keeps if its
    // room allows.
    void combine(Matrix& r2, Matrix& r1, Matrix& out, unsigned need) {
        u32* const mark = stack_.mark();
        const std::size_t p = r2.progress + r1.progress, len = std::max(kMinLength, std::bit_ceil(p));
        for (int i = 0; i < 2; ++i)
            if (need & (3u << 2 * i)) extend(r2, 2 * i, len), extend(r2, 2 * i + 1, len);
        for (int j = 0; j < 2; ++j)
            if (need & (5u << j)) extend(r1, j, len), extend(r1, 2 + j, len);
        const ntt::Transform& t = transform(len);
        u32* const work = take(len);
        const bool keep = out.room >= len;
        const std::size_t count = std::min(p + 1, len);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                const int ij = 2 * i + j;
                out.length[ij] = 0;
                const u32 x0 = u32((u64(r2.at0[2 * i]) * r1.at0[j] + u64(r2.at0[2 * i + 1]) * r1.at0[2 + j]) % kP);
                out.at0[ij] = x0;
                if (!(need >> ij & 1)) continue;
                // out[ij] = r2[2i] r1[j] + r2[2i + 1] r1[2 + j]: kept times 2^32, its coefficients
                // from the plain values in work.
                products(r2.transform[2 * i], r1.transform[j], r2.transform[2 * i + 1], r1.transform[2 + j],
                         keep ? out.transform[ij] : nullptr, work, len, true);
                if (keep) out.length[ij] = len;
                t.inverse(work, ntt::Form::values);
                u32* const e = out.entry[ij];
                std::copy_n(work, count, e);
                if (p == len) e[len] = sub1(e[0], x0), e[0] = x0;
                std::size_t size = p + 1;
                while (size > 0 && e[size - 1] == 0) --size;
                out.size[ij] = size;
            }
        out.progress = p;
        stack_.release(mark);
    }

    Stack& stack_;
    std::ptrdiff_t direct_;
};

// ---- Sizes --------------------------------------------------------------------------------------

inline std::size_t trimmed(Span a) {
    std::size_t n = a.size();
    while (n > 0 && a[n - 1] == 0) --n;
    return n;
}

// Stack words of jumps on polynomials of degree <= n: 56 N for N = bit_ceil(n) (lc-opt: 48 N, plus
// the fused ops' work arrays), and per level of the recursion the arrays' gaps.
inline std::size_t half_gcd_words(std::size_t n) {
    const std::size_t len = std::max(kMinLength, std::bit_ceil(std::max<std::size_t>(n, 1)));
    const std::size_t levels = 2 * std::size_t(std::bit_width(len)) + 4;
    return 56 * len + levels * 48 * footprint(0) + 4096;
}

// Words of f mod g (q and division's scratch).
inline std::size_t remainder_words(std::size_t n, std::size_t m) {
    if (division::quotient_size(n, m) == 0) return 0;
    return footprint(division::quotient_size(n, m)) + footprint(division::scratch_words(n, m, true));
}

inline std::size_t gcd_words(std::size_t nf, std::size_t ng) {  // trimmed sizes, both > 0, nf >= ng
    return 2 * footprint(nf) + std::max(remainder_words(nf, ng), 2 * footprint(ng) + half_gcd_words(ng));
}

inline std::size_t inverse_mod_words(std::size_t n, std::size_t m) {  // m >= 2
    const std::size_t d = m - 1;
    return footprint(d) + std::max(remainder_words(n, m), half_gcd_words(d));
}

inline constexpr std::size_t kAlignSlack = 16;

// Runs body(stack) in the caller's scratch, or in this thread's region when scratch is empty.
template <class Body>
auto with_stack(std::size_t words, std::span<u32> scratch, const Body& body) {
    words += kAlignSlack;
    if (!scratch.empty()) {
        if (scratch.size() < words) throw std::invalid_argument("fps::half_gcd: scratch too small");
        Stack stack(scratch.data(), scratch.size());
        return body(stack);
    }
    const series::detail::RegionGuard guard{ntt::detail::take_region(words * sizeof(u32))};
    Stack stack(reinterpret_cast<u32*>(guard.region.start), guard.region.bytes / sizeof(u32));
    return body(stack);
}

inline void check_length(std::size_t n) {
    if (n > (std::size_t(1) << ntt::kMaxLog)) throw std::length_error("fps::half_gcd: longer than the transforms' 2^25 words");
}

// gcd of a (na) and b (nb) words, trimmed, na >= nb >= 1, in the stack: out[0, t) monic.
inline std::size_t gcd_run(Stack& stack, Span a, Span b, Out out, std::ptrdiff_t direct) {
    const std::size_t na = a.size(), nb = b.size();
    // (a, b) <- (b, a mod b), then the spine.
    u32* const x = stack.take(na);
    u32* const y = stack.take(na);
    std::copy(b.begin(), b.end(), x);
    std::fill(x + nb, x + na, 0u);
    std::fill(y, y + na, 0u);
    HalfGcd h(stack, direct);
    {
        u32* const mark = stack.mark();
        const std::size_t qs = division::quotient_size(na, nb);
        const std::size_t words = division::scratch_words(na, nb, true);
        division::divide(a, b, stack.take_span(qs), Out(y, nb - 1), stack.take_span(words));
        stack.release(mark);
    }
    std::ptrdiff_t n = std::ptrdiff_t(nb) - 1, m = n - 1;
    while (m >= 0 && y[m] == 0) --m;
    return h.spine(x, n, y, m, out);
}

// inverse_mod of f by g (m >= 2 words, g[m - 1] != 0) in the stack.
inline std::ptrdiff_t inverse_mod_run(Stack& stack, Span f, Span g, Out h, std::ptrdiff_t direct) {
    const std::ptrdiff_t d = std::ptrdiff_t(g.size()) - 1;
    u32* const b = stack.take(std::size_t(d));
    {
        u32* const mark = stack.mark();
        const std::size_t qs = division::quotient_size(f.size(), g.size());
        const std::size_t words = division::scratch_words(f.size(), g.size(), true);
        division::divide(f, g, stack.take_span(qs), Out(b, std::size_t(d)), stack.take_span(words));
        stack.release(mark);
    }
    std::ptrdiff_t m = d - 1;
    while (m >= 0 && b[m] == 0) --m;
    if (m < 0) return -1;
    HalfGcd gcd(stack, direct);
    Matrix r = gcd.matrix(std::size_t(d), 0);
    gcd.jump(g.data(), d, b, m, d, r, 0x2);
    if (std::ptrdiff_t(r.progress) != d) return -1;  // deg gcd = d - progress
    // r (g, b) = (c, 0) with c a nonzero constant: h = r[0][1] / c.
    const u32 c = u32((u64(r.at0[0]) * g[0] + u64(r.at0[1]) * b[0]) % kP);
    const u32 scale = inverse(c);
    std::size_t t = r.size[1];
    while (t > 0 && r.entry[1][t - 1] == 0) --t;
    for (std::size_t i = 0; i < t; ++i) h[i] = mul(r.entry[1][i], scale);
    std::fill(h.begin() + std::ptrdiff_t(t), h.end(), 0u);
    return std::ptrdiff_t(t);
}

// The gcd with Euclid below `direct` (tests force small values).
inline std::size_t gcd(Span f, Span g, Out out, std::span<u32> scratch, std::ptrdiff_t direct) {
    std::size_t nf = trimmed(f), ng = trimmed(g);
    if (nf < ng) std::swap(f, g), std::swap(nf, ng);
    const std::size_t need = ng ? ng : nf;
    if (out.size() < need) throw std::invalid_argument("fps::half_gcd::gcd: out shorter than the gcd can be");
    if (nf == 0) return 0;
    check_length(nf);
    if (ng == 0) {  // gcd(f, 0) = f, monic
        const u32 scale = inverse(f[nf - 1]);
        for (std::size_t i = 0; i < nf; ++i) out[i] = mul(f[i], scale);
        return nf;
    }
    if (ng == 1) {  // a nonzero constant
        out[0] = 1;
        return 1;
    }
    return with_stack(gcd_words(nf, ng), scratch,
                      [&](Stack& stack) { return gcd_run(stack, f.first(nf), g.first(ng), out, direct); });
}

inline std::ptrdiff_t inverse_mod(Span f, Span g, Out h, std::span<u32> scratch, std::ptrdiff_t direct) {
    const std::size_t m = g.size();
    if (m == 0 || g[m - 1] == 0 || h.size() != m - 1)
        throw std::invalid_argument("fps::half_gcd::inverse_mod: g empty or g[m - 1] = 0, or h not m - 1 words");
    if (m == 1) return 0;
    check_length(std::max(f.size(), m));
    f = f.first(trimmed(f));
    return with_stack(inverse_mod_words(f.size(), m), scratch,
                      [&](Stack& stack) { return inverse_mod_run(stack, f, g, h, direct); });
}

}  // namespace detail

// Words of scratch gcd() takes for f of n and g of m words (trimmed or not; enough for any values).
inline std::size_t gcd_scratch(std::size_t n, std::size_t m) {
    if (n < m) std::swap(n, m);
    if (m <= 1) return 0;
    return detail::kAlignSlack + detail::gcd_words(n, m);
}

// Words of scratch inverse_mod() takes for f of n and g of m words.
inline std::size_t inverse_mod_scratch(std::size_t n, std::size_t m) {
    if (m <= 1) return 0;
    return detail::kAlignSlack + detail::inverse_mod_words(n, m);
}

// out[0, t) = the monic gcd of f and g, t returned (above).
inline std::size_t gcd(std::span<const u32> f, std::span<const u32> g, std::span<u32> out, std::span<u32> scratch = {}) {
    return detail::gcd(f, g, out, scratch, detail::kDirect);
}

// h = 1 / f mod g; the number of coefficients of h, or -1 if gcd(f, g) != 1 (above).
inline std::ptrdiff_t inverse_mod(std::span<const u32> f, std::span<const u32> g, std::span<u32> h,
                                  std::span<u32> scratch = {}) {
    return detail::inverse_mod(f, g, h, scratch, detail::kDirect);
}

}  // namespace fps::half_gcd
