// Polynomial gcd modulo P = 998244353 by half-gcd jumps, and the inverse modulo a polynomial
// (backend kernel; namespace half_gcd, as fps::gcd is the frontend's function). From lc-opt's
// lib/poly/gcd.hpp (its issues #82, #85), rewritten on ntt::Transform's values form and
// division::divide, so every target runs it. Design and measurements: notes.md.
//
//   std::size_t t = fps::half_gcd::gcd(f, g, out);           // out[0, t) = monic gcd(f, g); t = 0 if both 0
//   std::ptrdiff_t t = fps::half_gcd::inverse_mod(f, g, h);  // h = 1 / f mod g, deg h < deg g; -1 if
//                                                            // gcd(f, g) != 1
//   // The same in the caller's scratch (gcd_scratch(n, m), inverse_mod_scratch(n, m) words for f,
//   // g of n, m words past their trailing zeros; any alignment) instead of this thread's NTT region:
//   fps::half_gcd::gcd(f, g, out, scratch);
//
// f, g: words < P, any sizes; zeros at their ends are ignored. gcd: out needs as many words as the
// gcd can have, min of the two sizes past trailing zeros if both are nonzero, else the nonzero one's
// (max(f.size(), g.size()) always suffices), else std::invalid_argument before anything is written.
// inverse_mod: g of m >= 1 words with g[m - 1] != 0, h of m - 1 words; returns the number t of
// coefficients of h past trailing zeros (h[t, m - 1) = 0; t = 0: h = 0, also for m = 1), or -1 if
// gcd(f, g) != 1 (h unspecified). Throws std::invalid_argument for other sizes or a short scratch,
// std::length_error past the transforms' 2^25 words.
//
// A jump of k on (a, b), deg a = n > deg b, is the matrix R = Q_h ... Q_1 of the Euclidean
// quotients q_i of (a, b) with deg r_h >= n - k > deg r_(h+1): R (a, b) = (r_h, r_(h+1)). It
// depends only on a and b divided by x^(n - 2k) (Thull and Yap). Remainders are scaled
// (fraction-free steps), so R (a, b) = (c r_h, c' r_(h+1)) for nonzero constants c, c'.
// - jump(k): jump(k1) on the top 2 k1 + 1 coefficients (k1 = k / 2 for a power of two, else the
//   largest power of two below k), apply R1 to (a, b) by cyclic products of length bit_ceil(n / 2),
//   jump(k - progress) on the top of the result, combine R2 R1 by cyclic products of length
//   bit_ceil(progress), keeping the products' transforms (the parent doubles them by
//   forward_upper). Jumps of k <= kDirect: Euclid's algorithm with fraction-free steps on the top
//   coefficients (3 products per word in 64-bit sums, one Montgomery reduction), each step's
//   factors from the last's in scalar code.
// - Products in the values form (ntt.hpp): a pointwise product per word instead of an 8 x 8 leaf
//   product, for a split per forward and a join per inverse. Each product takes one Montgomery
//   reduction and leaves its factor 2^-32: matrices and remainders only matter up to constants.
// - gcd: the spine of a full jump without its matrix: jump(k1) on the top, apply to the whole of
//   (a, b), repeat; Euclid's fraction-free steps on whole remainders below kDirect.
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
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "backend/division/division.hpp"
#include "backend/gcd/select.hpp"
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
constexpr u32 kNegPinv = u32(0) - M::kPinv;  // -P^-1 mod 2^32

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

// t 2^-32 mod P, canonical, for t < 2^64 - 2^32 P.
constexpr u32 montgomery(u64 t) {
    const u32 r = u32((t + u64(u32(t) * kNegPinv) * kP) >> 32);  // < 2P
    return r >= kP ? r - kP : r;
}

// The factors of one fraction-free step on (c, d), deg c = D + 1, deg d = D: e = x c + (y X + z) d
// with x = alpha^2, y = -alpha beta, z = beta d1 - alpha c1 (alpha = d[D], beta = c[D + 1], c1 =
// c[D], d1 = d[D - 1]), so e = alpha^2 c - (alpha beta X + gamma) d for gamma = alpha c1 - beta d1,
// whose coefficients D + 1 and D vanish. One product per factor on the scalar chain (notes/157-r1.md).
// The passes return e times kScale (2^-32 where they reduce by Montgomery): remainders only matter
// up to constants, and the rows take the same factors. Not kWide: with Shoup quotients.
struct Step {
    u32 x, y, z;
    u32 xq = 0, yq = 0, zq = 0;
    Step(u32 alpha, u32 beta, u32 c1, u32 d1)
        : x(mul(alpha, alpha)), y(u32(u64(kP - alpha) * beta % kP)), z(u32((u64(beta) * d1 + u64(kP - alpha) * c1) % kP)) {
        if constexpr (!kWide) xq = M::shoup_quotient(x), yq = M::shoup_quotient(y), zq = M::shoup_quotient(z);
    }
};

// x86: from the 64-bit sums of the even and odd words, (sum + m P) / 2^32 joined, canonical.
template <class T>
[[gnu::always_inline]] inline typename T::V montgomery_join(typename T::V even, typename T::V odd) {
    using W = typename T::V;
    const W ninv = T::broadcast(kNegPinv), p = T::broadcast(kP);
    even = T::add64(even, T::mul_even(T::mul_even(even, ninv), p));
    odd = T::add64(odd, T::mul_even(T::mul_even(odd, ninv), p));
    return modarith::Ops<T, kP>::template reduce<kP>(T::join_high(even, odd));
}

// Neon: the same from the sums of lanes 0, 1 (low) and 2, 3 (high).
template <class T, class Wide>
[[gnu::always_inline]] inline typename T::V montgomery_halves(Wide low, Wide high) {
    using W = typename T::V;
    const W m = T::mul_lo(T::lo_halves(low, high), T::broadcast(kNegPinv)), p = T::broadcast(kP);
    return modarith::Ops<T, kP>::template reduce<kP>(T::hi_halves(T::mla_wide_lo(low, m, p), T::mla_wide_hi(high, m, p)));
}

// (x u + y X v + z v) kScale mod P word by word: u, v the words at j, v1 those of v at j - 1. Words
// of u, v < 2^32 give a result < P; canonical ones give the right values. x86: 3 products in 64-bit
// sums (< 3 P 2^32 + 2^32 P < 2^64), one Montgomery reduction (lc-opt's); the odd words of X v are
// the even words of v, so 2 shifts, not 3. Neon: the same with umull / umlal. Otherwise Shoup
// products, reduced in pairs.
template <class T>
[[gnu::always_inline]] inline typename T::V step_vector(typename T::V u, typename T::V v1, typename T::V v, const Step& s) {
    using W = typename T::V;
    using Ops = modarith::Ops<T, kP>;
    if constexpr (requires { T::mul_even(u, v); }) {
        const W xs = T::broadcast(s.x), ys = T::broadcast(s.y), zs = T::broadcast(s.z);
        const W uo = T::template shr64<32>(u), vo = T::template shr64<32>(v);
        const W even = T::add64(T::add64(T::mul_even(u, xs), T::mul_even(v1, ys)), T::mul_even(v, zs));
        const W odd = T::add64(T::add64(T::mul_even(uo, xs), T::mul_even(v, ys)), T::mul_even(vo, zs));
        return montgomery_join<T>(even, odd);
    } else if constexpr (requires { T::mul_wide_lo(u, v); }) {
        const W xs = T::broadcast(s.x), ys = T::broadcast(s.y), zs = T::broadcast(s.z);
        const auto low = T::mla_wide_lo(T::mla_wide_lo(T::mul_wide_lo(u, xs), v1, ys), v, zs);
        const auto high = T::mla_wide_hi(T::mla_wide_hi(T::mul_wide_hi(u, xs), v1, ys), v, zs);
        return montgomery_halves<T>(low, high);
    } else {
        const W t = Ops::template reduce<2 * kP>(T::add(Ops::shoup(u, T::broadcast(s.x), T::broadcast(s.xq)),
                                                         Ops::shoup(v1, T::broadcast(s.y), T::broadcast(s.yq))));
        const W r = Ops::template reduce<2 * kP>(T::add(t, Ops::shoup(v, T::broadcast(s.z), T::broadcast(s.zq))));
        return Ops::template reduce<kP>(r);
    }
}

// out = (x u + y X v + z v) kScale (step_vector) at the words [from, to), rounded out to whole
// vectors (from down, to up), from the top vector down: the next step's scalars read the top words
// first. Callers keep kPad words before u, v and out, and 2 kLanes after. v[j - 1] by unaligned
// loads: lane shifts of aligned ones ran 0.99-1.07x their time (notes/157-r1.md).
inline void combine3(const u32* u, const u32* v, u32* out, std::ptrdiff_t from, std::ptrdiff_t to, const Step& s) {
    constexpr std::ptrdiff_t L = std::ptrdiff_t(kLanes);
    const std::ptrdiff_t first = from & ~(L - 1);
    for (std::ptrdiff_t j = (to - 1) & ~(L - 1); j >= first; j -= L)
        S::store(out + j, step_vector<S>(S::load(u + j), S::load(v + j - 1), S::load(v + j), s));
}

// Word j of combine3 from u[j], v[j - 1], v[j], in scalar: the same canonical value.
inline u32 combine3_word(const Step& s, u32 uj, u32 vj1, u32 vj) {
    const u64 t = u64(s.x) * uj + u64(s.y) * vj1 + u64(s.z) * vj;
    if constexpr (kWide) return montgomery(t);
    else return u32(t % kP);
}

inline constexpr std::size_t kPad = std::max<std::size_t>(kLanes, 4);  // zero words before Euclid's arrays

// ---- Values-form passes ------------------------------------------------------------------------

// (x0 y0 + x1 y1) 2^-32 mod P, canonical, for canonical words. kWide: the two products in 64-bit
// sums (< 2 P^2 + 2^32 P < 2^64), one Montgomery reduction; else Montgomery products, added.
template <class T>
[[gnu::always_inline]] inline typename T::V products_vector(typename T::V x0, typename T::V y0, typename T::V x1, typename T::V y1) {
    using Ops = modarith::Ops<T, kP>;
    if constexpr (requires { T::mul_even(x0, y0); }) {
        const auto odd = [](typename T::V v) { return T::template shr64<32>(v); };
        return montgomery_join<T>(T::add64(T::mul_even(x0, y0), T::mul_even(x1, y1)),
                                  T::add64(T::mul_even(odd(x0), odd(y0)), T::mul_even(odd(x1), odd(y1))));
    } else if constexpr (requires { T::mul_wide_lo(x0, y0); }) {
        return montgomery_halves<T>(T::mla_wide_lo(T::mul_wide_lo(x0, y0), x1, y1), T::mla_wide_hi(T::mul_wide_hi(x0, y0), x1, y1));
    } else {
        return Ops::add(Ops::template reduce<kP>(Ops::montgomery(x0, y0)), Ops::template reduce<kP>(Ops::montgomery(x1, y1)));
    }
}

// out[i] = (x0 y0 + x1 y1) 2^-32 mod P word by word (canonical words, n a multiple of kLanes), and
// kept[i] the same if kept is not null.
inline void products(const u32* x0, const u32* y0, const u32* x1, const u32* y1, u32* kept, u32* out, std::size_t n) {
    if (!kept) {
        for (std::size_t i = 0; i < n; i += kLanes)
            S::store(out + i, products_vector<S>(S::load(x0 + i), S::load(y0 + i), S::load(x1 + i), S::load(y1 + i)));
        return;
    }
    for (std::size_t i = 0; i < n; i += kLanes) {
        const V m = products_vector<S>(S::load(x0 + i), S::load(y0 + i), S::load(x1 + i), S::load(y1 + i));
        S::store(kept + i, m);
        S::store(out + i, m);
    }
}

// ---- Division ------------------------------------------------------------------------------------

inline constexpr std::size_t kShortQuotient = 32;  // long division (k d word steps) up to this many words of q

// Stack words divide_in() takes for f of n and g of m >= 1 words.
inline std::size_t divide_words(std::size_t n, std::size_t m) {
    const std::size_t k = division::quotient_size(n, m);
    if (k == 0) return 0;
    if (k <= kShortQuotient) return footprint(k) + footprint(n);
    return footprint(k) + footprint(division::scratch_words(n, m, true));
}

// r = f mod g (m - 1 words, m = g.size(), g[m - 1] != 0), and q = f quo g into q if not empty (else
// into the stack): long division for short quotients, else division::divide in the stack (or the
// heap, if the stack is short; not for the sizes divide_words() counts).
inline void divide_in(Stack& stack, Span f, Span g, Out q, Out r) {
    const std::size_t k = division::quotient_size(f.size(), g.size());
    if (k == 0) {
        std::copy(f.begin(), f.end(), r.begin());
        std::fill(r.begin() + std::ptrdiff_t(f.size()), r.end(), 0u);
        return;
    }
    u32* const mark = stack.mark();
    if (q.empty()) q = stack.take_span(k);
    if (k <= kShortQuotient) {
        division::detail::long_division(f, g, q, r, stack.take(f.size()));
    } else {
        const std::size_t words = division::scratch_words(f.size(), g.size(), true);
        if (stack.room() >= footprint(words)) {
            division::divide(f, g, q, r, stack.take_span(words));
        } else {
            std::vector<u32> scratch(words);
            division::divide(f, g, q, r, scratch);
        }
    }
    stack.release(mark);
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
    // direct: jumps of at most this many degrees run Euclid's algorithm (tests force others), and
    // jumps of k on fewer than 2k + 1 words up to whole (at least direct), as the spine's tail.
    HalfGcd(Stack& stack, std::ptrdiff_t direct, std::ptrdiff_t whole = 0)
        : stack_(stack), direct_(direct), whole_(std::max(direct, whole)) {}

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
        if (k <= direct_ || (n < 2 * k && k <= whole_)) return euclid(a, n, b, m, k, out, need);
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
        while (m >= 0 && n > whole_) {
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
        steps<false>(a, n, b, m, e, 0, kNoFloor, nullptr);
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

    // n words, zero, with kPad zero words before them and 2 kLanes after (combine3, steps()).
    u32* take_padded(std::size_t n) {
        u32* const p = take(n + kPad + 2 * kLanes);
        std::fill_n(p, n + kPad + 2 * kLanes, 0u);
        return p + kPad;
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

    // One column of Euclid's matrix by rows: row0 of c (u a + v b = c), row1 of d, row2 free; zero
    // past their sizes. limit 1: its coefficient 0 only (for at0), when no entry in need has it.
    struct Column {
        u32 *row0, *row1, *row2;
        std::size_t size0, size1, size2, limit;

        // row2 = the next row by s, then rotated: (row0, row1) = (d's row, e's row).
        [[gnu::always_inline]] void step(const Step& s) {
            const std::size_t se = std::min(limit, std::max(size0, size1 ? size1 + 1 : 0));
            combine3(row0, row1, row2, 0, std::ptrdiff_t(se), s);
            // The passes write whole vectors past the sizes: zeros, from zeros (sizes stay below
            // the limit), or words nobody reads (limit 1).
            if (size2 > se) std::fill(row2 + se, row2 + size2, 0u);
            size2 = se;
            rotate();
        }
        [[gnu::always_inline]] void rotate() {
            u32* const r = row0;
            row0 = row1, row1 = row2, row2 = r;
            const std::size_t z = size0;
            size0 = size1, size1 = size2, size2 = z;
        }
    };
    // The d row of the last step only if last_d_row.
    struct Rows {
        Column column[2];
        bool last_d_row;
    };

    static constexpr std::ptrdiff_t kNoFloor = std::numeric_limits<std::ptrdiff_t>::min() / 2;  // steps()' floor: all words

    // Euclid's fraction-free steps on (c, d), deg c = n > deg d = m, while m >= target, on the
    // coefficients from floor + (progress so far) (all of them for kNoFloor), which are all the
    // remaining quotients read: (c, d) end as (r_h, r_(h+1)) up to constants, m < target (m = -1:
    // d = 0). c, d, e: take_padded arrays of n + 1 words. The rows (kRows) take the same steps.
    // A run of degree-1 steps keeps the top two words of c and d in scalars and computes the next
    // remainder's from them (combine3_word): the chain from one step's factors to the next is
    // scalar, and the vectors feed it words 3 and 4 from the top a step later. The state lives in
    // locals: in memory (the old Rows), its loads waited on its own stores each step (notes/157-r1.md).
    template <bool kRows>
    static void steps(u32*& c_io, std::ptrdiff_t& n_io, u32*& d_io, std::ptrdiff_t& m_io, u32*& e_io, std::ptrdiff_t target,
                      std::ptrdiff_t floor, Rows* rows_io) {
        u32 *c = c_io, *d = d_io, *e = e_io;
        std::ptrdiff_t n = n_io, m = m_io;
        Rows rows;
        if constexpr (kRows) rows = *rows_io;
        while (m >= target) {
            if (n - m > 1) {
                floor += n - m;
                if constexpr (kRows) *rows_io = rows;
                pseudo_divide(c, n, d, m, e, std::max<std::ptrdiff_t>(0, floor), kRows ? rows_io : nullptr);
                if constexpr (kRows) rows = *rows_io, rows.column[0].rotate(), rows.column[1].rotate();
                u32* const old = c;
                c = d, d = e, e = old;
                n = m, m = n - 1;
                while (m >= target && d[m] == 0) --m;
                continue;
            }
            u32 beta = c[n], c1 = c[m], alpha = d[m], d1 = d[m - 1];
            while (true) {  // n = m + 1; alpha, beta, c1, d1 = d[m], c[n], c[m], d[m - 1]
                const Step s(alpha, beta, c1, d1);
                const u32 e1 = combine3_word(s, c[m - 1], d[m - 2], d1);  // e[m - 1], e[m - 2]
                const u32 e2 = combine3_word(s, c[m - 2], d[m - 3], d[m - 2]);
                ++floor;
                combine3(c, d, e, std::max<std::ptrdiff_t>(0, floor), m, s);
                if constexpr (kRows)
                    if (m > target || rows.last_d_row) rows.column[0].step(s), rows.column[1].step(s);
                    else rows.column[0].rotate(), rows.column[1].rotate();
                u32* const old = c;
                c = d, d = e, e = old;
                n = m, m = n - 1;
                beta = alpha, c1 = d1, alpha = e1, d1 = e2;
                if (m < target || e1 == 0) break;
            }
            while (m >= target && d[m] == 0) --m;
        }
        c_io = c, d_io = d, e_io = e, n_io = n, m_io = m;
        if constexpr (kRows) *rows_io = rows;
    }

    // Euclid's algorithm with fraction-free steps on the coefficients of degree >= the floor
    // n - 2k + (progress so far); only the entries in need (and every at0 of row 0).
    void euclid(const u32* a, std::ptrdiff_t n, const u32* b, std::ptrdiff_t m, std::ptrdiff_t k, Matrix& out, unsigned need) {
        u32* const mark = stack_.mark();
        const std::ptrdiff_t floor = n - 2 * k, lo = std::max<std::ptrdiff_t>(0, floor);
        u32* p[3];
        for (auto& x : p) x = take_padded(std::size_t(n + 1));
        std::copy(a + lo, a + n + 1, p[0] + lo);
        std::copy(b + lo, b + m + 1, p[1] + lo);
        Rows rows;
        for (int j = 0; j < 2; ++j) {
            Column& col = rows.column[j];
            col.limit = need & (5u << j) ? std::size_t(k) + 1 : 1;
            col.row0 = take_padded(col.limit), col.row1 = take_padded(col.limit), col.row2 = take_padded(col.limit);
            col.size0 = j == 0, col.size1 = j == 1, col.size2 = 0;
        }
        rows.column[0].row0[0] = 1, rows.column[1].row1[0] = 1;
        rows.last_d_row = need & 12;
        u32 *c = p[0], *d = p[1], *e = p[2];
        std::ptrdiff_t c_deg = n, d_deg = m;
        steps<true>(c, c_deg, d, d_deg, e, n - k, floor, &rows);
        for (int j = 0; j < 2; ++j) {
            const Column& col = rows.column[j];
            const u32* const row[2] = {col.row0, col.row1};
            const std::size_t size[2] = {col.size0, col.size1};
            for (int i = 0; i < 2; ++i) {
                std::copy_n(row[i], size[i], out.entry[2 * i + j]);
                out.size[2 * i + j] = size[i];
                out.at0[2 * i + j] = size[i] ? row[i][0] : 0;
                out.length[2 * i + j] = 0;
            }
        }
        out.progress = std::size_t(n - c_deg);
        stack_.release(mark);
    }

    // A quotient of degree delta = c_deg - d_deg > 1: e = c, then e = alpha e - e[d_deg + t] x^t d
    // for t = delta .. 0 (alpha = lc d), on the coefficients from lo; the rows alike (row[2] from
    // row[0] and row[1]) if given.
    static void pseudo_divide(const u32* c, std::ptrdiff_t c_deg, const u32* d, std::ptrdiff_t d_deg, u32* e,
                              std::ptrdiff_t lo, Rows* rows) {
        const u32 alpha = d[d_deg];
        std::copy(c + lo, c + c_deg + 1, e + lo);
        if (rows)
            for (Column& col : rows->column) {
                std::copy_n(col.row0, col.size0, col.row2);
                if (col.size2 > col.size0) std::fill(col.row2 + col.size0, col.row2 + col.size2, 0u);
                col.size2 = col.size0;
            }
        for (std::ptrdiff_t t = c_deg - d_deg; t >= 0; --t) {
            const u32 minus_tau = negate(e[d_deg + t]);
            for (std::ptrdiff_t j = lo; j <= d_deg + t; ++j) {
                const u32 dj = j - t >= 0 ? d[j - t] : 0;
                e[j] = u32((u64(alpha) * e[j] + u64(minus_tau) * dj) % kP);
            }
            if (!rows) continue;
            for (Column& col : rows->column) {
                const std::size_t shift = std::size_t(t);
                const std::size_t se = std::min(col.limit, std::max(col.size2, col.size1 ? col.size1 + shift : 0));
                for (std::size_t j = 0; j < se; ++j) {
                    const u32 dj = j >= shift ? col.row1[j - shift] : 0;  // zero past size1 (limit 1: j = 0 only)
                    col.row2[j] = u32((u64(alpha) * col.row2[j] + u64(minus_tau) * dj) % kP);
                }
                col.size2 = se;
            }
        }
    }

    // rest = a mod b (m words; b of m + 1 coefficients, b[m] != 0), q (n - m + 1 words) = a quo b if
    // given (divide_in).
    void remainder(const u32* a, std::ptrdiff_t n, const u32* b, std::ptrdiff_t m, u32* rest, u32* q = nullptr) {
        const Out quotient = q ? Out(q, std::size_t(n - m + 1)) : Out();
        divide_in(stack_, Span(a, std::size_t(n + 1)), Span(b, std::size_t(m + 1)), quotient, Out(rest, std::size_t(m)));
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

    // out (len words) = the values form of length len of c mod (x^len - 1), c of size coefficients.
    static void values_of(const u32* c, std::size_t size, u32* out, std::size_t len) {
        division::detail::fold(Span(c, size), Out(out, len));
        transform(len).forward(out, ntt::Form::values);
    }

    // r.transform[i] = the values form of length len of entry i (room >= len): kept, doubled from
    // the half held (the upper half from the entry mod x^(len/2) + 1), or computed.
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
            transform(len).forward_upper(upper, ntt::Form::values);
        } else {
            values_of(r.entry[i], size, t, len);
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
        values_of(a, std::size_t(n + 1), ta, len);
        values_of(b, std::size_t(m + 1), tb, len);
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
                values_of(r1.entry[i], std::min(low, r1.size[i]), rt[i], short_len);
            }
            values_of(a, low, ta, short_len);
            values_of(b, mb, tb, short_len);
            apply_products(rt, ta, tb, lc, ld, short_len);
        } else {
            values_of(a, low, ta, len);
            values_of(b, mb, tb, len);
            apply_products(r1.transform, ta, tb, lc, ld, len);
            // Coefficient L of the low products (if n = 2L) wraps onto 0; the products are 2^-32 r1 (a, b).
            lc[0] = montgomery(u64(r1.at0[0]) * a[0] + u64(r1.at0[1]) * b[0]);
            ld[0] = montgomery(u64(r1.at0[2]) * a[0] + u64(r1.at0[3]) * b[0]);
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

    // c, d (len words each) = the coefficients of 2^-32 r (a, b) mod (x^len - 1) from the values
    // forms of length len of r's entries and of a, b.
    static void apply_products(u32* const (&r)[4], const u32* ta, const u32* tb, u32* c, u32* d, std::size_t len) {
        const ntt::Transform& t = transform(len);
        products(r[0], ta, r[1], tb, nullptr, c, len);
        products(r[2], ta, r[3], tb, nullptr, d, len);
        t.inverse(c, ntt::Form::values);
        t.inverse(d, ntt::Form::values);
    }

    // out = 2^-32 r2 r1 (the entries in need), progress p = r2.progress + r1.progress: cyclic
    // products of length L >= p (coefficient L, if p = L, wraps onto 0), whose values forms out
    // keeps if its room allows. Matrices only matter up to constants (one per matrix: the apply
    // and the combine of r1 read the same entries).
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
                const u32 x0 = montgomery(u64(r2.at0[2 * i]) * r1.at0[j] + u64(r2.at0[2 * i + 1]) * r1.at0[2 + j]);
                out.at0[ij] = x0;
                if (!(need >> ij & 1)) continue;
                // out[ij] = (r2[2i] r1[j] + r2[2i + 1] r1[2 + j]) 2^-32: its values kept, its
                // coefficients by the inverse of a copy in work.
                products(r2.transform[2 * i], r1.transform[j], r2.transform[2 * i + 1], r1.transform[2 + j],
                         keep ? out.transform[ij] : nullptr, work, len);
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
    std::ptrdiff_t direct_, whole_;
};

// ---- Sizes --------------------------------------------------------------------------------------

inline std::size_t trimmed(Span a) {
    std::size_t n = a.size();
    while (n > 0 && a[n - 1] == 0) --n;
    return n;
}

// Stack words of jumps on polynomials of degree <= n: 48 N for N = bit_ceil(n) (lc-opt's bound),
// and per level of the recursion the arrays' gaps. Random inputs used 31-34 N from 2^12 on (notes.md).
inline std::size_t half_gcd_words(std::size_t n) {
    const std::size_t len = std::max(kMinLength, std::bit_ceil(std::max<std::size_t>(n, 1)));
    const std::size_t levels = 2 * std::size_t(std::bit_width(len)) + 4;
    return 48 * len + levels * 32 * footprint(0) + 4096;
}

inline std::size_t gcd_words(std::size_t nf, std::size_t ng) {  // trimmed sizes, both > 0, nf >= ng
    return 2 * footprint(nf) + std::max(divide_words(nf, ng), 2 * footprint(ng) + half_gcd_words(ng));
}

inline std::size_t inverse_mod_words(std::size_t n, std::size_t m) {  // m >= 2
    const std::size_t d = m - 1;
    return footprint(d) + std::max(divide_words(n, m), half_gcd_words(d));
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
inline std::size_t gcd_run(Stack& stack, Span a, Span b, Out out, std::ptrdiff_t direct, std::ptrdiff_t whole) {
    const std::size_t na = a.size(), nb = b.size();
    // (a, b) <- (b, a mod b), then the spine.
    u32* const x = stack.take(na);
    u32* const y = stack.take(na);
    std::copy(b.begin(), b.end(), x);
    std::fill(x + nb, x + na, 0u);
    std::fill(y, y + na, 0u);
    HalfGcd h(stack, direct, whole);
    divide_in(stack, a, b, Out(), Out(y, nb - 1));
    std::ptrdiff_t n = std::ptrdiff_t(nb) - 1, m = n - 1;
    while (m >= 0 && y[m] == 0) --m;
    return h.spine(x, n, y, m, out);
}

// inverse_mod of f by g (m >= 2 words, g[m - 1] != 0) in the stack.
inline std::ptrdiff_t inverse_mod_run(Stack& stack, Span f, Span g, Out h, std::ptrdiff_t direct, std::ptrdiff_t whole) {
    const std::ptrdiff_t d = std::ptrdiff_t(g.size()) - 1;
    u32* const b = stack.take(std::size_t(d));
    divide_in(stack, f, g, Out(), Out(b, std::size_t(d)));
    std::ptrdiff_t m = d - 1;
    while (m >= 0 && b[m] == 0) --m;
    if (m < 0) return -1;
    HalfGcd gcd(stack, direct, whole);
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

// The gcd with Euclid below `direct` (tests force small values; whole: HalfGcd's).
inline std::size_t gcd(Span f, Span g, Out out, std::span<u32> scratch, std::ptrdiff_t direct, std::ptrdiff_t whole = 0) {
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
                      [&](Stack& stack) { return gcd_run(stack, f.first(nf), g.first(ng), out, direct, whole); });
}

inline std::ptrdiff_t inverse_mod(Span f, Span g, Out h, std::span<u32> scratch, std::ptrdiff_t direct,
                                  std::ptrdiff_t whole = 0) {
    const std::size_t m = g.size();
    if (m == 0 || g[m - 1] == 0 || h.size() != m - 1)
        throw std::invalid_argument("fps::half_gcd::inverse_mod: g empty or g[m - 1] = 0, or h not m - 1 words");
    if (m == 1) return 0;
    check_length(std::max(f.size(), m));
    f = f.first(trimmed(f));
    return with_stack(inverse_mod_words(f.size(), m), scratch,
                      [&](Stack& stack) { return inverse_mod_run(stack, f, g, h, direct, whole); });
}

}  // namespace detail

// Words of scratch gcd() takes for f of n and g of m words past their trailing zeros (any values).
inline std::size_t gcd_scratch(std::size_t n, std::size_t m) {
    if (n < m) std::swap(n, m);
    if (m <= 1) return 0;
    return detail::kAlignSlack + detail::gcd_words(n, m);
}

// Words of scratch inverse_mod() takes for f of n words past its trailing zeros and g of m.
inline std::size_t inverse_mod_scratch(std::size_t n, std::size_t m) {
    if (m <= 1) return 0;
    return detail::kAlignSlack + detail::inverse_mod_words(n, m);
}

// out[0, t) = the monic gcd of f and g, t returned (above).
inline std::size_t gcd(std::span<const u32> f, std::span<const u32> g, std::span<u32> out, std::span<u32> scratch = {}) {
    return detail::gcd(f, g, out, scratch, detail::kDirect, detail::kDirectWhole);
}

// h = 1 / f mod g; the number of coefficients of h, or -1 if gcd(f, g) != 1 (above).
inline std::ptrdiff_t inverse_mod(std::span<const u32> f, std::span<const u32> g, std::span<u32> h,
                                  std::span<u32> scratch = {}) {
    return detail::inverse_mod(f, g, h, scratch, detail::kDirect, detail::kDirectWhole);
}

}  // namespace fps::half_gcd
