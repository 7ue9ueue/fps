// Polynomials of degree < cap modulo P = 998244353 with Montgomery coefficients, for the
// factorization: gcd, division, products modulo a monic g by a table of x^(m + j) mod g, powers
// modulo g, square roots in F_P. Templates on a backend/simd struct S (kernels.hpp).
//
// Memory: a Poly's words come from a Pool of equal buffers (cap words, 64-byte aligned), taken
// zeroed and given back zeroed (the words below n are cleared on return), so a Poly costs no
// allocation once the pool is warm and no clearing of words it never used. One Pool per
// factorization; Polys must not outlive it.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "backend/factor/kernels.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace fps::factorization::detail {

class Pool {
public:
    // Buffers of cap words (a multiple of 16).
    explicit Pool(int cap) : cap_(cap) {}
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    ~Pool() {
        for (u32* p : free_) ::operator delete[](p, std::align_val_t(64));
    }

    int cap() const { return cap_; }
    u32* take() {
        if (free_.empty()) {
            u32* p = static_cast<u32*>(::operator new[](std::size_t(cap_) * sizeof(u32), std::align_val_t(64)));
            std::fill(p, p + cap_, 0);
            return p;
        }
        u32* p = free_.back();
        free_.pop_back();
        return p;
    }
    // p is zero from `used` words on.
    void give(u32* p, int used) {
        std::fill(p, p + used, 0);
        free_.push_back(p);
    }

private:
    int cap_;
    std::vector<u32*> free_;
};

// c[0, n) with c[n - 1] != 0 (n = 0: the zero polynomial); c is zero from n to the pool's cap, so
// kernels may read and write whole vectors past n.
class Poly {
public:
    int n = 0;
    u32* c = nullptr;

    explicit Poly(Pool& pool) : c(pool.take()), pool_(&pool) {}
    Poly(const Poly& o) : n(o.n), c(o.pool_->take()), pool_(o.pool_) { std::copy(o.c, o.c + o.n, c); }
    Poly(Poly&& o) noexcept : n(o.n), c(o.c), pool_(o.pool_) { o.c = nullptr, o.n = 0; }
    Poly& operator=(const Poly& o) {
        if (this != &o) {
            if (!c) c = pool_->take(), n = 0;
            if (n > o.n) std::fill(c + o.n, c + n, 0);
            std::copy(o.c, o.c + o.n, c);
            n = o.n;
        }
        return *this;
    }
    Poly& operator=(Poly&& o) noexcept {
        std::swap(n, o.n), std::swap(c, o.c), std::swap(pool_, o.pool_);
        return *this;
    }
    ~Poly() {
        if (c) pool_->give(c, n);
    }

    Pool& pool() const { return *pool_; }
    u32 lead() const { return c[n - 1]; }
    void trim() {
        while (n > 0 && c[n - 1] == 0) --n;
    }
    // Sets n to k and trims; the words from k on must be zero.
    void set_size(int k) {
        n = k;
        trim();
    }

private:
    Pool* pool_;
};

// A monic g of degree m >= 1 with T_j = x^(m + j) mod g, j < m - 1, at table + j stride.
struct Modulus {
    Poly g;
    int m = 0;
    int stride = 0;
    std::vector<u32> table;
};

// Whether S runs polynomials modulo g of degree <= 8 in one ymm register (lc-opt's register path).
// -DFPS_FACTOR_SMALL=0 turns it off.
template <class S>
constexpr bool small_path() {
#if defined(__AVX2__) && (!defined(FPS_FACTOR_SMALL) || FPS_FACTOR_SMALL)
    return S::lanes >= 8;
#else
    return false;
#endif
}

template <class S>
class Ring {
public:
    static constexpr int W = S::lanes;

    // Polynomials of degree < limit; pool buffers of cap(limit) words.
    static constexpr int cap(int limit) { return round_up(limit, 16) + 16; }

    explicit Ring(int limit)
        : pool_(cap(limit)), scratch_(std::size_t(23) * cap(limit) + 2 * kBlock<S> * W) {}

    Pool& pool() { return pool_; }
    int stride(int m) const { return round_up(m, std::max(W, 8)); }  // rows of m words

    Poly zero() { return Poly(pool_); }
    Poly constant(u32 a) {
        Poly r(pool_);
        r.c[0] = a;
        r.set_size(1);
        return r;
    }
    // x - z.
    Poly linear(u32 z) {
        Poly r(pool_);
        r.c[0] = neg(z), r.c[1] = kOne, r.n = 2;
        return r;
    }
    Poly from_words(const u32* a, int n) {
        Poly r(pool_);
        std::copy(a, a + n, r.c);
        r.set_size(n);
        return r;
    }

    void make_monic(Poly& a) const {
        const u32 s = inverse(a.lead());
        for (int i = 0; i < a.n; ++i) a.c[i] = mul(a.c[i], s);
    }

    // Monic gcd (zero if both are zero). Fraction-free Euclid: a <- lc(b) a - lc(a) x^s b kills
    // the leading term without inverses; remainders are right up to constant factors.
    Poly gcd(const Poly& a0, const Poly& b0) {
        int na = a0.n, nb = b0.n;
        const int lead = std::max(na, nb), span = round_up(lead, 16) + 16;  // reads reach a - lead, a + span
        u32* a = scratch_.data() + lead;
        u32* b = a + span + lead;
        for (u32* x : {a, b}) std::fill(x - lead, x + span, 0);
        std::copy(a0.c, a0.c + na, a);
        std::copy(b0.c, b0.c + nb, b);
        if (na < nb) std::swap(a, b), std::swap(na, nb);
        while (nb > 0) {
            while (na >= nb) {
                lincomb<S>(a, a, b[nb - 1], b - (na - nb), neg(a[na - 1]), na - 1);
                a[--na] = 0;
                while (na > 0 && a[na - 1] == 0) --na;
            }
            std::swap(a, b), std::swap(na, nb);
        }
        Poly g = from_words(a, na);
        if (na > 0) make_monic(g);
        return g;
    }

    // h = q d + r with deg r < deg d, d monic. q or r may be null.
    void divide(const Poly& h, const Poly& d, Poly* q, Poly* r) { divide_words(h.c, h.n, d, q, r); }

    // The same for h of hn <= 2 cap words (q only for hn <= cap).
    void divide_words(const u32* h, int hn, const Poly& d, Poly* q, Poly* r) {
        u32* a = scratch_.data();
        const int dd = d.n - 1;
        std::copy(h, h + hn, a);
        std::fill(a + hn, a + round_up(hn, 16) + 16, 0);
        if (q) *q = zero();
        for (int i = hn - 1; i >= dd; --i) {
            const u32 c = a[i];
            if (q) q->c[i - dd] = c;
            if (c != 0) axpy<S>(a + i - dd, a + i - dd, neg(c), d.c, dd + 1);  // a[i] -> 0
        }
        if (q) q->set_size(std::max(0, hn - dd));
        if (r) {
            *r = zero();
            const int k = std::min(hn, dd);
            std::copy(a, a + k, r->c);
            r->set_size(k);
        }
    }
    Poly quotient(const Poly& h, const Poly& d) {
        Poly q = zero();
        divide(h, d, &q, nullptr);
        return q;
    }
    Poly remainder(const Poly& h, const Poly& d) {
        if (h.n < d.n) return h;
        Poly r = zero();
        divide(h, d, nullptr, &r);
        return r;
    }

    // y = x a mod g for a of degree < m = deg g (y may be a).
    void times_x(const u32* a, u32* y, const Poly& g) {
        if (a == y) {
            u32* copy = scratch_.data();
            std::copy(a, a + round_up(g.n - 1, 16), copy);
            a = copy;
        }
        factorization::times_x<S>(y, a, g.c, g.n - 1);
    }

    // Column s = x column_(s-1) mod g at first + s stride for s = 1 .. count - 1, from column 0;
    // columns of m = deg g words, zero from m to the stride.
    void times_x_chain(u32* first, int count, std::ptrdiff_t stride, const Poly& g) {
        factorization::times_x_chain<S>(first, count, stride, g.c, g.n - 1);
    }

    Poly times_x(const Poly& a, const Modulus& M) {
        Poly r = zero();
        if (a.n == 0) return r;
        times_x(a.c, r.c, M.g);
        r.set_size(M.m);
        return r;
    }

    // The table goes into storage (its capacity is reused).
    Modulus modulus(const Poly& g, std::vector<u32>&& storage = {}) {
        Modulus M{g, 0, 0, {}};
        M.m = g.n - 1;
        M.stride = stride(M.m);
        M.table = std::move(storage);
        M.table.assign(std::size_t(std::max(M.m - 1, 0)) * M.stride, 0);
        if (M.m >= 2) {
            u32* t = M.table.data();
            for (int i = 0; i < M.m; ++i) t[i] = neg(g.c[i]);
            times_x_chain(t, M.m - 1, M.stride, g);
        }
        return M;
    }

    // a b mod g; a and b of degree < m.
    Poly multiply_mod(const Poly& a, const Poly& b, const Modulus& M) {
        Poly r = zero();
        if (a.n == 0 || b.n == 0) return r;
#if defined(__AVX2__)
        if constexpr (small_path<S>()) {
            if (M.m <= 8) {
                const __m256i x = multiply_small(load8(a.c), load8(b.c), M);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(r.c), x);
                r.set_size(M.m);
                return r;
            }
        }
#endif
        u32* product = scratch_.data() + 4 * pool_.cap();
        multiply<S>(a.c, a.n, b.c, b.n, product, product + 2 * pool_.cap());
        const int len = a.n + b.n - 1;
        const int k = std::min(len, M.m);
        std::copy(product, product + k, r.c);
        if (len > M.m) combine<S>(r.c, M.m, r.c, product + M.m, len - M.m, M.table.data(), M.stride);
        r.set_size(k);
        return r;
    }

    // a b mod h without a table; a and b of degree < cap.
    Poly multiply_remainder(const Poly& a, const Poly& b, const Poly& h) {
        Poly r = zero();
        if (a.n == 0 || b.n == 0) return r;
        u32* product = scratch_.data() + 4 * pool_.cap();
        multiply<S>(a.c, a.n, b.c, b.n, product, product + 2 * pool_.cap());
        divide_words(product, a.n + b.n - 1, h, nullptr, &r);
        return r;
    }

    // x^e mod g, e >= 1, deg g >= 2.
    Poly power_of_x(u64 e, const Modulus& M) {
        Poly r = linear(0);
        for (int bit = 62 - __builtin_clzll(e); bit >= 0; --bit) {
            r = multiply_mod(r, r, M);
            if (e >> bit & 1) r = times_x(r, M);
        }
        return r;
    }

    // (x + a)^e mod g, e >= 1, deg g >= 2.
    Poly power_of_linear(u32 a, u64 e, const Modulus& M) {
        Poly r = linear(neg(a));
#if defined(__AVX2__)
        if constexpr (small_path<S>()) {
            if (M.m <= 8) {
                __m256i x = load8(r.c);
                for (int bit = 62 - __builtin_clzll(e); bit >= 0; --bit) {
                    x = multiply_small(x, x, M);
                    if (e >> bit & 1) x = times_linear_small(x, a, M);
                }
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(r.c), x);
                r.set_size(M.m);
                return r;
            }
        }
#endif
        for (int bit = 62 - __builtin_clzll(e); bit >= 0; --bit) {
            r = multiply_mod(r, r, M);
            if (e >> bit & 1) {
                Poly s = times_x(r, M);
                axpy<S>(s.c, s.c, a, r.c, M.m);
                s.set_size(M.m);
                r = std::move(s);
            }
        }
        return r;
    }

    Poly derivative(const Poly& f) {
        Poly d = zero();
        for (int i = 1; i < f.n; ++i) d.c[i - 1] = mul(f.c[i], to_montgomery(u32(i)));
        d.set_size(std::max(f.n - 1, 0));
        return d;
    }

    // A square root of a square a (Montgomery) by Cipolla: for t with w = t^2 - a a non-square,
    // (t + y)^((P + 1) / 2) in F_P[y] / (y^2 - w). About 2 log P products in F_P^2, whatever the
    // power of 2 in P - 1 (Tonelli-Shanks takes up to log^2 P: 23 for 998244353).
    static u32 sqrt(u32 a) {
        const u32 minus_one = neg(kOne);
        u32 t = 0, w = neg(a);
        while (w != 0 && power(w, (kP - 1) / 2) != minus_one) t = add(t, kOne), w = sub(mul(t, t), a);
        if (w == 0) return t;
        u32 x0 = kOne, x1 = 0, b0 = t, b1 = kOne;  // x = x0 + x1 y, b = b0 + b1 y
        for (u64 e = (kP + 1) / 2; e; e >>= 1) {
            if (e & 1) {
                const u32 y0 = add(mul(x0, b0), mul(mul(x1, b1), w));
                x1 = add(mul(x0, b1), mul(x1, b0)), x0 = y0;
            }
            const u32 c0 = add(mul(b0, b0), mul(mul(b1, b1), w));
            b1 = mul(add(b0, b0), b1), b0 = c0;
        }
        return x0;
    }

    // 16 rows of stride(limit) words, not used by Ring's functions.
    u32* rows() { return scratch_.data() + 7 * pool_.cap() + 2 * kBlock<S> * W; }

private:
    Pool pool_;
    // Words from 0: gcd (2 (lead + span) <= 4 cap), divide and times_x (cap + 16); from 4 cap: a
    // product (2 cap), then multiply's scratch (cap + 2 kBlock W); then rows() (16 cap).
    std::vector<u32> scratch_;

#if defined(__AVX2__)
    static __m256i load8(const u32* a) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a)); }

    // Polynomials modulo g of degree m <= 8 in one vector (lanes from m on zero): chains of 30
    // squarings at tiny degrees (root finding, splitting) are latency-bound, so no memory trips.
    // a b mod g: product in two vectors (b shifted in registers), then c[0, m) + sum c[m + j] T_j.
    static __m256i multiply_small(__m256i a, __m256i b, const Modulus& M) {
        using A = EvenOdd<simd::Avx2>;
        const int m = M.m;
        const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), zero = _mm256_setzero_si256();
        A::Acc lo{zero, zero}, hi{zero, zero};
        for (int i = 0; i < m; ++i) {  // at most 8 products per sum: below 2^32 P after a fold
            const __m256i x = _mm256_permutevar8x32_epi32(a, _mm256_set1_epi32(i));
            const __m256i index = _mm256_sub_epi32(lanes, _mm256_set1_epi32(i));  // lane t: b[t - i]
            const __m256i rotated = _mm256_permutevar8x32_epi32(b, index);
            const __m256i below = _mm256_cmpgt_epi32(_mm256_set1_epi32(i), lanes);  // t < i: high half
            A::mac(lo, x, _mm256_andnot_si256(below, rotated));
            A::mac(hi, x, _mm256_and_si256(below, rotated));
        }
        A::fold(lo), A::fold(hi);
        const __m256i low = A::reduce(lo), high = A::reduce(hi);
        // c[m + j] is lane m + j of low, or lane m + j - 8 of high.
        const __m256i in_low = _mm256_cmpgt_epi32(_mm256_set1_epi32(m), lanes);
        const __m256i c = _mm256_and_si256(low, in_low);  // c 2^32 to start: the last step divides it out
        A::Acc s{_mm256_slli_epi64(c, 32), _mm256_and_si256(c, _mm256_set1_epi64x(~0xFFFFFFFFll))};
        for (int j = 0; j < m - 1; ++j) {
            const int t = m + j;
            const __m256i x = _mm256_permutevar8x32_epi32(t < 8 ? low : high, _mm256_set1_epi32(t & 7));
            A::mac(s, x, load8(M.table.data() + j * M.stride));
        }
        A::fold(s);
        return A::reduce(s);
    }

    // (x + a) u mod g = shifted u + a u - u[m - 1] g.
    static __m256i times_linear_small(__m256i u, u32 a, const Modulus& M) {
        using A = EvenOdd<simd::Avx2>;
        const int m = M.m;
        const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i shifted = _mm256_and_si256(
            _mm256_permutevar8x32_epi32(u, _mm256_sub_epi32(lanes, _mm256_set1_epi32(1))),
            _mm256_cmpgt_epi32(lanes, _mm256_setzero_si256()));
        const __m256i top = _mm256_permutevar8x32_epi32(u, _mm256_set1_epi32(m - 1));
        const __m256i minus_top = _mm256_sub_epi32(_mm256_set1_epi32(int(kP)), top);  // P - top, top < P
        A::Acc s = A::zero();
        A::mac(s, _mm256_set1_epi32(int(kOne)), shifted);
        A::mac(s, _mm256_set1_epi32(int(a)), u);
        A::mac(s, minus_top, load8(M.g.c));  // 3 products < 2^32 P
        return _mm256_and_si256(A::reduce(s), _mm256_cmpgt_epi32(_mm256_set1_epi32(m), lanes));  // lane m: 0
    }
#endif
};

}  // namespace fps::factorization::detail
