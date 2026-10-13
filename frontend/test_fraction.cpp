// Tests of fps/fraction.hpp: take(n) and [k] against the series p / q by its O(n^2) recurrence on
// edge sizes and random inputs; [k] at huge k against Kitamasa's method (x^k mod the reversed q,
// O(d^2 log k)), around the Bostan-Mori kernel's size limits, partial fractions (q with distinct
// roots, d up to 2048) and closed forms; every operation against its termwise definition;
// DESIGN.md's examples; threads; every error.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "fps/fps.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = 998244353;
using Poly = fps::Polynomial<P>;
using Frac = fps::Fraction<P>;

int failures = 0;

void check(bool ok, const char* what, int line, u64 a = 0, u64 b = 0, u64 c = 0) {
    if (ok) return;
    if (++failures <= 20)
        std::printf("FAIL line %d: %s (%llu, %llu, %llu)\n", line, what, (unsigned long long)a, (unsigned long long)b,
                    (unsigned long long)c);
}
#define CHECK(cond, ...) check(bool(cond), #cond, __LINE__ __VA_OPT__(, ) __VA_ARGS__)

template <class Exception, class F>
bool throws(F f) {
    try {
        f();
    } catch (const Exception&) {
        return true;
    }
    return false;
}

std::mt19937_64 rng(23);

constexpr u64 K18 = 1'000'000'000'000'000'000ull;
// Huge indices: around powers of two, 10^18, and the largest.
const std::vector<u64> kHuge = {(u64(1) << 20) - 1, u64(1) << 32, (u64(1) << 40) + 1, K18, K18 + 1,
                                (u64(1) << 63) - 1, u64(1) << 63, ~u64(0) - 1, ~u64(0)};

// 0, 1, powers of two and their neighbours (sizes of p and q).
const std::vector<std::size_t> kSizes = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65};
const std::vector<std::size_t> kLargeSizes = {127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025};

u32 add(u32 a, u32 b) { return a + b >= P ? a + b - P : a + b; }
u32 sub(u32 a, u32 b) { return a >= b ? a - b : a + P - b; }
u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }

u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}

u32 inv(u32 a) { return power(a, P - 2); }

// Kinds: 0 random, 1 mostly zeros, 2 all P - 1, 3 at most 20 nonzero terms past index 0 (the
// recurrence of inverse for <= 16). first: a[0] for n > 0 (any for p; nonzero for q).
Vec random_vec(std::size_t n, int kind) {
    Vec a(n);
    if (kind == 3) {
        const std::size_t t = 1 + rng() % 20;
        if (n) a[0] = u32(rng() % P);
        for (std::size_t i = 0; i < t && n > 1; ++i) a[1 + rng() % (n - 1)] = u32(1 + rng() % (P - 1));
        return a;
    }
    for (u32& c : a) {
        const u64 r = rng();
        c = kind == 2 ? P - 1 : kind == 1 && r % 4 ? 0 : u32(r % P);
    }
    return a;
}

// A denominator of n >= 1 coefficients: q[0] != 0.
Vec random_q(std::size_t n, int kind) {
    Vec q = random_vec(std::max<std::size_t>(n, 1), kind);
    if (q[0] == 0) q[0] = u32(1 + rng() % (P - 1));
    return q;
}

Vec vec(const Poly& f) { return Vec(f.span().begin(), f.span().end()); }
Poly poly(const Vec& a) { return Poly(std::span<const u32>(a)); }
u32 at(const Vec& f, std::size_t i) { return i < f.size() ? f[i] : 0; }

// Coefficients [0, n) of p / q: q[0] s[i] = p[i] - sum_{j >= 1} q[j] s[i - j].
Vec naive_series(const Vec& p, const Vec& q, std::size_t n) {
    Vec s(n);
    const u32 c = inv(q[0]);
    for (std::size_t i = 0; i < n; ++i) {
        u32 t = at(p, i);
        for (std::size_t j = 1; j <= i && j < q.size(); ++j) t = sub(t, mul(q[j], s[i - j]));
        s[i] = mul(t, c);
    }
    return s;
}

Vec naive_mul(const Vec& a, const Vec& b, std::size_t n) {
    Vec c(n);
    for (std::size_t i = 0; i < a.size() && i < n; ++i)
        for (std::size_t j = 0; j < b.size() && i + j < n; ++j) c[i + j] = add(c[i + j], mul(a[i], b[j]));
    return c;
}

// Coefficient k of p / q by Kitamasa's method, O(d^2 log k) for d = q.size() - 1. With q[0] = 1
// (both scaled), b_i = a_(s + i), s = max(0, p.size() - d), satisfies b_i = -sum_j q_j b_(i-j) for
// i >= d, so b_m = sum_i c_i b_i for x^m mod Q = sum_i c_i x^i, Q = x^d + q_1 x^(d-1) + ... + q_d.
u32 kitamasa(Vec p, Vec q, u64 k) {
    const u32 c = inv(q[0]);
    for (u32& v : p) v = mul(v, c);
    for (u32& v : q) v = mul(v, c);
    const std::size_t d = q.size() - 1;
    if (d == 0) return k < p.size() ? p[std::size_t(k)] : 0;
    const std::size_t s = p.size() > d ? p.size() - d : 0;
    if (k < s + d) return naive_series(p, q, std::size_t(k) + 1)[std::size_t(k)];
    const Vec b = [&] {
        const Vec a = naive_series(p, q, s + d);
        return Vec(a.begin() + std::ptrdiff_t(s), a.end());
    }();
    // a b mod Q, for a and b of d words.
    const auto mulmod = [&](const Vec& x, const Vec& y) {
        Vec t(2 * d);
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = 0; j < d; ++j) t[i + j] = add(t[i + j], mul(x[i], y[j]));
        for (std::size_t i = 2 * d - 1; i >= d; --i)  // x^i = -sum_j q_j x^(i - j)
            for (std::size_t j = 1; j <= d; ++j) t[i - j] = sub(t[i - j], mul(q[j], t[i]));
        t.resize(d);
        return t;
    };
    Vec r(d), base(d);
    r[0] = 1;  // x^0
    if (d == 1) base[0] = sub(0, q[1]);  // x mod (x + q_1)
    else base[1] = 1;
    for (u64 e = k - s; e; e >>= 1, base = mulmod(base, base))
        if (e & 1) r = mulmod(r, base);
    u32 sum = 0;
    for (std::size_t i = 0; i < d; ++i) sum = add(sum, mul(r[i], b[i]));
    return sum;
}

// F(k) mod P by fast doubling: F(k), F(k + 1).
std::pair<u32, u32> fibonacci(u64 k) {
    if (k == 0) return {0, 1};
    const auto [a, b] = fibonacci(k / 2);
    const u32 c = mul(a, sub(mul(2, b), a)), d = add(mul(a, a), mul(b, b));
    return k % 2 ? std::pair{d, add(c, d)} : std::pair{c, d};
}

// take(n) for n = 0, 1, the reference's length and some between; [k] for every k < n on small
// fractions, else edges and samples; [k] at huge k against Kitamasa when q is short.
void check_fraction(const Vec& p, const Vec& q, bool every_k) {
    const Frac a(poly(p), poly(q));
    CHECK(vec(a.numerator()) == p && vec(a.denominator()) == q);
    const std::size_t n = 2 * std::max(p.size(), q.size()) + 70;
    const Vec want = naive_series(p, q, n);
    for (const std::size_t m : {std::size_t(0), std::size_t(1), std::size_t(2), q.size(), n / 2, n})
        CHECK(vec(a.take(m)) == Vec(want.begin(), want.begin() + std::ptrdiff_t(m)), p.size(), q.size(), m);
    std::vector<std::size_t> ks;
    if (every_k) {
        for (std::size_t k = 0; k < n; ++k) ks.push_back(k);
    } else {
        for (const std::size_t k : {std::size_t(0), std::size_t(1), p.size() - 1, p.size(), q.size() - 1, q.size(),
                                    q.size() + 1, n - 1})
            if (k < n) ks.push_back(k);
        for (int t = 0; t < 8; ++t) ks.push_back(std::size_t(rng() % n));
    }
    for (const std::size_t k : ks) CHECK(a[k] == want[k], p.size(), q.size(), k);
    if (q.size() <= 17)
        for (const u64 k : kHuge) CHECK(a[k] == kitamasa(p, q, k), p.size(), q.size(), k);
    else if (q.size() <= 65)
        for (int t = 0; t < 3; ++t) {
            const u64 k = kHuge[rng() % kHuge.size()];
            CHECK(a[k] == kitamasa(p, q, k), p.size(), q.size(), k);
        }
}

void test_coefficients() {
    for (const std::size_t np : kSizes)
        for (const std::size_t nq : kSizes)
            for (const int kind : {0, 1, 2, 3}) {
                if (kind && (nq % 8 != 1 || np % 8 != 1)) continue;  // other kinds on 1, 9, 17, 33, 65
                check_fraction(random_vec(np, kind), random_q(nq, kind), np <= 17 && nq <= 17);
            }
    for (const std::size_t nq : kLargeSizes)
        for (const int kind : {0, 3}) {
            check_fraction(random_vec(nq - 1, 0), random_q(nq, kind), false);
            check_fraction(random_vec(2 * nq + 5, kind), random_q(nq, kind), false);
        }
    for (const std::size_t np : kLargeSizes) check_fraction(random_vec(np, 0), random_q(1 + rng() % 9, 0), false);
    // A q with trailing zeros, p longer than q, and the zero series.
    check_fraction(random_vec(40, 0), {3, 5, 0, 0, 0, 0}, true);
    check_fraction({}, {1}, true);
    check_fraction({0, 0, 0}, {7, 1}, true);
    CHECK(Frac(Poly{})[~u64(0)] == 0 && vec(Frac(Poly{}).take(5)) == Vec(5));
    CHECK(Frac(Poly{1, 2, 3})[2] == 3 && Frac(Poly{1, 2, 3})[3] == 0 && Frac(Poly{1, 2, 3})[~u64(0)] == 0);
}

// [k] is fps::bostan_mori::coefficient: schoolbook steps up to kSchoolbookMax words (64 to 96 by
// target), the values form beyond, which hands off to schoolbook once the cut to k + 1 words
// fits. Sizes on both sides of each limit; every k below the reference's length, and huge k
// against Kitamasa (on q without its trailing zeros, the same series).
void test_kernel_paths() {
    // At least 2: -DFPS_BOSTAN_MORI_SCHOOLBOOK_MAX=0 (never schoolbook) is a valid choice.
    const std::size_t m = std::max<std::size_t>(fps::bostan_mori::kSchoolbookMax, 2);
    const std::size_t cap = fps::bostan_mori::kSchoolbookCapacity;
    const auto check_all = [](const Vec& p, const Vec& q) {
        const Frac a(poly(p), poly(q));
        const std::size_t n = 2 * std::max(p.size(), q.size()) + 70;
        const Vec want = naive_series(p, q, n);
        for (std::size_t k = 0; k < n; ++k) CHECK(a[k] == want[k], p.size(), q.size(), k);
        Vec t = q;
        while (t.size() > 1 && t.back() == 0) t.pop_back();
        for (const u64 k : {u64(1) << 32, K18, ~u64(0)}) CHECK(a[k] == kitamasa(p, t, k), p.size(), q.size(), k);
    };
    for (const std::size_t s : {m - 1, m, m + 1, 2 * m + 1, cap, cap + 1}) {
        check_all(random_vec(s, 0), random_q(s, 0));
        check_all(random_vec(s - 1, 0), random_q(s, 3));
        check_all(random_vec(2 * s + 3, 0), random_q(s, 0));
        check_all(random_vec(s, 1), random_q(9, 0));
        // Trailing zeros: q of s words with 3 nonzero; p of s words with s / 2 nonzero.
        Vec q(s), p = random_vec(s, 0);
        q[0] = 3, q[1] = 5, q[2] = P - 1;
        std::fill(p.begin() + std::ptrdiff_t(s / 2), p.end(), 0);
        check_all(p, q);
    }
}

// q = prod (1 - r_i x), r_i distinct and nonzero, deg p < d: a_k = sum_i c_i r_i^k with
// c_i = p(1 / r_i) / prod_{j != i} (1 - r_j / r_i).
void test_partial_fractions() {
    for (const std::size_t d : {1, 2, 100, 1000, 2048}) {
        Vec r(d);
        for (std::size_t i = 0; i < d; ++i) r[i] = u32(1 + i + rng() % 400'000 * 2048);  // distinct
        Vec q{1};
        for (const u32 ri : r) {
            Vec t(q.size() + 1);
            for (std::size_t i = 0; i < q.size(); ++i) t[i] = add(t[i], q[i]), t[i + 1] = sub(t[i + 1], mul(q[i], ri));
            q = t;
        }
        const Vec p = random_vec(d, 0);
        Vec c(d);
        for (std::size_t i = 0; i < d; ++i) {
            const u32 x = inv(r[i]);
            u32 num = 0, den = 1;
            for (std::size_t j = p.size(); j-- > 0;) num = add(mul(num, x), p[j]);
            for (std::size_t j = 0; j < d; ++j)
                if (j != i) den = mul(den, sub(1, mul(r[j], x)));
            c[i] = mul(num, inv(den));
        }
        const Frac a(poly(p), poly(q));
        for (const u64 k : {u64(0), u64(d), u64(12345), K18, ~u64(0)}) {
            u32 want = 0;
            for (std::size_t i = 0; i < d; ++i) want = add(want, mul(c[i], power(r[i], k)));
            CHECK(a[k] == want, d, k);
        }
    }
    // Large d at an index past a cheap reference: against take.
    const std::size_t d = std::size_t(1) << 15;
    const Frac big(poly(random_vec(d, 0)), poly(random_q(d + 1, 0)));
    const std::size_t k = 200'000;
    CHECK(big[k] == big.take(k + 1)[k]);
}

void test_closed_forms() {
    const Poly x = Poly::x();
    const Frac fib(x, 1 - x - x * x), geo(Poly{1}, 1 - x);
    for (const u64 k : {u64(0), u64(1), u64(2), u64(6), u64(1000)}) CHECK(fib[k] == fibonacci(k).first, k);
    for (const u64 k : kHuge) {
        CHECK(fib[k] == fibonacci(k).first, k);
        CHECK(geo[k] == 1, k);
        CHECK(Frac(Poly{1}, 1 - 5 * x)[k] == power(5, k), k);                          // 5^k
        CHECK(Frac(Poly{1}, square(1 - x))[k] == u32((k % P + 1) % P), k);              // k + 1
        CHECK(Frac(Poly{1}, 1 - Poly::x(3))[k] == (k % 3 == 0), k);                     // period 3
        CHECK(Frac(Poly::x(2), 1 - 2 * Poly::x(2))[k] == (k % 2 == 0 && k ? power(2, k / 2 - 1) : 0), k);
    }
}

void test_operations() {
    const Poly x = Poly::x();
    const std::vector<std::pair<std::size_t, std::size_t>> shapes = {{0, 1},  {1, 1},   {3, 2},   {5, 3},
                                                                     {2, 5},  {17, 16}, {33, 9},  {64, 65},
                                                                     {1, 300}, {200, 3}, {600, 513}};
    for (const auto& [np, nq] : shapes)
        for (const int kind : {0, 1, 3}) {
            const Vec pa = random_vec(np, kind), qa = random_q(nq, kind);
            const Vec pb = random_vec(nq, kind), qb = random_q(np ? np : 1, kind);
            const Frac a(poly(pa), poly(qa)), b(poly(pb), poly(qb));
            constexpr std::size_t n = 120;
            const Vec ta = naive_series(pa, qa, n + 1), tb = naive_series(pb, qb, n);
            const auto take = [](const Frac& f) { return vec(f.take(n)); };
            Vec sum(n), diff(n), neg(n), prefix(n), shifted(n), deriv(n);
            u32 running = 0;
            for (std::size_t i = 0; i < n; ++i) {
                sum[i] = add(ta[i], tb[i]), diff[i] = sub(ta[i], tb[i]), neg[i] = sub(0, ta[i]);
                running = add(running, ta[i]), prefix[i] = running;
                shifted[i] = i >= 4 ? ta[i - 4] : 0;
                deriv[i] = mul(u32(i + 1), ta[i + 1]);
            }
            CHECK(take(a + b) == sum, np, nq);
            CHECK(take(a - b) == diff, np, nq);
            CHECK(take(-a) == neg, np, nq);
            CHECK(take(a * b) == naive_mul(ta, tb, n), np, nq);
            CHECK(take(a / (1 - x)) == prefix, np, nq);
            CHECK(take(a * Poly::x(4)) == shifted && take(Poly::x(4) * a) == shifted, np, nq);
            CHECK(take(fps::derivative(a)) == deriv, np, nq);
            // Scaling, with c taken mod P.
            for (const auto& [c, r] : std::vector<std::pair<long long, u32>>{
                     {0, 0}, {1, 1}, {P - 1, P - 1}, {-1, P - 1}, {-3LL * P - 5, P - 5}, {1LL << 40, power(2, 40)}}) {
                Vec scaled(n);
                for (std::size_t i = 0; i < n; ++i) scaled[i] = mul(r, ta[i]);
                CHECK(take(c * a) == scaled && take(a * c) == scaled, np, nq, r);
            }
            Vec scaled(n);
            for (std::size_t i = 0; i < n; ++i) scaled[i] = mul(u32((~u64(0)) % P), ta[i]);
            CHECK(take(~u64(0) * a) == scaled);
            // Products and quotients with a polynomial f, f[0] != 0.
            const Vec f = random_q(1 + rng() % 40, kind);
            CHECK(take(poly(f) * a) == naive_mul(f, ta, n) && take(a * poly(f)) == naive_mul(f, ta, n), np, nq);
            CHECK(take(a / poly(f)) == naive_series(pa, naive_mul(qa, f, qa.size() + f.size() - 1), n), np, nq);
            // a / b = (a / b) b / b: ((a / b) * b) has the series of a.
            if (!pb.empty() && pb[0] != 0) {
                CHECK(take(a / b * b) == Vec(ta.begin(), ta.begin() + n), np, nq);
                CHECK(take(a / b) == naive_series(naive_mul(pa, qb, pa.size() + qb.size()),
                                                  naive_mul(qa, pb, qa.size() + pb.size()), n), np, nq);
            }
            // Results are fractions: their huge coefficients against Kitamasa.
            if (nq <= 65 && np <= 65)
                for (const Frac& r : {a + b, a * b, a / (1 - x), fps::derivative(a)}) {
                    const u64 k = kHuge[rng() % kHuge.size()];
                    CHECK(r[k] == kitamasa(vec(r.numerator()), vec(r.denominator()), k), np, nq, k);
                }
        }
}

// DESIGN.md's examples (Fractions).
void test_examples() {
    const Poly x = Poly::x();
    const Frac geo(Poly{1}, 1 - x);
    CHECK(geo[K18] == 1);
    const Frac fib(x, 1 - x - x * x);
    CHECK(fib[6] == 8);
    CHECK(fib[K18] == fibonacci(K18).first);
    CHECK(vec(fib.take(10)) == (Vec{0, 1, 1, 2, 3, 5, 8, 13, 21, 34}));
    CHECK((fib / (1 - x))[K18] == sub(fibonacci(K18 + 2).first, 1));  // F_0 + ... + F_K = F_(K+2) - 1
    {
        // sum_i F_i F_(K-i) = (K L_K - F_K) / 5, L_K = F_(K-1) + F_(K+1).
        const u64 K = K18;
        const u32 lucas = add(fibonacci(K - 1).first, fibonacci(K + 1).first);
        const u32 want = mul(sub(mul(u32(K % P), lucas), fibonacci(K).first), inv(5));
        CHECK((fib * fib)[K] == want);
        const Vec t = vec(fib.take(301));
        CHECK((fib * fib)[300] == naive_mul(t, t, 301)[300]);
    }
    CHECK((x * fps::derivative(fib))[K18] == mul(u32(K18 % P), fibonacci(K18).first));
    CHECK((fib + 3 * geo)[K18] == add(fibonacci(K18).first, 3));
    const Frac strings(1 + x, 1 - x - x * x);
    CHECK(strings[3] == 5);
    CHECK(vec(strings.take(6)) == (Vec{1, 2, 3, 5, 8, 13}));
    const Frac tilings(1 - x * x, 1 - 4 * x * x + Poly::x(4));
    CHECK(tilings[10] == 571);
    CHECK(vec(tilings.take(9)) == (Vec{1, 0, 3, 0, 11, 0, 41, 0, 153}));
}

void test_threads() {
    const Poly x = Poly::x();
    const Frac fib(x, 1 - x - x * x);
    const Frac big(poly(random_vec(3000, 0)), poly(random_q(3001, 0)));
    const u32 want = big[K18];
    std::vector<std::thread> threads;
    std::vector<int> ok(4, 1);
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&, t] {
            for (int i = 0; i < 3; ++i)
                ok[t] &= big[K18] == want && fib[K18 + u64(i)] == fibonacci(K18 + u64(i)).first;
        });
    for (std::thread& t : threads) t.join();
    CHECK(std::ranges::all_of(ok, [](int v) { return v == 1; }));
}

void test_errors() {
    using std::domain_error;
    const Poly x = Poly::x();
    const Frac a(Poly{1, 2}, Poly{1, 3});
    CHECK(throws<domain_error>([] { (void)Frac(Poly{1}, Poly{}); }));
    CHECK(throws<domain_error>([] { (void)Frac(Poly{1}, Poly{0, 1}); }));
    CHECK(throws<domain_error>([] { (void)Frac(Poly{}, Poly{0}); }));
    CHECK(throws<domain_error>([&] { (void)(a / x); }));
    CHECK(throws<domain_error>([&] { (void)(a / Poly{}); }));
    CHECK(throws<domain_error>([&] { (void)(a / Poly{0, 0, 5}); }));
    CHECK(throws<domain_error>([&] { (void)(a / Frac(x, 1 - x)); }));
    CHECK(throws<domain_error>([&] { (void)(a / Frac(Poly{})); }));
    CHECK(throws<domain_error>([&] { (void)(a / (a - a)); }));
    CHECK(!throws<domain_error>([&] { (void)(a / Frac(Poly{2}, x + 1)); }));
}

}  // namespace

int main() {
    test_coefficients();
    test_kernel_paths();
    test_partial_fractions();
    test_closed_forms();
    test_operations();
    test_examples();
    test_threads();
    test_errors();
    if (failures) {
        std::printf("fraction: %d failures\n", failures);
        return 1;
    }
    std::printf("fraction: all passed\n");
    return 0;
}
