// Tests of fps/series.hpp: pow, sqrt and their raw forms, against O(n^2) references (repeated
// truncated products for any e < 2^64, the ODE recurrence, the root's own recurrence) on edge sizes
// and random inputs with leading zeros, on both paths (the backend's backend/series, and the
// recurrence for at most 16 terms past the lowest), against closed forms and identities at large
// sizes, with threads and every precondition; each path's limit past 2^25 coefficients. Also
// detail::mod_sqrt.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include "fps/fps.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = 998244353;
using Poly = fps::Polynomial<P>;

#if defined(__SANITIZE_ADDRESS__)
constexpr bool kSanitized = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
constexpr bool kSanitized = true;
#else
constexpr bool kSanitized = false;
#endif
#else
constexpr bool kSanitized = false;
#endif

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

std::mt19937_64 rng(22);

// 0, 1, powers of two and their neighbours.
const std::vector<std::size_t> kSizes = {0,  1,  2,  3,   4,   5,   7,   8,   9,   15,  16,   17,   31,  32,  33,
                                         63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025};

// Exponents: small, around P and 2^64, random.
const std::vector<u64> kExponents = {0, 1, 2, 3, 7, P - 1, P, P + 1, 2 * u64(P) + 3, ~u64(0), u64(1) << 63};

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 add(u32 a, u32 b) { return a + b >= P ? a + b - P : a + b; }
u32 sub(u32 a, u32 b) { return a >= b ? a - b : a + P - b; }

u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}

u32 inv(u32 a) { return power(a, P - 2); }
bool is_square(u32 a) { return a == 0 || power(a, (P - 1) / 2) == 1; }

u64 random_exponent() {
    switch (rng() % 4) {
    case 0: return rng() % 10;
    case 1: return rng();
    case 2: return kExponents[rng() % kExponents.size()];
    default: return rng() % (3 * u64(P));
    }
}

// Kinds: 0 random, 1 mostly zeros, 2 all P - 1, 3 random with t terms past index 0 (the
// recurrence for t <= 16), t = 1 + rng() % 20.
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

// z zeros, then a with a[0] = c (nonzero), size coefficients in all (or z + 1 if more).
Vec with_zeros(std::size_t z, Vec a, u32 c) {
    if (a.empty()) a.push_back(0);
    a[0] = c;
    a.insert(a.begin(), z, 0);
    return a;
}

Vec vec(const Poly& f) { return Vec(f.span().begin(), f.span().end()); }
Poly poly(const Vec& a) { return Poly(std::span<const u32>(a)); }
u32 at(const Vec& f, std::size_t i) { return i < f.size() ? f[i] : 0; }

// Frees n words of P - 1, so the next allocation of n words likely reuses them.
void dirty_heap(std::size_t n) { (void)poly(Vec(n, P - 1)); }

// a b mod x^n, O(n^2).
Vec naive_mul(const Vec& a, const Vec& b, std::size_t n) {
    Vec c(n);
    for (std::size_t i = 0; i < std::min(a.size(), n); ++i)
        if (a[i])
            for (std::size_t j = 0; j < b.size() && i + j < n; ++j) c[i + j] = add(c[i + j], mul(a[i], b[j]));
    return c;
}

// f^e mod x^n by repeated squaring: O(n^2 log e), for any e (no reasoning about e mod P).
Vec pow_by_squaring(Vec f, u64 e, std::size_t n) {
    Vec g(n);
    if (n) g[0] = 1;
    f.resize(std::min(f.size(), n));
    for (; e; e >>= 1) {
        if (e & 1) g = naive_mul(g, f, n);
        if (e > 1) f = naive_mul(f, f, n);
    }
    return g;
}

// f^e mod x^n for f = x^z u: x^(z e) u[0]^e (u / u[0])^e, the last by the ODE u g' = e u' g,
// u[0] i g[i] = sum_d u[d] ((e + 1) d - i) g[i - d]. O(n^2).
Vec naive_pow(const Vec& f, u64 e, std::size_t n) {
    Vec g(n);
    if (n == 0) return g;
    if (e == 0) return g[0] = 1, g;
    std::size_t z = 0;
    while (z < std::min(f.size(), n) && f[z] == 0) ++z;
    if (z >= std::min(f.size(), n) || (z > 0 && e >= (n + z - 1) / z)) return g;
    const std::size_t shift = z * e, m = n - shift;
    const Vec u(f.begin() + z, f.begin() + std::min(f.size(), z + m));
    const u32 c = inv(u[0]), e1 = u32((e % P + 1) % P);
    Vec h(m);
    h[0] = power(u[0], e);
    for (std::size_t i = 1; i < m; ++i) {
        u32 s = 0;
        for (std::size_t d = 1; d <= i; ++d) s = add(s, mul(mul(at(u, d), sub(mul(e1, u32(d)), u32(i))), h[i - d]));
        h[i] = mul(mul(s, c), inv(u32(i)));
    }
    std::copy(h.begin(), h.end(), g.begin() + shift);
    return g;
}

// The root of the polynomial f mod x^n = x^(2k) u as fps::sqrt defines it, for c the smaller
// root of u[0] != 0: x^k s, 2 c s[i] = u[i] - sum_{0 < j < i} s[j] s[i - j]. O(n^2).
Vec naive_sqrt(const Vec& f, std::size_t n, u32 c) {
    Vec g(n);
    std::size_t z = 0;
    while (f[z] == 0) ++z;
    const std::size_t k = z / 2, m = n - k;
    const u32 i2c = inv(mul(2, c));
    Vec s(m);
    s[0] = c;
    for (std::size_t i = 1; i < m; ++i) {
        u32 t = at(f, z + i);
        if (z + i >= std::min(f.size(), n)) t = 0;
        for (std::size_t j = 1; j < i; ++j) t = sub(t, mul(s[j], s[i - j]));
        s[i] = mul(t, i2c);
    }
    std::copy(s.begin(), s.end(), g.begin() + k);
    return g;
}

// raw::<op> into out of n words that start as P - 1: every word must be written.
Vec raw_pow(const Vec& f, u64 e, std::size_t n) {
    Vec out(n, P - 1);
    fps::raw::pow<P>(std::span<const u32>(f), e, std::span<u32>(out));
    return out;
}

std::optional<Vec> raw_sqrt(const Vec& f, std::size_t n) {
    Vec out(n, P - 1);
    if (!fps::raw::sqrt<P>(std::span<const u32>(f), std::span<u32>(out))) {
        if (out != Vec(n, P - 1)) return Vec{12345};  // written though it has no root
        return std::nullopt;
    }
    return out;
}

void test_mod_sqrt() {
    for (u32 a = 0; a < 2000; ++a) {
        const auto r = fps::detail::mod_sqrt<P>(a);
        CHECK(r.has_value() == is_square(a), a);
        if (r) CHECK(mul(*r, *r) == a && *r <= P - *r, a, *r);
    }
    for (int t = 0; t < 2000; ++t) {
        const u32 x = u32(rng() % P), a = mul(x, x);
        const auto r = fps::detail::mod_sqrt<P>(a);
        CHECK(r && (*r == x || *r == P - x) && *r <= P - *r, a, x);
        const u32 b = u32(rng() % P);
        CHECK(fps::detail::mod_sqrt<P>(b).has_value() == is_square(b), b);
    }
    CHECK(*fps::detail::mod_sqrt<P>(P - 1) == power(3, (P - 1) / 4) || *fps::detail::mod_sqrt<P>(P - 1) == P - power(3, (P - 1) / 4));
    // Small primes, every residue: the smallest root from a table of squares.
    const auto all = []<u32 Q>(std::integral_constant<u32, Q>) {
        Vec root(Q, Q);
        for (u32 x = Q; x-- > 0;) root[u64(x) * x % Q] = x;
        for (u32 a = 0; a < Q; ++a) {
            const auto r = fps::detail::mod_sqrt<Q>(a);
            CHECK(r.has_value() == (root[a] < Q), Q, a);
            if (r) CHECK(*r == root[a], Q, a, *r);
        }
    };
    all(std::integral_constant<u32, 3>{});
    all(std::integral_constant<u32, 5>{});
    all(std::integral_constant<u32, 17>{});
    all(std::integral_constant<u32, 97>{});
    all(std::integral_constant<u32, 7681>{});
    for (int t = 0; t < 1000; ++t) {  // the largest prime below 2^32
        constexpr u32 Q = 4294967291u;
        const u32 x = u32(rng() % Q), a = u32(u64(x) * x % Q);
        const auto r = fps::detail::mod_sqrt<Q>(a);
        CHECK(r && u64(*r) * *r % Q == a && *r <= Q - *r, a);
    }
}

// pow at n against the ODE reference, for f of n - 1, n, n + 2 coefficients with z leading zeros,
// and against repeated squaring up to 65 coefficients.
void test_pow_small() {
    for (const std::size_t n : kSizes)
        for (const int kind : {0, 1, 2, 3})
            for (const std::size_t z : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(5)})
                for (const std::size_t size : {n ? n - 1 : 0, n, n + 2}) {
                    const u64 e = random_exponent();
                    const Vec f = size > z ? with_zeros(z, random_vec(size - z, kind), u32(1 + rng() % (P - 1)))
                                           : Vec(size, 0);
                    const Vec g = naive_pow(f, e, n);
                    dirty_heap(n);
                    CHECK(vec(fps::pow(poly(f), e, n)) == g, n, e, z);
                    CHECK(raw_pow(f, e, n) == g, n, e, z);
                    if (size == n) CHECK(vec(fps::pow(poly(f), e)) == g, n, e);
                    if (n <= 65) CHECK(pow_by_squaring(f, e, n) == g, n, e, z);
                }
    // Every exponent of the list on short series, against repeated squaring.
    for (const std::size_t n : {std::size_t(1), std::size_t(2), std::size_t(9), std::size_t(33), std::size_t(64)})
        for (const u64 e : kExponents)
            for (const std::size_t z : {0, 1, 3})
                for (const int kind : {0, 3}) {
                    const Vec f = with_zeros(z, random_vec(n, kind), u32(1 + rng() % (P - 1)));
                    CHECK(vec(fps::pow(poly(f), e, n)) == pow_by_squaring(f, e, n), n, e, z);
                }
}

// z e around n, and z e past 2^64.
void test_pow_shifts() {
    for (const std::size_t n : {std::size_t(1), std::size_t(10), std::size_t(100), std::size_t(1000)})
        for (const std::size_t z : {std::size_t(1), std::size_t(3), std::size_t(7)})
            for (u64 e = (n - 1) / z; e <= (n - 1) / z + 2; ++e) {
                for (const int kind : {0, 3}) {
                    const Vec f = with_zeros(z, random_vec(n, kind), u32(1 + rng() % (P - 1)));
                    CHECK(vec(fps::pow(poly(f), e, n)) == naive_pow(f, e, n), n, z, e);
                    if (e && n <= 100) CHECK(vec(fps::pow(poly(f), e, n)) == pow_by_squaring(f, e, n), n, z, e);
                }
            }
    const Vec f = with_zeros(2, random_vec(50, 0), 3), zeros(40, 0);
    CHECK(vec(fps::pow(poly(f), u64(1) << 63, 40)) == zeros);
    CHECK(vec(fps::pow(poly(f), ~u64(0), 40)) == zeros);
    CHECK(vec(fps::pow(poly(with_zeros(3, {}, 1)), ~u64(0) / 3 + 1, 40)) == zeros);
    // Monomials c x^k: the recurrence with no taps.
    for (const u64 e : kExponents) {
        Vec g(30);
        if (e == 0) g[0] = 1;
        else if (e <= 9) g[3 * e] = power(5, e);
        CHECK(vec(fps::pow(5 * Poly::x(3), e, 30)) == g, e);
        Vec h(30);
        h[0] = power(7, e);
        CHECK(vec(fps::pow(Poly{7}, e, 30)) == h, e);
    }
    CHECK(vec(fps::pow(Poly{0, 3}, 4, 6)) == (Vec{0, 0, 0, 0, 81, 0}));
    // e = 0: 1, also for f = 0 and empty f (0^0 = 1).
    CHECK(vec(fps::pow(Poly(), 0, 3)) == (Vec{1, 0, 0}));
    CHECK(vec(fps::pow(Poly(5), 0, 3)) == (Vec{1, 0, 0}));
    CHECK(vec(fps::pow(Poly{0, 0, 1}, 0)) == (Vec{1, 0, 0}));
    CHECK(vec(fps::pow(Poly(), 3, 3)) == (Vec{0, 0, 0}));
    CHECK(vec(fps::pow(Poly{0, 0, 1}, 1, 2)) == (Vec{0, 0}));
    CHECK(fps::pow(Poly{1, 2}, 5, 0).size() == 0);
    CHECK(fps::pow(Poly(), 0).size() == 0);
}

// sqrt at n against the root's recurrence: squares c^2 x^(2k) (...) with z leading zeros, plus
// odd z and non-squares (nullopt).
void test_sqrt_small() {
    for (const std::size_t n : kSizes)
        for (const int kind : {0, 1, 2, 3})
            for (const std::size_t z : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(4), std::size_t(7)})
                for (const std::size_t size : {n ? n - 1 : 0, n, n + 2}) {
                    const u32 x = u32(1 + rng() % (P - 1)), c = std::min(x, P - x);
                    const bool square = rng() % 4 != 0;
                    u32 lead = mul(x, x);
                    if (!square)
                        do lead = u32(1 + rng() % (P - 1));
                        while (is_square(lead));
                    const Vec f = size > z ? with_zeros(z, random_vec(size - z, kind), lead) : Vec(size, 0);
                    const bool zero = z >= std::min(size, n);
                    std::optional<Vec> g;
                    if (zero) g = Vec(n, 0);
                    else if (square && z % 2 == 0) g = naive_sqrt(f, n, c);
                    dirty_heap(n);
                    const auto r = fps::sqrt(poly(f), n);
                    CHECK(r.has_value() == g.has_value(), n, z, size);
                    if (r && g) CHECK(vec(*r) == *g, n, z, size);
                    CHECK(raw_sqrt(f, n) == g, n, z, size);
                    if (size == n) {
                        const auto r2 = fps::sqrt(poly(f));
                        CHECK(r2.has_value() == g.has_value() && (!r2 || vec(*r2) == *g), n, z);
                    }
                    // A root squares back: g^2 = f mod x^n.
                    if (g) {
                        Vec fn(f.begin(), f.begin() + std::min(f.size(), n));
                        fn.resize(n);
                        CHECK(naive_mul(*g, *g, n) == fn, n, z);
                    }
                }
    // The root of the polynomial f mod x^n: x sqrt(1 + x) mod x^4 for x^2 (1 + x).
    const u32 i2 = inv(2), i8 = inv(8);
    CHECK(vec(*fps::sqrt(Poly{0, 0, 1, 1}, 4)) == (Vec{0, 1, i2, P - i8}));
    CHECK(vec(*fps::sqrt(Poly{0, 0, 1, 1}, 3)) == (Vec{0, 1, 0}));  // f mod x^3 = x^2
    CHECK(!fps::sqrt(Poly{0, 1}, 3));
    CHECK(!fps::sqrt(Poly{0, 0, 0, 2, 1}, 5));
    CHECK(vec(*fps::sqrt(Poly{0, 0, 0, 2, 1}, 3)) == (Vec{0, 0, 0}));  // f mod x^3 = 0
    CHECK(vec(*fps::sqrt(Poly(), 3)) == (Vec{0, 0, 0}));
    CHECK(fps::sqrt(Poly()).value().size() == 0);
    CHECK(fps::sqrt(Poly{5}, 0).value().size() == 0);
    u32 non_square = 2;
    while (is_square(non_square)) ++non_square;
    CHECK(!fps::sqrt(Poly{non_square, 1}, 3));
    CHECK(!fps::sqrt(Poly{non_square}));
    // The smaller root: 4 has roots 2 and P - 2; P - 1 has two roots, the smaller is taken.
    CHECK(vec(*fps::sqrt(Poly{4}, 3)) == (Vec{2, 0, 0}));
    const u32 i = *fps::detail::mod_sqrt<P>(P - 1);
    CHECK(mul(i, i) == P - 1);
    CHECK(vec(*fps::sqrt(Poly{P - 1}, 2)) == (Vec{i, 0}) && i < P - i);
    CHECK(vec(*fps::sqrt(Poly{0, 0, 4, 4, 1}, 5)) == (Vec{0, 2, 1, 0, 0}));  // x^2 (2 + x)^2
}

// Exactly t terms past u[0] around the paths' bounds (kSparseTerms = 16, pow's kPowTerms = 8,
// sqrt's kRootTerms = 4 up to kRootShort = 256 coefficients of s), near index 0 and spread.
void test_term_counts() {
    namespace d = fps::detail::series;
    static_assert(d::kSparseTerms == 16 && d::kPowTerms == 8 && d::kRootTerms == 4 && d::kRootShort == 256);
    for (const std::size_t n : {17, 40, 256, 257, 258, 1000})
        for (const std::size_t t : {0, 1, 2, 4, 5, 8, 9, 16, 17})
            for (const bool spread : {false, true})
                for (const std::size_t z : {std::size_t(0), std::size_t(2)}) {
                    if (t + z >= n) continue;
                    Vec u(n - z);
                    std::vector<std::size_t> idx;
                    for (std::size_t i = 1; i < u.size(); ++i) idx.push_back(i);
                    if (spread) std::shuffle(idx.begin(), idx.end(), rng);
                    for (std::size_t i = 0; i < t; ++i) u[idx[i]] = u32(1 + rng() % (P - 1));
                    const u64 e = random_exponent();
                    const Vec f = with_zeros(z, u, 9), fs = with_zeros(z, u, 49);
                    CHECK(vec(fps::pow(poly(f), e, n)) == naive_pow(f, e, n), n, t, e);
                    const auto r = fps::sqrt(poly(fs), n);
                    CHECK(r && vec(*r) == naive_sqrt(fs, n, 7), n, t, z);
                }
}

// The bounds kPowLong and kRootLong: below, the backend runs; from them on, the recurrence. So
// a result at the bound, cut by one coefficient, is the other path's; also f^2 = f f, g^2 = f.
void test_long_bounds() {
    namespace d = fps::detail::series;
    for (const std::size_t t : {9, 12, 16})
        for (const std::size_t z : {std::size_t(0), std::size_t(3)}) {
            const u64 e = z ? 1 + rng() % 3 : random_exponent();
            const std::size_t n = d::kPowLong + z * e;  // u's part: kPowLong coefficients
            Vec u(n);
            for (std::size_t i = 1; i <= t; ++i) u[i * (n / 32)] = u32(1 + rng() % (P - 1));
            const Poly f = poly(with_zeros(z, u, u32(1 + rng() % (P - 1))));
            const Poly g = fps::pow(f, e, n);
            CHECK(vec(g.truncate(n - 1)) == vec(fps::pow(f, e, n - 1)), t, z, e);
            CHECK(vec(fps::pow(f, 2, n)) == vec(fps::mul(f, f, n)), t, z);
        }
    // sqrt's bound kRootLong for t <= kRootTerms = 4 (5 terms run the backend on both sides).
    for (const std::size_t t : {1, 4, 5})
        for (const std::size_t z : {std::size_t(0), std::size_t(4)}) {
            const std::size_t n = d::kRootLong + z / 2;
            Vec u(n);
            for (std::size_t i = 1; i <= t; ++i) u[i * (n / 8)] = u32(1 + rng() % (P - 1));
            const Poly f = poly(with_zeros(z, u, 9));
            const auto g = fps::sqrt(f, n), h = fps::sqrt(f, n - 1);
            CHECK(g && h && vec(g->truncate(n - 1)) == vec(*h), t, z);
            CHECK(g && (*g)[z / 2] == 3 && vec(fps::mul(*g, *g, n)) == vec(f.truncate(n)), t, z);
        }
}

// Closed forms at large n: (1 + x)^e = sum C(e, i) x^i (e < P, and e mod P for e >= P with n <= P),
// f^P = f[0] below x^P, sqrt(1 - 4x) and the Catalan numbers (DESIGN.md, Examples).
void test_closed_forms() {
    const Poly x = Poly::x();
    const std::size_t n = 1'000'000;
    const fps::Factorials<P> fact(2 * n + 2);
    for (const u64 e : {u64(5), u64(123456789), u64(P) * 7 + 1000, ~u64(0)}) {
        Vec b(n);
        u32 r = 1;
        for (std::size_t i = 0; i < n; ++i) {
            b[i] = mul(r, fact.inverse(i));
            r = mul(r, sub(u32(e % P), u32(i)));
        }
        CHECK(vec(fps::pow(1 + x, e, n)) == b, e);
    }
    const std::size_t catalan = 100'000;
    const Poly s = *fps::sqrt(1 - 4 * x, catalan + 1);
    // C = (1 - s) / (2 x).
    const Vec sv = vec(s);
    const u32 half = inv(2);
    bool ok = sv[0] == 1;
    for (std::size_t k = 0; k < catalan; ++k) {
        const u32 c = mul(mul(fact[2 * k], fact.inverse(k)), fact.inverse(k + 1));
        ok = ok && mul(sub(0, sv[k + 1]), half) == c;
    }
    CHECK(ok);
    // Dense series: f^P = f[0], f^(P + 1) = f[0] f, f^(2P + 2) = f[0]^2 f^2 below x^P.
    const std::size_t m = (std::size_t(1) << 14) + 1;
    const Poly f = poly(with_zeros(0, random_vec(m, 0), 3));
    Vec c(m);
    c[0] = 3;
    CHECK(vec(fps::pow(f, P)) == c);
    CHECK(vec(fps::pow(f, u64(P) + 1)) == vec(3 * f));
    CHECK(vec(fps::pow(f, 2 * u64(P) + 2)) == vec(9 * fps::mul(f, f, m)));
}

// Identities on the Newton paths at large sizes: pow(f, 2) = f f, pow(f, 3) = f f f,
// pow(pow(f, a), b) = pow(f, a b), sqrt(s^2) = s, with leading zeros.
void test_large() {
    for (const std::size_t n : {(std::size_t(1) << 14) - 1, std::size_t(1) << 14, (std::size_t(1) << 14) + 1,
                                std::size_t(1) << 17, std::size_t(300'001)}) {
        const Poly f = poly(with_zeros(0, random_vec(n, 0), 7));
        CHECK(vec(fps::pow(f, 2)) == vec(fps::mul(f, f, n)), n);
        CHECK(vec(fps::pow(f, 3)) == vec(fps::mul(fps::mul(f, f, n), f, n)), n);
        // f^(a b) depends on a b mod P (f / f[0]) and mod P - 1 (f[0]).
        const u64 a = rng() % 1000 + 2, b = rng(), order = u64(P) * (P - 1);
        const u64 ab = u64((unsigned __int128)a * b % order) + order;
        CHECK(vec(fps::pow(fps::pow(f, a), b)) == vec(fps::pow(f, ab)), n);
        // g = x^5 v: g^2 = x^10 u, and sqrt(g^2) = x^5 s with s = v mod x^(n - 10) (u is known to
        // n - 10 terms; s past that is the root of the polynomial u).
        const Poly g = poly(with_zeros(5, random_vec(n - 5, 1), 2));
        const Poly g2 = fps::mul(g, g, n);
        CHECK(vec(fps::pow(g, 2)) == vec(g2), n);
        const auto r5 = fps::sqrt(g2);
        CHECK(r5 && vec(fps::mul(*r5, *r5, n)) == vec(g2), n);
        CHECK(r5 && vec(r5->truncate(n - 5)) == vec(g.truncate(n - 5)), n);
        const Poly h = poly(with_zeros(0, random_vec(n, 0), u32(P / 2)));
        const auto r = fps::sqrt(fps::mul(h, h, n));
        CHECK(r && vec(*r) == vec(h), n);
    }
    // The recurrences at a large size, against the Newton identities.
    const std::size_t n = 200'000;
    Vec a(n);
    for (int t = 0; t < 16; ++t) a[1 + rng() % (n - 1)] = u32(rng() % P);
    a[0] = 4;
    const Poly f = poly(a);
    CHECK(vec(fps::pow(f, 2)) == vec(fps::mul(f, f, n)));
    const Poly s = *fps::sqrt(f);
    CHECK(vec(fps::mul(s, s, n)) == vec(f));
    CHECK(s[0] == 2);
}

void test_threads() {
    std::vector<std::thread> threads;
    std::vector<int> ok(4, 0);
    std::vector<Poly> inputs;
    for (int t = 0; t < 4; ++t) inputs.push_back(poly(with_zeros(2 * t, random_vec(30'000 + 1000 * t, 0), 1)));
    std::vector<Poly> squares, cubes, roots;
    for (const Poly& f : inputs) {
        squares.push_back(fps::mul(f, f, f.size()));
        cubes.push_back(fps::pow(f, 3));
        roots.push_back(*fps::sqrt(squares.back()));
    }
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&, t] {
            bool good = true;
            for (int r = 0; r < 3; ++r) {
                good = good && vec(fps::pow(inputs[t], 3)) == vec(cubes[t]);
                const auto s = fps::sqrt(squares[t]);
                good = good && s && vec(*s) == vec(roots[t]);
            }
            ok[t] = good;
        });
    for (std::thread& t : threads) t.join();
    for (int t = 0; t < 4; ++t) CHECK(ok[t], t);
}

void test_errors() {
    using std::domain_error;
    Vec buffer = with_zeros(0, random_vec(100, 0), 4);
    const std::span<u32> all(buffer);
    CHECK(throws<domain_error>([&] { fps::raw::pow<P>(all.first(50), 3, all.first(50)); }));
    CHECK(throws<domain_error>([&] { fps::raw::pow<P>(all.first(50), 3, all.subspan(19, 20)); }));
    CHECK(throws<domain_error>([&] { fps::raw::pow<P>(all.subspan(30, 50), 0, all.first(31)); }));
    CHECK(!throws<domain_error>([&] { fps::raw::pow<P>(all.first(50), 3, all.subspan(50, 20)); }));
    // Only f mod x^n is read, so a longer f may overlap past it.
    buffer[0] = 4;
    CHECK(!throws<domain_error>([&] { fps::raw::pow<P>(all.first(80), 3, all.subspan(50, 20)); }));
    buffer[0] = 4;
    CHECK(throws<domain_error>([&] { (void)fps::raw::sqrt<P>(all.first(50), all.first(50)); }));
    CHECK(throws<domain_error>([&] { (void)fps::raw::sqrt<P>(all.first(50), all.subspan(10, 20)); }));
    CHECK(!throws<domain_error>([&] { (void)fps::raw::sqrt<P>(all.first(50), all.subspan(50, 20)); }));
    // n > P for pow: 4 GB of unwritten memory, never touched.
    {
        const std::size_t big = std::size_t(P) + 1;
        const auto words = std::make_unique_for_overwrite<u32[]>(big);
        const std::span<u32> out(words.get(), big);
        CHECK(throws<domain_error>([&] { fps::raw::pow<P>(Vec{1, 1}, 2, out); }));
        CHECK(throws<domain_error>([&] { fps::raw::pow<P>(Vec{}, 0, out); }));
    }
}

// Past the backend's 2^25 coefficients: a dense f throws std::length_error before out is written
// (the zeros of x^(z e) and x^k too); the recurrence has no such limit (not under ASan: 2^25 steps).
void test_limits() {
    const std::size_t big = (std::size_t(1) << 25) + 1;
    Vec f(big + 2, P - 1), out(big + 2, 5);
    f[0] = 4;
    const std::span<const u32> dense(f);
    CHECK(throws<std::length_error>([&] { fps::raw::pow<P>(dense, 3, std::span<u32>(out).first(big)); }));
    CHECK(throws<std::length_error>([&] { (void)fps::raw::sqrt<P>(dense, std::span<u32>(out).first(big)); }));
    f[0] = 0, f[1] = 0, f[2] = 4;  // x^2 u: the result's u part starts at 2 (pow, e = 1) or 1 (sqrt)
    CHECK(throws<std::length_error>([&] { fps::raw::pow<P>(dense, 1, std::span<u32>(out)); }));
    CHECK(throws<std::length_error>([&] { (void)fps::raw::sqrt<P>(dense, std::span<u32>(out)); }));
    CHECK(std::ranges::all_of(out, [](u32 c) { return c == 5; }));
    if (kSanitized) return;
    // (1 - x)^(P - 1) = (1 - x^P) / (1 - x): all ones below x^P. sqrt((1 - x)^2) = 1 - x.
    const Poly x = Poly::x();
    const Poly g = fps::pow(1 - x, P - 1, big);
    CHECK(g.size() == big && std::ranges::all_of(g.span(), [](u32 c) { return c == 1; }));
    const auto s = fps::sqrt(1 - 2 * x + x * x, big);
    Vec root(big, 0);
    root[0] = 1, root[1] = P - 1;
    CHECK(s && vec(*s) == root);
}

}  // namespace

int main() {
    test_mod_sqrt();
    test_pow_small();
    test_pow_shifts();
    test_sqrt_small();
    test_term_counts();
    test_long_bounds();
    test_closed_forms();
    test_large();
    test_threads();
    test_errors();
    test_limits();
    if (failures) {
        std::printf("pow_sqrt: %d failures\n", failures);
        return 1;
    }
    std::printf("pow_sqrt: all passed\n");
    return 0;
}
