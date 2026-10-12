// Tests of fps/points.hpp: evaluate against Horner's rule, interpolate against Lagrange's formula
// (O(m^2)), to_newton_basis against expanding the basis back, on every pair of edge sizes, random,
// repeated and zero points; large sizes at random points; threads and every error.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "fps/fps.hpp"

// Under ASan, every allocation starts filled with 0xbe bytes (not only its first 4 KiB), so a
// result word the library forgets to write shows up on any allocator.
extern "C" const char* __asan_default_options() { return "max_malloc_fill_size=2147483647"; }

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = 998244353;
using Poly = fps::Polynomial<P>;

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

std::mt19937_64 rng(24);

// 0, 1, powers of two and their neighbours (the leaves hold 32 points).
const std::vector<std::size_t> kSizes = {0,  1,  2,  3,   4,   5,   7,   8,   9,   15,  16,  17,  31,  32,  33,
                                         63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025};

u32 add(u32 a, u32 b) { return a + b >= P ? a + b - P : a + b; }
u32 sub(u32 a, u32 b) { return a >= b ? a - b : a + P - b; }
u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }

u32 inv(u32 a) {
    u32 r = 1;
    for (u64 e = P - 2; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}

std::size_t pick(std::size_t n) { return std::size_t(rng() % n); }

// Kinds: 0 random, 1 all P - 1, 2 mostly zeros.
Vec random_vec(std::size_t n, int kind = 0) {
    Vec a(n);
    for (u32& c : a) {
        const u64 r = rng();
        c = kind == 1 ? P - 1 : kind == 2 && r % 4 ? 0 : u32(r % P);
    }
    return a;
}

// m distinct points: random, or (kind 1) 0 .. m - 1 and P - 1, P - 2, ... shuffled.
Vec distinct_points(std::size_t m, int kind = 0) {
    Vec x;
    if (kind == 1) {
        for (std::size_t i = 0; i < m; ++i) x.push_back(i % 2 ? u32(i / 2) : P - 1 - u32(i / 2));
    } else {
        while (x.size() < m) {
            for (std::size_t i = x.size(); i < m; ++i) x.push_back(u32(rng() % P));
            std::ranges::sort(x);
            x.erase(std::unique(x.begin(), x.end()), x.end());
        }
    }
    std::ranges::shuffle(x, rng);
    return x;
}

// Points with repeats: drawn from a few values (kind 0), or all zero (kind 1).
Vec repeated_points(std::size_t m, int kind) {
    const Vec pool = random_vec(3);
    Vec x(m);
    for (u32& v : x) v = kind == 1 ? 0 : pool[pick(3)];
    return x;
}

Vec vec(const Poly& f) { return Vec(f.span().begin(), f.span().end()); }
Poly poly(const Vec& a) { return Poly(std::span<const u32>(a)); }

u32 horner(const Vec& f, u32 x) {
    u32 r = 0;
    for (std::size_t i = f.size(); i-- > 0;) r = add(mul(r, x), f[i]);
    return r;
}

Vec horner_all(const Vec& f, const Vec& x) {
    Vec v(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) v[i] = horner(f, x[i]);
    return v;
}

// a (x - c) in place, a one longer.
void times_linear(Vec& a, u32 c) {
    a.push_back(0);
    for (std::size_t j = a.size() - 1; j > 0; --j) a[j] = sub(a[j - 1], mul(c, a[j]));
    a[0] = sub(0, mul(c, a[0]));
}

// Lagrange: M = prod (x - x_i); f = sum y_i (M / (x - x_i)) / (M / (x - x_i))(x_i). O(m^2).
Vec lagrange(const Vec& x, const Vec& y) {
    const std::size_t m = x.size();
    Vec M{1}, f(m);
    for (u32 c : x) times_linear(M, c);
    for (std::size_t i = 0; i < m; ++i) {
        Vec q(m);
        u32 carry = 0;
        for (std::size_t j = m; j-- > 0;) q[j] = carry = add(M[j + 1], mul(x[i], carry));
        const u32 scale = mul(y[i], inv(horner(q, x[i])));
        for (std::size_t j = 0; j < m; ++j) f[j] = add(f[j], mul(scale, q[j]));
    }
    return f;
}

// sum c_k prod_(i < k) (x - x_i).
Vec expand_newton(const Vec& c, const Vec& x) {
    Vec f(c.size()), basis{1};
    for (std::size_t k = 0; k < c.size(); ++k) {
        for (std::size_t i = 0; i < basis.size(); ++i) f[i] = add(f[i], mul(c[k], basis[i]));
        if (k + 1 < c.size()) times_linear(basis, x[k]);
    }
    return f;
}

// sum c_k prod_(i < k) (a - x_i), O(n).
u32 newton_at(const Vec& c, const Vec& x, u32 a) {
    u32 sum = 0, basis = 1;
    for (std::size_t k = 0; k < c.size(); ++k) {
        sum = add(sum, mul(c[k], basis));
        if (k + 1 < c.size()) basis = mul(basis, sub(a, x[k]));
    }
    return sum;
}

void test_evaluate() {
    for (const std::size_t n : kSizes)
        for (const std::size_t m : kSizes) {
            const int kind = int((n + m) % 3);
            const Vec f = random_vec(n, kind);
            const Vec x = m % 4 == 3 ? repeated_points(m, 0) : random_vec(m, kind == 1 ? 0 : kind);
            CHECK(fps::evaluate(poly(f), x) == horner_all(f, x), n, m, kind);
        }
    // Zero points (nodes x^k: the monomial division), P - 1, and both at once.
    for (const std::size_t m : {1, 32, 33, 100, 1000}) {
        const Vec f = random_vec(1500);
        Vec x = repeated_points(m, 1);
        CHECK(fps::evaluate(poly(f), x) == horner_all(f, x), m);
        for (std::size_t i = 0; i < m; i += 2) x[i] = P - 1;
        CHECK(fps::evaluate(poly(f), x) == horner_all(f, x), m);
    }
    // Long f by few points, few coefficients by many points, and long both: at random points.
    const auto spot = [](const Vec& f, const Vec& x, const Vec& got, std::size_t count) {
        bool ok = got.size() == x.size();
        for (std::size_t t = 0; t < count && ok; ++t) {
            const std::size_t i = pick(x.size());
            ok = got[i] == horner(f, x[i]);
        }
        return ok;
    };
    for (const auto [n, m] : {std::pair<std::size_t, std::size_t>{(1 << 17) + 3, 5}, {(1 << 17) + 3, 33},
                              {3, (1 << 16) + 1}, {40, 1 << 16}, {1 << 15, (1 << 15) - 1}, {(1 << 16) + 1, 1 << 16},
                              {(1 << 16) - 1, (1 << 17) + 1}}) {
        const Vec f = random_vec(n), x = random_vec(m);
        CHECK(spot(f, x, fps::evaluate(poly(f), x), 300), n, m);
    }
}

void test_interpolate() {
    for (const std::size_t m : kSizes)
        for (int kind = 0; kind < 2; ++kind) {
            const Vec x = distinct_points(m, kind), y = random_vec(m, m % 3);
            const Poly f = fps::interpolate<P>(x, y);
            CHECK(vec(f) == lagrange(x, y), m, kind);
        }
    // Large: back through evaluate, and at random points by Horner.
    for (const std::size_t m : {std::size_t(1) << 14, (std::size_t(1) << 17) + 1}) {
        const Vec x = distinct_points(m), y = random_vec(m);
        const Vec f = vec(fps::interpolate<P>(x, y));
        CHECK(f.size() == m && fps::evaluate(poly(f), x) == y, m);
        for (int t = 0; t < 100; ++t) {
            const std::size_t i = pick(m);
            CHECK(horner(f, x[i]) == y[i], m, i);
        }
    }
    // A polynomial of degree < m comes back exactly, and with its trailing zeros.
    const Vec g = random_vec(700);
    const Vec x = distinct_points(1000);
    Vec want = g;
    want.resize(1000);
    CHECK(vec(fps::interpolate<P>(x, fps::evaluate(poly(g), x))) == want);
}

void test_newton() {
    for (const std::size_t n : kSizes)
        for (int kind = 0; kind < 3; ++kind) {
            // kind 0: random points; 1: repeats; 2: all zero. One spare point past n - 1 for odd n.
            const std::size_t count = n == 0 ? 0 : n - 1 + n % 2;
            const Vec x = kind ? repeated_points(count, kind - 1) : random_vec(count);
            const Vec f = random_vec(n, kind);
            const Poly c = fps::to_newton_basis(poly(f), x);
            CHECK(c.size() == n && expand_newton(vec(c), x) == f, n, kind);
        }
    for (const std::size_t n : {(std::size_t(1) << 15) + 1, std::size_t(100000)}) {
        const Vec f = random_vec(n), x = random_vec(n - 1);
        const Vec c = vec(fps::to_newton_basis(poly(f), x));
        CHECK(c.size() == n, n);
        for (int t = 0; t < 20; ++t) {
            const u32 a = t < 10 ? x[pick(n - 1)] : u32(rng() % P);
            CHECK(newton_at(c, x, a) == horner(f, a), n, t);
        }
    }
}

void test_threads() {
    std::vector<std::thread> threads;
    int bad[4] = {};
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([t, &bad] {
            std::mt19937_64 r(100 + t);
            for (int round = 0; round < 3; ++round) {
                const std::size_t m = 3000 + 1000 * std::size_t(t);
                Vec x(m), y(m);
                for (std::size_t i = 0; i < m; ++i) x[i] = u32(i * 7919 + t), y[i] = u32(r() % P);
                const Poly f = fps::interpolate<P>(x, y);
                bad[t] += fps::evaluate(f, x) != y;
                const Vec c = vec(fps::to_newton_basis(f, x));
                bad[t] += newton_at(c, x, 12345) != f(12345);
            }
        });
    for (auto& th : threads) th.join();
    for (int t = 0; t < 4; ++t) CHECK(bad[t] == 0, t);
}

void test_errors() {
    const Poly f = poly(random_vec(10));
    CHECK(throws<std::domain_error>([&] { (void)fps::evaluate(f, Vec{1, P, 2}); }));
    CHECK(throws<std::domain_error>([&] { (void)fps::evaluate(f, Vec{u32(-1)}); }));
    CHECK(throws<std::domain_error>([] { (void)fps::interpolate<P>(Vec{1, 2}, Vec{3}); }));
    CHECK(throws<std::domain_error>([] { (void)fps::interpolate<P>(Vec{}, Vec{3}); }));
    CHECK(throws<std::domain_error>([] { (void)fps::interpolate<P>(Vec{1, P}, Vec{3, 4}); }));
    CHECK(throws<std::domain_error>([] { (void)fps::interpolate<P>(Vec{1, 2}, Vec{3, P}); }));
    // Repeats: next to each other, in one leaf, in different leaves, and all equal.
    for (const auto [m, i, j] : {std::tuple<std::size_t, std::size_t, std::size_t>{2, 0, 1}, {32, 3, 30},
                                 {1000, 5, 6}, {1000, 0, 999}, {5000, 17, 4000}}) {
        Vec x = distinct_points(m);
        x[j] = x[i];
        CHECK(throws<std::domain_error>([&] { (void)fps::interpolate<P>(x, random_vec(m)); }), m, i, j);
    }
    CHECK(throws<std::domain_error>([] { (void)fps::interpolate<P>(Vec(100, 7), random_vec(100)); }));
    CHECK(throws<std::domain_error>([&] { (void)fps::to_newton_basis(f, Vec(8)); }));
    CHECK(throws<std::domain_error>([] { (void)fps::to_newton_basis(Poly{1, 2}, Vec{}); }));
    CHECK(throws<std::domain_error>([&] { (void)fps::to_newton_basis(f, Vec{1, 2, 3, 4, 5, 6, 7, 8, P}); }));
    CHECK(fps::to_newton_basis(f, Vec(9)).size() == 10);
    CHECK(fps::to_newton_basis(Poly{5}, Vec{}).size() == 1 && fps::to_newton_basis(Poly{5}, Vec{})[0] == 5);
    CHECK(fps::to_newton_basis(Poly(), Vec{}).size() == 0);
    CHECK(fps::interpolate<P>(Vec{}, Vec{}).size() == 0);
    CHECK(fps::evaluate(Poly(), Vec{1, 2}) == (Vec{0, 0}) && fps::evaluate(f, Vec{}).empty());
    // 2^25 points: the root's product would have 2^25 + 1 coefficients.
    const Vec x(std::size_t(1) << 25, 3);
    CHECK(throws<std::length_error>([&] { (void)fps::evaluate(f, x); }));
}

}  // namespace

int main() {
    test_evaluate();
    test_interpolate();
    test_newton();
    test_threads();
    test_errors();
    if (failures) {
        std::printf("points: %d failures\n", failures);
        return 1;
    }
    std::printf("points: all passed\n");
    return 0;
}
