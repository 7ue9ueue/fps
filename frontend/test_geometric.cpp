// Tests of fps/geometric.hpp: evaluate on geometric points against Horner's rule at each point,
// interpolate by evaluating back (the f of degree < m through m points is unique), extrapolate
// against Horner on the polynomial that made the samples, dft against its sum and idft back,
// raw::dft and raw::idft against them (into other memory and over their input); on
// pairs of edge sizes, special a, r and c, roots of unity, large sizes; threads and every error.
#include <sys/mman.h>

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

std::mt19937_64 rng(27);

// 0, 1, powers of two and their neighbours (the kernels switch paths at n m = 256 and n = 32).
const std::vector<std::size_t> kSizes = {0,  1,  2,  3,   4,   5,   7,   8,   9,   15,  16,  17,  31,  32,  33,
                                         63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025};

u32 add(u32 a, u32 b) { return a + b >= P ? a + b - P : a + b; }
u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }

u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}

u32 random_word() { return u32(rng() % P); }

// Kinds: 0 random, 1 all P - 1, 2 mostly zeros.
Vec random_vec(std::size_t n, int kind = 0) {
    Vec a(n);
    for (u32& c : a) {
        const u64 r = rng();
        c = kind == 1 ? P - 1 : kind == 2 && r % 4 ? 0 : u32(r % P);
    }
    return a;
}

Vec vec(const Poly& f) { return Vec(f.span().begin(), f.span().end()); }
Poly poly(const Vec& a) { return Poly(std::span<const u32>(a)); }

u32 horner(const Vec& f, u32 x) {
    u32 r = 0;
    for (std::size_t i = f.size(); i-- > 0;) r = add(mul(r, x), f[i]);
    return r;
}

Vec geometric_points(u32 a, u32 r, std::size_t m) {
    Vec x(m);
    for (std::size_t i = 0; i < m; ++i, a = mul(a, r)) x[i] = a;
    return x;
}

Vec horner_all(const Vec& f, const Vec& x) {
    Vec v(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) v[i] = horner(f, x[i]);
    return v;
}

// The multiplicative order of r != 0.
u64 order(u32 r) {
    u64 k = P - 1;
    for (u64 q : {2, 7, 17})
        while (k % q == 0 && power(r, k / q) == 1) k /= q;
    return k;
}

// A primitive k-th root of unity, k | P - 1.
u32 root_of_unity(u64 k) { return power(3, (P - 1) / k); }

// (a, r) pairs: random, zeros, ones, P - 1, small orders.
std::pair<u32, u32> pick_ar(std::size_t i) {
    switch (i % 9) {
        case 0: return {0, random_word()};
        case 1: return {random_word(), 0};
        case 2: return {random_word(), 1};
        case 3: return {1, P - 1};
        case 4: return {random_word(), root_of_unity(7)};
        case 5: return {P - 1, random_word()};
        case 6: return {0, 0};
        default: return {random_word(), random_word()};
    }
}

void test_evaluate() {
    std::size_t pair = 0;
    for (std::size_t n : kSizes)
        for (std::size_t m : kSizes) {
            const auto [a, r] = pick_ar(pair++);
            const Vec f = random_vec(n, int(pair % 3));
            CHECK(fps::evaluate(poly(f), fps::geometric(a, r, m)) == horner_all(f, geometric_points(a, r, m)), n, m,
                  pair);
        }
    // Long f by few points and back; 0^0 = 1 at a = 0.
    for (std::size_t n : {3000, 40000})
        for (std::size_t m : {2, 5, 33, 300}) {
            const Vec f = random_vec(n);
            const u32 a = random_word(), r = random_word();
            CHECK(fps::evaluate(poly(f), fps::geometric(a, r, m)) == horner_all(f, geometric_points(a, r, m)), n, m);
            CHECK(fps::evaluate(poly(Vec(f.begin(), f.begin() + std::ptrdiff_t(m))), fps::geometric(a, r, n)) ==
                      horner_all(Vec(f.begin(), f.begin() + std::ptrdiff_t(m)), geometric_points(a, r, n)),
                  n, m);
        }
    CHECK(fps::evaluate(Poly{5, 6, 7}, fps::geometric(0, 9, 3)) == (Vec{5, 5, 5}));
    CHECK(fps::evaluate(Poly{5, 6, 7}, fps::geometric(2, 0, 3)) == (Vec{5 + 12 + 28, 5, 5}));
}

void test_interpolate() {
    std::size_t pair = 0;
    for (std::size_t m : kSizes)
        for (int kind = 0; kind < 3; ++kind) {
            u32 a = random_word(), r = random_word();
            if (kind == 1) r = root_of_unity(1 << 11);  // order 2048 >= m: distinct
            if (kind == 2) a = P - 1, r = root_of_unity(7 * 17 * 16);
            if (r != 0 && order(r) < m) continue;
            const Vec y = random_vec(m, int(pair++ % 3));
            const Poly f = fps::interpolate<P>(fps::geometric(a, r, m), y);
            CHECK(f.size() == m && horner_all(vec(f), geometric_points(a, r, m)) == y, m, kind);
        }
    // Edge cases of distinct points: one point (any a), two with r = 0, roots of unity of order m.
    CHECK(vec(fps::interpolate<P>(fps::geometric(0, 0, 1), Vec{7})) == Vec{7});
    CHECK(vec(fps::interpolate<P>(fps::geometric(0, 5, 1), Vec{7})) == Vec{7});
    const Vec two{11, 13};
    CHECK(horner_all(vec(fps::interpolate<P>(fps::geometric(3, 0, 2), two)), Vec{3, 0}) == two);
    CHECK(horner_all(vec(fps::interpolate<P>(fps::geometric(3, P - 1, 2), two)), Vec{3, P - 3}) == two);
    for (u64 k : {2, 7, 16, 17, 32, 34, 64, 119, 256, 1024}) {
        const u32 w = root_of_unity(k), a = random_word();
        const Vec y = random_vec(k);
        const Poly f = fps::interpolate<P>(fps::geometric(a, w, k), y);
        CHECK(f.size() == k && horner_all(vec(f), geometric_points(a, w, k)) == y, k);
        CHECK(throws<std::domain_error>([&] { (void)fps::interpolate<P>(fps::geometric(a, w, k + 1), random_vec(k + 1)); }), k);
        CHECK(throws<std::domain_error>([&] { (void)fps::interpolate<P>(fps::geometric(a, w, 2 * k + 5), random_vec(2 * k + 5)); }), k);
    }
    for (std::size_t m : {2, 3, 33, 100}) {
        CHECK(throws<std::domain_error>([&] { (void)fps::interpolate<P>(fps::geometric(0, 5, m), random_vec(m)); }), m);
        CHECK(throws<std::domain_error>([&] { (void)fps::interpolate<P>(fps::geometric(5, 1, m), random_vec(m)); }), m);
        if (m > 2)
            CHECK(throws<std::domain_error>([&] { (void)fps::interpolate<P>(fps::geometric(5, 0, m), random_vec(m)); }), m);
    }
    // Large: back through evaluate, and a polynomial of degree < m back exactly.
    for (std::size_t m : {std::size_t(1) << 14, (std::size_t(1) << 14) + 1, std::size_t(100003)}) {
        const u32 a = random_word(), r = random_word();
        const Vec y = random_vec(m);
        const Poly f = fps::interpolate<P>(fps::geometric(a, r, m), y);
        CHECK(f.size() == m && fps::evaluate(f, fps::geometric(a, r, m)) == y, m);
        const Vec g = random_vec(m);
        CHECK(vec(fps::interpolate<P>(fps::geometric(a, r, m), fps::evaluate(poly(g), fps::geometric(a, r, m)))) == g, m);
        for (int t = 0; t < 5; ++t) {
            const std::size_t i = rng() % m;
            CHECK(horner(vec(f), mul(a, power(r, i))) == y[i], m, i);
        }
    }
}

void test_extrapolate() {
    std::size_t pair = 0;
    for (std::size_t n : kSizes)
        for (std::size_t m : {0, 1, 2, 17, 64, 257, 300}) {
            if (n > 300 && (m == 64 || m == 257)) continue;  // the Horner reference is O(n m)
            const Vec f = random_vec(n, int(pair++ % 3));  // degree < n
            Vec values(n);
            for (std::size_t j = 0; j < n; ++j) values[j] = horner(f, u32(j));
            for (u32 c : {u32(0), u32(1), u32(n / 2), u32(n), random_word(), P - 1, u32((P - n / 2) % P), u32((P - m / 2) % P)}) {
                Vec want(m);
                for (std::size_t i = 0; i < m; ++i) want[i] = horner(f, u32((c + i) % P));
                CHECK(fps::extrapolate<P>(values, c, m) == want, n, m, c);
            }
        }
    // Large: random points of the result against Horner.
    for (std::size_t n : {std::size_t(1) << 14, std::size_t(20001)}) {
        const Vec f = random_vec(n);
        Vec x(n);
        for (std::size_t j = 0; j < n; ++j) x[j] = u32(j);
        const Vec values = fps::evaluate(poly(f), x);  // fps/points.hpp, tested in test_points.cpp
        for (u32 c : {random_word(), u32(P - n / 3)}) {
            const std::size_t m = n + 5;
            const Vec got = fps::extrapolate<P>(values, c, m);
            CHECK(got.size() == m);
            for (int t = 0; t < 8; ++t) {
                const std::size_t i = rng() % m;
                CHECK(got[i] == horner(f, u32((c + i) % P)), n, c, i);
            }
        }
    }
}

void test_dft() {
    // 3 generates the units: w = 3^((P - 1) / L) has order exactly L.
    CHECK(order(3) == P - 1);
    // The table of roots, for every divisor L of P - 1.
    for (u64 L = 1; L <= P - 1; L <<= 1)
        for (u64 odd : {1, 7, 17, 119})
            if ((P - 1) % (L * odd) == 0) {
                const u32 w = fps::detail::geometric::root<P>(L * odd, false, "");
                CHECK(w == root_of_unity(L * odd) && order(w) == L * odd, L * odd);
                CHECK(mul(w, fps::detail::geometric::root<P>(L * odd, true, "")) == 1, L * odd);
            }
    const std::vector<std::size_t> lengths = {1, 2, 4, 7, 8, 14, 16, 17, 32, 34, 64, 119, 128, 238, 256, 1024, 1904};
    for (std::size_t L : lengths) {
        const u32 w = root_of_unity(L);
        CHECK(order(w) == L, L);
        for (std::size_t n : {std::size_t(0), std::size_t(1), L - 1, L, L + 1, 2 * L + 3}) {
            const Vec f = random_vec(n, int(n % 3));
            const Vec got = fps::dft(poly(f), L);
            CHECK(got == horner_all(f, geometric_points(1, w, L)), L, n);
            // idft gives f mod (x^L - 1).
            Vec folded(L);
            for (std::size_t i = 0; i < n; ++i) folded[i % L] = add(folded[i % L], f[i]);
            CHECK(vec(fps::idft<P>(got)) == folded, L, n);
        }
        const Vec y = random_vec(L);
        const Poly g = fps::idft<P>(y);
        CHECK(g.size() == L && fps::dft(g, L) == y, L);
    }
    // The documented order at L = 4: f(1), f(w), f(w^2) = f(-1), f(w^3).
    const u32 w4 = power(3, (P - 1) / 4);
    CHECK(fps::dft(Poly{1, 2, 3}, 4) == (Vec{6, horner({1, 2, 3}, w4), 2, horner({1, 2, 3}, mul(w4, mul(w4, w4)))}));
    CHECK(fps::dft(Poly{1, 2, 3}, 0).empty() && fps::idft<P>(Vec{}).size() == 0);
    // Large: 2^17, 119 2^10 and 2^23 (the longest power of two), round trips and a few values.
    for (std::size_t L : {std::size_t(1) << 17, std::size_t(119) << 10, std::size_t(1) << 23}) {
        const Vec f = random_vec(L);
        const Vec v = fps::dft(poly(f), L);
        CHECK(vec(fps::idft<P>(v)) == f, L);
        const u32 w = root_of_unity(L);
        for (int t = 0; t < 5; ++t) {
            const std::size_t i = rng() % L;
            CHECK(v[i] == horner(f, power(w, i)), L, i);
        }
    }
}

// raw::dft and raw::idft against fps::dft and fps::idft (tested above): into other memory (filled
// with garbage), in place, and into out shifted by one word over the input; the transform's lengths
// and the chirp's.
void test_raw_dft() {
    const std::vector<std::size_t> lengths = {0, 1, 2, 7, 8, 17, 32, 34, 56, 64, 119, 128, 1024, 1904, 4096, std::size_t(1) << 17};
    for (std::size_t L : lengths)
        for (std::size_t n : {std::size_t(0), std::size_t(1), L / 2, L - (L > 0), L, L + 1, 2 * L + 3}) {
            if (L == std::size_t(1) << 17 && n != L) continue;
            const Vec f = random_vec(n, int(n % 3));
            const Vec want = fps::dft(poly(f), L);
            Vec out(L, ~u32(0));
            fps::raw::dft<P>(f, out);
            CHECK(out == want, L, n);
            // In one buffer of max(n, L) + 1 words: the input at word `in`, out at word `at`.
            const std::pair<std::size_t, std::size_t> places[] = {{0, 0}, {0, 1}, {1, 0}};
            for (auto [in, at] : places) {
                Vec buffer(std::max(n, L) + 1, ~u32(0));
                std::ranges::copy(f, buffer.begin() + std::ptrdiff_t(in));
                fps::raw::dft<P>(std::span<const u32>(buffer).subspan(in, n), std::span(buffer).subspan(at, L));
                CHECK(Vec(buffer.begin() + std::ptrdiff_t(at), buffer.begin() + std::ptrdiff_t(at + L)) == want, L, n, in * 2 + at);
            }
            if (n != L) continue;
            const Vec v = random_vec(L, int(L % 3));
            const Vec back = vec(fps::idft<P>(v));
            Vec coefficients(L, ~u32(0));
            fps::raw::idft<P>(v, coefficients);
            CHECK(coefficients == back, L);
            for (auto [in, at] : places) {
                Vec buffer(L + 1, ~u32(0));
                std::ranges::copy(v, buffer.begin() + std::ptrdiff_t(in));
                fps::raw::idft<P>(std::span<const u32>(buffer).subspan(in, L), std::span(buffer).subspan(at, L));
                CHECK(Vec(buffer.begin() + std::ptrdiff_t(at), buffer.begin() + std::ptrdiff_t(at + L)) == back, L, in * 2 + at);
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
                Vec y(m);
                for (u32& v : y) v = u32(r() % P);
                const u32 a = u32(r() % (P - 1)) + 1, q = u32(r() % (P - 1)) + 1;
                const Poly f = fps::interpolate<P>(fps::geometric(a, q, m), y);
                bad[t] += fps::evaluate(f, fps::geometric(a, q, m)) != y;
                Vec padded = vec(f);
                padded.resize(8192);
                bad[t] += vec(fps::idft<P>(fps::dft(f, 8192))) != padded;
                const Vec e = fps::extrapolate<P>(y, u32(m), m);
                bad[t] += fps::extrapolate<P>(e, u32(P - m), 2 * m) != [&] {
                    Vec both(e);
                    both.insert(both.begin(), y.begin(), y.end());
                    return both;
                }();
            }
        });
    for (std::thread& th : threads) th.join();
    for (int t = 0; t < 4; ++t) CHECK(bad[t] == 0, t);
}

void test_errors() {
    using std::domain_error;
    const Poly f{1, 2, 3};
    CHECK(throws<domain_error>([&] { (void)fps::evaluate(f, fps::geometric(P, 2, 3)); }));
    CHECK(throws<domain_error>([&] { (void)fps::evaluate(f, fps::geometric(2, P, 3)); }));
    CHECK(throws<domain_error>([&] { (void)fps::evaluate(f, fps::geometric(~u32(0), 2, 0)); }));
    CHECK(!throws<domain_error>([&] { (void)fps::evaluate(f, fps::geometric(P - 1, P - 1, 3)); }));
    CHECK(throws<domain_error>([] { (void)fps::interpolate<P>(fps::geometric(P, 2, 2), Vec{1, 2}); }));
    CHECK(throws<domain_error>([] { (void)fps::interpolate<P>(fps::geometric(2, P, 2), Vec{1, 2}); }));
    CHECK(throws<domain_error>([] { (void)fps::interpolate<P>(fps::geometric(2, 3, 2), Vec{1, P}); }));
    CHECK(throws<domain_error>([] { (void)fps::interpolate<P>(fps::geometric(2, 3, 2), Vec{1}); }));
    CHECK(throws<domain_error>([] { (void)fps::interpolate<P>(fps::geometric(2, 3, 2), Vec{1, 2, 3}); }));
    CHECK(throws<domain_error>([] { (void)fps::interpolate<P>(fps::geometric(2, 3, 0), Vec{1}); }));
    CHECK(fps::interpolate<P>(fps::geometric(2, 3, 0), Vec{}).size() == 0);
    CHECK(throws<domain_error>([] { (void)fps::extrapolate<P>(Vec{1, 2}, P, 3); }));
    CHECK(throws<domain_error>([] { (void)fps::extrapolate<P>(Vec{1, P}, 5, 3); }));
    CHECK(fps::extrapolate<P>(Vec{}, 5, 3) == Vec(3, 0) && fps::extrapolate<P>(Vec{4}, 5, 2) == (Vec{4, 4}));
    CHECK(fps::extrapolate<P>(Vec{1, 2}, 5, 0).empty());
    for (std::size_t L : {3, 5, 9, 239, 1 << 24, 1 << 25}) {
        CHECK(throws<domain_error>([&] { (void)fps::dft(f, L); }), L);
        if (L < 1000) CHECK(throws<domain_error>([&] { (void)fps::idft<P>(Vec(L)); }), L);
    }
    CHECK(throws<domain_error>([] { (void)fps::idft<P>(Vec{1, P}); }));
    // One word >= P anywhere: each vector chain and the tail of the check.
    for (std::size_t n : {1, 7, 8, 16, 63, 64, 65, 100}) {
        Vec v = random_vec(n, 1);  // all P - 1: canonical
        CHECK(!throws<domain_error>([&] { (void)fps::extrapolate<P>(v, 0, 1); }), n);
        for (std::size_t i = 0; i < n; ++i)
            for (u32 bad : {P, P + 1, ~u32(0)}) {
                v[i] = bad;
                CHECK(throws<domain_error>([&] { (void)fps::extrapolate<P>(v, 0, 1); }), n, i, bad);
                CHECK(throws<domain_error>([&] { (void)fps::interpolate<P>(fps::geometric(2, 3, n), v); }), n, i, bad);
                v[i] = P - 1;
            }
    }
    {
        Vec v = random_vec(1024);
        v[rng() % 1024] = P;
        CHECK(throws<domain_error>([&] { (void)fps::idft<P>(v); }));
    }
    // n > P values, 2^25 + 1 geometric points: unwritten memory, never read.
    {
        const std::size_t words = std::size_t(P) + 1, bytes = words * sizeof(u32);
        void* map = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        CHECK(map != MAP_FAILED);
        if (map != MAP_FAILED) {
            const std::span<const u32> big(static_cast<const u32*>(map), words);
            CHECK(throws<domain_error>([&] { (void)fps::extrapolate<P>(big, 0, 1); }));
            const std::size_t m = (std::size_t(1) << 25) + 1;
            CHECK(throws<std::length_error>([&] { (void)fps::interpolate<P>(fps::geometric(2, 3, m), big.first(m)); }));
            // idft: 2^24 and 2^25 have no roots, 7 2^22 divides P - 1 but needs 2L - 1 > 2^25 words.
            for (std::size_t L : {std::size_t(1) << 24, std::size_t(1) << 25})
                CHECK(throws<domain_error>([&] { (void)fps::idft<P>(big.first(L)); }), L);
            CHECK(throws<std::length_error>([&] { (void)fps::idft<P>(big.first(std::size_t(7) << 22)); }));
            munmap(map, bytes);
        }
    }
    // The middle product past 2^25 words: 2 coefficients at 2^25 points.
    CHECK(throws<std::length_error>([] { (void)fps::evaluate(Poly{1, 2}, fps::geometric(2, 3, std::size_t(1) << 25)); }));
    // dft: divisors of P - 1 past the middle product's 2^25 words, checked before the L-word result.
    CHECK(throws<std::length_error>([] { (void)fps::dft(Poly{1, 2}, std::size_t(17) << 21); }));
    CHECK(throws<std::length_error>([] { (void)fps::dft(Poly((std::size_t(1) << 22) + 2), std::size_t(7) << 22); }));
    CHECK(fps::dft(Poly{5}, std::size_t(17) << 21).size() == std::size_t(17) << 21);  // one coefficient: no product

    // raw::dft and raw::idft: the same errors, and out of the wrong size; nothing written.
    for (std::size_t L : {3, 5, 9, 239}) {
        Vec out(L, 7), values(L, 1);
        CHECK(throws<domain_error>([&] { fps::raw::dft<P>(f.span(), out); }) && out == Vec(L, 7), L);
        CHECK(throws<domain_error>([&] { fps::raw::idft<P>(values, out); }) && out == Vec(L, 7), L);
    }
    for (std::size_t L : {0, 1, 8, 64, 1024}) {
        Vec values(L, 1), shorter(L + 1, 7), longer(L + 1, 7);
        CHECK(throws<domain_error>([&] { fps::raw::idft<P>(values, std::span(longer)); }) && longer == Vec(L + 1, 7), L);
        if (L > 0) CHECK(throws<domain_error>([&] { fps::raw::idft<P>(values, std::span(shorter).first(L - 1)); }), L);
    }
    {
        // Lengths past the chirp's middle product, into memory never touched (thrown before writing).
        const std::size_t words = std::size_t(17) << 21, bytes = words * sizeof(u32);
        void* map = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        CHECK(map != MAP_FAILED);
        if (map != MAP_FAILED) {
            const std::span<u32> big(static_cast<u32*>(map), words);
            CHECK(throws<std::length_error>([&] { fps::raw::dft<P>(Vec{1, 2}, big); }));
            CHECK(throws<std::length_error>([&] { fps::raw::idft<P>(big.first(std::size_t(7) << 22), big.first(std::size_t(7) << 22)); }));
            for (std::size_t L : {std::size_t(1) << 24, std::size_t(1) << 25})
                CHECK(throws<domain_error>([&] { fps::raw::dft<P>(Vec{1, 2}, big.first(L)); }), L);
            munmap(map, bytes);
        }
    }
}

}  // namespace

int main() {
    test_evaluate();
    test_interpolate();
    test_extrapolate();
    test_dft();
    test_raw_dft();
    test_threads();
    test_errors();
    if (failures) {
        std::printf("geometric: %d failures\n", failures);
        return 1;
    }
    std::printf("geometric: all passed\n");
    return 0;
}
