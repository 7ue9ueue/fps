// Tests of backend/composition: compose and power_projection against O(n^3) references on 0, 1,
// powers of two and their neighbours, random sizes and inputs (random, mostly zero, all P - 1), f
// and g shorter and longer than n, g[0] != 0 (not read); compositional_inverse by f(g) = x; at
// large sizes identities (exp(g), 1 / (1 - g), the transposition <a, f> = f(g)[n - 1], a closed
// form of power projection); the leaf passes of every simd struct against the scalar ones;
// threads; the length error.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "backend/composition/composition.hpp"
#include "backend/series/series.hpp"
#include "backend/simd/select.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = 998244353;
namespace c = fps::composition;

int failures = 0;

void check(bool ok, const char* what, int line, u64 a = 0, u64 b = 0) {
    if (ok) return;
    if (++failures <= 20)
        std::printf("FAIL line %d: %s (%llu, %llu)\n", line, what, (unsigned long long)a, (unsigned long long)b);
}
#define CHECK(cond, ...) check(bool(cond), #cond, __LINE__ __VA_OPT__(, ) __VA_ARGS__)

std::mt19937_64 rng(105);

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 add(u32 a, u32 b) { return (a + b) % P; }
u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}

// Kinds: 0 random, 1 mostly zeros, 2 all P - 1.
Vec random_vec(std::size_t n, int kind) {
    Vec a(n);
    for (auto& x : a) x = kind == 0 ? u32(rng() % P) : kind == 1 ? (rng() % 8 == 0 ? u32(rng() % P) : 0) : P - 1;
    return a;
}

// a b mod x^n.
Vec product(const Vec& a, const Vec& b, std::size_t n) {
    Vec c(n, 0);
    for (std::size_t i = 0; i < std::min(a.size(), n); ++i)
        if (a[i])
            for (std::size_t j = 0; j < b.size() && i + j < n; ++j) c[i + j] = add(c[i + j], mul(a[i], b[j]));
    return c;
}

// f(g) mod x^n with g[0] taken as 0: sum of f_i g^i.
Vec compose_reference(const Vec& f, Vec g, std::size_t n) {
    if (!g.empty()) g[0] = 0;
    Vec h(n, 0), pw(n, 0);
    if (n) pw[0] = 1;
    for (std::size_t i = 0; i < std::min(f.size(), n); ++i) {
        for (std::size_t k = 0; k < n; ++k) h[k] = add(h[k], mul(f[i], pw[k]));
        pw = product(pw, g, n);
    }
    return h;
}

// [x^(n-1)] g^k for k < n, g[0] taken as 0.
Vec projection_reference(Vec g, std::size_t n) {
    if (!g.empty()) g[0] = 0;
    Vec a(n), pw(n, 0);
    if (n) pw[0] = 1;
    for (std::size_t k = 0; k < n; ++k) {
        a[k] = pw[n - 1];
        pw = product(pw, g, n);
    }
    return a;
}

Vec compose(const Vec& f, const Vec& g, std::size_t n) {
    Vec h(n, 12345);
    c::compose(f, g, h);
    return h;
}

std::vector<std::size_t> small_sizes() {
    std::vector<std::size_t> s = {0, 1, 2, 3, 5, 15, 16, 17, 31, 32, 33};
    for (std::size_t p = 64; p <= 512; p *= 2) s.insert(s.end(), {p - 1, p, p + 1});
    for (int i = 0; i < 6; ++i) s.push_back(34 + rng() % 400);
    return s;
}

void test_compose_small() {
    for (std::size_t n : small_sizes())
        for (int kind = 0; kind < 3; ++kind) {
            const std::size_t fs = kind == 1 ? n / 2 + 1 : n + 3, gs = kind == 2 ? n + 5 : (n + 1) / 3 + 2;
            Vec f = random_vec(fs, kind), g = random_vec(gs, kind);
            if (kind != 2 && !g.empty()) g[0] = u32(rng() % P);  // not read
            if (kind == 0 && g.size() > 1) g[1] = 0;
            const Vec want = compose_reference(f, g, n), got = compose(f, g, n);
            CHECK(got == want, n, kind);
        }
    // f or g empty.
    CHECK(compose({}, {1, 2, 3}, 40) == Vec(40, 0));
    const Vec f = random_vec(40, 0);
    Vec constant(40, 0);
    constant[0] = f[0];
    CHECK(compose(f, {}, 40) == constant);
}

void test_projection_small() {
    for (std::size_t n : small_sizes())
        for (int kind = 0; kind < 3; ++kind) {
            Vec g = random_vec(kind == 1 ? n / 2 + 1 : n + 2, kind);
            if (!g.empty()) g[0] = 7;
            Vec a(n, 12345);
            c::power_projection(g, a);
            CHECK(a == projection_reference(g, n), n, kind);
        }
}

void test_inverse(std::size_t n, int kind) {
    Vec f = random_vec(n + 2, kind);
    if (f.size() > 0) f[0] = 0;
    if (f.size() > 1) f[1] = f[1] ? f[1] : 1;
    Vec g(n, 12345);
    c::compositional_inverse(f, g);
    Vec x(n, 0);
    if (n > 1) x[1] = 1;
    CHECK(compose(f, g, n) == x, n, kind);
    if (n) CHECK(g[0] == 0, n);
}

void test_inverse_sizes() {
    for (std::size_t n : small_sizes())
        for (int kind = 0; kind < 3; ++kind) test_inverse(n, kind);
    for (std::size_t n : {std::size_t(4096), std::size_t(16384 + 5)}) test_inverse(n, 0);
}

// exp(g) = compose(sum x^k / k!, g), 1 / (1 - g) = compose(sum x^k, g).
void test_compose_large() {
    for (std::size_t n : {std::size_t(1000), std::size_t(4096), std::size_t(4097), std::size_t(65536 + 3), std::size_t(1) << 17}) {
        Vec g = random_vec(n, 0);
        g[0] = 0;
        Vec e(n), ones(n, 1), want(n), one_minus(n);
        u32 fact = 1;
        for (std::size_t k = 0; k < n; ++k) {
            if (k) fact = mul(fact, u32(k));
            e[k] = power(fact, P - 2);
        }
        fps::series::exp(g, want);
        CHECK(compose(e, g, n) == want, n);
        for (std::size_t k = 0; k < n; ++k) one_minus[k] = g[k] ? P - g[k] : 0;
        one_minus[0] = 1;
        fps::series::inverse(one_minus, want);
        CHECK(compose(ones, g, n) == want, n);
    }
}

void test_projection_large() {
    for (std::size_t n : {std::size_t(2048), std::size_t(5000), std::size_t(1) << 16}) {
        // The transposition: sum_k a[k] f[k] = [x^(n-1)] f(g).
        const Vec g = random_vec(n, 0), f = random_vec(n, 0);
        Vec a(n);
        c::power_projection(g, a);
        u32 dot = 0;
        for (std::size_t k = 0; k < n; ++k) dot = add(dot, mul(a[k], f[k]));
        CHECK(dot == compose(f, g, n)[n - 1], n);
        // g = x / (1 - x): [x^(n-1)] g^k = C(n - 2, k - 1) for k >= 1.
        Vec h(n, 1), b(n);
        h[0] = 0;
        c::power_projection(h, b);
        u32 binomial = 1;  // C(n - 2, k - 1)
        bool ok = b[0] == 0;
        for (std::size_t k = 1; k < n; ++k) {
            ok = ok && b[k] == binomial;
            binomial = mul(mul(binomial, u32(n - 1 - k)), power(u32(k), P - 2));
        }
        CHECK(ok, n);
    }
}

// The leaf passes of every simd struct against the scalar ones.
void test_passes() {
    const std::size_t m = 256, pairs = m / 4;
    const auto& tables = c::detail::pair_tables(pairs);
    const Vec F = random_vec(4 * m, 0), F2 = random_vec(4 * m, 0), H = random_vec(2 * m, 0);
    using Ref = c::detail::Pairs<fps::simd::Scalar>;
    Vec rF = F, rG(2 * m), rV(2 * m), rW(2 * m), rA(4 * m);
    Ref::graeffe_pass(rF.data(), rG.data(), pairs, tables, true);
    Ref::projection_pass(F.data(), F2.data(), rV.data(), rW.data(), pairs, tables);
    Ref::spread_pass(H.data(), rA.data(), pairs, tables);
    // The scalar Graeffe against its definition on pair 3: c(z) c(-z) mod (z^8 -+ s), then the CRT.
    {
        const std::size_t p = 3;
        const u32 s = mul(tables.s[p], power(u32((u64(1) << 32) % P), P - 2));
        u32 V[2][8] = {};
        for (int half = 0; half < 2; ++half) {
            const u32 t = half ? P - s : s;
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j) {
                    u32 x = mul(F[16 * p + 8 * half + i], F[16 * p + 8 * half + j]);
                    if (j % 2) x = x ? P - x : 0;
                    if (i + j >= 8) x = mul(x, t);
                    V[half][(i + j) % 8] = add(V[half][(i + j) % 8], x);
                }
        }
        const u32 half = (P + 1) / 2, inverse = power(mul(2, s), P - 2);
        bool ok = true;
        for (int k = 0; k < 4; ++k) {
            ok = ok && V[0][2 * k + 1] == 0 && V[1][2 * k + 1] == 0;
            ok = ok && rG[8 * p + k] == mul(add(V[0][2 * k], V[1][2 * k]), half);
            ok = ok && rG[8 * p + k + 4] == mul(add(V[0][2 * k], P - V[1][2 * k]), inverse);
        }
        CHECK(ok);
    }
    fps::simd::for_each(fps::simd::All{}, [&]<class S> {
        using N = c::detail::Pairs<c::detail::Narrow<S>>;
        Vec f = F, G(2 * m), V(2 * m), W(2 * m), A(4 * m);
        N::graeffe_pass(f.data(), G.data(), pairs, tables, true);
        CHECK(f == rF && G == rG, S::lanes);
        N::projection_pass(F.data(), F2.data(), V.data(), W.data(), pairs, tables);
        CHECK(V == rV && W == rW, S::lanes);
        N::spread_pass(H.data(), A.data(), pairs, tables);
        CHECK(A == rA, S::lanes);
    });
}

void test_threads() {
    const std::size_t n = 3000;
    const Vec f = random_vec(n, 0), g = random_vec(n, 0);
    const Vec want = compose(f, g, n);
    std::vector<Vec> got(4, Vec(n));
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back([&, t] {
        for (int r = 0; r < 3; ++r) c::compose(f, g, got[std::size_t(t)]);
    });
    for (auto& t : threads) t.join();
    for (const auto& x : got) CHECK(x == want);
}

void test_length_error() {
    Vec big(c::kMaxSize + 1);
    bool thrown = false;
    try {
        const Vec f = {1}, g = {0, 1};
        c::compose(f, g, big);
    } catch (const std::length_error&) {
        thrown = true;
    }
    CHECK(thrown);
}

}  // namespace

int main() {
    test_passes();
    test_compose_small();
    test_projection_small();
    test_inverse_sizes();
    test_compose_large();
    test_projection_large();
    test_threads();
    test_length_error();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
