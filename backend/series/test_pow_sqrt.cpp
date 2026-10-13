// Tests of backend/series pow and sqrt (issue #107): against O(n^2) recurrences on 1, powers of two
// and their neighbours, 3 2^k and its neighbours, random sizes; f[0] = 1, a square, random; random,
// mostly zero and all P - 1 inputs; f shorter and longer than n; e = 0, 1, 2, P - 1, (P + 1) / 2,
// random; at large sizes against identities (f^2 = f f, f^-1 = inverse, sqrt(f)^2 = f, pow at e =
// 1/2 = sqrt, f^a f^b = f^(a+b), pow against exp(e log f)); the caller's scratch at every
// alignment; aliasing; threads; the length error.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "backend/ntt/ntt.hpp"
#include "backend/series/series.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = 998244353;

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

void check(bool ok, const char* what, int line, u64 a = 0, u64 b = 0) {
    if (ok) return;
    if (++failures <= 20)
        std::printf("FAIL line %d: %s (%llu, %llu)\n", line, what, (unsigned long long)a, (unsigned long long)b);
}
#define CHECK(cond, ...) check(bool(cond), #cond, __LINE__ __VA_OPT__(, ) __VA_ARGS__)

std::mt19937_64 rng(107);

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
u32 inv(u32 a) { return power(a, P - 2); }
u32 at(const Vec& f, std::size_t i) { return i < f.size() ? f[i] : 0; }
u32 random_word() { return u32(rng() % P); }
u32 random_nonzero() { return 1 + u32(rng() % (P - 1)); }

// Kinds: 0 random, 1 mostly zeros, 2 all P - 1.
Vec random_vec(std::size_t n, int kind) {
    Vec a(n);
    for (u32& c : a) {
        const u64 r = rng();
        c = kind == 2 ? P - 1 : kind == 1 && r % 4 ? 0 : u32(r % P);
    }
    return a;
}

// g = c (f / f[0])^e mod x^n from f g' = e f' g: i f_0 g_i = sum_(0<j<=i) (e j - (i - j)) f_j g_(i-j).
Vec ref_pow(const Vec& f, u32 e, u32 c, std::size_t n) {
    Vec g(n);
    if (n == 0) return g;
    g[0] = c;
    const u32 f0 = inv(f[0]);
    for (std::size_t i = 1; i < n; ++i) {
        u64 s = 0;
        for (std::size_t j = 1; j <= i; ++j) {
            const u32 w = u32((u64(mul(e, u32(j))) + P - (i - j) % P) % P);
            s = (s + u64(mul(w, at(f, j))) * g[i - j]) % P;
        }
        g[i] = mul(mul(u32(s), f0), inv(u32(i)));
    }
    return g;
}

// g with g^2 = f mod x^n, g[0] = c: 2 c g_i = f_i - sum_(0<j<i) g_j g_(i-j).
Vec ref_sqrt(const Vec& f, u32 c, std::size_t n) {
    Vec g(n);
    if (n == 0) return g;
    g[0] = c;
    const u32 inverse = inv(mul(2, c));
    for (std::size_t i = 1; i < n; ++i) {
        u64 s = at(f, i);
        for (std::size_t j = 1; j < i; ++j) s = (s + P - u64(g[j]) * g[i - j] % P) % P;
        g[i] = mul(u32(s), inverse);
    }
    return g;
}

Vec run_pow(const Vec& f, u32 e, u32 c, std::size_t n) {
    Vec g(n, P - 1);
    fps::series::pow(f, e, c, g);
    return g;
}
Vec run_sqrt(const Vec& f, u32 c, std::size_t n) {
    Vec g(n, P - 1);
    fps::series::sqrt(f, c, g);
    return g;
}

const std::vector<std::size_t> kSizes = {1,   2,   3,   4,   5,   7,   8,   9,   15,  16,  17,  31,   32,   33,  48,
                                         63,  64,  65,  95,  96,  97,  127, 128, 129, 191, 192, 193,  255,  256,
                                         257, 383, 384, 385, 511, 512, 513, 767, 768, 769, 1023, 1024, 1025, 1500,
                                         2047, 2048, 2049, 3071, 3073, 4095, 4096, 4097};

void test_references() {
    std::vector<std::size_t> sizes = kSizes;
    for (int i = 0; i < 12; ++i) sizes.push_back(1 + rng() % 1500);
    const u32 exponents[] = {0, 1, 2, P - 1, (P + 1) / 2};
    for (std::size_t n : sizes) {
        const bool large = n > 1100;  // fewer cases: the references are O(n^2)
        for (int kind = 0; kind < (large ? 1 : 3); ++kind) {
            for (std::size_t fsize : {n, n + 3, n > 2 ? n / 2 : n}) {
                if (large && fsize != n) continue;
                const Vec base = random_vec(fsize, kind);
                for (int first = 0; first < 3; ++first) {  // f[0] = 1, a square, random
                    Vec f = base;
                    const u32 root = random_nonzero();
                    f[0] = first == 0 ? 1 : first == 1 ? mul(root, root) : kind == 2 ? P - 1 : random_nonzero();
                    if (first == 1) CHECK(run_sqrt(f, root, n) == ref_sqrt(f, root, n), n, fsize);
                    if (first == 0) CHECK(run_sqrt(f, P - 1, n) == ref_sqrt(f, P - 1, n), n, fsize);
                    const u32 e = large ? random_word() : exponents[rng() % 5], c = first == 0 ? 1 : random_word();
                    CHECK(run_pow(f, e, c, n) == ref_pow(f, e, c, n), n, e);
                    if (!large && first == 2) {
                        const u32 r = random_word();
                        CHECK(run_pow(f, r, c, n) == ref_pow(f, r, c, n), n, r);
                    }
                }
            }
        }
        // f of one word: a constant
        Vec one = {9}, want(n);
        want[0] = 5;
        CHECK(run_pow(one, random_word(), 5, n) == want, n);
        want[0] = 3;
        CHECK(run_sqrt(one, 3, n) == want, n);
    }
    // n = 0 writes nothing
    Vec none;
    fps::series::pow(Vec{1}, 2, 1, none);
    fps::series::sqrt(Vec{1}, 1, none);
}

Vec product(const Vec& f, const Vec& g, std::size_t n) {
    Vec out(n);
    fps::ntt::multiply(f, g, out);
    return out;
}

// f mod x^n.
Vec cut(Vec f, std::size_t n) {
    f.resize(n);
    return f;
}

void test_identities() {
    std::vector<std::size_t> sizes = {4097, 6143, 16383, 16384, 16385, 131072, 196609, 300001};
    if (!kSanitized) sizes.insert(sizes.end(), {1048576, 1048579, 1572864, (std::size_t(1) << 22) + 1});
    for (std::size_t n : sizes) {
        Vec f = random_vec(n, 0);
        const u32 root = random_nonzero();
        f[0] = mul(root, root);
        // f^2 = f f, f^3 = f f f, f^-1 = inverse(f)
        const Vec f2 = run_pow(f, 2, mul(f[0], f[0]), n);
        CHECK(f2 == product(f, f, n), n);
        CHECK(run_pow(f, 3, mul(f2[0], f[0]), n) == product(f2, f, n), n);
        Vec fi(n);
        fps::series::inverse(f, fi);
        CHECK(run_pow(f, P - 1, inv(f[0]), n) == fi, n);
        // sqrt(f)^2 = f; pow at e = 1/2 with the same root
        const Vec g = run_sqrt(f, root, n);
        CHECK(product(g, g, n) == f, n);
        CHECK(run_pow(f, (P + 1) / 2, root, n) == g, n);
        // f^a f^b = f^(a + b) with c = 1, and pow = exp(e log(f / f[0]))
        const u32 a = random_word(), b = random_word();
        const Vec fa = run_pow(f, a, 1, n), fb = run_pow(f, b, 1, n);
        CHECK(product(fa, fb, n) == run_pow(f, (a + b) % P, 1, n), n);
        Vec l(n), u = f;
        const u32 f0 = inv(f[0]);
        for (u32& x : u) x = mul(x, f0);
        fps::series::log(u, l);
        for (u32& x : l) x = mul(x, a);
        fps::series::exp(l, l);
        CHECK(l == fa, n);
        // f shorter than n: a polynomial
        const Vec p = cut(f, n / 3);
        const Vec ps = run_sqrt(product(p, p, n), p[0], n);
        CHECK(ps == cut(p, n), n);
    }
    // Closed forms at 10^6 (2^18 under ASan): (1 - x)^-1 = sum x^i; sqrt(1 - 4x) has
    // -2 Catalan(i - 1) at i >= 1.
    const std::size_t n = kSanitized ? 262147 : 1000000;
    const Vec g = run_pow(Vec{1, P - 1}, P - 1, 1, n);
    CHECK(std::all_of(g.begin(), g.end(), [](u32 c) { return c == 1; }), n);
    const Vec s = run_sqrt(Vec{1, P - 4}, 1, n);
    bool ok = s[0] == 1;
    u32 catalan = 1;  // Catalan(i - 1)
    for (std::size_t i = 1; i < n && ok; ++i) {
        ok = s[i] == (catalan ? P - mul(2, catalan) : 0);
        catalan = mul(mul(catalan, u32((4 * (i - 1) + 2) % P)), inv(u32(i + 1)));  // C(i) = C(i-1) (4i - 2) / (i + 1)
    }
    CHECK(ok, n);
}

// The caller's scratch at each offset of a cache line, filled with garbage; words past it unwritten.
void test_scratch() {
    for (std::size_t n : {33ul, 65ul, 100ul, 200ul, 1025ul, 4096ul, 70000ul}) {
        for (std::size_t offset = 0; offset < 16; offset += 3) {
            Vec f = random_vec(n, 0);
            f[0] = 4;
            const std::size_t words = std::max(fps::series::pow_scratch(n), fps::series::sqrt_scratch(n));
            Vec scratch(words + offset + 8, P - 1);
            const std::span<u32> s(scratch.data() + offset, words);
            Vec g(n);
            fps::series::pow(f, 12345, 7, g, s.first(fps::series::pow_scratch(n)));
            CHECK(g == run_pow(f, 12345, 7, n), n, offset);
            fps::series::sqrt(f, 2, g, s.first(fps::series::sqrt_scratch(n)));
            CHECK(g == run_sqrt(f, 2, n), n, offset);
            CHECK(std::all_of(scratch.end() - 8, scratch.end(), [](u32 c) { return c == P - 1; }), n, offset);
        }
    }
    const std::size_t n = 1000;
    Vec f = random_vec(n, 0), g(n, 5);
    f[0] = 1;
    for (int op = 0; op < 2; ++op) {  // too little scratch: std::invalid_argument, out untouched
        Vec small((op == 0 ? fps::series::pow_scratch(n) : fps::series::sqrt_scratch(n)) - 1);
        bool threw = false;
        try {
            if (op == 0) fps::series::pow(f, 3, 1, g, small);
            else fps::series::sqrt(f, 1, g, small);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw && std::all_of(g.begin(), g.end(), [](u32 c) { return c == 5; }), op);
    }
}

// out overlapping f: in place and shifted both ways.
void test_aliasing() {
    for (std::size_t n : {40ul, 64ul, 65ul, 100ul, 1024ul, 1500ul, 5000ul}) {
        for (std::ptrdiff_t shift : {0, 1, -1, 7, -7, 300, -300}) {
            if (std::size_t(std::abs(shift)) >= n) continue;
            Vec f = random_vec(n, 0);
            f[0] = 9;
            for (int op = 0; op < 2; ++op) {
                const Vec want = op == 0 ? ref_pow(f, 777, 2, n) : run_sqrt(f, 3, n);
                Vec buffer(3 * n);
                std::copy(f.begin(), f.end(), buffer.begin() + std::ptrdiff_t(n));
                const std::span<const u32> in(buffer.data() + n, n);
                const std::span<u32> out(buffer.data() + std::ptrdiff_t(n) + shift, n);
                if (op == 0) fps::series::pow(in, 777, 2, out);
                else fps::series::sqrt(in, 3, out);
                CHECK(Vec(out.begin(), out.end()) == want, n, op);
            }
        }
    }
}

void test_threads() {
    std::vector<std::thread> threads;
    std::vector<int> ok(4, 0);
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([t, &ok] {
            std::mt19937_64 r(t);
            bool good = true;
            for (std::size_t n : {100ul, 5000ul, 70000ul}) {
                Vec f(n);
                for (u32& c : f) c = u32(r() % P);
                f[0] = 1;
                Vec s(n), sq(n);
                fps::series::sqrt(f, 1, s);
                fps::series::pow(s, 2, 1, sq);
                good = good && sq == f;
            }
            ok[t] = good;
        });
    for (auto& th : threads) th.join();
    for (int t = 0; t < 4; ++t) CHECK(ok[t], t);
}

void test_length() {
    const std::size_t n = fps::series::kMaxSize + 1;
    auto words = std::make_unique_for_overwrite<u32[]>(n);  // never written
    const Vec f = {1, 2};
    const std::span<u32> out(words.get(), n);
    for (int op = 0; op < 2; ++op) {
        bool threw = false;
        try {
            if (op == 0) fps::series::pow(f, 5, 1, out);
            else fps::series::sqrt(f, 1, out);
        } catch (const std::length_error&) {
            threw = true;
        }
        CHECK(threw, op);
    }
}

}  // namespace

int main() {
    test_references();
    test_identities();
    test_scratch();
    test_aliasing();
    test_threads();
    test_length();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("series pow, sqrt: all tests passed\n");
    return 0;
}
