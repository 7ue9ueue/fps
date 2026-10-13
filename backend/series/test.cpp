// Tests of backend/series: inverse, log and exp against O(n^2) recurrences on 1, powers of two and
// their neighbours, random sizes and inputs (random, mostly zero, all P - 1), f shorter and longer
// than n; at large sizes against identities (f inverse(f) = 1, log exp f = f, exp log f = f) and
// closed forms; the caller's scratch at every alignment; aliasing; threads; the length error.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

std::mt19937_64 rng(87);

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
u32 inv(u32 a) { return power(a, P - 2); }

u32 at(const Vec& f, std::size_t i) { return i < f.size() ? f[i] : 0; }

// Kinds: 0 random, 1 mostly zeros, 2 all P - 1.
Vec random_vec(std::size_t n, int kind) {
    Vec a(n);
    for (u32& c : a) {
        const u64 r = rng();
        c = kind == 2 ? P - 1 : kind == 1 && r % 4 ? 0 : u32(r % P);
    }
    return a;
}

// The references, O(n^2), on f mod x^n.
Vec ref_inverse(const Vec& f, std::size_t n) {
    Vec g(n);
    const u32 c = inv(f[0]);
    for (std::size_t i = 0; i < n; ++i) {
        u64 s = i == 0;
        for (std::size_t j = 1; j <= i; ++j) s = (s + P - u64(at(f, j)) * g[i - j] % P) % P;
        g[i] = mul(u32(s), c);
    }
    return g;
}

Vec ref_log(const Vec& f, std::size_t n) {  // f g' = f' with G = i g_i
    Vec G(n), g(n);
    for (std::size_t i = 1; i < n; ++i) {
        u64 s = mul(u32(i), at(f, i));
        for (std::size_t k = 1; k < i; ++k) s = (s + P - u64(G[k]) * at(f, i - k) % P) % P;
        G[i] = u32(s);
        g[i] = mul(G[i], inv(u32(i)));
    }
    return g;
}

Vec ref_exp(const Vec& f, std::size_t n) {  // g' = f' g
    Vec g(n);
    if (n) g[0] = 1;
    for (std::size_t i = 1; i < n; ++i) {
        u64 s = 0;
        for (std::size_t k = 1; k <= i; ++k) s = (s + u64(mul(u32(k), at(f, k))) * g[i - k]) % P;
        g[i] = mul(u32(s), inv(u32(i)));
    }
    return g;
}

Vec with_first(Vec f, u32 c) {
    if (!f.empty()) f[0] = c;
    return f;
}

Vec run_inverse(const Vec& f, std::size_t n) {
    Vec g(n, P - 1);
    fps::series::inverse(f, g);
    return g;
}
Vec run_log(const Vec& f, std::size_t n) {
    Vec g(n, P - 1);
    fps::series::log(f, g);
    return g;
}
Vec run_exp(const Vec& f, std::size_t n) {
    Vec g(n, P - 1);
    fps::series::exp(f, g);
    return g;
}

const std::vector<std::size_t> kSizes = {1,   2,   3,   4,   5,   7,   8,   9,   15,  16,  17,  31,   32,   33,  48,
                                         63,  64,  65,  95,  96,  97,  127, 128, 129, 191, 192, 193,  255,  256,
                                         257, 383, 384, 385, 511, 512, 513, 767, 768, 769, 1023, 1024, 1025, 1500,
                                         2047, 2048, 2049, 3073, 4095, 4096, 4097};

void test_references() {
    std::vector<std::size_t> sizes = kSizes;
    for (int i = 0; i < 12; ++i) sizes.push_back(1 + rng() % 1500);
    for (std::size_t n : sizes) {
        const bool large = n > 1100;  // one kind and size of f: the references are O(n^2)
        for (int kind = 0; kind < (large ? 1 : 3); ++kind) {
            for (std::size_t fsize : {n, n + 3, n > 2 ? n / 2 : n}) {
                if (large && fsize != n) continue;
                const Vec f = random_vec(fsize, kind);
                const Vec fi = with_first(f, kind == 2 ? P - 1 : 1 + u32(rng() % (P - 1)));
                CHECK(run_inverse(fi, n) == ref_inverse(fi, n), n, fsize);
                const Vec fl = with_first(f, 1);
                CHECK(run_log(fl, n) == ref_log(fl, n), n, fsize);
                const Vec fe = with_first(f, 0);
                CHECK(run_exp(fe, n) == ref_exp(fe, n), n, fsize);
            }
        }
        // f of one word
        CHECK(run_inverse(Vec{7}, n) == ref_inverse(Vec{7}, n), n);
        CHECK(run_log(Vec{1}, n) == Vec(n, 0), n);
        CHECK(run_exp(Vec{0}, n) == ref_exp(Vec{0}, n), n);
        CHECK(run_exp(Vec{}, n) == ref_exp(Vec{}, n), n);
    }
    // n = 0 writes nothing
    Vec none;
    fps::series::inverse(Vec{1}, none);
    fps::series::log(Vec{1}, none);
    fps::series::exp(Vec{}, none);
}

// f g mod x^n.
Vec product(const Vec& f, const Vec& g, std::size_t n) {
    Vec out(n);
    fps::ntt::multiply(f, g, out);
    return out;
}

void test_identities() {
    std::vector<std::size_t> sizes = {16383, 16384, 16385, 131072, 300001};
    if (!kSanitized) sizes.insert(sizes.end(), {1048576, 1048579, (std::size_t(1) << 22) + 1});
    for (std::size_t n : sizes) {
        Vec f = random_vec(n, 0);
        f[0] = 3;
        const Vec g = run_inverse(f, n);
        Vec one(n);
        one[0] = 1;
        CHECK(product(f, g, n) == one, n);
        f[0] = 0;
        const Vec e = run_exp(f, n);
        CHECK(run_log(e, n) == f, n);
        f[0] = 1;
        const Vec l = run_log(f, n);
        CHECK(run_exp(l, n) == f, n);
    }
    // Closed forms at 10^6 (2^18 under ASan): 1 / (1 - x), log 1/(1 - x) = sum x^i / i, exp x.
    const std::size_t n = kSanitized ? 262147 : 1000000;
    Vec q = {1, P - 1};
    const Vec g = run_inverse(q, n);
    CHECK(std::all_of(g.begin(), g.end(), [](u32 c) { return c == 1; }), n);
    const Vec l = run_log(g, n);
    bool ok = l[0] == 0;
    for (std::size_t i = 1; i < n && ok; ++i) ok = l[i] == inv(u32(i));
    CHECK(ok, n);
    const Vec e = run_exp(Vec{0, 1}, n);
    u32 fact_inv = 1;
    ok = e[0] == 1;
    for (std::size_t i = 1; i < n && ok; ++i) ok = e[i] == (fact_inv = mul(fact_inv, inv(u32(i))));
    CHECK(ok, n);
}

// The caller's scratch at each offset of a cache line, filled with garbage; words past it unwritten.
void test_scratch() {
    for (std::size_t n : {33ul, 65ul, 200ul, 1025ul, 4096ul, 70000ul}) {
        for (std::size_t offset = 0; offset < 16; offset += 3) {
            const Vec f = with_first(random_vec(n, 0), 1);
            const std::size_t words = std::max({fps::series::inverse_scratch(n), fps::series::log_scratch(n),
                                                fps::series::exp_scratch(n)});
            Vec scratch(words + offset + 8, P - 1);
            const std::span<u32> s(scratch.data() + offset, words);
            Vec g(n);
            fps::series::inverse(f, g, s.first(fps::series::inverse_scratch(n)));
            CHECK(g == run_inverse(f, n), n, offset);
            fps::series::log(f, g, s.first(fps::series::log_scratch(n)));
            CHECK(g == run_log(f, n), n, offset);
            const Vec f0 = with_first(f, 0);
            fps::series::exp(f0, g, s.first(fps::series::exp_scratch(n)));
            CHECK(g == run_exp(f0, n), n, offset);
            CHECK(std::all_of(scratch.end() - 8, scratch.end(), [](u32 c) { return c == P - 1; }), n, offset);
        }
    }
    // Too little scratch: std::invalid_argument, out untouched.
    const std::size_t n = 1000;
    Vec f = with_first(random_vec(n, 0), 1), g(n, 5), small(fps::series::log_scratch(n) - 1);
    bool threw = false;
    try {
        fps::series::log(f, g, small);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw && std::all_of(g.begin(), g.end(), [](u32 c) { return c == 5; }));
}

// log and exp with out overlapping f: in place and shifted both ways.
void test_aliasing() {
    for (std::size_t n : {40ul, 64ul, 65ul, 100ul, 1024ul, 1500ul, 5000ul}) {
        for (std::ptrdiff_t shift : {0, 1, -1, 7, -7, 300, -300}) {
            if (std::size_t(std::abs(shift)) >= n) continue;
            Vec f = random_vec(n, 0);
            for (int op = 0; op < 2; ++op) {
                f[0] = op == 0 ? 1 : 0;
                const Vec want = op == 0 ? ref_log(f, n) : run_exp(f, n);
                Vec buffer(3 * n);
                std::copy(f.begin(), f.end(), buffer.begin() + std::ptrdiff_t(n));
                const std::span<const u32> in(buffer.data() + n, n);
                const std::span<u32> out(buffer.data() + std::ptrdiff_t(n) + shift, n);
                if (op == 0) fps::series::log(in, out);
                else fps::series::exp(in, out);
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
                f[0] = 0;
                Vec e(n), l(n);
                fps::series::exp(f, e);
                fps::series::log(e, l);
                good = good && l == f;
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
    for (int op = 0; op < 3; ++op) {
        bool threw = false;
        try {
            if (op == 0) fps::series::inverse(f, out);
            if (op == 1) fps::series::log(f, out);
            if (op == 2) fps::series::exp(Vec{0, 2}, out);
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
    std::printf("series: all tests passed\n");
    return 0;
}
