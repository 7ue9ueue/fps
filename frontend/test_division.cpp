// Tests of fps/division.hpp: divmod, f / g, f % g, /= and %=, against long division on edge sizes
// and random inputs, on both paths (a monomial divisor in the frontend; every other divisor in
// backend/division, here on divisors with trailing zeros, few terms, short and long quotients),
// against f = q g + r at large sizes, with aliasing, threads and every error.
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

std::mt19937_64 rng(21);

// 0, 1, powers of two and their neighbours.
const std::vector<std::size_t> kSizes = {0,  1,  2,  3,   4,   5,   7,   8,   9,   15,  16,   17,   31,  32,  33,
                                         63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025};

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 sub(u32 a, u32 b) { return a >= b ? a - b : a + P - b; }

u32 inv(u32 a) {
    u32 r = 1;
    for (u64 e = P - 2; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}

u32 nonzero() { return u32(1 + rng() % (P - 1)); }

// Kinds: 0 random, 1 mostly zeros, 2 all P - 1, 3 at most 20 nonzero words.
Vec random_vec(std::size_t n, int kind) {
    Vec a(n);
    if (kind == 3) {
        for (std::size_t t = rng() % 21; t-- > 0 && n;) a[rng() % n] = nonzero();
        return a;
    }
    for (u32& c : a) {
        const u64 r = rng();
        c = kind == 2 ? P - 1 : kind == 1 && r % 4 ? 0 : u32(r % P);
    }
    return a;
}

// A divisor of m >= 1 coefficients: kinds of random_vec, then the last word nonzero; kind 4: of
// degree below m - 1 (trailing zeros).
Vec random_divisor(std::size_t m, int kind) {
    if (kind == 4) {
        Vec g = random_divisor(1 + rng() % m, int(rng() % 4));
        g.resize(m);
        return g;
    }
    Vec g = random_vec(m, kind);
    if (g.back() == 0) g.back() = nonzero();
    return g;
}

Vec vec(const Poly& f) { return Vec(f.span().begin(), f.span().end()); }
Poly poly(const Vec& a) { return Poly(std::span<const u32>(a)); }

// Frees n words of P - 1, so the next allocation of n words likely reuses them.
void dirty_heap(std::size_t n) { (void)poly(Vec(n, P - 1)); }

int degree(const Vec& g) {
    int d = int(g.size()) - 1;
    while (d >= 0 && g[std::size_t(d)] == 0) --d;
    return d;
}

// Long division: q of max(n - d, 0) coefficients, r of d, for d = deg g >= 0.
std::pair<Vec, Vec> naive_divmod(const Vec& f, const Vec& g) {
    const std::size_t d = std::size_t(degree(g)), n = f.size(), k = n > d ? n - d : 0;
    Vec rem = f, q(k);
    rem.resize(std::max(n, d));
    const u32 c = inv(g[d]);
    for (std::size_t i = k; i-- > 0;) {
        const u32 t = mul(rem[i + d], c);
        q[i] = t;
        for (std::size_t j = 0; j <= d; ++j) rem[i + j] = sub(rem[i + j], mul(t, g[j]));
    }
    rem.resize(d);
    return {q, rem};
}

// Every form against long division; out-of-place results after dirty_heap.
void check_all(const Vec& f, const Vec& g, u64 tag = 0) {
    const auto [wq, wr] = naive_divmod(f, g);
    const Poly pf = poly(f), pg = poly(g);
    dirty_heap(wq.size());
    const auto [q, r] = fps::divmod(pf, pg);
    CHECK(vec(q) == wq && vec(r) == wr, f.size(), g.size(), tag);
    dirty_heap(wq.size());
    CHECK(vec(pf / pg) == wq, f.size(), g.size(), tag);
    dirty_heap(wr.size());
    CHECK(vec(pf % pg) == wr, f.size(), g.size(), tag);
    Poly h = pf;
    h /= pg;
    CHECK(vec(h) == wq, f.size(), g.size(), tag);
    h = pf;
    h %= pg;
    CHECK(vec(h) == wr, f.size(), g.size(), tag);
}

// Every pair of edge sizes, kinds of f and g by turns.
void test_small() {
    std::size_t turn = 0;
    for (const std::size_t n : kSizes)
        for (const std::size_t m : kSizes) {
            if (m == 0) continue;
            for (int rep = 0; rep < 2; ++rep, ++turn) {
                const Vec f = random_vec(n, int(turn % 4)), g = random_divisor(m, int(turn / 4 % 5));
                check_all(f, g, turn);
            }
        }
}

// The sparse path's bounds: g = c x^d plus t terms, the highest `gap` below d (the path needs
// gap >= 8 (t + 1)), the others below it, near it or spread; t around kSparseTerms = 16. f of n
// around d and around multiples of the gap (the blocks' edges), so terms reach the remainder from
// inside a block.
void test_sparse() {
    for (const std::size_t d : {std::size_t(13), std::size_t(30), std::size_t(100), std::size_t(140), std::size_t(300)})
        for (const std::size_t gap : {std::size_t(1), std::size_t(15), std::size_t(16), std::size_t(23), std::size_t(24),
                                      std::size_t(127), std::size_t(128), std::size_t(135), std::size_t(136), d})
            for (const std::size_t t : {1, 2, 15, 16, 17})
                for (const bool spread : {false, true}) {
                    const std::size_t top = d - gap;
                    if (gap > d || t - 1 > top) continue;
                    Vec g(d + 1);
                    g[d] = nonzero(), g[top] = nonzero();
                    std::vector<std::size_t> idx;  // t - 1 more terms below top
                    for (std::size_t j = top; j-- > 0;) idx.push_back(j);
                    if (spread) std::shuffle(idx.begin(), idx.end(), rng);
                    for (std::size_t i = 0; i + 1 < t; ++i) g[idx[i]] = nonzero();
                    for (const std::size_t n : {d - 1, d, d + 1, d + gap, d + gap + 1, d + 3 * gap - 1, 5 * d + 7})
                        check_all(random_vec(n, int(t % 3)), g, d * 1000 + gap * 10 + t);
                }
}

// The paths: monomials c x^k, constants, trailing zeros; divisors whose reversal G has t = 0 .. 18
// terms past G[0] among its first k (the frontend's recurrence before #147, at most 16, and its
// Newton path), sparse divisors such as x^d - x - 1.
void test_paths() {
    for (const std::size_t n : std::vector<std::size_t>{0, 1, 10, 1000, 4097}) {
        const Vec f = random_vec(n, 0);
        for (const std::size_t k : {std::size_t(0), std::size_t(1), std::size_t(5), n ? n - 1 : 0, n, n + 3}) {
            for (const u32 c : {u32(1), u32(2), P - 1, nonzero()}) {
                Vec g(k + 1 + rng() % 3);
                g[k] = c;
                check_all(f, g, k);
            }
            const Poly x_k = Poly::x(k), pf = poly(f);
            const Vec dropped(f.begin() + std::ptrdiff_t(std::min(k, n)), f.end());
            CHECK(vec(pf / x_k) == dropped, n, k);
            CHECK(vec(pf % x_k) == vec(pf.truncate(k)), n, k);
        }
        Vec scaled(n);
        for (std::size_t i = 0; i < n; ++i) scaled[i] = mul(f[i], inv(7));
        CHECK(vec(poly(f) / Poly{7}) == scaled && (poly(f) % Poly{7}).size() == 0, n);
    }
    const Poly f = poly(random_vec(500, 0)), g{3, 1, 4, 0, 0};
    CHECK(fps::divmod(f, g).first.size() == 498 && fps::divmod(f, g).second.size() == 2);
    CHECK((Poly{1, 2} / Poly{3, 1, 4}).size() == 0 && vec(Poly{1, 2} % Poly{3, 1, 4}) == (Vec{1, 2}));
    CHECK(vec(Poly() % Poly{3, 1, 4}) == (Vec{0, 0}) && (Poly() / Poly{3, 1, 4}).size() == 0);

    // Exactly t nonzero terms of G past G[0] among its first k, near the top of g and spread.
    for (const auto& [n, m] : {std::pair<std::size_t, std::size_t>{40, 30}, {300, 100}, {300, 250}, {2000, 40}})
        for (const std::size_t t : {0, 1, 2, 15, 16, 17, 18})
            for (const bool spread : {false, true}) {
                const std::size_t d = m - 1, k = n - d, top = std::min(k, d + 1);
                if (t >= top) continue;
                Vec g = random_vec(m, 0);
                for (std::size_t j = 0; j < top; ++j) g[d - j] = 0;
                g[d] = nonzero();
                std::vector<std::size_t> idx;
                for (std::size_t j = 1; j < top; ++j) idx.push_back(j);
                if (spread) std::shuffle(idx.begin(), idx.end(), rng);
                for (std::size_t i = 0; i < t; ++i) g[d - idx[i]] = nonzero();
                check_all(random_vec(n, 0), g, t);
            }
    // x^d - x - 1 and 1 - x^d / 2.
    for (const std::size_t d : {std::size_t(2), std::size_t(64), std::size_t(1000)}) {
        Vec g(d + 1);
        g[0] = P - 1, g[1] = P - 1, g[d] = 1;
        check_all(random_vec(3 * d + 5, 0), g, d);
        Vec h(d + 1);
        h[0] = 1, h[d] = P - inv(2);
        check_all(random_vec(5 * d, 0), h, d);
    }
    test_sparse();
}

// a = b up to trailing zeros.
bool same(const Vec& a, const Vec& b) {
    const std::size_t n = std::min(a.size(), b.size());
    return std::equal(a.begin(), a.begin() + std::ptrdiff_t(n), b.begin()) &&
           std::all_of(a.begin() + std::ptrdiff_t(n), a.end(), [](u32 c) { return c == 0; }) &&
           std::all_of(b.begin() + std::ptrdiff_t(n), b.end(), [](u32 c) { return c == 0; });
}

// f = q g + r exactly at large sizes, with q and r of the stated sizes (so deg r < deg g).
void check_identity(const Vec& f, const Vec& g) {
    const std::size_t d = std::size_t(degree(g)), k = f.size() > d ? f.size() - d : 0;
    const Poly pf = poly(f), pg = poly(g);
    const auto [q, r] = fps::divmod(pf, pg);
    CHECK(q.size() == k && r.size() == d, f.size(), g.size());
    CHECK(same(vec(q * pg + r), f), f.size(), g.size());
    CHECK(vec(pf / pg) == vec(q) && vec(pf % pg) == vec(r), f.size(), g.size());
}

void test_large() {
    for (const auto& [n, m] : {std::pair<std::size_t, std::size_t>{(1 << 17) + 1, 1 << 16},
                               {300'000, 3000},
                               {150'000, 149'990},
                               {1 << 18, (1 << 18) - 20},
                               {500'000, 53'336},
                               {500'000, 277'012},
                               {1 << 20, 4}})
        check_identity(random_vec(n, 0), random_divisor(m, 0));
    // Sparse divisors (the sparse path): x^500000 - x - 1, x^(2^18) - 1, 16 terms spread below
    // x^300000; then a kernel's: a sparse top over a dense rest.
    const std::size_t d = 500'000;
    Vec g(d + 1);
    g[0] = P - 1, g[1] = P - 1, g[d] = 1;
    check_identity(random_vec(1'000'000, 0), g);
    Vec cyclic((1 << 18) + 1);
    cyclic[0] = P - 1, cyclic.back() = 1;
    check_identity(random_vec(1'000'003, 0), cyclic);
    Vec spread(300'001);
    spread.back() = nonzero();
    for (int t = 0; t < 16; ++t) spread[rng() % 299'000] = nonzero();
    check_identity(random_vec(700'000, 0), spread);
    Vec h = random_vec(d + 1, 0);
    for (std::size_t j = 0; j < 1000; ++j) h[d - j] = 0;
    h[d] = 5, h[d - 7] = 3;
    check_identity(random_vec(d + 1000, 0), h);
}

// g the same object as f, and f / f, f % f.
void test_aliasing() {
    for (const std::size_t n : {std::size_t(1), std::size_t(17), std::size_t(300), std::size_t(5000)}) {
        Vec a = random_vec(n, 0);
        a.back() = nonzero();
        const auto [wq, wr] = naive_divmod(a, a);
        Poly f = poly(a);
        f /= f;
        CHECK(vec(f) == wq && wq == (Vec{1}), n);
        f = poly(a);
        f %= f;
        CHECK(vec(f) == wr && wr == Vec(n - 1), n);
        f = poly(a);
        const auto [q, r] = fps::divmod(f, f);
        CHECK(vec(q) == wq && vec(r) == wr, n);
    }
}

void test_threads() {
    std::vector<std::pair<Poly, Poly>> inputs, expected;
    for (int t = 0; t < 4; ++t) {
        const std::size_t u = std::size_t(t);
        const Poly f = poly(random_vec(20'000 + 1000 * u, 0)), g = poly(random_divisor(3000 * (u + 1), 0));
        inputs.emplace_back(f, g);
        expected.push_back(fps::divmod(f, g));
    }
    std::vector<int> ok(4, 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&, t] {
            bool good = true;
            const auto& [f, g] = inputs[std::size_t(t)];
            const auto& [wq, wr] = expected[std::size_t(t)];
            for (int r = 0; r < 3; ++r) {
                const auto [q, rem] = fps::divmod(f, g);
                good = good && vec(q) == vec(wq) && vec(rem) == vec(wr);
                good = good && vec(f / g) == vec(q) && vec(f % g) == vec(rem);
            }
            ok[std::size_t(t)] = good;
        });
    for (std::thread& t : threads) t.join();
    for (int t = 0; t < 4; ++t) CHECK(ok[std::size_t(t)], t);
}

void test_errors() {
    using std::domain_error;
    const Poly f = poly(random_vec(100, 0));
    for (const Poly& zero : {Poly(), Poly{0}, Poly(5)}) {
        CHECK(throws<domain_error>([&] { (void)fps::divmod(f, zero); }));
        CHECK(throws<domain_error>([&] { (void)(f / zero); }));
        CHECK(throws<domain_error>([&] { (void)(f % zero); }));
        CHECK(throws<domain_error>([&] { (void)(Poly() / zero); }));
        CHECK(throws<domain_error>([&] { (void)(zero % zero); }));
        Poly h = f;
        CHECK(throws<domain_error>([&] { h /= zero; }));
        CHECK(throws<domain_error>([&] { h %= zero; }));
        CHECK(vec(h) == vec(f));
    }
    // Past 2^25 coefficients: the kernel's blocks split the work into transforms of at most 2^25
    // words, so f of 2^25 + 2^10 by x^(2^25) + x + 1 (which threw std::length_error before #147)
    // gives q and r. Checked at random points: f(c) = q(c) g(c) + r(c).
    {
        const std::size_t d = std::size_t(1) << 25, k = std::size_t(1) << 10;
        const Poly big = poly(random_vec(d + k, 0));
        Poly g(d + 1);
        g[0] = 1, g[1] = 1, g[d] = 1;
        const auto [q, r] = fps::divmod(big, g);
        CHECK(q.size() == k && r.size() == d);
        CHECK(vec(big / g) == vec(q));
        for (int t = 0; t < 3; ++t) {
            const u32 c = u32(rng() % P);
            CHECK(big(c) == (u64(q(c)) * g(c) + r(c)) % P, c);
        }
    }
}

}  // namespace

int main() {
    test_small();
    test_paths();
    test_large();
    test_aliasing();
    test_threads();
    test_errors();
    if (failures) {
        std::printf("division: %d failures\n", failures);
        return 1;
    }
    std::printf("division: all passed\n");
    return 0;
}
