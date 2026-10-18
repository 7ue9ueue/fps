// Tests the small/NTT choice of squares and middle products (fps/polynomial.hpp, Products,
// issue #271) against O(n^2) references: both sides of each switch of small_square and
// small_middle on this target, found at run time (they move with the target's tuned costs), and a
// product of a polynomial with itself (f * f, mul(f, f, n), raw::mul(a, a, out)), which runs as a
// square.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fps/fps.hpp"

// Under ASan, every allocation starts filled with 0xbe bytes, so a result word the library forgets
// to write shows up on any allocator.
extern "C" const char* __asan_default_options() { return "max_malloc_fill_size=2147483647"; }

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = 998244353;
using Poly = fps::Polynomial<P>;
using Products = fps::detail::Products<P>;

int failures = 0;

void check(bool ok, const char* what, int line, u64 a = 0, u64 b = 0, u64 c = 0) {
    if (ok) return;
    if (++failures <= 20)
        std::printf("FAIL line %d: %s (%llu, %llu, %llu)\n", line, what, (unsigned long long)a, (unsigned long long)b,
                    (unsigned long long)c);
}
#define CHECK(cond, ...) check(bool(cond), #cond, __LINE__ __VA_OPT__(, ) __VA_ARGS__)

std::mt19937_64 rng(271);

// Kinds: 0 random, 1 mostly zeros, 2 mostly P - 1, 3 all P - 1 (the largest sums).
Vec random_vec(std::size_t n, int kind = 0) {
    Vec a(n);
    for (u32& c : a) {
        const u64 r = rng();
        c = kind == 3 || (kind == 2 && r % 4) ? P - 1 : kind == 1 && r % 4 ? 0 : u32(r % P);
    }
    // Two nonzero terms at least, so the product is not a shift.
    if (n >= 2) a[0] = a[0] ? a[0] : 1, a[n - 1] = a[n - 1] ? a[n - 1] : 1;
    return a;
}

Vec vec(const Poly& f) { return Vec(f.span().begin(), f.span().end()); }

Poly poly(const Vec& a) { return Poly(std::span<const u32>(a)); }

Vec naive_product(const Vec& a, const Vec& b) {
    if (a.empty() || b.empty()) return {};
    Vec c(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = u32((c[i + j] + u64(a[i]) * b[j]) % P);
    return c;
}

Vec first(const Vec& c, std::size_t n) {
    Vec r(n, 0);
    std::copy_n(c.begin(), std::min(n, c.size()), r.begin());
    return r;
}

// h[i] = sum_j f[i + j] g[m - 1 - j].
Vec naive_middle(const Vec& f, const Vec& g) {
    const std::size_t n = f.size(), m = g.size();
    Vec h(n - m + 1);
    for (std::size_t i = 0; i < h.size(); ++i) {
        u64 s = 0;
        for (std::size_t j = 0; j < m; ++j) s = (s + u64(f[i + j]) * g[m - 1 - j]) % P;
        h[i] = u32(s);
    }
    return h;
}

// The n where small_square(n) != small_square(n + 1), for n < 1024 (a switch on both sides).
std::vector<std::size_t> square_switches() {
    std::vector<std::size_t> at;
    for (std::size_t n = 1; n < 1024; ++n)
        if (Products::small_square(n) != Products::small_square(n + 1)) at.push_back(n);
    return at;
}

// a^2 of a of n words every way the API runs it: square, f * f, mul(f, f, k) and raw::mul(a, a,
// out) cut around the square's size, against the reference.
void check_square(const Vec& a) {
    if (a.empty()) return;
    const Vec want = naive_product(a, a);
    const Poly f = poly(a);
    CHECK(vec(fps::square(f)) == want, a.size());
    CHECK(vec(f * f) == want, a.size());
    Poly g = f;
    g *= g;
    CHECK(vec(g) == want, a.size());
    const std::size_t s = want.size();
    for (const std::size_t k : {std::size_t(1), a.size(), s - 1, s, s + 3}) {
        CHECK(vec(fps::mul(f, f, k)) == first(want, k), a.size(), k);
        Vec out(k, P - 1);
        fps::raw::mul<P>(a, a, out);
        CHECK(out == first(want, k), a.size(), k);
    }
}

void test_square() {
    const std::vector<std::size_t> at = square_switches();
    CHECK(!at.empty() && Products::small_square(1) && !Products::small_square(1024));
    for (const std::size_t n : at)
        for (const std::size_t k : {n - 1, n, n + 1, n + 2})
            for (const int kind : {0, 3}) check_square(random_vec(k, kind));
    // Every size to 160, and the NTT's square at lengths past a power of two.
    for (std::size_t n = 1; n <= 160; ++n) check_square(random_vec(n, int(rng() % 4)));
    for (const std::size_t n : {std::size_t(257), std::size_t(1025), std::size_t(2049)}) check_square(random_vec(n));
    // Long: against the product of two copies, which test_products checks.
    for (const std::size_t n : {std::size_t(1) << 15, (std::size_t(1) << 16) + 1}) {
        const Poly f = poly(random_vec(n)), copy = f;
        const Vec want = vec(f * copy);
        CHECK(vec(f * f) == want && vec(fps::square(f)) == want, n);
        CHECK(vec(fps::mul(f, f, n + 7)) == first(want, n + 7), n);
    }
}

// Spans of the same words that differ in length or start are a product, not a square.
void test_overlapping_factors() {
    const Vec w = random_vec(300);
    for (const std::size_t n : {std::size_t(20), std::size_t(150)}) {
        const std::span<const u32> a = std::span(w).first(n), b = std::span(w).first(n + 10),
                                   c = std::span(w).subspan(1, n);
        const auto product = [](std::span<const u32> x, std::span<const u32> y) {
            Vec out(x.size() + y.size() - 1, P - 1);
            fps::raw::mul<P>(x, y, out);
            return out;
        };
        CHECK(product(a, b) == naive_product(Vec(a.begin(), a.end()), Vec(b.begin(), b.end())), n);
        CHECK(product(a, c) == naive_product(Vec(a.begin(), a.end()), Vec(c.begin(), c.end())), n);
    }
}

void check_middle(std::size_t n, std::size_t m) {
    for (const int kind : {0, 3}) {
        const Vec f = random_vec(n, kind), g = random_vec(m, kind);
        CHECK(vec(fps::middle_product(poly(f), poly(g))) == naive_middle(f, g), n, m, kind);
    }
}

// Both sides of each switch of small_middle(n, m): in n for each m, and in m for each n.
void test_middle() {
    std::size_t switches = 0;
    for (const std::size_t m : {1, 2, 8, 16, 24, 32, 40, 48, 64, 80, 96, 128})
        for (std::size_t n = m; n < 8 * m + 64; ++n)
            if (Products::small_middle(n, m) != Products::small_middle(n + 1, m)) {
                ++switches;
                check_middle(n, m);
                check_middle(n + 1, m);
            }
    for (const std::size_t n : {std::size_t(1000), std::size_t(1024), std::size_t(1025), std::size_t(4096)})
        for (std::size_t m = 1; m < 256; ++m)
            if (Products::small_middle(n, m) != Products::small_middle(n, m + 1)) {
                ++switches;
                check_middle(n, m);
                check_middle(n, m + 1);
            }
    CHECK(switches > 0, switches);
    CHECK(Products::small_middle(1 << 20, 4) && !Products::small_middle(1 << 20, 1024));
    // The NTT's middle product past 2^25 still throws: f of 2^25 + 1 words by a dense g.
    Poly big((std::size_t(1) << 25) + 1), g(4096);
    big[0] = big[1] = g[0] = g[1] = 1;
    CHECK(!Products::small_middle(big.size(), g.size()));
    bool threw = false;
    try {
        (void)fps::middle_product(big, g);
    } catch (const std::length_error&) {
        threw = true;
    }
    CHECK(threw);
    // A square past 2^25 words throws too, written as f * f.
    big.resize((std::size_t(1) << 24) + 1);
    threw = false;
    try {
        (void)(big * big);
    } catch (const std::length_error&) {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

int main() {
    test_square();
    test_overlapping_factors();
    test_middle();
    if (failures) std::printf("%d failures\n", failures);
    return failures != 0;
}
