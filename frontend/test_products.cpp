// Tests of fps/products.hpp against O(n^2) references: mul, square, product, middle_product and
// raw::mul on each path (shift, small::multiply, ntt::multiply, raw::mul's truncated loop), on
// edge sizes and random inputs.
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

std::mt19937_64 rng(18);

// 0, 1, powers of two and their neighbours.
const std::vector<std::size_t> kSizes = {0,  1,  2,  3,   4,   5,   7,   8,   9,   15,  16,   17,   31,  32,  33,
                                         63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025};

// Kinds: 0 random, 1 mostly zeros, 2 mostly P - 1, 3 all P - 1 (the largest sums).
Vec random_vec(std::size_t n, int kind = 0) {
    Vec a(n);
    for (u32& c : a) {
        const u64 r = rng();
        c = kind == 3 || (kind == 2 && r % 4) ? P - 1 : kind == 1 && r % 4 ? 0 : u32(r % P);
    }
    return a;
}

Vec vec(const Poly& f) { return Vec(f.span().begin(), f.span().end()); }

Poly poly(const Vec& a) { return Poly(std::span<const u32>(a)); }

// Frees n words of P - 1, so the next allocation of n words likely reuses them: a result word the
// library forgets to write then shows up.
void dirty_heap(std::size_t n) { (void)poly(Vec(n, P - 1)); }

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }

Vec naive_product(const Vec& a, const Vec& b) {
    if (a.empty() || b.empty()) return {};
    Vec c(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = u32((c[i + j] + u64(a[i]) * b[j]) % P);
    return c;
}

// The first n coefficients of c, zeros past its end.
Vec first(const Vec& c, std::size_t n) {
    Vec r(n, 0);
    std::copy_n(c.begin(), std::min(n, c.size()), r.begin());
    return r;
}

// The definition: h[i] = sum_j f[i + j] g[m - 1 - j].
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

// raw::mul into out of n words that start as P - 1: every word must be written.
Vec raw_mul(const Vec& a, const Vec& b, std::size_t n) {
    Vec out(n, P - 1);
    fps::raw::mul<P>(a, b, out);
    return out;
}

// mul and raw::mul for outputs around the product's size, and square for equal factors.
void check_mul(const Vec& a, const Vec& b) {
    const Vec full = naive_product(a, b);
    const Poly f = poly(a), g = poly(b);
    const std::size_t s = full.size();
    for (const std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(2), std::min(a.size(), b.size()),
                                s > 0 ? s - 1 : 0, s, s + 1, s + 70, 3 * s + 100, std::size_t(rng() % (s + 2))}) {
        dirty_heap(n);
        CHECK(vec(fps::mul(f, g, n)) == first(full, n), a.size(), b.size(), n);
        CHECK(raw_mul(a, b, n) == first(full, n), a.size(), b.size(), n);
    }
    if (a == b) CHECK(vec(fps::square(f)) == full, a.size());
}

void test_mul() {
    for (const std::size_t n : kSizes)
        for (const std::size_t m : kSizes) check_mul(random_vec(n, int(rng() % 4)), random_vec(m, int(rng() % 4)));
    // Both sides of small::beats_ntt, which differs per target, and the NTT path; each truncated.
    for (const auto& [n, m] : std::vector<std::pair<std::size_t, std::size_t>>{
             {24, 24}, {33, 32}, {64, 64}, {97, 96}, {128, 129}, {443, 444}, {1024, 1024}, {2000, 1500},
             {3000, 16}, {3000, 64}, {96, 3000}, {3000, 128}, {33, 9000}})
        for (const int kind : {0, 3}) check_mul(random_vec(n, kind), random_vec(m, kind));
    // Shifts: a factor with one nonzero term, or none; truncated anywhere.
    for (const std::size_t n : {std::size_t(1), std::size_t(64), std::size_t(3000)})
        for (const std::size_t k : {std::size_t(0), std::size_t(1), std::size_t(5), std::size_t(700)}) {
            const Vec a = random_vec(n);
            for (const u32 c : {u32(0), u32(1), u32(rng() % P)}) {
                Vec t(k + 1, 0);
                t[k] = c;
                check_mul(a, t);
                check_mul(t, a);
            }
        }
    // Long factors: mul against the exact product, which test_polynomial checks.
    for (int lg = 11; lg <= 19; lg += 4) {
        const std::size_t h = std::size_t(1) << (lg - 1);
        const Poly f = poly(random_vec(h + 1)), g = poly(random_vec(h));
        const Vec full = vec(f * g);
        for (const std::size_t n : {h / 3, h, h + 1, 2 * h - 1, 2 * h, 2 * h + 1, 4 * h})
            CHECK(vec(fps::mul(f, g, n)) == first(full, n) && raw_mul(vec(f), vec(g), n) == first(full, n), lg, n);
        CHECK(vec(fps::square(f)) == vec(f * f), lg);
    }
}

void test_square() {
    for (const std::size_t n : kSizes)
        for (const int kind : {0, 1, 2, 3}) {
            const Vec a = random_vec(n, kind);
            dirty_heap(n ? 2 * n - 1 : 0);
            CHECK(vec(fps::square(poly(a))) == naive_product(a, a), n, kind);
        }
    for (const std::size_t n : {std::size_t(24), std::size_t(64), std::size_t(104), std::size_t(105), std::size_t(300),
                                std::size_t(2049)}) {
        const Vec a = random_vec(n);
        CHECK(vec(fps::square(poly(a))) == naive_product(a, a), n);
    }
    // c x^k
    for (const std::size_t k : {std::size_t(0), std::size_t(3), std::size_t(1000)}) {
        Vec t(k + 1, 0);
        t[k] = u32(rng() % P);
        CHECK(vec(fps::square(poly(t))) == naive_product(t, t), k);
    }
}

void check_middle(const Vec& f, const Vec& g) {
    dirty_heap(f.size() - g.size() + 1);
    const Vec h = vec(fps::middle_product(poly(f), poly(g)));
    CHECK(h == naive_middle(f, g), f.size(), g.size());
    const Vec full = naive_product(f, g);
    CHECK(h == Vec(full.begin() + std::ptrdiff_t(g.size() - 1), full.begin() + std::ptrdiff_t(f.size())), f.size(),
          g.size());
}

void test_middle() {
    for (const std::size_t n : kSizes)
        for (const std::size_t m : kSizes)
            if (m >= 1 && m <= n) check_middle(random_vec(n, int(rng() % 4)), random_vec(m, int(rng() % 4)));
    // Both sides of small::beats_ntt and the NTT path.
    for (const auto& [n, m] : std::vector<std::pair<std::size_t, std::size_t>>{
             {4096, 32}, {4096, 128}, {4096, 176}, {4096, 4064}, {1024, 512}, {1025, 513}, {2048, 1024},
             {2049, 1025}, {3000, 1}, {5000, 4000}, {104, 104}, {128, 64}, {200, 100}})
        for (const int kind : {0, 3}) check_middle(random_vec(n, kind), random_vec(m, kind));
    // Long: against the exact product.
    for (const auto& [n, m] : std::vector<std::pair<std::size_t, std::size_t>>{
             {std::size_t(1) << 16, std::size_t(1) << 15}, {(std::size_t(1) << 18) + 1, 3000}, {100000, 99999}}) {
        const Poly f = poly(random_vec(n)), g = poly(random_vec(m));
        const Vec full = vec(f * g);
        CHECK(vec(fps::middle_product(f, g)) == Vec(full.begin() + std::ptrdiff_t(m - 1), full.begin() + std::ptrdiff_t(n)),
              n, m);
    }
}

Vec naive_product_all(const std::vector<Vec>& fs) {
    Vec p{1};
    for (const Vec& f : fs) p = naive_product(p, f);
    return p;
}

void check_product(const std::vector<Vec>& fs) {
    std::vector<Poly> polys;
    for (const Vec& f : fs) polys.push_back(poly(f));
    const Vec want = naive_product_all(fs);
    CHECK(vec(fps::product(polys)) == want, fs.size());
    CHECK(vec(fps::product(std::span<const Poly>(polys))) == want, fs.size());
}

void test_product() {
    check_product({});
    CHECK(vec(fps::product(std::vector<Poly>{})) == Vec{1});
    for (const std::size_t n : {0, 1, 2, 5, 64})
        check_product({random_vec(n)});
    for (const std::size_t count : {2, 3, 4, 5, 7, 8, 9, 16, 17, 100}) {
        std::vector<Vec> fs;
        for (std::size_t i = 0; i < count; ++i) fs.push_back(random_vec(1 + rng() % 40, int(rng() % 4)));
        check_product(fs);
    }
    // Unbalanced sizes: one long factor among short ones, at each place; sizes 1 (constants).
    for (const std::size_t at : {0, 3, 9}) {
        std::vector<Vec> fs;
        for (std::size_t i = 0; i < 10; ++i) fs.push_back(random_vec(i == at ? 1500 : 1 + rng() % 3));
        check_product(fs);
    }
    check_product({random_vec(1), random_vec(1), random_vec(1)});
    check_product({random_vec(1000), random_vec(1000)});
    // An empty factor makes the product empty.
    check_product({random_vec(5), {}, random_vec(7)});
    // 2^k + 1 linear factors x - r_i: the product at random points.
    for (const std::size_t count : {std::size_t(1025), std::size_t(1) << 14}) {
        std::vector<Poly> fs;
        Vec roots(count);
        for (u32& r : roots) {
            r = u32(rng() % P);
            fs.push_back(Poly{(P - r) % P, 1});
        }
        const Poly f = fps::product(fs);
        CHECK(f.size() == count + 1 && f[count] == 1, count);
        for (int t = 0; t < 3; ++t) {
            const u32 z = u32(rng() % P);
            u32 want = 1;
            for (const u32 r : roots) want = mul(want, (z + P - r) % P);
            CHECK(f(z) == want, count);
        }
    }
}

// Results of the NTT path hold more memory than their size: copies, resizes and sums still see
// size() coefficients only.
void test_results() {
    const Vec a = random_vec(700), b = random_vec(600);
    const Vec want = naive_product(a, b);
    Poly h = fps::mul(poly(a), poly(b), 1299);
    CHECK(vec(h) == want);
    const Poly copy = h;
    CHECK(vec(copy) == want);
    h.resize(1000);
    h.resize(1400);
    CHECK(vec(h) == first(first(want, 1000), 1400));
    h = poly(a) * poly(b);
    h += Poly(1310);
    CHECK(vec(h) == first(want, 1310));
    h = fps::middle_product(poly(want), poly(b));
    CHECK(h.size() == 700 && vec(h) == naive_middle(want, b));
}

void test_errors() {
    const Poly f{1, 2, 3};
    CHECK(throws<std::domain_error>([&] { (void)fps::middle_product(f, Poly()); }));
    CHECK(throws<std::domain_error>([&] { (void)fps::middle_product(Poly(), Poly()); }));
    CHECK(throws<std::domain_error>([&] { (void)fps::middle_product(f, Poly{1, 2, 3, 4}); }));
    CHECK(!throws<std::domain_error>([&] { (void)fps::middle_product(f, f); }));
    // raw::mul: out overlapping a or b, at its start, inside or at its end; adjacent is fine.
    Vec w = random_vec(100);
    const std::span<u32> all(w);
    const std::span<const u32> a = all.subspan(10, 20);
    CHECK(throws<std::domain_error>([&] { fps::raw::mul<P>(a, a, all.subspan(10, 20)); }));
    CHECK(throws<std::domain_error>([&] { fps::raw::mul<P>(a, all.subspan(60, 5), all.subspan(29, 30)); }));
    CHECK(throws<std::domain_error>([&] { fps::raw::mul<P>(all.subspan(60, 5), a, all.subspan(0, 11)); }));
    CHECK(throws<std::domain_error>([&] { fps::raw::mul<P>(all.subspan(60, 5), a, all.subspan(15, 2)); }));
    const Vec want = first(naive_product(Vec(a.begin(), a.end()), Vec(a.begin(), a.end())), 39);
    fps::raw::mul<P>(a, a, all.subspan(30, 39));
    CHECK(Vec(w.begin() + 30, w.begin() + 69) == want);
    fps::raw::mul<P>(a, a, all.subspan(0, 0));
    // NTT products past 2^25 coefficients, before any memory of that size is written; the factors
    // are not short.
    Poly big(std::size_t(1) << 25), g(4096);
    big[0] = big[1] = g[0] = g[1] = 1;
    CHECK(throws<std::length_error>([&] { (void)fps::mul(big, g, big.size()); }));
    CHECK(throws<std::length_error>([&] { (void)fps::middle_product(big, g); }));
    CHECK(throws<std::length_error>([&] {
        Vec out(1 << 25);
        fps::raw::mul<P>(big.span(), g.span(), out);
    }));
    big.resize((std::size_t(1) << 24) + 1);
    CHECK(throws<std::length_error>([&] { (void)fps::square(big); }));
    // Cut to n first: the same factors fit.
    CHECK(vec(fps::mul(big, g, 3)) == (Vec{1, 2, 1}));
    // Past 2^25 with a short factor: small::multiply, no error.
    big.resize(std::size_t(1) << 25);
    const Poly h = fps::mul(big, Poly{1, 1}, big.size());
    CHECK(h.size() == big.size() && h.degree() == 2 && h[0] == 1 && h[1] == 2 && h[2] == 1);
}

void test_threads() {
    std::vector<std::thread> threads;
    std::vector<int> ok(4, 0);
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([t, &ok] {
            std::mt19937_64 local{u64(t)};
            bool good = true;
            for (int r = 0; r < 3; ++r) {
                Vec a(2000), b(1500);
                for (u32& c : a) c = u32(local() % P);
                for (u32& c : b) c = u32(local() % P);
                const Vec full = naive_product(a, b);
                good &= vec(fps::mul(poly(a), poly(b), 3000)) == first(full, 3000);
                good &= raw_mul(a, b, 3499) == full;
                good &= vec(fps::square(poly(b))) == naive_product(b, b);
                good &= vec(fps::middle_product(poly(a), poly(b))) == naive_middle(a, b);
            }
            ok[std::size_t(t)] = good;
        });
    for (std::thread& t : threads) t.join();
    for (const int good : ok) CHECK(good);
}

}  // namespace

int main() {
    test_mul();
    test_square();
    test_middle();
    test_product();
    test_results();
    test_errors();
    test_threads();
    if (failures) std::printf("%d failures\n", failures);
    return failures != 0;
}
