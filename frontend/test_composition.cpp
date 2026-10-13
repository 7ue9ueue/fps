// Tests of fps/composition.hpp: compose on every path (a constant, a linear and a general inner
// function, with g[0] = 0 and not) against Horner's rule mod x^n on edge sizes and random inputs;
// compositional_inverse by f(g) = g(f) = x; identities at large sizes; threads and every error.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "fps/fps.hpp"

// Under ASan, every allocation starts filled with 0xbe bytes, so a result word the library
// forgets to write shows up on any allocator.
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

// f throws an Exception whose message contains where.
template <class Exception, class F>
bool throws_from(const char* where, F f) {
    try {
        f();
    } catch (const Exception& e) {
        return std::string(e.what()).find(where) != std::string::npos;
    }
    return false;
}

template <class Exception, class F>
bool throws(F f) {
    try {
        f();
    } catch (const Exception&) {
        return true;
    }
    return false;
}

std::mt19937_64 rng(26);

// 0, 1, powers of two and their neighbours.
const std::vector<std::size_t> kSizes = {0,  1,  2,  3,   4,   5,   7,   8,   9,   15,  16,   17,   31,  32,  33,
                                         63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025};

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 add(u32 a, u32 b) { return (a + b) % P; }
u32 nonzero() { return u32(1 + rng() % (P - 1)); }

// Kinds: 0 random, 1 mostly zeros, 2 all P - 1.
Vec random_vec(std::size_t n, int kind = 0) {
    Vec a(n);
    for (u32& c : a) {
        const u64 r = rng();
        c = kind == 2 ? P - 1 : kind == 1 && r % 4 ? 0 : u32(r % P);
    }
    return a;
}

Vec vec(const Poly& f) { return Vec(f.span().begin(), f.span().end()); }
Poly poly(const Vec& a) { return Poly(std::span<const u32>(a)); }
bool operator==(const Poly& a, const Poly& b) { return std::ranges::equal(a.span(), b.span()); }

// a b mod x^n, schoolbook.
Vec mul_mod(const Vec& a, const Vec& b, std::size_t n) {
    Vec c(n);
    for (std::size_t i = 0; i < a.size() && i < n; ++i)
        for (std::size_t j = 0; j < b.size() && i + j < n; ++j) c[i + j] = add(c[i + j], mul(a[i], b[j]));
    return c;
}

// f(g) mod x^n by Horner's rule, schoolbook: O(f.size() n min(n, g.size())).
Vec naive_compose(const Vec& f, const Vec& g, std::size_t n) {
    Vec h(n);
    for (std::size_t i = f.size(); i-- > 0;) {
        h = mul_mod(h, g, n);
        if (n) h[0] = add(h[0], f[i]);
    }
    return h;
}

// The same, Horner on fps::mul (tested in test_products.cpp): for f of up to a few thousand terms.
Poly horner(const Poly& f, const Poly& g, std::size_t n) {
    Poly h(n);
    for (std::size_t i = f.size(); i-- > 0;) {
        h = fps::mul(h, g, n);
        if (n) h[0] = add(h[0], f[i]);
    }
    return h;
}

Vec x_mod(std::size_t n) {
    Vec x(n);
    if (n > 1) x[1] = 1;
    return x;
}

// Inner functions of every shape, for precision n: the shapes are those of g mod x^n.
std::vector<Vec> inner_functions(std::size_t n) {
    std::vector<Vec> gs;
    gs.push_back({});                                       // 0
    gs.push_back({nonzero()});                              // a constant
    gs.push_back({nonzero(), 0, 0});                        // a constant with trailing zeros
    gs.push_back({0, 1});                                   // x
    gs.push_back({0, nonzero()});                           // b x
    gs.push_back({nonzero(), 1});                           // x + a
    gs.push_back({nonzero(), nonzero(), 0, 0});             // a + b x
    gs.push_back({P - 1, P - 1});                           // -1 - x
    Vec high(n + 2);                                        // a + b x + c x^n: linear mod x^n
    high[0] = nonzero(), high[1] = nonzero(), high[n] = nonzero(), high[n + 1] = nonzero();
    gs.push_back(high);
    gs.push_back({0, 0, 1});                                // x^2
    gs.push_back({nonzero(), 0, nonzero()});                // a + c x^2
    gs.push_back({1, 1});                                   // a = 1, b = 1
    gs.push_back({1, 0, nonzero()});                        // a = 1
    gs.push_back({P - 1, nonzero(), nonzero()});            // a = -1
    for (int kind = 0; kind < 3; ++kind) {
        for (const std::size_t size : {std::size_t(3), n, n + 3, std::size_t(rng() % (n + 4))}) {
            Vec g = random_vec(size, kind);
            gs.push_back(g);
            if (!g.empty()) g[0] = 0;
            gs.push_back(g);
        }
    }
    return gs;
}

void test_compose_small() {
    for (const std::size_t n : kSizes) {
        if (n > 65) continue;
        for (const Vec& g : inner_functions(n)) {
            for (const std::size_t m : {std::size_t(0), std::size_t(1), n ? n - 1 : 0, n + 3, std::size_t(rng() % 100)}) {
                const Vec f = random_vec(m);
                CHECK(vec(fps::compose(poly(f), poly(g), n)) == naive_compose(f, g, n), n, m, g.size());
            }
            for (int kind = 0; kind < 3; ++kind) {
                const Vec f = random_vec(n, kind);
                CHECK(vec(fps::compose(poly(f), poly(g), n)) == naive_compose(f, g, n), n, kind, g.size());
            }
        }
    }
}

void test_compose_large() {
    for (const std::size_t n : kSizes) {
        if (n <= 65) continue;
        for (const Vec& g : inner_functions(n)) {
            // Short f against the schoolbook; f of n + 3 against Horner on fps::mul.
            const Vec f = random_vec(1 + rng() % 6);
            CHECK(vec(fps::compose(poly(f), poly(g), n)) == naive_compose(f, g, n), n, f.size(), g.size());
            const Poly F = poly(random_vec(n + 3));
            CHECK(fps::compose(F, poly(g), n) == horner(F, poly(g), n), n, g.size());
        }
    }
}

void test_defaults() {
    for (const std::size_t m : {std::size_t(0), std::size_t(1), std::size_t(5), std::size_t(40)})
        for (const std::size_t k : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(3), std::size_t(40), std::size_t(77)}) {
            const Poly f = poly(random_vec(m)), g = poly(random_vec(k));
            const Poly h = fps::compose(f, g);
            CHECK(h.size() == std::max(m, k), m, k);
            CHECK(vec(h) == naive_compose(vec(f), vec(g), std::max(m, k)), m, k);
        }
    // A constant g evaluates all of f, whatever n.
    const Poly f = poly(random_vec(1000));
    const u32 c = nonzero();
    const Poly h = fps::compose(f, Poly{c}, 7);
    CHECK(h.size() == 7 && h[0] == f(c) && h.degree() <= 0);
    CHECK(fps::compose(f, Poly{c}, 0).size() == 0);
    CHECK(fps::compose(Poly{}, poly(random_vec(9)), 5) == Poly(5));
}

// Identities at large n, every path: f(g) for f the series of exp and of log(1 + y); a Taylor
// shift there and back; a scaling there and back; long f with short n.
void test_identities() {
    for (const std::size_t n : {std::size_t(1) << 14, (std::size_t(1) << 14) + 1, std::size_t(100003)}) {
        const fps::Factorials<P> fact(n + 1);
        Poly e(n), l(n);
        for (std::size_t i = 0; i < n; ++i) e[i] = fact.inverse(i);  // exp(y)
        for (std::size_t i = 1; i < n; ++i)                          // log(1 + y) = sum (-1)^(i+1) y^i / i
            l[i] = mul(i % 2 ? 1 : P - 1, mul(fact[i - 1], fact.inverse(i)));
        Poly g = poly(random_vec(n));
        g[0] = 0;
        CHECK(fps::compose(e, g) == fps::exp(g), n);
        Poly one_plus_g = g;
        one_plus_g[0] = 1;
        CHECK(fps::compose(l, g) == fps::log(one_plus_g), n);
        // g[0] = a != 0, through the Taylor shift: f = 1 + y + ... + y^(n-1) gives
        // f(g) (1 - g) = 1 - g^n mod x^n.
        const u32 a = nonzero();
        g[0] = a;
        const Poly geometric = poly(Vec(n, 1));
        const Poly h = fps::compose(geometric, g, n);
        const Poly lhs = fps::mul(h, 1 - g, n);
        const Poly rhs = 1 - fps::pow(g, n, n);
        CHECK(lhs == rhs, n);
        const Poly x = Poly::x(), p = poly(random_vec(n));
        CHECK(fps::compose(fps::compose(p, x + a), x - a) == p, n);
        const u32 b = nonzero();
        CHECK(fps::compose(fps::compose(p, b * x), fps::mod_inverse<P>(b) * x) == p, n);
        CHECK(fps::compose(fps::compose(p, a + b * x), (x - a) * fps::mod_inverse<P>(b)) == p, n);
        // Short n of a long f: the Taylor shift's middle product with fewer outputs.
        CHECK(fps::compose(p, a + b * x, 37) == fps::compose(p, a + b * x).truncate(37), n);
        CHECK(fps::compose(p, g, 37) == fps::compose(p, g).truncate(37), n);
    }
}

void test_inverse() {
    for (const std::size_t n : kSizes)
        for (int kind = 0; kind < 3; ++kind)
            for (const std::size_t m : {std::size_t(2), std::size_t(3), n, n + 3}) {
                if (m < 2) continue;
                Vec f = random_vec(m, kind);
                f[0] = 0;
                if (f[1] == 0) f[1] = nonzero();
                const Poly g = fps::compositional_inverse(poly(f), n);
                CHECK(g.size() == n);
                if (n <= 129) {
                    CHECK(naive_compose(f, vec(g), n) == x_mod(n), n, m, kind);
                    CHECK(naive_compose(vec(g), f, n) == x_mod(n), n, m, kind);
                } else {
                    CHECK(vec(horner(poly(f), g, n)) == x_mod(n), n, m, kind);
                    CHECK(vec(horner(g, poly(f), n)) == x_mod(n), n, m, kind);
                }
            }
    // Closed forms: x / (1 - x) has inverse x / (1 + x); x e^x (Lambert): [x^k] = (-k)^(k-1) / k!.
    const std::size_t n = 2000;
    const Poly x = Poly::x();
    const Poly g = fps::compositional_inverse(fps::mul(x, fps::inverse(1 - x, n), n));
    for (std::size_t k = 1; k < n; ++k) CHECK(g[k] == (k % 2 ? 1 : P - 1), k);
    const fps::Factorials<P> fact(n);
    Poly lambert(n);
    for (std::size_t k = 1; k < n; ++k) lambert[k] = fact.inverse(k - 1);
    const Poly w = fps::compositional_inverse(lambert);
    for (std::size_t k = 1; k < n; ++k)
        CHECK(w[k] == mul(fps::mod_pow<P>(P - u32(k), k - 1), fact.inverse(k)), k);
    // The default n, and n past f.size().
    CHECK(fps::compositional_inverse(Poly{0, 3}).size() == 2);
    CHECK(fps::compositional_inverse(Poly{0, 3}, 6) == poly(Vec{0, fps::mod_inverse<P>(3), 0, 0, 0, 0}));
    for (const std::size_t big : {std::size_t(1) << 14, (std::size_t(1) << 14) + 1, std::size_t(100003)}) {
        Poly f = poly(random_vec(big));
        f[0] = 0, f[1] = nonzero();
        const Poly gi = fps::compositional_inverse(f);
        CHECK(fps::compose(f, gi) == x.truncate(big), big);
        CHECK(fps::compose(gi, f) == x.truncate(big), big);
    }
}

void test_threads() {
    std::vector<Poly> fs, gs, want;
    for (int t = 0; t < 4; ++t) {
        fs.push_back(poly(random_vec(3000)));
        Poly g = poly(random_vec(3000));
        if (t % 2) g[0] = 0;
        gs.push_back(g);
        want.push_back(fps::compose(fs.back(), g));
    }
    std::vector<int> ok(4);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&, t] {
            bool good = true;
            for (int r = 0; r < 3; ++r) good &= fps::compose(fs[t], gs[t]) == want[t];
            Poly f = fs[t];
            f[0] = 0, f[1] = 1 + u32(t);
            good &= fps::compose(f, fps::compositional_inverse(f)) == Poly::x().truncate(3000);
            ok[t] = good;
        });
    for (std::thread& t : threads) t.join();
    for (int t = 0; t < 4; ++t) CHECK(ok[t], t);
}

void test_errors() {
    CHECK(throws<std::domain_error>([] { (void)fps::compositional_inverse(Poly{1, 1}); }));
    CHECK(throws<std::domain_error>([] { (void)fps::compositional_inverse(Poly{1, 1}, 1); }));
    CHECK(throws<std::domain_error>([] { (void)fps::compositional_inverse(Poly{0, 0, 1}); }));
    CHECK(throws<std::domain_error>([] { (void)fps::compositional_inverse(Poly{}, 2); }));
    CHECK(throws<std::domain_error>([] { (void)fps::compositional_inverse(Poly{0, 1}, std::size_t(P) + 1); }));
    // Only f mod x^n is read.
    CHECK(fps::compositional_inverse(Poly{1, 1}, 0).size() == 0);
    CHECK(fps::compositional_inverse(Poly{0, 0, 1}, 1) == Poly{0});
    CHECK(fps::compositional_inverse(Poly{}, 1) == Poly{0});
    // Past the kernel's kMaxSize = 2^23: thrown by the frontend's own check, before any work, from
    // n = 2^23 + 1 (test_limits runs n = 2^23). g[0] != 0 too: the check comes before the shift.
    constexpr std::size_t past = fps::composition::kMaxSize + 1;
    static_assert(past == (std::size_t(1) << 23) + 1);
    CHECK(throws_from<std::length_error>("fps::compose", [] { (void)fps::compose(Poly{1, 2}, Poly{0, 1, 1}, past); }));
    CHECK(throws_from<std::length_error>("fps::compose", [] { (void)fps::compose(Poly{1, 2}, Poly{5, 1, 1}, past); }));
    CHECK(throws_from<std::length_error>("fps::compose", [] { (void)fps::compose(Poly{1, 2}, Poly{0, 1, 1}, std::size_t(1) << 24); }));
    CHECK(throws_from<std::length_error>("fps::compositional_inverse",
                                         [] { (void)fps::compositional_inverse(Poly{0, 1, 1}, past); }));
    CHECK(throws_from<std::length_error>("fps::compositional_inverse",
                                         [] { (void)fps::compositional_inverse(Poly{0, 1, 1}, std::size_t(1) << 24); }));
    // An empty f runs no product: n zeros, whatever n.
    const Poly zeros = fps::compose(Poly{}, Poly{0, 1, 1}, std::size_t(1) << 24);
    CHECK(zeros.size() == std::size_t(1) << 24 && zeros.degree() == -1);
    // The Taylor shift: a cyclic product of f.size() + t - 1 words, t = min(n, f.size()); past 2^25
    // it throws before its tables are built.
    Poly top((std::size_t(1) << 24) + 1);
    top[top.size() - 1] = 1;
    CHECK(throws_from<std::length_error>("Taylor shift", [&top] { (void)fps::compose(top, Poly{1, 1}); }));
    // On the general path n <= 2^23 (the check above), so f needs more than 2^25 - 2^23 words.
    Poly longer((std::size_t(1) << 25) - (std::size_t(1) << 23) + 2);
    longer[longer.size() - 1] = 1;
    CHECK(throws_from<std::length_error>("Taylor shift",
                                         [&longer] { (void)fps::compose(longer, Poly{1, 1, 1}, std::size_t(1) << 23); }));
}

// At the limits: the largest sizes the checks above pass, n = 2^23, through every level of the
// kernel. Vector targets at -O2 only (scalar: minutes; ASan: gigabytes more). The kernel keeps its
// work per thread (about 3.3 GiB for compose at n = 2^23), so each part runs in a thread of its
// own, which frees it on exit.
void in_thread(auto f) { std::thread(f).join(); }

void test_limits() {
    if (kSanitized || std::is_same_v<fps::simd::Native, fps::simd::Scalar>) return;
    constexpr std::size_t n = fps::composition::kMaxSize;
    // compose: f = 1 + y + y^2, g of n random words with g[0] != 0 (the Taylor shift, then the
    // kernel): 1 + g + g^2.
    Poly g = poly(random_vec(n));
    g[0] = nonzero();
    in_thread([&g] {
        const Poly want = 1 + g + fps::mul(g, g, n);
        CHECK(fps::compose(Poly{1, 1, 1}, g) == want);
    });
    // compositional_inverse of x + x^2: sum over k >= 1 of (-1)^(k-1) C(k-1) x^k, C the Catalan
    // numbers.
    in_thread([] {
        const fps::Factorials<P> fact(2 * n + 1);
        const Poly g = fps::compositional_inverse(Poly{0, 1, 1}, n);
        CHECK(g.size() == n && g[0] == 0);
        for (const std::size_t k : {std::size_t(1), std::size_t(2), std::size_t(3), std::size_t(1000), n / 2 + 1, n - 1}) {
            const std::size_t j = k - 1;  // C(j) = (2j)! / (j! (j + 1)!)
            const u32 c = mul(fact[2 * j], mul(fact.inverse(j), fact.inverse(j + 1)));
            CHECK(g[k] == (j % 2 ? (c ? P - c : 0) : c), k);
        }
    });
    // The Taylor shift at exactly 2^25 words: f = x^(2^24), n = 2^24, so f(x + 1) = (x + 1)^(2^24),
    // coefficient k C(2^24, k).
    {
        const std::size_t m = (std::size_t(1) << 24) + 1, t = std::size_t(1) << 24;
        const fps::Factorials<P> big(m);
        Poly f(m);
        f[m - 1] = 1;
        const Poly h = fps::compose(f, Poly{1, 1}, t);
        CHECK(h.size() == t);
        for (const std::size_t k : {std::size_t(0), std::size_t(1), std::size_t(12345), t / 2, t - 1})
            CHECK(h[k] == mul(big[m - 1], mul(big.inverse(k), big.inverse(m - 1 - k))), k);
    }
}

}  // namespace

int main() {
    test_compose_small();
    test_compose_large();
    test_defaults();
    test_identities();
    test_inverse();
    test_threads();
    test_errors();
    test_limits();
    if (failures) {
        std::printf("composition: %d failures\n", failures);
        return 1;
    }
    std::printf("composition: all passed\n");
    return 0;
}
