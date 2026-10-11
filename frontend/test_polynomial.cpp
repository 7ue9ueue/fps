// Tests of fps/polynomial.hpp against O(n^2) references: the class, the sums and scalings, and
// f * g on each path (shift, small::multiply, ntt::Convolution), on edge sizes and random inputs.
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "fps/fps.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = 998244353;
using Poly = fps::Polynomial<P>;

int failures = 0;

void check(bool ok, const char* what, int line, u64 a = 0, u64 b = 0) {
    if (ok) return;
    if (++failures <= 20)
        std::printf("FAIL line %d: %s (%llu, %llu)\n", line, what, (unsigned long long)a, (unsigned long long)b);
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

std::mt19937_64 rng(16);

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

// sign 1: a + b, sign -1: a - b, zero-extended to the longer.
Vec naive_sum(const Vec& a, const Vec& b, int sign) {
    Vec c(std::max(a.size(), b.size()));
    for (std::size_t i = 0; i < c.size(); ++i) {
        const u64 x = i < a.size() ? a[i] : 0, y = i < b.size() ? b[i] : 0;
        c[i] = u32((x + (sign > 0 ? y : P - y)) % P);
    }
    return c;
}

Vec naive_scale(const Vec& a, u32 c) {
    Vec b(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) b[i] = mul(a[i], c);
    return b;
}

u32 horner(const Vec& f, u64 x) {
    u64 r = 0;
    for (std::size_t i = f.size(); i-- > 0;) r = (r * (x % P) + f[i]) % P;
    return u32(r);
}

void test_class() {
    CHECK(Poly().size() == 0 && Poly().degree() == -1);
    CHECK(Poly(5).size() == 5 && Poly(5).degree() == -1 && vec(Poly(5)) == Vec(5, 0));
    const Poly f{1, 2, 0, 0};
    CHECK(f.size() == 4 && f.degree() == 1 && f[1] == 2 && f.coeff(3) == 0 && f.coeff(100) == 0);
    for (const std::size_t k : kSizes) {
        const Poly m = Poly::x(k);
        Vec want(k + 1, 0);
        want[k] = 1;
        CHECK(vec(m) == want && m.degree() == int(k), k);
    }
    CHECK(vec(Poly::x()) == (Vec{0, 1}));
    const Vec c = random_vec(100);
    CHECK(vec(poly(c)) == c);
    CHECK(vec(Poly(std::span<const u32>(c.data(), 0))).empty());
    CHECK(vec(Poly{P - 1, 0}) == (Vec{P - 1, 0}));
    // Copies, moves, self-assignment.
    Poly copy = poly(c), moved = copy;
    CHECK(vec(copy) == c && vec(moved) == c);
    const Poly taken = std::move(moved);
    moved = copy;
    copy = std::as_const(copy);
    CHECK(vec(taken) == c && vec(moved) == c && vec(copy) == c);
    moved = Poly{4, 5};
    CHECK(vec(moved) == (Vec{4, 5}));

    Poly g{1, 2, 3};
    g[1] = 7;
    g.span()[2] = 9;
    CHECK(vec(g) == (Vec{1, 7, 9}) && g.span().size() == 3 && std::as_const(g).span()[0] == 1);
    g.resize(5);
    CHECK(vec(g) == (Vec{1, 7, 9, 0, 0}) && g.degree() == 2);
    g.resize(2);
    CHECK(vec(g) == (Vec{1, 7}));
    g.resize(0);
    CHECK(g.size() == 0 && g.degree() == -1);

    for (const std::size_t n : kSizes) {
        const Vec a = random_vec(n, 1);
        int degree = int(n) - 1;
        while (degree >= 0 && a[std::size_t(degree)] == 0) --degree;
        CHECK(poly(a).degree() == degree, n);
        for (const std::size_t k : {std::size_t(0), std::size_t(1), n / 2, n, n + 3}) {
            Vec t(k, 0), r(k, 0);
            for (std::size_t i = 0; i < std::min(k, n); ++i) t[i] = a[i], r[k - 1 - i] = a[i];
            const Poly f = poly(a);
            dirty_heap(k);
            CHECK(vec(f.truncate(k)) == t, n, k);
            dirty_heap(k);
            CHECK(vec(f.reverse(k)) == r, n, k);
        }
        const Poly f = poly(a);
        for (const u64 x : {u64(0), u64(1), u64(2), u64(P - 1), u64(P), u64(P + 5), u64(~u32(0)), rng() % P})
            CHECK(f(u32(x)) == horner(a, u32(x)), n, x);
    }
}

void test_errors() {
    CHECK(throws<std::domain_error>([] { Poly{1, P}; }));
    CHECK(throws<std::domain_error>([] { Poly{~u32(0)}; }));
    const Vec bad = {0, 1, 2, P + 1};
    CHECK(throws<std::domain_error>([&] { (void)poly(bad); }));
    // An NTT product past the backend's 2^25 coefficients; the factors are not short.
    Poly f(std::size_t(1) << 25), g(4096);
    f[0] = f[1] = g[0] = g[1] = 1;
    CHECK(throws<std::length_error>([&] { (void)(f * g); }));
}

// Scalars are integers of at most 64 bits: wider ones (__int128) would be cut, so they do not
// compile.
template <class T>
constexpr bool takes_scalar = requires(const Poly& f, Poly& g, T c) {
    f + c, c + f, f - c, c - f, f * c, c * f, g += c, g -= c, g *= c;
};
static_assert(takes_scalar<char> && takes_scalar<int> && takes_scalar<std::int64_t> && takes_scalar<u64>);
static_assert(!takes_scalar<__int128> && !takes_scalar<unsigned __int128> && !takes_scalar<double>);

template <class T>
void check_scalar(const Vec& a, T c) {
    const u32 r = u32(((__int128)c % P + P) % P);
    const Poly f = poly(a);
    Vec plus = a.empty() ? Vec{0} : a, minus = plus, from = plus;
    plus[0] = u32((u64(plus[0]) + r) % P);
    minus[0] = u32((u64(minus[0]) + P - r) % P);
    from = naive_sum(Vec{r}, a, -1);
    CHECK(vec(f + c) == plus && vec(c + f) == plus, a.size(), r);
    CHECK(vec(f - c) == minus && vec(c - f) == from, a.size(), r);
    CHECK(vec(f * c) == naive_scale(a, r) && vec(c * f) == naive_scale(a, r), a.size(), r);
    CHECK(vec(Poly(f) + c) == plus && vec(c - Poly(f)) == from && vec(Poly(f) * c) == naive_scale(a, r));
    Poly g = f;
    g += c;
    CHECK(vec(g) == plus);
    g = f;
    g -= c;
    CHECK(vec(g) == minus);
    g = f;
    g *= c;
    CHECK(vec(g) == naive_scale(a, r));
}

void test_sums_and_scalars() {
    for (const std::size_t n : kSizes)
        for (const std::size_t m : {std::size_t(0), std::size_t(1), std::size_t(8), n / 2, n, n + 1, 2 * n + 9}) {
            const Vec a = random_vec(n, int(rng() % 4)), b = random_vec(m, int(rng() % 4));
            const Poly f = poly(a), g = poly(b);
            CHECK(vec(f + g) == naive_sum(a, b, 1), n, m);
            CHECK(vec(f - g) == naive_sum(a, b, -1), n, m);
            CHECK(vec(Poly(f) + g) == naive_sum(a, b, 1), n, m);
            CHECK(vec(Poly(f) - g) == naive_sum(a, b, -1), n, m);
            Poly h = f;
            h += g;
            CHECK(vec(h) == naive_sum(a, b, 1), n, m);
            h = f;
            h -= g;
            CHECK(vec(h) == naive_sum(a, b, -1), n, m);
        }
    for (const std::size_t n : kSizes) {
        const Vec a = random_vec(n, int(rng() % 4));
        CHECK(vec(-poly(a)) == naive_sum({}, a, -1), n);
        for (const u32 c : {u32(0), u32(1), u32(2), P - 1, P, ~u32(0), u32(rng() % P)}) check_scalar(a, c);
        for (const int c : {-1, -2, 3, -int(P), std::numeric_limits<int>::min()}) check_scalar(a, c);
        for (const std::int64_t c : {std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max()})
            check_scalar(a, c);
        check_scalar(a, std::numeric_limits<u64>::max());
    }
    // A left operand shorter than g: +=, -= and the rvalue forms build a new result.
    for (const std::size_t n : {0, 1, 5, 64, 1000})
        for (const std::size_t m : {n + 1, n + 7, 2 * n + 33}) {
            const Vec a = random_vec(n), b = random_vec(m);
            const Poly g = poly(b);
            Poly h = poly(a);
            dirty_heap(m);
            h += g;
            CHECK(vec(h) == naive_sum(a, b, 1), n, m);
            h = poly(a);
            dirty_heap(m);
            h -= g;
            CHECK(vec(h) == naive_sum(a, b, -1), n, m);
            dirty_heap(m);
            CHECK(vec(poly(a) + g) == naive_sum(a, b, 1), n, m);
            dirty_heap(m);
            CHECK(vec(poly(a) - g) == naive_sum(a, b, -1), n, m);
        }
    // Aliasing: f with itself.
    const Vec a = random_vec(77);
    Poly f = poly(a);
    f += f;
    CHECK(vec(f) == naive_sum(a, a, 1));
    f -= f;
    CHECK(vec(f) == Vec(77, 0));
    // DESIGN.md's forms.
    const Poly x = Poly::x();
    CHECK(vec(1 - 4 * x) == (Vec{1, P - 4}));
    CHECK(vec(1 - x - x * x) == naive_sum(naive_sum({1}, {0, 1}, -1), {0, 0, 1}, -1));
    CHECK(vec(1 - x - x * x) == (Vec{1, P - 1, P - 1}));
    CHECK(vec(Poly{3} + x * x * x - x) == (Vec{3, P - 1, 0, 1}));
    CHECK(vec(1 - 4 * x * x + Poly::x(4)) == (Vec{1, 0, P - 4, 0, 1}));
}

void check_product(const Vec& a, const Vec& b) {
    const Poly f = poly(a), g = poly(b);
    const Vec want = naive_product(a, b);
    CHECK(vec(f * g) == want, a.size(), b.size());
    Poly h = f;
    h *= g;
    CHECK(vec(h) == want, a.size(), b.size());
}

// f * g at random points, for products too long for the O(n m) reference.
void check_long_product(std::size_t n, std::size_t m) {
    const Poly f = poly(random_vec(n)), g = poly(random_vec(m));
    const Poly h = f * g;
    CHECK(h.size() == n + m - 1, n, m);
    for (int t = 0; t < 4; ++t) {
        const u32 z = u32(rng() % P);
        CHECK(h(z) == mul(f(z), g(z)), n, m);
    }
}

void test_products() {
    for (const std::size_t n : kSizes)
        for (const std::size_t m : kSizes) check_product(random_vec(n, int(rng() % 4)), random_vec(m, int(rng() % 4)));
    // Around the cutoffs of small::beats_ntt, which differ per target: both short (16 to 128
    // each) and one short (4 to 128 against thousands); the largest sums on both paths.
    for (const auto& [n, m] : std::vector<std::pair<std::size_t, std::size_t>>{
             {24, 24}, {33, 32}, {48, 47}, {64, 64}, {97, 96}, {128, 129}, {3000, 4}, {3000, 16}, {3000, 24},
             {32, 3000}, {3000, 48}, {3000, 64}, {96, 3000}, {3000, 128}, {9000, 32}, {33, 9000}})
        for (const int kind : {0, 3}) check_product(random_vec(n, kind), random_vec(m, kind));
    // Shifts: a factor with one nonzero term, or none.
    for (const std::size_t n : {std::size_t(1), std::size_t(64), std::size_t(3000)})
        for (const std::size_t k : {std::size_t(0), std::size_t(1), std::size_t(5), std::size_t(700)}) {
            const Vec a = random_vec(n);
            for (const u32 c : {u32(0), u32(1), u32(rng() % P)}) {
                Vec t(k + 1, 0);
                t[k] = c;
                check_product(a, t);
                check_product(t, a);
            }
            check_product(a, Vec(k + 1, 0));
        }
    const Poly x = Poly::x();
    const Poly f = poly(random_vec(1000));
    CHECK(vec(f * Poly::x(3)) == naive_product(vec(f), vec(Poly::x(3))));
    CHECK(vec(x * f) == naive_product(vec(x), vec(f)));
    Poly g = f;
    g *= g;
    CHECK(vec(g) == naive_product(vec(f), vec(f)));
    // ntt::Convolution's lengths: 2^lg - 1 and 2^lg coefficients.
    for (int lg = 10; lg <= 20; ++lg) {
        const std::size_t half = std::size_t(1) << (lg - 1);
        check_long_product(half, half);
        check_long_product(half + 1, half);
    }
    check_long_product(1 << 19, 3);
    check_long_product(5, 1 << 18);
    check_long_product(std::size_t(1) << 25, 2);  // past the NTT's 2^25: small::multiply
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
                good &= vec(poly(a) * poly(b)) == naive_product(a, b);
            }
            ok[std::size_t(t)] = good;
        });
    for (std::thread& t : threads) t.join();
    for (const int good : ok) CHECK(good);
}

}  // namespace

int main() {
    test_class();
    test_errors();
    test_sums_and_scalars();
    test_products();
    test_threads();
    if (failures) std::printf("%d failures\n", failures);
    return failures != 0;
}
