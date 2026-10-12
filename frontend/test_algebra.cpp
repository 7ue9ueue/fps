// Tests of fps/algebra.hpp: gcd against Euclid's algorithm on long division, inverse_mod by f h = 1
// mod m and against that gcd, roots against gcd(f, x^P - x) by naive powers and on polynomials
// built from known roots, factor on products of known irreducibles (exact lists) and on random
// polynomials (product, irreducibility by Rabin's test, order); threads and every error.
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

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = 998244353;
using Poly = fps::Polynomial<P>;
using Factors = std::vector<std::pair<Vec, int>>;

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

std::mt19937_64 rng(25);

// 0, 1, powers of two and their neighbours.
const std::vector<std::size_t> kSizes = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129};

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 sub(u32 a, u32 b) { return a >= b ? a - b : a + P - b; }

u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}

u32 inv(u32 a) { return power(a, P - 2); }
u32 nonzero() { return u32(1 + rng() % (P - 1)); }
u32 random_value() { return u32(rng() % P); }

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

// Degree exactly d.
Vec random_degree(std::size_t d) {
    Vec f = random_vec(d + 1, 0);
    f[d] = nonzero();
    return f;
}

Vec vec(const Poly& f) { return Vec(f.span().begin(), f.span().end()); }
Poly poly(const Vec& a) { return Poly(std::span<const u32>(a)); }

Vec trim(Vec f) {
    while (!f.empty() && f.back() == 0) f.pop_back();
    return f;
}

Vec monic(Vec f) {
    f = trim(std::move(f));
    if (!f.empty()) {
        const u32 c = inv(f.back());
        for (u32& x : f) x = mul(x, c);
    }
    return f;
}

Vec naive_mul(const Vec& a, const Vec& b) {
    if (a.empty() || b.empty()) return {};
    Vec c(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = u32((c[i + j] + u64(a[i]) * b[j]) % P);
    return c;
}

Vec naive_sub(Vec a, const Vec& b) {
    a.resize(std::max(a.size(), b.size()));
    for (std::size_t i = 0; i < b.size(); ++i) a[i] = sub(a[i], b[i]);
    return trim(a);
}

// Long division, b != 0: quotient and remainder, trimmed.
std::pair<Vec, Vec> naive_divmod(Vec a, Vec b) {
    a = trim(a), b = trim(b);
    if (a.size() < b.size()) return {{}, a};
    const std::size_t d = b.size() - 1;
    Vec q(a.size() - d);
    const u32 c = inv(b[d]);
    for (std::size_t i = q.size(); i-- > 0;) {
        q[i] = mul(a[i + d], c);
        for (std::size_t j = 0; j <= d; ++j) a[i + j] = sub(a[i + j], mul(q[i], b[j]));
    }
    a.resize(d);
    return {trim(q), trim(a)};
}

Vec naive_mod(const Vec& a, const Vec& b) { return naive_divmod(a, b).second; }

Vec naive_gcd(Vec a, Vec b) {
    a = trim(a), b = trim(b);
    while (!b.empty()) a = std::exchange(b, naive_mod(a, b));
    return monic(a);
}

// a^e mod m, deg m >= 1.
Vec naive_pow_mod(Vec a, u64 e, const Vec& m) {
    Vec r = naive_mod({1}, m);
    for (a = naive_mod(a, m); e; e >>= 1, a = naive_mod(naive_mul(a, a), m))
        if (e & 1) r = naive_mod(naive_mul(r, a), m);
    return r;
}

const Vec kX = {0, 1};

// Rabin: f (degree d >= 1) is irreducible iff gcd(x^(P^i) - x, f) = 1 for i <= d / 2.
bool irreducible(const Vec& f) {
    const std::size_t d = trim(f).size() - 1;
    Vec h = kX;
    for (std::size_t i = 1; i <= d / 2; ++i) {
        h = naive_pow_mod(h, P, f);
        if (naive_gcd(naive_sub(h, kX), f) != Vec{1}) return false;
    }
    return true;
}

// A random monic irreducible of degree d (about 1 in d random polynomials is one).
Vec random_irreducible(std::size_t d) {
    for (;;) {
        Vec f = random_vec(d + 1, 0);
        f[d] = 1;
        if (irreducible(f)) return f;
    }
}

// Ordered as fps::factor's: by degree, then coefficients from the constant term.
void sort_factors(Factors& fs) {
    std::ranges::sort(fs, [](const auto& a, const auto& b) {
        if (a.first.size() != b.first.size()) return a.first.size() < b.first.size();
        return std::ranges::lexicographical_compare(a.first, b.first);
    });
}

Factors to_vecs(const std::vector<std::pair<Poly, int>>& fs) {
    Factors out;
    for (const auto& [g, e] : fs) out.emplace_back(vec(g), e);
    return out;
}

// gcd on every pair of edge sizes and kinds, with and without a common factor, and with trailing
// zeros; then long ones on both of division's paths.
void test_gcd() {
    std::size_t turn = 0;
    for (const std::size_t n : kSizes)
        for (const std::size_t m : kSizes) {
            ++turn;
            Vec f = random_vec(n, int(turn % 4)), g = random_vec(m, int(turn / 4 % 4));
            if (turn % 3 == 0 && n && m) {
                const Vec c = random_degree(rng() % std::min(n, m));
                f = naive_mul(f, c), g = naive_mul(g, c);
            }
            if (turn % 5 == 0) f.resize(f.size() + 3);
            const Vec want = naive_gcd(f, g);
            CHECK(vec(fps::gcd(poly(f), poly(g))) == want, n, m, turn);
            CHECK(vec(fps::gcd(poly(g), poly(f))) == want, n, m, turn);
        }
    CHECK(fps::gcd(Poly{}, Poly{}).size() == 0 && fps::gcd(Poly(5), Poly(3)).size() == 0);
    CHECK(vec(fps::gcd(Poly{0, 2, 4, 0}, Poly{})) == (Vec{0, inv(2), 1}));
    CHECK(vec(fps::gcd(Poly{3}, Poly{1, 2, 3})) == Vec{1});
    CHECK(vec(fps::gcd(Poly{3, 1}, Poly{6, 2, 0})) == (Vec{3, 1}));
    // gcd(a c, b c) = monic c: short quotients (the recurrence), then a long one (Newton).
    for (const auto& [na, nb, nc] : {std::tuple{2000, 1500, 301}, {5000, 40, 100}, {40, 3000, 1}, {700, 700, 1}}) {
        const Vec c = random_degree(std::size_t(nc - 1)), a = random_degree(std::size_t(na)),
                  b = random_degree(std::size_t(nb));
        const Vec want = naive_gcd(naive_mul(a, c), naive_mul(b, c));
        CHECK(vec(fps::gcd(poly(a) * poly(c), poly(b) * poly(c))) == want, std::size_t(na), std::size_t(nc));
        if (nc > 1) CHECK(want.size() >= std::size_t(nc), std::size_t(na));
    }
    // Fibonacci-like pairs: every quotient of degree 1, deg f steps.
    const Poly x = Poly::x();
    Poly a{1}, b = x + 1;
    for (int i = 0; i < 300; ++i) a = std::exchange(b, x * b + a);
    CHECK(vec(fps::gcd(a, b)) == Vec{1});
    CHECK(vec(fps::gcd(a * b, b * (x + 3))) == monic(vec(b)));
}

// inverse_mod by multiplication, and existence against naive_gcd.
void test_inverse_mod() {
    std::size_t turn = 0;
    for (const std::size_t d : kSizes) {
        if (d == 0) continue;
        for (const std::size_t n : {std::size_t(0), std::size_t(1), d - 1, d, d + 1, 2 * d + 3}) {
            ++turn;
            Vec m = random_vec(d + 1, int(turn % 4)), f = random_vec(n, int(turn / 4 % 4));
            if (trim(m).empty()) m[d] = nonzero();
            if (turn % 4 == 0) m.resize(m.size() + 2);
            const std::size_t dm = trim(m).size() - 1;
            const auto h = fps::inverse_mod(poly(f), poly(m));
            const bool coprime = dm == 0 || naive_gcd(f, m) == Vec{1};
            CHECK(h.has_value() == coprime, d, n, turn);
            if (!h) continue;
            CHECK(h->size() == dm, d, n, turn);
            if (dm > 0) CHECK(naive_mod(naive_mul(f, vec(*h)), m) == Vec{1}, d, n, turn);
        }
        // A shared factor: no inverse.
        const Vec m = random_degree(d), f = random_vec(d + 2, 0), l = {random_value(), 1};
        CHECK(!fps::inverse_mod(poly(naive_mul(f, l)), poly(naive_mul(m, l))).has_value(), d);
        CHECK(!fps::inverse_mod(poly(naive_mul(m, {3})), poly(m)).has_value() || d == 0, d);
    }
    // Long: m of 3000, f of 5000.
    const Vec m = random_degree(3000), f = random_vec(5000, 0);
    const auto h = fps::inverse_mod(poly(f), poly(m));
    CHECK(h && naive_mod(naive_mul(f, vec(*h)), m) == Vec{1});
    CHECK(vec(*fps::inverse_mod(Poly{5}, Poly{0, 0, 2})) == (Vec{inv(5), 0}));
    CHECK(fps::inverse_mod(Poly{5}, Poly{7, 0})->size() == 0 && fps::inverse_mod(Poly{}, Poly{7})->size() == 0);
    CHECK(!fps::inverse_mod(Poly{}, Poly{1, 1}).has_value() && !fps::inverse_mod(Poly{0, 1}, Poly{0, 1}).has_value());
}

u32 non_square() {
    u32 z = 2;
    while (power(z, (P - 1) / 2) == 1) ++z;
    return z;
}

// (x - a)^2 - z: irreducible for z a non-square.
Vec irreducible_quadratic(u32 a, u32 z) { return {sub(mul(a, a), z), sub(0, mul(2, a)), 1}; }

// The product of (x - r) over the distinct roots, monic: gcd(f, x^P - x).
Vec naive_root_product(const Vec& f) { return naive_gcd(f, naive_sub(naive_pow_mod(kX, P, f), kX)); }

Vec from_roots(const Vec& roots) {
    Vec f{1};
    for (const u32 r : roots) f = naive_mul(f, {sub(0, r), 1});
    return f;
}

void test_roots() {
    // Random polynomials: the product of (x - r) over the roots found is gcd(f, x^P - x).
    std::size_t turn = 0;
    for (const std::size_t n : kSizes) {
        for (int kind = 0; kind < 4; ++kind, ++turn) {
            Vec f = random_vec(n, kind);
            if (turn % 3 == 0) f = naive_mul(f, from_roots({0, random_value(), random_value()}));
            if (trim(f).empty()) continue;
            const Vec found = fps::roots(poly(f));
            CHECK(std::ranges::is_sorted(found) && std::ranges::adjacent_find(found) == found.end(), n, turn);
            CHECK(from_roots(found) == (trim(f).size() == 1 ? Vec{1} : naive_root_product(f)), n, turn);
        }
    }
    // Known roots with multiplicities, times irreducible quadratics.
    const u32 z = non_square();
    for (const std::size_t count : {0, 1, 2, 3, 5, 16, 17, 40, 200}) {
        Vec f{nonzero()}, want;
        for (std::size_t i = 0; i < count; ++i) {
            const u32 r = i == 0 && count % 2 ? 0 : i % 7 == 6 ? want[rng() % want.size()] : random_value();
            want.push_back(r);
            for (std::size_t e = 0; e <= rng() % 3; ++e) f = naive_mul(f, {sub(0, r), 1});
        }
        for (std::size_t i = 0; i < count / 3 + 1; ++i) f = naive_mul(f, irreducible_quadratic(random_value(), z));
        std::ranges::sort(want);
        want.erase(std::unique(want.begin(), want.end()), want.end());
        CHECK(fps::roots(poly(f)) == want, count);
    }
    CHECK(fps::roots(Poly{5}).empty() && fps::roots(Poly{5, 0, 0}).empty());
    CHECK(fps::roots(Poly{0, 0, 3}) == Vec{0} && fps::roots(Poly{0, 7}) == Vec{0});
    CHECK(fps::roots(Poly{P - 1, 0, 1}) == (Vec{1, P - 1}));
    CHECK(fps::roots(poly(irreducible_quadratic(5, z))).empty());
    // x^(2^23) = 1 has 2^23 roots; x^4 - 1 its four of order 4.
    const u32 i4 = power(3, (P - 1) / 4);
    Vec want4 = {1, P - 1, i4, P - i4};
    std::ranges::sort(want4);
    CHECK(fps::roots(Poly::x(4) - 1) == want4);
    // Long: 2000 random roots, some repeated, times a quadratic.
    std::vector<Poly> linear;
    Vec want;
    for (int i = 0; i < 2000; ++i) {
        want.push_back(i % 10 == 9 ? want[rng() % want.size()] : random_value());
        linear.push_back(Poly{sub(0, want.back()), 1});
    }
    std::ranges::sort(want);
    want.erase(std::unique(want.begin(), want.end()), want.end());
    CHECK(fps::roots(fps::product(linear) * poly(irreducible_quadratic(3, z))) == want);
}

// factor of the product of the given factors (to their multiplicities), times c.
void check_factor(Factors parts, u32 c, std::size_t tag) {
    Vec f{c};
    for (const auto& [g, e] : parts)
        for (int i = 0; i < e; ++i) f = naive_mul(f, g);
    sort_factors(parts);
    CHECK(to_vecs(fps::factor(poly(f))) == parts, tag, parts.size());
}

void test_factor() {
    const u32 z = non_square();
    // Known irreducibles: linear, quadratic, random of degree up to 8; several of one degree (the
    // equal-degree split); multiplicities 1 to 4; x itself.
    for (std::size_t trial = 0; trial < 24; ++trial) {
        Factors parts;
        const auto add = [&](const Vec& g, int e) {
            if (std::ranges::none_of(parts, [&](const auto& p) { return p.first == g; })) parts.emplace_back(g, e);
        };
        for (std::size_t i = 0; i < trial % 5; ++i) add({random_value(), 1}, 1 + int(rng() % 4));
        for (std::size_t i = 0; i < trial % 3; ++i) add(irreducible_quadratic(random_value(), z), 1 + int(rng() % 3));
        for (std::size_t i = 0; i < trial % 4; ++i) add(random_irreducible(3 + rng() % 6), 1 + int(rng() % 2));
        if (trial % 6 == 5) add(kX, 1 + int(trial % 4));
        if (trial % 7 == 3)
            for (int i = 0; i < 5; ++i) add(random_irreducible(4), 1);
        check_factor(parts, trial % 2 ? 1 : nonzero(), trial);
    }
    // Random polynomials: monic irreducible factors, ordered, distinct, whose product is monic f.
    std::size_t turn = 0;
    for (const std::size_t n : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33}) {
        for (int kind = 0; kind < 4; ++kind, ++turn) {
            Vec f = random_vec(n, kind);
            if (turn % 3 == 0) f = naive_mul(f, naive_mul(f, {1, 1}));
            if (trim(f).empty()) continue;
            if (turn % 4 == 1) f.resize(f.size() + 2);
            const auto fs = fps::factor(poly(f));
            Vec back{1};
            for (const auto& [g, e] : fs) {
                CHECK(g.degree() >= 1 && g.size() == std::size_t(g.degree() + 1) && g[g.size() - 1] == 1, n, turn);
                CHECK(e >= 1 && irreducible(vec(g)), n, turn, g.size());
                for (int i = 0; i < e; ++i) back = naive_mul(back, vec(g));
            }
            CHECK(back == monic(f), n, turn);
            Factors sorted = to_vecs(fs);
            sort_factors(sorted);
            CHECK(sorted == to_vecs(fs), n, turn);
            for (std::size_t i = 1; i < fs.size(); ++i) CHECK(vec(fs[i].first) != vec(fs[i - 1].first), n, turn);
        }
    }
    CHECK(fps::factor(Poly{7}).empty() && fps::factor(Poly{7, 0, 0}).empty());
    check_factor({{{1, 1}, 2}}, 1, 0);
    check_factor({{kX, 5}}, 3, 0);
    check_factor({{{1, 1}, 1}, {{2, 1}, 1}}, 1, 0);
    // x^4 + 1 splits into linear factors (8 divides P - 1); x^2 + 1 into two.
    const u32 r8 = power(3, (P - 1) / 8);
    Factors linear;
    for (int k = 1; k < 8; k += 2) linear.push_back({{sub(0, power(r8, u64(k))), 1}, 1});
    check_factor(linear, 1, 8);
    // Long: 60 linear factors, 10 quadratics, 4 of degree 6, multiplicities.
    Factors parts;
    for (int i = 0; i < 60; ++i) parts.push_back({{random_value(), 1}, 1 + i % 3});
    for (int i = 0; i < 10; ++i) parts.push_back({irreducible_quadratic(random_value(), z), 1 + i % 2});
    for (int i = 0; i < 4; ++i) parts.push_back({random_irreducible(6), 1});
    check_factor(parts, nonzero(), 1);
}

void test_threads() {
    const Vec f = naive_mul(from_roots({1, 2, 3, 4, 5, 6, 7, 8}), random_irreducible(5)), g = random_degree(100);
    const Vec want_gcd = vec(fps::gcd(poly(f), poly(naive_mul(g, {sub(0, 3), 1}))));
    const Vec want_roots = fps::roots(poly(f));
    const Factors want_factors = to_vecs(fps::factor(poly(f)));
    std::vector<std::thread> threads;
    std::vector<int> ok(4, 1);
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&, t] {
            for (int rep = 0; rep < 5; ++rep) {
                ok[std::size_t(t)] &= vec(fps::gcd(poly(f), poly(naive_mul(g, {sub(0, 3), 1})))) == want_gcd;
                ok[std::size_t(t)] &= fps::roots(poly(f)) == want_roots;
                ok[std::size_t(t)] &= to_vecs(fps::factor(poly(f))) == want_factors;
            }
        });
    for (std::thread& t : threads) t.join();
    CHECK(std::ranges::all_of(ok, [](int v) { return v == 1; }));
    CHECK(want_gcd == (Vec{sub(0, 3), 1}) && want_roots == (Vec{1, 2, 3, 4, 5, 6, 7, 8}));
}

void test_errors() {
    using std::domain_error;
    CHECK(throws<domain_error>([] { (void)fps::inverse_mod(Poly{1, 2}, Poly{}); }));
    CHECK(throws<domain_error>([] { (void)fps::inverse_mod(Poly{1, 2}, Poly(3)); }));
    CHECK(throws<domain_error>([] { (void)fps::roots(Poly{}); }));
    CHECK(throws<domain_error>([] { (void)fps::roots(Poly(4)); }));
    CHECK(throws<domain_error>([] { (void)fps::factor(Poly{}); }));
    CHECK(throws<domain_error>([] { (void)fps::factor(Poly(4)); }));
    // deg f >= P: 4 GB of unwritten memory, only the top word written.
    {
        Poly big = fps::detail::unwritten_polynomial<P>(std::size_t(P) + 1);
        big[P] = 1;
        CHECK(throws<domain_error>([&] { (void)fps::factor(big); }));
    }
    CHECK(!throws<domain_error>([] { (void)fps::gcd(Poly{}, Poly(3)); }));
}

}  // namespace

int main() {
    test_gcd();
    test_inverse_mod();
    test_roots();
    test_factor();
    test_threads();
    test_errors();
    if (failures) {
        std::printf("test_algebra: %d failures\n", failures);
        return 1;
    }
    std::printf("test_algebra: ok\n");
    return 0;
}
