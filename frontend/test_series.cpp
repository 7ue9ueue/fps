// Tests of fps/series.hpp: derivative, integral, inverse, log, exp and their raw forms, against
// O(n^2) recurrences on edge sizes and random inputs, on both paths (Newton, and the recurrence
// for at most 16 terms past f[0]), against identities at large sizes, with aliasing, threads and
// every precondition.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "fps/fps.hpp"

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

std::mt19937_64 rng(20);

// 0, 1, powers of two and their neighbours.
const std::vector<std::size_t> kSizes = {0,  1,  2,  3,   4,   5,   7,   8,   9,   15,  16,   17,   31,  32,  33,
                                         63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025};

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 sub(u32 a, u32 b) { return a >= b ? a - b : a + P - b; }

u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}

u32 inv(u32 a) { return power(a, P - 2); }

// Kinds: 0 random, 1 mostly zeros, 2 all P - 1 (the largest sums), 3 random with t terms past
// index 0 (the recurrence for t <= 16), t = 1 + rng() % 20.
Vec random_vec(std::size_t n, int kind) {
    Vec a(n);
    if (kind == 3) {
        const std::size_t t = 1 + rng() % 20;
        if (n) a[0] = u32(rng() % P);
        for (std::size_t i = 0; i < t && n > 1; ++i) a[1 + rng() % (n - 1)] = u32(1 + rng() % (P - 1));
        return a;
    }
    for (u32& c : a) {
        const u64 r = rng();
        c = kind == 2 ? P - 1 : kind == 1 && r % 4 ? 0 : u32(r % P);
    }
    return a;
}

// a with a[0] = c (a nonempty).
Vec with_constant(Vec a, u32 c) {
    if (a.empty()) a.push_back(0);
    a[0] = c;
    return a;
}

Vec vec(const Poly& f) { return Vec(f.span().begin(), f.span().end()); }
Poly poly(const Vec& a) { return Poly(std::span<const u32>(a)); }
u32 at(const Vec& f, std::size_t i) { return i < f.size() ? f[i] : 0; }

// Frees n words of P - 1, so the next allocation of n words likely reuses them.
void dirty_heap(std::size_t n) { (void)poly(Vec(n, P - 1)); }

// The references: f[0] g[i] = [i = 0] - sum_j f[j] g[i - j]; for log, G = i g[i] with
// G[i] = i f[i] - sum_j f[j] G[i - j]; for exp, i g[i] = sum_j j f[j] g[i - j].
Vec naive_inverse(const Vec& f, std::size_t n) {
    Vec g(n);
    if (n == 0) return g;
    const u32 c = inv(f[0]);
    for (std::size_t i = 0; i < n; ++i) {
        u32 s = i == 0 ? 1 : 0;
        for (std::size_t j = 1; j <= i; ++j) s = sub(s, mul(at(f, j), g[i - j]));
        g[i] = mul(s, c);
    }
    return g;
}

Vec naive_log(const Vec& f, std::size_t n) {
    Vec G(n), g(n);
    for (std::size_t i = 1; i < n; ++i) {
        u32 s = mul(u32(i), at(f, i));
        for (std::size_t j = 1; j < i; ++j) s = sub(s, mul(at(f, j), G[i - j]));
        G[i] = s;
        g[i] = mul(s, inv(u32(i)));
    }
    return g;
}

Vec naive_exp(const Vec& f, std::size_t n) {
    Vec g(n);
    if (n) g[0] = 1;
    for (std::size_t i = 1; i < n; ++i) {
        u32 s = 0;
        for (std::size_t j = 1; j <= i; ++j) s = (s + u64(mul(u32(j), at(f, j))) * g[i - j]) % P;
        g[i] = mul(s, inv(u32(i)));
    }
    return g;
}

// raw::<op> into out of n words that start as P - 1: every word must be written.
template <class Raw>
Vec raw_out(Raw raw, const Vec& f, std::size_t n) {
    Vec out(n, P - 1);
    raw(std::span<const u32>(f), std::span<u32>(out));
    return out;
}

void test_calculus() {
    for (const std::size_t n : kSizes)
        for (const int kind : {0, 1, 2}) {
            const Vec f = random_vec(n, kind);
            Vec d(n ? n - 1 : 0), q(n + 1);
            for (std::size_t i = 0; i + 1 < n; ++i) d[i] = mul(u32(i + 1), f[i + 1]);
            for (std::size_t i = 0; i < n; ++i) q[i + 1] = mul(f[i], inv(u32(i + 1)));
            dirty_heap(d.size());
            CHECK(vec(fps::derivative(poly(f))) == d, n, kind);
            dirty_heap(q.size());
            CHECK(vec(fps::integral(poly(f))) == q, n, kind);
            CHECK(fps::derivative(fps::integral(poly(f))).span().size() == n, n);
            CHECK(vec(fps::derivative(fps::integral(poly(f)))) == f, n, kind);
        }
    CHECK(vec(fps::integral(Poly())) == Vec{0});
    CHECK(fps::derivative(Poly{5}).size() == 0);
    CHECK(fps::derivative(Poly()).size() == 0);
    const std::size_t n = (std::size_t(1) << 20) + 3;
    const Poly f = poly(random_vec(n, 0));
    CHECK(vec(fps::derivative(fps::integral(f))) == vec(f));
    const Poly q = fps::integral(f);
    for (int t = 0; t < 100; ++t) {
        const std::size_t i = rng() % n;
        CHECK(mul(q[i + 1], u32(i + 1)) == f[i], i);
    }
}

// Every function at n against its reference, for f of n - 1, n and n + 2 coefficients.
void test_small() {
    for (const std::size_t n : kSizes)
        for (const int kind : {0, 1, 2, 3})
            for (const std::size_t size : {n ? n - 1 : 0, n, n + 2}) {
                const Vec a = random_vec(size, kind);
                const Vec fi = n ? with_constant(a, 1 + u32(rng() % (P - 1))) : a;
                const Vec fl = n ? with_constant(a, 1) : a, fe = n ? with_constant(a, 0) : a;
                const Vec gi = naive_inverse(fi, n), gl = naive_log(fl, n), ge = naive_exp(fe, n);
                dirty_heap(n);
                CHECK(vec(fps::inverse(poly(fi), n)) == gi, n, kind, size);
                dirty_heap(n);
                CHECK(vec(fps::log(poly(fl), n)) == gl, n, kind, size);
                dirty_heap(n);
                CHECK(vec(fps::exp(poly(fe), n)) == ge, n, kind, size);
                CHECK(raw_out([](auto f, auto out) { fps::raw::inverse<P>(f, out); }, fi, n) == gi, n, kind, size);
                CHECK(raw_out([](auto f, auto out) { fps::raw::log<P>(f, out); }, fl, n) == gl, n, kind, size);
                CHECK(raw_out([](auto f, auto out) { fps::raw::exp<P>(f, out); }, fe, n) == ge, n, kind, size);
                if (size == n) {
                    CHECK(vec(fps::inverse(poly(fi))) == gi, n);
                    CHECK(vec(fps::log(poly(fl))) == gl, n);
                    CHECK(vec(fps::exp(poly(fe))) == ge, n);
                }
            }
}

// Exactly t terms past f[0] for t around kSparseTerms = 16, near index 0 and spread out, so both
// paths run on the same kind of input.
void test_term_counts() {
    for (const std::size_t n : {std::size_t(17), std::size_t(18), std::size_t(40), std::size_t(300), std::size_t(1000)})
        for (const std::size_t t : {0, 1, 2, 15, 16, 17, 18})
            for (const bool spread : {false, true}) {
                if (t >= n) continue;
                Vec f(n);
                std::vector<std::size_t> idx;
                for (std::size_t i = 1; i < n; ++i) idx.push_back(i);
                if (spread) std::shuffle(idx.begin(), idx.end(), rng);
                for (std::size_t i = 0; i < t; ++i) f[idx[i]] = u32(1 + rng() % (P - 1));
                CHECK(vec(fps::inverse(poly(with_constant(f, 3)), n)) == naive_inverse(with_constant(f, 3), n), n, t);
                CHECK(vec(fps::log(poly(with_constant(f, 1)), n)) == naive_log(with_constant(f, 1), n), n, t);
                CHECK(vec(fps::exp(poly(f), n)) == naive_exp(f, n), n, t);
            }
}

// Closed forms: 1 / (1 - x) = sum x^i, log(1 / (1 - x)) = sum x^i / i, exp(x) = sum x^i / i!,
// exp(c x^k) = sum c^j x^(jk) / j!.
void test_closed_forms() {
    const Poly x = Poly::x();
    const std::size_t n = 1'000'000;
    CHECK(vec(fps::inverse(1 - x, n)) == Vec(n, 1));
    CHECK(fps::inverse(1 - x).size() == 2);
    Vec h(n);
    for (std::size_t i = 1; i < n; ++i) h[i] = inv(u32(i));
    CHECK(vec(fps::log(fps::inverse(1 - x, n))) == h);
    CHECK(vec(fps::log(1 - x, n)) == vec(-poly(h)));
    const fps::Factorials<P> fact(n);
    Vec e(n);
    for (std::size_t i = 0; i < n; ++i) e[i] = fact.inverse(i);
    CHECK(vec(fps::exp(x, n)) == e);
    Vec e3(n);
    for (std::size_t j = 0; 3 * j < n; ++j) e3[3 * j] = mul(power(5, j), fact.inverse(j));
    CHECK(vec(fps::exp(5 * Poly::x(3), n)) == e3);
}

// Identities on the Newton paths at large sizes: f inverse(f) = 1, log(exp f) = f, exp(log f) = f.
void test_large() {
    for (const std::size_t n : {(std::size_t(1) << 14) - 1, std::size_t(1) << 14, (std::size_t(1) << 14) + 1,
                                std::size_t(1) << 17, std::size_t(300'001)}) {
        const Poly fi = poly(with_constant(random_vec(n, 0), 7));
        const Poly g = fps::inverse(fi);
        Poly one(n);
        one[0] = 1;
        CHECK(vec(fps::mul(fi, g, n)) == vec(one), n);
        const Poly fe = poly(with_constant(random_vec(n, 1), 0)), fl = poly(with_constant(random_vec(n, 0), 1));
        CHECK(vec(fps::log(fps::exp(fe))) == vec(fe), n);
        CHECK(vec(fps::exp(fps::log(fl))) == vec(fl), n);
    }
    // The recurrence at a large size, against the Newton path's identity.
    const std::size_t n = 200'000;
    Vec a(n);
    for (int t = 0; t < 16; ++t) a[1 + rng() % (n - 1)] = u32(rng() % P);
    const Poly fi = poly(with_constant(a, 9)), fl = poly(with_constant(a, 1)), fe = poly(a);
    Poly one(n);
    one[0] = 1;
    CHECK(vec(fps::mul(fi, fps::inverse(fi), n)) == vec(one));
    CHECK(vec(fps::exp(fps::log(fl))) == vec(fl));
    CHECK(vec(fps::log(fps::exp(fe))) == vec(fe));
}

// The raw forms on overlapping memory: exp in place (also with f shorter than out, the words past
// f not read), log on any overlap; random words left in out before.
void test_aliasing() {
    for (const std::size_t n : {std::size_t(1), std::size_t(17), std::size_t(300), std::size_t(5000)})
        for (const int kind : {0, 3}) {
            const Vec fe = with_constant(random_vec(n, kind), 0);
            Vec buffer = fe;
            fps::raw::exp<P>(buffer, buffer);
            CHECK(buffer == naive_exp(fe, n), n, kind);
            const std::size_t k = n / 2;
            buffer = with_constant(random_vec(n, 0), 0);
            const Vec head(buffer.begin(), buffer.begin() + std::max<std::size_t>(k, 1));
            fps::raw::exp<P>(std::span<const u32>(buffer).first(head.size()), buffer);
            CHECK(buffer == naive_exp(head, n), n, kind);

            const Vec fl = with_constant(random_vec(n + 100, kind), 1);
            const Vec gl = naive_log(fl, n);
            for (const std::size_t shift : {std::size_t(0), std::size_t(1), std::size_t(50)}) {
                buffer = fl;
                buffer.resize(n + 200, P - 1);
                const std::span<const u32> f(buffer.data() + 100, n);
                std::copy(fl.begin(), fl.begin() + n, buffer.begin() + 100);
                fps::raw::log<P>(f, std::span(buffer).subspan(100 - shift, n));
                CHECK(Vec(buffer.begin() + 100 - shift, buffer.begin() + 100 - shift + n) == gl, n, kind, shift);
                std::copy(fl.begin(), fl.begin() + n, buffer.begin() + 100);
                fps::raw::log<P>(f, std::span(buffer).subspan(100 + shift, n));
                CHECK(Vec(buffer.begin() + 100 + shift, buffer.begin() + 100 + shift + n) == gl, n, kind, shift);
            }
        }
}

void test_threads() {
    std::vector<std::thread> threads;
    std::vector<int> ok(4, 0);
    std::vector<Poly> inputs;
    for (int t = 0; t < 4; ++t) inputs.push_back(poly(with_constant(random_vec(30'000 + 1000 * t, 0), 0)));
    std::vector<Poly> expected;
    for (const Poly& f : inputs) expected.push_back(fps::exp(f));
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&, t] {
            bool good = true;
            for (int r = 0; r < 3; ++r) {
                good = good && vec(fps::exp(inputs[t])) == vec(expected[t]);
                good = good && vec(fps::log(fps::exp(inputs[t]))) == vec(inputs[t]);
            }
            ok[t] = good;
        });
    for (std::thread& t : threads) t.join();
    for (int t = 0; t < 4; ++t) CHECK(ok[t], t);
}

void test_errors() {
    using std::domain_error;
    CHECK(throws<domain_error>([] { (void)fps::inverse(Poly{0, 1}, 3); }));
    CHECK(throws<domain_error>([] { (void)fps::inverse(Poly{0, 1}); }));
    CHECK(throws<domain_error>([] { (void)fps::inverse(Poly(), 1); }));
    CHECK(fps::inverse(Poly{0, 1}, 0).size() == 0);
    CHECK(fps::inverse(Poly()).size() == 0);
    CHECK(throws<domain_error>([] { (void)fps::log(Poly{2, 1}, 3); }));
    CHECK(throws<domain_error>([] { (void)fps::log(Poly{0, 1}); }));
    CHECK(throws<domain_error>([] { (void)fps::log(Poly(), 2); }));
    CHECK(fps::log(Poly{2, 1}, 0).size() == 0);
    CHECK(throws<domain_error>([] { (void)fps::exp(Poly{1, 1}, 3); }));
    CHECK(throws<domain_error>([] { (void)fps::exp(Poly{P - 1}); }));
    CHECK(fps::exp(Poly{1, 1}, 0).size() == 0);
    CHECK(vec(fps::exp(Poly(), 3)) == (Vec{1, 0, 0}));
    CHECK(vec(fps::exp(Poly())).empty());

    // Overlaps: raw::inverse never, raw::exp only from the same first word.
    Vec buffer = with_constant(random_vec(100, 0), 1);
    const std::span<u32> all(buffer);
    CHECK(throws<domain_error>([&] { fps::raw::inverse<P>(all.first(50), all.first(50)); }));
    CHECK(throws<domain_error>([&] { fps::raw::inverse<P>(all.first(50), all.subspan(19, 20)); }));
    CHECK(throws<domain_error>([&] { fps::raw::inverse<P>(all.subspan(30, 50), all.first(31)); }));
    CHECK(!throws<domain_error>([&] { fps::raw::inverse<P>(all.first(50), all.subspan(50, 20)); }));
    buffer[0] = 0;
    CHECK(throws<domain_error>([&] { fps::raw::exp<P>(all.first(50), all.subspan(1, 50)); }));
    buffer[10] = 0;
    CHECK(throws<domain_error>([&] { fps::raw::exp<P>(all.subspan(10, 50), all.first(20)); }));
    CHECK(!throws<domain_error>([&] { fps::raw::exp<P>(all.subspan(10, 50), all.subspan(60, 40)); }));

    // n > P for log and exp, f.size() >= P for integral: 4 GB of unwritten memory, never touched.
    {
        const std::size_t big = std::size_t(P) + 1;
        const auto words = std::make_unique_for_overwrite<u32[]>(big);
        const std::span<u32> out(words.get(), big);
        const Vec one{1}, zero{0};
        CHECK(throws<domain_error>([&] { fps::raw::log<P>(one, out); }));
        CHECK(throws<domain_error>([&] { fps::raw::exp<P>(zero, out); }));
        CHECK(throws<domain_error>([&] { fps::raw::exp<P>(Vec{}, out); }));
    }
    CHECK(throws<domain_error>([&] { (void)fps::integral(fps::detail::unwritten_polynomial<P>(P)); }));
}

}  // namespace

int main() {
    test_calculus();
    test_small();
    test_term_counts();
    test_closed_forms();
    test_large();
    test_aliasing();
    test_threads();
    test_errors();
    if (failures) {
        std::printf("series: %d failures\n", failures);
        return 1;
    }
    std::printf("series: all passed\n");
    return 0;
}
