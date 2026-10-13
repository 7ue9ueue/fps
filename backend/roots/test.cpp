// Tests of backend/roots: the distinct roots of polynomials built from known roots (distinct,
// repeated, 0, all equal, roots that collide under the Graeffe map, times irreducible quadratics),
// and of random polynomials against the degree of gcd(f, x^P - x) (naive powers to degree 60, the
// backend's products, division and half-gcd above); sizes around powers of two to 2^13; the shift
// loop (roots at the first shifts); every struct of 8-word rows the target runs; argument checks.
// tools/kernel.py builds it at -O2 and with ASan/UBSan.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/division/division.hpp"
#include "backend/gcd/gcd.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/roots/roots.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = fps::root_finding::kP;
namespace rd = fps::root_finding::detail;

int failures = 0, checks = 0;
std::mt19937_64 rng(122);

void fail(const std::string& what) {
    if (++failures <= 200) std::printf("FAIL %s\n", what.c_str());
}

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 add(u32 a, u32 b) { return a + b >= P ? a + b - P : a + b; }
u32 sub(u32 a, u32 b) { return a >= b ? a - b : a + P - b; }
u32 pw(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
u32 inv(u32 a) { return pw(a, P - 2); }
u32 random_word() { return u32(rng() % P); }
u32 random_nonzero() { return 1 + u32(rng() % (P - 1)); }

void trim(Vec& a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
}

u32 eval(const Vec& f, u32 x) {
    u32 r = 0;
    for (std::size_t i = f.size(); i-- > 0;) r = add(mul(r, x), f[i]);
    return r;
}

Vec multiply(const Vec& a, const Vec& b) {
    if (a.empty() || b.empty()) return {};
    Vec c(a.size() + b.size() - 1);
    fps::ntt::multiply(a, b, c);
    return c;
}

// The product of (x - a) over the list (repeats are multiplicities), by halves.
Vec from_roots(std::span<const u32> roots) {
    if (roots.empty()) return {1};
    if (roots.size() == 1) return {roots[0] ? P - roots[0] : 0, 1};
    const std::size_t h = roots.size() / 2;
    return multiply(from_roots(roots.first(h)), from_roots(roots.subspan(h)));
}

// A product of `count` irreducible quadratics x^2 + b x + c (discriminant a non-residue).
Vec junk(std::size_t count) {
    Vec f{1};
    for (std::size_t i = 0; i < count; ++i) {
        for (;;) {
            const u32 b = random_word(), c = random_word();
            const u32 disc = sub(mul(b, b), mul(4, c));
            if (disc != 0 && pw(disc, (P - 1) / 2) == P - 1) {
                f = multiply(f, Vec{c, b, 1});
                break;
            }
        }
    }
    return f;
}

// ---- references -------------------------------------------------------------------------------

// a mod f, f trimmed (deg >= 1), naive.
Vec naive_mod(Vec a, const Vec& f) {
    trim(a);
    const std::size_t m = f.size();
    const u32 lead = inv(f.back());
    while (a.size() >= m) {
        const u32 c = mul(a.back(), lead);
        const std::size_t shift = a.size() - m;
        for (std::size_t i = 0; i < m; ++i) a[shift + i] = sub(a[shift + i], mul(c, f[i]));
        trim(a);
    }
    return a;
}

Vec naive_mul(const Vec& a, const Vec& b) {
    if (a.empty() || b.empty()) return {};
    Vec c(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = add(c[i + j], mul(a[i], b[j]));
    return c;
}

// x^P - x mod f: square and multiply; mulmod(a, b) by `product` and `reduce`.
template <class Product, class Reduce>
Vec frobenius_minus_x(const Vec& f, Product product, Reduce reduce) {
    Vec h = reduce(Vec{0, 1});
    for (int bit = 28; bit >= 0; --bit) {  // P < 2^30, top bit 29
        h = reduce(product(h, h));
        if (P >> bit & 1) {
            Vec xh(h.size() + 1);
            std::copy(h.begin(), h.end(), xh.begin() + 1);
            h = reduce(xh);
        }
    }
    h.resize(std::max<std::size_t>(h.size(), 2));
    h[1] = sub(h[1], 1);
    trim(h);
    return h;
}

// deg gcd(a, b) by naive Euclid.
std::size_t naive_gcd_degree(Vec a, Vec b) {
    trim(a), trim(b);
    while (!b.empty()) {
        Vec r = naive_mod(a, b);
        a = std::move(b), b = std::move(r);
    }
    return a.size() - 1;
}

// The number of distinct roots of f (trimmed, deg >= 1) in F_P: deg gcd(f, x^P - x).
std::size_t root_count_naive(const Vec& f) {
    const Vec h = frobenius_minus_x(f, naive_mul, [&](const Vec& a) { return naive_mod(a, f); });
    return naive_gcd_degree(f, h);
}

std::size_t root_count_fast(const Vec& f) {
    const auto reduce = [&](const Vec& a) {
        Vec t = a;
        trim(t);
        if (t.size() < f.size()) return t;
        Vec q(fps::division::quotient_size(t.size(), f.size())), r(f.size() - 1);
        fps::division::divide(t, f, q, r);
        trim(r);
        return r;
    };
    const Vec h = frobenius_minus_x(f, multiply, reduce);
    if (h.empty()) return f.size() - 1;
    Vec out(f.size());
    return fps::half_gcd::gcd(f, h, out) - 1;
}

// ---- checks -----------------------------------------------------------------------------------

// The kernels' structs: Graeffe steps on L, cosets' folds on F.
template <class L, class F = L>
struct Structs {
    using Steps = L;
    using Folds = F;
    static std::string name() { return std::string(L::name) + "/" + F::name; }
};
using Selected = Structs<fps::root_finding::Steps, fps::root_finding::Folds>;

template <class C>
Vec run(const Vec& f, u64 state = rd::kSeed) {
    std::size_t d = f.size();
    while (d > 0 && f[d - 1] == 0) --d;
    Vec out(d ? d - 1 : 0, 0xFFFFFFFFu);
    out.resize(out.size() + 4, 0xFFFFFFFFu);  // guard words
    const std::size_t m = rd::roots<typename C::Steps, typename C::Folds>(f, std::span<u32>(out).first(d - 1), state);
    for (std::size_t i = d - 1; i < out.size(); ++i)
        if (out[i] != 0xFFFFFFFFu) fail(std::string("guard word written, structs ") + C::name());
    out.resize(m);
    return out;
}

// f's roots, on the structs C, against `expected` (a set of distinct roots) or, if `count` is set,
// against a root count with each output checked by evaluation.
template <class C>
void check(const std::string& name, const Vec& f, const Vec* expected, std::size_t count = 0, u64 state = rd::kSeed) {
    ++checks;
    Vec got;
    try {
        got = run<C>(f, state);
    } catch (const std::exception& e) {
        return fail(name + ": threw " + e.what() + " (" + C::name() + ")");
    }
    if (!std::is_sorted(got.begin(), got.end()) || std::adjacent_find(got.begin(), got.end()) != got.end())
        return fail(name + ": not increasing (" + C::name() + ")");
    if (expected) {
        Vec e = *expected;
        std::sort(e.begin(), e.end());
        e.erase(std::unique(e.begin(), e.end()), e.end());
        if (got != e) {
            char buf[128];
            std::snprintf(buf, sizeof buf, ": %zu roots, expected %zu (%s)", got.size(), e.size(), C::name().c_str());
            return fail(name + buf);
        }
        return;
    }
    for (u32 r : got)
        if (r >= P || eval(f, r) != 0) return fail(name + ": a non-root (" + C::name() + ")");
    if (got.size() != count) {
        char buf[128];
        std::snprintf(buf, sizeof buf, ": %zu roots, expected %zu (%s)", got.size(), count, C::name().c_str());
        fail(name + buf);
    }
}

// Every struct of 8-word rows this target runs, and Avx512 for the folds.
template <class F>
void each_struct(F f) {
    f.template operator()<Structs<fps::simd::Scalar>>();
#if defined(__SSE4_1__)
    f.template operator()<Structs<fps::simd::Sse>>();
#endif
#if defined(__AVX2__)
    f.template operator()<Structs<fps::simd::Avx2>>();
#endif
#if defined(__AVX2__) && defined(__AVX512VL__)
    f.template operator()<Structs<fps::simd::Avx512Vl>>();
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    f.template operator()<Structs<fps::simd::Avx512Vl, fps::simd::Avx512>>();
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
    f.template operator()<Structs<fps::simd::Neon>>();
#endif
}

// f from roots (with repeats) times a junk part of `quadratics` irreducible quadratics times c.
void check_planted(const std::string& name, const Vec& roots, std::size_t quadratics = 0, bool all_structs = false) {
    Vec f = multiply(from_roots(roots), junk(quadratics));
    const u32 c = random_nonzero();
    for (u32& x : f) x = mul(x, c);
    if (all_structs) each_struct([&]<class C> { check<C>(name, f, &roots); });
    else check<Selected>(name, f, &roots);
}

Vec random_roots(std::size_t n) {
    Vec r(n);
    for (u32& x : r) x = random_word();
    return r;
}

// n roots drawn from `values` distinct values (repeats).
Vec repeated_roots(std::size_t n, std::size_t values) {
    const Vec pool = random_roots(values);
    Vec r(n);
    for (u32& x : r) x = pool[rng() % values];
    return r;
}

// The first shift of the generator from state.
u32 first_shift(u64 state = rd::kSeed) { return rd::next_shift(state); }

// Roots that collide under the Graeffe map: s + y z^j for z of order 2^c (s the kernel's shift),
// so the roots y z^j of g = f(x + s) share their 2^k-th powers.
Vec colliding(std::size_t groups, int c, std::size_t per_group, u32 s) {
    const u32 z = pw(3, (P - 1) >> c);
    Vec r;
    for (std::size_t g = 0; g < groups; ++g) {
        const u32 y = random_nonzero();
        std::set<std::size_t> used;
        for (std::size_t j = 0; j < per_group && used.size() < (std::size_t(1) << c); ++j) {
            std::size_t e = rng() % (std::size_t(1) << c);
            while (used.count(e)) e = (e + 1) % (std::size_t(1) << c);
            used.insert(e);
            r.push_back(add(s, mul(y, pw(z, e))));
        }
    }
    return r;
}

}  // namespace

int main() {
    const auto all = [](const std::string& name, const Vec& f, const Vec* expected, std::size_t count = 0) {
        each_struct([&]<class C> { check<C>(name, f, expected, count); });
    };

    // Small cases.
    {
        const Vec none;
        all("constant", Vec{5}, &none);
        all("constant with zeros", Vec{7, 0, 0}, &none);
        const Vec zero{0};
        all("x", Vec{0, 1}, &zero);
        all("x^5", Vec{0, 0, 0, 0, 0, 3}, &zero);
        const Vec one{P - 2};
        all("x + 2", Vec{2, 1}, &one);
        const Vec two{0, P - 2};
        all("x^3 + 2 x^2", Vec{0, 0, 2, 1}, &two);
        all("x^2 + 1", Vec{1, 0, 1}, nullptr, 2);  // P = 1 mod 4
        all("x^2 - 3 (3 a non-residue)", Vec{P - 3, 0, 1}, &none);
        const Vec ab{5, 9};
        all("(x - 5)(x - 9)", from_roots(ab), &ab);
        const Vec aa{5, 5};
        all("(x - 5)^2", from_roots(aa), &aa);
        const Vec zero_five{0, 5};
        all("x^2 (x - 5)^2", multiply(Vec{0, 0, 1}, from_roots(aa)), &zero_five);
        all("x^1024 - 1", [] { Vec f(1025); f[0] = P - 1, f[1024] = 1; return f; }(), nullptr, 1024);
        all("x^119 - 1", [] { Vec f(120); f[0] = P - 1, f[119] = 1; return f; }(), nullptr, 119);
        all("x^1000 - 1", [] { Vec f(1001); f[0] = P - 1, f[1000] = 1; return f; }(), nullptr, 8);  // gcd(1000, P - 1)
        all("x^4096 - 3", [] { Vec f(4097); f[0] = P - 3, f[4096] = 1; return f; }(), nullptr, 0);
    }

    // Random f against the naive reference, every struct.
    for (std::size_t d = 2; d <= 60; ++d)
        for (int seed = 0; seed < (d <= 12 ? 6 : 2); ++seed) {
            Vec f(d + 1);
            for (u32& x : f) x = random_word();
            if (f[d] == 0) f[d] = 1;
            if (seed == 1) f[0] = 0;  // root 0
            Vec r = f;
            trim(r);
            std::size_t z = 0;
            while (r[z] == 0) ++z;
            Vec u(r.begin() + std::ptrdiff_t(z), r.end());
            const std::size_t count = (z > 0) + (u.size() > 1 ? root_count_naive(u) : 0);
            all("random d=" + std::to_string(d), f, nullptr, count);
        }

    // Planted roots, small: distinct, repeated, with zeros and junk; every struct.
    for (std::size_t n = 1; n <= 70; ++n) {
        check_planted("distinct n=" + std::to_string(n), random_roots(n), rng() % 4, true);
        check_planted("repeated n=" + std::to_string(n), repeated_roots(n, 1 + rng() % n), rng() % 3, true);
        Vec z = random_roots(n);
        z[0] = 0;
        if (n > 2) z[1] = 0;
        check_planted("with 0 n=" + std::to_string(n), z, rng() % 3, true);
    }

    // Sizes around powers of two: distinct, all equal, repeated with junk.
    for (int j = 1; j <= 13; ++j)
        for (std::size_t n : {(std::size_t(1) << j) - 1, std::size_t(1) << j, (std::size_t(1) << j) + 1}) {
            const std::string at = std::to_string(n);
            const bool every = n <= 600;
            check_planted("distinct " + at, random_roots(n), 0, every);
            check_planted("all equal " + at, Vec(n, random_nonzero()), 0, every);
            check_planted("repeated " + at, repeated_roots(n, std::max<std::size_t>(1, n / 3)), n / 8, every);
        }

    // Degrees 2^j - 1 .. 2^j + 17: the Graeffe steps at length 2^j with wrapped top coefficients
    // (to 2^j + kWrap - 1) and the shift's top coefficients added directly (to 2^j + kShiftWrap - 1).
    for (int j : {6, 7, 9, 11})
        for (std::size_t e = 0; e <= 18; ++e) {
            const std::size_t d = (std::size_t(1) << j) + e - 1;
            const std::string at = std::to_string(d);
            const bool every = j <= 7;
            check_planted("distinct d=" + at, random_roots(d), 0, every);
            const std::size_t m = d / 2 + rng() % (d / 4);  // repeated roots, junk to degree d
            Vec r = repeated_roots(m - (d - m) % 2, m / 3 + 1);
            check_planted("repeated d=" + at, r, (d - r.size()) / 2, every);
            if (j <= 7) {
                Vec f(d + 1);
                for (u32& x : f) x = random_word();
                f[d] = 1;
                all("random d=" + at, f, nullptr, root_count_naive(f));
            }
        }

    // lc-opt's test shapes at N = 4000.
    check_planted("all_distinct 4000", random_roots(4000), 0, true);
    check_planted("all_same 4000", Vec(4000, random_nonzero()), 0, true);
    for (int t = 0; t < 4; ++t) {  // max_random: m linear factors over k values times junk
        const std::size_t m = 1000 + rng() % 2000, k = t % 2 ? 1 + rng() % 400 : m;
        check_planted("max_random " + std::to_string(m) + "/" + std::to_string(k), repeated_roots(m, k), (4000 - m) / 2, t == 0);
    }

    // Collisions under the Graeffe map, at every climb's level: groups of roots of g whose 2^c-th
    // powers agree, with repeats and junk.
    {
        const u32 s = first_shift();
        for (std::size_t d : {20, 100, 500, 2000, 5000})
            for (int c : {1, 3, 7, 8, 9, 12, 15, 16, 20, 23}) {
                if ((std::size_t(1) << std::min(c, 20)) > 4 * d && c > 3) continue;
                Vec r = colliding(1 + d / 64, c, std::min<std::size_t>(std::size_t(1) << c, 48), s);
                for (std::size_t i = 0; i < r.size() / 4; ++i) r.push_back(r[rng() % r.size()]);  // repeats
                if (r.size() > d) r.resize(d);
                const std::string at = "collisions d=" + std::to_string(d) + " c=" + std::to_string(c);
                check_planted(at, r, (d - r.size()) / 2, d <= 500);
            }
        // One group of 2^c roots that all collide down to level c.
        for (int c : {2, 5, 8, 10, 12}) {
            const Vec r = colliding(1, c, std::size_t(1) << c, s);
            check_planted("full coset c=" + std::to_string(c), r, 3, c <= 8);
        }
    }

    // The shift loop: f(t) = 0 at the first shifts t (repeated), so roots are divided out first.
    {
        u64 state = rd::kSeed;
        const u32 s1 = rd::next_shift(state), s2 = rd::next_shift(state), s3 = rd::next_shift(state);
        check_planted("root at shift 1", Vec{s1, 7, 8, 9}, 2, true);
        check_planted("roots at shifts 1-3", Vec{s1, s1, s2, s3, s3, 11}, 1, true);
        check_planted("only shift roots", Vec{s1, s1, s2}, 0, true);
        check_planted("shift root and one more", Vec{s1, 3}, 0, true);
        check_planted("shift root and junk", Vec{s1}, 2, true);
        Vec r = random_roots(300);
        r.push_back(s1), r.push_back(s2);
        check_planted("shift roots in 302", r, 5, true);
    }

    // Random f of larger degree with planted roots, against the fast reference.
    for (std::size_t d : {200, 1000, 3000}) {
        Vec f(d + 1);
        for (u32& x : f) x = random_word();
        f[d] = 1;
        f = multiply(f, from_roots(repeated_roots(d / 4, d / 8)));
        check<Selected>("random times planted d=" + std::to_string(f.size() - 1), f, nullptr, root_count_fast(f));
    }

    // Other seeds of the shifts (the results do not depend on them).
    for (u64 seed : {1ull, 2ull, 12345ull}) {
        const Vec r = repeated_roots(500, 300);
        Vec f = multiply(from_roots(r), junk(40));
        check<Selected>("seed " + std::to_string(seed), f, &r, 0, seed);
    }

    // Argument checks.
    {
        ++checks;
        Vec out(4);
        try {
            fps::root_finding::roots(Vec{0, 0, 0}, out);
            fail("f = 0 did not throw");
        } catch (const std::invalid_argument&) {
        }
        try {
            fps::root_finding::roots(Vec{1, 2, 3, 4, 5, 6}, out);
            fail("short out did not throw");
        } catch (const std::invalid_argument&) {
        }
        try {
            fps::root_finding::roots(Vec{}, out);
            fail("empty f did not throw");
        } catch (const std::invalid_argument&) {
        }
        // out of exactly deg f words, f with trailing zeros
        Vec f = from_roots(Vec{1, 2, 3});
        f.push_back(0), f.push_back(0);
        Vec o(3);
        if (fps::root_finding::roots(f, o) != 3 || o != Vec{1, 2, 3}) fail("trailing zeros");
    }

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
