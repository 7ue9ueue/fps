// Tests of backend/chirp. Terms and chirp / multiply_chirp, for every struct this target runs and
// K in {1, 2, 4, 8}: terms against the recurrence x_(k+1) = x_k s q^k, sizes 0 .. 99, powers of
// two and neighbours, random sizes up to 2^16 + small; c, s, q in {0, 1, 2, P - 1} and random;
// spans at offsets 0 .. 3 words with guard words around them. evaluate_geometric against Horner
// (all points up to n m = 2^22, sampled beyond); interpolate_geometric by the round trip from a
// random f (against O(n^2) Lagrange for small n), its errors; extrapolate from random samples
// against Lagrange at c + k, with c at, inside, past and wrapping around the samples. Exits
// nonzero on a failure. Also the prefix products under them (scan.hpp), every variant.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/chirp/chirp.hpp"
#include "backend/chirp/geometric.hpp"
#include "backend/chirp/shift.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
using fps::chirp::kP;
constexpr u32 P = kP;
constexpr u32 kGuard = 0xdeadbeef;

int failures = 0;
long checks = 0;
std::mt19937_64 rng(20261012);

void fail(const std::string& what) {
    if (++failures <= 30) std::printf("FAIL %s\n", what.c_str());
}

void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) fail(what);
}

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 random_word() { return u32(rng() % P); }

u32 horner(const Vec& f, u32 x) {
    u64 r = 0;
    for (std::size_t i = f.size(); i-- > 0;) r = (r * x + f[i]) % P;
    return u32(r);
}

Vec random_vec(std::size_t n) {
    Vec f(n);
    for (u32& x : f) x = random_word();
    return f;
}

// x_k by the recurrence.
Vec reference_chirp(u32 c, u32 s, u32 q, std::size_t n) {
    Vec x(n);
    u32 term = c, ratio = s;
    for (std::size_t k = 0; k < n; ++k) x[k] = term, term = mul(term, ratio), ratio = mul(ratio, q);
    return x;
}

std::vector<std::size_t> sizes() {
    std::vector<std::size_t> out;
    for (std::size_t n = 0; n < 100; ++n) out.push_back(n);
    for (int lg = 7; lg <= 16; ++lg)
        for (long d : {-1, 0, 1}) out.push_back((std::size_t(1) << lg) + d);
    for (int i = 0; i < 6; ++i) out.push_back(100 + rng() % 3000);
    out.push_back((1 << 16) + 5);
    return out;
}

std::vector<u32> parameters() { return {0, 1, 2, P - 1, random_word(), random_word()}; }

// A buffer with n words at the given offset and guard words around them.
struct Guarded {
    Vec words;
    std::size_t offset, n;
    Guarded(std::size_t n, std::size_t offset) : words(n + offset + 8, kGuard), offset(offset), n(n) {}
    std::span<u32> span() { return {words.data() + offset, n}; }
    bool guards_intact() const {
        for (std::size_t i = 0; i < offset; ++i)
            if (words[i] != kGuard) return false;
        for (std::size_t i = offset + n; i < words.size(); ++i)
            if (words[i] != kGuard) return false;
        return true;
    }
};

template <class S, int K>
void test_terms() {
    const std::string name = std::string(S::name) + " K=" + std::to_string(K);
    // Terms directly: every vector after a few steps.
    for (int trial = 0; trial < 20; ++trial) {
        const u32 c = random_word(), s = trial % 3 ? random_word() : 0, q = random_word();
        constexpr int W = K * S::lanes;
        const Vec x = reference_chirp(c, s, q, 5 * W);
        fps::chirp::Terms<S, K> terms(c, s, q);
        bool ok = true;
        for (int j = 0; j < 5; ++j, terms.next())
            for (int v = 0; v < K; ++v) {
                u32 lanes[S::lanes];
                S::store(lanes, terms[v]);
                for (int l = 0; l < S::lanes; ++l) ok = ok && lanes[l] < 2 * P && lanes[l] % P == x[j * W + v * S::lanes + l];
            }
        check(ok, name + " Terms");
    }
    const auto all = sizes();
    for (std::size_t n : all) {
        const auto ps = parameters();
        const int trials = n < 100 ? 6 : 2;
        for (int trial = 0; trial < trials; ++trial) {
            const u32 c = ps[rng() % ps.size()], s = ps[rng() % ps.size()], q = ps[rng() % ps.size()];
            const std::size_t offset = rng() % 4;
            const Vec x = reference_chirp(c, s, q, n);
            const std::string what = name + " n=" + std::to_string(n) + " c=" + std::to_string(c) +
                                      " s=" + std::to_string(s) + " q=" + std::to_string(q);
            Guarded out(n, offset);
            fps::chirp::chirp<S, K>(c, s, q, out.span());
            check(out.guards_intact() && std::equal(x.begin(), x.end(), out.span().begin()), what + " chirp");
            const Vec f = random_vec(n);
            Vec expect(n);
            for (std::size_t k = 0; k < n; ++k) expect[k] = mul(f[k], x[k]);
            Guarded into(n, (offset + 1) % 4);
            fps::chirp::multiply_chirp<S, K>(c, s, q, f, into.span());
            check(into.guards_intact() && std::equal(expect.begin(), expect.end(), into.span().begin()),
                  what + " multiply_chirp");
            Guarded in_place(n, offset);
            std::copy(f.begin(), f.end(), in_place.span().begin());
            fps::chirp::multiply_chirp<S, K>(c, s, q, in_place.span());
            check(in_place.guards_intact() && std::equal(expect.begin(), expect.end(), in_place.span().begin()),
                  what + " multiply_chirp in place");
        }
    }
}

template <class S>
void test_struct() {
    test_terms<S, 1>();
    test_terms<S, 2>();
    test_terms<S, 4>();
    test_terms<S, 8>();
}

// products_scalar and products_lanes<S, K> against running products, both kinds of factors.
template <class U>
void check_products(const U& u, std::size_t n, const std::string& what) {
    namespace d = fps::chirp::detail;
    const u32 sp = random_word() | 1, si = random_word() | 1;
    Vec prefix(n + 1), inverse(n + 1);
    u32 run = 1;
    bool nonzero = true;
    for (std::size_t k = 0; k <= n; ++k) {
        prefix[k] = mul(sp, run);
        inverse[k] = mul(si, d::inverse(run));
        if (k < n) run = mul(run, u.at(k)), nonzero = nonzero && u.at(k) != 0;
    }
    const auto run_one = [&](auto products, const std::string& variant) {
        Guarded p(n + 1, rng() % 4), q(n + 1, rng() % 4);
        products(u, n, p.span().data(), sp, q.span().data(), si);
        check(p.guards_intact() && q.guards_intact() && std::equal(prefix.begin(), prefix.end(), p.span().begin()) &&
                  (!nonzero || std::equal(inverse.begin(), inverse.end(), q.span().begin())),
              what + " n=" + std::to_string(n) + " " + variant);
    };
    run_one([](auto&&... a) { d::products_scalar(a...); }, "scalar");
    run_one([](auto&&... a) { d::products(a...); }, "selected");
    fps::simd::for_each(fps::simd::All{}, [&]<class S> {
        const auto lanes = [&]<int K> {
            if (n >= std::size_t(K) * S::lanes * S::lanes)
                run_one([](auto&&... a) { d::products_lanes<S, K>(a...); }, std::string(S::name) + " K=" + std::to_string(K));
        };
        lanes.template operator()<1>();
        lanes.template operator()<2>();
        lanes.template operator()<4>();
    });
}

void test_products() {
    std::vector<std::size_t> ns;
    for (std::size_t n = 0; n < 80; ++n) ns.push_back(n);
    for (std::size_t n : {255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4095, 4096, 4097, 65537}) ns.push_back(n);
    for (int i = 0; i < 6; ++i) ns.push_back(100 + rng() % 20000);
    for (std::size_t n : ns) {
        check_products(fps::chirp::detail::Arithmetic{u32(1 + rng() % (P - 70000))}, n, "products arithmetic");
        check_products(fps::chirp::detail::Arithmetic{u32(P - n / 2)}, n, "products arithmetic through 0");
        check_products(fps::chirp::detail::Geometric(random_word(), random_word(), random_word() | 1), n, "products geometric");
    }
}

void test_evaluate() {
    const std::vector<std::size_t> small = {0, 1, 2, 3, 5, 8, 31, 32, 33, 64, 65, 100, 127, 200, 1000};
    for (std::size_t n : small)
        for (std::size_t m : small) {
            for (int trial = 0; trial < 3; ++trial) {
                const auto ps = parameters();
                const u32 a = ps[rng() % ps.size()], r = ps[rng() % ps.size()];
                const Vec f = random_vec(n);
                Guarded out(m, rng() % 4);
                fps::chirp::evaluate_geometric(f, a, r, out.span());
                bool ok = out.guards_intact();
                u32 x = a;
                for (std::size_t i = 0; i < m; ++i, x = mul(x, r)) ok = ok && out.span()[i] == horner(f, x);
                check(ok, "evaluate_geometric n=" + std::to_string(n) + " m=" + std::to_string(m) +
                              " a=" + std::to_string(a) + " r=" + std::to_string(r));
            }
        }
    // Larger: sampled points.
    const std::vector<std::pair<std::size_t, std::size_t>> large = {
        {1 << 12, 1 << 12}, {(1 << 16) + 3, (1 << 16) + 5}, {1 << 17, 10}, {10, 1 << 17}, {4097, 70000}, {70001, 4095}};
    for (const auto& [n, m] : large) {
        const u32 a = random_word(), r = random_word();
        const Vec f = random_vec(n);
        Vec out(m);
        fps::chirp::evaluate_geometric(f, a, r, out);
        bool ok = true;
        for (int t = 0; t < 40; ++t) {
            const std::size_t i = t < 3 ? std::size_t(t) : t < 6 ? m - 1 - (t - 3) : rng() % m;
            ok = ok && out[i] == horner(f, mul(a, fps::chirp::detail::power(r, i)));
        }
        check(ok, "evaluate_geometric large n=" + std::to_string(n) + " m=" + std::to_string(m));
    }
}

// The f of degree < n through (a r^i, y_i), by Lagrange in O(n^2).
Vec lagrange(u32 a, u32 r, const Vec& y) {
    const std::size_t n = y.size();
    Vec x(n), out(n, 0);
    for (std::size_t i = 0; i < n; ++i) x[i] = i ? mul(x[i - 1], r) : a;
    for (std::size_t i = 0; i < n; ++i) {
        Vec basis{1};  // prod_(j != i) (X - x_j)
        u32 denominator = 1;
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            Vec next(basis.size() + 1, 0);
            for (std::size_t k = 0; k < basis.size(); ++k) {
                next[k + 1] = (next[k + 1] + basis[k]) % P;
                next[k] = u32((next[k] + u64(P - x[j]) * basis[k]) % P);
            }
            basis = next;
            denominator = mul(denominator, (x[i] + P - x[j]) % P);
        }
        const u32 w = mul(y[i], fps::chirp::detail::inverse(denominator));
        for (std::size_t k = 0; k < n; ++k) out[k] = u32((out[k] + u64(w) * basis[k]) % P);
    }
    return out;
}

void test_interpolate() {
    for (std::size_t n : {0, 1, 2, 3, 4, 7, 8, 9, 31, 32, 33, 34, 63, 64, 65, 100, 127, 128, 129, 300}) {
        for (int trial = 0; trial < 3; ++trial) {
            const u32 a = trial == 0 ? 1 : random_word() | 1, r = trial == 1 ? 2 : random_word() | 1;
            const Vec y = random_vec(n);
            Vec out(n);
            fps::chirp::interpolate_geometric(a, r, y, out);
            check(out == lagrange(a, r, y), "interpolate_geometric n=" + std::to_string(n) + " a=" + std::to_string(a) +
                                                " r=" + std::to_string(r));
        }
    }
    // r = 0 with two points; r = P - 1 (order 2) with two.
    for (u32 r : {0u, P - 1}) {
        const Vec y = {5, 7};
        Vec out(2);
        fps::chirp::interpolate_geometric(3, r, y, out);
        check(out == lagrange(3, r, y), "interpolate_geometric n=2 r=" + std::to_string(r));
    }
    // Round trips.
    for (std::size_t n : {std::size_t(1000), std::size_t(1) << 12, (std::size_t(1) << 15) + 1, (std::size_t(1) << 16) + 3,
                          std::size_t(1) << 17}) {
        const u32 a = random_word() | 1, r = random_word() | 1;
        const Vec f = random_vec(n);
        Vec y(n), out(n);
        fps::chirp::evaluate_geometric(f, a, r, y);
        fps::chirp::interpolate_geometric(a, r, y, out);
        check(out == f, "interpolate_geometric round trip n=" + std::to_string(n));
    }
    // Errors: points not distinct; out of another size.
    const auto throws = [](u32 a, u32 r, std::size_t n, std::size_t out_size) {
        const Vec y(n, 1);
        Vec out(out_size, 12345);
        try {
            fps::chirp::interpolate_geometric(a, r, y, out);
        } catch (const std::invalid_argument&) {
            return std::all_of(out.begin(), out.end(), [](u32 x) { return x == 12345; });
        }
        return false;
    };
    // 3 has order P - 1; 3^((P - 1) / 7 / 17) has order 7 * 17 = 119.
    const u32 order119 = fps::chirp::detail::power(3, (P - 1) / 119);
    check(throws(0, 2, 2, 2), "throws a = 0");
    check(throws(5, 0, 3, 3), "throws r = 0, n = 3");
    check(throws(5, 1, 2, 2), "throws r = 1");
    check(throws(5, P - 1, 3, 3), "throws r of order 2");
    check(throws(5, order119, 120, 120), "throws r of order 119, n = 120");
    check(throws(5, order119, 1000, 1000), "throws r of order 119, n = 1000");
    check(throws(5, 2, 10, 9), "throws out size");
    Vec y(119, 3), out(119);
    for (std::size_t i = 0; i < 119; ++i) y[i] = random_word();
    fps::chirp::interpolate_geometric(5, order119, y, out);
    check(out == lagrange(5, order119, y), "interpolate_geometric r of order 119, n = 119");
}

// f(x) for the f of degree < n with f(i) = values[i]: Lagrange, O(n) products and one inverse.
u32 lagrange_at(const Vec& values, u32 x) {
    const std::size_t n = values.size();
    if (n == 0) return 0;
    if (x < n) return values[x];
    // f(x) = sum_i values[i] prod_(j != i) (x - j) / (i - j); prefix and suffix products of x - j.
    Vec prefix(n + 1, 1), suffix(n + 1, 1);
    for (std::size_t j = 0; j < n; ++j) prefix[j + 1] = mul(prefix[j], (x + P - u32(j)) % P);
    for (std::size_t j = n; j-- > 0;) suffix[j] = mul(suffix[j + 1], (x + P - u32(j)) % P);
    // 1 / (i! (n - 1 - i)!) from 1 / (n - 1)!.
    Vec inverse_fact(n);
    u32 fact = 1;
    for (std::size_t i = 1; i < n; ++i) fact = mul(fact, u32(i));
    inverse_fact[n - 1] = fps::chirp::detail::inverse(fact);
    for (std::size_t i = n - 1; i > 0; --i) inverse_fact[i - 1] = mul(inverse_fact[i], u32(i));
    u64 sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        u32 term = mul(mul(values[i], mul(prefix[i], suffix[i + 1])), mul(inverse_fact[i], inverse_fact[n - 1 - i]));
        if ((n - 1 - i) % 2) term = term ? P - term : 0;
        sum += term;
    }
    return u32(sum % P);
}

void test_extrapolate() {
    struct Case {
        std::size_t n, m;
        u32 c;
    };
    std::vector<Case> cases;
    for (std::size_t n : {1, 2, 3, 8, 33, 100})
        for (std::size_t m : {0, 1, 2, 7, 64, 130})
            for (u32 c : {0u, 1u, u32(n / 2), u32(n), u32(n + 1), 1000u, P - 1, P - 3, P - u32(m / 2), random_word()})
                cases.push_back({n, m, c});
    cases.push_back({0, 5, 3});
    cases.push_back({1 << 12, 1 << 12, random_word()});
    cases.push_back({1 << 12, 1 << 13, 100});           // inside, then past the samples
    cases.push_back({1 << 12, 1 << 13, P - 4000});      // before, the samples, after
    cases.push_back({5000, 300, 4900});
    cases.push_back({(1 << 16) + 3, (1 << 16) + 1, random_word()});
    cases.push_back({100000, 1000, P - 500});
    for (const auto& [n, m, c] : cases) {
        const Vec values = random_vec(n);
        Guarded out(m, rng() % 4);
        fps::chirp::extrapolate(values, c, out.span());
        bool ok = out.guards_intact();
        const bool all = n * m <= (1 << 20);
        for (std::size_t t = 0; t < (all ? m : 40); ++t) {
            const std::size_t k = all ? t : t < 20 ? t * (m / 20) : rng() % m;
            ok = ok && out.span()[k] == lagrange_at(values, u32((c + k) % P));
        }
        check(ok, "extrapolate n=" + std::to_string(n) + " m=" + std::to_string(m) + " c=" + std::to_string(c));
    }
}

}  // namespace

int main() {
    fps::simd::for_each(fps::simd::All{}, []<class S> { test_struct<S>(); });
    test_terms<fps::chirp::Wide, fps::chirp::kVectors>();
    test_products();
    test_evaluate();
    test_interpolate();
    test_extrapolate();
    std::printf("%s: %ld checks, %d failures\n", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
