// Tests of backend/bostan_mori: every variant against a model (Bostan and Mori with O(n^2)
// products, and the recurrence q[0] s_i = p_i - sum q_j s_(i-j) for small k) on sizes 0, 1,
// powers of two and their neighbours, the transform lengths' edges, p longer and shorter than q,
// k from 0 to 2^64 - 1 and k below the sizes (the cuts), random, mostly zero and all P - 1 inputs;
// the halving pass and the schoolbook steps on every simd struct this machine runs, the pass
// against the values form of U and V; large sizes against the products variant; Fibonacci at
// 10^18; threads; the errors.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "backend/bostan_mori/bostan_mori.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/simd/select.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
namespace bm = fps::bostan_mori;
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

void check(bool ok, const char* what, int line, u64 a = 0, u64 b = 0, u64 c = 0) {
    if (ok) return;
    if (++failures <= 30)
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

std::mt19937_64 rng(114);

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 add(u32 a, u32 b) { return a + b >= P ? a + b - P : a + b; }
u32 sub(u32 a, u32 b) { return a >= b ? a - b : a + P - b; }
u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
u32 inv(u32 a) { return power(a, P - 2); }

// Kinds: 0 random, 1 mostly zeros, 2 all P - 1. first: a[0] (q[0] != 0).
Vec random_vec(std::size_t n, int kind, bool nonzero_first = false) {
    Vec a(n);
    for (u32& x : a) {
        if (kind == 0) x = u32(rng() % P);
        else if (kind == 1) x = rng() % 8 == 0 ? u32(rng() % P) : 0;
        else x = P - 1;
    }
    if (nonzero_first && n && a[0] == 0) a[0] = 1 + u32(rng() % (P - 1));
    return a;
}

Vec product(const Vec& f, const Vec& g) {
    if (f.empty() || g.empty()) return {};
    Vec out(f.size() + g.size() - 1);
    for (std::size_t i = 0; i < f.size(); ++i)
        for (std::size_t j = 0; j < g.size(); ++j) out[i + j] = add(out[i + j], mul(f[i], g[j]));
    return out;
}

Vec negate_odd(Vec f) {
    for (std::size_t i = 1; i < f.size(); i += 2) f[i] = f[i] ? P - f[i] : 0;
    return f;
}

Vec every_other(const Vec& f, std::size_t r) {
    Vec g;
    for (std::size_t i = r; i < f.size(); i += 2) g.push_back(f[i]);
    return g;
}

// The model: Bostan and Mori with schoolbook products, p and q cut to k + 1 words.
u32 model(Vec p, Vec q, u64 k) {
    const auto cut = [](Vec& f, u64 k) {
        if (k < f.size()) f.resize(std::size_t(k) + 1);
    };
    cut(p, k), cut(q, k);
    for (; k > 0; k /= 2) {
        if (p.empty()) return 0;
        const Vec minus = negate_odd(q);
        p = every_other(product(p, minus), k % 2);
        q = every_other(product(q, minus), 0);
        cut(p, k / 2), cut(q, k / 2);
    }
    return p.empty() ? 0 : mul(p[0], inv(q[0]));
}

// The first n terms of p / q by the recurrence.
Vec series(const Vec& p, const Vec& q, std::size_t n) {
    Vec s(n);
    const u32 qi = inv(q[0]);
    for (std::size_t i = 0; i < n; ++i) {
        u32 x = i < p.size() ? p[i] : 0;
        for (std::size_t j = 1; j < q.size() && j <= i; ++j) x = sub(x, mul(q[j], s[i - j]));
        s[i] = mul(x, qi);
    }
    return s;
}

// Indices: small, around powers of two, 10^18, the largest.
const std::vector<u64> kHuge = {(u64(1) << 20) - 1, u64(1) << 32, (u64(1) << 40) + 1, 1'000'000'000'000'000'000ull,
                                (u64(1) << 63) - 1, u64(1) << 63, ~u64(0) - 1, ~u64(0)};

// Every variant on (p, q, k) against `want`.
void check_all(const Vec& p, const Vec& q, u64 k, u32 want, int line) {
    const auto cut = [k](std::size_t n) { return k < n ? std::size_t(k) + 1 : n; };
    const std::size_t n = std::max(cut(p.size()), cut(q.size()));
    const u32 got = bm::coefficient(p, q, k);
    check(got == want, "coefficient", line, p.size(), q.size(), k);
    check(bm::variant::values(p, q, k, 0) == want, "values, no schoolbook", line, p.size(), q.size(), k);
    check(bm::variant::values(p, q, k, 16) == want, "values, schoolbook from 16", line, p.size(), q.size(), k);
    check(bm::variant::values(p, q, k, bm::kSchoolbookCapacity) == want, "values, schoolbook from capacity", line,
          p.size(), q.size(), k);
    if (n <= bm::kSchoolbookCapacity)
        check(bm::variant::schoolbook(p, q, k) == want, "schoolbook", line, p.size(), q.size(), k);
    check(bm::variant::products(p, q, k) == want, "products", line, p.size(), q.size(), k);
}

// Sizes 0 .. 65 around powers of two, every k up to 70 against the recurrence, huge k against the
// model.
void small_sizes() {
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65};
    for (std::size_t np : sizes)
        for (std::size_t nq : sizes) {
            if (nq == 0) continue;
            for (int kind = 0; kind < 3; ++kind) {
                const Vec p = random_vec(np, kind), q = random_vec(nq, kind, true);
                const bool few = np > 17 && nq > 17 && kind != 0;  // fewer cases for the larger pairs
                const Vec s = series(p, q, 71);
                for (u64 k = 0; k <= 70; k += few ? 7 : 1) check_all(p, q, k, s[k], __LINE__);
                for (u64 k : kHuge) {
                    if (few && k != kHuge[3]) continue;
                    check_all(p, q, k, model(p, q, k), __LINE__);
                }
            }
        }
}

// Lengths of the values form at their edges (N = 64 .. 2048: deg q and deg p + deg q around N and
// 2N), p longer and shorter than q, k below the sizes (the cut shortens the transforms).
void transform_edges() {
    const std::vector<std::size_t> sizes = kSanitized ? std::vector<std::size_t>{63, 64, 65, 127, 128, 129, 257, 600}
                                                      : std::vector<std::size_t>{63, 64, 65, 127, 128, 129, 255, 256,
                                                                                 257, 511, 512, 513, 1025, 2049};
    for (std::size_t d : sizes) {
        for (int kind = 0; kind < 3; ++kind) {
            const std::size_t shapes[][2] = {{d, d + 1}, {d + 1, d}, {d, d}, {2 * d, d / 2 + 1}, {3, d}, {d, 3}, {1, d}};
            for (const auto& shape : shapes) {
                const Vec p = random_vec(shape[0], kind), q = random_vec(shape[1], kind, true);
                const bool slow = std::max(shape[0], shape[1]) > 700;  // the model is O(n^2) per step
                const u64 ks[] = {1, 2, d / 2, d - 1, d, d + 1, 2 * d + 1, 3 * d, 1000003, kHuge[3], ~u64(0)};
                for (u64 k : ks) {
                    const u32 want = slow ? bm::variant::products(p, q, k) : model(p, q, k);
                    check_all(p, q, k, want, __LINE__);
                }
            }
        }
    }
}

// Large sizes against products (and the recurrence where k is small).
void large() {
    const std::vector<std::size_t> sizes = kSanitized ? std::vector<std::size_t>{4097, 10000}
                                                      : std::vector<std::size_t>{4095, 4096, 4097, 20000, 70000};
    for (std::size_t d : sizes) {
        const Vec p = random_vec(d, 0), q = random_vec(d + 1, 0, true);
        for (u64 k : {u64(d / 3), u64(5 * d / 2), u64(1) << 40, ~u64(0)}) {
            const u32 want = bm::variant::products(p, q, k);
            CHECK(bm::coefficient(p, q, k) == want, d, k);
            CHECK(bm::variant::values(p, q, k, 0) == want, d, k);
        }
        const Vec s = series(p, q, 300);
        CHECK(bm::coefficient(p, q, 299) == s[299], d);
    }
}

// Fibonacci: F_k by fast doubling.
u32 fibonacci(u64 k) {
    u32 a = 0, b = 1;  // F_n, F_(n+1)
    for (int bit = 63; bit >= 0; --bit) {
        const u32 c = mul(a, sub(mul(2, b), a)), d = add(mul(a, a), mul(b, b));  // F_2n, F_(2n+1)
        if (k >> bit & 1) a = d, b = add(c, d);
        else a = c, b = d;
    }
    return a;
}

void closed_forms() {
    const Vec p = {0, 1}, q = {1, P - 1, P - 1};
    for (u64 k : {u64(0), u64(1), u64(2), u64(10), u64(90), u64(1) << 33, u64(1'000'000'000'000'000'000ull), ~u64(0)})
        check_all(p, q, k, fibonacci(k), __LINE__);
    // 1 / (1 - c x)^2: (k + 1) c^k
    const u32 c = 12345;
    const Vec q2 = {1, P - mul(2, c), mul(c, c)};
    for (u64 k : {u64(5), u64(1) << 50, ~u64(0)}) check_all({1}, q2, k, mul(u32((k % P + 1) % P), power(c, k)), __LINE__);
    // x^100 / (1 - x): 1 from k = 100 on; and a p longer than k
    Vec x100(101);
    x100[100] = 1;
    for (u64 k : {u64(0), u64(99), u64(100), u64(1) << 40}) check_all(x100, {1, P - 1}, k, k >= 100, __LINE__);
}

// The halving pass on every struct, against the values forms of U and V (scaled) at length N.
template <class S>
void pass_struct() {
    if constexpr (S::lanes <= 8) {
        for (int lg = 7; lg <= (kSanitized ? 11 : 13); ++lg) {
            const std::size_t L = std::size_t(1) << lg, n = L / 2;
            const fps::ntt::Transform tl(lg), tn(lg - 1);
            for (int kind = 0; kind < 3; ++kind)
                for (int odd = 0; odd < 2; ++odd) {
                    const Vec p = random_vec(n, kind), q = random_vec(n, kind, true);
                    // aligned buffers: L + padding words
                    std::vector<u32> buffer(4 * (L + 64));
                    u32* pv = buffer.data() + (64 - reinterpret_cast<std::uintptr_t>(buffer.data()) / 4 % 16) % 16;
                    u32* qv = pv + L + 64;
                    u32* want_u = qv + L + 64;
                    u32* want_v = want_u + L + 64;
                    std::fill_n(pv, L + 16, 0u), std::fill_n(qv, L + 16, 0u);
                    std::copy(p.begin(), p.end(), pv), std::copy(q.begin(), q.end(), qv);
                    tl.forward(pv, fps::ntt::Form::values), tl.forward(qv, fps::ntt::Form::values);
                    const u32* winv = bm::detail::inverse_points(n);
                    if (odd) bm::detail::Pass<S>::template run<true>(pv, qv, L, winv);
                    else bm::detail::Pass<S>::template run<false>(pv, qv, L, winv);
                    const Vec minus = negate_odd(q);
                    const Vec u = every_other(product(p, minus), std::size_t(odd)), v = every_other(product(q, minus), 0);
                    const u32 su = inv(power(2, 31)), sv = inv(power(2, 32));  // U times 2^-31, V times 2^-32
                    std::fill_n(want_u, n + 16, 0u), std::fill_n(want_v, n + 16, 0u);
                    for (std::size_t i = 0; i < u.size(); ++i) want_u[i] = mul(u[i], su);
                    for (std::size_t i = 0; i < v.size(); ++i) want_v[i] = mul(v[i], sv);
                    tn.forward(want_u, fps::ntt::Form::values), tn.forward(want_v, fps::ntt::Form::values);
                    CHECK(std::equal(pv, pv + n, want_u), lg, kind, odd);
                    CHECK(std::equal(qv, qv + n, want_v), lg, kind, odd);
                }
        }
    }
}

// The schoolbook steps on every struct against the model, up to the capacity.
template <class S>
void schoolbook_struct() {
    if constexpr (S::lanes <= 8) {
        const std::vector<std::size_t> sizes = {1, 2, 3, 5, 8, 9, 16, 17, 31, 33, 64, 100, 128, 129, 200, 255, 256};
        for (std::size_t np : sizes)
            for (std::size_t nq : sizes) {
                if (kSanitized && np * nq > 20000) continue;
                for (int kind = 0; kind < 3; ++kind) {
                    const Vec p = random_vec(np, kind), q = random_vec(nq, kind, true);
                    for (u64 k : {u64(1), u64(np + nq), kHuge[3], ~u64(0)}) {
                        if (std::max(np, nq) > bm::kSchoolbookCapacity && k >= bm::kSchoolbookCapacity) continue;
                        const auto [a, b] = bm::detail::trim(p, q, k);
                        if (a.empty()) continue;
                        CHECK(bm::detail::Schoolbook<S>::run(a, b, k) == model(p, q, k), np, nq, k);
                    }
                }
            }
    }
}

// The scalar steps for tiny sizes against the model.
void tiny() {
    for (std::size_t np = 1; np <= bm::detail::kTinyMax; ++np)
        for (std::size_t nq = 1; nq <= bm::detail::kTinyMax; ++nq)
            for (int kind = 0; kind < 3; ++kind) {
                const Vec p = random_vec(np, kind), q = random_vec(nq, kind, true);
                for (u64 k : {u64(1), u64(2), u64(3), u64(7), kHuge[3], ~u64(0)}) {
                    const auto [a, b] = bm::detail::trim(p, q, k);
                    if (a.empty()) continue;
                    CHECK(bm::detail::tiny(a, b, k) == model(p, q, k), np, nq, k);
                }
            }
}

void structs() {
    fps::simd::for_each(fps::simd::All{}, []<class S> {
        pass_struct<S>();
        schoolbook_struct<S>();
    });
}

void threads() {
    std::vector<std::thread> pool;
    int bad[4] = {};
    for (int t = 0; t < 4; ++t)
        pool.emplace_back([t, &bad] {
            std::mt19937_64 r{u64(t)};
            for (int i = 0; i < 6; ++i) {
                const std::size_t d = std::size_t(1) << (5 + 2 * ((t + i) % 4));
                Vec p(d), q(d + 1);
                for (u32& x : p) x = u32(r() % P);
                for (u32& x : q) x = u32(r() % P);
                q[0] = 1;
                const u64 k = r();
                if (bm::coefficient(p, q, k) != bm::variant::products(p, q, k)) ++bad[t];
            }
        });
    for (auto& t : pool) t.join();
    for (int t = 0; t < 4; ++t) CHECK(bad[t] == 0, t);
}

void errors() {
    CHECK(throws<std::domain_error>([] { bm::coefficient(Vec{1}, Vec{}, 5); }));
    CHECK(throws<std::domain_error>([] { bm::coefficient(Vec{1}, Vec{0, 1}, 5); }));
    CHECK(throws<std::domain_error>([] { bm::variant::values(Vec{1}, Vec{0, 1}, 5); }));
    CHECK(throws<std::domain_error>([] { bm::variant::schoolbook(Vec{1}, Vec{0}, 5); }));
    CHECK(throws<std::domain_error>([] { bm::variant::products(Vec{1}, Vec{}, 5); }));
    CHECK(throws<std::length_error>([] { bm::variant::schoolbook(Vec(300, 1), Vec(300, 1), 1000); }));
    CHECK(bm::coefficient(Vec{}, Vec{3}, 0) == 0);
    CHECK(bm::coefficient(Vec{0, 0, 0}, Vec{3, 4}, 7) == 0);
    CHECK(bm::coefficient(Vec{6}, Vec{3}, 0) == 2);
    CHECK(bm::coefficient(Vec{6}, Vec{3}, 1) == 0);
    CHECK(bm::coefficient(Vec{6}, Vec{3, 0, 0}, ~u64(0)) == 0);
}

}  // namespace

int main() {
    errors();
    closed_forms();
    tiny();
    structs();
    small_sizes();
    transform_edges();
    large();
    threads();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("bostan_mori: all tests passed\n");
    return 0;
}
