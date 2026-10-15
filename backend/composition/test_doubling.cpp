// Tests of backend/composition's round-2 paths (issue #149 round 2): the doubled levels (every
// block size, m from 512 on, and the plain levels below; both parities of m - n),
// compositional_inverse on series::pow (n - 1 a power of two or not, both directions f(g) = g(f)
// = x), the scaled CRT of the hooks and their writes in place (V and W over the leaves read), on
// every simd struct.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <vector>

#include "backend/composition/composition.hpp"
#include "backend/series/series.hpp"
#include "backend/simd/select.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = 998244353;
namespace c = fps::composition;

int failures = 0;

void check(bool ok, const char* what, int line, u64 a = 0, u64 b = 0) {
    if (ok) return;
    if (++failures <= 20)
        std::printf("FAIL line %d: %s (%llu, %llu)\n", line, what, (unsigned long long)a, (unsigned long long)b);
}
#define CHECK(cond, ...) check(bool(cond), #cond, __LINE__ __VA_OPT__(, ) __VA_ARGS__)

std::mt19937_64 rng(149);

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 add(u32 a, u32 b) { return (a + b) % P; }
u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}

Vec random_vec(std::size_t n) {
    Vec a(n);
    for (auto& x : a) x = u32(rng() % P);
    return a;
}

Vec compose(const Vec& f, const Vec& g, std::size_t n) {
    Vec h(n, 12345);
    c::compose(f, g, h);
    return h;
}

// Sizes with m = 2^T from 128 (doubled levels from m = 512 on, blocks 32 .. m/4), m - n even and odd.
std::vector<std::size_t> sizes() {
    std::vector<std::size_t> s;
    for (std::size_t m = 128; m <= (std::size_t(1) << 14); m *= 2) s.insert(s.end(), {m / 2 + 1, m - 1, m});
    s.insert(s.end(), {std::size_t(1) << 17, (std::size_t(1) << 17) - 3});
    return s;
}

// compose against exp(g) and 1 / (1 - g); power projection against the transposition and the
// binomials of g = x / (1 - x).
void test_levels() {
    for (std::size_t n : sizes()) {
        Vec g = random_vec(n);
        g[0] = 0;
        Vec e(n), ones(n, 1), want(n), one_minus(n);
        u32 fact = 1;
        for (std::size_t k = 0; k < n; ++k) {
            if (k) fact = mul(fact, u32(k));
            e[k] = power(fact, P - 2);
        }
        fps::series::exp(g, want);
        CHECK(compose(e, g, n) == want, n);
        for (std::size_t k = 0; k < n; ++k) one_minus[k] = g[k] ? P - g[k] : 0;
        one_minus[0] = 1;
        fps::series::inverse(one_minus, want);
        CHECK(compose(ones, g, n) == want, n);

        const Vec f = random_vec(n);
        Vec a(n);
        c::power_projection(g, a);
        u32 dot = 0;
        for (std::size_t k = 0; k < n; ++k) dot = add(dot, mul(a[k], f[k]));
        CHECK(dot == compose(f, g, n)[n - 1], n);
        Vec h(n, 1), b(n);
        h[0] = 0;
        c::power_projection(h, b);
        u32 binomial = 1;  // C(n - 2, k - 1)
        bool ok = b[0] == 0;
        for (std::size_t k = 1; k < n; ++k) {
            ok = ok && b[k] == binomial;
            binomial = mul(mul(binomial, u32(n - 1 - k)), power(u32(k), P - 2));
        }
        CHECK(ok, n);
    }
}

// f(g) = g(f) = x mod x^n.
void test_inverse() {
    for (std::size_t n : {std::size_t(3), std::size_t(4), std::size_t(17), std::size_t(33), std::size_t(65), std::size_t(129),
                          std::size_t(1000), std::size_t(1025), std::size_t(4097), std::size_t(8192), std::size_t(1) << 15}) {
        Vec f = random_vec(n + 1);
        f[0] = 0;
        f[1] = f[1] ? f[1] : 1;
        Vec g(n, 12345);
        c::compositional_inverse(f, g);
        Vec x(n, 0);
        x[1] = 1;
        CHECK(compose(f, g, n) == x, n);
        CHECK(compose(g, f, n) == x, n);
    }
}

// The hooks' runs with a scaled Crt are c times the plain ones; projection_run in place (V over
// Q's leaves, W over P's) as out of place.
void test_hooks() {
    const std::size_t m = 512, pairs = m / 4;
    const auto& tables = c::detail::pair_tables(pairs);
    const Vec F = random_vec(4 * m), F2 = random_vec(4 * m);
    const u32 factor = u32(rng() % P);
    Vec table(pairs + 64);
    const c::detail::Crt scaled = c::detail::scaled_crt(tables, pairs, factor, table.data());
    fps::simd::for_each(fps::simd::All{}, [&]<class S> {
        using N = c::detail::Pairs<c::detail::Narrow<S>>;
        Vec G(2 * m), Gs(2 * m), V(2 * m), W(2 * m), Vs(2 * m), Ws(2 * m);
        N::graeffe_run(F.data(), G.data(), 0, pairs, tables);
        N::graeffe_run(F.data(), Gs.data(), 0, pairs, tables, scaled);
        N::projection_run(F.data(), F2.data(), V.data(), W.data(), 0, pairs, tables);
        N::projection_run(F.data(), F2.data(), Vs.data(), Ws.data(), 0, pairs, tables, scaled);
        bool ok = true;
        for (std::size_t i = 0; i < 2 * m; ++i)
            ok = ok && Gs[i] == mul(G[i], factor) && Vs[i] == mul(V[i], factor) && Ws[i] == mul(W[i], factor);
        CHECK(ok, S::lanes);
        // In place, in runs of 32 pairs from the start, as the hooks write.
        Vec Q = F, Pp = F2;
        for (std::size_t p0 = 0; p0 < pairs; p0 += 32)
            N::projection_run(Q.data() + 16 * p0, Pp.data() + 16 * p0, Q.data() + 8 * p0, Pp.data() + 8 * p0, p0, 32, tables);
        CHECK(std::equal(V.begin(), V.end(), Q.begin()) && std::equal(W.begin(), W.end(), Pp.begin()), S::lanes);
    });
}

}  // namespace

int main() {
    test_hooks();
    test_levels();
    test_inverse();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
