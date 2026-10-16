// Tests for issue #136 round 1: the direct paths for few points, on every simd struct the target
// has: product_of_linear against scalar loops, interpolate_small (f(a_i) = y_i exactly, repeated
// points refused), newton_small with 1, 2 and 4 vectors against scalar synthetic divisions; and
// the API on both sides of kDirectInterpolate and kDirectNewton.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/evaluation/evaluation.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
constexpr u32 P = 998244353;
namespace d = fps::evaluation::detail;

std::mt19937_64 rng(1361);
int failures = 0, checks = 0;

void expect(bool ok, const std::string& what, u64 detail = 0) {
    ++checks;
    if (ok) return;
    if (++failures <= 30) std::printf("FAIL: %s (%llu)\n", what.c_str(), static_cast<unsigned long long>(detail));
}

u32 mul(u32 x, u32 y) { return u32(u64(x) * y % P); }
u32 add(u32 x, u32 y) { return u32((u64(x) + y) % P); }
u32 sub(u32 x, u32 y) { return u32((u64(x) + P - y) % P); }

u32 horner(std::span<const u32> f, u32 x) {
    u32 r = 0;
    for (std::size_t i = f.size(); i-- > 0;) r = add(mul(r, x), f[i]);
    return r;
}

// Words of a kind: random, all P - 1, all 0, small.
std::vector<u32> words(std::size_t n, int kind) {
    std::vector<u32> v(n);
    for (u32& x : v) x = kind == 0 ? u32(rng() % P) : kind == 1 ? P - 1 : kind == 2 ? 0 : u32(rng() % 10);
    return v;
}

std::vector<u32> distinct(std::size_t m) {
    std::vector<u32> v;
    if (m > 2) v.push_back(0), v.push_back(P - 1);
    while (v.size() < m) {
        const u32 x = u32(rng() % P);
        if (std::find(v.begin(), v.end(), x) == v.end()) v.push_back(x);
    }
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

std::vector<std::size_t> counts() {
    std::vector<std::size_t> s;
    for (std::size_t m = 1; m <= 70; ++m) s.push_back(m);
    for (std::size_t m : {95, 96, 97, 127, 128, 129, 200, 255, 256, 257, 300}) s.push_back(m);
    return s;
}

template <class S>
void test_struct() {
    const std::string name = S::name;
    for (std::size_t m : counts()) {
        // M = prod (x - a_i)
        const auto a = m % 3 ? distinct(m) : words(m, int(m % 4));
        std::vector<u32> big(m + 1 + 16), ref(m + 1, 0);
        ref[0] = 1;
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = i + 1; j > 0; --j) ref[j] = sub(ref[j - 1], mul(a[i], ref[j]));
            ref[0] = sub(0, mul(a[i], ref[0]));
        }
        d::product_of_linear<S>(a.data(), m, big.data());
        expect(std::equal(ref.begin(), ref.end(), big.begin()), "product_of_linear " + name, m);
        // interpolate_small, distinct points
        const auto points = distinct(m);
        const auto y = words(m, int(rng() % 4));
        d::product_of_linear<S>(points.data(), m, big.data());
        std::vector<u32> scratch((S::lanes + 2) * m + 16), f(m);
        const auto check = [&](bool ok, const char* chains) {
            for (std::size_t i = 0; i < m; ++i) ok &= horner(f, points[i]) == y[i];
            expect(ok, "interpolate_small " + name + " chains " + chains, m);
        };
        check(d::interpolate_small<S, 1>(big.data(), m, points.data(), y.data(), f.data(), scratch.data()), "1");
        check(d::interpolate_small<S, 2>(big.data(), m, points.data(), y.data(), f.data(), scratch.data()), "2");
        check(d::interpolate_small<S, 4>(big.data(), m, points.data(), y.data(), f.data(), scratch.data()), "4");
        if (m >= 2) {
            auto twice = points;
            twice[m - 1] = twice[m / 3];
            d::product_of_linear<S>(twice.data(), m, big.data());
            expect(!d::interpolate_small<S>(big.data(), m, twice.data(), y.data(), f.data(), scratch.data()),
                   "interpolate_small repeated " + name, m);
        }
        // newton_small against synthetic divisions, repeated and zero points among them
        const std::size_t n = m;
        const auto g = words(n, int(n % 4));
        auto p = words(n, int((n + 1) % 4));
        if (n > 5) p[1] = p[4] = 0, p[2] = p[3];
        std::vector<u32> want = g;
        for (std::size_t k = 0; k + 1 < n; ++k)
            for (std::size_t j = n - 1; j-- > k;) want[j] = add(want[j], mul(want[j + 1], p[k]));
        std::vector<u32> e(n + 8 * S::lanes + 16);
        const auto newton = [&]<std::size_t K>(std::integral_constant<std::size_t, K>) {
            std::vector<u32> c = g;
            d::newton_small<S, K>(c.data(), n, p.data(), e.data());
            expect(c == want, "newton_small " + name + " vectors " + std::to_string(K), n);
        };
        newton(std::integral_constant<std::size_t, 1>{});
        newton(std::integral_constant<std::size_t, 2>{});
        newton(std::integral_constant<std::size_t, 4>{});
    }
}

// The API across the cutoffs.
void test_api() {
    using namespace fps::evaluation;
    for (std::size_t m : {kDirectInterpolate - 1, kDirectInterpolate, kDirectInterpolate + 1}) {
        const auto points = distinct(m);
        const auto y = words(m, 0);
        std::vector<u32> f(m);
        interpolate(points, y, f);
        bool ok = true;
        for (std::size_t i = 0; i < m; ++i) ok &= horner(f, points[i]) == y[i];
        expect(ok, "interpolate at the cutoff", m);
    }
    for (std::size_t n : {kDirectNewton - 1, kDirectNewton, kDirectNewton + 1}) {
        const auto f = words(n, 0), p = words(n, 0);
        std::vector<u32> c(n);
        to_newton_basis(f, std::span<const u32>(p).first(n - 1), c);
        std::vector<u32> want = f;
        for (std::size_t k = 0; k + 1 < n; ++k)
            for (std::size_t j = n - 1; j-- > k;) want[j] = add(want[j], mul(want[j + 1], p[k]));
        expect(c == want, "to_newton_basis at the cutoff", n);
    }
}

}  // namespace

int main() {
    fps::simd::for_each(fps::simd::All{}, []<class S> { test_struct<S>(); });
    test_api();
    std::printf("%s evaluation small: %d checks, %d failures\n", failures ? "FAIL" : "pass", checks, failures);
    return failures ? 1 : 0;
}
