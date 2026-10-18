// Tests for issue #136 round 3: the interpolation's groups of bases (one inversion per group;
// interpolate_tree at slot counts around multiples of kInterpolateGroup and of its bases, short
// last slots, exact workspaces; repeated points in each base of a group and past it throw), the
// Newton basis' deferred combine (combine_product on every simd struct the machine runs against
// scalar loops; newton_tree at C = L and C < L checked by its nested form at random points) and
// the descent's root window read in place (evaluate_tree at C = L). Exits nonzero on a failure.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/evaluation/evaluation.hpp"

namespace {

namespace d = fps::evaluation::detail;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
constexpr u32 P = 998244353;

std::mt19937_64 rng(1363);
int failures = 0, checks = 0;

void expect(bool ok, const std::string& what, u64 detail = 0) {
    ++checks;
    if (!ok && ++failures <= 30) std::printf("FAIL: %s (%llu)\n", what.c_str(), static_cast<unsigned long long>(detail));
}

u32 mul(u32 x, u32 y) { return u32(u64(x) * y % P); }
u32 add(u32 x, u32 y) { return u32((u64(x) + y) % P); }
u32 sub(u32 x, u32 y) { return u32((u64(x) + P - y) % P); }

u32 horner(std::span<const u32> f, u32 x) {
    u32 r = 0;
    for (std::size_t i = f.size(); i-- > 0;) r = add(mul(r, x), f[i]);
    return r;
}

std::vector<u32> random_words(std::size_t n) {
    std::vector<u32> v(n);
    for (u32& x : v) x = u32(rng() % P);
    return v;
}

// m distinct points.
std::vector<u32> distinct(std::size_t m) {
    std::vector<u32> v = random_words(m + m / 4 + 16);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    v.resize(m);
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

// Indices: all below 600, else the ends, every 8th of the first group and 64 random ones.
std::vector<std::size_t> samples(std::size_t m) {
    std::vector<std::size_t> s;
    if (m <= 600) {
        for (std::size_t i = 0; i < m; ++i) s.push_back(i);
        return s;
    }
    for (std::size_t i = 0; i < 8 * fps::evaluation::kInterpolateGroup && i < m; i += 7) s.push_back(i);
    s.push_back(m - 1), s.push_back(m - 2);
    for (int i = 0; i < 64; ++i) s.push_back(rng() % m);
    return s;
}

void test_interpolate(std::size_t m) {
    const auto points = distinct(m), values = random_words(m);
    std::vector<u32> f(m, 777);
    d::workspace_spilled = false;
    d::interpolate_tree(points, values, f);
    expect(!d::workspace_spilled, "interpolate_tree memory as computed", m);
    bool ok = true;
    for (std::size_t i : samples(m)) ok &= horner(f, points[i]) == values[i];
    expect(ok, "interpolate_tree", m);
}

// A repeated point: points[j] = points[i].
void test_repeated(std::size_t m, std::size_t i, std::size_t j) {
    auto points = distinct(m);
    points[j] = points[i];
    const auto values = random_words(m);
    std::vector<u32> f(m);
    bool threw = false;
    try {
        d::interpolate_tree(points, values, f);
    } catch (const std::domain_error&) {
        threw = true;
    }
    expect(threw, "interpolate_tree repeated " + std::to_string(i) + " " + std::to_string(j), m);
}

void test_groups() {
    const std::size_t G = fps::evaluation::kInterpolateGroup;
    std::vector<std::size_t> slots;
    for (std::size_t base : {G, 2 * G, 3 * G, 4 * G, 8 * G})
        for (std::size_t s : {base - 1, base, base + 1, base + 2}) slots.push_back(s);
    slots.push_back(17 * G + 9);
    for (std::size_t s : slots)
        for (std::size_t r : {0, 3, 7})
            if (8 * s > r && s >= fps::evaluation::kMinSlots) test_interpolate(8 * s - r);
    // Points 8 k + l are slot k of lane l: repeats within one base, across a group's bases, in
    // the padded last slots and across groups.
    const std::size_t m = 8 * (4 * G + 3) - 5;
    for (std::size_t base = 0; base < G / 16; ++base) test_repeated(m, 8 * 16 * base + 3, 8 * 16 * base + 100);
    test_repeated(m, 1, 8 * (G - 1) + 7);
    test_repeated(m, 8 * G + 5, 8 * (2 * G - 1));
    test_repeated(m, 2, m - 1);
    test_repeated(m, m - 2, m - 1);
}

// f against sum_k c[k] prod_(i < k) (x - p_i) at random x (nested form, from the top).
void test_newton(std::size_t n) {
    const auto f = random_words(n), points = random_words(n);
    std::vector<u32> c(n);
    d::workspace_spilled = false;
    d::newton_tree(f, points, c);
    expect(!d::workspace_spilled, "newton_tree memory as computed", n);
    bool ok = true;
    for (int t = 0; t < 8; ++t) {
        const u32 x = u32(rng() % P);
        u32 r = 0;
        for (std::size_t k = n; k-- > 0;) r = add(mul(r, sub(x, points[k])), c[k]);
        ok &= r == horner(f, x);
    }
    expect(ok, "newton_tree", n);
}

void test_evaluate(std::size_t n, std::size_t m) {
    const auto f = random_words(n), points = random_words(m);
    std::vector<u32> values(m);
    d::workspace_spilled = false;
    d::evaluate_tree(f, points, values);
    expect(!d::workspace_spilled, "evaluate_tree memory as computed", m);
    bool ok = true;
    for (std::size_t i : samples(m)) ok &= values[i] == horner(f, points[i]);
    expect(ok, "evaluate_tree", n * 1000003 + m);
}

template <class S, int kSign>
void test_combine_product(const char* name, std::size_t n) {
    auto s = random_words(2 * n);
    const auto b = random_words(n), keep = s;
    std::vector<u32> out(n + 16, 0xdeadbeef);
    d::combine_product<S, kSign>(s.data(), b.data(), out.data(), n);
    bool ok = out[n] == 0xdeadbeef;
    for (std::size_t i = 0; i < n; ++i) {
        const bool minus = kSign < 0 || (kSign == 2 && i % 2 == 0);
        const u32 x = minus ? sub(keep[i], keep[n + i]) : add(keep[i], keep[n + i]);
        ok &= s[i] == x && s[n + i] == keep[n + i] && out[i] == fps::product_tree::detail::montgomery(x, b[i]);
    }
    expect(ok, std::string("combine_product<") + name + ", " + std::to_string(kSign) + ">", n);
}

template <class S>
void test_struct(const char* name) {
    for (std::size_t n : {16, 32, 64, 512, 4096}) {
        test_combine_product<S, -1>(name, n);
        test_combine_product<S, 1>(name, n);
        test_combine_product<S, 2>(name, n);
    }
}

}  // namespace

int main() {
    d::workspace_exact = true;
    test_groups();
    for (std::size_t n : {136, 137, 200, 256, 512, 520, 700, 1024, 1500, 2048, 4096, 5000, 8192, 1 << 15})
        test_newton(n), test_evaluate(n, n);
    test_evaluate(1 << 14, 1 << 12);
    test_struct<fps::simd::Scalar>("Scalar");
    test_struct<fps::product_tree::Lanes>("Lanes");
    test_struct<fps::product_tree::Wide>("Wide");
    std::printf("%s evaluation groups: %d checks, %d failures\n", failures ? "FAIL" : "pass", checks, failures);
    return failures != 0;
}
