// Tests for backend/evaluation: evaluate against Horner's rule, interpolate by evaluating its
// result at the points, to_newton_basis by expanding the Newton form back; the tree paths at every
// size from the smallest tree (detail::*_tree) and the API on n, m in {0, 1, 2^k +- 1, random},
// repeated and zero points, values at the edges (0, P - 1). Large sizes are checked at sampled
// points. Kernels: middle and cross products against schoolbook, Horner, batch inversion.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "backend/evaluation/evaluation.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
constexpr u32 P = 998244353;

std::mt19937_64 rng(104);
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

// Words of a kind: random, all P - 1, all 0, small (< 10).
std::vector<u32> words(std::size_t n, int kind) {
    std::vector<u32> v(n);
    for (u32& x : v) x = kind == 0 ? u32(rng() % P) : kind == 1 ? P - 1 : kind == 2 ? 0 : u32(rng() % 10);
    return v;
}

// m distinct random points (some 0 and P - 1 among them).
std::vector<u32> distinct(std::size_t m) {
    std::vector<u32> v;
    v.reserve(m);
    if (m > 2) v.push_back(0), v.push_back(P - 1);
    while (v.size() < m) v.push_back(u32(rng() % P));
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    while (v.size() < m) {
        const u32 x = u32(rng() % P);
        if (!std::binary_search(v.begin(), v.end(), x)) v.insert(std::lower_bound(v.begin(), v.end(), x), x);
    }
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

// Sampled indices: all below `all`, else 64 random ones and the ends.
std::vector<std::size_t> samples(std::size_t m, std::size_t all = 3000) {
    std::vector<std::size_t> s;
    if (m <= all) {
        for (std::size_t i = 0; i < m; ++i) s.push_back(i);
        return s;
    }
    s = {0, 1, m / 2, m - 2, m - 1};
    for (int i = 0; i < 64; ++i) s.push_back(rng() % m);
    return s;
}

void check_evaluate(std::span<const u32> f, std::span<const u32> points, std::span<const u32> values, const std::string& name) {
    bool ok = true;
    for (std::size_t i : samples(points.size(), 4'000'000 / (f.size() + 1))) ok &= values[i] == horner(f, points[i]);
    expect(ok, name, f.size() * 1000003 + points.size());
}

using Evaluate = void (*)(std::span<const u32>, std::span<const u32>, std::span<u32>);

void test_evaluate(Evaluate run, const std::string& name, std::size_t n, std::size_t m, int kind) {
    const auto f = words(n, kind);
    auto points = words(m, kind == 3 ? 3 : 0);
    if (kind == 1 && m > 4) points[1] = points[3] = 0, points[2] = P - 1;
    const std::vector<u32> keep = points;
    std::vector<u32> values(m, 12345);
    fps::evaluation::detail::workspace_spilled = false;
    run(f, points, values);
    expect(!fps::evaluation::detail::workspace_spilled, name + ": memory as computed", m);
    expect(points == keep, name + ": points kept");
    check_evaluate(f, points, values, name + " kind " + std::to_string(kind));
}

void test_interpolate(void (*run)(std::span<const u32>, std::span<const u32>, std::span<u32>), const std::string& name, std::size_t m,
                      int kind) {
    const auto points = distinct(m);
    const auto values = words(m, kind);
    std::vector<u32> f(m, 777);
    fps::evaluation::detail::workspace_spilled = false;
    run(points, values, f);
    expect(!fps::evaluation::detail::workspace_spilled, name + ": memory as computed", m);
    bool ok = true;
    for (std::size_t i : samples(m, 4'000'000 / (m + 1))) ok &= horner(f, points[i]) == values[i];
    expect(ok, name + " kind " + std::to_string(kind), m);
}

// f against sum_k c[k] prod_(i < k) (x - p_i): exactly for small n, else at sampled x.
void test_newton(void (*run)(std::span<const u32>, std::span<const u32>, std::span<u32>), const std::string& name, std::size_t n,
                 int kind, bool repeated) {
    const auto f = words(n, kind);
    auto points = words(n > 0 ? n - 1 : 0, 0);
    if (repeated && points.size() > 4) points[2] = points[1] = points[3];
    std::vector<u32> c(n, 999);
    fps::evaluation::detail::workspace_spilled = false;
    run(f, points, c);
    expect(!fps::evaluation::detail::workspace_spilled, name + ": memory as computed", n);
    bool ok = true;
    if (n <= 2000) {
        std::vector<u32> g{};  // g = c[n-1]; g = g (x - p_k) + c[k]
        for (std::size_t k = n; k-- > 0;) {
            std::vector<u32> h(g.size() + 1, 0);
            for (std::size_t j = 0; j < g.size(); ++j) h[j + 1] = add(h[j + 1], g[j]), h[j] = sub(h[j], mul(points[k], g[j]));
            if (g.empty()) h.pop_back();
            if (h.empty()) h.push_back(0);
            h[0] = add(h[0], c[k]);
            g = h;
        }
        g.resize(n);
        ok = g == f;
    } else {
        for (int t = 0; t < 8; ++t) {
            const u32 x = u32(rng() % P);
            u32 v = 0;
            for (std::size_t k = n; k-- > 0;) v = add(mul(v, sub(x, k < points.size() ? points[k] : 0)), c[k]);
            ok &= v == horner(f, x);
        }
    }
    expect(ok, name + " kind " + std::to_string(kind) + (repeated ? " repeated" : ""), n);
}

std::vector<std::size_t> sizes(std::size_t top) {
    std::vector<std::size_t> s = {1, 2, 3, 7, 8, 9, 31, 32, 33, 100};
    for (std::size_t p = 64; p <= top; p *= 2) s.insert(s.end(), {p - 1, p, p + 1});
    for (int i = 0; i < 6; ++i) s.push_back(1 + rng() % top);
    std::erase_if(s, [top](std::size_t x) { return x > top; });
    return s;
}

#if __has_feature(address_sanitizer)
constexpr std::size_t kTop = std::size_t(1) << 14;
#else
constexpr std::size_t kTop = std::size_t(1) << 18;
#endif

void test_trees() {
    namespace d = fps::evaluation::detail;
    for (std::size_t m : sizes(kTop)) {
        for (std::size_t n : {std::size_t(1), std::size_t(2), m / 3 + 1, m, m + 5, 3 * m + 1}) {
            if (n * m > (std::size_t(1) << 32) / 64 && n != m) continue;
            test_evaluate(d::evaluate_tree, "evaluate_tree", n, m, n == m ? int(rng() % 4) : 0);
        }
        test_interpolate(d::interpolate_tree, "interpolate_tree", m, int(rng() % 4));
        test_newton(d::newton_tree, "newton_tree", m, int(rng() % 4), m % 2);
    }
    // Every count of slots from the smallest tree to 3 lane levels, at both paddings.
    for (std::size_t m = 8 * fps::evaluation::kMinSlots - 7; m <= 8 * 70; m += 5) {
        test_evaluate(d::evaluate_tree, "evaluate_tree slots", m + 3, m, 0);
        test_interpolate(d::interpolate_tree, "interpolate_tree slots", m, 0);
        test_newton(d::newton_tree, "newton_tree slots", m, 0, false);
    }
}

void test_api() {
    using namespace fps::evaluation;
    for (std::size_t m : {0, 1, 2, 31, 32, 33, 64, 129, 1000, 2049}) {
        for (std::size_t n : {0, 1, 2, 31, 32, 33, 64, 129, 1000, 2049, 5000}) test_evaluate(evaluate, "evaluate", n, m, int((n + m) % 4));
        if (m) test_interpolate(interpolate, "interpolate", m, int(m % 4));
        test_newton(to_newton_basis, "to_newton_basis", m, int(m % 4), m % 2 == 0);
    }
    test_newton(to_newton_basis, "to_newton_basis", 1 << 16, 0, false);
    test_interpolate(interpolate, "interpolate", 1 << 16, 0);
    test_evaluate(evaluate, "evaluate", 1 << 16, 1 << 16, 0);
    test_evaluate(evaluate, "evaluate long f", 1 << 17, 200, 0);
    test_evaluate(evaluate, "evaluate many points", 300, 1 << 17, 0);
    // Repeated points: evaluate takes them; interpolate throws.
    {
        std::vector<u32> points(500, 42), values(500), f = words(700, 0);
        evaluate(f, points, values);
        check_evaluate(f, points, values, "evaluate repeated");
        std::vector<u32> pts = distinct(300), ys = words(300, 0), out(300);
        pts[200] = pts[7];
        bool threw = false;
        try {
            interpolate(pts, ys, out);
        } catch (const std::domain_error&) {
            threw = true;
        }
        expect(threw, "interpolate repeated throws");
        threw = false;
        pts.resize(10);
        pts[3] = pts[5];
        ys.resize(10), out.resize(10);
        try {
            interpolate(pts, ys, out);
        } catch (const std::domain_error&) {
            threw = true;
        }
        expect(threw, "interpolate repeated throws (direct)");
    }
    // Size checks.
    {
        std::vector<u32> a(5), b(4);
        bool threw = false;
        try {
            evaluate(a, a, b);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        expect(threw, "evaluate sizes");
        threw = false;
        try {
            to_newton_basis(a, std::span<const u32>(b).first(2), a);  // 2 points for 5 coefficients
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        expect(threw, "to_newton_basis sizes");
    }
}

// The lane kernels against schoolbook, on Lanes (8-word coefficients).
void test_kernels() {
    namespace d = fps::evaluation::detail;
    using L = d::Lanes;
    for (std::size_t k = 1; k <= 8; ++k)
        for (std::size_t count = 1; count <= 8; ++count) {
            const auto w = words(8 * (k + count), int(k % 3 == 0)), q = words(8 * (k + 1), int(count % 3 == 0));
            std::vector<u32> out(8 * count);
            d::middle_lanes<L>(w.data(), k, q.data(), count, out.data());
            bool ok = true;
            const u32 r_inverse = d::inverse_mod(d::kR);
            for (std::size_t t = 0; t < count; ++t)
                for (int l = 0; l < 8; ++l) {
                    u32 s = 0;
                    for (std::size_t j = 0; j <= k; ++j) s = add(s, mul(w[8 * (k + t - j) + l], q[8 * j + l]));
                    ok &= out[8 * t + l] == mul(s, r_inverse);
                }
            expect(ok, "middle_lanes", k * 100 + count);
            const auto a = words(8 * k, 0), b = words(8 * count, 0), c = words(8 * (k + 1), 1), e = words(8 * (count + 1), 0);
            std::vector<u32> cross(8 * (k + count));
            d::cross_lanes<L>(a.data(), k, b.data(), count, c.data(), e.data(), cross.data());
            ok = true;
            for (std::size_t t = 0; t < k + count; ++t)
                for (int l = 0; l < 8; ++l) {
                    u32 s = 0;
                    for (std::size_t i = 0; i < k; ++i)
                        if (t >= i && t - i <= count) s = add(s, mul(a[8 * i + l], e[8 * (t - i) + l]));
                    for (std::size_t i = 0; i < count; ++i)
                        if (t >= i && t - i <= k) s = add(s, mul(b[8 * i + l], c[8 * (t - i) + l]));
                    ok &= cross[8 * t + l] == mul(s, r_inverse);
                }
            expect(ok, "cross_lanes", k * 100 + count);
        }
    fps::simd::for_each(fps::simd::All{}, []<class S> {
        for (std::size_t n : {1, 2, 5, 64, 255, 256, 257, 1000}) {
            for (std::size_t m : {1, 3, 8, 17, 33, 64, 100}) {
                const auto f = words(n, int(n % 4)), x = words(m, 0);
                std::vector<u32> v(m);
                d::horner<S>(f.data(), n, x.data(), m, v.data());
                check_evaluate(f, x, v, std::string("horner ") + S::name);
            }
        }
    });
    // divide_slots: y / v on slots [first, first + count / 8), 0 past m.
    using L = d::Lanes;
    for (std::size_t count : {8, 16, 72, 128})
        for (std::size_t first : {0, 8, 64})
            for (std::size_t m : {std::size_t(0), first + 3, first + count - 1, first + count, first + count + 9}) {
                auto v = words(count, 0), y = words(count, 0);
                for (u32& x : v) x = x ? x : 1;
                v[0] = P - 1;
                for (std::size_t i = 0; i < count; ++i)
                    if (first + i >= m) v[i] = 0, y[i] = 0;  // padding: values not read, y zero
                const auto v0 = v;
                bool ok = d::divide_slots<L>(v.data(), y.data(), first, m, count);
                for (std::size_t i = 0; i < count; ++i) ok &= first + i < m ? mul(v[i], v0[i]) == y[i] : v[i] == 0;
                expect(ok, "divide_slots", count * 1000 + m);
                if (m > first) {
                    auto z = v0;
                    z[std::min(m - first, count) / 2] = 0;
                    expect(!d::divide_slots<L>(z.data(), y.data(), first, m, count), "divide_slots zero", count * 1000 + m);
                }
            }
}

}  // namespace

// evaluate's chunks of f and of the points (2^24 each in evaluate), with small chunks.
void test_chunks() {
    namespace d = fps::evaluation::detail;
    for (auto [n, m, chunk] : {std::tuple<std::size_t, std::size_t, std::size_t>{5000, 300, 1024}, {3000, 2500, 1000}, {1025, 4000, 1024}}) {
        const auto f = words(n, 0), points = words(m, 0);
        std::vector<u32> values(m);
        d::evaluate_chunks(f, points, values, chunk);
        check_evaluate(f, points, values, "evaluate_chunks");
    }
}

int main() {
    fps::evaluation::detail::workspace_exact = true;
    test_kernels();
    test_chunks();
    test_trees();
    test_api();
    std::printf("%s evaluation: %d checks, %d failures\n", failures ? "FAIL" : "pass", checks, failures);
    return failures ? 1 : 0;
}
