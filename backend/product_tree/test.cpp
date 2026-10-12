// Tests product_tree::product against a sequential product (O(n m) per step) up to a few thousand
// coefficients, and at random points (Horner) beyond, in each mode (automatic, standard alone,
// lanes for every factor): no factors, empty factors, constants, one factor, 2^k and 2^k +- 1
// factors of degree 1, random mixed degrees with constants, one large factor among small ones,
// equal degrees (sums at powers of two: the wrap trick), trailing zero coefficients, values P - 1.
// Then the projection forms, and the errors. Exits nonzero on a failure.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/product_tree/product_tree.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
using fps::product_tree::detail::Mode;

constexpr u32 P = 998244353;
int failures = 0, checks = 0;
std::mt19937_64 rng(20261011);

void fail(const std::string& what) {
    if (++failures <= 20) std::printf("FAIL %s\n", what.c_str());
}

Vec naive(const Vec& a, const Vec& b) {
    if (a.empty() || b.empty()) return {};
    Vec c(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = u32((c[i + j] + u64(a[i]) * b[j]) % P);
    return c;
}

u32 evaluate(std::span<const u32> f, u32 x) {
    u64 r = 0;
    for (std::size_t i = f.size(); i-- > 0;) r = (r * x + f[i]) % P;
    return u32(r);
}

// kind 0: random below P; 1: P - 1; 2: random, a quarter 0 and a quarter P - 1.
u32 value(int kind) {
    const u32 r = u32(rng() % P);
    if (kind == 1) return P - 1;
    if (kind == 2) {
        const int q = int(rng() % 4);
        return q == 0 ? 0 : q == 1 ? P - 1 : r;
    }
    return r;
}

Vec random_poly(std::size_t n, int kind) {
    Vec f(n);
    for (u32& x : f) x = value(kind);
    return f;
}

const char* name(Mode mode) { return mode == Mode::automatic ? "automatic" : mode == Mode::standard ? "standard" : "lanes"; }

void check(const std::vector<Vec>& fs, const std::string& what) {
    std::vector<std::span<const u32>> spans(fs.begin(), fs.end());
    const std::size_t n = fps::product_tree::size(spans);
    std::size_t expect = 1;
    bool empty = false;
    for (const Vec& f : fs) empty |= f.empty(), expect += f.empty() ? 0 : f.size() - 1;
    if (empty) expect = 0;
    if (n != expect) return fail(what + ": size " + std::to_string(n) + ", expected " + std::to_string(expect));
    Vec exact;
    if (n <= 4000) {
        exact = empty ? Vec{} : Vec{1};
        for (const Vec& f : fs) exact = naive(exact, f);
    }
    for (const Mode mode : {Mode::automatic, Mode::standard, Mode::lanes}) {
        ++checks;
        const std::string where = what + " (" + name(mode) + ")";
        Vec out(n + 16, 0xdeadbeef);
        fps::product_tree::detail::product(spans, std::span(out.data(), n), mode);
        for (std::size_t i = n; i < n + 16; ++i)
            if (out[i] != 0xdeadbeef) return fail(where + ": wrote past the product");
        out.resize(n);
        if (n <= 4000) {
            if (out != exact) {
                std::size_t i = 0;
                while (i < n && out[i] == exact[i]) ++i;
                return fail(where + ": coefficient " + std::to_string(i) + " of " + std::to_string(n));
            }
            continue;
        }
        for (u32& x : out)
            if (x >= P) return fail(where + ": not canonical");
        for (int t = 0; t < 4; ++t) {
            const u32 x = u32(rng() % P);
            u64 want = 1;
            for (const Vec& f : fs) want = want * evaluate(f, x) % P;
            if (evaluate(out, x) != want) return fail(where + ": value at a random point");
        }
    }
}

std::vector<Vec> linear(std::size_t count, int kind = 0) {
    std::vector<Vec> fs;
    for (std::size_t i = 0; i < count; ++i) fs.push_back({value(kind), kind == 1 ? P - 1 : 1});
    return fs;
}

// Degrees: a quarter constants, the rest 1 .. top, some with trailing zeros.
std::vector<Vec> mixed(std::size_t count, u32 top, int kind = 0) {
    std::vector<Vec> fs;
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t d = rng() % 4 == 0 ? 0 : 1 + rng() % top;
        Vec f = random_poly(d + 1, kind);
        if (rng() % 8 == 0) f.back() = 0;
        fs.push_back(f);
    }
    return fs;
}

void test_small_cases() {
    check({}, "no factors");
    check({{}}, "one empty factor");
    check({{1, 2}, {}, {3}}, "an empty factor among others");
    check({{5}}, "one constant");
    check({{5}, {7}, {P - 1}}, "constants");
    check({{0}, {1, 2, 3}}, "zero constant");
    check({{1, 2, 3}}, "one factor");
    check({{3}, {1, 2, 3}, {4}}, "one factor among constants");
    check({{1, 2}, {3, 4}}, "two linear factors");
    check({{0, 0, 0}, {1, 1}}, "zero factor");
    for (int kind = 0; kind < 3; ++kind)
        for (std::size_t count : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
                                  511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049})
            check(linear(count, kind), "linear x" + std::to_string(count) + " kind " + std::to_string(kind));
    for (std::size_t count : {16, 64, 100, 300, 1000})
        for (u32 top : {2u, 4u, 9u, 40u}) check(mixed(count, top), "mixed x" + std::to_string(count) + " up to " + std::to_string(top));
    for (u32 d : {1u, 2u, 3u, 4u, 7u, 8u, 16u, 31u, 32u, 33u, 64u, 100u, 128u, 255u})
        for (std::size_t count : {2, 3, 4, 8, 16, 64, 65}) {
            if (count * d > 3000) continue;
            std::vector<Vec> fs;
            for (std::size_t i = 0; i < count; ++i) fs.push_back(random_poly(d + 1, 0));
            check(fs, "equal degree " + std::to_string(d) + " x" + std::to_string(count));
        }
}

void test_shapes() {
    // One large factor among small ones, first, middle and last.
    for (std::size_t at : {0, 50, 99}) {
        std::vector<Vec> fs = linear(100);
        fs.insert(fs.begin() + at, random_poly(2001, 0));
        check(fs, "large among 100 linear at " + std::to_string(at));
    }
    {
        std::vector<Vec> fs = mixed(500, 6);
        fs.push_back(random_poly(1500, 0));
        fs.push_back(random_poly(900, 2));
        check(fs, "two large among mixed");
    }
    // Geometric degrees: unbalanced splits.
    {
        std::vector<Vec> fs;
        for (std::size_t d = 1024; d >= 1; d /= 2) fs.push_back(random_poly(d + 1, 0));
        check(fs, "geometric degrees");
        std::reverse(fs.begin(), fs.end());
        check(fs, "geometric degrees, increasing");
    }
    // Sums at powers of two with leading coefficients P - 1 and 0.
    check(linear(1024, 1), "linear x1024, all P - 1");
    {
        std::vector<Vec> fs = linear(512);
        for (Vec& f : fs) f.push_back(0);
        check(fs, "linear with a zero x^2 coefficient x512");
    }
}

void test_large() {
    for (std::size_t count : {4095, 4096, 4097, 16384, 65537}) check(linear(count), "linear x" + std::to_string(count));
    check(mixed(20000, 8), "mixed x20000");
    check(mixed(3000, 60, 2), "mixed x3000 up to 60");
    {
        std::vector<Vec> fs = linear(10000);
        fs.push_back(random_poly(30000, 0));
        check(fs, "large among 10000 linear");
    }
    {
        std::vector<Vec> fs;
        for (int i = 0; i < 3; ++i) fs.push_back(random_poly(1 << 15, 0));
        check(fs, "three of 2^15");
    }
}

// product(items, out, proj) and size(items, proj) against the span-of-spans form.
void test_projection() {
    for (std::size_t count : {0, 1, 3, 100, 5000}) {
        const std::vector<Vec> fs = mixed(count, 5);
        const std::span<const Vec> items(fs);
        const auto proj = [](const Vec& f) { return std::span<const u32>(f); };
        std::vector<std::span<const u32>> spans(fs.begin(), fs.end());
        const std::size_t n = fps::product_tree::size(spans);
        ++checks;
        if (fps::product_tree::size(items, proj) != n) {
            fail("size(items, proj) x" + std::to_string(count));
            continue;
        }
        Vec a(n), b(n);
        fps::product_tree::product(spans, a);
        fps::product_tree::product(items, std::span(b), proj);
        if (a != b) fail("product(items, out, proj) x" + std::to_string(count));
    }
}

void test_errors() {
    const Vec f{1, 2};
    const std::span<const u32> spans[] = {f, f};
    Vec out(2);
    bool thrown = false;
    try {
        fps::product_tree::product(spans, out);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    ++checks;
    if (!thrown) fail("out of the wrong size: no std::invalid_argument");
}

}  // namespace

int main() {
    test_small_cases();
    test_shapes();
    test_large();
    test_projection();
    test_errors();
    std::printf("%s product_tree: %d checks, %d failures\n", failures ? "FAIL" : "pass", checks, failures);
    return failures ? 1 : 0;
}
