// Tests for products of more than 2^24 small factors (issue #253): the lanes' pairs stop where the
// lane root's transform at twice its length passes ntt's 2^kMaxLog (pairs_up_to), and the standard
// tree takes the lane products' forwards. With pairs_up_to lowered, that path runs at small sizes,
// against a sequential product and against the default path's output; untwist_factors past ntt's
// kRoots against roots computed apart; and 2^24 + 8 linear factors at random points and at their
// roots (not under ASan: a lowered pairs_up_to covers that path there). Exits nonzero on a failure.
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/product_tree/product_tree.hpp"

namespace {

namespace pt = fps::product_tree::detail;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
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

std::mt19937_64 rng(253);
int failures = 0, checks = 0;

void expect(bool ok, const std::string& what, u64 detail = 0) {
    ++checks;
    if (!ok && ++failures <= 20) std::printf("FAIL: %s (%llu)\n", what.c_str(), static_cast<unsigned long long>(detail));
}

u32 mul(u32 x, u32 y) { return u32(u64(x) * y % P); }
u32 pow_mod(u32 x, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, x = mul(x, x))
        if (e & 1) r = mul(r, x);
    return r;
}

u32 evaluate(std::span<const u32> f, u32 x) {
    u64 r = 0;
    for (std::size_t i = f.size(); i-- > 0;) r = (r * x + f[i]) % P;
    return u32(r);
}

Vec naive(const Vec& a, const Vec& b) {
    Vec c(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = u32((c[i + j] + u64(a[i]) * b[j]) % P);
    return c;
}

Vec run(const std::vector<std::span<const u32>>& spans, std::size_t n, std::size_t pairs_up_to, const std::string& where) {
    pt::pairs_up_to = pairs_up_to;
    Vec out(n + 16, 0xdeadbeef);
    pt::product(spans, std::span(out.data(), n), pt::Mode::automatic);
    pt::pairs_up_to = std::size_t(1) << fps::ntt::kMaxLog;
    bool guard = true;
    for (std::size_t i = n; i < n + 16; ++i) guard &= out[i] == 0xdeadbeef;
    expect(guard, where + ": wrote past the product");
    out.resize(n);
    return out;
}

// product() with the pairs stopped below each lane length the factors reach, against a sequential
// product (to 3000 coefficients), random points, and the output with the default pairs_up_to.
void check(const std::vector<Vec>& fs, const std::string& what) {
    std::vector<std::span<const u32>> spans(fs.begin(), fs.end());
    const std::size_t n = fps::product_tree::size(spans);
    const Vec base = run(spans, n, std::size_t(1) << fps::ntt::kMaxLog, what);
    Vec exact;
    if (n <= 3000) {
        exact = {1};
        for (const Vec& f : fs) exact = naive(exact, f);
        expect(base == exact, what + ": default path");
    }
    for (std::size_t up_to = 16 * 16; up_to <= 16 * (n / 8 + 64); up_to *= 2) {
        const std::string where = what + " pairs_up_to " + std::to_string(up_to);
        const Vec out = run(spans, n, up_to, where);
        if (n <= 3000) {
            std::size_t i = 0;
            while (i < n && out[i] == exact[i]) ++i;
            expect(i == n, where + ": coefficient", i);
            continue;
        }
        bool ok = out == base;
        for (u32 x : out) ok &= x < P;
        for (int t = 0; t < 2; ++t) {
            const u32 x = u32(rng() % P);
            u64 want = 1;
            for (const Vec& f : fs) want = want * evaluate(f, x) % P;
            ok &= evaluate(out, x) == want;
        }
        expect(ok, where + ": values");
    }
}

std::vector<Vec> polys(std::size_t count, u32 degree) {
    std::vector<Vec> fs(count, Vec(degree + 1));
    for (Vec& f : fs)
        for (u32& x : f) x = u32(rng() % P);
    return fs;
}

void test_lowered() {
    for (const std::size_t count : {136, 256, 257, 1000, 4096, 4099, 20000})
        check(polys(count, 1), std::to_string(count) + " linear factors");
    for (const u32 d : {2, 5}) check(polys(4096 / d, d), std::to_string(4096 / d) + " factors of degree " + std::to_string(d));
    std::vector<Vec> fs = polys(3000, 1);  // mixed degrees and constants, a large factor
    for (std::size_t i = 0; i < 600; ++i) fs.push_back(polys(1, u32(rng() % 9))[0]);
    fs.push_back(polys(1, 700)[0]);
    check(fs, "mixed factors");
}

// untwist_factors(2^20): rho_u^-1 from the bits of u (roots of order 2^(b + 4) for bit b, past
// ntt's kRoots for b = 19), and (rho_u^-1)^8 w_u = 1 (leaf_weights).
void test_untwist() {
    constexpr std::size_t kCount = std::size_t(1) << 20;
    const u32* table = pt::untwist_factors(kCount);
    const u32* weights = pt::leaf_weights(kCount);
    const u32 r_inverse = pow_mod(u32((u64(1) << 32) % P), P - 2), eighth = pow_mod(8, P - 2);
    std::vector<std::size_t> us = {0, 1, 2, 3, (1 << 18) + 5, 1 << 19, (1 << 19) + 1, (1 << 19) + 12345, kCount - 1};
    for (int t = 0; t < 40; ++t) us.push_back(rng() % kCount);
    for (const std::size_t u : us) {
        u32 rho = 1;
        for (int b = 0; b < 20; ++b)
            if (u >> b & 1) rho = mul(rho, pow_mod(3, (P - 1) >> (b + 4)));
        const u32 rho_inverse = pow_mod(rho, P - 2);
        bool ok = true;
        for (int k = 0; k < 8; ++k) ok &= mul(table[8 * u + k], r_inverse) == mul(pow_mod(rho_inverse, k), eighth);
        expect(ok, "untwist_factors: leaf " + std::to_string(u), u);
        const u32 one = mul(pow_mod(mul(mul(table[8 * u + 1], r_inverse), 8), 8), mul(weights[u], r_inverse));
        expect(one == 1, "untwist_factors: rho_u^8 = w_u, leaf " + std::to_string(u), u);
    }
}

// 2^24 + 8 linear factors x - r_i (degree 2^24 + 8 <= 2^25): lanes of length 2^22, past the pairs.
void test_large() {
    const std::size_t n = (std::size_t(1) << 24) + 8;
    Vec words(2 * n);
    std::vector<std::span<const u32>> spans(n);
    for (std::size_t i = 0; i < n; ++i) {
        const u32 r = u32(rng() % P);
        words[2 * i] = r ? P - r : 0, words[2 * i + 1] = 1;
        spans[i] = {&words[2 * i], 2};
    }
    Vec out(n + 1);
    try {
        fps::product_tree::product(std::span<const std::span<const u32>>(spans), out);
    } catch (const std::exception& e) {
        expect(false, std::string("2^24 + 8 linear factors: threw ") + e.what());
        return;
    }
    bool canonical = out[n] == 1;
    for (u32 x : out) canonical &= x < P;
    expect(canonical, "2^24 + 8 linear factors: canonical, monic");
    for (int t = 0; t < 3; ++t) {
        const u32 x = u32(rng() % P);
        u64 want = 1;
        for (std::size_t i = 0; i < n; ++i) want = want * (x + words[2 * i]) % P;
        expect(evaluate(out, x) == want, "2^24 + 8 linear factors: value at a random point", x);
    }
    for (const std::size_t i : {std::size_t(0), n / 3, n - 1}) {
        const u32 r = words[2 * i] ? P - words[2 * i] : 0;
        expect(evaluate(out, r) == 0, "2^24 + 8 linear factors: zero at a root", i);
    }
}

}  // namespace

int main() {
    test_untwist();
    test_lowered();
    if (!kSanitized) test_large();
    std::printf("%s product_tree large: %d checks, %d failures%s\n", failures ? "FAIL" : "pass", checks, failures,
                kSanitized ? " (2^24 + 8 factors skipped under ASan)" : "");
    return failures != 0;
}
