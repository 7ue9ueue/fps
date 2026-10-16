// Tests for issue #136 round 1: the base kernels that take the products' unit terms for free
// (middle_lanes<L, true>, cross_units, multiply_units, with q[0] = kR) against schoolbook, and the
// root division (root_window: series::inverse and fused cyclic products) against a schoolbook
// series division or its defining identity, in exactly division_words() of memory.
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "backend/evaluation/evaluation.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
constexpr u32 P = 998244353;
namespace d = fps::evaluation::detail;

std::mt19937_64 rng(136);
int failures = 0, checks = 0;

void expect(bool ok, const std::string& what, u64 detail = 0) {
    ++checks;
    if (ok) return;
    if (++failures <= 30) std::printf("FAIL: %s (%llu)\n", what.c_str(), static_cast<unsigned long long>(detail));
}

u32 mul(u32 x, u32 y) { return u32(u64(x) * y % P); }
u32 add(u32 x, u32 y) { return u32((u64(x) + y) % P); }
u32 sub(u32 x, u32 y) { return u32((u64(x) + P - y) % P); }

// Words of a kind: random, all P - 1, all 0, small.
std::vector<u32> words(std::size_t n, int kind) {
    std::vector<u32> v(n);
    for (u32& x : v) x = kind == 0 ? u32(rng() % P) : kind == 1 ? P - 1 : kind == 2 ? 0 : u32(rng() % 10);
    return v;
}

// coefficients of 8 words with coefficient 0 = kR (1 in Montgomery form) in every lane.
std::vector<u32> unit_words(std::size_t coefficients, int kind) {
    auto v = words(8 * coefficients, kind);
    std::fill(v.begin(), v.begin() + 8, d::kR);
    return v;
}

void test_kernels() {
    using L = d::Lanes;
    const u32 r_inverse = d::inverse_mod(d::kR);
    for (int kind = 0; kind < 4; ++kind) {
        for (std::size_t k = 1; k <= 8; ++k)
            for (std::size_t count = 1; count <= 8; ++count) {
                const auto w = words(8 * (k + count), kind), q = unit_words(k + 1, (kind + 1) % 4);
                std::vector<u32> out(8 * count);
                d::middle_lanes<L, true>(w.data(), k, q.data(), count, out.data());
                bool ok = true;
                for (std::size_t t = 0; t < count; ++t)
                    for (int l = 0; l < 8; ++l) {
                        u32 s = 0;
                        for (std::size_t j = 0; j <= k; ++j) s = add(s, mul(w[8 * (k + t - j) + l], q[8 * j + l]));
                        ok &= out[8 * t + l] == mul(s, r_inverse);
                    }
                expect(ok, "middle_lanes unit", k * 100 + count);
            }
        for (std::size_t n = 1; n <= 8; ++n)
            for (std::size_t k = 1; k <= 8; ++k) {
                const auto a = words(8 * n, kind), b = words(8 * k, (kind + 2) % 4);
                const auto c = unit_words(n + 1, (kind + 1) % 4), e = unit_words(k + 1, kind);  // c = Q_l, e = Q_r
                std::vector<u32> cross(8 * (n + k));
                d::cross_units<L>(a.data(), n, b.data(), k, c.data(), e.data(), cross.data());
                bool ok = true;
                for (std::size_t t = 0; t < n + k; ++t)
                    for (int l = 0; l < 8; ++l) {
                        u32 s = 0;
                        for (std::size_t i = 0; i < n; ++i)
                            if (t >= i && t - i <= k) s = add(s, mul(a[8 * i + l], e[8 * (t - i) + l]));
                        for (std::size_t i = 0; i < k; ++i)
                            if (t >= i && t - i <= n) s = add(s, mul(b[8 * i + l], c[8 * (t - i) + l]));
                        ok &= cross[8 * t + l] == mul(s, r_inverse);
                    }
                expect(ok, "cross_units", n * 100 + k);
                const auto x = unit_words(n + 1, kind), y = unit_words(k + 1, (kind + 3) % 4);
                std::vector<u32> product(8 * (n + k + 1));
                d::multiply_units<L>(x.data(), n, y.data(), k, product.data());
                ok = true;
                for (std::size_t t = 0; t <= n + k; ++t)
                    for (int l = 0; l < 8; ++l) {
                        u32 s = 0;
                        for (std::size_t i = 0; i <= n; ++i)
                            if (t >= i && t - i <= k) s = add(s, mul(x[8 * i + l], y[8 * (t - i) + l]));
                        ok &= product[8 * t + l] == mul(s, r_inverse);
                    }
                expect(ok, "multiply_units", n * 100 + k);
            }
    }
}

// d = G / q mod x^K, G = rev_K(f), against the schoolbook division (K up to 3000) or, beyond, the
// identity (q d)[i] = G[i] at sampled i.
void test_division(std::size_t big, std::size_t n, std::size_t qn, bool transform, int kind) {
    auto q = words(qn, kind);
    q[0] = 1;
    const auto f = words(n, (kind + 1) % 4);
    std::vector<u32> g(big, 0);
    for (std::size_t j = 0; j < n; ++j) g[big - 1 - j] = f[j];
    const std::size_t len = std::max<std::size_t>(64, std::bit_ceil(big));
    // The tree's form of q's transform: q mod (x^L - 1) times 2^32 (64-byte aligned, as the
    // transforms need).
    std::vector<u32> buffer(len + 64, 0);
    u32* const qt = buffer.data() + (64 - reinterpret_cast<std::uintptr_t>(buffer.data()) % 64) % 64 / sizeof(u32);
    if (transform) {
        for (std::size_t i = 0; i < qn; ++i) qt[i % len] = add(qt[i % len], q[i]);
        for (std::size_t i = 0; i < len; ++i) qt[i] = mul(qt[i], d::kR);
        d::transform(len).forward(qt);
    }
    std::vector<u32> out(big, 12345);
    d::workspace_spilled = false;
    {
        d::Workspace work(d::division_words(big));
        d::root_window(f.data(), n, q.data(), qn, big, out.data(), work.stack, transform ? qt : nullptr, transform ? len : 0);
    }
    const std::string name = "root_window K " + std::to_string(big) + " n " + std::to_string(n) + " qn " + std::to_string(qn) +
                             (transform ? " qt" : "") + " kind " + std::to_string(kind);
    expect(!d::workspace_spilled, name + ": memory as computed");
    bool ok = true;
    if (big <= 3000) {
        std::vector<u32> e(big);
        for (std::size_t i = 0; i < big; ++i) {
            u32 s = g[i];
            for (std::size_t j = 1; j <= std::min(i, qn - 1); ++j) s = sub(s, mul(q[j], e[i - j]));
            e[i] = s;
        }
        ok = e == out;
    } else {
        std::vector<std::size_t> at = {0, 1, big / 2, big - 2, big - 1};
        for (int i = 0; i < 40; ++i) at.push_back(rng() % big);
        for (std::size_t i : at) {
            u32 s = 0;
            for (std::size_t j = 0; j <= std::min(i, qn - 1); ++j) s = add(s, mul(q[j], out[i - j]));
            ok &= s == g[i];
        }
    }
    expect(ok, name);
}

}  // namespace

int main() {
    d::workspace_exact = true;
    test_kernels();
    for (std::size_t big : {136, 137, 200, 255, 256, 257, 1000, 1024, 1025, 2999, 4096, 5003, 1 << 15, (1 << 15) + 1})
        for (std::size_t n : {std::size_t(1), big / 3 + 1, big})
            for (bool transform : {false, true}) {
                const std::size_t qn = n == 1 ? big / 2 + 1 : big + 1;  // a short divisor too
                test_division(big, n, qn, transform, int(rng() % 4));
            }
    std::printf("%s evaluation units: %d checks, %d failures\n", failures ? "FAIL" : "pass", checks, failures);
    return failures ? 1 : 0;
}
