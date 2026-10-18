// Tests for the standard layout's leaf products (issue #136 round 3): leaf_products and
// leaf_products_twice on every simd struct the machine runs against a schoolbook product mod
// x^8 - w per leaf, for leaf counts 1 .. 40 and powers of two to 2^12 (short batches, odd counts:
// Neon's and zmm's pairs of leaves with a tail), random inputs, P - 1 and zeros, random weights;
// outputs into a or b, the copy, and nothing written past the leaves. Exits nonzero on a failure.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "backend/product_tree/product_tree.hpp"

namespace {

namespace pt = fps::product_tree::detail;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = 998244353;

std::mt19937_64 rng(1336);
int failures = 0, checks = 0;

void expect(bool ok, const std::string& what, u64 detail = 0) {
    ++checks;
    if (!ok && ++failures <= 20) std::printf("FAIL: %s (%llu)\n", what.c_str(), static_cast<unsigned long long>(detail));
}

u32 mul(u32 x, u32 y) { return u32(u64(x) * y % P); }

// a b 2^-32 mod (x^8 - w), w = weight 2^-32, leaf by leaf.
Vec reference(const Vec& a, const Vec& b, const Vec& weights) {
    const u32 r_inverse = pt::kRInverse;
    Vec c(a.size());
    for (std::size_t v = 0; v < weights.size(); ++v) {
        const u32 w = mul(weights[v], r_inverse);
        for (int j = 0; j < 8; ++j) {
            u64 s = 0;
            for (int i = 0; i < 8; ++i) {
                const int k = j - i;
                const u32 t = mul(a[8 * v + i], b[8 * v + (k + 8) % 8]);
                s += k >= 0 ? t : mul(w, t);
            }
            c[8 * v + j] = mul(u32(s % P), r_inverse);
        }
    }
    return c;
}

Vec words(std::size_t n, int kind) {
    Vec v(n);
    for (u32& x : v) x = kind == 1 ? P - 1 : kind == 2 && rng() % 2 ? 0 : u32(rng() % P);
    return v;
}

// Guard words around each output.
constexpr std::size_t kGuard = 32;
constexpr u32 kMark = 0xdeadbeef;

bool guards_intact(const Vec& v, std::size_t n) {
    for (std::size_t i = 0; i < kGuard; ++i)
        if (v[i] != kMark || v[kGuard + n + i] != kMark) return false;
    return true;
}

template <class L>
void test_count(const char* name, std::size_t count, int kind) {
    const std::string where = std::string("<") + name + "> count " + std::to_string(count) + " kind " + std::to_string(kind);
    const std::size_t n = 8 * count;
    const Vec a = words(n, kind), b0 = words(n, kind), b1 = words(n, kind == 2 ? 0 : kind);
    Vec weights(count);
    for (u32& w : weights) w = kind == 1 ? P - 1 : u32(rng() % P);
    const Vec c0 = reference(a, b0, weights), c1 = reference(a, b1, weights);
    const auto fresh = [&] { return Vec(n + 2 * kGuard, kMark); };

    Vec out = fresh(), copy = fresh();
    pt::leaf_products<L>(a.data(), b0.data(), out.data() + kGuard, count, weights.data(), copy.data() + kGuard);
    expect(std::equal(c0.begin(), c0.end(), out.begin() + kGuard) && guards_intact(out, n), "leaf_products" + where);
    expect(std::equal(c0.begin(), c0.end(), copy.begin() + kGuard) && guards_intact(copy, n), "leaf_products copy" + where);

    Vec in_a = a, in_b = b0;
    pt::leaf_products<L>(in_a.data(), b0.data(), in_a.data(), count, weights.data());
    expect(in_a == c0, "leaf_products into a" + where);
    pt::leaf_products<L>(a.data(), in_b.data(), in_b.data(), count, weights.data());
    expect(in_b == c0, "leaf_products into b" + where);

    Vec o0 = fresh(), o1 = fresh();
    pt::leaf_products_twice<L>(a.data(), b0.data(), b1.data(), o0.data() + kGuard, o1.data() + kGuard, count, weights.data());
    expect(std::equal(c0.begin(), c0.end(), o0.begin() + kGuard) && guards_intact(o0, n), "leaf_products_twice 0" + where);
    expect(std::equal(c1.begin(), c1.end(), o1.begin() + kGuard) && guards_intact(o1, n), "leaf_products_twice 1" + where);

    // The descent's use: the first product into a.
    Vec twice_a = a, second = fresh();
    pt::leaf_products_twice<L>(twice_a.data(), b0.data(), b1.data(), twice_a.data(), second.data() + kGuard, count, weights.data());
    expect(twice_a == c0, "leaf_products_twice into a" + where);
    expect(std::equal(c1.begin(), c1.end(), second.begin() + kGuard) && guards_intact(second, n), "leaf_products_twice second" + where);
}

template <class L>
void test_struct(const char* name) {
    for (std::size_t count = 1; count <= 40; ++count)
        for (int kind = 0; kind < 3; ++kind) test_count<L>(name, count, kind);
    for (std::size_t count = 64; count <= 4096; count *= 4) test_count<L>(name, count, 0);
}

}  // namespace

int main() {
    test_struct<fps::simd::Scalar>("Scalar");
    test_struct<fps::product_tree::Lanes>("Lanes");
#if defined(__SSE4_1__)
    test_struct<fps::simd::Sse>("Sse");
#endif
#if defined(__AVX2__)
    test_struct<fps::simd::Avx2>("Avx2");
#endif
#if defined(__AVX512VL__)
    test_struct<fps::simd::Avx512Vl>("Avx512Vl");
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
    test_struct<fps::simd::Neon>("Neon");
#endif
    std::printf("%s product_tree leaves: %d checks, %d failures\n", failures ? "FAIL" : "pass", checks, failures);
    return failures != 0;
}
