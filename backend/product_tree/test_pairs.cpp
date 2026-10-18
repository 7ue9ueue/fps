// Tests for the standard tree's nodes over two lane products from the lane root's transform
// (issue #91 round 2): lane_pairs on every simd struct the machine runs against the forward
// transform of each pair's cyclic product computed apart; the lane root's transform at twice its
// length against the forward of its lanes; pair_nodes on lane products alone and among large
// factors; and products through product() where the pairs run (lane lengths from kPairsLength,
// degrees at powers of two for the wrap trick, lanes among large factors), against a sequential
// product or random points. Exits nonzero on a failure.
#include <algorithm>
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

std::mt19937_64 rng(912);
int failures = 0, checks = 0;

void expect(bool ok, const std::string& what, u64 detail = 0) {
    ++checks;
    if (!ok && ++failures <= 20) std::printf("FAIL: %s (%llu)\n", what.c_str(), static_cast<unsigned long long>(detail));
}

u32 mul(u32 x, u32 y) { return u32(u64(x) * y % P); }

// 64-byte aligned words with kPadding after, zero.
struct Words {
    std::vector<u32> memory;
    u32* data;
    explicit Words(std::size_t n) : memory(n + fps::ntt::kPadding + 16, 0) {
        data = memory.data() + (16 - reinterpret_cast<std::uintptr_t>(memory.data()) / 4 % 16) % 16;
    }
};

// a b mod (x^n - 1), schoolbook.
Vec cyclic(const Vec& a, const Vec& b, std::size_t n) {
    Vec c(n);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[(i + j) % n] = u32((c[(i + j) % n] + u64(a[i]) * b[j]) % P);
    return c;
}

// lane_pairs<L> on random lanes of length T (degree < T: their cyclic products of length T).
template <class L>
void test_lane_pairs(const char* struct_name, std::size_t T, int kind) {
    const std::string where = std::string("lane_pairs<") + struct_name + "> T " + std::to_string(T) + " kind " + std::to_string(kind);
    std::vector<Vec> lanes(8, Vec(T));
    for (Vec& l : lanes)
        for (u32& x : l) x = kind == 1 ? P - 1 : kind == 2 && rng() % 2 ? 0 : u32(rng() % P);
    Words v(8 * T);
    for (std::size_t j = 0; j < T; ++j)
        for (int l = 0; l < 8; ++l) v.data[8 * j + l] = lanes[l][j];
    pt::forward(v.data, 8 * T);
    std::vector<Words> outs;
    for (int j = 0; j < 4; ++j) outs.emplace_back(2 * T + 64);
    u32* out[4];
    for (int j = 0; j < 4; ++j) {
        std::fill_n(outs[j].data, 2 * T + 64, 0xdeadbeefu);
        out[j] = j == 2 ? nullptr : outs[j].data;  // pair 2 not written
    }
    pt::lane_pairs<L>(v.data, T, out, pt::untwist_factors(T / 8));
    const u32 r_inverse = pt::kRInverse;
    for (int j = 0; j < 4; ++j) {
        if (!out[j]) {
            bool untouched = true;
            for (std::size_t i = 0; i < 2 * T + 64; ++i) untouched &= outs[j].data[i] == 0xdeadbeefu;
            expect(untouched, where + ": a null pair written");
            continue;
        }
        Vec q = cyclic(lanes[2 * j], lanes[2 * j + 1], T);
        Words want(T);
        for (std::size_t i = 0; i < T; ++i) want.data[i] = mul(q[i], r_inverse);
        pt::forward(want.data, T);
        std::size_t bad = 0;
        for (std::size_t i = 0; i < T; ++i) bad += out[j][i] != want.data[i];
        expect(bad == 0, where + ": pair " + std::to_string(j), bad);
        bool guard = true;
        for (std::size_t i = T; i < 2 * T + 64; ++i) guard &= out[j][i] == 0xdeadbeefu;
        expect(guard, where + ": wrote past the pair");
    }
}

void test_kernels() {
    for (const std::size_t T : {64, 128, 256, 512, 1024})
        for (int kind = 0; kind < 3; ++kind) {
            test_lane_pairs<fps::simd::Scalar>("Scalar", T, kind);
            test_lane_pairs<fps::product_tree::Lanes>("Lanes", T, kind);
#if defined(__SSE4_1__)
            test_lane_pairs<fps::simd::Sse>("Sse", T, kind);
#endif
#if defined(__AVX2__)
            test_lane_pairs<fps::simd::Avx2>("Avx2", T, kind);
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
            test_lane_pairs<fps::simd::Neon>("Neon", T, kind);
#endif
        }
    // The untwist factors are prefix compatible: a shorter table first, then a longer one.
    const Vec first(pt::untwist_factors(4), pt::untwist_factors(4) + 32);
    const u32* longer = pt::untwist_factors(1 << 12);
    expect(std::equal(first.begin(), first.end(), longer), "untwist_factors prefix");
}

// Slots of 8 factors, every factor's words given (degree >= 1), in their order.
struct Factors {
    const std::vector<Vec>* fs;
    std::size_t size() const { return fs->size(); }
    std::span<const u32> operator[](std::size_t i) const { return (*fs)[i]; }
};

// LaneTree::root_doubled: the root's transform at twice its length against the forward of its
// lanes, the stack's layout; pair_nodes on the lane products, as product() builds them.
void test_root_values(std::size_t count, u32 degree) {
    const std::string where = "root values: " + std::to_string(count) + " factors of degree " + std::to_string(degree);
    std::vector<Vec> fs(count, Vec(degree + 1));
    for (Vec& f : fs)
        for (u32& x : f) x = u32(rng() % P);
    std::vector<u32> order(count);
    for (std::size_t i = 0; i < count; ++i) order[i] = u32(i);
    const pt::Slots<Factors> slots{Factors{&fs}, order.data(), order.size()};
    Words region(1 << 22);
    pt::Stack stack(region.data, 1 << 22);
    pt::LaneTree<pt::Slots<Factors>> tree(slots, stack);
    const std::size_t len = pt::LaneTree<pt::Slots<Factors>>::length(tree.degree(0, slots.count()));
    const auto [root, values, room, coefficients_mark, values_mark] = tree.root_doubled();
    // The stack: room (the root's left child), the coefficients, values.
    expect(room + pt::Stack::footprint(8 * len + 8) == root.coefficients && root.coefficients + pt::Stack::footprint(8 * len + 8) == values,
           where + ": room, coefficients, values in the stack");
    {  // The coefficients and the node as root()'s.
        Words other(1 << 22);
        pt::Stack other_stack(other.data, 1 << 22);
        pt::LaneTree<pt::Slots<Factors>> plain(slots, other_stack);
        const auto r = plain.root();
        expect(r.length == root.length && std::equal(r.coefficients, r.coefficients + 8 * (len + 1), root.coefficients) &&
                   std::equal(r.node.degree, r.node.degree + 8, root.node.degree) && std::equal(r.node.lead, r.node.lead + 8, root.node.lead),
               where + ": coefficients as root()'s");
    }
    // The lanes' coefficients (at most len + 1 each) mod X^(2 len) - 1 = themselves.
    Words want(16 * len);
    for (std::size_t j = 0; j <= len && j < 2 * len; ++j)
        for (int l = 0; l < 8; ++l) want.data[8 * j + l] = root.coefficients[8 * j + l];
    pt::forward(want.data, 16 * len);
    std::size_t bad = 0;
    for (std::size_t i = 0; i < 16 * len; ++i) bad += values[i] != want.data[i];
    expect(bad == 0, where + ": the root's transform at twice its length", bad);
    std::vector<pt::Leaf> leaves;
    for (int l = 0; l < 8; ++l) leaves.push_back({root.coefficients + l, root.node.degree[l], true, 8});
    const unsigned bits = pt::pair_nodes(leaves, 0, 2 * len);
    expect(bits == 15, where + ": pair_nodes", bits);
    stack.release(values_mark);
    expect(stack.mark().top == values_mark.top, where + ": values released");
    (void)coefficients_mark;
}

// pair_nodes among large leaves: the bits name the nodes over (first + 2j, first + 2j + 1).
void test_pair_nodes() {
    Vec words(1 << 12, 1);
    const auto bits = [&](const std::vector<u32>& degrees, std::size_t first, std::size_t length) {
        std::vector<pt::Leaf> leaves;
        for (u32 d : degrees) leaves.push_back({words.data(), d, true});
        return pt::pair_nodes(leaves, first, length);
    };
    expect(bits({100, 100, 100, 100, 100, 100, 100, 100}, 0, 256) == 15, "pair_nodes: 8 of 100");
    expect(bits({128, 128, 128, 128, 128, 128, 128, 128}, 0, 256) == 15, "pair_nodes: 8 of 128 (wrap)");
    expect(bits({100, 100, 100, 100, 100, 100, 100, 100}, 0, 512) == 0, "pair_nodes: other length");
    expect(bits({60, 60, 60, 60, 60, 60, 60, 60}, 0, 128) == 15, "pair_nodes: 8 of 60");
    expect(bits({10, 10, 10, 10, 10, 10, 10, 10}, 0, 64) == 0, "pair_nodes: base nodes");
    // A large leaf first: the lanes pair up below it or not at all.
    const unsigned with_large = bits({2000, 100, 100, 100, 100, 100, 100, 100, 100}, 1, 256);
    expect(with_large == 15, "pair_nodes: lanes after a large leaf", with_large);
    expect(bits({900, 100, 100, 100, 100, 100, 100, 100, 100}, 0, 256) == 0, "pair_nodes: first not a lane");
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

// product() in each mode against a sequential product (to 3000 coefficients) or random points.
void check(const std::vector<Vec>& fs, const std::string& what) {
    std::vector<std::span<const u32>> spans(fs.begin(), fs.end());
    const std::size_t n = fps::product_tree::size(spans);
    Vec exact;
    if (n <= 3000) {
        exact = {1};
        for (const Vec& f : fs) exact = naive(exact, f);
    }
    for (const pt::Mode mode : {pt::Mode::automatic, pt::Mode::lanes}) {
        const std::string where = what + (mode == pt::Mode::lanes ? " (lanes)" : "");
        Vec out(n + 16, 0xdeadbeef);
        pt::product(spans, std::span(out.data(), n), mode);
        bool guard = true;
        for (std::size_t i = n; i < n + 16; ++i) guard &= out[i] == 0xdeadbeef;
        expect(guard, where + ": wrote past the product");
        out.resize(n);
        if (n <= 3000) {
            std::size_t i = 0;
            while (i < n && out[i] == exact[i]) ++i;
            expect(i == n, where + ": coefficient", i);
            continue;
        }
        bool ok = true;
        for (u32 x : out) ok &= x < P;
        for (int t = 0; t < 3; ++t) {
            const u32 x = u32(rng() % P);
            u64 want = 1;
            for (const Vec& f : fs) want = want * evaluate(f, x) % P;
            ok &= evaluate(out, x) == want;
        }
        expect(ok, where + ": values at random points");
    }
}

std::vector<Vec> polys(std::size_t count, u32 degree, int kind = 0) {
    std::vector<Vec> fs(count, Vec(degree + 1));
    for (Vec& f : fs)
        for (u32& x : f) x = kind == 1 ? P - 1 : u32(rng() % P);
    return fs;
}

void test_products() {
    // Lanes of length 32 (the smallest with pairs) and up; degrees of lanes at powers of two (the
    // lane root and the pair nodes wrap), and one past.
    for (const std::size_t count : {136, 200, 255, 256, 257, 300, 511, 512, 513, 1000, 2048, 4096, 4099, 10000, 1 << 14})
        check(polys(count, 1), std::to_string(count) + " linear factors");
    check(polys(4096, 1, 1), "4096 linear factors, words P - 1");
    for (const u32 d : {2, 3, 4, 7, 8}) check(polys(4096 / d, d), std::to_string(4096 / d) + " factors of degree " + std::to_string(d));
    // Mixed degrees and constants.
    for (const std::size_t count : {300, 3000, 20000}) {
        std::vector<Vec> fs;
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t d = rng() % 4 == 0 ? 0 : 1 + rng() % 8;
            Vec f(d + 1);
            for (u32& x : f) x = u32(rng() % P);
            fs.push_back(f);
        }
        check(fs, std::to_string(count) + " mixed factors");
    }
    // Large factors among lanes: before them in the standard tree's leaves.
    for (const u32 big : {100, 300, 700, 2000, 5000}) {
        std::vector<Vec> fs = polys(2000, 1);
        const std::vector<Vec> large = polys(1 + big % 3, big);
        fs.insert(fs.begin() + 700, large.begin(), large.end());
        check(fs, "2000 linear factors and " + std::to_string(large.size()) + " of degree " + std::to_string(big));
    }
}

}  // namespace

int main() {
    test_kernels();
    for (const std::size_t count : {136, 256, 512, 1000, 4096})
        for (const u32 d : {1, 2}) test_root_values(count, d);
    test_pair_nodes();
    test_products();
    std::printf("%s product_tree pairs: %d checks, %d failures\n", failures ? "FAIL" : "pass", checks, failures);
    return failures != 0;
}
