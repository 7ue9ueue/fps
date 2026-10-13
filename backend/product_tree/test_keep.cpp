// Tests for the trees' Keep hook (issue #104): a Keep that records every room, on lane trees and
// standard trees of random degrees. Each kept transform must be the forward transform, at the
// node's length, of the child's product in Montgomery form computed apart (schoolbook); the
// standard root's kept transform that of the root's product mod (x^L - 1); the products the same
// as without a Keep.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/product_tree/tree.hpp"

namespace {

namespace pt = fps::product_tree::detail;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
constexpr u32 P = 998244353;

std::mt19937_64 rng(104);
int failures = 0, checks = 0;

void expect(bool ok, const std::string& what, u64 detail = 0) {
    ++checks;
    if (!ok && ++failures <= 20) std::printf("FAIL: %s (%llu)\n", what.c_str(), static_cast<unsigned long long>(detail));
}

u32 to_montgomery(u32 x) { return u32((u64(x) << 32) % P); }

std::vector<u32> multiply(const std::vector<u32>& a, const std::vector<u32>& b) {
    std::vector<u32> c(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = u32((c[i + j] + u64(a[i]) * b[j]) % P);
    return c;
}

// Memory for rooms: 64-byte lines.
struct alignas(64) Line {
    u32 w[16];
};

struct Entry {
    std::size_t lo, mid, hi, words;
    u32 *left, *right;
};

// Records every room, both children's, from one pool; and the root's room.
struct Recorder {
    std::vector<Line>* pool;
    std::size_t* used;
    std::vector<Entry>* entries;
    u32** root_room;
    std::size_t* root_words;

    u32* take(std::size_t words) const {
        const std::size_t lines = pt::Stack::footprint(words) / 16;
        u32* p = (*pool)[*used].w;
        *used += lines;
        return p;
    }
    pt::Rooms place(std::size_t lo, std::size_t mid, std::size_t hi, std::size_t words) const {
        const pt::Rooms r{take(words), take(words)};
        entries->push_back({lo, mid, hi, words, r.left, r.right});
        return r;
    }
    u32* root(std::size_t words) const {
        *root_words = words;
        return *root_room = take(words);
    }
};

struct Harness {
    std::vector<Line> pool = std::vector<Line>(1 << 18);
    std::size_t used = 0;
    std::vector<Entry> entries;
    u32* root_room = nullptr;
    std::size_t root_words = 0;
    std::vector<Line> stack_memory = std::vector<Line>(1 << 18);
    pt::Stack stack{stack_memory[0].w, 16 * stack_memory.size()};
    Recorder recorder() { return {&pool, &used, &entries, &root_room, &root_words}; }
};

// Lanes: slot k holds 8 polynomials of degree d_k (normal form here, Montgomery when loaded).
struct LaneLeaves {
    std::vector<std::vector<std::vector<u32>>> slots;  // [k][l]: coefficients

    std::size_t count() const { return slots.size(); }
    u32 degree(std::size_t k) const { return u32(slots[k][0].size() - 1); }
    pt::LaneNode load(std::size_t k, u32* c) const {
        pt::LaneNode node;
        const u32 d = degree(k);
        for (std::size_t l = 0; l < 8; ++l) {
            for (u32 j = 0; j <= d; ++j) c[8 * j + l] = to_montgomery(slots[k][l][j]);
            node.degree[l] = d, node.lead[l] = c[8 * d + l];
        }
        return node;
    }
    std::vector<u32> product(std::size_t lo, std::size_t hi, std::size_t l) const {
        std::vector<u32> p{1};
        for (std::size_t k = lo; k < hi; ++k) p = multiply(p, slots[k][l]);
        return p;
    }
};

void test_lanes(std::size_t count) {
    LaneLeaves leaves;
    for (std::size_t k = 0; k < count; ++k) {
        const std::size_t d = 1 + rng() % 3;
        leaves.slots.push_back(std::vector<std::vector<u32>>(8, std::vector<u32>(d + 1)));
        for (auto& f : leaves.slots.back())
            for (u32& x : f) x = u32(rng() % P);
    }
    Harness h;
    pt::LaneTree<LaneLeaves, Recorder> tree(leaves, h.stack, h.recorder());
    const auto root = tree.root();
    for (std::size_t l = 0; l < 8; ++l) {
        const auto p = leaves.product(0, count, l);
        bool ok = true;
        for (std::size_t j = 0; j < p.size(); ++j) ok &= root.coefficients[8 * j + l] == to_montgomery(p[j]);
        expect(ok, "lane root product", count);
    }
    expect(!h.entries.empty() || tree.base(0, count), "lane rooms placed", count);
    for (const Entry& e : h.entries) {
        expect(tree.split(e.lo, e.hi) == e.mid && !tree.base(e.lo, e.hi), "lane node structure", count);
        for (int side = 0; side < 2; ++side) {
            std::vector<Line> memory(e.words / 16 + 2);
            u32* const t = memory[0].w;
            std::fill(t, t + e.words, 0u);
            for (std::size_t l = 0; l < 8; ++l) {
                const auto p = side ? leaves.product(e.mid, e.hi, l) : leaves.product(e.lo, e.mid, l);
                for (std::size_t j = 0; j < p.size(); ++j) t[8 * j + l] = to_montgomery(p[j]);
            }
            pt::forward(t, e.words);
            const u32* kept = side ? e.right : e.left;
            expect(std::equal(t, t + e.words, kept), "lane kept transform", count * 1000 + e.lo);
        }
    }
}

void test_standard(std::size_t count, u32 top_degree) {
    std::vector<std::vector<u32>> fs;
    std::vector<pt::Leaf> leaves;
    for (std::size_t k = 0; k < count; ++k) {
        fs.emplace_back(2 + rng() % top_degree);
        for (u32& x : fs.back()) x = u32(rng() % P);
    }
    for (const auto& f : fs) leaves.push_back({f.data(), u32(f.size() - 1), false});
    const auto product = [&](std::size_t lo, std::size_t hi) {
        std::vector<u32> p{1};
        for (std::size_t k = lo; k < hi; ++k) p = multiply(p, fs[k]);
        return p;
    };
    Harness h;
    pt::StandardTree<Recorder> tree(leaves, h.stack, h.recorder());
    const u32* root = tree.root();
    const auto all = product(0, count);
    bool ok = true;
    for (std::size_t j = 0; j < all.size(); ++j) ok &= root[j] == to_montgomery(all[j]);
    expect(ok, "standard root product", count);
    const auto transform_of = [](const std::vector<u32>& p, std::size_t words) {  // of p mod (x^words - 1), Montgomery
        std::vector<Line> memory(words / 16 + 2);
        u32* const t = memory[0].w;
        std::fill(t, t + words, 0u);
        for (std::size_t j = 0; j < p.size(); ++j) t[j % words] = u32((t[j % words] + u64(to_montgomery(p[j]))) % P);
        pt::forward(t, words);
        return std::vector<u32>(t, t + words);
    };
    if (!tree.base(0, count)) {
        expect(h.root_room != nullptr, "standard root room", count);
        if (h.root_room) expect(transform_of(all, h.root_words) == std::vector<u32>(h.root_room, h.root_room + h.root_words), "standard root transform", count);
    }
    for (const Entry& e : h.entries) {
        expect(tree.split(e.lo, e.hi) == e.mid && !tree.base(e.lo, e.hi), "standard node structure", count);
        expect(transform_of(product(e.lo, e.mid), e.words) == std::vector<u32>(e.left, e.left + e.words), "standard kept left", count);
        expect(transform_of(product(e.mid, e.hi), e.words) == std::vector<u32>(e.right, e.right + e.words), "standard kept right", count);
    }
}

}  // namespace

int main() {
    for (std::size_t count : {1, 2, 16, 17, 33, 40, 100, 257}) test_lanes(count);
    for (std::size_t count : {1, 2, 3, 5, 9, 33}) {
        test_standard(count, 8);
        test_standard(count, 70);
    }
    test_standard(8, 300);
    std::printf("%s product_tree keep: %d checks, %d failures\n", failures ? "FAIL" : "pass", checks, failures);
    return failures ? 1 : 0;
}
