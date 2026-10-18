// Tests for two hooks backend/evaluation uses (issue #136 round 2): LaneTree::root() writes the
// root's product transform (the lanes at the roots of X^L - 1) to keep.root(8 L), against the
// forward of the root's lanes mod X^L - 1, with the coefficients as without the hook; and
// lane_pairs' copy output, the same words as its output. Exits nonzero on a failure.
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

std::mt19937_64 rng(1363);
int failures = 0, checks = 0;

void expect(bool ok, const std::string& what, u64 detail = 0) {
    ++checks;
    if (!ok && ++failures <= 20) std::printf("FAIL: %s (%llu)\n", what.c_str(), static_cast<unsigned long long>(detail));
}

struct Words {
    std::vector<u32> memory;
    u32* data;
    explicit Words(std::size_t n, u32 fill = 0) : memory(n + fps::ntt::kPadding + 16, fill) {
        data = memory.data() + (16 - reinterpret_cast<std::uintptr_t>(memory.data()) / 4 % 16) % 16;
    }
};

struct Factors {
    const std::vector<Vec>* fs;
    std::size_t size() const { return fs->size(); }
    std::span<const u32> operator[](std::size_t i) const { return (*fs)[i]; }
};

// A Keep that keeps no children and takes the root's product transform into `room`.
struct RootOnly {
    u32* room;
    std::size_t* words;
    pt::Rooms place(std::size_t, std::size_t, std::size_t, std::size_t) const { return {}; }
    u32* root(std::size_t n) const {
        *words = n;
        return room;
    }
};

void test_root(std::size_t count, u32 degree) {
    const std::string where = "lane root transform: " + std::to_string(count) + " factors of degree " + std::to_string(degree);
    std::vector<Vec> fs(count, Vec(degree + 1));
    for (Vec& f : fs)
        for (u32& x : f) x = u32(rng() % P);
    const pt::Slots<Factors> slots{Factors{&fs}, nullptr, count};
    Words region(1 << 22), other(1 << 22);
    pt::Stack stack(region.data, 1 << 22), other_stack(other.data, 1 << 22);
    pt::LaneTree<pt::Slots<Factors>> plain(slots, other_stack);
    const auto want = plain.root();
    const std::size_t len = want.length;
    Words room(8 * len + 64, 0xdeadbeefu);
    std::size_t words = 0;
    pt::LaneTree<pt::Slots<Factors>, RootOnly> tree(slots, stack, RootOnly{room.data, &words});
    const auto root = tree.root();
    expect(root.length == len && std::equal(root.coefficients, root.coefficients + 8 * (len + 1), want.coefficients),
           where + ": coefficients as without the hook");
    expect(words == 8 * len, where + ": root(8 L)", words);
    // The lanes mod X^L - 1: the top coefficient (at L) wraps onto 0.
    Words t(8 * len);
    for (std::size_t j = 0; j < len; ++j)
        for (int l = 0; l < 8; ++l) t.data[8 * j + l] = root.coefficients[8 * j + l];
    for (int l = 0; l < 8; ++l) t.data[l] = u32((t.data[l] + u64(root.coefficients[8 * len + l])) % P);
    pt::forward(t.data, 8 * len);
    std::size_t bad = 0;
    for (std::size_t i = 0; i < 8 * len; ++i) bad += room.data[i] != t.data[i];
    expect(bad == 0, where + ": the root's product transform", bad);
    bool guard = true;
    for (std::size_t i = 8 * len; i < 8 * len + 64; ++i) guard &= room.data[i] == 0xdeadbeefu;
    expect(guard, where + ": wrote past it");
}

template <class L>
void test_copy(const char* name, std::size_t T) {
    Words v(8 * T);
    for (std::size_t i = 0; i < 8 * T; ++i) v.data[i] = u32(rng() % P);
    std::vector<Words> outs, copies;
    for (int j = 0; j < 4; ++j) outs.emplace_back(T, 1), copies.emplace_back(T + 64, 0xdeadbeefu);
    u32* out[4];
    u32* copy[4];
    for (int j = 0; j < 4; ++j) out[j] = j == 3 ? nullptr : outs[j].data, copy[j] = copies[j].data;
    pt::lane_pairs<L>(v.data, T, out, pt::untwist_factors(T / 8), copy);
    for (int j = 0; j < 3; ++j) expect(std::equal(out[j], out[j] + T, copy[j]), std::string("lane_pairs copy <") + name + ">", T);
    bool untouched = true;
    for (std::size_t i = 0; i < T + 64; ++i) untouched &= copies[3].data[i] == 0xdeadbeefu;
    expect(untouched, std::string("lane_pairs copy of a null pair <") + name + ">", T);
}

}  // namespace

int main() {
    for (const auto [count, degree] : {std::pair<std::size_t, u32>{136, 1}, {256, 1}, {296, 1}, {1024, 1}, {200, 3}, {64, 4}, {4096, 1}})
        test_root(count, degree);
    for (const std::size_t T : {64, 256, 1024}) {
        test_copy<fps::simd::Scalar>("Scalar", T);
        test_copy<fps::product_tree::Lanes>("Lanes", T);
    }
    std::printf("%s product_tree root copy: %d checks, %d failures\n", failures ? "FAIL" : "pass", checks, failures);
    return failures ? 1 : 0;
}
