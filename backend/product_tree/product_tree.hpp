// Products of many polynomials modulo P = 998244353 by product trees (backend kernel).
//
//   fps::product_tree::size(fs)           // coefficients of the product: 1 + the sum of
//                                         // (f.size() - 1); 0 if a factor is empty, 1 if none
//   fps::product_tree::product(fs, out);  // out = f_0 f_1 ... f_(k-1), out.size() == size(fs)
//   fps::product_tree::size(items, proj), fps::product_tree::product(items, out, proj);
//                                         // the factors proj(items[i]), std::span<const uint32_t>:
//                                         // no array of spans, e.g. for a span of polynomials
//
// fs: a span of spans of words < P, any alignment; out: the caller's memory, not overlapping them,
// canonical out. Throws std::invalid_argument if out.size() != size(fs), and std::length_error,
// before writing out, if size(fs) - 1 > 2^kMaxLog (ntt.hpp). The product's length counts each
// factor's size, so trailing zero coefficients are kept as such.
//
// Constants are multiplied into one number. The other factors: if at least kMinLaneFactors
// (select.hpp) have degree at most 1/64 of those factors' sum, they are sorted by degree and dealt
// to 8 lanes, and one tree multiplies the 8 lanes side by side (pointwise products of transforms);
// a standard tree (one polynomial per node, leaf products) multiplies the 8 lane products and the
// larger factors. Both trees use one transform of each length per node (transform doubling, the
// wrap trick): tree.hpp. Memory: one region of this thread, kept across calls (ntt's), and ntt's
// transforms per length, kept per thread. Design and measurements: notes.md.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "backend/product_tree/tree.hpp"

namespace fps::product_tree {

namespace detail {

struct Identity {
    std::span<const std::uint32_t> operator()(std::span<const std::uint32_t> f) const { return f; }
};

// fs[i] = proj(items[i]).
template <class T, class Proj>
struct Factors {
    std::span<const T> items;
    Proj proj;

    std::size_t size() const { return items.size(); }
    std::span<const u32> operator[](std::size_t i) const { return proj(items[i]); }
};

}  // namespace detail

template <class T, class Proj>
std::size_t size(std::span<const T> items, Proj proj) {
    std::size_t n = 1;
    for (const T& item : items) {
        const std::span<const std::uint32_t> f = proj(item);
        if (f.empty()) return 0;
        n += f.size() - 1;
    }
    return n;
}

inline std::size_t size(std::span<const std::span<const std::uint32_t>> fs) { return size(fs, detail::Identity{}); }

namespace detail {

// Which trees run: automatic as above; standard alone; lanes for every nonconstant factor.
enum class Mode { automatic, standard, lanes };

// Slots of 8 factors: slot k holds factors order[8k + l], l < 8; past the end of order, 1.
template <class F>
struct Slots {
    const F& fs;
    std::span<const u32> order;  // indices into fs, by degree, largest first

    std::size_t count() const { return (order.size() + 7) / 8; }
    u32 degree(std::size_t k) const { return u32(fs[order[8 * k]].size() - 1); }

    LaneNode load(std::size_t k, u32* c) const {
        static constexpr u32 kOne = 1;
        const u32 d = degree(k);
        const u32* p[8];
        LaneNode node;
        bool full = true;
        for (std::size_t l = 0; l < 8; ++l) {
            if (8 * k + l < order.size()) {
                const std::span<const u32> f = fs[order[8 * k + l]];
                p[l] = f.data(), node.degree[l] = u32(f.size() - 1);
            } else {
                p[l] = &kOne, node.degree[l] = 0;
            }
            full &= node.degree[l] == d;
        }
        if (full) {
            for (u32 j = 0; j <= d; ++j)
                for (std::size_t l = 0; l < 8; ++l) c[8 * j + l] = p[l][j];
        } else {
            for (std::size_t l = 0; l < 8; ++l) {
                for (u32 j = 0; j <= node.degree[l]; ++j) c[8 * j + l] = p[l][j];
                for (u32 j = node.degree[l] + 1; j <= d; ++j) c[8 * j + l] = 0;
            }
        }
        scale<Wide>(c, kR2, c, 8 * (d + 1));
        for (std::size_t l = 0; l < 8; ++l) node.lead[l] = c[8 * node.degree[l] + l];
        return node;
    }
};

// Gives back ntt's region of this thread on any exit.
struct RegionGuard {
    ntt::detail::Region region;
    ~RegionGuard() { ntt::detail::give_region(region); }
};

template <class F>
void product(const F& fs, std::span<u32> out, Mode mode) {
    // The constants' product; the other factors' count, degree sum and largest degree.
    u32 constant = 1, top = 0;
    std::size_t count = 0, last = 0;
    u64 total = 0;
    bool empty = false;
    for (std::size_t i = 0; i < fs.size(); ++i) {
        const std::size_t s = fs[i].size();
        if (s <= 1) {
            if (s == 0) empty = true;
            else constant = multiply_mod(constant, fs[i][0]);
            continue;
        }
        total += s - 1, ++count, last = i;
        top = u32(std::min<u64>(std::max<u64>(top, s - 1), u64(1) << 31));
    }
    const std::size_t n = empty ? 0 : total + 1;
    if (out.size() != n) throw std::invalid_argument("fps::product_tree::product: out.size() != size(fs)");
    if (n == 0) return;
    if (total > (u64(1) << ntt::kMaxLog))
        throw std::length_error("fps::product_tree::product: product longer than 2^kMaxLog + 1");
    if (count == 0) return void(out[0] = constant);
    if (count == 1) {  // f c: montgomery(x, c 2^32) = x c
        const std::span<const u32> f = fs[last];
        if (constant == 1) std::copy(f.begin(), f.end(), out.begin());
        else scale<Wide>(f.data(), multiply_mod(constant, kR), out.data(), n);
        return;
    }

    // Small factors (lanes): degree at most cut. Region: the counting sort's starts, the standard
    // tree's leaves and the small factors' order, then the lane tree (below 4 lengths of 8 words in
    // balanced trees), whose root stays for the standard tree (below 4 lengths). Past it, Stack
    // spills to the heap. Nothing of size count goes to the heap: fresh pages cost about 0.4 ms per
    // MB on Zen 3 (notes.md).
    const u32 cut = std::min(top, mode == Mode::lanes ? top : mode == Mode::standard ? 0 : u32(total / 64));
    const std::size_t lane_length = LaneTree<Slots<F>>::length(u32(cut + total / 8 + 1));
    const std::size_t lane_words = 8 * 4 * (lane_length + 1);
    const std::size_t standard_words = 8 * (lane_length + 1) + 4 * (StandardTree::length(u32(total)) + 1);
    constexpr std::size_t kLeafWords = sizeof(Leaf) / sizeof(u32);
    const std::size_t words = cut + 2 + (kLeafWords + 1) * (count + 8) + std::max(lane_words, standard_words) + 8192;
    RegionGuard guard{ntt::detail::take_region(words * sizeof(u32))};
    Stack stack(reinterpret_cast<u32*>(guard.region.start), guard.region.bytes / sizeof(u32));

    // A counting sort of the small factors by degree, largest first: start[cut - d + 1] counts
    // degree d, then start[cut - d] is where the next of degree d goes. Stable: the slots read
    // factors of one degree in their order (notes.md).
    u32* const start = stack.take(cut + 2);
    std::fill(start, start + cut + 2, 0u);
    std::size_t small = 0;
    for (std::size_t i = 0; i < fs.size(); ++i) {
        const std::size_t s = fs[i].size();
        if (s >= 2 && s - 1 <= cut) ++start[cut - (s - 1) + 1], ++small;
    }
    if (mode == Mode::automatic && small < kMinLaneFactors) small = 0;
    static_assert(sizeof(Leaf) % sizeof(u32) == 0 && alignof(Leaf) <= 64);
    Leaf* const leaves = reinterpret_cast<Leaf*>(stack.take(kLeafWords * (count - small + 8)));
    u32* const order = small ? stack.take(small) : nullptr;
    std::size_t leaf_count = 0;
    for (u32 d = 1; d <= cut + 1; ++d) start[d] += start[d - 1];
    for (std::size_t i = 0; i < fs.size(); ++i) {
        const std::size_t s = fs[i].size();
        if (s < 2) continue;
        if (small && s - 1 <= cut) order[start[cut - (s - 1)]++] = u32(i);
        else leaves[leaf_count++] = {fs[i].data(), u32(s - 1), false};
    }
    if (small) {
        const Slots<F> slots{fs, {order, small}};
        LaneTree<Slots<F>> tree(slots, stack);
        const auto root = tree.root();
        for (std::size_t l = 0; l < 8; ++l)  // a lane of degree 0 holds 1: only with fewer than 8 small factors
            if (root.node.degree[l]) leaves[leaf_count++] = {root.coefficients + l, root.node.degree[l], true, 8};
    }
    StandardTree tree({leaves, leaf_count}, stack);
    scale<Wide>(tree.root(), constant, out.data(), n);
}

inline void product(std::span<const std::span<const u32>> fs, std::span<u32> out, Mode mode) {
    product(Factors<std::span<const u32>, Identity>{fs, {}}, out, mode);
}

}  // namespace detail

template <class T, class Proj>
void product(std::span<const T> items, std::span<std::uint32_t> out, Proj proj) {
    detail::product(detail::Factors<T, Proj>{items, proj}, out, detail::Mode::automatic);
}

inline void product(std::span<const std::span<const std::uint32_t>> fs, std::span<std::uint32_t> out) {
    detail::product(fs, out, detail::Mode::automatic);
}

}  // namespace fps::product_tree
