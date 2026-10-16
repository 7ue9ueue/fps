// The point tree and its passes: the product tree of Q = prod (1 - a_i x) with every node's
// children's transforms kept, the root's window by a division, the transposed descent (values),
// its transpose (interpolation's sums) and the Newton basis' descent. Design: notes.md.
//
// Points: m8 = 8 C of them (C slots of 8 lanes), C > kLaneBase. Slot k of lane l is x[8 k + l];
// its factor 1 - x[8 k + l] x. The lane tree (backend/product_tree LaneTree) multiplies the 8 lanes
// side by side; the top tree (StandardTree) the 8 lane products, lane l its leaf l. Both keep each
// node's children's transforms at the node's length (Keep, Store): for the descents.
//
// Windows (Bostan, Lecerf and Schost's transposed evaluation): for a series G and K >= m8, node v
// of s_v points gets W_v = (G / Q_v)[K - s_v, K); a child c with sibling d gets W_c = (W_v
// Q_d)[s_d, s_v), exact in a cyclic product of length L >= s_v. A point's window is [x^(K-1)] G /
// (1 - a x): f(a) for G = rev_K(f). A node's state is f_v times the transform of length L of its
// window right-aligned (terms below it are garbage, harmless); a child of half the length takes
// the upper half of the product by halving (to_child): one product, one inverse_upper and one
// forward of L / 2 words, factor f_c = 2 f_v.
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>

#include "backend/evaluation/kernels.hpp"
#include "backend/product_tree/tree.hpp"
#include "backend/series/series.hpp"

namespace fps::evaluation::detail {

using product_tree::kLaneBase;
using product_tree::detail::LaneNode;
using product_tree::detail::LaneTree;
using product_tree::detail::Leaf;
using product_tree::detail::leaf_products;
using product_tree::detail::leaf_weights;
using product_tree::detail::multiply_lanes;
using product_tree::detail::pointwise;
using product_tree::detail::Rooms;
using product_tree::detail::scale;
using product_tree::detail::Stack;
using product_tree::detail::StandardTree;
using product_tree::detail::transform;

inline void forward(u32* f, std::size_t words) { transform(words).forward(f); }
inline void inverse(u32* f, std::size_t words) { transform(words).inverse(f); }
inline void forward_upper(u32* h, std::size_t words) { transform(words).forward_upper(h); }
inline void inverse_upper(u32* h, std::size_t words) { transform(words).inverse_upper(h); }

// Node lengths: LaneTree::length and StandardTree::length.
inline std::size_t lane_length(std::size_t slots) { return std::max<std::size_t>(8, std::bit_ceil(slots)); }
inline std::size_t top_length(std::size_t degree) { return StandardTree<>::length(u32(degree)); }

// Slot k of lane l: the factor 1 - x[8 k + l] x, in Montgomery form.
struct PointSlots {
    const u32* x;
    std::size_t slots;

    std::size_t count() const { return slots; }
    u32 degree(std::size_t) const { return 1; }
    LaneNode load(std::size_t k, u32* c) const {
        using L = Lanes;
        LaneNode node;
        for (int l = 0; l < 8; l += L::lanes) {
            const typename L::V lead = montgomery_canonical<L>(Ops<L>::sub(L::broadcast(0), L::load(x + 8 * k + l)), L::broadcast(kR2));
            L::store(c + l, L::broadcast(kR)), L::store(c + 8 + l, lead);
            L::store(node.degree + l, L::broadcast(1)), L::store(node.lead + l, lead);
        }
        return node;
    }
};

// A tree's kept transforms: node [lo, hi)'s children's (or the right child's alone) in rooms of
// one block, found by the node's split point.
class Store {
public:
    struct Keeper {
        Store* store;
        Rooms place(std::size_t, std::size_t mid, std::size_t, std::size_t words) const { return store->place(mid, words); }
        u32* root(std::size_t words) const { return words == store->root_words_ ? store->root_ : nullptr; }
    };

    // Rooms for a tree of `count` leaves, `words` words in all, from the stack.
    void reset(Stack& stack, std::size_t count, std::size_t words, bool left) {
        base_ = stack.take(words), size_ = words, used_ = 0, left_ = left;
        offset_ = reinterpret_cast<std::size_t*>(stack.take(2 * (count + 1)));
    }

    Rooms place(std::size_t mid, std::size_t words) {
        const std::size_t room = Stack::footprint(words);
        if (used_ + (left_ ? 2 : 1) * room > size_) throw std::logic_error("fps::evaluation: kept transforms past their block");
        offset_[mid] = used_;
        Rooms r;
        if (left_) r.left = base_ + used_, used_ += room;
        r.right = base_ + used_, used_ += room;
        return r;
    }

    // Room for the root's product transform of `words` words (StandardTree::root).
    void keep_root(Stack& stack, std::size_t words) { root_ = stack.take(words), root_words_ = words; }
    const u32* root() const { return root_; }

    const u32* left(std::size_t mid) const { return base_ + offset_[mid]; }
    const u32* right(std::size_t mid, std::size_t words) const { return base_ + offset_[mid] + (left_ ? Stack::footprint(words) : 0); }

    // Words of the rooms of node [lo, hi) and below, for a tree's structure and words per length.
    template <class Tree, class Words>
    static std::size_t words(const Tree& tree, std::size_t lo, std::size_t hi, Words of, bool left) {
        if (tree.base(lo, hi)) return 0;
        const std::size_t mid = tree.split(lo, hi);
        return (left ? 2 : 1) * Stack::footprint(of(tree.degree(lo, hi))) + words(tree, lo, mid, of, left) + words(tree, mid, hi, of, left);
    }

private:
    u32* base_ = nullptr;
    u32* root_ = nullptr;
    std::size_t size_ = 0, used_ = 0, root_words_ = 0;
    std::size_t* offset_ = nullptr;  // by split point
    bool left_ = true;
};

// Words of the lane tree's rooms for C slots (its split and base for slots of degree 1: the
// middle, and at most kLaneBase slots), for sizing memory before the tree exists.
inline std::size_t lane_kept_words(std::size_t slots, bool left) {
    if (slots <= kLaneBase) return 0;
    return (left ? 2 : 1) * Stack::footprint(8 * lane_length(slots)) + lane_kept_words(slots / 2, left) +
           lane_kept_words(slots - slots / 2, left);
}

// The base of the lane tree (at most kLaneBase slots): the products of its nodes by schoolbook,
// nodes split at the middle, in heap order (node 1 the base node, children 2v, 2v + 1).
class BaseProducts {
public:
    BaseProducts(const PointSlots& slots, std::size_t lo, std::size_t hi) : slots_(slots) { build(1, lo, hi); }

    // The product of node v: degree + 1 coefficients of 8 words, Montgomery form. Not node 1's.
    const u32* operator[](std::size_t v) const { return product_[v]; }

private:
    void build(std::size_t v, std::size_t lo, std::size_t hi) {
        u32* const c = free_;
        if (hi - lo == 1) {
            slots_.load(lo, c);
            free_ += 16;
            product_[v] = c;
            return;
        }
        const std::size_t mid = lo + (hi - lo) / 2;
        build(2 * v, lo, mid);
        build(2 * v + 1, mid, hi);
        if (v == 1) return;
        multiply_lanes<Lanes>(product_[2 * v], mid - lo, product_[2 * v + 1], hi - mid, free_);
        product_[v] = free_;
        free_ += 8 * (hi - lo + 1);
    }

    // 16 slots: 2 coefficients per slot and s + 1 per node of s slots below the top: 32 + 24 +
    // 20 + 18 coefficients.
    static constexpr std::size_t kWords = 8 * (6 * kLaneBase + 8);
    alignas(64) u32 memory_[kWords];
    const u32* product_[4 * kLaneBase];
    u32* free_ = memory_;
    const PointSlots& slots_;
};

// Layouts of the passes: the top tree's (one polynomial per transform, leaf products) or the lanes'
// (8 side by side, pointwise products). W words per coefficient.
template <int W>
struct Layout {
    static constexpr int kWidth = W;
    // out = a b, b a kept transform (Montgomery form): a's form out.
    static void product(const u32* a, const u32* b, u32* out, std::size_t words) {
        if constexpr (W == 1) leaf_products<Lanes>(a, b, out, words / 8, leaf_weights(words / 8));
        else pointwise<Wide>(a, b, out, words);
    }
    // out = a b + c d, as product(); and the same into copy if not null.
    static void product_sums(const u32* a, const u32* b, const u32* c, const u32* d, u32* out, u32* copy, u32* scratch,
                             std::size_t words) {
        if constexpr (W == 1) {
            const u32* weights = leaf_weights(words / 8);
            leaf_products<Lanes>(a, b, out, words / 8, weights);
            leaf_products<Lanes>(c, d, scratch, words / 8, weights);
            add_words<Wide>(out, scratch, out, words);
            if (copy) std::copy_n(out, words, copy);
        } else {
            pointwise_sums<Wide>(a, b, c, d, out, copy, words);
        }
    }
};

// p (the transform of length L of a polynomial X, W L words) -> in place, the transform of length
// Lc of X[e - Lc, e), terms below e - Lc garbage; times 2 if it halves (returns true). Halving:
// Lc = L / 2 and e = L (P_lo - T(X mod (x^(L/2) + 1)) = 2 T(X_hi)) or e = L / 2 (P_lo + ... = 2
// T(X_lo)); else an inverse, the words moved, a forward.
template <int W>
bool to_child(u32* p, std::size_t length, std::size_t e, std::size_t child_length) {
    const std::size_t words = W * length, h = words / 2;
    if (child_length == length && e == length) return false;
    if (2 * child_length == length && (e == length || 2 * e == length)) {
        inverse_upper(p + h, words);
        forward(p + h, h);
        combine_halves<Wide>(p, h, e == length ? -1 : 1);
        return true;
    }
    inverse(p, words);
    std::memmove(p, p + W * (e - child_length), W * child_length * sizeof(u32));
    forward(p, W * child_length);
    return false;
}

class PointTree {
public:
    // x: 8 slots words (slot order), canonical, slots > kLaneBase; kept until the tree is gone.
    // left: keep left children's transforms too (evaluate, interpolate), else right ones only (the
    // Newton basis). Everything goes in the stack.
    PointTree(const u32* x, std::size_t slots, Stack& stack, bool left)
        : slots_{x, slots}, count_(slots), m8_(8 * slots), lane_length_(lane_length(slots)), top_length_(top_length(8 * slots)),
          lanes_(slots_, stack, Store::Keeper{&lane_store_}) {
        const auto lane_words = [](u32 d) { return 8 * lane_length(d); };
        lane_store_.reset(stack, count_, Store::words(lanes_, 0, count_, lane_words, left), left);
        const auto root = lanes_.root();
        lane_root_ = root.coefficients;
        for (std::size_t l = 0; l < 8; ++l) leaves_[l] = {lane_root_ + l, u32(count_), true, 8};
        top_.emplace(std::span<const Leaf>(leaves_), stack, Store::Keeper{&top_store_});
        const auto top_words = [](u32 d) { return top_length(d); };
        top_store_.reset(stack, 8, Store::words(*top_, 0, 8, top_words, left), left);
        top_store_.keep_root(stack, top_length_);
        q_ = top_->root();
    }

    PointTree(const PointTree&) = delete;
    PointTree& operator=(const PointTree&) = delete;

    std::size_t size() const { return m8_; }

    // Q = prod (1 - a_i x): size() + 1 coefficients, Montgomery form, in the stack.
    const u32* product() const { return q_; }
    // The transform of Q mod (x^L - 1), L = top_length(size()), Montgomery form.
    const u32* product_transform() const { return top_store_.root(); }

    // Stack words the passes take past the tree, at most.
    static std::size_t pass_words(std::size_t m8) {
        const std::size_t lanes = 8 * lane_length(m8 / 8), top = top_length(m8);
        return 10 * Stack::footprint(lanes) + 8 * Stack::footprint(top) + 4096;
    }

    // values[8 k + l] = the window of slot k of lane l, from the root's window w (m8 words) of
    // the evaluation's descent; canonical.
    void descend(const u32* w, u32* values, Stack& stack) const {
        const Stack::Mark mark = stack.mark();
        u32* const lanes = stack.take(8 * lane_length_);
        u32* const rows = stack.take(8 * lane_length_);
        u32* const s = stack.take(top_length_);
        std::fill(s, s + top_length_ - m8_, 0u);
        std::copy_n(w, m8_, s + top_length_ - m8_);
        forward(s, top_length_);
        descend_top(0, 8, s, 1, rows, stack);
        rows_to_lanes<Lanes>(rows, lane_length_, lane_length_, lanes);
        forward(lanes, 8 * lane_length_);
        descend_lanes(0, count_, lanes, 1, values, stack);
        stack.release(mark);
    }

    // The Newton basis' descent: values[8 k + l] = the window of slot k of lane l (with x in the
    // blocked order: point l C + k there), from the root's window w (m8 words). Needs a tree
    // that keeps right children's transforms.
    void descend_newton(const u32* w, u32* values, Stack& stack) const {
        const Stack::Mark mark = stack.mark();
        u32* const lanes = stack.take(8 * lane_length_);
        u32* const rows = stack.take(8 * lane_length_);
        u32* const s = stack.take(top_length_);
        std::fill(s, s + top_length_ - m8_, 0u);
        std::copy_n(w, m8_, s + top_length_ - m8_);
        forward(s, top_length_);
        newton_top(0, 8, s, 1, rows, stack);
        rows_to_lanes<Lanes>(rows, lane_length_, lane_length_, lanes);
        forward(lanes, 8 * lane_length_);
        newton_lanes(0, count_, lanes, 1, values, stack);
        stack.release(mark);
    }

    // Interpolation, the descent and its transpose in one pass, depth first: from the root's
    // window w (m8 words) of rev(M') / Q (M = prod (x - a_i) over the m points) and the values y
    // (m8 words, slot order, zero past m), r = R = sum_i (y_i / M'(a_i)) prod_(j != i) (1 - a_j x),
    // canonical (top length words, R's in the first m8). Each lane node's sum R_v = R_l Q_r + R_r
    // Q_l right after its children's descents and sums, while its kept transforms are in cache; a
    // base inverts its values once. Throws std::domain_error if some M'(a_i) is 0, i < m.
    void interpolate(const u32* w, const u32* y, std::size_t m, u32* r, Stack& stack) const {
        const Stack::Mark mark = stack.mark();
        u32* const lanes = stack.take(8 * lane_length_);
        u32* const rows = stack.take(8 * lane_length_);
        {
            const Stack::Mark top = stack.mark();
            u32* const s = stack.take(top_length_);
            std::fill(s, s + top_length_ - m8_, 0u);
            std::copy_n(w, m8_, s + top_length_ - m8_);
            forward(s, top_length_);
            descend_top(0, 8, s, 1, rows, stack);
            stack.release(top);
        }
        rows_to_lanes<Lanes>(rows, lane_length_, lane_length_, lanes);
        forward(lanes, 8 * lane_length_);
        u32* const sums = stack.take(8 * lane_length_);
        interpolate_lanes(0, count_, lanes, 1, y, m, sums, lane_length_, stack);
        inverse(sums, 8 * lane_length_);
        lanes_to_rows<Lanes>(sums, lane_length_, rows, lane_length_);
        ascend_top(0, 8, rows, r, top_length_, stack);
        inverse(r, top_length_);
        stack.release(mark);
    }

private:
    using Top = Layout<1>;
    using Lane = Layout<8>;

    // Factor u = 1 / f_v times 2^32 mod P, for scale().
    static u32 montgomery_factor(u32 u) { return multiply_mod(u, kR); }

    // The top tree's lane products' windows (right-aligned in lane_length_ coefficients, times u)
    // into row l of rows (lane_length_ words each; rows_to_lanes makes them the lanes): p is the
    // transform of length `length` of X; the window ends at e.
    void to_lane(u32* p, std::size_t length, std::size_t e, std::size_t l, u32 u, u32* rows) const {
        inverse(p, length);
        scale<Wide>(p + e - lane_length_, montgomery_factor(u), rows + l * lane_length_, lane_length_);
    }

    // Node [lo, hi) of the top tree with state s (factor 1 / u).
    void descend_top(std::size_t lo, std::size_t hi, const u32* s, u32 u, u32* lanes, Stack& stack) const {
        const StandardTree<Store::Keeper>& tree = *top_;
        const std::size_t length = top_length(tree.degree(lo, hi)), mid = tree.split(lo, hi);
        for (int side = 0; side < 2; ++side) {
            const std::size_t clo = side ? mid : lo, chi = side ? hi : mid;
            const Stack::Mark mark = stack.mark();
            u32* const p = stack.take(length);
            Top::product(s, side ? top_store_.left(mid) : top_store_.right(mid, length), p, length);
            if (chi - clo == 1) {
                to_lane(p, length, length, clo, u, lanes);
            } else {
                const bool halved = to_child<1>(p, length, length, top_length(tree.degree(clo, chi)));
                descend_top(clo, chi, p, halved ? half(u) : u, lanes, stack);
            }
            stack.release(mark);
        }
    }

    // Node [lo, hi) of the lane tree with state s (factor 1 / u).
    void descend_lanes(std::size_t lo, std::size_t hi, const u32* s, u32 u, u32* values, Stack& stack) const {
        const std::size_t length = lane_length(lanes_.degree(lo, hi)), words = 8 * length, mid = lanes_.split(lo, hi);
        for (int side = 0; side < 2; ++side) {
            const std::size_t clo = side ? mid : lo, chi = side ? hi : mid;
            const Stack::Mark mark = stack.mark();
            u32* const p = stack.take(words);
            Lane::product(s, side ? lane_store_.left(mid) : lane_store_.right(mid, words), p, words);
            if (lanes_.base(clo, chi)) {
                inverse(p, words);
                u32* const w = p + 8 * (length - (chi - clo));
                scale<Wide>(w, montgomery_factor(u), w, 8 * (chi - clo));
                base_descend(clo, chi, w, values + 8 * clo);
            } else {
                const bool halved = to_child<8>(p, length, length, lane_length(lanes_.degree(clo, chi)));
                descend_lanes(clo, chi, p, halved ? half(u) : u, values, stack);
            }
            stack.release(mark);
        }
    }

    // The descent below a base node from its window w (hi - lo coefficients of 8 words): its
    // slots' windows to out (8 words per slot, slot lo first).
    void base_descend(std::size_t lo, std::size_t hi, const u32* w, u32* out) const {
        const BaseProducts products(slots_, lo, hi);
        base_descend(products, 1, lo, hi, w, out);
    }
    void base_descend(const BaseProducts& products, std::size_t v, std::size_t lo, std::size_t hi, const u32* w, u32* out) const {
        if (hi - lo == 1) return void(std::copy_n(w, 8, out));
        const std::size_t mid = lo + (hi - lo) / 2;
        alignas(64) u32 left[8 * kLaneBase], right[8 * kLaneBase];
        middle_lanes<Lanes>(w, hi - mid, products[2 * v + 1], mid - lo, left);
        middle_lanes<Lanes>(w, mid - lo, products[2 * v], hi - mid, right);
        base_descend(products, 2 * v, lo, mid, left, out);
        base_descend(products, 2 * v + 1, mid, hi, right, out + 8 * (mid - lo));
    }

    // The Newton basis: node v's right child's window is W_v[0, s_r), its left child's (W_v
    // Q_r)[s_r, s_v): the left one from the product, the right one from the state (to_child with
    // e = L - s_l). The product first, as the right child's state overwrites s.
    void newton_top(std::size_t lo, std::size_t hi, u32* s, u32 u, u32* lanes, Stack& stack) const {
        const StandardTree<Store::Keeper>& tree = *top_;
        const std::size_t length = top_length(tree.degree(lo, hi)), mid = tree.split(lo, hi);
        const std::size_t e = length - tree.degree(lo, mid);
        const Stack::Mark mark = stack.mark();
        u32* const p = stack.take(length);
        Top::product(s, top_store_.right(mid, length), p, length);
        if (mid - lo == 1) {
            to_lane(p, length, length, lo, u, lanes);
            to_lane(s, length, e, mid, u, lanes);
        } else {
            const std::size_t child_length = top_length(tree.degree(lo, mid));  // both children's
            const bool left_halved = to_child<1>(p, length, length, child_length);
            const bool right_halved = to_child<1>(s, length, e, child_length);
            newton_top(lo, mid, p, left_halved ? half(u) : u, lanes, stack);
            newton_top(mid, hi, s, right_halved ? half(u) : u, lanes, stack);
        }
        stack.release(mark);
    }

    void newton_lanes(std::size_t lo, std::size_t hi, u32* s, u32 u, u32* values, Stack& stack) const {
        const std::size_t length = lane_length(lanes_.degree(lo, hi)), words = 8 * length, mid = lanes_.split(lo, hi);
        const std::size_t e = length - (mid - lo);
        const Stack::Mark mark = stack.mark();
        u32* const p = stack.take(words);
        Lane::product(s, lane_store_.right(mid, words), p, words);
        const std::size_t sides[2][3] = {{lo, mid, length}, {mid, hi, e}};
        u32* const states[2] = {p, s};
        bool halved[2] = {};
        for (int side = 0; side < 2; ++side) {
            const auto [clo, chi, end] = sides[side];
            if (lanes_.base(clo, chi)) {
                inverse(states[side], words);
                u32* const w = states[side] + 8 * (end - (chi - clo));
                scale<Wide>(w, montgomery_factor(u), w, 8 * (chi - clo));
            } else {
                halved[side] = to_child<8>(states[side], length, end, lane_length(lanes_.degree(clo, chi)));
            }
        }
        for (int side = 0; side < 2; ++side) {
            const auto [clo, chi, end] = sides[side];
            if (lanes_.base(clo, chi)) newton_base(clo, chi, states[side] + 8 * (end - (chi - clo)), values);
            else newton_lanes(clo, chi, states[side], halved[side] ? half(u) : u, values, stack);
        }
        stack.release(mark);
    }

    void newton_base(std::size_t lo, std::size_t hi, const u32* w, u32* values) const {
        const BaseProducts products(slots_, lo, hi);
        newton_base(products, 1, lo, hi, w, values);
    }
    void newton_base(const BaseProducts& products, std::size_t v, std::size_t lo, std::size_t hi, const u32* w, u32* values) const {
        if (hi - lo == 1) return void(std::copy_n(w, 8, values + 8 * lo));
        const std::size_t mid = lo + (hi - lo) / 2;
        alignas(64) u32 left[8 * kLaneBase];
        middle_lanes<Lanes>(w, hi - mid, products[2 * v + 1], mid - lo, left);
        newton_base(products, 2 * v, lo, mid, left, values);
        newton_base(products, 2 * v + 1, mid, hi, w, values);
    }

    // Interpolation. Node [lo, hi)'s sum R_v = sum over its points of w_i Q_v / (1 - a_i x) =
    // R_l Q_r + R_r Q_l, of degree < s_v <= L: its transform of length L from the children's
    // (at length L) and the kept ones; then, for a longer parent, doubled as the product tree's
    // nodes (inverse, forward_upper). out: the transform at out_length >= L.
    template <class Layout, class Child>
    void sums(const Store& store, std::size_t mid, std::size_t length, u32* out, std::size_t out_length, Stack& stack,
              Child child) const {
        constexpr int W = Layout::kWidth;
        const std::size_t words = W * length;
        const Stack::Mark mark = stack.mark();
        u32* const upper = out + words;
        u32* const a = out_length >= 2 * length ? upper : stack.take(words);
        u32* const b = stack.take(words);
        child(0, a);
        child(1, b);
        const bool longer = out_length > 2 * length;
        u32* const copy = out_length == length ? nullptr : longer ? stack.take(words + 8) : upper;
        Layout::product_sums(a, store.right(mid, words), b, store.left(mid), out, copy, b, words);
        if (copy) {
            inverse(copy, words);  // R_v: no wrap
            if (longer) std::copy_n(copy, words, upper);
            forward_upper(upper, 2 * words);
            for (std::size_t m = 2 * length; m < out_length; m *= 2) {  // R mod (x^m + 1) = R
                std::fill(std::copy_n(copy, words, out + W * m), out + 2 * W * m, 0u);
                forward_upper(out + W * m, 2 * W * m);
            }
        }
        stack.release(mark);
    }

    // Node [lo, hi) of the lane tree with descent state s (factor 1 / u): its children's
    // descents and subtrees, then its sum's transform of length out_length into out.
    void interpolate_lanes(std::size_t lo, std::size_t hi, const u32* s, u32 u, const u32* y, std::size_t m, u32* out,
                           std::size_t out_length, Stack& stack) const {
        const std::size_t length = lane_length(lanes_.degree(lo, hi)), words = 8 * length, mid = lanes_.split(lo, hi);
        sums<Lane>(lane_store_, mid, length, out, out_length, stack, [&](int side, u32* to) {
            const std::size_t clo = side ? mid : lo, chi = side ? hi : mid;
            const Stack::Mark mark = stack.mark();
            u32* const p = stack.take(words);
            Lane::product(s, side ? lane_store_.left(mid) : lane_store_.right(mid, words), p, words);
            if (lanes_.base(clo, chi)) {
                inverse(p, words);
                u32* const w = p + 8 * (length - (chi - clo));
                scale<Wide>(w, montgomery_factor(u), w, 8 * (chi - clo));
                base_interpolate(clo, chi, w, y, m, to);
                std::fill(to + 8 * (chi - clo), to + words, 0u);
                forward(to, words);
            } else {
                const bool halved = to_child<8>(p, length, length, lane_length(lanes_.degree(clo, chi)));
                interpolate_lanes(clo, chi, p, halved ? half(u) : u, y, m, to, length, stack);
            }
            stack.release(mark);
        });
    }

    // Node [lo, hi) of the top tree, from the lane products' sums (rows: lane l's coefficients at
    // l lane_length_).
    void ascend_top(std::size_t lo, std::size_t hi, const u32* rows, u32* out, std::size_t out_length, Stack& stack) const {
        const StandardTree<Store::Keeper>& tree = *top_;
        const std::size_t length = top_length(tree.degree(lo, hi)), mid = tree.split(lo, hi);
        sums<Top>(top_store_, mid, length, out, out_length, stack, [&](int side, u32* to) {
            const std::size_t clo = side ? mid : lo, chi = side ? hi : mid;
            if (chi - clo > 1) return ascend_top(clo, chi, rows, to, length, stack);
            std::fill(std::copy_n(rows + clo * lane_length_, count_, to), to + length, 0u);
            forward(to, length);
        });
    }

    // The sum of base node [lo, hi) (hi - lo coefficients of 8 words) from its slots' weights w
    // (8 words per slot, slot lo first).
    void base_ascend(const BaseProducts& products, std::size_t v, std::size_t lo, std::size_t hi, const u32* w, u32* out) const {
        if (hi - lo == 1) return void(std::copy_n(w, 8, out));
        const std::size_t mid = lo + (hi - lo) / 2;
        alignas(64) u32 left[8 * kLaneBase], right[8 * kLaneBase];
        base_ascend(products, 2 * v, lo, mid, w, left);
        base_ascend(products, 2 * v + 1, mid, hi, w + 8 * (mid - lo), right);
        cross_lanes<Lanes>(left, mid - lo, right, hi - mid, products[2 * v], products[2 * v + 1], out);
    }

    // Base node [lo, hi) of the interpolation from its window w: the descent to its slots' values
    // M'(a_i), the weights y_i / M'(a_i) (0 past the m points; one inversion per base), then its
    // sum (hi - lo coefficients of 8 words) into out from the same products.
    void base_interpolate(std::size_t lo, std::size_t hi, const u32* w, const u32* y, std::size_t m, u32* out) const {
        const BaseProducts products(slots_, lo, hi);
        alignas(64) u32 values[8 * kLaneBase];
        base_descend(products, 1, lo, hi, w, values);
        if (!divide_slots<Lanes>(values, y + 8 * lo, 8 * lo, m, 8 * (hi - lo)))
            throw std::domain_error("fps::evaluation::interpolate: repeated points");
        base_ascend(products, 1, lo, hi, values, out);
    }

    PointSlots slots_;
    std::size_t count_, m8_, lane_length_, top_length_;
    Store lane_store_, top_store_;
    LaneTree<PointSlots, Store::Keeper> lanes_;
    const u32* lane_root_ = nullptr;
    Leaf leaves_[8];
    std::optional<StandardTree<Store::Keeper>> top_;
    const u32* q_ = nullptr;
};

// d = G / q mod x^K for G = rev_K(f) (G[K - 1 - j] = f[j], f of n <= K words) and q of qn words
// (q[0] = 1, qn <= K + 1), by Karp and Markstein's division with k = ceil(K / 2): h = 1 / q mod
// x^k (fps::series::inverse), q0 = G h mod x^k, r = (G - q q0)[k, K), q1 = h r mod x^(K - k), d =
// q0 + x^k q1; each product one ntt::Transform::cyclic_product of length L = bit_ceil(K) >=
// max(2k - 1, K) (sources of at most L / 2 words: the forwards skip the zero half), in d itself
// (G, then q0 and r, then q1). All canonical. qt: the transform of q mod (x^L - 1) of length
// qt_length times 2^32 (PointTree::product_transform), used if that is L; r's pass takes the
// 2^32 out.
inline void root_window(const u32* f, std::size_t n, const u32* q, std::size_t qn, std::size_t big, u32* d, Stack& stack,
                        const u32* qt = nullptr, std::size_t qt_length = 0) {
    const std::size_t k = (big + 1) / 2, rest = big - k, len = std::max<std::size_t>(64, std::bit_ceil(big));
    const Stack::Mark mark = stack.mark();
    std::fill(d, d + big - n, 0u);
    std::reverse_copy(f, f + n, d + big - n);  // G
    u32* const h = stack.take(k);
    {
        const Stack::Mark inverse = stack.mark();
        const std::size_t words = series::inverse_scratch(k);
        series::inverse({q, std::min(qn, k)}, {h, k}, {stack.take(words), words});
        stack.release(inverse);
    }
    const ntt::Transform& t = transform(len);
    u32* const H = stack.take(len);
    u32* const work = stack.take(len);
    t.forward({h, k}, H);
    t.cyclic_product({d, k}, H, work, {d, k});  // q0
    const bool montgomery = qt && qt_length == len;
    if (!montgomery) {  // q mod (x^L - 1): deg q <= K <= L
        u32* const b = stack.take(len);
        const std::size_t terms = std::min(qn, len);
        std::fill(std::copy_n(q, terms, b), b + len, 0u);
        if (qn > len) b[0] = add_mod(b[0], q[len]);
        t.forward(b);
        qt = b;
    }
    // c = (q q0)[k, K), times 2^32 with the tree's qt: the upper half if k = L / 2, else the first K.
    u32* const c = stack.take(len);
    const std::size_t from = 2 * k == len ? k : 0;
    t.cyclic_product({d, k}, qt, work, {c, big - from}, {.offset = from});
    subtract_scaled<Wide>(d + k, c + k - from, montgomery ? 1 : kR, d + k, rest);  // r
    t.cyclic_product({d + k, rest}, H, work, {d + k, rest});  // q1
    stack.release(mark);
}

inline std::size_t division_words(std::size_t big) {
    const std::size_t k = (big + 1) / 2, len = std::max<std::size_t>(64, std::bit_ceil(big));
    return Stack::footprint(k) + std::max(Stack::footprint(series::inverse_scratch(k)), 4 * Stack::footprint(len)) + 1024;
}

}  // namespace fps::evaluation::detail
