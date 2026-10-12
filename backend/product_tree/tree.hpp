// The product trees: memory, transforms and tables, the lane tree and the standard tree.
// Design: notes.md. Coefficients are in Montgomery form (x 2^32 mod P) throughout.
//
// Lengths. A node covers leaves [lo, hi) and splits where the prefix sums of the leaves' degrees
// come closest to halves. Its length L is a power of two >= its degree (the wrap trick): its
// product p comes from its children's transforms of length L as p mod (x^L - 1); if deg p = L,
// p[L] is the product of the leading coefficients and p[0] is corrected by it. A node returns its
// transform at its parent's length (>= L), or its coefficients at the root. Nodes up to a base
// degree, and leaves, are multiplied out and transformed.
//
// Layouts.
// - Lanes: 8 polynomials side by side, word 8 j + l = coefficient j of polynomial l. A transform of
//   8 m words (ntt::Transform's leaf form) then holds polynomial l at the m roots w_v of X^m - 1
//   in word l of leaf v, so products are pointwise. The parent's length 2L: the lower half is the
//   node's own product transform; the upper half is ntt's forward_upper of p mod (X^L + 1) (8 L
//   words: the words' polynomial mod x^(8L) + 1 is each lane's mod X^L + 1). Per node: one
//   pointwise pass (which also writes the copy the inverse runs on), one inverse and one
//   forward_upper of length L.
// - Standard: one polynomial per node, leaf products mod (x^8 - w_v), the upper half as the
//   lanes'. Per node: leaf products (and the copy), one inverse and one forward_upper of length L.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "backend/ntt/memory.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/product_tree/kernels.hpp"
#include "backend/small/small.hpp"

namespace fps::product_tree::detail {

// LIFO allocation of word arrays from one span: each 64-byte aligned, with ntt::kPadding words
// after it. Past the span, blocks from the heap, freed by release().
class Stack {
public:
    struct Mark {
        std::size_t top, spilled;
    };

    Stack(u32* base, std::size_t words) : base_(base), size_(words) {}

    Mark mark() const { return {top_, spilled_.size()}; }
    void release(Mark m) {
        top_ = m.top;
        spilled_.resize(m.spilled);
    }

    static std::size_t footprint(std::size_t n) { return (n + ntt::kPadding + 15) / 16 * 16; }

    u32* take(std::size_t n) {
        const std::size_t need = footprint(n);
        if (top_ + need <= size_) {
            u32* p = base_ + top_;
            top_ += need;
            return p;
        }
        spilled_.push_back(std::make_unique<Line[]>(need / 16));
        return spilled_.back()[0].words;
    }

private:
    struct alignas(64) Line {
        u32 words[16];
    };
    u32* base_;
    std::size_t size_, top_ = 0;
    std::vector<std::unique_ptr<Line[]>> spilled_;
};

// This thread's transforms, one per length, kept across calls.
inline const ntt::Transform& transform(std::size_t words) {
    thread_local std::array<std::unique_ptr<ntt::Transform>, ntt::kMaxLog + 1> cache;
    const int lg = std::countr_zero(words);
    if (!cache[lg]) cache[lg] = std::make_unique<ntt::Transform>(lg);
    return *cache[lg];
}

inline void forward(u32* f, std::size_t words) { transform(words).forward(f); }
inline void inverse(u32* f, std::size_t words) { transform(words).inverse(f); }
// h: words / 2 words of p mod (x^(words/2) + 1) -> the upper half of p's transform of `words` words.
inline void forward_upper(u32* h, std::size_t words) { transform(words).forward_upper(h); }

// x - y and x + y mod P, canonical in and out.
constexpr u32 subtract(u32 x, u32 y) { return x >= y ? x - y : x + kP - y; }
constexpr u32 add(u32 x, u32 y) { return x >= kP - y ? x - (kP - y) : x + y; }

// w_v 2^32 mod P for the leaves v < count of the standard layout: w_v = (-1)^v r(v / 2), r(k) the
// product of ntt's roots[j] over the set bits j of k. This thread's, kept for the most leaves yet.
inline const u32* leaf_weights(std::size_t count) {
    thread_local std::vector<u32> weights;
    if (weights.size() < count) {
        const std::size_t half = std::max<std::size_t>(count / 2, 1);
        std::vector<u32> r(half);
        r[0] = 1;
        for (std::size_t bit = 1, j = 0; bit < half; bit *= 2, ++j)
            for (std::size_t i = 0; i < bit; ++i) r[bit + i] = multiply_mod(r[i], ntt::detail::kRoots[0][j]);
        weights.resize(2 * half);
        for (std::size_t v = 0; v < 2 * half; ++v) weights[v] = multiply_mod(v % 2 ? kP - r[v / 2] : r[v / 2], kR);
    }
    return weights.data();
}

// The leaf of a tree's leaves where node [lo, hi) splits, from prefix[k] = degree(0) + ... +
// degree(k - 1): the sums before and after it closest to halves.
inline std::size_t split(const u32* prefix, std::size_t lo, std::size_t hi) {
    const u32 target = prefix[lo] + (prefix[hi] - prefix[lo]) / 2;
    std::size_t mid = std::size_t(std::lower_bound(prefix + lo + 1, prefix + hi - 1, target) - prefix);  // in [lo + 1, hi - 1]
    if (mid > lo + 1 && prefix[mid] >= target && target - prefix[mid - 1] < prefix[mid] - target) --mid;
    return mid;
}

// Lanes: degrees and leading coefficients (Montgomery form) per lane.
struct LaneNode {
    alignas(32) u32 degree[8];
    alignas(32) u32 lead[8];
};

inline LaneNode combine(const LaneNode& x, const LaneNode& y) {
    LaneNode z;
    for (int l = 0; l < 8; l += Lanes::lanes) {
        Lanes::store(z.degree + l, Lanes::add(Lanes::load(x.degree + l), Lanes::load(y.degree + l)));
        Lanes::store(z.lead + l, montgomery_canonical<Lanes>(Lanes::load(x.lead + l), Lanes::load(y.lead + l)));
    }
    return z;
}

// The lanes' tree over slots of 8 factors. Leaves:
//   std::size_t count() const;                // slots, >= 1
//   u32 degree(std::size_t k) const;          // the largest degree in slot k, >= 1
//   LaneNode load(std::size_t k, u32* c) const;
//       // writes the degree(k) + 1 coefficients of slot k at c (8 words each, Montgomery form,
//       // zero past a lane's degree)
template <class Leaves>
class LaneTree {
public:
    struct Root {
        u32* coefficients;  // 8 (length + 1) words: lane l's at 8 j + l, zero past its degree
        LaneNode node;
        std::size_t length;
    };

    static std::size_t length(u32 degree) { return std::max<std::size_t>(8, std::bit_ceil(degree)); }

    // The prefix sums of the slots' degrees go in the stack.
    LaneTree(const Leaves& leaves, Stack& stack) : leaves_(leaves), stack_(stack), prefix_(stack.take(leaves.count() + 1)) {
        prefix_[0] = 0;
        for (std::size_t k = 0; k < leaves.count(); ++k) prefix_[k + 1] = prefix_[k] + leaves.degree(k);
    }

    // The product of all leaves. Its coefficients stay in the stack; what root() takes past them
    // is released.
    Root root() {
        const std::size_t count = leaves_.count();
        const u32 d = degree(0, count);
        const std::size_t len = length(d);
        if (count == 1 || d <= kLaneBase) {
            u32* const c = stack_.take(8 * (d + 1));
            return {c, coefficients(0, count, c), d};
        }
        u32* const a = stack_.take(8 * len + 8);
        const Stack::Mark mark = stack_.mark();
        u32* const b = stack_.take(8 * len);
        const std::size_t mid = split(prefix_, 0, count);
        const LaneNode left = build(0, mid, a, len);
        const LaneNode node = combine(left, build(mid, count, b, len));
        pointwise<Wide>(a, b, a, 8 * len);
        inverse(a, 8 * len);
        for (int l = 0; l < 8; ++l) {
            const u32 top = node.degree[l] == len ? node.lead[l] : 0;
            a[l] = a[l] >= top ? a[l] - top : a[l] + kP - top;
            a[8 * len + l] = top;
        }
        stack_.release(mark);
        return {a, node, len};
    }

private:
    u32 degree(std::size_t lo, std::size_t hi) const { return prefix_[hi] - prefix_[lo]; }

    // The degree(lo, hi) + 1 coefficients of node [lo, hi) at c: a leaf, or a node of degree <=
    // kLaneBase multiplied out in consecutive pairs, level by level.
    LaneNode coefficients(std::size_t lo, std::size_t hi, u32* c) const {
        if (hi - lo == 1) return leaves_.load(lo, c);
        struct Part {
            u32* c;
            std::size_t degree;
        };
        Part part[kLaneBase];
        alignas(64) u32 memory[2][8 * 2 * kLaneBase + 16];  // a level: degree + parts <= 2 kLaneBase coefficients
        std::size_t count = hi - lo;
        u32* to = memory[0];
        LaneNode node{};
        for (std::size_t k = 0; k < count; ++k) {
            const LaneNode n = leaves_.load(lo + k, to);
            node = k ? combine(node, n) : n;
            part[k] = {to, leaves_.degree(lo + k)};
            to += 8 * (part[k].degree + 1);
        }
        for (int level = 1; count > 1; ++level) {
            to = count == 2 ? c : memory[level & 1];
            std::size_t next = 0;
            for (std::size_t k = 0; k + 1 < count; k += 2) {
                const Part &x = part[k], &y = part[k + 1];
                multiply_lanes<Lanes>(x.c, x.degree, y.c, y.degree, to);
                part[next++] = {to, x.degree + y.degree};
                to += 8 * (x.degree + y.degree + 1);
            }
            if (count % 2) {  // the odd one moves up as it is
                const Part& x = part[count - 1];
                std::copy_n(x.c, 8 * (x.degree + 1), to);
                part[next++] = {to, x.degree};
            }
            count = next;
        }
        return node;
    }

    // out = the transform of node [lo, hi) of 8 out_length words (out_length >= its length).
    LaneNode build(std::size_t lo, std::size_t hi, u32* out, std::size_t out_length) {
        const u32 d = degree(lo, hi);
        if (hi - lo == 1 || d <= kLaneBase) {  // out_length > d: not the root
            const LaneNode node = coefficients(lo, hi, out);
            std::fill(out + 8 * (d + 1), out + 8 * out_length, 0u);
            forward(out, 8 * out_length);
            return node;
        }
        const std::size_t len = length(d), words = 8 * len;
        const Stack::Mark mark = stack_.mark();
        u32* const a = out_length >= 2 * len ? out + words : stack_.take(words);
        u32* const c = stack_.take(words + 8);  // the right child's transform, then p's coefficients
        const std::size_t mid = split(prefix_, lo, hi);
        const LaneNode left = build(lo, mid, a, len);
        const LaneNode node = combine(left, build(mid, hi, c, len));
        if (out_length == len) {
            pointwise<Wide>(a, c, out, words);
        } else {
            // a = out + words, the upper half: the forward_upper of p mod (X^L + 1). Longer
            // extensions (out_length > 2L) need p itself: then c keeps it.
            const bool longer = out_length > 2 * len;
            u32* const h = longer ? c : a;
            pointwise2<Wide>(a, c, out, h, words);
            inverse(h, words);
            // h = p mod (X^L - 1) = p + top (1 - X^L), top = p[L] or 0.
            alignas(32) u32 top[8];
            for (int l = 0; l < 8; ++l) top[l] = node.degree[l] == len ? node.lead[l] : 0;
            if (longer) {
                for (int l = 0; l < 8; ++l) c[l] = subtract(c[l], top[l]), c[words + l] = top[l];
                std::copy_n(c, words, a);
                for (int l = 0; l < 8; ++l) a[l] = subtract(a[l], top[l]);
            } else {
                for (int l = 0; l < 8; ++l) a[l] = subtract(a[l], add(top[l], top[l]));
            }
            forward_upper(a, 2 * words);
            for (std::size_t m = 2 * len; m < out_length; m *= 2) {  // p mod (X^m + 1) = p
                std::fill(std::copy_n(c, words + 8, out + 8 * m), out + 16 * m, 0u);
                forward_upper(out + 8 * m, 16 * m);
            }
        }
        stack_.release(mark);
        return node;
    }

    const Leaves& leaves_;
    Stack& stack_;
    u32* prefix_;  // prefix_[k] = degree(0) + ... + degree(k - 1)
};

// One polynomial per node. A leaf: degree >= 1, coefficient j < P at p[stride j], in Montgomery
// form or not (then stride 1).
struct Leaf {
    const u32* p;
    u32 degree;
    bool montgomery;
    std::size_t stride = 1;
};

struct Node {
    u32 degree, lead;  // lead in Montgomery form
};

class StandardTree {
public:
    static std::size_t length(u32 degree) { return std::max<std::size_t>(64, std::bit_ceil(degree)); }

    StandardTree(std::span<const Leaf> leaves, Stack& stack) : leaves_(leaves), stack_(stack), prefix_(stack.take(leaves.size() + 1)) {
        prefix_[0] = 0;
        for (std::size_t k = 0; k < leaves.size(); ++k) prefix_[k + 1] = prefix_[k] + leaves[k].degree;
        const std::size_t len = length(prefix_[leaves.size()]);
        weights_ = leaf_weights(len / 8);
    }

    // The degree + 1 coefficients of the product of all leaves, in the stack.
    u32* root() {
        const std::size_t count = leaves_.size();
        const u32 d = degree(0, count);
        if (count == 1 || d <= kStandardBase) {
            u32* const c = stack_.take(d + 1);
            coefficients(0, count, c);
            return c;
        }
        const std::size_t len = length(d);
        u32* const a = stack_.take(len + 1);
        u32* const b = stack_.take(len);
        const std::size_t mid = split(prefix_, 0, count);
        const Node left = build(0, mid, a, len);
        const Node right = build(mid, count, b, len);
        leaf_products<Lanes>(a, b, a, len / 8, weights_);
        inverse(a, len);
        const u32 top = d == len ? montgomery(left.lead, right.lead) : 0;
        a[0] = a[0] >= top ? a[0] - top : a[0] + kP - top;
        a[len] = top;
        return a;
    }

private:
    u32 degree(std::size_t lo, std::size_t hi) const { return prefix_[hi] - prefix_[lo]; }

    Node load(std::size_t k, u32* c) const {
        const Leaf& f = leaves_[k];
        if (f.stride != 1) {
            for (std::size_t j = 0; j <= f.degree; ++j) c[j] = f.p[f.stride * j];
        } else if (f.montgomery) {
            std::copy_n(f.p, f.degree + 1, c);
        } else {
            scale<Wide>(f.p, kR2, c, f.degree + 1);
        }
        return {f.degree, c[f.degree]};
    }

    // The degree(lo, hi) + 1 coefficients of node [lo, hi) at c: a leaf, or a node of degree <=
    // kStandardBase multiplied out in consecutive pairs by small::multiply, level by level. Each
    // product of Montgomery forms carries one more 2^32; one pass at the end takes them out.
    Node coefficients(std::size_t lo, std::size_t hi, u32* c) const {
        if (hi - lo == 1) return load(lo, c);
        struct Part {
            u32* c;
            std::size_t degree;
        };
        Part part[kStandardBase];
        u32 memory[2][2 * kStandardBase + 16];
        std::size_t count = hi - lo;
        u32* to = memory[0];
        u32 lead = kR;
        for (std::size_t k = 0; k < count; ++k) {
            const Node n = load(lo + k, to);
            lead = montgomery(lead, n.lead);
            part[k] = {to, n.degree};
            to += part[k].degree + 1;
        }
        for (int level = 1; count > 1; ++level) {
            to = count == 2 ? c : memory[level & 1];
            std::size_t next = 0;
            for (std::size_t k = 0; k + 1 < count; k += 2) {
                const Part &x = part[k], &y = part[k + 1];
                small::multiply<kP>({x.c, x.degree + 1}, {y.c, y.degree + 1}, {to, x.degree + y.degree + 1});
                part[next++] = {to, x.degree + y.degree};
                to += x.degree + y.degree + 1;
            }
            if (count % 2) {
                const Part& x = part[count - 1];
                std::copy_n(x.c, x.degree + 1, to);
                part[next++] = {to, x.degree};
            }
            count = next;
        }
        const u32 d = degree(lo, hi);
        scale<Wide>(c, power(kRInverse, hi - lo - 2), c, d + 1);  // 2^(32 (k - 1)) too many, k leaves
        return {d, lead};
    }

    // out = the transform of node [lo, hi) of out_length words (out_length >= its length).
    Node build(std::size_t lo, std::size_t hi, u32* out, std::size_t out_length) {
        const u32 d = degree(lo, hi);
        if (hi - lo == 1 || d <= kStandardBase) {
            const Node node = coefficients(lo, hi, out);
            std::fill(out + d + 1, out + out_length, 0u);
            forward(out, out_length);
            return node;
        }
        const std::size_t len = length(d);
        const Stack::Mark mark = stack_.mark();
        u32* const a = out_length >= 2 * len ? out + len : stack_.take(len);
        u32* const c = stack_.take(len + 8);  // the right child's transform, then p's coefficients
        const std::size_t mid = split(prefix_, lo, hi);
        const Node left = build(lo, mid, a, len);
        const Node right = build(mid, hi, c, len);
        const Node node{d, montgomery(left.lead, right.lead)};
        if (out_length == len) {
            leaf_products<Lanes>(a, c, out, len / 8, weights_);
        } else {
            // As the lanes' (LaneTree::build): a = out + len gets the forward_upper of p mod
            // (x^L + 1); c keeps p for longer extensions.
            const bool longer = out_length > 2 * len;
            u32* const h = longer ? c : a;
            leaf_products<Lanes>(a, c, out, len / 8, weights_, h);
            inverse(h, len);
            const u32 top = d == len ? node.lead : 0;
            if (longer) {
                c[0] = subtract(c[0], top), c[len] = top;
                std::copy_n(c, len, a);
                a[0] = subtract(a[0], top);
            } else {
                a[0] = subtract(a[0], add(top, top));
            }
            forward_upper(a, 2 * len);
            for (std::size_t m = 2 * len; m < out_length; m *= 2) {  // p mod (x^m + 1) = p
                std::fill(std::copy_n(c, len + 1, out + m), out + 2 * m, 0u);
                forward_upper(out + m, 2 * m);
            }
        }
        stack_.release(mark);
        return node;
    }

    std::span<const Leaf> leaves_;
    Stack& stack_;
    u32* prefix_;
    const u32* weights_;
};

}  // namespace fps::product_tree::detail
