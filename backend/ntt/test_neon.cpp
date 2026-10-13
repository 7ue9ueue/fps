// Tests for backend/ntt: the neon variant (neon.hpp, issue #151). Its generated asm kernels against
// the intrinsics they replace (Bottom<Neon>), with inputs at the edges of their ranges; then the
// variant's transforms, products and squares at every length against test_util.hpp's references
// (test_fused.cpp covers its fused ops as the tuned variant). Off arm (or without SVE2: the variant
// is then intrinsics<Neon>) only what runs.
#include "backend/ntt/test_util.hpp"

#include <algorithm>
#include <cstdio>
#include <string_view>
#include <vector>

#if defined(__ARM_NEON) && defined(__aarch64__)
namespace {

namespace intrinsics = fps::ntt::detail::intrinsics;
using Plain = intrinsics::Ntt<fps::simd::Neon>;
using Asm = intrinsics::Ntt<fps::ntt::detail::neon::Vec>;

// Words of a forward level's output, < 4P: random, or an edge (0, P - 1, P, 2P - 1, 2P, 4P - 1).
u32 forward_word(int kind) {
    static constexpr u32 kEdges[] = {0, P - 1, P, 2 * P - 1, 2 * P, 4 * P - 1};
    if (kind == 0) return random_below(4ull * P);
    if (kind == 1) return kEdges[random_below(std::size(kEdges))];
    return kind == 2 ? 4 * P - 1 : 0;
}

// A bottom stage on groups [first, first + count) of a (and b), every pair count and group offset
// the transforms use up to 2^16 (count = 4^j, first a multiple of count), against Bottom<Neon>:
// out < 2P and equal mod P. b: inputs < 4P (product), unread (square), canonical leaves (stored).
enum class Kind { product, square, stored };
void test_bottom(Kind kind, const char* name) {
    const int lg = 16;
    const std::size_t leaves = (std::size_t(1) << lg) / 8;
    Words r(leaves), ir(leaves);  // leaves / 2 entries, each a value and a quotient
    Plain::build_table(r.begin(), leaves / 2, false);
    Plain::build_table(ir.begin(), leaves / 2, true);
    for (std::size_t count : {2, 4, 16, 64}) {
        for (int trial = 0; trial < 24; ++trial) {
            const std::size_t first = trial < 2 ? trial * count : count * random_below(leaves / 4 / count);
            const int pick = trial % 4;
            Words a(32 * count), b(32 * count), expected(32 * count);
            for (std::size_t i = 0; i < 32 * count; ++i) {
                a.begin()[i] = expected.begin()[i] = forward_word(pick == 3 ? int(random_below(4)) : pick);
                b.begin()[i] = kind == Kind::stored ? (pick == 1 ? P - 1 : random_below(P))
                                                    : forward_word(pick == 3 ? int(random_below(4)) : (pick + 1) % 3);
            }
            const u32 *rt = r.begin(), *irt = ir.begin();
            if (kind == Kind::product) {
                Plain::bottom_multiply<false>(expected.begin(), b.begin(), count, first, rt, irt);
                Asm::bottom_multiply<false>(a.begin(), b.begin(), count, first, rt, irt);
            } else if (kind == Kind::square) {
                Plain::bottom_multiply<true>(expected.begin(), b.begin(), count, first, rt, irt);
                Asm::bottom_multiply<true>(a.begin(), b.begin(), count, first, rt, irt);
            } else {
                Plain::bottom_multiply_stored(expected.begin(), b.begin(), count, first, rt, irt);
                Asm::bottom_multiply_stored(a.begin(), b.begin(), count, first, rt, irt);
            }
            bool ok = true;
            for (std::size_t i = 0; i < 32 * count; ++i)
                ok &= a.begin()[i] < 2 * P && a.begin()[i] % P == expected.begin()[i] % P;
            expect(ok, name, count * 1000000 + first);
        }
    }
}

// Bottom::inverse_terms: sums of 1 to 4 terms, product terms (p: a b) and plain ones (x: a times
// 2^-32) in any order, against Bottom<Neon>: out < 2P, equal mod P. Canonical leaves: random, all
// P - 1, or 0 and P - 1. The asm runs one term a b, or up to 4 products and one plain term in
// passes; the other lists (no product, two plain terms) fall back to the intrinsics.
void test_terms() {
    const std::size_t leaves = std::size_t(1) << 13;
    Words r(leaves), ir(leaves);
    Plain::build_table(r.begin(), leaves / 2, false);
    Plain::build_table(ir.begin(), leaves / 2, true);
    static constexpr const char* kLists[] = {"p", "px", "xp", "pp", "ppx", "xpp", "pxp", "ppp", "pppp", "xppp", "pppx", "x", "xx", "pxx"};
    for (std::size_t list = 0; list < std::size(kLists); ++list) {
        const std::string_view spec = kLists[list];
        const int n = int(spec.size());
        for (std::size_t count : {2, 4, 16, 64}) {
            for (int trial = 0; trial < 9; ++trial) {
                const std::size_t first = trial < 2 ? trial * count : count * random_below(leaves / 4 / count);
                const int pick = trial % 3;
                const std::size_t words = 32 * (first + count);
                std::vector<Words> a, b;
                fps::ntt::detail::LeafTerm terms[fps::ntt::detail::kMaxTerms];
                for (int i = 0; i < n; ++i) {
                    a.emplace_back(words), b.emplace_back(words);
                    for (std::size_t w = 0; w < words; ++w) {
                        const auto word = [&] { return pick == 0 ? random_below(P) : pick == 1 ? P - 1 : u32(random_below(2) * (P - 1)); };
                        a.back().begin()[w] = word(), b.back().begin()[w] = word();
                    }
                }
                for (int i = 0; i < n; ++i) terms[i] = {a[i].begin(), spec[i] == 'p' ? b[i].begin() : nullptr};
                Words out(32 * count), expected(32 * count);
                Plain::bottom_inverse_terms(expected.begin(), terms, n, count, first, r.begin(), ir.begin());
                Asm::bottom_inverse_terms(out.begin(), terms, n, count, first, r.begin(), ir.begin());
                bool ok = true;
                for (std::size_t i = 0; i < 32 * count; ++i)
                    ok &= out.begin()[i] < 2 * P && out.begin()[i] % P == expected.begin()[i] % P;
                expect(ok, "neon: inverse_terms bottom against Bottom<Neon>", list * 10000000 + count * 10000 + first);
            }
        }
    }
}

// The group loops: group k on quarters of q words, against Butterflies<Neon>: forward in < 4P in
// place, or in < 2P from other memory or in place, out < 4P; inverse in < 2P, out < 2P; equal
// mod P. Then the bottom level of transforms (Bottom::forward, canonical out; Bottom::inverse).
enum class Op { forward, forward2, forward2_from, inverse };
void test_groups() {
    const std::size_t entries = 4096;
    Words r(2 * entries), ir(2 * entries);
    Plain::build_table(r.begin(), entries, false);
    Plain::build_table(ir.begin(), entries, true);
    for (const Op op : {Op::forward, Op::forward2, Op::forward2_from, Op::inverse}) {
        const u32 in = op == Op::forward ? 4 : 2, out = op == Op::inverse ? 2 : 4;
        for (const std::size_t q : {32, 48, 64, 128, 512, 2048}) {
            for (int trial = 0; trial < 12; ++trial) {
                const std::size_t k = trial < 2 ? trial : 1 + random_below(entries / 2 - 1);
                const int pick = trial % 4;
                Words f(4 * q), expected(4 * q), from(4 * q);
                for (std::size_t i = 0; i < 4 * q; ++i) {
                    const u32 w = forward_word(pick == 3 ? int(random_below(4)) : pick);
                    from.begin()[i] = f.begin()[i] = expected.begin()[i] = in == 4 ? w : w % (2 * P);
                }
                if (op == Op::forward) {
                    Plain::forward_group(expected.begin(), q, k, r.begin());
                    Asm::forward_group(f.begin(), q, k, r.begin());
                } else if (op == Op::forward2) {
                    Plain::forward_group<2>(expected.begin(), q, k, r.begin());
                    Asm::forward_group<2>(f.begin(), q, k, r.begin());
                } else if (op == Op::forward2_from) {
                    Plain::forward_group<2>(from.begin(), expected.begin(), q, k, r.begin());
                    for (std::size_t i = 0; i < 4 * q; ++i) f.begin()[i] = u32(rng());
                    Asm::forward_group<2>(from.begin(), f.begin(), q, k, r.begin());
                } else {
                    Plain::inverse_group(expected.begin(), q, k, ir.begin());
                    Asm::inverse_group(f.begin(), q, k, ir.begin());
                }
                bool ok = true;
                for (std::size_t i = 0; i < 4 * q; ++i)
                    ok &= f.begin()[i] < out * P && f.begin()[i] % P == expected.begin()[i] % P;
                expect(ok, "neon: group loop against Butterflies<Neon>", int(op) * 10000000 + q * 10000 + k);
            }
        }
    }
    for (const bool forward : {true, false}) {
        for (const std::size_t count : {2, 4, 16, 64}) {
            for (int trial = 0; trial < 12; ++trial) {
                const std::size_t first = trial < 2 ? trial * count : count * random_below(entries / 2 / count);
                const int pick = trial % 4;
                Words f(32 * count), expected(32 * count);
                for (std::size_t i = 0; i < 32 * count; ++i) {
                    const u32 w = forward_word(pick == 3 ? int(random_below(4)) : pick);
                    f.begin()[i] = expected.begin()[i] = forward ? w : w % (2 * P);
                }
                if (forward) {
                    Plain::bottom_forward(expected.begin(), count, first, r.begin());
                    Asm::bottom_forward(f.begin(), count, first, r.begin());
                } else {
                    Plain::bottom_inverse(expected.begin(), count, first, ir.begin());
                    Asm::bottom_inverse(f.begin(), count, first, ir.begin());
                }
                bool ok = true;
                for (std::size_t i = 0; i < 32 * count; ++i)
                    ok &= forward ? f.begin()[i] == expected.begin()[i]
                                  : f.begin()[i] < 2 * P && f.begin()[i] % P == expected.begin()[i] % P;
                expect(ok, forward ? "neon: transform bottom forward" : "neon: transform bottom inverse", count * 1000000 + first);
            }
        }
    }
}

// Round 2's kernels against the intrinsics, on quarters of q words: the forward top group on
// canonical inputs (in place, from other memory), the scaled inverse groups k < 4 (scaled blocks of
// random scales), the inverse top group and radix-2 level with and without a scale (in place, and
// from other memory into words [0, w) of the quarters or halves, w a multiple of 8 or 4). Canonical
// outputs equal, others mod P.
void test_tops() {
    const std::size_t entries = 512;
    Words r(2 * entries), ir(2 * entries);
    Plain::build_table(r.begin(), entries, false);
    Plain::build_table(ir.begin(), entries, true);
    const auto fill = [](Words& w, std::size_t n, u32 bound, int pick) {
        for (std::size_t i = 0; i < n; ++i) w.begin()[i] = forward_word(pick == 3 ? int(random_below(4)) : pick) % bound;
    };
    const auto same = [](Words& a, Words& b, std::size_t n, u32 bound) {
        bool ok = true;
        for (std::size_t i = 0; i < n; ++i) ok &= a.begin()[i] < bound && a.begin()[i] % P == b.begin()[i] % P;
        return ok;
    };
    for (const std::size_t q : {16, 32, 48, 64, 128, 512}) {
        for (int trial = 0; trial < 16; ++trial) {
            const int pick = trial % 4;
            Words f(4 * q), expected(4 * q), from(4 * q);
            // Forward top group, canonical in.
            fill(f, 4 * q, P, pick);
            for (std::size_t i = 0; i < 4 * q; ++i) expected.begin()[i] = from.begin()[i] = f.begin()[i];
            Plain::forward_group<1>(expected.begin(), q, 0, r.begin());
            Asm::forward_group<1>(f.begin(), q, 0, r.begin());
            expect(same(f, expected, 4 * q, 4 * P), "neon: forward top group", q * 100 + trial);
            for (std::size_t i = 0; i < 4 * q; ++i) f.begin()[i] = u32(rng());
            Asm::forward_group<1>(from.begin(), f.begin(), q, 0, r.begin());
            expect(same(f, expected, 4 * q, 4 * P), "neon: forward top group from", q * 100 + trial);
            // Columns [0, w) of a group from other memory (a product's first level), in < 2P.
            for (const std::size_t k : {std::size_t(0), std::size_t(1 + random_below(entries / 2 - 1))}) {
                const std::size_t w = 8 * (1 + random_below(q / 8));
                Words to(4 * q), want(4 * q);
                for (std::size_t i = 0; i < 4 * q; ++i) to.begin()[i] = want.begin()[i] = u32(rng());
                Plain::forward_group_part<2, 4>(from.begin(), want.begin(), q, w, k, r.begin());
                Asm::forward_group_part<2, 4>(from.begin(), to.begin(), q, w, k, r.begin());
                bool ok = true;
                for (std::size_t i = 0; i < 4 * q; ++i)
                    ok &= i % q < w ? to.begin()[i] < 4 * P && to.begin()[i] % P == want.begin()[i] % P : to.begin()[i] == want.begin()[i];
                expect(ok, "neon: forward group columns from other memory", q * 100000 + w * 100 + k);
            }
            // Scaled inverse groups, in < 2P.
            const u32 s = trial == 0 ? 1 : random_below(P);
            const auto block = intrinsics::scaled_block(s);
            for (std::size_t k = 0; k < 4; ++k) {
                fill(f, 4 * q, 2 * P, pick);
                for (std::size_t i = 0; i < 4 * q; ++i) expected.begin()[i] = f.begin()[i];
                Plain::inverse_group_scaled(expected.begin(), q, k, ir.begin(), block.words);
                Asm::inverse_group_scaled(f.begin(), q, k, ir.begin(), block.words);
                expect(same(f, expected, 4 * q, 2 * P), "neon: scaled inverse group", q * 1000 + trial * 10 + k);
            }
            // Inverse top group, s and 0.
            for (const u32 scale : {s, 0u}) {
                fill(f, 4 * q, 2 * P, pick);
                for (std::size_t i = 0; i < 4 * q; ++i) expected.begin()[i] = from.begin()[i] = f.begin()[i];
                Plain::inverse_radix4(expected.begin(), q, ir.begin(), scale);
                Asm::inverse_radix4(f.begin(), q, ir.begin(), scale);
                expect(std::equal(f.begin(), f.begin() + 4 * q, expected.begin()), "neon: inverse top group", q * 100 + trial);
                const std::size_t w = 8 * (1 + random_below(q / 8));
                Words to(4 * q), want(4 * q);
                for (std::size_t i = 0; i < 4 * q; ++i) to.begin()[i] = want.begin()[i] = u32(rng());
                Plain::inverse_radix4_part<4>(from.begin(), want.begin(), q, w, ir.begin(), scale);
                Asm::inverse_radix4_part<4>(from.begin(), to.begin(), q, w, ir.begin(), scale);
                expect(std::equal(to.begin(), to.begin() + 4 * q, want.begin()), "neon: inverse top group to other memory",
                       q * 100000 + w * 100 + trial);
                for (const bool upper : {false, true}) {  // outputs 0, 1 or 2, 3 only (cut products, upper halves)
                    for (std::size_t i = 0; i < 4 * q; ++i) to.begin()[i] = want.begin()[i] = u32(rng());
                    if (upper) {
                        Plain::inverse_radix4_part<2, true>(from.begin(), want.begin(), q, w, ir.begin(), scale);
                        Asm::inverse_radix4_part<2, true>(from.begin(), to.begin(), q, w, ir.begin(), scale);
                    } else {
                        Plain::inverse_radix4_part<2>(from.begin(), want.begin(), q, w, ir.begin(), scale);
                        Asm::inverse_radix4_part<2>(from.begin(), to.begin(), q, w, ir.begin(), scale);
                    }
                    expect(std::equal(to.begin(), to.begin() + 4 * q, want.begin()), "neon: inverse top group, half out",
                           q * 100000 + w * 100 + trial * 2 + upper);
                }
                // The inverse radix-2 level on halves of 2q words.
                const std::size_t half = 2 * q, w2 = 4 * (1 + random_below(half / 4));
                for (std::size_t i = 0; i < 4 * q; ++i) expected.begin()[i] = f.begin()[i] = from.begin()[i];
                Plain::inverse_radix2(expected.begin(), half, scale);
                Asm::inverse_radix2(f.begin(), half, scale);
                expect(std::equal(f.begin(), f.begin() + 4 * q, expected.begin()), "neon: inverse radix-2 level", q * 100 + trial);
                for (std::size_t i = 0; i < 4 * q; ++i) to.begin()[i] = want.begin()[i] = u32(rng());
                Plain::inverse_radix2_part<2>(from.begin(), want.begin(), half, w2, scale);
                Asm::inverse_radix2_part<2>(from.begin(), to.begin(), half, w2, scale);
                expect(std::equal(to.begin(), to.begin() + 4 * q, want.begin()), "neon: inverse radix-2 level to other memory",
                       q * 100000 + w2 * 100 + trial);
                for (const bool upper : {false, true}) {  // the sums or the differences only
                    for (std::size_t i = 0; i < 4 * q; ++i) to.begin()[i] = want.begin()[i] = u32(rng());
                    if (upper) {
                        Plain::inverse_radix2_part<1, true>(from.begin(), want.begin(), half, w2, scale);
                        Asm::inverse_radix2_part<1, true>(from.begin(), to.begin(), half, w2, scale);
                    } else {
                        Plain::inverse_radix2_part<1>(from.begin(), want.begin(), half, w2, scale);
                        Asm::inverse_radix2_part<1>(from.begin(), to.begin(), half, w2, scale);
                    }
                    expect(std::equal(to.begin(), to.begin() + 4 * q, want.begin()), "neon: inverse radix-2 level, half out",
                           q * 100000 + w2 * 100 + trial * 2 + upper);
                }
            }
        }
    }
}

// square<V>(f, out) against schoolbook (to 40 words, and short transforms) or the product by
// evaluation, out of 2n - 1 words, at every length.
template <class V>
void test_squares(const char* name, int max_lg) {
    for (std::size_t n = 1; n <= 40; ++n) {
        const auto f = random_poly(n, n % 4);
        std::vector<u32> out(2 * n - 1);
        square<V>(f, out);
        expect(out == schoolbook(f, f), name, n);
    }
    for (int lg = kMinLog; lg <= max_lg; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        for (const std::size_t n : {len / 2, len / 4 + 1}) {
            const auto f = random_poly(n, lg % 2);
            std::vector<u32> out(2 * n - 1);
            square<V>(f, out);
            if (lg <= 11) {
                expect(out == schoolbook(f, f), name, n);
                continue;
            }
            bool ok = true;
            for (int i = 0; i < 2; ++i) {
                const u32 x = random_below(P), fx = evaluate(f, x);
                ok &= evaluate(out, x) == mul(fx, fx);
            }
            expect(ok, name, n);
        }
    }
}

}  // namespace
#endif

int main() {
#if defined(__ARM_NEON) && defined(__aarch64__)
#ifdef FPS_NTT_NEON_KERNELS
    test_bottom(Kind::product, "neon: product bottom against Bottom<Neon>");
    test_bottom(Kind::square, "neon: square bottom against Bottom<Neon>");
    test_bottom(Kind::stored, "neon: stored bottom against Bottom<Neon>");
    test_terms();
    test_groups();
    test_tops();
#else
    std::printf("no SVE2: neon is intrinsics<Neon>\n");
#endif
    test_transforms<variant::neon>("neon", kVectorMaxLog);
    test_products<variant::neon>("neon", kVectorMaxLog);
    test_multiply_spans<variant::neon>("neon", kVectorMaxLog);
    test_squares<variant::neon>("neon square", kVectorMaxLog);
#else
    std::printf("not arm: no neon variant\n");
#endif
    return finish();
}
