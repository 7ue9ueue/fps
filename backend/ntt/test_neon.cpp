// Tests for backend/ntt: the neon variant (neon.hpp, issue #151). Its generated asm kernels against
// the intrinsics they replace (Bottom<Neon>), with inputs at the edges of their ranges; then the
// variant's transforms, products and squares at every length against test_util.hpp's references
// (test_fused.cpp covers its fused ops as the tuned variant). Off arm (or without SVE2: the variant
// is then intrinsics<Neon>) only what runs.
#include "backend/ntt/test_util.hpp"

#include <cstdio>

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
    test_groups();
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
