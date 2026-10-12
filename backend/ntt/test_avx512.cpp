// Tests for the generated zmm kernels (avx512_kernels.hpp, variant avx512, issue #150) against the
// intrinsics they replace, on the inputs' whole lazy ranges: random words, every word at the top
// of the range, zeros. The whole ops of the variant run in test.cpp, test_spans.cpp,
// test_leaf.cpp, test_fused.cpp and test_upper.cpp (for_each_variant and their lists).
//   forward groups k > 0 (inputs < 2P and < 4P), every column count the asm takes, quarters from
//   64 words to past the prefetch threshold: the same words as Butterflies<Avx512>, the other
//   words untouched;
//   product bottoms (multiply, square, a leaf form stored): the same residues as WideBottom's,
//   < 2P, for 2 to 64 groups from even first groups; the leaf buffers' stores stay in bounds.
#include "backend/ntt/test_util.hpp"

#include <cstring>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)

namespace {

namespace in = fps::ntt::detail::intrinsics;
using Zmm = fps::ntt::detail::avx512::Zmm;
using Old = in::Butterflies<fps::simd::Avx512>;
using New = in::Butterflies<Zmm>;
using fps::ntt::detail::kStaticTables;

const u32* const kR = kStaticTables.words[0];
const u32* const kIr = kStaticTables.words[1];

// count words < bound (kind 0 random, 1 all bound - 1, 2 zeros, 3 mixed edges), 64-byte aligned,
// then 64 guard words.
struct Block {
    explicit Block(std::size_t count) : data(count / 16 + 8), count(count) {}
    u32* begin() { return data.data()->w; }
    void fill(u32 bound, int kind) {
        for (std::size_t i = 0; i < 16 * data.size(); ++i) {
            const u32 edge = i % 3 == 0 ? bound - 1 : i % 3 == 1 ? 0 : random_below(bound);
            begin()[i] = kind == 0 ? random_below(bound) : kind == 1 ? bound - 1 : kind == 2 ? 0 : edge;
        }
    }
    struct alignas(64) Line {
        u32 w[16];
    };
    std::vector<Line> data;
    std::size_t count;
};

bool same(Block& x, Block& y) { return std::memcmp(x.begin(), y.begin(), 64 * x.data.size()) == 0; }

template <int kIn>
void test_forward_groups() {
    for (const std::size_t q : {64u, 128u, 256u, 1024u, 32768u}) {
        for (const std::size_t words : {q, std::size_t(64), q / 2}) {
            if (words % 64 || words > q) continue;
            for (const std::size_t k : {1u, 2u, 3u, 5u, 6u, 7u, 100u, 2047u}) {
                for (int kind = 0; kind < 4; ++kind) {
                    Block x(4 * q), y(4 * q);
                    x.fill(kIn * P, kind);
                    std::memcpy(y.begin(), x.begin(), 64 * x.data.size());
                    Old::forward_group_part<kIn, 4>(x.begin(), x.begin(), q, words, k, kR);
                    New::forward_group_part<kIn, 4>(y.begin(), y.begin(), q, words, k, kR);
                    expect(same(x, y), kIn == 2 ? "forward group, in < 2P" : "forward group, in < 4P", q * 1000 + k);
                }
            }
        }
    }
}

// in::Generated<Zmm>'s product bottoms against WideBottom's (op 0 multiply, 1 square, 2 stored).
void test_bottoms() {
    using W = in::WideBottom<fps::simd::Avx512>;
    using G = in::Generated<Zmm>;
    for (int op = 0; op < 3; ++op) {
        for (const std::size_t count : {2u, 4u, 6u, 8u, 32u, 64u}) {
            for (const std::size_t first : {0u, 2u, 64u, 1000u}) {
                for (int kind = 0; kind < 4; ++kind) {
                    const std::size_t words = 32 * count;
                    Block a(words), b(words), a2(words);
                    a.fill(4 * P, kind);
                    b.fill(op == 2 ? P : 4 * P, (kind + 1) % 4);
                    std::memcpy(a2.begin(), a.begin(), 64 * a.data.size());
                    if (op == 0) {
                        W::multiply<false>(a.begin(), b.begin(), count, first, kR, kIr);
                        G::multiply<false>(a2.begin(), b.begin(), count, first, kR, kIr);
                    } else if (op == 1) {
                        W::multiply<true>(a.begin(), a.begin(), count, first, kR, kIr);
                        G::multiply<true>(a2.begin(), a2.begin(), count, first, kR, kIr);
                    } else {
                        W::multiply_stored(a.begin(), b.begin(), count, first, kR, kIr);
                        G::multiply_stored(a2.begin(), b.begin(), count, first, kR, kIr);
                    }
                    bool ok = true;
                    for (std::size_t i = 0; i < words; ++i)
                        ok &= a2.begin()[i] < 2 * P && a2.begin()[i] % P == a.begin()[i] % P;
                    ok &= std::memcmp(a.begin() + words, a2.begin() + words, 64 * a.data.size() - 4 * words) == 0;
                    const char* name[] = {"bottom multiply", "bottom square", "bottom stored"};
                    expect(ok, name[op], count * 10000 + first * 10 + std::size_t(kind));
                }
            }
        }
    }
}

}  // namespace

int main() {
    test_forward_groups<2>();
    test_forward_groups<4>();
    test_bottoms();
    return finish();
}

#else

int main() {
    std::printf("PASS (no AVX-512)\n");
    return 0;
}

#endif
