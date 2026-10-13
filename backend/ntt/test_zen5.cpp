// Tests for the kernels variant avx512 runs through intrinsics.hpp's Generated<Zmm> hooks
// (avx512.hpp; on the zen5 target the asm of zen5_kernels.hpp, issue #152), against the
// intrinsics on simd::Avx512, word for word, on the inputs' whole lazy ranges (random words, every
// word at the top of the range, zeros, mixed edges), with the words around the outputs untouched:
//   forward groups k = 0 (inputs < P, < 2P, < 4P) and k > 0 (< 2P, < 4P), every column count
//   from 2 to a whole group, quarters of 32 to 2^15 words;
//   inverse groups k = 0 and k > 0, quarters of 32 to 2^15 words;
//   forward levels of groups of two columns (quarters of 32 words), 1 to 64 groups from group 0
//   or others (runs of 4 from a multiple of 4 in asm, the rest intrinsics);
//   the bottom level forward and inverse, 2 to 64 groups (statements of 4 and 2 pairs, a last
//   pair by the intrinsics) from even first groups.
// On other targets (avx512) the hooks without asm run the intrinsics, and the checks hold
// trivially. The whole ops of the variant run in test.cpp and the other test files.
#include "backend/ntt/test_util.hpp"

#include <cstring>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)

namespace {

namespace in = fps::ntt::detail::intrinsics;
using Zmm = fps::ntt::detail::avx512::Zmm;
using fps::ntt::detail::kStaticTables;

const u32* const kR = kStaticTables.words[0];
const u32* const kIr = kStaticTables.words[1];

// count words < bound (kind 0 random, 1 all bound - 1, 2 zeros, 3 mixed edges), 64-byte aligned,
// then 64 guard words.
struct Block {
    explicit Block(std::size_t count) : data(count / 16 + 4) {}
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
};

bool same(Block& x, Block& y) { return std::memcmp(x.begin(), y.begin(), 64 * x.data.size()) == 0; }

template <int kIn>
void test_forward_groups() {
    using Old = in::Butterflies<fps::simd::Avx512>;
    using New = in::Butterflies<Zmm>;
    for (const std::size_t q : {32u, 128u, 512u, 2048u, 32768u}) {
        for (const std::size_t words : {q, std::size_t(32), std::size_t(64), std::size_t(96), q / 2}) {
            if (words > q || words % 32) continue;
            for (const std::size_t k : {0u, 1u, 2u, 3u, 7u, 100u, 2047u}) {
                for (int kind = 0; kind < 4; ++kind) {
                    Block x(4 * q), y(4 * q);
                    x.fill(kIn * P, kind);
                    std::memcpy(y.begin(), x.begin(), 64 * x.data.size());
                    Old::forward_group_part<kIn, 4>(x.begin(), x.begin(), q, words, k, kR);
                    New::forward_group_part<kIn, 4>(y.begin(), y.begin(), q, words, k, kR);
                    expect(same(x, y), kIn == 1 ? "forward group, in < P" : kIn == 2 ? "forward group, in < 2P"
                                                                                       : "forward group, in < 4P",
                           (q * 10000 + words) * 10000 + k);
                }
            }
        }
    }
}

void test_inverse_groups() {
    using Old = in::Butterflies<fps::simd::Avx512>;
    using New = in::Butterflies<Zmm>;
    for (const std::size_t q : {32u, 128u, 512u, 2048u, 32768u}) {
        for (const std::size_t k : {0u, 1u, 2u, 3u, 7u, 100u, 2047u}) {
            for (int kind = 0; kind < 4; ++kind) {
                Block x(4 * q), y(4 * q);
                x.fill(2 * P, kind);
                std::memcpy(y.begin(), x.begin(), 64 * x.data.size());
                Old::inverse_group(x.begin(), q, k, kIr);
                New::inverse_group(y.begin(), q, k, kIr);
                expect(same(x, y), "inverse group", q * 10000 + k);
            }
        }
    }
}

// The forward levels of groups of two columns (quarters of 32 words): forward_groups.
void test_small_groups() {
    using Old = in::Butterflies<fps::simd::Avx512>;
    using New = in::Butterflies<Zmm>;
    for (const std::size_t count : {1u, 3u, 4u, 8u, 16u, 64u}) {
        for (const std::size_t k : {0u, 1u, 4u, 8u, 12u, 64u, 1000u, 1980u}) {  // the tables hold 4096 entries
            for (int kind = 0; kind < 4; ++kind) {
                Block x(128 * count), y(128 * count);
                x.fill(4 * P, kind);
                std::memcpy(y.begin(), x.begin(), 64 * x.data.size());
                Old::forward_groups(x.begin(), 32, k, count, kR), New::forward_groups(y.begin(), 32, k, count, kR);
                expect(same(x, y), "forward groups of 2 columns", (count * 10000 + k) * 10 + kind);
            }
        }
    }
}

void test_bottoms() {
    using Old = in::Ntt<fps::simd::Avx512>;
    using New = in::Ntt<Zmm>;
    for (const std::size_t count : {2u, 4u, 6u, 8u, 10u, 12u, 14u, 16u, 30u, 64u}) {
        for (const std::size_t first : {0u, 2u, 6u, 64u, 1000u}) {
            for (int kind = 0; kind < 4; ++kind) {
                for (const bool inverse : {false, true}) {
                    Block x(32 * count), y(32 * count);
                    x.fill(inverse ? 2 * P : 4 * P, kind);
                    std::memcpy(y.begin(), x.begin(), 64 * x.data.size());
                    if (inverse) Old::bottom_inverse(x.begin(), count, first, kIr), New::bottom_inverse(y.begin(), count, first, kIr);
                    else Old::bottom_forward(x.begin(), count, first, kR), New::bottom_forward(y.begin(), count, first, kR);
                    expect(same(x, y), inverse ? "bottom inverse" : "bottom forward", (count * 10000 + first) * 10 + kind);
                }
            }
        }
    }
}

}  // namespace

int main() {
    test_forward_groups<1>();
    test_forward_groups<2>();
    test_forward_groups<4>();
    test_inverse_groups();
    test_small_groups();
    test_bottoms();
    return finish();
}

#else

int main() {
    std::printf("PASS (no AVX-512)\n");
    return 0;
}

#endif
