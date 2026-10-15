// Tests of the values form's split and join on 16 lanes (intrinsics::WideValues, issue #157): the
// same words as the scalar Values for every first block and count (odd ones, which take a lone block
// on 8 lanes), with the value rows of 2^6 to 2^14; join undoes split, in place and apart; and
// Transform's values-form forward, forward_upper and inverse (every variant this machine runs, and
// the tuned one) against the leaf form and the scalar split. On targets without AVX-512 only the
// Transform checks run (ValuesFor is Values there).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/ntt/ntt.hpp"

namespace {

using u32 = std::uint32_t;
using namespace fps::ntt;
namespace in = fps::ntt::detail::intrinsics;
constexpr u32 P = kModulus;

std::mt19937_64 rng(157);
int failures = 0, checks = 0;

void expect(bool ok, const std::string& what) {
    ++checks;
    if (!ok && ++failures <= 20) std::printf("FAIL: %s\n", what.c_str());
}

// count words, 64-byte aligned, then kPadding words.
struct Words {
    struct alignas(64) Line {
        u32 w[16];
    };
    std::vector<Line> lines;
    explicit Words(std::size_t count) : lines((count + kPadding + 15) / 16) {}
    u32* data() { return lines.data()->w; }
};

void fill(u32* p, std::size_t n, int kind) {
    for (std::size_t i = 0; i < n; ++i) p[i] = kind == 0 ? u32(rng() % P) : kind == 1 ? P - 1 : 0;
}

using Reference = in::Values<fps::simd::Scalar>;
using Wide = in::ValuesFor<fps::simd::Native>;

void blocks() {
    for (int lg = 6; lg <= 14; ++lg) {
        const std::size_t len = std::size_t(1) << lg, total = len / 64;
        std::vector<u32> rows(32 * fps::ntt::detail::value_row_count(lg));
        fps::ntt::detail::build_value_rows(rows.data(), lg);
        for (int kind = 0; kind < 3; ++kind) {
            Words from(len), expect_split(len), got(len), back(len);
            fill(from.data(), len, kind);
            Reference::split(from.data(), expect_split.data(), total, 0, rows.data());
            for (std::size_t first = 0; first < std::min<std::size_t>(total, 5); ++first)
                for (std::size_t count = 0; first + count <= total; count += count < 4 ? 1 : total / 4 + 1) {
                    const std::size_t at = 64 * first, words = 64 * count;
                    const std::string what = "lg " + std::to_string(lg) + " kind " + std::to_string(kind) + " first " +
                                             std::to_string(first) + " count " + std::to_string(count);
                    std::fill_n(got.data(), len, 0xdeadbeefu);
                    Wide::split(from.data() + at, got.data() + at, count, first, rows.data());
                    expect(std::equal(got.data() + at, got.data() + at + words, expect_split.data() + at), what + ": split");
                    expect(std::all_of(got.data(), got.data() + at, [](u32 x) { return x == 0xdeadbeefu; }) &&
                               std::all_of(got.data() + at + words, got.data() + len, [](u32 x) { return x == 0xdeadbeefu; }),
                           what + ": split wrote outside its blocks");
                    Wide::join(got.data() + at, back.data() + at, count, first, rows.data());
                    expect(std::equal(back.data() + at, back.data() + at + words, from.data() + at), what + ": join");
                    Wide::join(got.data() + at, got.data() + at, count, first, rows.data());  // in place
                    expect(std::equal(got.data() + at, got.data() + at + words, from.data() + at), what + ": join in place");
                }
            // join against the scalar join on values that are not a split's (any canonical words).
            Words values(len), expect_join(len), got_join(len);
            fill(values.data(), len, kind);
            Reference::join(values.data(), expect_join.data(), total, 0, rows.data());
            Wide::join(values.data(), got_join.data(), total, 0, rows.data());
            expect(std::equal(got_join.data(), got_join.data() + len, expect_join.data()), "lg " + std::to_string(lg) + ": join of any words");
        }
    }
}

template <class Variant>
void transforms(const char* name) {
    for (int lg = kMinLog; lg <= 15; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        const Transform t(lg);
        std::vector<u32> rows(32 * fps::ntt::detail::value_row_count(lg));
        fps::ntt::detail::build_value_rows(rows.data(), lg);
        Words f(len), leaves(len), values(len);
        fill(f.data(), len, 0);
        std::copy_n(f.data(), len, leaves.data());
        t.forward<Variant>(leaves.data());
        Reference::split(leaves.data(), values.data(), len / 64, 0, rows.data());
        Words g(len);
        std::copy_n(f.data(), len, g.data());
        t.forward<Variant>(g.data(), Form::values);
        const std::string what = std::string(name) + " lg " + std::to_string(lg);
        expect(std::equal(g.data(), g.data() + len, values.data()), what + ": forward values");
        t.inverse<Variant>(g.data(), Form::values);
        expect(std::equal(g.data(), g.data() + len, f.data()), what + ": inverse values");
        if (lg > kMinLog) {  // the upper half from p mod (x^(L/2) + 1), p of degree < L / 2 here: p itself
            Words p(len), h(len / 2);
            fill(p.data(), len / 2, 0);
            std::fill_n(p.data() + len / 2, len / 2, 0u);
            std::copy_n(p.data(), len / 2, h.data());
            t.forward(p.data(), Form::values);
            t.forward_upper<Variant>(h.data(), Form::values);
            expect(std::equal(h.data(), h.data() + len / 2, p.data() + len / 2), what + ": forward_upper values");
        }
    }
}

}  // namespace

int main() {
    blocks();
    transforms<variant::scalar>("scalar");
    fps::simd::for_each(fps::simd::All{}, []<class S> { transforms<variant::intrinsics<S>>(S::name); });
#ifdef __AVX2__
    transforms<variant::avx2>("avx2");
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    transforms<variant::avx512>("avx512");
#endif
    transforms<variant::tuned>("tuned");
    std::printf("%s: %d checks, %d failures (values on %s)\n", failures ? "FAIL" : "PASS", checks, failures,
                fps::simd::Native::lanes == 16 ? "16 lanes" : "Values");
    return failures ? 1 : 0;
}
