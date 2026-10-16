// Tests of the values form's split and join by block range (issue #98): for ValuesFor<S> of every
// simd struct this machine runs (Values on 1, 4 or 8 lanes; WideValues on 16), every first block
// below 20 (and some near the end) and count up to 20, against the scalar model (detail::scalar::split
// and join) over the whole transform. The ranges cross block 8 (A(g / 8) = 1 below it: no A
// products), the steps of 2 blocks or 2 pairs and the lone blocks around them. In place and apart;
// words outside the range untouched; random words, all P - 1, all 0. The join leaves the 8-point
// transform's gain in: join<true> is canonical and join<false> below 2P, both 8 times the model's.
// Transform's values-form ops for every variant are test_leaf.cpp's.
#include "backend/ntt/test_util.hpp"

#include "backend/ntt/scalar.hpp"

namespace {

namespace in = fps::ntt::detail::intrinsics;

constexpr u32 kUnwritten = 0xdeadbeefu;

std::vector<u32> words_of(std::size_t n, int kind) {
    std::vector<u32> w(n);
    for (u32& x : w) x = kind == 0 ? random_below(P) : kind == 1 ? P - 1 : 0;
    return w;
}

template <class Values>
void test_ranges(const char* name) {
    for (const int lg : {6, 9, 12, 14}) {
        const std::size_t len = std::size_t(1) << lg, total = len / 64;
        std::vector<u32> rows(32 * fps::ntt::detail::value_row_count(lg));
        fps::ntt::detail::build_value_rows(rows.data(), lg);
        for (int kind = 0; kind < 3; ++kind) {
            // The model: split of leaves and join of values, over the whole length.
            const auto leaves = words_of(len, kind), values = words_of(len, kind);
            std::vector<u32> split(len), join(len);
            fps::ntt::detail::scalar::split(leaves.data(), split.data(), lg, rows.data());
            fps::ntt::detail::scalar::join(values.data(), join.data(), lg, rows.data());
            std::vector<std::size_t> firsts;
            for (std::size_t first = 0; first < std::min<std::size_t>(total, 20); ++first) firsts.push_back(first);
            for (std::size_t back = 1; back <= 3 && back < total; ++back) firsts.push_back(total - back);
            for (const std::size_t first : firsts)
                for (std::size_t count = 0; count <= std::min<std::size_t>(total - first, 20); ++count) {
                    const std::size_t at = 64 * first, n = 64 * count;
                    const u64 detail = u64(lg) << 32 | u64(kind) << 24 | first << 8 | count;
                    const auto outside = [&](const Words& w) {
                        const u32* p = const_cast<Words&>(w).begin();
                        return std::all_of(p, p + at, [](u32 x) { return x == kUnwritten; }) &&
                               std::all_of(p + at + n, p + len, [](u32 x) { return x == kUnwritten; });
                    };
                    Words got(len), in_place(len);
                    std::fill_n(got.begin(), len, kUnwritten);
                    Values::split(leaves.data() + at, got.begin() + at, count, first, rows.data());
                    expect(std::equal(got.begin() + at, got.begin() + at + n, split.data() + at) && outside(got), name, detail);
                    std::fill_n(in_place.begin(), len, kUnwritten);
                    std::copy_n(leaves.data() + at, n, in_place.begin() + at);
                    Values::split(in_place.begin() + at, in_place.begin() + at, count, first, rows.data());
                    expect(std::equal(in_place.begin() + at, in_place.begin() + at + n, split.data() + at) && outside(in_place),
                           name, detail | u64(1) << 60);
                    // join: 8 times the model's, canonical or below 2P.
                    const auto check = [&](const Words& w, bool canonical) {
                        const u32* p = const_cast<Words&>(w).begin();
                        bool ok = outside(w);
                        for (std::size_t i = at; i < at + n; ++i)
                            ok &= p[i] < (canonical ? P : 2 * P) && p[i] % P == mul(8, join[i]);
                        return ok;
                    };
                    std::fill_n(got.begin(), len, kUnwritten);
                    Values::template join<true>(values.data() + at, got.begin() + at, count, first, rows.data());
                    expect(check(got, true), name, detail | u64(2) << 60);
                    std::fill_n(got.begin(), len, kUnwritten);
                    Values::template join<false>(values.data() + at, got.begin() + at, count, first, rows.data());
                    expect(check(got, false), name, detail | u64(3) << 60);
                    std::fill_n(in_place.begin(), len, kUnwritten);
                    std::copy_n(values.data() + at, n, in_place.begin() + at);
                    Values::template join<false>(in_place.begin() + at, in_place.begin() + at, count, first, rows.data());
                    expect(check(in_place, false), name, detail | u64(4) << 60);
                    std::copy_n(values.data() + at, n, in_place.begin() + at);
                    Values::template join<true>(in_place.begin() + at, in_place.begin() + at, count, first, rows.data());
                    expect(check(in_place, true), name, detail | u64(5) << 60);
                }
        }
    }
}

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
// The zmm leaf products (WideBottom::products, Transform::multiply on Zen 4 and 5), on every
// AVX-512 target: against Bottom<Avx512Vl>::products for 2 to 64 groups from group 0 and others,
// out apart, = a, = b, and accumulating.
void test_wide_products() {
    using Wide = in::WideBottom<fps::simd::Avx512>;
    using Narrow = in::Bottom<fps::simd::Avx512Vl>;
    const int lg = 12;
    std::vector<u32> table(fps::ntt::detail::table_words(lg) + 16);
    Native::build_table(table.data(), (std::size_t(1) << lg) / 16, false);
    for (const std::size_t first : {0, 2, 6, 64})
        for (std::size_t count = 2; count <= 64; count += count < 8 ? 2 : 14) {
            const std::size_t n = 32 * count;
            for (int kind = 0; kind < 3; ++kind) {
                const auto a = words_of(n, kind), b = words_of(n, kind == 2 ? 0 : 1 - kind), c = words_of(n, 0);
                for (int where = 0; where < 3; ++where)  // out apart (starting as c), = a, = b
                    for (const bool add : {false, true}) {
                        const auto& start = where == 0 ? c : where == 1 ? a : b;
                        Words x(n), y(n), apart(n), expect_out(n);
                        std::copy(a.begin(), a.end(), x.begin()), std::copy(b.begin(), b.end(), y.begin());
                        std::copy(start.begin(), start.end(), apart.begin()), std::copy(start.begin(), start.end(), expect_out.begin());
                        if (add) Narrow::products<true>(a.data(), b.data(), expect_out.begin(), count, first, table.data());
                        else Narrow::products<false>(a.data(), b.data(), expect_out.begin(), count, first, table.data());
                        u32* out = where == 0 ? apart.begin() : where == 1 ? x.begin() : y.begin();
                        if (add) Wide::products<true>(x.begin(), y.begin(), out, count, first, table.data());
                        else Wide::products<false>(x.begin(), y.begin(), out, count, first, table.data());
                        expect(std::equal(out, out + n, expect_out.begin()), "wide leaf products",
                               u64(first) << 32 | count << 8 | u64(kind) << 4 | u64(where) << 1 | add);
                    }
            }
        }
}
#endif

}  // namespace

int main() {
    fps::simd::for_each(fps::simd::All{}, []<class S> {
        const std::string name = std::string("values ") + S::name;
        test_ranges<in::ValuesFor<S>>(name.c_str());
    });
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    test_wide_products();
#endif
    return finish();
}
