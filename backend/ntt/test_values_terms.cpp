// Tests for the fused ops in the values form (issue #251): forward(in, V, Form::values) from a span
// against forward(f, Form::values) in place on the scalar variant (in of every length class: full,
// half, short, empty), to_values(G, V) against it, and inverse_product(terms, ..., Form::values)
// of 1 to 8 values-form terms (products and terms alone) against the scalar model's leaf-form
// inverse_product of the same polynomials, word for word: both output halves, negated or not, short
// counts, words around out untouched. On every variant this machine runs and the tuned one, at every
// length from 2^kMinLog to 2^12 and at 2^14, 2^16, 2^18 (vector variants); random words, all P - 1,
// zeros.
#include "backend/ntt/test_util.hpp"

#include <string>

namespace {

constexpr u32 kSentinel = 0xdeadbeef;
constexpr int kShortMaxLog = 12;  // every length to here
#if __has_feature(address_sanitizer)
constexpr int kLongMaxLog = 16;
#else
constexpr int kLongMaxLog = 18;
#endif
constexpr int kScalarTestLog = 12;

// out: words [3, 3 + count) of a buffer whose other words hold kSentinel.
struct Out {
    explicit Out(std::size_t count) : buffer(count + 8, kSentinel), count(count) {}
    std::span<u32> span() { return {buffer.data() + 3, count}; }
    std::vector<u32> words() const { return {buffer.begin() + 3, buffer.begin() + 3 + std::ptrdiff_t(count)}; }
    bool edges_kept() const {
        return std::all_of(buffer.begin(), buffer.begin() + 3, [](u32 x) { return x == kSentinel; }) &&
               std::all_of(buffer.begin() + 3 + std::ptrdiff_t(count), buffer.end(), [](u32 x) { return x == kSentinel; });
    }
    std::vector<u32> buffer;
    std::size_t count;
};

std::vector<int> lengths(int max_lg) {
    std::vector<int> v;
    for (int lg = kMinLog; lg <= std::min(max_lg, kShortMaxLog); ++lg) v.push_back(lg);
    for (int lg : {14, 16, 18})
        if (lg <= max_lg && lg <= kLongMaxLog) v.push_back(lg);
    return v;
}

// The values form of p (zero past it) by the scalar variant, in place.
std::vector<u32> values_of(const Transform& t, const std::vector<u32>& p) {
    Words w(t.length());
    std::fill_n(w.begin(), t.length(), 0u);
    std::copy(p.begin(), p.end(), w.begin());
    t.forward<variant::scalar>(w.begin(), Form::values);
    return {w.begin(), w.begin() + t.length()};
}

template <class V>
void test_forward(const char* name, const Transform& t, int lg) {
    const std::size_t len = t.length();
    for (const std::size_t n : {len, len - 5, len / 2, len / 2 - 3, len / 2 + 1, std::size_t(1), std::size_t(0)}) {
        for (int kind = 0; kind < 3; ++kind) {
            if (kind && lg > 10 && n != len) continue;
            const std::vector<u32> p = random_poly(n, kind);
            Words in(n), F(len);
            std::copy(p.begin(), p.end(), in.begin());
            std::fill_n(F.begin(), len, kSentinel);
            t.forward<V>(std::span<const u32>(in.begin(), n), F.begin(), Form::values);
            const std::vector<u32> want = values_of(t, p);
            expect(std::equal(want.begin(), want.end(), F.begin()), name, u64(lg) * 1000000 + n * 10 + u64(kind));
            expect(std::equal(p.begin(), p.end(), in.begin()), "forward values: in kept", u64(lg));
            Words G(len);
            std::copy(p.begin(), p.end(), G.begin());
            std::fill_n(G.begin() + n, len - n, 0u);
            t.forward<variant::scalar>(G.begin());
            Words W(len);
            t.to_values(G.begin(), W.begin());
            expect(std::equal(want.begin(), want.end(), W.begin()), "to_values", u64(lg) * 1000000 + n);
            t.to_values(G.begin(), G.begin());
            expect(std::equal(want.begin(), want.end(), G.begin()), "to_values in place", u64(lg) * 1000000 + n);
        }
    }
    // Form::leaves is forward(in, F).
    const std::vector<u32> p = random_poly(len / 2, 0);
    Words A(len), B(len);
    t.forward<V>(std::span<const u32>(p), A.begin(), Form::leaves);
    t.forward<V>(std::span<const u32>(p), B.begin());
    expect(std::equal(A.begin(), A.begin() + len, B.begin()), "forward Form::leaves", u64(lg));
}

template <class V>
void test_terms(const char* name, const Transform& t, int lg) {
    const std::size_t len = t.length(), half = len / 2;
    for (int kind = 0; kind < 3; ++kind) {
        if (kind && lg > 10) break;
        std::vector<std::vector<u32>> polys(2 * kMaxTerms);
        std::vector<Words> leaves, values;
        for (int i = 0; i < 2 * kMaxTerms; ++i) {
            polys[i] = random_poly(len, i % 3 == 2 ? (kind + 1) % 3 : kind);
            leaves.emplace_back(len);
            values.emplace_back(len);
            std::copy(polys[i].begin(), polys[i].end(), leaves.back().begin());
            t.forward<variant::scalar>(leaves.back().begin());
            const std::vector<u32> v = values_of(t, polys[i]);
            std::copy(v.begin(), v.end(), values.back().begin());
        }
        for (int n = 1; n <= kMaxTerms; ++n) {
            for (int alone = 0; alone < 2; ++alone) {  // alone: odd-numbered terms without b
                if (alone && lg > 10 && n != 3) continue;
                std::vector<Term> lt(n), vt(n);
                for (int i = 0; i < n; ++i) {
                    const bool a = alone && i % 2;
                    lt[i] = {leaves[2 * i].begin(), a ? nullptr : leaves[2 * i + 1].begin()};
                    vt[i] = {values[2 * i].begin(), a ? nullptr : values[2 * i + 1].begin()};
                }
                struct Case {
                    std::size_t offset, count;
                    bool negate;
                };
                std::vector<Case> cases = {{half, half, false}, {half, half, true}, {0, half, false}, {0, len, true}};
                if (lg <= 10) cases.insert(cases.end(), {{half, half - 7, false}, {0, 1, true}, {half, 3, true}, {0, len - 9, false}});
                for (const Case& c : cases) {
                    Out want(c.count), got(c.count);
                    Words work(len), work2(len);
                    t.inverse_product<variant::scalar>(lt, work.begin(), want.span(), {.offset = c.offset, .negate = c.negate});
                    t.inverse_product<V>(vt, work2.begin(), got.span(), {.offset = c.offset, .negate = c.negate}, Form::values);
                    const u64 detail = u64(lg) * 1000000 + u64(n) * 10000 + u64(alone) * 1000 + c.count % 1000 + (c.negate ? 500 : 0);
                    expect(want.words() == got.words(), name, detail);
                    expect(got.edges_kept(), "inverse_product values: words around out", detail);
                }
            }
        }
    }
}

template <class V>
void test_variant(const char* name, int max_lg) {
    for (const int lg : lengths(max_lg)) {
        const Transform t(lg);
        test_forward<V>((std::string(name) + " forward values").c_str(), t, lg);
        test_terms<V>((std::string(name) + " inverse_product values").c_str(), t, lg);
    }
}

}  // namespace

int main() {
    test_variant<variant::scalar>("scalar", kScalarTestLog);
    fps::simd::for_each(fps::simd::All{}, [&]<class S> {
        const std::string name = std::string("intrinsics ") + S::name;
        test_variant<variant::intrinsics<S>>(name.c_str(), S::lanes == 1 ? kScalarTestLog : kLongMaxLog);
    });
#ifdef __AVX2__
    test_variant<variant::avx2>("avx2", kLongMaxLog);
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    test_variant<variant::avx512>("avx512", kLongMaxLog);
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
    test_variant<variant::neon>("neon", kLongMaxLog);
#endif
    test_variant<variant::tuned>("tuned", kLongMaxLog);
    // Past 2^kValuesMaxLog both throw.
    bool threw = false;
    try {
        const Transform t(kValuesMaxLog + 1);
        Words F(16);
        t.forward(std::span<const u32>(F.begin(), 1), F.begin(), Form::values);
    } catch (const std::length_error&) {
        threw = true;
    }
    expect(threw, "forward(in, F, Form::values) past 2^kValuesMaxLog throws");
    return finish();
}
