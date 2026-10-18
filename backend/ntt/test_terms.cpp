// Tests for Transform::inverse_product with 1 to kMaxTerms (8) terms (issue #143): every list of
// product terms (p: a b) and terms alone (x: a) of 1 to 8 terms, in every order, at the short
// lengths; lists of products and some mixed ones at every length; on every variant this machine
// runs and the tuned one, against the scalar model (its leaf products, sums and inverse). Leaf
// forms random, all P - 1, or 0 and P - 1 (the edges of the bottoms' lazy sums); both output
// halves, negated or not, short counts; words around out and the work array's padding untouched.
#include "backend/ntt/test_util.hpp"

#include <string>

namespace {

constexpr u32 kSentinel = 0xdeadbeef;
#if __has_feature(address_sanitizer)
constexpr int kMaxTestLog = 16;
#else
constexpr int kMaxTestLog = 18;
#endif
constexpr int kEveryListMaxLog = 8;  // every list of p and x to here (510 lists)
constexpr int kScalarTestLog = 13;   // the scalar variant and intrinsics on 1 lane
constexpr int kThoroughMaxLog = 12;  // past it, fewer kinds and outputs

// A leaf form the ops read: 32-byte aligned, kPadding readable words past it.
struct Form {
    explicit Form(std::size_t len) : words(len) {}
    u32* data() { return words.begin(); }
    Words words;
};

// out: words [3, 3 + count) of a buffer whose other words hold kSentinel (an odd address).
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

using Run = std::function<void(const Transform&, std::span<const Term>, u32*, std::span<u32>, Output)>;
struct Variant {
    std::string name;
    int max_lg;
    Run run;
};

template <class V>
Variant variant_of(std::string name, int max_lg) {
    return {std::move(name), max_lg,
            [](const Transform& t, std::span<const Term> terms, u32* work, std::span<u32> out, Output o) {
                t.inverse_product<V>(terms, work, out, o);
            }};
}

std::vector<Variant> all_variants() {
    std::vector<Variant> v;
    v.push_back(variant_of<variant::scalar>("scalar", kScalarTestLog));
    fps::simd::for_each(fps::simd::All{}, [&]<class S> {
        v.push_back(variant_of<variant::intrinsics<S>>(std::string("intrinsics ") + S::name, S::lanes == 1 ? kScalarTestLog : kMaxTestLog));
    });
#ifdef __AVX2__
    v.push_back(variant_of<variant::avx2>("avx2", kMaxTestLog));
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    v.push_back(variant_of<variant::avx512>("avx512", kMaxTestLog));
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
    v.push_back(variant_of<variant::neon>("neon", kMaxTestLog));
#endif
    v.push_back(variant_of<variant::tuned>("tuned", kMaxTestLog));
    return v;
}

// Every list of n terms, 'p' or 'x' each: bit i of mask set is a product.
std::vector<std::string> every_list() {
    std::vector<std::string> lists;
    for (int n = 1; n <= kMaxTerms; ++n)
        for (int mask = 0; mask < 1 << n; ++mask) {
            std::string s;
            for (int i = 0; i < n; ++i) s += mask >> i & 1 ? 'p' : 'x';
            lists.push_back(s);
        }
    return lists;
}

// Products only (the callers' lists), and terms alone at the ends and between.
std::vector<std::string> some_lists() {
    std::vector<std::string> lists;
    for (int n = 1; n <= kMaxTerms; ++n) lists.push_back(std::string(std::size_t(n), 'p'));
    for (const char* s : {"x", "px", "xp", "xpppp", "ppppx", "xppppppp", "pppppppx", "ppxpppxp", "pxpxpxpx", "xxxxxxxx"})
        lists.push_back(s);
    return lists;
}

void fill(u32* w, std::size_t len, int kind) {
    for (std::size_t i = 0; i < len; ++i) w[i] = kind == 0 ? random_below(P) : kind == 1 ? P - 1 : u32(random_below(2)) * (P - 1);
}

// The coefficients of the sum of the list's terms over a[i], b[i]: the scalar model.
std::vector<u32> model(const Transform& t, const std::string& list, std::vector<Form>& a, std::vector<Form>& b) {
    const std::size_t len = t.length();
    std::vector<u32> sum(len, 0);
    Words product(len);
    for (std::size_t i = 0; i < list.size(); ++i) {
        const u32* x = a[i].data();
        if (list[i] == 'p') t.multiply<variant::scalar>(a[i].data(), b[i].data(), product.begin()), x = product.begin();
        for (std::size_t j = 0; j < len; ++j) sum[j] = add(sum[j], x[j]);
    }
    Words w(len);
    std::copy(sum.begin(), sum.end(), w.begin());
    t.inverse<variant::scalar>(w.begin());
    return {w.begin(), w.begin() + len};
}

void test_length(int lg, const std::vector<Variant>& vs) {
    const Transform t(lg);
    const std::size_t len = t.length();
    const std::vector<std::string> lists = lg <= kEveryListMaxLog ? every_list() : some_lists();
    for (int kind = 0; kind < (lg <= kThoroughMaxLog ? 3 : 2); ++kind) {
        std::vector<Form> a, b;
        for (int i = 0; i < kMaxTerms; ++i) {
            a.emplace_back(len), b.emplace_back(len);
            fill(a.back().data(), len, kind), fill(b.back().data(), len, kind);
        }
        for (std::size_t l = 0; l < lists.size(); ++l) {
            const std::string& list = lists[l];
            std::vector<Term> terms;
            for (std::size_t i = 0; i < list.size(); ++i) terms.push_back({a[i].data(), list[i] == 'p' ? b[i].data() : nullptr});
            const std::vector<u32> c = model(t, list, a, b);
            // Outputs: every half and sign for the product lists past the every-list lengths, else
            // one per list in turn; full halves, and a short count at the short lengths.
            std::vector<std::pair<Output, std::size_t>> outs;
            for (int m = 0; m < 4; ++m) {
                if (lg <= kEveryListMaxLog && m != int(l % 4)) continue;
                const Output o{m & 1 ? len / 2 : 0, (m & 2) != 0};
                outs.push_back({o, len / 2});
                if (lg <= kThoroughMaxLog) outs.push_back({o, m & 1 ? std::size_t(5) : len});
            }
            for (const auto& [o, count] : outs) {
                std::vector<u32> expected(count);
                for (std::size_t i = 0; i < count; ++i) expected[i] = o.negate ? sub(0, c[o.offset + i]) : c[o.offset + i];
                for (const Variant& v : vs) {
                    if (lg > v.max_lg) continue;
                    Words work(len);
                    std::fill(work.begin() + len, work.begin() + len + kPadding, kSentinel);
                    Out out(count);
                    v.run(t, terms, work.begin(), out.span(), o);
                    const bool ok = out.words() == expected && out.edges_kept() &&
                                    std::all_of(work.begin() + len, work.begin() + len + kPadding, [](u32 x) { return x == kSentinel; });
                    if (!ok && failures < 20)
                        std::printf("  %s lg %d kind %d list %s offset %zu%s count %zu\n", v.name.c_str(), lg, kind, list.c_str(),
                                    o.offset, o.negate ? " negated" : "", count);
                    expect(ok, "inverse_product of terms");
                }
            }
        }
    }
}

// Terms that share leaf forms (the same a in several terms, a term's a as another's b), and the
// argument checks: no term, more than kMaxTerms.
void test_shared_and_checks(const std::vector<Variant>& vs) {
    const Transform t(10);
    const std::size_t len = t.length();
    std::vector<Form> a, b;
    for (int i = 0; i < 2; ++i) a.emplace_back(len), b.emplace_back(len), fill(a.back().data(), len, 0), fill(b.back().data(), len, 0);
    const Term terms[] = {{a[0].data(), b[0].data()}, {a[0].data(), b[1].data()}, {b[0].data(), a[0].data()},
                          {a[1].data(), a[1].data()}, {a[0].data(), b[0].data()}, {b[1].data(), nullptr},
                          {a[1].data(), b[0].data()}, {b[1].data(), b[1].data()}};
    std::vector<Form> fa, fb;
    std::string list;
    for (const Term& term : terms) {
        fa.emplace_back(len), fb.emplace_back(len);
        std::copy_n(term.a, len, fa.back().data());
        if (term.b) std::copy_n(term.b, len, fb.back().data());
        list += term.b ? 'p' : 'x';
    }
    const std::vector<u32> c = model(t, list, fa, fb);
    for (const Variant& v : vs) {
        Words work(len);
        Out out(len);
        v.run(t, terms, work.begin(), out.span(), {});
        expect(out.words() == c, "inverse_product of shared leaf forms");
    }
    std::vector<u32> work(len + kPadding), out(len);
    const Term nine[9] = {};
    const auto throws = [&](std::span<const Term> list, const char* what) {
        bool thrown = false;
        try {
            t.inverse_product(list, work.data(), std::span<u32>(out));
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        expect(thrown, what);
    };
    throws(std::span<const Term>(), "inverse_product: no term");
    throws(std::span<const Term>(nine, 9), "inverse_product: 9 terms");
}

}  // namespace

int main() {
    const std::vector<Variant> vs = all_variants();
    for (int lg = kMinLog; lg <= kMaxTestLog; ++lg) test_length(lg, vs);
    test_shared_and_checks(vs);
    return finish();
}
