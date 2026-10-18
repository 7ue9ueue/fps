// Tests for leaf products whose rows [w A, A] keep Shoup's product w A not reduced (below 1.24 P;
// issue #218: avx2's ToeplitzLeaf and inverse_sum with AsmLeaf): toeplitz_product and inverse_product
// of 2 to 8 products, with leaf forms chosen so that every word of w A is near its largest value
// (the most of 64 random candidates per word) and the other factor P - 1, on every variant this
// machine runs and the tuned one, against the scalar model. Upper and lower halves of the output;
// A3 exact; the 64-bit sums near their bound overflow if the rows are larger than the bottoms assume.
#include "backend/ntt/test_util.hpp"

#include <string>

namespace {

#if __has_feature(address_sanitizer)
constexpr int kMaxTestLog = 14;
#else
constexpr int kMaxTestLog = 16;
#endif
constexpr int kScalarTestLog = 12;

// The modulus of leaf v, x^8 - w_v: w_v = (-1)^v r(v / 2) (test_util's leaf_form).
u32 leaf_weight(std::size_t v) { return v % 2 ? sub(0, root(v / 2)) : root(v / 2); }

// Shoup's product x w before its reduction: in [0, 2P), below (1 + x / 2^32) P.
u64 shoup(u32 x, u32 w) {
    const u64 q = (u64(w) << 32) / P;
    return u64(x) * w - (u64(x) * q >> 32) * P;
}

// A canonical leaf form whose words x make shoup(x, w_v) the largest of 64 random candidates.
std::vector<u32> large_rows(std::size_t len) {
    std::vector<u32> a(len);
    for (std::size_t v = 0; v < len / 8; ++v) {
        const u32 w = leaf_weight(v);
        for (int l = 0; l < 8; ++l) {
            u32 best = 0;
            for (int c = 0; c < 64; ++c) {
                const u32 x = random_below(P);
                if (shoup(x, w) > shoup(best, w)) best = x;
            }
            a[8 * v + l] = best;
        }
    }
    return a;
}

struct Form {
    explicit Form(const std::vector<u32>& v) : words(v.size()) { std::copy(v.begin(), v.end(), words.begin()); }
    u32* data() { return words.begin(); }
    std::vector<u32> get(std::size_t len) { return {data(), data() + len}; }
    Words words;
};

std::vector<u32> top(std::size_t len) { return std::vector<u32>(len, P - 1); }
std::vector<u32> plus(const std::vector<u32>& a, u32 c) {
    std::vector<u32> b(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) b[i] = add(a[i], c);
    return b;
}

std::vector<u32> product(const Transform& t, std::vector<u32> a, std::vector<u32> b) {
    Form x(a), y(b);
    Words z(t.length());
    t.multiply<variant::scalar>(x.data(), y.data(), z.begin());
    return {z.begin(), z.begin() + t.length()};
}
std::vector<u32> sum(std::vector<u32> a, const std::vector<u32>& b) {
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = add(a[i], b[i]);
    return a;
}
std::vector<u32> coefficients(const Transform& t, const std::vector<u32>& F) {
    Form w(F);
    t.inverse<variant::scalar>(w.data());
    return w.get(t.length());
}

template <class V>
void test_length(const std::string& name, int lg) {
    const Transform t(lg);
    const std::size_t len = t.length();
    for (std::size_t offset : {std::size_t(0), len / 2}) {
        const Output o{offset, false};
        const std::string at = name + " lg " + std::to_string(lg) + (offset ? " upper" : " lower");
        // toeplitz_product: the rows are of S = B0 + B1, D1 = A1 - A2, D3 = A3 - A2; the leaf
        // products' other factors A2, B1, B0. A2 = B1 = P - 1, and S, D1, D3 large rows.
        {
            const std::vector<u32> s = large_rows(len), d1 = large_rows(len), d3 = large_rows(len);
            const std::vector<u32> a2 = top(len), b1 = top(len), b0 = plus(s, 1), a1 = plus(d1, P - 1), a3 = plus(d3, P - 1);
            const std::vector<u32> c = coefficients(t, sum(product(t, a2, b0), product(t, a1, b1)));
            const std::vector<u32> side = sum(product(t, a3, b0), product(t, a2, b1));
            Form A1(a1), A2(a2), A3(a3), B0(b0), B1(b1);
            Words work(len);
            std::vector<u32> out(len / 2);
            t.toeplitz_product<V>(A1.data(), A2.data(), A3.data(), B0.data(), B1.data(), work.begin(), std::span<u32>(out), o);
            expect(std::equal(out.begin(), out.end(), c.begin() + std::ptrdiff_t(offset)), (at + " toeplitz_product: out").c_str());
            expect(A3.get(len) == side, (at + " toeplitz_product: A3").c_str());
        }
        // inverse_product of k products a_i b_i: rows of a_i large, b_i = P - 1.
        for (int k = 2; k <= kMaxTerms; ++k) {
            std::vector<Form> a, b;
            std::vector<u32> leaves(len, 0);
            a.reserve(std::size_t(k)), b.reserve(std::size_t(k));
            for (int i = 0; i < k; ++i) {
                a.emplace_back(large_rows(len)), b.emplace_back(top(len));
                leaves = sum(leaves, product(t, a.back().get(len), b.back().get(len)));
            }
            std::vector<Term> terms;
            for (int i = 0; i < k; ++i) terms.push_back({a[std::size_t(i)].data(), b[std::size_t(i)].data()});
            const std::vector<u32> c = coefficients(t, leaves);
            Words work(len);
            std::vector<u32> out(len / 2);
            t.inverse_product<V>(terms, work.begin(), std::span<u32>(out), o);
            expect(std::equal(out.begin(), out.end(), c.begin() + std::ptrdiff_t(offset)),
                   (at + " inverse_product of " + std::to_string(k)).c_str());
        }
    }
}

template <class V>
void test_variant(const std::string& name, int max_lg) {
    for (int lg = kMinLog; lg <= max_lg; ++lg) test_length<V>(name, lg);
}

}  // namespace

int main() {
    test_variant<variant::scalar>("scalar", kScalarTestLog);
    fps::simd::for_each(fps::simd::All{}, []<class S> {
        test_variant<variant::intrinsics<S>>(std::string("intrinsics ") + S::name, S::lanes == 1 ? kScalarTestLog : kMaxTestLog);
    });
#ifdef __AVX2__
    test_variant<variant::avx2>("avx2", kMaxTestLog);
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    test_variant<variant::avx512>("avx512", kMaxTestLog);
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
    test_variant<variant::neon>("neon", kMaxTestLog);
#endif
    test_variant<variant::tuned>("tuned", kMaxTestLog);
    return finish();
}
