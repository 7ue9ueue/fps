// Tests for issue #210: Transform's hooked ops in place, on every variant this machine runs and
// the tuned one, against the same op apart. forward_hooked with in at F (sources of every size
// class: whole, past half, half and below, short), every column bit, the hook changing its leaves;
// inverse_hooked whose out is the memory the hook copies its leaves from (both halves, negated or
// not, every column bit). Words past F and out untouched.
#include "backend/ntt/test_util.hpp"

namespace {

#if __has_feature(address_sanitizer)
constexpr int kMaxTestLog = 16;
#else
constexpr int kMaxTestLog = 20;
#endif
constexpr int kScalarTestLog = 13;
constexpr u32 kSentinel = 0xdeadbeef;

// count words, 32-byte aligned, kSentinel in the kPadding words past them.
struct Padded {
    explicit Padded(std::size_t count) : size(count), w(count) { std::fill_n(w.begin() + count, kPadding, kSentinel); }
    u32* begin() { return w.begin(); }
    bool padding_kept() { return std::all_of(begin() + size, begin() + size + kPadding, [](u32 x) { return x == kSentinel; }); }
    std::size_t size;
    Words w;
};

std::vector<std::size_t> column_bits(std::size_t len) {
    std::vector<std::size_t> bits = {0};
    for (std::size_t b = 1; b <= len / 4; b *= 2) bits.push_back(b);
    return bits;
}

template <class V>
void test_forward(const std::string& name, const Transform& t, int lg) {
    const std::size_t len = t.length();
    const auto bump = [](u32* leaves, std::size_t, std::size_t count) {
        for (std::size_t i = 0; i < 8 * count; ++i) leaves[i] = leaves[i] + 1 == P ? 0 : leaves[i] + 1;
    };
    for (std::size_t zero : column_bits(len)) {
        for (std::size_t n : {len, len / 2 + 3, len / 2, len / 2 - 5, std::size_t(1 + rng() % (len / 4))}) {
            std::vector<u32> in = random_poly(n, n % 3 == 0 ? 1 : 0);
            for (std::size_t i = 0; i < n; ++i)
                if (i & zero) in[i] = 0;
            Padded apart(len), here(len);
            t.forward_hooked<V>(std::span<const u32>(in), apart.begin(), bump, zero);
            for (std::size_t i = 0; i < len; ++i) here.begin()[i] = i < n ? in[i] : u32(rng());  // past n: not read
            t.forward_hooked<V>(std::span<const u32>(here.begin(), n), here.begin(), bump, zero);
            const bool ok = std::equal(here.begin(), here.begin() + len, apart.begin()) && here.padding_kept();
            expect(ok, (name + " forward_hooked in place, zero " + std::to_string(zero) + " n " + std::to_string(n)).c_str(),
                   std::size_t(lg));
            if (lg > 14) break;
        }
    }
}

template <class V>
void test_inverse(const std::string& name, const Transform& t, int lg) {
    const std::size_t len = t.length();
    std::vector<u32> A(len);
    for (auto& x : A) x = random_below(P);
    for (std::size_t unused : column_bits(len)) {
        for (std::size_t offset : {std::size_t(0), len / 2}) {
            for (bool negate : {false, true}) {
                const std::size_t count = len - offset;
                Words work(len);
                std::vector<u32> apart(count);
                const auto from = [](const u32* source) {
                    return [source](u32* leaves, std::size_t first, std::size_t n) { std::copy_n(source + 8 * first, 8 * n, leaves); };
                };
                t.inverse_hooked<V>(from(A.data()), work.begin(), std::span<u32>(apart), {offset, negate}, unused);
                Padded here(len);  // the hook reads here, out is its first count words
                std::copy(A.begin(), A.end(), here.begin());
                t.inverse_hooked<V>(from(here.begin()), work.begin(), std::span<u32>(here.begin(), count), {offset, negate}, unused);
                bool ok = here.padding_kept() && std::equal(here.begin() + count, here.begin() + len, A.begin() + std::ptrdiff_t(count));
                for (std::size_t i = 0; i < count; ++i) ok &= ((offset + i) & unused) || here.begin()[i] == apart[i];
                expect(ok, (name + " inverse_hooked out at the hook's source, unused " + std::to_string(unused) + " offset " +
                            std::to_string(offset) + (negate ? " negated" : "")).c_str(), std::size_t(lg));
                if (lg > 14) break;
            }
        }
    }
}

template <class V>
void test_variant(const std::string& name, int max_lg) {
    for (int lg = kMinLog; lg <= max_lg; ++lg) {
        const Transform t(lg);
        test_forward<V>(name, t, lg);
        test_inverse<V>(name, t, lg);
    }
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
    test_variant<variant::tuned>("tuned", kMaxTestLog);
    return finish();
}
