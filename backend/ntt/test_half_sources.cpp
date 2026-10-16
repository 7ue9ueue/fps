// Tests for issue #87 round 2 (notes/87-r2.md): the forward's lowest levels as loops (avx2:
// forward_h4, forward_leaves_loop), the radix-8 top level from a source in the lower half (avx2:
// forward_radix8_lower_from, two passes; cyclic_product then ends with the radix-8 inverse top)
// and forward(in, F) in place (in at F, at most L/2 words). On every variant this machine runs and
// the tuned one, against the scalar model: forward in place at every length, and forward(in, F)
// apart and in place, forward_upper(in, H), forward_product and cyclic_product (both halves out,
// negated or not) for sources around the radix-8 level's blocks (L/8 words), at every length to
// 2^kMaxTestLog. Words past each output untouched.
#include "backend/ntt/ntt.hpp"

#include <algorithm>
#include <cstdio>
#include <new>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using namespace fps::ntt;

constexpr u32 P = kModulus;
#if __has_feature(address_sanitizer)
constexpr int kMaxTestLog = 16;
#else
constexpr int kMaxTestLog = 20;
#endif
constexpr int kSourcesMaxLog = 14;  // past it, fewer source sizes
constexpr int kScalarMaxLog = 14;
constexpr u32 kSentinel = 0xdeadbeef;

std::mt19937_64 rng(872);
int failures = 0;

void expect(bool ok, const std::string& what, u64 detail = 0) {
    if (ok) return;
    if (++failures <= 30) std::printf("FAIL: %s (%llu)\n", what.c_str(), static_cast<unsigned long long>(detail));
}

u32 neg(u32 x) { return x ? P - x : 0; }

// count words, 64-byte aligned, then kPadding words of kSentinel, which no op may change.
struct Words {
    explicit Words(std::size_t count)
        : size(count), data(static_cast<u32*>(::operator new((count + kPadding) * sizeof(u32), std::align_val_t(64)))) {
        std::fill(data + count, data + count + kPadding, kSentinel);
    }
    ~Words() { ::operator delete(data, std::align_val_t(64)); }
    Words(const Words&) = delete;
    Words& operator=(const Words&) = delete;
    bool padding_kept() const { return std::all_of(data + size, data + size + kPadding, [](u32 x) { return x == kSentinel; }); }
    std::vector<u32> words() const { return {data, data + size}; }
    std::size_t size;
    u32* data;
};

std::vector<u32> random_poly(std::size_t n, bool top) {
    std::vector<u32> p(n);
    for (auto& x : p) x = top ? P - 1 - u32(rng() % 2) : u32(rng() % P);
    return p;
}

// The scalar model: the leaf form of in (zero past it), coefficients of a leaf form, leaf products.
std::vector<u32> leaves(const Transform& t, const std::vector<u32>& in) {
    Words w(t.length());
    std::fill(std::copy(in.begin(), in.end(), w.data), w.data + t.length(), 0u);
    t.forward<variant::scalar>(w.data);
    return w.words();
}
std::vector<u32> coefficients(const Transform& t, const std::vector<u32>& F) {
    Words w(t.length());
    std::copy(F.begin(), F.end(), w.data);
    t.inverse<variant::scalar>(w.data);
    return w.words();
}
std::vector<u32> leaf_product(const Transform& t, const std::vector<u32>& a, const std::vector<u32>& b) {
    Words x(t.length()), y(t.length()), z(t.length());
    std::copy(a.begin(), a.end(), x.data), std::copy(b.begin(), b.end(), y.data);
    t.multiply<variant::scalar>(x.data, y.data, z.data);
    return z.words();
}

// Sources in the lower half around the radix-8 level's blocks of L/8 words (and steps of 8 words).
std::vector<std::size_t> source_sizes(std::size_t len) {
    const std::size_t block = len / 8;
    std::vector<std::size_t> s = {0, 1, 8, len / 2 - 1, len / 2, len / 2 + 1, 1 + rng() % (len / 2)};
    if (len <= (std::size_t(1) << kSourcesMaxLog)) {
        for (std::size_t b = 1; b <= 4; ++b)
            for (std::size_t n : {b * block - 9, b * block - 1, b * block, b * block + 1, b * block + 7}) s.push_back(n);
    } else {
        s.insert(s.end(), {block + 1, 3 * block - 1});
    }
    std::erase_if(s, [len](std::size_t n) { return n > len; });
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
    return s;
}

template <class V>
void test_length(const std::string& name, int lg) {
    const Transform t(lg);
    const std::size_t len = t.length();
    // In place, every value range of the input.
    for (bool top : {false, true}) {
        const std::vector<u32> in = random_poly(len, top);
        Words F(len);
        std::copy(in.begin(), in.end(), F.data);
        t.forward<V>(F.data);
        expect(F.words() == leaves(t, in) && F.padding_kept(), name + " forward in place", top);
    }
    const std::vector<u32> b = random_poly(len, false);
    const std::vector<u32> B = leaves(t, b);
    Words Bw(len);
    std::copy(B.begin(), B.end(), Bw.data);
    for (std::size_t n : source_sizes(len)) {
        const std::vector<u32> in = random_poly(n, n % 2 == 1);
        const std::vector<u32> A = leaves(t, in), AB = leaf_product(t, A, B), full = coefficients(t, AB);
        Words F(len), G(len);
        t.forward<V>(std::span<const u32>(in), F.data);
        expect(F.words() == A && F.padding_kept(), name + " forward(in, F)", n);
        if (2 * n <= len && lg > kMinLog) {  // forward_upper(in, H): the upper half of A
            Words H(len / 2);
            t.forward_upper<V>(std::span<const u32>(in), H.data);
            expect(std::equal(A.begin() + std::ptrdiff_t(len / 2), A.end(), H.data) && H.padding_kept(),
                   name + " forward_upper(in, H)", n);
        }
        if (2 * n <= len) {  // in place: the source in F's lower half, garbage past it
            Words H(len);
            std::fill(std::copy(in.begin(), in.end(), H.data), H.data + len, P - 1);
            t.forward<V>(std::span<const u32>(H.data, n), H.data);
            expect(H.words() == A && H.padding_kept(), name + " forward(in, F) in place", n);
        }
        t.forward_product<V>(std::span<const u32>(in), Bw.data, G.data);
        expect(G.words() == AB && G.padding_kept(), name + " forward_product", n);
        for (std::size_t offset : {std::size_t(0), len / 2}) {
            for (bool negate : {false, true}) {
                const std::size_t count = len / 2 - (negate ? 3 : 0);
                Words work(len);
                std::vector<u32> out(count + 4, kSentinel);
                t.cyclic_product<V>(std::span<const u32>(in), Bw.data, work.data, std::span<u32>(out.data(), count),
                                    {.offset = offset, .negate = negate});
                bool ok = work.padding_kept() && std::all_of(out.begin() + std::ptrdiff_t(count), out.end(), [](u32 x) { return x == kSentinel; });
                for (std::size_t i = 0; i < count; ++i) ok = ok && out[i] == (negate ? neg(full[offset + i]) : full[offset + i]);
                expect(ok, name + " cyclic_product offset " + std::to_string(offset) + (negate ? " negated" : ""), n);
            }
        }
    }
}

template <class V>
void test_variant(const std::string& name, int max_lg = kMaxTestLog) {
    for (int lg = kMinLog; lg <= max_lg; ++lg) test_length<V>(name + " lg " + std::to_string(lg), lg);
}

}  // namespace

int main() {
    test_variant<variant::scalar>("scalar", kScalarMaxLog);
    fps::simd::for_each(fps::simd::All{}, []<class S> {
        test_variant<variant::intrinsics<S>>(std::string("intrinsics ") + S::name, S::lanes == 1 ? kScalarMaxLog : kMaxTestLog);
    });
#ifdef __AVX2__
    test_variant<variant::avx2>("avx2");
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    test_variant<variant::avx512>("avx512");
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
    test_variant<variant::neon>("neon");
#endif
    test_variant<variant::tuned>("tuned");
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
