// Tests for Transform::toeplitz_product (issue #87 round 2, notes/87-r2.md) and inverse_product of a
// product and a term alone (avx2: one asm bottom), on every variant this machine runs and the tuned
// one, against the scalar model's leaf products and inverse: both output halves, negated or not,
// several output sizes; A3 = A3 B0 + A2 B1 written exactly, the other leaf forms and the words past
// every array untouched. Leaf forms random and all P - 1.
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
constexpr int kMaxTestLog = 18;
#endif
constexpr int kScalarMaxLog = 14;
constexpr int kThoroughMaxLog = 12;  // past it, fewer output sizes
constexpr u32 kSentinel = 0xdeadbeef;

std::mt19937_64 rng(87);
int failures = 0;

void expect(bool ok, const std::string& what, u64 detail = 0) {
    if (ok) return;
    if (++failures <= 30) std::printf("FAIL: %s (%llu)\n", what.c_str(), static_cast<unsigned long long>(detail));
}

u32 add(u32 x, u32 y) { return u32((u64(x) + y) % P); }
u32 neg(u32 x) { return x ? P - x : 0; }

// count words, 64-byte aligned, then kPadding words of kSentinel, which no op may change.
struct Words {
    explicit Words(std::size_t count)
        : size(count), data(static_cast<u32*>(::operator new((count + kPadding) * sizeof(u32), std::align_val_t(64)))) {
        std::fill(data + count, data + count + kPadding, kSentinel);
    }
    explicit Words(const std::vector<u32>& v) : Words(v.size()) { std::copy(v.begin(), v.end(), data); }
    ~Words() { ::operator delete(data, std::align_val_t(64)); }
    Words(const Words&) = delete;
    Words& operator=(const Words&) = delete;
    bool padding_kept() const { return std::all_of(data + size, data + size + kPadding, [](u32 x) { return x == kSentinel; }); }
    std::vector<u32> words() const { return {data, data + size}; }
    std::size_t size;
    u32* data;
};

std::vector<u32> random_form(std::size_t n, bool top) {
    std::vector<u32> p(n);
    for (auto& x : p) x = top ? P - 1 : u32(rng() % P);
    return p;
}

// The scalar model: leaf products, sums, the inverse.
std::vector<u32> leaf_product(const Transform& t, const std::vector<u32>& a, const std::vector<u32>& b) {
    Words x(a), y(b), z(t.length());
    t.multiply<variant::scalar>(x.data, y.data, z.data);
    return z.words();
}
std::vector<u32> sum(std::vector<u32> a, const std::vector<u32>& b) {
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = add(a[i], b[i]);
    return a;
}
std::vector<u32> coefficients(const Transform& t, const std::vector<u32>& F) {
    Words w(F);
    t.inverse<variant::scalar>(w.data);
    return w.words();
}

// out as Output o picks it from the coefficients c; out's words past count kept.
bool picked(const std::vector<u32>& out, const std::vector<u32>& c, std::size_t count, Output o) {
    for (std::size_t i = 0; i < count; ++i)
        if (out[i] != (o.negate ? neg(c[o.offset + i]) : c[o.offset + i])) return false;
    return std::all_of(out.begin() + std::ptrdiff_t(count), out.end(), [](u32 x) { return x == kSentinel; });
}

template <class V>
void test_length(const std::string& name, int lg) {
    const Transform t(lg);
    const std::size_t len = t.length();
    for (bool top : {false, true}) {
        const std::vector<u32> a1 = random_form(len, top), a2 = random_form(len, top), a3 = random_form(len, top),
                               b0 = random_form(len, top), b1 = random_form(len, false);
        const std::vector<u32> lower = coefficients(t, sum(leaf_product(t, a2, b0), leaf_product(t, a1, b1)));
        const std::vector<u32> side = sum(leaf_product(t, a3, b0), leaf_product(t, a2, b1));
        // inverse_product of a product and a term alone, both orders: a1 b1 + a3.
        const std::vector<u32> alone = coefficients(t, sum(leaf_product(t, a1, b1), a3));
        for (std::size_t offset : {std::size_t(0), len / 2}) {
            for (bool negate : {false, true}) {
                const Output o{offset, negate};
                for (std::size_t count : {std::size_t(0), std::size_t(1), len / 2 - 3, len / 2, len - offset}) {
                    if (count > len - offset || (lg > kThoroughMaxLog && count < len / 2 - 3)) continue;
                    const std::string what = name + (top ? " P - 1" : "") + " offset " + std::to_string(offset) +
                                             (negate ? " negated" : "") + " count " + std::to_string(count);
                    Words A1(a1), A2(a2), A3(a3), B0(b0), B1(b1), work(len);
                    std::vector<u32> out(count + 4, kSentinel);
                    t.toeplitz_product<V>(A1.data, A2.data, A3.data, B0.data, B1.data, work.data,
                                          std::span<u32>(out.data(), count), o);
                    expect(picked(out, lower, count, o), what + " toeplitz_product: out");
                    const bool written = count > 0;  // an empty out returns before the products
                    expect(A3.words() == (written ? side : a3), what + " toeplitz_product: A3");
                    expect(A1.words() == a1 && A2.words() == a2 && B0.words() == b0 && B1.words() == b1,
                           what + " toeplitz_product: inputs kept");
                    expect(A1.padding_kept() && A2.padding_kept() && A3.padding_kept() && B0.padding_kept() &&
                               B1.padding_kept() && work.padding_kept(),
                           what + " toeplitz_product: padding kept");
                    for (bool first_alone : {false, true}) {
                        const Term product{A1.data, B1.data}, single{A3.data, nullptr};
                        const Term terms[] = {first_alone ? single : product, first_alone ? product : single};
                        std::copy(a3.begin(), a3.end(), A3.data);
                        std::vector<u32> out2(count + 4, kSentinel);
                        t.inverse_product<V>(terms, work.data, std::span<u32>(out2.data(), count), o);
                        expect(picked(out2, alone, count, o), what + " inverse_product, a product and a term alone" +
                                                                  (first_alone ? " (alone first)" : ""));
                    }
                }
            }
        }
    }
}

template <class V>
void test_variant(const std::string& name, int max_lg) {
    for (int lg = kMinLog; lg <= max_lg; ++lg) test_length<V>(name + " lg " + std::to_string(lg), lg);
}

}  // namespace

int main() {
    test_variant<variant::scalar>("scalar", kScalarMaxLog);
    fps::simd::for_each(fps::simd::All{}, []<class S> {
        test_variant<variant::intrinsics<S>>(std::string("intrinsics ") + S::name, S::lanes == 1 ? kScalarMaxLog : kMaxTestLog);
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
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
