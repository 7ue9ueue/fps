// Tests for Transform::forward_upper(h, Form::values) (issue #103, backend/gcd): for every variant
// this machine runs and the tuned one, from h = p mod (x^(L/2) + 1) it gives the upper half of p's
// values form, word for word, at every length from 2^(kMinLog + 1) to 2^kValuesMaxLog (fewer under
// ASan), on random words, all P - 1 and zeros; Form::leaves is forward_upper(h); past
// 2^kValuesMaxLog it throws.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/ntt/ntt.hpp"

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
constexpr int kScalarMaxLog = 16;

std::mt19937_64 rng(103);
int failures = 0;

void expect(bool ok, const std::string& what, u64 detail) {
    if (ok) return;
    if (++failures <= 20) std::printf("FAIL: %s (%llu)\n", what.c_str(), static_cast<unsigned long long>(detail));
}

// count words, 32-byte aligned, then kPadding words.
struct Words {
    struct alignas(32) Lanes {
        u32 lane[8];
    };
    explicit Words(std::size_t count) : count(count), data((count + kPadding) / 8 + 1) {}
    u32* begin() { return data.data()->lane; }
    std::vector<u32> vector() { return {begin(), begin() + count}; }
    std::size_t count;
    std::vector<Lanes> data;
};

std::vector<u32> random_words(std::size_t n, int kind) {
    std::vector<u32> p(n);
    for (u32& x : p) x = kind == 0 ? u32(rng() % P) : kind == 1 ? P - 1 : 0;
    return p;
}

template <class V>
void test_variant(const std::string& name, int max_lg) {
    for (int lg = kMinLog + 1; lg <= max_lg; ++lg) {
        const Transform t(lg);
        const std::size_t len = t.length(), half = len / 2;
        for (int kind = 0; kind < 3; ++kind) {
            if (kind && lg > 12) break;
            const std::vector<u32> p = random_words(len, kind);
            Words full(len), h(half), leaves(half);
            std::copy(p.begin(), p.end(), full.begin());
            t.forward<variant::scalar>(full.begin(), Form::values);
            for (std::size_t i = 0; i < half; ++i) h.begin()[i] = leaves.begin()[i] = (p[i] + P - p[i + half]) % P;
            t.forward_upper<V>(h.begin(), Form::values);
            expect(std::equal(full.begin() + half, full.begin() + len, h.begin()), name + " values", u64(lg) * 10 + u64(kind));
            Words copy(half);
            std::copy_n(leaves.begin(), half, copy.begin());
            t.forward_upper<V>(leaves.begin(), Form::leaves);
            t.forward_upper<V>(copy.begin());
            expect(leaves.vector() == copy.vector(), name + " Form::leaves", u64(lg));
        }
    }
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
    test_variant<variant::tuned>("tuned", kMaxTestLog);
    bool threw = false;
    try {
        Words h(std::size_t(1) << 23);
        Transform(kValuesMaxLog + 1).forward_upper(h.begin(), Form::values);
    } catch (const std::length_error&) {
        threw = true;
    }
    expect(threw, "forward_upper(h, Form::values) past 2^kValuesMaxLog throws", 0);
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
