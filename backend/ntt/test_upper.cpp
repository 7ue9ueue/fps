// Tests for Transform::inverse_upper (issue #104): for every variant this machine runs and the
// tuned one, it undoes forward_upper (p mod (x^(L/2) + 1) from the upper half of p's leaf form),
// agrees with the scalar model word for word, and returns canonical words, at every length from
// 2^(kMinLog + 1), on random words, all P - 1 and zeros.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "backend/ntt/ntt.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using namespace fps::ntt;
constexpr u32 P = kModulus;

#if __has_feature(address_sanitizer)
constexpr int kMaxTestLog = 18;
#else
constexpr int kMaxTestLog = 22;
#endif
constexpr int kScalarMaxLog = 18;

std::mt19937_64 rng(104);
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
    explicit Words(std::size_t count) : count(count), data((count + kPadding) / 8) {}
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
            // p of len words: its leaf form's upper half, by forward of p (scalar), then back.
            const std::vector<u32> p = random_words(len, kind);
            Words leaves(len), h(half);
            std::copy(p.begin(), p.end(), leaves.begin());
            t.forward<variant::scalar>(leaves.begin());
            std::copy_n(leaves.begin() + half, half, h.begin());
            t.inverse_upper<V>(h.begin());
            bool ok = true;
            for (std::size_t i = 0; i < half; ++i) ok &= h.begin()[i] == (p[i] + P - p[i + half]) % P;
            expect(ok, name + " undoes forward_upper", u64(lg) * 10 + u64(kind));
            // Any words: the scalar model's result, canonical, and forward_upper gives them back.
            const std::vector<u32> x = random_words(half, kind);
            Words a(half), b(half);
            std::copy(x.begin(), x.end(), a.begin());
            std::copy(x.begin(), x.end(), b.begin());
            t.inverse_upper<V>(a.begin());
            if (lg <= kScalarMaxLog) {
                t.inverse_upper<variant::scalar>(b.begin());
                expect(a.vector() == b.vector(), name + " against scalar", u64(lg) * 10 + u64(kind));
            }
            expect(std::ranges::all_of(a.vector(), [](u32 w) { return w < P; }), name + " canonical", u64(lg));
            t.forward_upper<variant::scalar>(a.begin());
            expect(a.vector() == x, name + " forward_upper of it", u64(lg) * 10 + u64(kind));
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
        Words h(32);
        Transform(kMinLog).inverse_upper(h.begin());
    } catch (const std::length_error&) {
        threw = true;
    }
    expect(threw, "inverse_upper at 2^kMinLog throws", 0);
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
