// Tests the factorials kernel: the model against the definition (fact[i] = fact[i - 1] i,
// fact[i] inverse[i] = 1), then every Chains variant this machine can run (every simd struct,
// K = 2, 4, 8) and the selected `tables` against the model. Sizes: every n up to 1100, powers of
// two and their neighbours up to 2^20, both sides of every change of the lane length C up to
// 2^14, random n, and n = P - 1, P for small P. Output words past n must stay untouched.
// Exits nonzero on a failure.
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "backend/factorials/factorials.hpp"
#include "backend/simd/select.hpp"

namespace {

using fps::factorials::u32;
using fps::factorials::u64;

constexpr u32 kGuard = 0xdeadbeef;
constexpr std::size_t kGuardWords = 16;
int failures = 0;

void fail(const std::string& what) {
    if (++failures <= 20) std::printf("FAIL %s\n", what.c_str());
}

// Runs f(fact, inverse, n) on buffers with guard words after n and compares with the model.
template <u32 P, class F>
void check(const std::string& name, std::size_t n, F f) {
    std::vector<u32> want_fact(n), want_inverse(n);
    fps::factorials::model::tables<P>(want_fact.data(), want_inverse.data(), n);
    std::vector<u32> fact(n + kGuardWords, kGuard), inverse(n + kGuardWords, kGuard);
    f(fact.data(), inverse.data(), n);
    const std::string where = name + " P=" + std::to_string(P) + " n=" + std::to_string(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (fact[i] != want_fact[i] || inverse[i] != want_inverse[i]) {
            fail(where + " i=" + std::to_string(i) + ": " + std::to_string(fact[i]) + ", " +
                 std::to_string(inverse[i]) + "; want " + std::to_string(want_fact[i]) + ", " +
                 std::to_string(want_inverse[i]));
            return;
        }
    }
    for (std::size_t i = n; i < n + kGuardWords; ++i)
        if (fact[i] != kGuard || inverse[i] != kGuard) return fail(where + ": wrote past n");
}

// The model against the definition.
template <u32 P>
void model(std::size_t n) {
    std::vector<u32> fact(n), inverse(n);
    fps::factorials::model::tables<P>(fact.data(), inverse.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        const u64 want = i == 0 ? 1 : u64(fact[i - 1]) * i % P;
        if (fact[i] != want || u64(fact[i]) * inverse[i] % P != 1 || inverse[i] >= P)
            return fail("model P=" + std::to_string(P) + " n=" + std::to_string(n) + " i=" + std::to_string(i));
    }
}

std::vector<std::size_t> sizes(std::size_t limit, std::size_t min) {
    std::set<std::size_t> n;
    for (std::size_t i = min; i <= std::min<std::size_t>(1100, limit); ++i) n.insert(i);
    for (int k = 6; k <= 20; ++k)
        for (std::size_t m : {(std::size_t(1) << k) - 1, std::size_t(1) << k, (std::size_t(1) << k) + 1})
            if (m >= min && m <= limit) n.insert(m);
    std::mt19937_64 rng(20261011);
    for (int t = 0; t < 20; ++t) n.insert(min + rng() % (std::min<std::size_t>(limit, 1 << 18) - min + 1));
    return {n.begin(), n.end()};
}

template <class S, int K, u32 P>
void chains(std::size_t limit) {
    using C = fps::factorials::Chains<S, K, P>;
    const std::string name = std::string("Chains<") + S::name + ", " + std::to_string(K) + ">";
    const auto run = [](u32* fact, u32* inverse, std::size_t n) { C::run(fact, inverse, n); };
    const std::vector<std::size_t> base = sizes(limit, C::kMinSize);
    std::set<std::size_t> n(base.begin(), base.end());
    for (std::size_t m = C::kMinSize + 1; m <= std::min<std::size_t>(limit, 1 << 14); ++m)
        if (C::chunk(m) != C::chunk(m - 1)) n.insert({m - 1, m});
    if (limit == P) n.insert({P - 1, P});
    for (std::size_t m : n) check<P>(name, m, run);
}

template <class S>
void test_isa() {
    const int before = failures;
    chains<S, 2, 998244353>(1 << 20);
    chains<S, 4, 998244353>(1 << 20);
    chains<S, 8, 998244353>(1 << 16);
    chains<S, 2, 4099>(4099);
    chains<S, 4, 65537>(65537);
    chains<S, 4, 1073741789>(1 << 12);  // the largest prime below 2^30
    std::printf("%-9s %s\n", S::name, failures == before ? "ok" : "FAIL");
}

}  // namespace

int main() {
    for (std::size_t n : {0, 1, 2, 3, 100, 4099}) model<4099>(n);
    for (std::size_t n : {0, 1, 2, 1000, 1 << 16}) model<998244353>(n);
    fps::simd::for_each(fps::simd::All{}, []<class S> { test_isa<S>(); });
    const auto selected = [](u32* fact, u32* inverse, std::size_t n) {
        fps::factorials::tables<998244353>(std::span(fact, n), std::span(inverse, n));
    };
    for (std::size_t n : sizes(1 << 20, 0)) check<998244353>("tables", n, selected);
    const auto small = [](u32* fact, u32* inverse, std::size_t n) {
        fps::factorials::tables<4099>(std::span(fact, n), std::span(inverse, n));
    };
    for (std::size_t n : {std::size_t(0), std::size_t(4098), std::size_t(4099)}) check<4099>("tables", n, small);
    std::printf("%s\n", failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
