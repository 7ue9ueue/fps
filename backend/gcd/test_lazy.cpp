// Tests of backend/gcd's lazy Euclid words (issue #157 round 2): a step's vector pass without its
// last reduction (step_vector<S, kLazyEuclid>) keeps words below B = P / (1 - 3 P / 2^32) from words
// below B, with factors up to P - 1, and gives the residue of (x u + y v1 + z v) kScale; its scalar
// twin combine3_word and reduce_words give canonical words; shifted_up is X v from aligned vectors.
// Random words and the edges (0, 1, P - 1, P, 2P - 1, 2P, 3P, B - 1), on simd::Native's lanes.
// tools/kernel.py builds it at -O2 and with ASan/UBSan.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "backend/gcd/gcd.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
constexpr u32 P = fps::half_gcd::kP;
namespace gd = fps::half_gcd::detail;
using S = gd::S;
constexpr std::size_t L = gd::kLanes;
constexpr u32 kB = 3297426516u;  // ceil(P / (1 - 3 P / 2^32)): Euclid's words stay below it

int failures = 0, checks = 0;
std::mt19937_64 rng(1572);

void fail(const std::string& what) {
    if (++failures <= 20) std::printf("FAIL %s\n", what.c_str());
}

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
// The passes' scale: 2^-32 where they reduce by Montgomery, else 1.
const u32 kScale = gd::kWide ? power(power(2, 32), P - 2) : 1;

constexpr u32 kEdges[] = {0, 1, P - 1, P, 2 * P - 1, 2 * P, 3 * P, kB - 1};
// A word below kB (kLazy) or P.
template <bool kLazy>
u32 word() {
    if constexpr (!kLazy) return rng() % 4 ? u32(rng() % P) : (rng() % 2 ? 0 : P - 1);
    return rng() % 4 ? u32(rng() % kB) : kEdges[rng() % std::size(kEdges)];
}
u32 factor() {
    constexpr u32 edges[] = {0, 1, P - 1, P - 2};
    return rng() % 4 ? u32(rng() % P) : edges[rng() % std::size(edges)];
}

// The factors of a Step set directly (Shoup quotients for the targets without 64-bit lane products).
gd::Step step(u32 x, u32 y, u32 z) {
    gd::Step s(1, 0, 0, 0);
    s.x = x, s.y = y, s.z = z;
    if constexpr (!gd::kWide) s.xq = gd::M::shoup_quotient(x), s.yq = gd::M::shoup_quotient(y), s.zq = gd::M::shoup_quotient(z);
    return s;
}

u32 expected(const gd::Step& s, u32 u, u32 v1, u32 v) {
    return mul(u32((u64(s.x) * (u % P) + u64(s.y) * (v1 % P) + u64(s.z) * (v % P)) % P), kScale);
}

// The bound of step_vector<S, kLazy>'s words: kB where it leaves them unreduced.
template <bool kLazy>
constexpr u32 kOut = kLazy && gd::kWide ? kB : P;

template <bool kLazy>
void vectors() {
    const std::string tag = kLazy ? "lazy " : "";
    for (int round = 0; round < 20000; ++round) {
        const u32 top = kLazy ? kB - 1 : P - 1;
        const gd::Step s = round < 8 ? step(P - 1, P - 1, P - 1) : step(factor(), factor(), factor());
        // v1 = X v: its lanes are v's one lane up (x86 takes X v's odd lanes from v's even ones).
        alignas(64) u32 u[L], v1[L], v[L], out[L];
        for (std::size_t i = 0; i < L; ++i) u[i] = round < 8 ? top : word<kLazy>(), v[i] = round < 8 ? top : word<kLazy>();
        v1[0] = round < 8 ? top : word<kLazy>();
        for (std::size_t i = 1; i < L; ++i) v1[i] = v[i - 1];
        S::store(out, gd::step_vector<S, kLazy>(S::load(u), S::load(v1), S::load(v), s));
        for (std::size_t i = 0; i < L; ++i) {
            ++checks;
            const u32 want = expected(s, u[i], v1[i], v[i]);
            if (out[i] >= kOut<kLazy> || out[i] % P != want)
                fail(tag + "step_vector lane " + std::to_string(i) + ": " + std::to_string(out[i]) + ", want " + std::to_string(want));
            const u32 w = gd::combine3_word<kLazy>(s, u[i], v1[i], v[i]);
            if (w != want) fail(tag + "combine3_word: " + std::to_string(w) + ", want " + std::to_string(want));
        }
    }
}

// Steps in a row on the same words, the outputs fed back: the bound holds along a chain.
template <bool kLazy>
void chains() {
    for (int round = 0; round < 200; ++round) {
        alignas(64) u32 c[L], d[L], e[L], d1[L];
        for (std::size_t i = 0; i < L; ++i) c[i] = word<kLazy>(), d[i] = word<kLazy>();
        for (int t = 0; t < 200; ++t) {
            const gd::Step s = step(factor(), factor(), factor());
            d1[0] = word<kLazy>();
            for (std::size_t i = 1; i < L; ++i) d1[i] = d[i - 1];
            S::store(e, gd::step_vector<S, kLazy>(S::load(c), S::load(d1), S::load(d), s));
            for (std::size_t i = 0; i < L; ++i) {
                ++checks;
                if (e[i] >= kOut<kLazy> || e[i] % P != expected(s, c[i], d1[i], d[i]))
                    fail(std::string(kLazy ? "lazy " : "") + "chain step " + std::to_string(t));
            }
            std::copy_n(d, L, c), std::copy_n(e, L, d);
        }
    }
}

void shifts() {
    for (int round = 0; round < 1000; ++round) {
        alignas(64) u32 w[2 * L], out[L];
        for (auto& x : w) x = u32(rng());
        S::store(out, gd::shifted_up(S::load(w + L), S::load(w)));
        for (std::size_t i = 0; i < L; ++i) {
            ++checks;
            if (out[i] != w[L - 1 + i]) fail("shifted_up lane " + std::to_string(i));
        }
    }
}

void reductions() {
    std::vector<u32> a(8 * L + 2 * L);
    for (int round = 0; round < 1000; ++round) {
        for (auto& x : a) x = word<true>();
        std::vector<u32> b = a;
        const std::ptrdiff_t from = std::ptrdiff_t(rng() % (4 * L)), to = from + std::ptrdiff_t(rng() % (4 * L));
        gd::reduce_words<true>(b.data(), from, to);
        for (std::ptrdiff_t j = 0; j < std::ptrdiff_t(a.size()); ++j) {
            ++checks;
            const u32 x = a[std::size_t(j)], y = b[std::size_t(j)];
            if (y % P != x % P) fail("reduce_words changed a residue at " + std::to_string(j));
            if (j >= from && j < to && y >= P) fail("reduce_words left " + std::to_string(y));
        }
        for (const u32 x : kEdges) {
            ++checks;
            if (gd::reduced(x) != x % P) fail("reduced(" + std::to_string(x) + ")");
        }
    }
}

}  // namespace

int main() {
    vectors<true>(), vectors<false>();
    chains<true>(), chains<false>();
    shifts();
    reductions();
    std::printf("%s: %d checks, %d failures\n", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
