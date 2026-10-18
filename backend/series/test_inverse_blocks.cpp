// Tests of series::inverse by blocks (issue #214 r2: the blocked division with N = 1,
// detail::inverse_blocks) and of the choice between Newton, the ragged chain and blocks: the
// blocks against the O(n^2) recurrence at every n from 65 to 700 (3, 4 and 5 blocks, last blocks
// direct, low and full) and around block edges to 2^12 (by f g = 1 past 2^11); inverse() at the
// edges of each method; f shorter than n (inside a block, at a block edge), mostly zero and all
// P - 1; identities at large sizes; the caller's scratch.
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <vector>

#include "backend/ntt/ntt.hpp"
#include "backend/series/series.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
namespace d = fps::series::detail;
constexpr u32 P = 998244353;

#if defined(__SANITIZE_ADDRESS__)
constexpr bool kSanitized = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
constexpr bool kSanitized = true;
#else
constexpr bool kSanitized = false;
#endif
#else
constexpr bool kSanitized = false;
#endif

int failures = 0;

void check(bool ok, const char* what, int line, u64 a = 0, u64 b = 0) {
    if (ok) return;
    if (++failures <= 20)
        std::printf("FAIL line %d: %s (%llu, %llu)\n", line, what, (unsigned long long)a, (unsigned long long)b);
}
#define CHECK(cond, ...) check(bool(cond), #cond, __LINE__ __VA_OPT__(, ) __VA_ARGS__)

std::mt19937_64 rng(2142);

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
u32 at(const Vec& f, std::size_t i) { return i < f.size() ? f[i] : 0; }

// Kinds: 0 random, 1 mostly zeros, 2 all P - 1; f[0] != 0.
Vec random_vec(std::size_t n, int kind) {
    Vec a(n);
    for (u32& c : a) {
        const u64 r = rng();
        c = kind == 2 ? P - 1 : kind == 1 && r % 4 ? 0 : u32(r % P);
    }
    if (n) a[0] = kind == 2 ? P - 1 : 1 + u32(rng() % (P - 1));
    return a;
}

Vec ref_inverse(const Vec& f, std::size_t n) {
    Vec g(n);
    const u32 c = power(f[0], P - 2);
    for (std::size_t i = 0; i < n; ++i) {
        u64 s = i == 0;
        for (std::size_t j = 1; j <= i && j < f.size(); ++j) s = (s + P - u64(f[j]) * g[i - j] % P) % P;
        g[i] = mul(u32(s), c);
    }
    return g;
}

// inverse_blocks itself, in an arena of its words with garbage, f of any length (mod x^n read).
Vec run_blocks(const Vec& f, std::size_t n) {
    Vec memory(d::inverse_block_words(n) + d::kAlignSlack, P - 1), g(n, P - 1);
    const std::span<const u32> fs(f.data(), std::min(f.size(), n));
    d::inverse_blocks(fs, g, d::Arena(memory.data(), memory.size()));
    return g;
}

Vec run_inverse(const Vec& f, std::size_t n) {
    Vec g(n, P - 1);
    fps::series::inverse(f, g);
    return g;
}

// Whether g = 1 / f mod x^n (f mod x^n read): f g = 1 mod x^n, which only the inverse satisfies.
bool is_inverse(const Vec& f, const Vec& g) {
    const std::size_t n = g.size();
    Vec fg(n);
    fps::ntt::multiply(std::span<const u32>(f.data(), std::min(f.size(), n)), g, fg);
    return fg[0] == 1 && std::all_of(fg.begin() + 1, fg.end(), [](u32 c) { return c == 0; });
}

// Every n from 65 to 700: 3 blocks of 32 to 4 of 128, and from 257 on 5 blocks of 64 and 3 of 128:
// last blocks direct (to 16 words), low and full.
void test_every_size() {
    for (std::size_t n = 65; n <= 700; ++n) {
        const Vec f = random_vec(n, int(n % 3));
        CHECK(run_blocks(f, n) == ref_inverse(f, n), n);
    }
}

// Around the block edges to 2^12 (each shape's last block 1, 16, 17 words and full; f shorter):
// against the recurrence to 2^11, by f g = 1 beyond.
void test_edges() {
    for (int k = 7; k <= 12; ++k) {
        const std::size_t L = std::size_t(1) << k;
        std::vector<std::size_t> sizes;
        for (std::size_t r : {L / 32, L / 8, L / 4, L / 2 - 1, L / 2, L / 2 + 1, L - L / 4, L - 1})
            sizes.push_back(L + r);
        for (std::size_t b : {L / 4, L / 2})  // block edges of 5 blocks of L/4 and 3 or 4 of L/2
            for (std::size_t edge = L + b; edge < 2 * L; edge += b)
                for (std::size_t delta : {std::size_t(1), std::size_t(16), std::size_t(17)}) sizes.push_back(edge - b + delta);
        for (std::size_t n : sizes) {
            if (n <= L || n >= 2 * L) continue;
            const bool small = n <= 2048;
            const auto ok = [small, n](const Vec& f, const Vec& g) { return small ? g == ref_inverse(f, n) : is_inverse(f, g); };
            for (int kind = 0; kind < 3; ++kind) {
                const Vec f = random_vec(n, kind);
                CHECK(ok(f, run_blocks(f, n)), n, kind);
                CHECK(ok(f, run_inverse(f, n)), n, kind);
                // f shorter: inside the last block, at a block edge, below the first block's end
                const d::DivisionShape s(n);
                for (std::size_t fsize : {n - 5, (s.blocks - 1) * s.k, s.k + 3, s.k / 2}) {
                    const Vec fs(f.begin(), f.begin() + std::ptrdiff_t(fsize));
                    CHECK(ok(fs, run_blocks(fs, n)), n, fsize);
                    CHECK(ok(fs, run_inverse(fs, n)), n, fsize);
                }
            }
        }
    }
}

// inverse() at the edges of its methods: the chain to chain_up_to(L) / 32, blocks to 16 / 32, Newton
// past it and at powers of two, from L = 64 on.
void test_methods() {
    using M = d::InverseMethod;
    CHECK(d::inverse_method(32) == M::direct);
    CHECK(d::inverse_method(64) == M::newton);
    CHECK(d::inverse_method(1 << 20) == M::newton);
    for (int k = 6; k <= 12; ++k) {
        const std::size_t L = std::size_t(1) << k, chain = L * d::chain_up_to(L) / 32, blocks = L * d::kBlocksUpTo / 32;
        CHECK(d::inverse_method(L + 1) == M::chain, L);
        CHECK(d::inverse_method(L + chain) == M::chain, L);
        CHECK(d::inverse_method(L + chain + 1) == M::blocks, L);
        CHECK(d::inverse_method(L + blocks) == M::blocks, L);
        CHECK(d::inverse_method(L + blocks + 1) == M::newton, L);
        CHECK(d::inverse_method(2 * L - 1) == M::newton, L);
        for (std::size_t n : {L + chain, L + chain + 1, L + blocks, L + blocks + 1}) {
            const Vec f = random_vec(n, 0);
            CHECK(run_inverse(f, n) == ref_inverse(f, n), n);
        }
    }
}

// f g = 1 mod x^n at sizes too large for the reference.
void test_identities() {
    std::vector<std::size_t> sizes = {16384 + 4096, 16384 + 8192, 16384 + 8193, 16384 + 3000};
    if (!kSanitized) {
        const std::size_t K = std::size_t(1) << 18;
        for (std::size_t n : {K + K / 4, K + 6 * K / 32 + 1, K + 9 * K / 32, K + K / 2, 2 * K + K / 2, 4 * K + 7 * K / 32})
            sizes.push_back(n);
    }
    for (std::size_t n : sizes) {
        const Vec f = random_vec(n, 0);
        const Vec g = run_inverse(f, n);
        Vec fg(n), one(n);
        one[0] = 1;
        fps::ntt::multiply(f, g, fg);
        CHECK(fg == one, n);
        CHECK(run_blocks(f, n) == g, n);
    }
}

// The caller's scratch, filled with garbage, at offsets; words past it unwritten.
void test_scratch() {
    for (std::size_t n : {97ul, 160ul, 330ul, 1300ul, 1536ul, 5000ul, 6144ul}) {
        for (std::size_t offset = 0; offset < 16; offset += 5) {
            const Vec f = random_vec(n, 0);
            const std::size_t words = fps::series::inverse_scratch(n);
            Vec scratch(words + offset + 8, P - 1);
            Vec g(n);
            fps::series::inverse(f, g, std::span<u32>(scratch.data() + offset, words));
            CHECK(g == ref_inverse(f, n), n, offset);
            CHECK(std::all_of(scratch.end() - 8, scratch.end(), [](u32 c) { return c == P - 1; }), n, offset);
        }
    }
}

}  // namespace

int main() {
    test_every_size();
    test_edges();
    test_methods();
    test_identities();
    test_scratch();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("series inverse blocks: all tests passed\n");
    return 0;
}
