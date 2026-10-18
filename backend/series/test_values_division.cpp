// Tests of the blocked division in ntt's values form (issue #251), forced on for every block count
// (FPS_SERIES_VALUES below), so every target runs the path its default may not take: inverse by
// blocks, log and pow against O(n^2) recurrences at every n from 65 to 700 (3, 4 and 5 blocks; last
// blocks direct, low and full) and around block edges to 2^12 (f g = 1, f g' = f' and pow's
// exponent law past 2^11), f shorter than n, random, mostly zero and all P - 1 inputs; identities
// at large sizes; the caller's scratch, garbage in it, words past it unwritten.
#define FPS_SERIES_VALUES 0x3c  // blocks 2 .. 5
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
constexpr u32 kE = 0x2545f491u % P, kC = 0x3c6ef372u % P;  // pow's exponent and constant

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

std::mt19937_64 rng(251);

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
u32 inv(u32 a) { return power(a, P - 2); }
u32 at(const Vec& f, std::size_t i) { return i < f.size() ? f[i] : 0; }

// Kinds: 0 random, 1 mostly zeros, 2 all P - 1; then f[0] = first.
Vec random_vec(std::size_t n, int kind, u32 first) {
    Vec a(n);
    for (u32& c : a) {
        const u64 r = rng();
        c = kind == 2 ? P - 1 : kind == 1 && r % 4 ? 0 : u32(r % P);
    }
    if (n) a[0] = first;
    return a;
}

// The references, O(n^2), on f mod x^n.
Vec ref_inverse(const Vec& f, std::size_t n) {
    Vec g(n);
    const u32 c = inv(f[0]);
    for (std::size_t i = 0; i < n; ++i) {
        u64 s = i == 0;
        for (std::size_t j = 1; j <= i && j < f.size(); ++j) s = (s + P - u64(f[j]) * g[i - j] % P) % P;
        g[i] = mul(u32(s), c);
    }
    return g;
}
Vec ref_log(const Vec& f, std::size_t n) {  // f g' = f' with G = i g_i
    Vec G(n), g(n);
    for (std::size_t i = 1; i < n; ++i) {
        u64 s = mul(u32(i), at(f, i));
        for (std::size_t k = 1; k < i; ++k) s = (s + P - u64(G[k]) * at(f, i - k) % P) % P;
        G[i] = u32(s);
        g[i] = mul(G[i], inv(u32(i)));
    }
    return g;
}
Vec ref_pow(const Vec& f, u32 e, u32 c, std::size_t n) {  // f g' = e f' g, g[0] = c
    Vec g(n);
    g[0] = c;
    const u32 f0 = inv(at(f, 0));
    for (std::size_t i = 1; i < n; ++i) {
        u64 s = 0;
        for (std::size_t j = 1; j <= i && j < f.size(); ++j) {
            const u32 w = u32((u64(e) * j % P + P - (i - j) % P) % P);
            s = (s + u64(mul(w, f[j])) * g[i - j]) % P;
        }
        g[i] = mul(mul(u32(s), f0), inv(u32(i)));
    }
    return g;
}

// inverse_blocks itself, in an arena of its words with garbage, f of any length (mod x^n read).
Vec run_blocks(const Vec& f, std::size_t n) {
    Vec memory(d::inverse_block_words(n) + d::kAlignSlack, P - 1), g(n, P - 1);
    d::inverse_blocks(std::span<const u32>(f.data(), std::min(f.size(), n)), g, d::Arena(memory.data(), memory.size()));
    return g;
}
Vec run_log(const Vec& f, std::size_t n) {
    Vec g(n, P - 1);
    fps::series::log(f, g);
    return g;
}
Vec run_pow(const Vec& f, std::size_t n, u32 e = kE, u32 c = kC) {
    Vec g(n, P - 1);
    fps::series::pow(f, e, c, g);
    return g;
}

Vec product(const Vec& a, const Vec& b, std::size_t n) {
    Vec c(n);
    fps::ntt::multiply(std::span<const u32>(a.data(), std::min(a.size(), n)), std::span<const u32>(b.data(), std::min(b.size(), n)), c);
    return c;
}
bool is_inverse(const Vec& f, const Vec& g) {
    const Vec fg = product(f, g, g.size());
    return fg[0] == 1 && std::all_of(fg.begin() + 1, fg.end(), [](u32 c) { return c == 0; });
}
bool is_log(const Vec& f, const Vec& g) {  // f g' = f' mod x^(n-1), g[0] = 0
    const std::size_t n = g.size();
    Vec dg(n - 1);
    for (std::size_t i = 1; i < n; ++i) dg[i - 1] = mul(u32(i), g[i]);
    const Vec fdg = product(f, dg, n - 1);
    for (std::size_t i = 0; i + 1 < n; ++i)
        if (fdg[i] != mul(u32(i + 1), at(f, i + 1))) return false;
    return g[0] == 0;
}
bool is_pow(const Vec& f, const Vec& g) {  // pow(f, a, 1) pow(f, b, 1) = pow(f, a + b, 1), a + b = kE
    const std::size_t n = g.size();
    const u32 a = 12345, b = (kE + P - a) % P;
    return product(run_pow(f, n, a, 1), run_pow(f, n, b, 1), n) == run_pow(f, n, kE, 1) &&
           std::equal(g.begin(), g.end(), run_pow(f, n, kE, 1).begin(), [](u32 x, u32 y) { return x == mul(y, kC); });
}

// The forced mask is in effect: every block count from 2 runs in the values form (pow: where its
// kept transforms are not exp's).
void test_shapes() {
    for (std::size_t n : {100ul, 300ul, 1000ul, 1300ul, 1600ul, 2000ul, 5000ul})
        CHECK(d::DivisionShape(n).values, n, d::DivisionShape(n).blocks);
}

// Every n from 65 to 700: 3 blocks of 32 to 4 of 128, and from 257 on 5 blocks of 64 and 3 of 128:
// last blocks direct (to 16 words), low and full.
void test_every_size() {
    for (std::size_t n = 65; n <= 700; ++n) {
        const int kind = int(n % 3);
        const Vec f = random_vec(n, kind, kind == 2 ? P - 1 : 1 + u32(rng() % (P - 1)));
        CHECK(run_blocks(f, n) == ref_inverse(f, n), n);
        if (n % 2 == 0) CHECK(run_pow(f, n) == ref_pow(f, kE, kC, n), n);
        if (n % 2 == 1) {
            const Vec l = random_vec(n, kind, 1);
            CHECK(run_log(l, n) == ref_log(l, n), n);
        }
    }
}

// Around the block edges to 2^12 (each shape's last block 1, 16, 17 words and full; f shorter):
// against the recurrences to 2^11, by the identities beyond.
void test_edges() {
    for (int k = 7; k <= 12; ++k) {
        const std::size_t L = std::size_t(1) << k;
        std::vector<std::size_t> sizes;
        for (std::size_t r : {std::size_t(1), L / 32, L / 8, L / 4, L / 4 + 1, L / 2 - 1, L / 2, L / 2 + 1, L - L / 4, L - 1})
            sizes.push_back(L + r);
        for (std::size_t b : {L / 4, L / 2})  // block edges of 5 blocks of L/4 and 3 or 4 of L/2
            for (std::size_t edge = L + b; edge < 2 * L; edge += b)
                for (std::size_t delta : {std::size_t(1), std::size_t(16), std::size_t(17)}) sizes.push_back(edge - b + delta);
        for (std::size_t n : sizes) {
            if (n <= L || n >= 2 * L) continue;
            const bool small = n <= 2048;
            for (int kind = 0; kind < 3; ++kind) {
                if (!small && kind == 1) continue;
                const Vec f = random_vec(n, kind, kind == 2 ? P - 1 : 3);
                const Vec l = random_vec(n, kind, 1);
                const d::DivisionShape s(n);
                for (std::size_t fsize : {n, n - 5, (s.blocks - 1) * s.k, s.k + 3, s.k / 2}) {
                    const Vec fs(f.begin(), f.begin() + std::ptrdiff_t(fsize)), ls(l.begin(), l.begin() + std::ptrdiff_t(fsize));
                    const Vec g = run_blocks(fs, n), lg = run_log(ls, n), pg = run_pow(fs, n);
                    CHECK(small ? g == ref_inverse(fs, n) : is_inverse(fs, g), n, fsize);
                    CHECK(small ? lg == ref_log(ls, n) : is_log(ls, lg), n, fsize);
                    CHECK(small ? pg == ref_pow(fs, kE, kC, n) : is_pow(fs, pg), n, fsize);
                }
            }
        }
    }
}

// Identities at sizes too large for the references, with inverse() and the blocks equal.
void test_identities() {
    std::vector<std::size_t> sizes = {16384 + 4096, 16384 + 8192, 16384 + 8193, 16384 + 3000, 32768 - 77};
    if (!kSanitized) {
        const std::size_t K = std::size_t(1) << 18;
        for (std::size_t n : {K + 1, K + K / 4, K + 6 * K / 32 + 1, K + K / 2, K + 3 * K / 4, 2 * K + K / 2, 4 * K + 7 * K / 32})
            sizes.push_back(n);
    }
    for (std::size_t n : sizes) {
        const Vec f = random_vec(n, 0, 7), l = random_vec(n, 0, 1);
        const Vec g = run_blocks(f, n);
        CHECK(is_inverse(f, g), n);
        Vec h(n);
        fps::series::inverse(f, h);
        CHECK(h == g, n);
        CHECK(is_log(l, run_log(l, n)), n);
        CHECK(is_pow(f, run_pow(f, n)), n);
    }
}

// The caller's scratch, filled with garbage, at offsets; words past it unwritten.
void test_scratch() {
    for (std::size_t n : {97ul, 160ul, 330ul, 1300ul, 1536ul, 1700ul, 5000ul, 6144ul}) {
        for (std::size_t offset = 0; offset < 16; offset += 5) {
            const Vec f = random_vec(n, 0, 9), l = random_vec(n, 0, 1);
            const auto ok_past = [](const Vec& s) { return std::all_of(s.end() - 8, s.end(), [](u32 c) { return c == P - 1; }); };
            {
                const std::size_t words = fps::series::inverse_scratch(n);
                Vec scratch(words + offset + 8, P - 1), g(n);
                fps::series::inverse(f, g, std::span<u32>(scratch.data() + offset, words));
                CHECK(g == ref_inverse(f, n), n, offset);
                CHECK(ok_past(scratch), n, offset);
            }
            {
                const std::size_t words = fps::series::log_scratch(n);
                Vec scratch(words + offset + 8, P - 1), g(n);
                fps::series::log(l, g, std::span<u32>(scratch.data() + offset, words));
                CHECK(g == ref_log(l, n), n, offset);
                CHECK(ok_past(scratch), n, offset);
            }
            {
                const std::size_t words = fps::series::pow_scratch(n);
                Vec scratch(words + offset + 8, P - 1), g(n);
                fps::series::pow(f, kE, kC, g, std::span<u32>(scratch.data() + offset, words));
                CHECK(g == ref_pow(f, kE, kC, n), n, offset);
                CHECK(ok_past(scratch), n, offset);
            }
        }
    }
}

}  // namespace

int main() {
    test_shapes();
    test_every_size();
    test_edges();
    test_identities();
    test_scratch();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("series values division: all tests passed\n");
    return 0;
}
