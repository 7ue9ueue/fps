// Tests of backend/series at ragged n (issue #214): inverse, log, exp, sqrt and pow against O(n^2)
// recurrences at n = 2^k + 1, 2^k + 2, 1.25, 1.5 and 1.75 times 2^k and their neighbours, the edges
// of each step kind (the ragged chain's r = L/2 and L/2 + 1, the short last steps), random sizes; f
// shorter than n (inside a step's corner and below it), random, mostly zero and all P - 1 inputs;
// at large sizes against identities (f inverse(f) = 1, exp log, sqrt^2, pow against exp of e log)
// at 6e5, 2^20 + 1 and the other ragged shapes; the caller's scratch at those sizes; the helpers
// low_product and window() against the O(n m) product.
#include <algorithm>
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

std::mt19937_64 rng(214);

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
u32 inv(u32 a) { return power(a, P - 2); }
u32 at(const Vec& f, std::size_t i) { return i < f.size() ? f[i] : 0; }

// Kinds: 0 random, 1 mostly zeros, 2 all P - 1.
Vec random_vec(std::size_t n, int kind) {
    Vec a(n);
    for (u32& c : a) {
        const u64 r = rng();
        c = kind == 2 ? P - 1 : kind == 1 && r % 4 ? 0 : u32(r % P);
    }
    return a;
}
Vec with_first(Vec f, u32 c) {
    if (!f.empty()) f[0] = c;
    return f;
}

// The references, O(n^2), on f mod x^n.
Vec ref_inverse(const Vec& f, std::size_t n) {
    Vec g(n);
    const u32 c = inv(f[0]);
    for (std::size_t i = 0; i < n; ++i) {
        u64 s = i == 0;
        for (std::size_t j = 1; j <= i; ++j) s = (s + P - u64(at(f, j)) * g[i - j] % P) % P;
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
Vec ref_exp(const Vec& f, std::size_t n) {  // g' = f' g
    Vec g(n);
    if (n) g[0] = 1;
    for (std::size_t i = 1; i < n; ++i) {
        u64 s = 0;
        for (std::size_t k = 1; k <= i; ++k) s = (s + u64(mul(u32(k), at(f, k))) * g[i - k]) % P;
        g[i] = mul(u32(s), inv(u32(i)));
    }
    return g;
}
Vec ref_sqrt(const Vec& f, u32 c, std::size_t n) {  // 2 c g_i = f_i - sum_(0<j<i) g_j g_(i-j)
    Vec g(n);
    g[0] = c;
    const u32 inverse = inv(mul(2, c));
    for (std::size_t i = 1; i < n; ++i) {
        u64 s = at(f, i);
        for (std::size_t j = 1; j < i; ++j) s = (s + P - u64(g[j]) * g[i - j] % P) % P;
        g[i] = mul(u32(s), inverse);
    }
    return g;
}
Vec ref_pow(const Vec& f, u32 e, u32 c, std::size_t n) {  // f g' = e f' g, g[0] = c
    Vec g(n);
    g[0] = c;
    const u32 f0 = inv(at(f, 0));
    for (std::size_t i = 1; i < n; ++i) {
        // i f0 g_i = sum_(0<j<=i) (e j - (i - j)) f_j g_(i-j)
        u64 s = 0;
        for (std::size_t j = 1; j <= i; ++j) {
            const u32 w = u32((u64(e) * j % P + P - (i - j) % P) % P);
            s = (s + u64(mul(w, at(f, j))) * g[i - j]) % P;
        }
        g[i] = mul(mul(u32(s), f0), inv(u32(i)));
    }
    return g;
}

Vec run_inverse(const Vec& f, std::size_t n) {
    Vec g(n, P - 1);
    fps::series::inverse(f, g);
    return g;
}
Vec run_log(const Vec& f, std::size_t n) {
    Vec g(n, P - 1);
    fps::series::log(f, g);
    return g;
}
Vec run_exp(const Vec& f, std::size_t n) {
    Vec g(n, P - 1);
    fps::series::exp(f, g);
    return g;
}
Vec run_sqrt(const Vec& f, u32 c, std::size_t n) {
    Vec g(n, P - 1);
    fps::series::sqrt(f, c, g);
    return g;
}
Vec run_pow(const Vec& f, u32 e, u32 c, std::size_t n) {
    Vec g(n, P - 1);
    fps::series::pow(f, e, c, g);
    return g;
}

// f g mod x^n.
Vec product(const Vec& f, const Vec& g, std::size_t n) {
    Vec out(n);
    fps::ntt::multiply(f, g, out);
    return out;
}

// Sizes around each step kind: 2^k + 1, 2^k + 2, the chain's edges r = L/2 and L/2 + 1 (and r = L/4,
// L/4 + 1), 1.25, 1.5, 1.75 2^k, 2^(k+1) - 1, 3 2^k / 2 +- 1 (exp's and sqrt's blocks).
std::vector<std::size_t> ragged_sizes(int lo, int hi) {
    std::vector<std::size_t> sizes;
    for (int k = lo; k <= hi; ++k) {
        const std::size_t L = std::size_t(1) << k;
        for (std::size_t n : {L + 1, L + 2, L + 3, L + L / 8, L + L / 8 + 1, L + L / 4 - 1, L + L / 4, L + L / 4 + 1,
                              L + L / 2 - 1, L + L / 2, L + L / 2 + 1, 2 * L - L / 4, 2 * L - L / 4 + 1, 2 * L - 1})
            sizes.push_back(n);
    }
    return sizes;
}

void test_references() {
    std::vector<std::size_t> sizes = ragged_sizes(5, 10);
    for (int i = 0; i < 16; ++i) sizes.push_back(65 + rng() % 2500);
    sizes.insert(sizes.end(), {2049, 2050, 2561, 3072, 3073, 3200, 3201, 3500});
    for (std::size_t n : sizes) {
        const bool large = n > 1100;
        for (int kind = 0; kind < (large ? 1 : 3); ++kind) {
            // f of n words, shorter (inside the top step's f_hi, at L, and below), longer
            const std::size_t L = std::bit_floor(n - 1);
            for (std::size_t fsize : {n, n + 3, L + (n - L) / 2, L, L - 3, n / 3}) {
                if (large && fsize != n && fsize != L + (n - L) / 2) continue;
                const Vec f = random_vec(fsize, kind);
                const Vec fi = with_first(f, kind == 2 ? P - 1 : 1 + u32(rng() % (P - 1)));
                CHECK(run_inverse(fi, n) == ref_inverse(fi, n), n, fsize);
                const Vec fl = with_first(f, 1);
                CHECK(run_log(fl, n) == ref_log(fl, n), n, fsize);
                const Vec fe = with_first(f, 0);
                CHECK(run_exp(fe, n) == ref_exp(fe, n), n, fsize);
                const Vec fs = with_first(f, 4);
                CHECK(run_sqrt(fs, 2, n) == ref_sqrt(fs, 2, n), n, fsize);
                CHECK(run_sqrt(fs, P - 2, n) == ref_sqrt(fs, P - 2, n), n, fsize);
                const u32 e = u32(rng() % P), c = u32(rng() % P);
                const Vec fp = with_first(f, kind == 2 ? P - 1 : 1 + u32(rng() % (P - 1)));
                CHECK(run_pow(fp, e, c, n) == ref_pow(fp, e, c, n), n, fsize);
            }
        }
    }
}

// Identities at large ragged sizes (the references are O(n^2)).
void test_identities() {
    std::vector<std::size_t> sizes = {16385, 20480, 24576, 24577, 28672, 600000 / 8};
    if (!kSanitized) {
        const std::size_t K = std::size_t(1) << 16;  // 2^18 = 4K
        for (std::size_t n : {std::size_t(600000), 16 * K + 1, 5 * K, 6 * K, 13 * K, 8 * K + 75000, 32 * K + 7})
            sizes.push_back(n);
    }
    for (std::size_t n : sizes) {
        Vec f = random_vec(n, 0);
        f[0] = 3;
        const Vec g = run_inverse(f, n);
        Vec one(n);
        one[0] = 1;
        CHECK(product(f, g, n) == one, n);
        f[0] = 0;
        const Vec e = run_exp(f, n);
        CHECK(run_log(e, n) == f, n);
        f[0] = 4;
        const Vec s = run_sqrt(f, 2, n);
        CHECK(product(s, s, n) == f, n);
        // pow(f, a) pow(f, b) = pow(f, a + b) and pow(f, P - 1) = inverse(f) (c = f[0]^e)
        const u32 a = u32(rng() % P), b = u32(rng() % P);
        const Vec pa = run_pow(f, a, 1, n), pb = run_pow(f, b, 1, n), pab = run_pow(f, (a + b) % P, 1, n);
        CHECK(product(pa, pb, n) == pab, n);
        CHECK(run_pow(f, P - 1, inv(4), n) == run_inverse(f, n), n);
    }
}

// The caller's scratch at ragged sizes, filled with garbage; words past it unwritten.
void test_scratch() {
    for (std::size_t n : {65ul, 97ul, 129ul, 160ul, 4097ul, 5120ul, 6144ul, 6145ul, 75001ul}) {
        for (std::size_t offset = 0; offset < 16; offset += 5) {
            const Vec f = with_first(random_vec(n, 0), 1);
            const std::size_t words = std::max({fps::series::inverse_scratch(n), fps::series::log_scratch(n),
                                                fps::series::exp_scratch(n), fps::series::sqrt_scratch(n),
                                                fps::series::pow_scratch(n)});
            Vec scratch(words + offset + 8, P - 1);
            const std::span<u32> s(scratch.data() + offset, words);
            Vec g(n);
            fps::series::inverse(f, g, s.first(fps::series::inverse_scratch(n)));
            CHECK(g == run_inverse(f, n), n, offset);
            fps::series::log(f, g, s.first(fps::series::log_scratch(n)));
            CHECK(g == run_log(f, n), n, offset);
            const Vec f0 = with_first(f, 0);
            fps::series::exp(f0, g, s.first(fps::series::exp_scratch(n)));
            CHECK(g == run_exp(f0, n), n, offset);
            fps::series::sqrt(f, 1, g, s.first(fps::series::sqrt_scratch(n)));
            CHECK(g == run_sqrt(f, 1, n), n, offset);
            fps::series::pow(f, 12345, 6, g, s.first(fps::series::pow_scratch(n)));
            CHECK(g == run_pow(f, 12345, 6, n), n, offset);
            CHECK(std::all_of(scratch.end() - 8, scratch.end(), [](u32 c) { return c == P - 1; }), n, offset);
        }
    }
}

// The helpers against the O(n m) product: low_product (sums, small::multiply, plain and split NTT
// plans) on factors of c and fewer words, window() at the edges of the factors and in the middle.
void test_helpers() {
    namespace d = fps::series::detail;
    const auto naive = [](const Vec& a, const Vec& b) {
        Vec c(a.size() + b.size() - 1);
        for (std::size_t i = 0; i < a.size(); ++i)
            for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = u32((c[i + j] + u64(a[i]) * b[j]) % P);
        return c;
    };
    std::vector<std::size_t> sizes = {1, 2, 3, 16, 31, 32, 33, 40, 63, 64, 65, 100, 127, 128, 129, 255, 300, 1000, 1025,
                                      4097};
    if (!kSanitized) sizes.insert(sizes.end(), {20000, 32769});
    for (std::size_t c : sizes) {
        for (std::size_t na : {c, c - c / 3, std::size_t(1)}) {
            const Vec a = random_vec(na, 0), b = random_vec(c, rng() % 2 ? 0 : 2);
            const Vec full = naive(a, b);
            Vec work(d::low_product_words(c) + 64), out(c, 7);
            u32* const aligned = work.data() + (64 - reinterpret_cast<std::uintptr_t>(work.data()) % 64) % 64 / 4;
            d::low_product(a, b, c, out.data(), aligned);
            bool ok = true;
            for (std::size_t i = 0; i < c; ++i) ok = ok && out[i] == (i < full.size() ? full[i] : 0);
            CHECK(ok, c, na);
            // windows at the start, the middle and the end of the product
            if (c > 300) continue;
            for (std::size_t pos : {std::size_t(0), full.size() / 2, full.size() > 20 ? full.size() - 20 : 0}) {
                const std::size_t count = std::min<std::size_t>(19, full.size() - pos);
                Vec w(count);
                d::window(a.data(), a.size(), b.data(), b.size(), pos, count, w.data());
                CHECK(std::equal(w.begin(), w.end(), full.begin() + std::ptrdiff_t(pos)), c, pos);
            }
        }
    }
}

}  // namespace

int main() {
    test_helpers();
    test_references();
    test_identities();
    test_scratch();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("series ragged: all tests passed\n");
    return 0;
}
