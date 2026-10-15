// Tests of backend/gcd's jumps (issue #157): HalfGcd::jump against Euclid's algorithm, the matrix
// applied to (a, b) giving the remainders r_h and r_(h+1) up to constants, for Euclid's base and the
// recursion (cutoffs 1, 2, 3, 8, 64); entries outside `need` left out without changing the others
// (bit for bit, at0 included); quotient sequences of chosen degrees (degree drops in the middle of
// Euclid's runs of degree-1 steps, and at their ends), through jump, gcd and inverse_mod; the
// per-target cutoffs in the caller's scratch of exactly the bound.
// tools/kernel.py builds it at -O2 and with ASan/UBSan.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/gcd/gcd.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = fps::half_gcd::kP;
namespace gd = fps::half_gcd::detail;

int failures = 0, checks = 0;
std::mt19937_64 rng(157);

void fail(const std::string& what) {
    if (++failures <= 20) std::printf("FAIL %s\n", what.c_str());
}

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 add(u32 a, u32 b) { return a + b >= P ? a + b - P : a + b; }
u32 sub(u32 a, u32 b) { return a >= b ? a - b : a + P - b; }
u32 inv(u32 a) {
    u32 r = 1;
    for (u64 e = P - 2; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
u32 random_word() { return u32(rng() % P); }
u32 nonzero_word() { return 1 + u32(rng() % (P - 1)); }

void trim(Vec& a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
}

Vec times(const Vec& f, const Vec& g) {
    if (f.empty() || g.empty()) return {};
    Vec out(f.size() + g.size() - 1);
    for (std::size_t i = 0; i < f.size(); ++i)
        for (std::size_t j = 0; j < g.size(); ++j) out[i + j] = add(out[i + j], mul(f[i], g[j]));
    return out;
}

Vec plus(Vec f, const Vec& g) {
    if (f.size() < g.size()) f.resize(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) f[i] = add(f[i], g[i]);
    trim(f);
    return f;
}

// (q, r) = (a quo b, a mod b), b trimmed and nonzero.
std::pair<Vec, Vec> divmod(Vec a, const Vec& b) {
    trim(a);
    if (a.size() < b.size()) return {{}, a};
    Vec q(a.size() - b.size() + 1);
    const u32 lead = inv(b.back());
    for (std::size_t at = a.size() - b.size() + 1; at-- > 0;) {
        const u32 c = mul(a[at + b.size() - 1], lead);
        q[at] = c;
        for (std::size_t t = 0; t < b.size(); ++t) a[at + t] = sub(a[at + t], mul(c, b[t]));
    }
    a.resize(b.size() - 1);
    trim(a);
    return {q, a};
}

Vec monic(Vec a) {
    trim(a);
    if (a.empty()) return a;
    const u32 s = inv(a.back());
    for (u32& x : a) x = mul(x, s);
    return a;
}

std::ptrdiff_t degree(const Vec& a) { return std::ptrdiff_t(a.size()) - 1; }

// The Euclidean remainders r_0 = a, r_1 = b, ... down to the last nonzero one.
std::vector<Vec> remainders(Vec a, Vec b) {
    trim(a), trim(b);
    std::vector<Vec> r = {a, b};
    while (!r.back().empty()) r.push_back(divmod(r[r.size() - 2], r.back()).second);
    r.pop_back();
    return r;
}

// The matrix entry i of a jump as a polynomial.
Vec entry(const gd::Matrix& r, int i) { return Vec(r.entry[i], r.entry[i] + r.size[i]); }

// out = the jump of k on (a, b) by the kernel, in memory of its own.
struct Jump {
    std::vector<u32> memory;
    gd::Stack stack;
    gd::HalfGcd h;
    gd::Matrix r;
    Jump(const Vec& a, const Vec& b, std::ptrdiff_t k, std::ptrdiff_t direct, std::ptrdiff_t whole, unsigned need)
        : memory(gd::half_gcd_words(a.size()) + 4 * (a.size() + 64) + 4096),
          stack(memory.data(), memory.size()),
          h(stack, direct, whole),
          r(h.matrix(std::size_t(k), 0)) {
        h.jump(a.data(), degree(a), b.data(), degree(b), k, r, need);
    }
};

// The jump of k on (a, b) (deg a > deg b) against Euclid: R (a, b) = (c r_h, c' r_(h+1)) with
// deg r_h >= deg a - k > deg r_(h+1); the entries in need and at0 of their rows, the same bits as
// for need 0xF.
void check_jump(const Vec& a, const Vec& b, std::ptrdiff_t k, std::ptrdiff_t direct, const std::string& what,
                std::ptrdiff_t whole = 0) {
    ++checks;
    const std::vector<Vec> r = remainders(a, b);
    const std::ptrdiff_t target = degree(a) - k;
    std::size_t h = 0;
    while (h + 1 < r.size() && degree(r[h + 1]) >= target) ++h;
    const Vec rh = r[h], rh1 = h + 1 < r.size() ? r[h + 1] : Vec{};
    const Jump full(a, b, k, direct, whole, 0xF);
    const Vec c = plus(times(entry(full.r, 0), a), times(entry(full.r, 1), b));
    const Vec d = plus(times(entry(full.r, 2), a), times(entry(full.r, 3), b));
    const std::string tag = what + " (deg a " + std::to_string(degree(a)) + ", deg b " + std::to_string(degree(b)) +
                            ", k " + std::to_string(k) + ", direct " + std::to_string(direct) + ", whole " + std::to_string(whole) + ")";
    if (monic(c) != monic(rh) || monic(d) != monic(rh1)) return fail(tag + ": R (a, b) is not (r_h, r_h+1)");
    if (std::ptrdiff_t(full.r.progress) != degree(a) - degree(rh)) fail(tag + ": progress");
    for (int i = 0; i < 4; ++i) {
        const Vec e = entry(full.r, i);
        if ((e.empty() ? 0 : e[0]) != full.r.at0[i]) fail(tag + ": at0 of entry " + std::to_string(i));
        if (!e.empty() && e.back() == 0) fail(tag + ": entry " + std::to_string(i) + " not trimmed");
    }
    for (const unsigned need : {0x1u, 0x2u, 0x3u, 0x4u, 0x8u, 0xCu, 0x6u}) {
        const Jump part(a, b, k, direct, whole, need);
        if (part.r.progress != full.r.progress) fail(tag + ": progress with need " + std::to_string(need));
        for (int i = 0; i < 4; ++i) {
            const int row = i / 2;
            if (need >> i & 1 && entry(part.r, i) != entry(full.r, i))
                fail(tag + ": entry " + std::to_string(i) + " with need " + std::to_string(need));
            if (need >> (2 * row) & 3 && part.r.at0[i] != full.r.at0[i])
                fail(tag + ": at0 " + std::to_string(i) + " with need " + std::to_string(need));
        }
    }
}

Vec random_poly(std::size_t size) {
    Vec a(size);
    for (u32& x : a) x = random_word();
    if (size) a.back() = nonzero_word();
    return a;
}

// (a, b) whose Euclidean quotients have the given degrees, from the bottom up: r_last of degree
// last, then r_(i-1) = q_i r_i + r_(i+1) with deg q_i = degrees[i - 1] (random, nonzero leads).
std::pair<Vec, Vec> with_quotients(const std::vector<int>& degrees, std::size_t last) {
    Vec below = {}, r = random_poly(last + 1);
    for (std::size_t i = degrees.size(); i-- > 0;) {
        const Vec q = random_poly(std::size_t(degrees[i]) + 1);
        Vec above = plus(times(q, r), below);
        below = r, r = above;
    }
    return {r, below};
}

void random_pairs() {
    for (int n = 1; n <= 300; n += n < 40 ? 1 : 37) {
        for (const int gap : {1, 2, 5}) {
            if (n - gap < 0) continue;
            const Vec a = random_poly(std::size_t(n) + 1), b = random_poly(std::size_t(n - gap) + 1);
            for (const std::ptrdiff_t k : {std::ptrdiff_t(1), std::ptrdiff_t(n / 3 + 1), std::ptrdiff_t(n / 2), std::ptrdiff_t(n)}) {
                if (k < 1 || k > n) continue;
                for (const std::ptrdiff_t direct : {std::ptrdiff_t(1), std::ptrdiff_t(3), std::ptrdiff_t(8), std::ptrdiff_t(64)})
                    check_jump(a, b, k, direct, "random");
            }
        }
    }
}

// Degree drops: quotients of degree 2 or 3 among degree-1 ones, at every place of short runs, so
// that the scalar look-ahead of Euclid's steps meets a zero top word mid-run and at its end.
void chosen_quotients() {
    for (int length = 1; length <= 12; ++length)
        for (int at = 0; at < length; ++at)
            for (const int drop : {2, 3}) {
                std::vector<int> degrees(std::size_t(length), 1);
                degrees[std::size_t(at)] = drop;
                for (const std::size_t last : {std::size_t(0), std::size_t(1), std::size_t(5)}) {
                    const auto [a, b] = with_quotients(degrees, last);
                    const std::ptrdiff_t n = degree(a);
                    for (std::ptrdiff_t k = 1; k <= n; ++k)
                        for (const std::ptrdiff_t direct : {std::ptrdiff_t(1), std::ptrdiff_t(2), std::ptrdiff_t(64)})
                            check_jump(a, b, k, direct, "drop at " + std::to_string(at) + " of " + std::to_string(length));
                }
            }
    // Longer runs with drops spread out, through the recursion, gcd and inverse_mod.
    for (int trial = 0; trial < 40; ++trial) {
        std::vector<int> degrees;
        const int length = 20 + int(rng() % 200);
        for (int i = 0; i < length; ++i) degrees.push_back(rng() % 9 == 0 ? 2 + int(rng() % 3) : 1);
        const std::size_t last = rng() % 2 ? 0 : 1 + rng() % 40;
        const auto [a, b] = with_quotients(degrees, last);
        const std::ptrdiff_t n = degree(a);
        for (const std::ptrdiff_t k : {n, n / 2 + 1, n / 3 + 1})
            for (const std::ptrdiff_t direct : {std::ptrdiff_t(1), std::ptrdiff_t(5), std::ptrdiff_t(64)})
                check_jump(a, b, k, direct, "chosen run " + std::to_string(trial));
        ++checks;
        const Vec expect = monic(remainders(a, b).back());
        Vec out(a.size());
        const std::size_t t = gd::gcd(a, b, out, {}, 3);
        if (Vec(out.begin(), out.begin() + std::ptrdiff_t(t)) != expect) fail("chosen run gcd " + std::to_string(trial));
        ++checks;
        Vec h(a.size() - 1);
        const std::ptrdiff_t s = gd::inverse_mod(b, a, h, {}, 3);
        if ((s >= 0) != (expect.size() == 1)) fail("chosen run inverse_mod existence " + std::to_string(trial));
        if (s >= 0) {
            Vec product = times(b, Vec(h.begin(), h.begin() + s));
            if (divmod(product, a).second != Vec{1}) fail("chosen run inverse_mod " + std::to_string(trial));
        }
    }
}

// The leaves' shape (deg a = 2k, deg b = 2k - 1) at Euclid's cutoff and around it, and jumps that
// start mid-way (deg a > 2k: only the top 2k + 1 words read).
void leaves() {
    for (const std::ptrdiff_t k : {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128}) {
        const Vec a = random_poly(std::size_t(2 * k) + 1), b = random_poly(std::size_t(2 * k));
        for (const std::ptrdiff_t direct : {std::ptrdiff_t(2), std::ptrdiff_t(64), std::ptrdiff_t(128)}) check_jump(a, b, k, direct, "leaf");
        const Vec a2 = random_poly(std::size_t(3 * k) + 5), b2 = random_poly(std::size_t(3 * k) + 3);
        check_jump(a2, b2, k, 64, "top part");
        // Full jumps (k = deg a: Euclid up to whole) and the first part of inverse_mod's top.
        const Vec a3 = random_poly(std::size_t(k) + 1), b3 = random_poly(std::size_t(k));
        for (const std::ptrdiff_t direct : {std::ptrdiff_t(2), std::ptrdiff_t(8), std::ptrdiff_t(32)}) {
            check_jump(a3, b3, k, direct, "full", 64);
            check_jump(a, b, k, direct, "leaf", 64);
        }
    }
}

// Cutoffs around select.hpp's (32, 64 and 128, with whole 64 and without) in the caller's scratch
// of exactly gcd_scratch / inverse_mod_scratch words, garbage-filled.
void scratch_bounds() {
    const std::ptrdiff_t cutoffs[][2] = {{32, 0}, {32, 64}, {64, 0}, {128, 0}, {gd::kDirect, gd::kDirectWhole}};
    for (const std::size_t n : {2, 3, 33, 64, 65, 127, 128, 129, 200, 255, 256, 257, 511, 513, 1000, 2047, 2048, 3000})
        for (const auto [direct, whole] : cutoffs) {
            const Vec f = random_poly(n), g = random_poly(n - n / 7);
            const std::string what = "scratch n " + std::to_string(n) + " direct " + std::to_string(direct) + " whole " +
                                     std::to_string(whole);
            ++checks;
            Vec scratch(fps::half_gcd::gcd_scratch(f.size(), g.size()), 0xa5a5a5a5u), out(n);
            const std::size_t t = gd::gcd(f, g, out, scratch, direct, whole);
            const bool small = n <= 600;  // else coprime (random, this seed)
            const Vec expect = small ? monic(remainders(f, g).back()) : Vec{1};
            if (Vec(out.begin(), out.begin() + std::ptrdiff_t(t)) != expect) fail(what + ": gcd");
            ++checks;
            Vec scratch2(fps::half_gcd::inverse_mod_scratch(g.size(), f.size()), 0x5a5a5a5au), h(n - 1);
            const std::ptrdiff_t s = gd::inverse_mod(g, f, h, scratch2, direct, whole);
            if (s < 0) {
                if (small && expect.size() == 1) fail(what + ": inverse_mod returned -1");
            } else if (small && divmod(times(g, Vec(h.begin(), h.begin() + s)), f).second != Vec{1}) {
                fail(what + ": inverse_mod");
            }
        }
}

}  // namespace

int main() {
    random_pairs();
    chosen_quotients();
    leaves();
    scratch_bounds();
    std::printf("%s: %d checks, %d failures\n", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
