// Tests of backend/gcd: gcd against Euclid's O(n^2) algorithm, inverse_mod by f h = 1 mod g; random
// pairs, coprime and with common factors of degree 1, 2, 2^k +- 1, sizes around powers of two,
// sparse inputs (x^n - 1: many long quotients), with Euclid's base forced small so that the
// recursion runs on short inputs; this thread's region and the caller's scratch (garbage-filled);
// argument checks. tools/kernel.py builds it at -O2 and with ASan/UBSan.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/division/division.hpp"
#include "backend/gcd/gcd.hpp"
#include "backend/ntt/ntt.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = fps::half_gcd::kP;
namespace gd = fps::half_gcd::detail;

int failures = 0, checks = 0;
std::mt19937_64 rng(103);

void fail(const std::string& what) {
    if (++failures <= 20) std::printf("FAIL %s\n", what.c_str());
}

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 sub(u32 a, u32 b) { return a >= b ? a - b : a + P - b; }
u32 inv(u32 a) {
    u32 r = 1;
    for (u64 e = P - 2; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}

void trim(Vec& a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
}

// Random words; kind 0 random, 1 mostly zero. The last word nonzero if nonzero_top.
Vec make(std::size_t n, int kind = 0, bool nonzero_top = true) {
    Vec a(n);
    for (u32& x : a) x = kind == 0 || rng() % 5 == 0 ? u32(rng() % P) : 0;
    if (nonzero_top && n && a.back() == 0) a.back() = 1 + u32(rng() % (P - 1));
    return a;
}

// a mod b by long division (b trimmed, nonzero).
Vec mod(Vec a, const Vec& b) {
    trim(a);
    const std::size_t m = b.size();
    const u32 lead = inv(b.back());
    while (a.size() >= m) {
        const u32 q = mul(a.back(), lead);
        const std::size_t at = a.size() - m;
        for (std::size_t t = 0; t < m; ++t) a[at + t] = sub(a[at + t], mul(q, b[t]));
        a.pop_back();
        trim(a);
    }
    return a;
}

Vec reference_gcd(Vec a, Vec b) {
    trim(a), trim(b);
    while (!b.empty()) a = std::exchange(b, mod(a, b));
    if (!a.empty()) {
        const u32 s = inv(a.back());
        for (u32& x : a) x = mul(x, s);
    }
    return a;
}

// f g, by ntt::multiply.
Vec product(const Vec& f, const Vec& g) {
    if (f.empty() || g.empty()) return {};
    Vec out(f.size() + g.size() - 1);
    fps::ntt::multiply(f, g, out);
    return out;
}

// f mod g by division::divide.
Vec fast_mod(const Vec& f, const Vec& g) {
    Vec q(fps::division::quotient_size(f.size(), g.size())), r(g.size() - 1);
    fps::division::divide(f, g, q, r);
    trim(r);
    return r;
}

// gcd through the kernel: region or scratch (garbage-filled), out with guard words.
Vec run_gcd(const Vec& f, const Vec& g, std::ptrdiff_t direct, bool scratch) {
    Vec tf = f, tg = g;
    trim(tf), trim(tg);
    const std::size_t room = std::max(f.size(), g.size());
    Vec out(room + 2, 0xdeadbeef);
    Vec memory;
    if (scratch) memory.assign(fps::half_gcd::gcd_scratch(tf.size(), tg.size()) + 3, 0xa5a5a5a5u);
    const std::size_t t = gd::gcd(f, g, std::span<u32>(out.data() + 1, room), memory, direct);
    if (out[0] != 0xdeadbeef || out[room + 1] != 0xdeadbeef) fail("gcd wrote outside out");
    return Vec(out.begin() + 1, out.begin() + 1 + std::ptrdiff_t(t));
}

void check_gcd(const Vec& f, const Vec& g, const Vec& expect, std::ptrdiff_t direct, bool scratch, const std::string& what) {
    ++checks;
    try {
        const Vec got = run_gcd(f, g, direct, scratch);
        if (got != expect)
            fail(what + ": gcd of sizes " + std::to_string(f.size()) + ", " + std::to_string(g.size()) + " direct " +
                 std::to_string(direct) + ": got degree " + std::to_string(int(got.size()) - 1) + ", expected " +
                 std::to_string(int(expect.size()) - 1));
    } catch (const std::exception& e) {
        fail(what + ": threw " + e.what());
    }
}

// h = 1 / f mod g through the kernel, then checked: f h = 1 mod g, or (for -1) gcd != 1.
void check_inverse(const Vec& f, const Vec& g, std::ptrdiff_t direct, bool scratch, int expect_invertible,
                   const std::string& what) {
    ++checks;
    try {
        const std::size_t m = g.size();
        Vec h(m - 1 + 2, 0xdeadbeef);
        Vec memory;
        Vec tf = f;
        trim(tf);
        if (scratch) memory.assign(fps::half_gcd::inverse_mod_scratch(tf.size(), m) + 5, 0x5a5a5a5au);
        const std::ptrdiff_t t = gd::inverse_mod(f, g, std::span<u32>(h.data() + 1, m - 1), memory, direct);
        if (h[0] != 0xdeadbeef || h[m] != 0xdeadbeef) fail(what + ": inverse_mod wrote outside h");
        const Vec hs(h.begin() + 1, h.begin() + std::ptrdiff_t(m));
        const bool small = m <= 300;
        if (t < 0) {
            if (expect_invertible == 1) fail(what + ": inverse_mod returned -1 for an invertible f");
            if (small && reference_gcd(f, g).size() == 1) fail(what + ": -1 but gcd = 1");
            return;
        }
        if (expect_invertible == 0) fail(what + ": inverse_mod found an inverse with gcd != 1");
        for (std::size_t i = std::size_t(t); i < hs.size(); ++i)
            if (hs[i] != 0) return fail(what + ": h not zero past t");
        if (t > 0 && hs[std::size_t(t) - 1] == 0) fail(what + ": t not trimmed");
        if (m == 1) return;
        Vec hv(hs.begin(), hs.begin() + t);
        const Vec fh = product(tf, hv);
        const Vec r = fh.empty() ? Vec{} : small ? mod(fh, g) : fast_mod(fh, g);
        if (r != Vec{1}) fail(what + ": f h != 1 mod g (sizes " + std::to_string(f.size()) + ", " + std::to_string(m) + ")");
    } catch (const std::exception& e) {
        fail(what + ": threw " + e.what());
    }
}

const std::ptrdiff_t kDirects[] = {1, 2, 3, 5, 8, 64};

void small_pairs() {
    for (std::size_t n = 0; n <= 40; ++n)
        for (std::size_t m = 0; m <= 40; ++m)
            for (const std::ptrdiff_t direct : {1, 3, 64}) {
                const Vec f = make(n, int(rng() % 2), rng() % 4 != 0), g = make(m, int(rng() % 2), rng() % 4 != 0);
                const bool scratch = (n + m) % 2 == 0;
                check_gcd(f, g, reference_gcd(f, g), direct, scratch, "small");
                if (m >= 1 && g.back() != 0) check_inverse(f, g, direct, scratch, -1, "small inverse");
            }
}

// f = u w, g = v w with w of degree dw.
void common_factors() {
    const std::size_t sizes[] = {2, 3, 5, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1000};
    for (const std::size_t nu : sizes)
        for (const std::size_t dw : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(3), std::size_t(7),
                                     std::size_t(8), std::size_t(9), std::size_t(31), std::size_t(32), std::size_t(33),
                                     nu / 2, nu - 1}) {
            const Vec w = make(dw + 1), u = make(nu), v = make(nu - std::min(nu - 1, std::size_t(rng() % 4)));
            const Vec f = product(u, w), g = product(v, w);
            const Vec expect = reference_gcd(f, g);
            const std::string what = "common nu " + std::to_string(nu) + " dw " + std::to_string(dw);
            for (const std::ptrdiff_t direct : kDirects) {
                if (nu == 1000 && direct != 64 && direct != 3) continue;
                check_gcd(f, g, expect, direct, direct % 2 == 0, what);
                check_inverse(f, g, direct, direct % 2 == 1, expect.size() == 1 ? 1 : 0, what);
            }
        }
}

// Sizes around powers of two, coprime random pairs (the bench's shape), against Euclid.
void powers_of_two() {
    for (int lg = 1; lg <= 12; ++lg)
        for (const int delta : {-1, 0, 1})
            for (const int gap : {0, 1, 2, 37}) {
                const std::size_t n = (std::size_t(1) << lg) + std::size_t(delta + 1);
                if (n <= std::size_t(gap) + 1) continue;
                const Vec f = make(n), g = make(n - std::size_t(gap));
                const Vec expect = reference_gcd(f, g);
                for (const std::ptrdiff_t direct : {std::ptrdiff_t(4), std::ptrdiff_t(64)}) {
                    const std::string what = "pow2 n " + std::to_string(n) + " gap " + std::to_string(gap);
                    check_gcd(f, g, expect, direct, lg % 2 == 0, what);
                    check_inverse(f, g, direct, lg % 2 == 1, expect.size() == 1 ? 1 : 0, what);
                }
            }
}

// x^a - 1 and x^b - 1: gcd x^gcd(a, b) - 1, through long quotients (division steps in jumps).
void sparse() {
    const std::size_t pairs[][2] = {{12, 8}, {100, 75}, {128, 96}, {1000, 600}, {1024, 768}, {4096, 1024}, {3000, 1},
                                    {5000, 4999}, {65536, 49152}, {20000, 3}};
    for (const auto& [a, b] : pairs)
        for (const std::ptrdiff_t direct : {std::ptrdiff_t(2), std::ptrdiff_t(64)}) {
            Vec f(a + 1, 0), g(b + 1, 0), expect(std::gcd(a, b) + 1, 0);
            f[0] = g[0] = expect[0] = P - 1, f[a] = g[b] = expect.back() = 1;
            const std::string what = "x^" + std::to_string(a) + " - 1, x^" + std::to_string(b) + " - 1";
            check_gcd(f, g, expect, direct, direct == 2, what);
            if (std::gcd(a, b) > 1) check_inverse(f, g, direct, true, 0, what);
            // x^a - 1 + x^c terms: random sparse pairs, against Euclid where it is cheap.
            if (a <= 5000) {
                const Vec fs = make(a + 1, 1), gs = make(b + 1, 1);
                check_gcd(fs, gs, reference_gcd(fs, gs), direct, true, what + " sparse");
            }
        }
}

// Large: a known common factor w of random u, v (coprime with probability ~1 - n / P).
void large() {
    struct Case {
        std::size_t nu, nv, dw;
    };
    const Case cases[] = {{1 << 14, 1 << 14, 1}, {1 << 15, (1 << 15) - 3, 1000}, {50000, 50000, 0}, {1 << 16, 1 << 16, 1 << 12},
                          {100000, 30000, 77}, {1 << 18, 1 << 18, 5}};
    for (const auto& [nu, nv, dw] : cases) {
        const Vec w = make(dw + 1), u = make(nu), v = make(nv);
        Vec expect = w;
        const u32 s = inv(w.back());
        for (u32& x : expect) x = mul(x, s);
        const Vec f = product(u, w), g = product(v, w);
        const std::string what = "large " + std::to_string(nu) + " " + std::to_string(nv) + " " + std::to_string(dw);
        check_gcd(f, g, expect, 64, nu % 2 == 0, what);
        check_inverse(f, g, 64, nu % 2 == 1, dw == 0 ? 1 : 0, what);
        if (dw == 0) check_inverse(u, v, 64, false, 1, what + " (u, v)");
    }
}

void arguments() {
    const Vec f = make(10), g = make(6);
    Vec out(5), h(5);
    const auto expect_throw = [](const std::string& what, auto&& call) {
        ++checks;
        try {
            call();
            fail(what + ": no exception");
        } catch (const std::invalid_argument&) {
        }
    };
    expect_throw("gcd out short", [&] { fps::half_gcd::gcd(f, g, out); });
    expect_throw("inverse_mod h size", [&] { fps::half_gcd::inverse_mod(f, g, std::span<u32>(h.data(), 4)); });
    Vec g0 = g;
    g0.back() = 0;
    expect_throw("inverse_mod g top zero", [&] { fps::half_gcd::inverse_mod(f, g0, h); });
    expect_throw("inverse_mod g empty", [&] { fps::half_gcd::inverse_mod(f, Vec{}, std::span<u32>()); });
    Vec small(3);
    Vec big = make(200), big2 = make(200), out200(200);
    expect_throw("scratch short", [&] { fps::half_gcd::gcd(big, big2, out200, small); });
    // Constants and zeros.
    ++checks;
    Vec o(6, 7);
    if (fps::half_gcd::gcd(Vec{}, Vec{0, 0}, o) != 0) fail("gcd(0, 0) != 0");
    if (fps::half_gcd::gcd(Vec{0, 5, 0}, Vec{}, o) != 2 || o[0] != 0 || o[1] != 1) fail("gcd(5x, 0) != x");
    if (fps::half_gcd::gcd(Vec{3}, g, o) != 1 || o[0] != 1) fail("gcd(3, g) != 1");
    if (fps::half_gcd::inverse_mod(f, Vec{4}, std::span<u32>()) != 0) fail("inverse_mod by a constant != 0");
    Vec h2(5);
    if (fps::half_gcd::inverse_mod(Vec{}, g, h2) != -1) fail("inverse_mod(0, g) != -1");
}

}  // namespace

int main() {
    small_pairs();
    common_factors();
    powers_of_two();
    sparse();
    large();
    arguments();
    std::printf("%s: %d checks, %d failures\n", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
