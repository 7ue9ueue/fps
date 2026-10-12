// Tests of backend/division against long division (O(n m)): every plan the cost model can pick
// (block sizes, the last block's methods, the remainder's methods) on small and ragged sizes, the
// chosen plan through divide() in this thread's region and in the caller's scratch, and large
// divisions by f = q g + r at random points. tools/kernel.py builds it at -O2 and with ASan/UBSan.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/division/division.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = fps::division::kP;
namespace dd = fps::division::detail;

int failures = 0;
std::mt19937_64 rng(102);

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

// Coefficient kinds: random, all P - 1, mostly zero.
Vec make(std::size_t n, int kind) {
    Vec a(n);
    for (u32& x : a) {
        if (kind == 0) x = u32(rng() % P);
        else if (kind == 1) x = P - 1;
        else x = rng() % 7 == 0 ? u32(rng() % P) : 0;
    }
    return a;
}
Vec make_divisor(std::size_t m, int kind) {
    Vec g = make(m, kind);
    if (g.back() == 0) g.back() = 1 + u32(rng() % (P - 1));
    return g;
}

// q, r by long division.
void reference(const Vec& f, const Vec& g, Vec& q, Vec& r) {
    const std::size_t n = f.size(), d = g.size() - 1;
    Vec rest = f;
    q.assign(n > d ? n - d : 0, 0);
    const u32 lead = power(g[d], P - 2);
    for (std::size_t i = q.size(); i-- > 0;) {
        const u32 c = mul(rest[i + d], lead);
        q[i] = c;
        for (std::size_t t = 0; t <= d; ++t) rest[i + t] = (rest[i + t] + P - mul(c, g[t])) % P;
    }
    r.assign(d, 0);
    for (std::size_t i = 0; i < std::min(n, d); ++i) r[i] = rest[i];
}

const char* kLast[] = {"full", "short", "small", "dot"};
const char* kRemainder[] = {"none", "direct", "wrap", "split", "tail"};

std::string describe(const dd::Plan& p) {
    if (p.long_division) return "long";
    return "s=" + std::to_string(p.s) + " B=" + std::to_string(p.blocks) + " last=" + kLast[int(p.last_method)] +
           " rem=" + kRemainder[int(p.remainder)] + (p.tails ? " tails" : " windows=" + std::to_string(p.windows));
}

// Runs plan p in a scratch of garbage, with guard words, and compares with q0, r0.
void run_plan(const dd::Plan& p, const Vec& f, const Vec& g, const Vec& q0, const Vec& r0, bool remainder,
              const std::string& name) {
    const std::size_t words = dd::words(p) + 64;
    Vec scratch(words + 32);
    for (u32& x : scratch) x = u32(rng());
    Vec q(q0.size() + 2, 0xdeadbeef), r((remainder ? r0.size() : 0) + 2, 0xdeadbeef);
    std::span<u32> qs(q.data() + 1, q0.size()), rs(r.data() + 1, remainder ? r0.size() : 0);
    dd::run(p, f, g, qs, rs, scratch.data() + 1, words);
    const std::string what = name + " " + describe(p);
    if (q.front() != 0xdeadbeef || q.back() != 0xdeadbeef || r.front() != 0xdeadbeef || r.back() != 0xdeadbeef)
        fail(what + ": wrote outside q or r");
    if (!std::equal(qs.begin(), qs.end(), q0.begin())) fail(what + ": q");
    if (remainder && !std::equal(rs.begin(), rs.end(), r0.begin())) fail(what + ": r");
}

// Every plan for f of n and g of m words (n >= m).
void every_plan(std::size_t n, std::size_t m, int kind) {
    const Vec f = make(n, kind), g = make_divisor(m, kind);
    Vec q0, r0;
    reference(f, g, q0, r0);
    const std::string name = "n=" + std::to_string(n) + " m=" + std::to_string(m) + " kind=" + std::to_string(kind);
    const fps::division::Cost cost(fps::division::costs.get());
    const std::size_t k = n - m + 1;
    dd::Plan lp;
    lp.n = n, lp.m = m, lp.d = m - 1, lp.k = k, lp.long_division = true;
    run_plan(lp, f, g, q0, r0, true, name);
    if (m < 2) return;  // the blocks need d >= 1 (the plan sends d = 0 to long division)
    for (std::size_t s = 32;; s *= 2) {
        for (const auto last : {dd::Last::full, dd::Last::short_length, dd::Last::small, dd::Last::dot}) {
            const dd::Plan p = dd::block_plan(cost, n, m, s, last);
            if (!(p.ns < 1e300)) continue;
            run_plan(p, f, g, q0, r0, false, name);
            for (const auto how : {dd::Remainder::direct, dd::Remainder::wrap, dd::Remainder::split, dd::Remainder::tail}) {
                const dd::Plan pr = dd::with_remainder(cost, p, how);
                if (pr.ns < 1e300) run_plan(pr, f, g, q0, r0, true, name);
            }
        }
        if (s >= k) break;
    }
}

// The public API: this thread's region and the caller's scratch, with and without r.
void public_api(std::size_t n, std::size_t m, int kind) {
    const Vec f = make(n, kind), g = make_divisor(m, kind);
    Vec q0, r0;
    reference(f, g, q0, r0);
    const std::string name = "divide n=" + std::to_string(n) + " m=" + std::to_string(m);
    Vec q(q0.size()), r(r0.size());
    fps::division::divide(f, g, q, r);
    if (q != q0 || r != r0) fail(name + " region");
    Vec scratch(fps::division::scratch_words(n, m) + 3);
    std::fill(q.begin(), q.end(), 7u), std::fill(r.begin(), r.end(), 7u);
    fps::division::divide(f, g, q, r, std::span(scratch.data() + 3, scratch.size() - 3));
    if (q != q0 || r != r0) fail(name + " scratch");
    std::fill(q.begin(), q.end(), 7u);
    fps::division::divide(f, g, q, {});
    if (q != q0) fail(name + " quotient only");
}

// f = q g + r at 4 random points, for sizes past the reference's reach.
u32 eval(const u32* a, std::size_t n, u32 x) {
    u32 y = 0;
    for (std::size_t i = n; i-- > 0;) y = (mul(y, x) + a[i]) % P;
    return y;
}
void large(std::size_t n, std::size_t m) {
    const Vec f = make(n, 0), g = make_divisor(m, 0);
    Vec q(fps::division::quotient_size(n, m)), r(m - 1);
    fps::division::divide(f, g, q, r);
    const dd::Plan p = dd::plan(n, m, true);
    for (int i = 0; i < 4; ++i) {
        const u32 x = u32(rng() % P);
        const u32 lhs = eval(f.data(), n, x), rhs = (mul(eval(q.data(), q.size(), x), eval(g.data(), m, x)) + eval(r.data(), r.size(), x)) % P;
        if (lhs != rhs) {
            fail("large n=" + std::to_string(n) + " m=" + std::to_string(m) + " " + describe(p));
            break;
        }
    }
}

void arguments() {
    const Vec f = make(10, 0), g = {1, 2, 0};
    Vec q(8), r(2);
    bool threw = false;
    try {
        fps::division::divide(f, g, q, r);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) fail("g[m - 1] = 0 accepted");
    const Vec g2 = {1, 2, 3};
    Vec q2(7);
    threw = false;
    try {
        fps::division::divide(f, g2, q2, r);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) fail("a short q accepted");
    // n < m: q empty, r = f then zeros
    const Vec f3 = {5, 6};
    Vec r3(4, 9);
    fps::division::divide(f3, Vec{1, 1, 1, 1, 1}, std::span<u32>(), r3);
    if (r3 != Vec{5, 6, 0, 0}) fail("n < m");
}

}  // namespace

int main() {
    arguments();
    // Small: every n, m up to 40 through the API (long division and the smallest blocks).
    for (std::size_t m = 1; m <= 40; ++m)
        for (std::size_t n = 0; n <= 40; ++n) public_api(n, m, int((n + m) % 3));
    // Every plan on sizes around the block sizes and the transforms' lengths.
    const std::size_t degrees[] = {1, 2, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 700, 1023, 1024, 1025};
    const std::size_t quotients[] = {1, 2, 17, 31, 32, 33, 64, 65, 96, 127, 128, 129, 200, 255, 256, 257, 511, 513, 1000, 2047, 2049};
    int kind = 0;
    for (const std::size_t d : degrees)
        for (const std::size_t k : quotients) {
            if (k * d > 300000) continue;
            every_plan(d + k, d + 1, kind);
            kind = (kind + 1) % 3;
        }
    for (int i = 0; i < 30; ++i) {
        const std::size_t m = 1 + rng() % 1500, n = m + rng() % 2000;
        every_plan(n, m, int(i % 3));
        public_api(n, m, int(i % 3));
    }
#if !defined(__SANITIZE_ADDRESS__)
    const std::size_t big[][2] = {{1 << 18, 1 << 17}, {(1 << 18) + 5, 1 << 12}, {(1 << 20), (1 << 20) - 3}, {500000, 53336},
                                  {500000, 277012}, {455000, 92222}, {(1 << 20) + 1, 9}, {1 << 21, (1 << 15) + 1}};
#else
    const std::size_t big[][2] = {{1 << 16, 1 << 15}, {(1 << 16) + 5, 1 << 10}, {100000, 30001}, {70000, 9}};
#endif
    for (const auto& [n, m] : big) large(n, m);
    // The bench's shapes (bench.cpp): g of n / 2, n / 64 and 9 words.
    for (int lg = 7; lg <= 16; ++lg) {
        const std::size_t n = std::size_t(1) << lg;
        large(n, n / 2), large(n, 9);
        if (lg >= 11) large(n, n / 64);
    }
    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
