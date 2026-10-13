// Tests of backend/taylor_shift. shift(f, c, out) against the binomial sums
// out[k] = sum_(i >= k) f_i C(i, k) c^(i-k), O(n^2), for n in 0 .. 200, powers of two and their
// neighbours up to 2^12 and random sizes; beyond, out(x0) = f(x0 + c) at random points. c in
// {0, 1, 2, P - 1, P + 3 (taken mod P)} and random; inputs random or all P - 1; out shorter than f
// (0, 1, the lanes' minimum and around it, n / 2, n - 1); out over f (in place), out at word
// offsets 0 .. 3 with guard words around it. Every scan variant the machine runs (Scan<S, K, step>,
// S in simd::All, K in {1, 2, 4, 8} where K S::lanes >= 2, both chain steps), also with out over f
// and out = f + 1, and Horner on every struct. The argument checks. Exits nonzero on a failure.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/taylor_shift/taylor_shift.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
namespace ts = fps::taylor_shift;
constexpr u32 P = ts::kP;
constexpr u32 kGuard = 0xdeadbeef;

int failures = 0;
long checks = 0;
std::mt19937_64 rng(20261013);

void fail(const std::string& what) {
    if (++failures <= 30) std::printf("FAIL %s\n", what.c_str());
}

void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) fail(what);
}

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 random_word() { return u32(rng() % P); }

Vec random_vec(std::size_t n) {
    Vec f(n);
    for (u32& x : f) x = random_word();
    return f;
}

u32 horner(const Vec& f, u32 x) {
    u64 r = 0;
    for (std::size_t i = f.size(); i-- > 0;) r = (r * x + f[i]) % P;
    return u32(r);
}

// out[k] = sum_(i >= k) f_i C(i, k) c^(i-k), k < t: row by row of Pascal's triangle times powers.
Vec reference(const Vec& f, u32 c, std::size_t t) {
    const std::size_t n = f.size();
    c %= P;
    Vec out(t, 0), row{1};  // row = C(i, k), k <= i
    Vec powers(n + 1, 1);
    for (std::size_t j = 1; j <= n; ++j) powers[j] = mul(powers[j - 1], c);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k <= i && k < t; ++k) out[k] = u32((out[k] + u64(mul(f[i], row[k])) * powers[i - k]) % P);
        Vec next(i + 2, 1);
        for (std::size_t k = 1; k <= i; ++k) next[k] = (row[k - 1] + row[k]) % P;
        row = std::move(next);
    }
    return out;
}

// out against f by the reference (n <= 1024) or at points: out(x0) = f(x0 + c) when t = n. For
// t < n, against the full shift (given, or by shift()) cut to t, the full one checked at points.
using Run = void (*)(std::span<const u32>, u32, std::span<u32>);

bool agrees(const Vec& f, u32 c, const Vec& out, const Vec* full = nullptr) {
    const std::size_t n = f.size(), t = out.size();
    if (n <= 1024) return out == reference(f, c, t);
    if (t < n) {
        Vec own;
        if (!full) {
            own.resize(n);
            ts::shift(f, c, own);
            full = &own;
        }
        return agrees(f, c, *full) && std::equal(out.begin(), out.end(), full->begin());
    }
    for (int k = 0; k < 3; ++k) {
        const u32 x0 = random_word();
        if (horner(out, x0) != horner(f, u32((x0 + u64(c)) % P))) return false;
    }
    return true;
}

std::string describe(const char* name, std::size_t n, std::size_t t, u32 c, int offset) {
    return std::string(name) + " n=" + std::to_string(n) + " t=" + std::to_string(t) + " c=" + std::to_string(c) +
           " offset=" + std::to_string(offset);
}

// One call into a guarded buffer at a word offset; checks the guards and the result.
void run_case(const char* name, Run run, const Vec& f, u32 c, std::size_t t, int offset, const Vec* full = nullptr) {
    Vec buffer(t + offset + 8, kGuard);
    run(f, c, std::span<u32>(buffer.data() + offset, t));
    const Vec out(buffer.begin() + offset, buffer.begin() + offset + t);
    bool guards = true;
    for (int i = 0; i < offset; ++i) guards &= buffer[i] == kGuard;
    for (std::size_t i = offset + t; i < buffer.size(); ++i) guards &= buffer[i] == kGuard;
    check(guards, describe(name, f.size(), t, c, offset) + ": guard words");
    check(agrees(f, c, out, full), describe(name, f.size(), t, c, offset));
}

std::vector<u32> constants() { return {0, 1, 2, P - 1, P + 3, random_word(), random_word()}; }

// The sizes: 0 .. 200, powers of two and neighbours to 2^12, random sizes.
std::vector<std::size_t> sizes(std::size_t largest) {
    std::vector<std::size_t> s;
    for (std::size_t n = 0; n <= 200; ++n) s.push_back(n);
    for (std::size_t p = 256; p <= largest; p *= 2)
        for (std::size_t n : {p - 1, p, p + 1}) s.push_back(n);
    for (int i = 0; i < 6; ++i) s.push_back(201 + rng() % (largest - 200));
    return s;
}

void test_shift() {
    for (std::size_t n : sizes(4096)) {
        const Vec f = random_vec(n);
        for (u32 c : n <= 200 ? Vec{0, 1, P - 1, random_word()} : constants()) {
            run_case("shift", ts::shift, f, c, n, int(n % 4));
            if (n > 0) run_case("shift", ts::shift, f, c, n / 2, 1);
            if (n > 1) run_case("shift", ts::shift, f, c, n - 1, 0);
        }
    }
    // All words P - 1, and t around the lanes' minimum.
    for (std::size_t n : {65, 100, 513, 1000}) {
        const Vec f(n, P - 1);
        for (u32 c : {1u, P - 1, 12345u}) {
            for (std::size_t t : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(63), std::size_t(64),
                                  std::size_t(65), std::size_t(127), std::size_t(128), n})
                if (t <= n) run_case("shift all P-1", ts::shift, f, c, t, 3);
        }
    }
    // Large n at points; t < n against the full shift cut.
    for (std::size_t n : {std::size_t(1) << 14, (std::size_t(1) << 15) + 1, (std::size_t(1) << 16) - 1, std::size_t(100000)}) {
        const Vec f = random_vec(n);
        const u32 c = random_word();
        Vec full(n);
        ts::shift(f, c, full);
        check(agrees(f, c, full), "shift large n=" + std::to_string(n));
        for (std::size_t t : {std::size_t(1), std::size_t(100), n / 3, n - 1}) run_case("shift large", ts::shift, f, c, t, 2, &full);
    }
}

// In place: out over f.
void test_in_place() {
    for (std::size_t n : {0, 1, 5, 64, 65, 200, 1023, 1024, 1025, 5000}) {
        for (u32 c : {0u, 1u, random_word()}) {
            const Vec f = random_vec(n);
            Vec g = f;
            ts::shift(g, c, g);
            check(agrees(f, c, g), "in place n=" + std::to_string(n) + " c=" + std::to_string(c));
            if (n > 2) {
                Vec h = f, full(n);
                ts::shift(f, c, full);
                ts::shift(h, c, std::span<u32>(h).first(n / 2));  // the first n / 2 words over f
                check(std::equal(h.begin(), h.begin() + n / 2, full.begin()) &&
                          std::equal(h.begin() + n / 2, h.end(), f.begin() + n / 2),
                      "in place, cut n=" + std::to_string(n));
            }
        }
    }
}

// Every scan variant and Horner on every struct this machine runs.
template <class S, int K, fps::factorials::Step kStep>
void test_scan() {
    if constexpr (K * S::lanes >= 2) {
        using Sc = ts::detail::Scan<S, K, kStep>;
        const std::string name = std::string("scan ") + S::name + " K=" + std::to_string(K) +
                                 (kStep == fps::factorials::Step::shoup ? " shoup" : " montgomery");
        const Run run = [](std::span<const u32> f, u32 c, std::span<u32> out) {
            ts::detail::shift_lanes<S, K, kStep>(f.data(), f.size(), c % P, out.data(), out.size());
        };
        for (std::size_t n : {Sc::kMinSize, Sc::kMinSize + 1, std::size_t(100), std::size_t(255), std::size_t(256),
                              std::size_t(257), std::size_t(1000), std::size_t(2048), std::size_t(3001)}) {
            if (n < Sc::kMinSize) continue;
            const Vec f = random_vec(n);
            for (u32 c : {1u, P - 1, random_word()}) {
                run_case(name.c_str(), run, f, c, n, 1);
                run_case(name.c_str(), run, f, c, Sc::kMinSize, 0);
                run_case(name.c_str(), run, f, c, n / 2 + 1, 2);
                run_case(name.c_str(), run, f, c, 3, 0);
                // out over f: whole, its first half, and shifted by a word (out = f + 1).
                for (const std::size_t t : {n, n / 2 + 1}) {
                    Vec g = f;
                    ts::detail::shift_lanes<S, K, kStep>(g.data(), n, c % P, g.data(), t);
                    Vec expected(t);
                    run(f, c, expected);
                    expected.insert(expected.end(), f.begin() + std::ptrdiff_t(t), f.end());
                    check(g == expected, name + " over f n=" + std::to_string(n) + " t=" + std::to_string(t));
                }
                Vec g(n + 1);
                std::copy(f.begin(), f.end(), g.begin());
                ts::detail::shift_lanes<S, K, kStep>(g.data(), n, c % P, g.data() + 1, n);
                Vec expected(n);
                run(f, c, expected);
                check(std::equal(expected.begin(), expected.end(), g.begin() + 1) && g[0] == f[0],
                      name + " out = f + 1, n=" + std::to_string(n));
            }
        }
    }
}

template <class S>
void test_horner() {
    const std::string name = std::string("horner ") + S::name;
    const Run run = [](std::span<const u32> f, u32 c, std::span<u32> out) {
        ts::detail::horner<S>(f.data(), f.size(), c % P, out.data(), out.size());
    };
    for (std::size_t n = 1; n <= ts::kSmall; ++n) {
        const Vec f = random_vec(n);
        for (u32 c : {1u, P - 1, random_word()}) {
            run_case(name.c_str(), run, f, c, n, int(n % 3));
            run_case(name.c_str(), run, f, c, (n + 1) / 2, 0);
        }
    }
    const Vec top(ts::kSmall, P - 1);
    run_case(name.c_str(), run, top, P - 1, ts::kSmall, 0);
}

void test_checks() {
    const Vec f = random_vec(10);
    Vec out(11, kGuard);
    bool thrown = false;
    try {
        ts::shift(f, 3, out);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    check(thrown && std::all_of(out.begin(), out.end(), [](u32 x) { return x == kGuard; }), "out longer than f throws");
    thrown = false;
    try {
        ts::shift(std::span<const u32>(f.data(), ts::kMaxSize + 1), 3, std::span<u32>(out).first(1));
    } catch (const std::length_error&) {
        thrown = true;
    }
    check(thrown && out[0] == kGuard, "n > 2^24 throws");
}

template <class F>
void for_steps(F f) {
    f.template operator()<fps::factorials::Step::montgomery>();
    f.template operator()<fps::factorials::Step::shoup>();
}

}  // namespace

int main() {
    test_checks();
    test_shift();
    test_in_place();
    fps::simd::for_each(fps::simd::All{}, []<class S> {
        test_horner<S>();
        for_steps([]<fps::factorials::Step kStep> {
            test_scan<S, 1, kStep>();
            test_scan<S, 2, kStep>();
            test_scan<S, 4, kStep>();
            if constexpr (S::lanes <= 4) test_scan<S, 8, kStep>();
        });
    });
    std::printf("%s: %ld checks, %d failures\n", failures ? "FAIL" : "pass", checks, failures);
    return failures ? 1 : 0;
}
