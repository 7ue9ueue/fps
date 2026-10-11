// Tests of fps/factorials.hpp: the tables against running products, binomials against Pascal's
// triangle and products of k factors. Factorials<P> exists for P = 998244353 only (DESIGN.md);
// the tables it wraps (detail::factorial_tables) are also tested on small primes, up to n = P.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "fps/fps.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 P = 998244353;

int failures = 0;

void check(bool ok, const char* what, u64 p, u64 n, u64 i) {
    if (ok) return;
    if (++failures <= 10)
        std::printf("FAIL %s (P %llu, n %llu, i %llu)\n", what, (unsigned long long)p, (unsigned long long)n,
                    (unsigned long long)i);
}

// fact[i] against the running product i! mod Q; inverse[i] is the one value in [0, Q) with
// i! inverse[i] = 1.
template <u32 Q>
void check_tables(const std::vector<u32>& fact, const std::vector<u32>& inverse) {
    const std::size_t n = fact.size();
    u64 f = 1;
    for (std::size_t i = 0; i < n; ++i) {
        if (i) f = f * i % Q;
        check(fact[i] == f, "fact[i] = i!", Q, n, i);
        check(inverse[i] < Q && inverse[i] * f % Q == 1, "inverse(i) i! = 1", Q, n, i);
    }
}

template <u32 Q>
void test_detail(std::size_t n) {
    std::vector<u32> fact(n), inverse(n);
    fps::detail::factorial_tables<Q>(fact, inverse);
    check_tables<Q>(fact, inverse);
}

// Every n <= Q, up to the largest table, n = Q: there (Q - 1)! = -1 (Wilson).
template <u32 Q>
void test_small_prime() {
    for (std::size_t n = 0; n <= Q; ++n) test_detail<Q>(n);
    std::vector<u32> fact(Q), inverse(Q);
    fps::detail::factorial_tables<Q>(fact, inverse);
    check(fact[Q - 1] == Q - 1, "(Q - 1)! = -1", Q, Q, Q - 1);
}

void test_tables(std::size_t n) {
    const fps::Factorials<P> fact(n);
    check(fact.size() == n, "size", P, n, 0);
    std::vector<u32> f(n), inverse(n);
    for (std::size_t i = 0; i < n; ++i) f[i] = fact[i], inverse[i] = fact.inverse(i);
    check_tables<P>(f, inverse);
}

// C(m, k) = m (m - 1) ... (m - k + 1) / k!, for m < P.
u32 binomial_by_products(u64 m, u64 k) {
    if (k > m) return 0;
    k = std::min(k, m - k);
    u64 top = 1, bottom = 1;
    for (u64 j = 0; j < k; ++j) top = top * (m - j) % P, bottom = bottom * (j + 1) % P;
    return fps::mod_mul<P>(u32(top), fps::mod_inverse<P>(u32(bottom)));
}

// Every C(m, k), m < rows, k <= m + 2, against Pascal's triangle.
void test_pascal(std::size_t rows) {
    const fps::Factorials<P> fact(rows);
    std::vector<u32> row = {1};
    for (std::size_t m = 0; m < rows; ++m) {
        for (std::size_t k = 0; k <= m; ++k) check(fact.binomial(m, k) == row[k], "binomial = Pascal", P, m, k);
        check(fact.binomial(m, m + 1) == 0 && fact.binomial(m, m + 2) == 0, "binomial k > m", P, m, m + 1);
        std::vector<u32> next(m + 2, 1);
        for (std::size_t k = 1; k <= m; ++k) next[k] = u32((u64(row[k - 1]) + row[k]) % P);
        row = std::move(next);
    }
}

// Random C(m, k), m < n, against products of k factors; the edges k = 0, m, m + 1 and the largest k.
void test_random_binomials(std::size_t n, std::mt19937_64& rng) {
    if (n == 0) return;
    const fps::Factorials<P> fact(n);
    for (int t = 0; t < 100; ++t) {
        const u64 m = rng() % n, k = rng() % (m + 2);
        check(fact.binomial(m, k) == binomial_by_products(m, k), "binomial random", P, m, k);
    }
    const u64 m = n - 1;
    check(fact.binomial(m, 0) == 1 && fact.binomial(m, m) == 1, "binomial k = 0, m", P, m, 0);
    check(fact.binomial(m, m + 1) == 0, "binomial k = m + 1", P, m, m + 1);
    check(fact.binomial(m, std::numeric_limits<std::size_t>::max()) == 0, "binomial k = max", P, m, 0);
}

void test_throws(std::size_t n) {
    bool threw = false;
    try {
        fps::Factorials<P> fact(n);
    } catch (const std::domain_error& e) {
        threw = std::string_view(e.what()).starts_with("fps::Factorials");  // not later, from mod_inverse
    }
    check(threw, "n > P throws", P, n, 0);
}

}  // namespace

int main() {
    test_small_prime<2>();
    test_small_prime<3>();
    test_small_prime<5>();
    test_small_prime<101>();
    for (std::size_t n : {0, 1, 2, 4097}) test_detail<4294967291>(n);  // largest prime below 2^32

    std::mt19937_64 rng(1);
    std::vector<std::size_t> sizes = {0, 1, 2, 3};
    for (int lg = 2; lg <= 20; ++lg) {
        const std::size_t power = std::size_t(1) << lg;
        sizes.insert(sizes.end(), {power - 1, power, power + 1});
    }
    for (int t = 0; t < 10; ++t) sizes.push_back(rng() % 5000);
    for (std::size_t n : sizes) {
        test_tables(n);
        test_random_binomials(n, rng);
    }
    test_pascal(300);
    test_throws(std::size_t(P) + 1);
    test_throws(std::numeric_limits<std::size_t>::max());
    if (failures) std::printf("%d failures\n", failures);
    return failures != 0;
}
