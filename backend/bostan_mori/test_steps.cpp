// Tests of backend/bostan_mori's small steps (#198): the tiny steps on every shape up to kTinyMax
// words and one past it, and the schoolbook steps on every simd struct, with q[0] = 1 (no inverse
// at the end) and q[0] != 1, against the recurrence (k < 80) and the products variant (any k):
// p shorter than, as long as and longer than q, zero top words, random, mostly zero and all P - 1
// words; the schoolbook steps right after a wider call (its stack arrays hold that call's words
// where it does not clear them).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "backend/bostan_mori/bostan_mori.hpp"
#include "backend/simd/select.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
namespace bm = fps::bostan_mori;
constexpr u32 P = 998244353;

int failures = 0;

void check(bool ok, const char* what, int line, u64 a = 0, u64 b = 0, u64 c = 0) {
    if (ok) return;
    if (++failures <= 30)
        std::printf("FAIL line %d: %s (%llu, %llu, %llu)\n", line, what, (unsigned long long)a, (unsigned long long)b,
                    (unsigned long long)c);
}
#define CHECK(cond, ...) check(bool(cond), #cond, __LINE__ __VA_OPT__(, ) __VA_ARGS__)

std::mt19937_64 rng(198);

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 sub(u32 a, u32 b) { return a >= b ? a - b : a + P - b; }
u32 inv(u32 a) {
    u32 r = 1;
    for (u64 e = P - 2; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}

// Kinds: 0 random, 1 mostly zeros, 2 all P - 1. first: a[0] if nonzero.
Vec random_vec(std::size_t n, int kind, u32 first = 0) {
    Vec a(n);
    for (u32& x : a) {
        if (kind == 0) x = u32(rng() % P);
        else if (kind == 1) x = rng() % 4 == 0 ? u32(rng() % P) : 0;
        else x = P - 1;
    }
    if (first && n) a[0] = first;
    return a;
}

// The first n terms of p / q by the recurrence.
Vec series(const Vec& p, const Vec& q, std::size_t n) {
    Vec s(n);
    const u32 qi = inv(q[0]);
    for (std::size_t i = 0; i < n; ++i) {
        u32 x = i < p.size() ? p[i] : 0;
        for (std::size_t j = 1; j < q.size() && j <= i; ++j) x = sub(x, mul(q[j], s[i - j]));
        s[i] = mul(x, qi);
    }
    return s;
}

const std::vector<u64> kHuge = {80, 81, 127, 128, (u64(1) << 20) - 1, u64(1) << 32, (u64(1) << 40) + 1,
                                1'000'000'000'000'000'000ull, (u64(1) << 63) - 1, u64(1) << 63, ~u64(0)};

// The tiny steps on (p, q) cut by k, against `want`.
void check_tiny(const Vec& p, const Vec& q, u64 k, u32 want, int line) {
    const auto [a, b] = bm::detail::trim(p, q, k);
    if (a.empty()) return;
    if (std::max(a.size(), b.size()) <= bm::detail::kTinyMax)
        check(bm::detail::tiny(a, b, k) == want, "tiny", line, p.size(), q.size(), k);
    check(bm::variant::schoolbook(p, q, k) == want, "schoolbook", line, p.size(), q.size(), k);
    check(bm::coefficient(p, q, k) == want, "coefficient", line, p.size(), q.size(), k);
}

// Every shape up to kTinyMax + 1 words, also untrimmed (zero top words of p and q).
void tiny_shapes() {
    const std::size_t top = bm::detail::kTinyMax + 1;
    for (std::size_t np = 1; np <= top; ++np)
        for (std::size_t nq = 1; nq <= top; ++nq)
            for (int kind = 0; kind < 3; ++kind)
                for (u32 first : {1u, 0u}) {
                    Vec p = random_vec(np, kind), q = random_vec(nq, kind, first ? first : 1 + u32(rng() % (P - 1)));
                    const Vec s = series(p, q, 80);
                    for (u64 k = 0; k < 80; ++k) check_tiny(p, q, k, s[k], __LINE__);
                    for (u64 k : kHuge) check_tiny(p, q, k, bm::variant::products(p, q, k), __LINE__);
                    // zero top words: the shape is that of the sizes passed in
                    const auto [a, b] = bm::detail::trim(p, q, ~u64(0));
                    if (!a.empty() && std::max(a.size(), b.size()) < bm::detail::kTinyMax) {
                        Vec pz(a.begin(), a.end()), qz(b.begin(), b.end());
                        pz.push_back(0), qz.push_back(0);
                        for (u64 k : {u64(5), u64(77), kHuge[7]})
                            CHECK(bm::detail::tiny(pz, qz, k) == bm::variant::products(p, q, k), np, nq, k);
                    }
                }
}

// Linear recurrences (q[0] = 1): Fibonacci, tribonacci-like and order 7 with sparse words.
void recurrences() {
    const Vec fib_p = {0, 1}, fib_q = {1, P - 1, P - 1};
    const Vec trib_p = {0, 0, 1}, trib_q = {1, P - 1, P - 1, P - 1};
    Vec sparse_q(8), sparse_p(7);
    sparse_q[0] = 1, sparse_q[7] = P - 3, sparse_p[0] = 2, sparse_p[6] = 5;
    for (const auto& [p, q] : {std::pair{fib_p, fib_q}, std::pair{trib_p, trib_q}, std::pair{sparse_p, sparse_q}}) {
        const Vec s = series(p, q, 80);
        for (u64 k = 0; k < 80; ++k) check_tiny(p, q, k, s[k], __LINE__);
        for (u64 k : kHuge) check_tiny(p, q, k, bm::variant::products(p, q, k), __LINE__);
    }
}

// The schoolbook steps on every struct, q[0] = 1 and not, each after a call of the full width.
template <class S>
void schoolbook_struct() {
    if constexpr (S::lanes <= 8) {
        using Book = bm::detail::Schoolbook<S>;
        const std::size_t cap = bm::kSchoolbookCapacity;
        const Vec wide_p = random_vec(cap, 0), wide_q = random_vec(cap, 0, 7);
        for (std::size_t n : {1, 2, 3, 5, 8, 9, 15, 16, 17, 31, 40, 64, 97, 128, 200, 255, 256})
            for (int kind = 0; kind < 3; ++kind)
                for (u32 first : {1u, 0u})
                    for (int shape = 0; shape < 3; ++shape) {
                        const std::size_t np = shape == 0 ? n : shape == 1 ? std::max<std::size_t>(n / 2, 1) : n;
                        const std::size_t nq = shape == 0 ? n : shape == 1 ? n : std::max<std::size_t>(n / 3, 1);
                        const Vec p = random_vec(np, kind), q = random_vec(nq, kind, first ? first : 1 + u32(rng() % (P - 1)));
                        for (u64 k : {u64(1), u64(np + nq), kHuge[7], ~u64(0)}) {
                            const auto [a, b] = bm::detail::trim(p, q, k);
                            if (a.empty()) continue;
                            if (std::max(a.size(), b.size()) > cap) continue;
                            const u32 want = bm::variant::products(p, q, k);
                            Book::run(wide_p, wide_q, ~u64(0));  // leaves its words on the stack
                            CHECK(Book::run(a, b, k) == want, np, nq, k);
                        }
                    }
    }
}

}  // namespace

int main() {
    tiny_shapes();
    recurrences();
    fps::simd::for_each(fps::simd::All{}, []<class S> { schoolbook_struct<S>(); });
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("bostan_mori steps: all tests passed\n");
    return 0;
}
