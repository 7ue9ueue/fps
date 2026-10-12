// Shared by backend/ntt's tests (test*.cpp, one binary each): the checks, references and
// generators, and for_each_variant (every variant this machine runs). Each test file is a binary
// of its own, so a new area adds a file instead of editing a shared one (issue #116).
#pragma once

#include "backend/ntt/ntt.hpp"
#include "backend/ntt/tune.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <new>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using namespace fps::ntt;

constexpr u32 P = kModulus;
constexpr int kScalarMaxLog = 20;  // the scalar variant is slow; ASan makes it slower
// Under ASan every variant stops at 2^20: above it the run took 4x as long (156 s against 36 s on
// lc-zen5-test). -O2 builds test the vector variants up to 2^kMaxLog.
#if __has_feature(address_sanitizer)  // GCC 14+ and clang
constexpr int kVectorMaxLog = 20;
#else
constexpr int kVectorMaxLog = kMaxLog;
#endif

inline std::mt19937_64 rng(2026);
inline int failures = 0;

inline void expect(bool ok, const char* what, u64 detail = 0) {
    if (ok) return;
    if (++failures <= 20) std::printf("FAIL: %s (%llu)\n", what, static_cast<unsigned long long>(detail));
}

inline u32 random_below(u64 bound) { return u32(rng() % bound); }

inline u32 mul(u32 x, u32 y) { return u32(u64(x) * y % P); }
inline u32 add(u32 x, u32 y) { return u32((u64(x) + y) % P); }
inline u32 sub(u32 x, u32 y) { return u32((u64(x) % P + P - y % P) % P); }
inline u32 power(u32 x, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, x = mul(x, x))
        if (e & 1) r = mul(r, x);
    return r;
}

// count words, 32-byte aligned, then kPadding words: exactly what the API asks for, so ASan sees
// any read past it.
struct Words {
    struct alignas(32) Lanes {
        u32 lane[8];
    };
    explicit Words(std::size_t count) : data((count + kPadding) / 8) {}
    u32* begin() { return data.data()->lane; }
    std::vector<Lanes> data;
};

inline std::vector<u32> random_poly(std::size_t n, int kind) {
    std::vector<u32> p(n);
    for (auto& x : p) x = kind == 0 ? random_below(P) : kind == 1 ? P - 1 : kind == 2 ? 0 : random_below(10);
    return p;
}

// r(k) from its definition: the product of 3^((P - 1) / 2^(j + 2)) over the set bits j of k.
inline u32 root(u64 k) {
    u32 r = 1;
    for (int j = 0; k >> j; ++j)
        if (k >> j & 1) r = mul(r, power(3, (P - 1) >> (j + 2)));
    return r;
}

// f mod (x^8 - w_v) for every leaf v, computed directly: w_v = (-1)^v r(v / 2).
inline std::vector<u32> leaf_form(const std::vector<u32>& f) {
    std::vector<u32> out(f.size());
    for (std::size_t v = 0; v < f.size() / 8; ++v) {
        const u32 w = v % 2 ? sub(0, root(v / 2)) : root(v / 2);
        u32 wi = 1;
        for (std::size_t i = 0; i < f.size(); i += 8, wi = mul(wi, w))
            for (int l = 0; l < 8; ++l) out[8 * v + l] = add(out[8 * v + l], mul(f[i + l], wi));
    }
    return out;
}

inline std::vector<u32> run_forward(const Transform& t, const std::vector<u32>& f, auto variant) {
    Words w(f.size());
    std::copy(f.begin(), f.end(), w.begin());
    t.forward<decltype(variant)>(w.begin());
    return {w.begin(), w.begin() + f.size()};
}

inline std::vector<u32> run_inverse(const Transform& t, const std::vector<u32>& f, auto variant) {
    Words w(f.size());
    std::copy(f.begin(), f.end(), w.begin());
    t.inverse<decltype(variant)>(w.begin());
    return {w.begin(), w.begin() + f.size()};
}

// Forward against the leaf form (small lengths) and the scalar model; inverse undoes it.
template <class V>
inline void test_transforms(const char* name, int max_lg) {
    for (int lg = kMinLog; lg <= max_lg; ++lg) {
        const Transform t(lg);
        for (int kind = 0; kind < 4; ++kind) {
            if (kind > 1 && lg > 12) break;
            const auto f = random_poly(t.length(), kind);
            const auto leaves = run_forward(t, f, V{});
            if (lg <= 11) expect(leaves == leaf_form(f), name, lg);
            if (lg <= 18) expect(leaves == run_forward(t, f, variant::scalar{}), name, lg);
            expect(run_inverse(t, leaves, V{}) == f, name, lg);
            if (lg <= 18) {
                const auto g = random_poly(t.length(), kind);
                expect(run_inverse(t, g, V{}) == run_inverse(t, g, variant::scalar{}), name, lg);
            }
            if (lg > kMinLog) {  // forward_upper: from f mod (x^(L/2) + 1), the upper half of f's leaves
                const std::size_t half = t.length() / 2;
                Words h(half);
                for (std::size_t i = 0; i < half; ++i) h.begin()[i] = sub(f[i], f[i + half]);
                t.forward_upper<V>(h.begin());
                expect(std::equal(h.begin(), h.begin() + half, leaves.begin() + half), name, 100 + lg);
            }
        }
    }
}

inline std::vector<u32> schoolbook(const std::vector<u32>& a, const std::vector<u32>& b) {
    if (a.empty() || b.empty()) return {};
    std::vector<u64> c(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = (c[i + j] + u64(a[i]) * b[j]) % P;
    return {c.begin(), c.end()};
}

inline u32 evaluate(const std::vector<u32>& p, u32 x) {
    u32 r = 0;
    for (std::size_t i = p.size(); i--;) r = add(mul(r, x), p[i]);
    return r;
}

template <class V>
inline void test_product(const char* name, std::size_t n, std::size_t m, int kind, bool exact) {
    const auto a = random_poly(n, kind), b = random_poly(m, kind);
    Convolution conv(n, m);
    std::copy(a.begin(), a.end(), conv.a());
    std::copy(b.begin(), b.end(), conv.b());
    const u32* p = conv.template multiply<V>();
    const std::vector<u32> c(p, p + conv.size());
    bool ok = c.size() == (n && m ? n + m - 1 : 0);
    for (const u32 x : c) ok &= x < P;
    if (exact) {
        ok &= c == schoolbook(a, b);
    } else {
        for (int i = 0; i < 3; ++i) {
            const u32 x = random_below(P);
            ok &= evaluate(c, x) == mul(evaluate(a, x), evaluate(b, x));
        }
    }
    expect(ok, name, n * 100000000 + m);
}

template <class V>
inline void test_products(const char* name, int max_lg) {
    for (std::size_t n = 0; n <= 40; ++n)
        for (std::size_t m = 0; m <= 40; m += 3) test_product<V>(name, n, m, 0, true);
    for (int round = 0; round < 60; ++round)
        test_product<V>(name, 1 + random_below(1500), 1 + random_below(1500), round % 4, true);
    // Every transform length: sparse and dense first levels, factors at and around half the length.
    for (int lg = kMinLog; lg <= max_lg; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        const bool exact = lg <= 11;
        test_product<V>(name, len / 2, len / 2, lg % 2, exact);
        test_product<V>(name, len / 2 + 1, len / 2 - 1, 0, exact);
        test_product<V>(name, len - 1, 2, 0, exact);
        test_product<V>(name, len / 2, 2, 0, exact);
        if (lg <= 22) {
            test_product<V>(name, len / 4 + 1, 1, 3, exact);
            test_product<V>(name, len / 4 + 1, len / 4 + 1, 3, exact);
            test_product<V>(name, len / 2 - 5, len / 4 + 7, 0, exact);
        }
    }
}

// Factors of at most half the length with garbage in their upper halves, which must not matter.
template <class V>
inline void test_upper_halves_unread(const char* name, std::size_t n, std::size_t m) {
    const auto a = random_poly(n, 0), b = random_poly(m, 0);
    Convolution conv(n, m);
    std::copy(a.begin(), a.end(), conv.a());
    std::copy(b.begin(), b.end(), conv.b());
    const std::size_t half = conv.length() / 2;
    for (std::size_t i = half; i < 2 * half; ++i) conv.a()[i] = u32(rng()), conv.b()[i] = u32(rng());
    const u32* c = conv.template multiply<V>();
    expect(std::vector<u32>(c, c + n + m - 1) == schoolbook(a, b), name, n * 100000000 + m);
}

// count words `offset` words past a 4 KiB boundary, then `padding` words that end the allocation,
// so ASan sees an access past them.
class PageWords {
public:
    PageWords(std::size_t count, std::size_t offset, std::size_t padding = kPadding)
        : count_(count), offset_(offset),
          words_(static_cast<u32*>(::operator new((offset + count + padding) * sizeof(u32), std::align_val_t(4096)))) {}
    ~PageWords() { ::operator delete(words_, std::align_val_t(4096)); }
    PageWords(const PageWords&) = delete;
    PageWords& operator=(const PageWords&) = delete;
    u32* begin() { return words_ + offset_; }
    std::span<u32> span() { return {begin(), count_}; }

private:
    std::size_t count_, offset_;
    u32* words_;
};

// The words [0, count) of the product of f and g, zeros past it.
inline std::vector<u32> cut(const std::vector<u32>& product, std::size_t count) {
    std::vector<u32> c(count);
    std::copy_n(product.begin(), std::min(count, product.size()), c.begin());
    return c;
}

// multiply(f, g, out) for out of each count, against `product` (f g): f, g and out each `offset`
// words or more into a page and ending their allocation (ASan sees any access past them), out full
// of garbage; f and g unchanged.
template <class V>
inline void test_spans(const char* name, const std::vector<u32>& f, const std::vector<u32>& g,
                const std::vector<u32>& product, std::initializer_list<std::size_t> counts, std::size_t offset) {
    PageWords fw(f.size(), offset, 0), gw(g.size(), offset + 5, 0);
    std::copy(f.begin(), f.end(), fw.begin());
    std::copy(g.begin(), g.end(), gw.begin());
    for (const std::size_t count : counts) {
        PageWords out(count, offset + 3, 0);
        for (u32& w : out.span()) w = u32(rng());
        multiply<V>(fw.span(), gw.span(), out.span());
        bool ok = std::ranges::equal(out.span(), cut(product, count));
        ok &= std::ranges::equal(fw.span(), f) && std::ranges::equal(gw.span(), g);
        expect(ok, name, (f.size() * 100000 + g.size()) * 1000 + count % 1000);
    }
}

// out in the memory of f, of g, or of both: every variant reads the factors before it writes out.
template <class V>
inline void test_spans_overlap(const char* name, std::size_t n, std::size_t m) {
    const auto f = random_poly(n, 0), g = random_poly(m, 0), product = schoolbook(f, g);
    for (const std::size_t count : {n, n + m - 1, n / 2 + 1}) {
        std::vector<u32> both(n + m + count);
        std::copy(f.begin(), f.end(), both.begin());  // out at f
        std::copy(g.begin(), g.end(), both.begin() + n);
        multiply<V>(std::span(both).first(n), std::span(both).subspan(n, m), std::span(both).first(count));
        expect(std::ranges::equal(std::span(both).first(count), cut(product, count)), name, 1);
        std::copy(f.begin(), f.end(), both.begin());  // out over the end of f and the start of g
        std::copy(g.begin(), g.end(), both.begin() + n);
        multiply<V>(std::span(both).first(n), std::span(both).subspan(n, m), std::span(both).subspan(n / 2, count));
        expect(std::ranges::equal(std::span(both).subspan(n / 2, count), cut(product, count)), name, 2);
    }
    std::vector<u32> square(f);  // out = f = g
    multiply<V>(square, square, square);
    expect(square == cut(schoolbook(f, f), n), name, 3);
}

// multiply(f, g, out): sizes around each level's blocks, cuts, zero padding, alignments, overlap.
template <class V>
inline void test_multiply_spans(const char* name, int max_lg) {
    for (std::size_t n = 0; n <= 24; ++n)
        for (std::size_t m = 0; m <= 24; m += 5) {
            const auto f = random_poly(n, 1), g = random_poly(m, 1), product = schoolbook(f, g);
            const std::size_t size = n + m ? n + m - 1 : 0;
            test_spans<V>(name, f, g, product, {0, 1, n, size, size + 1, size + 9}, n % 4);
        }
    using Shapes = std::vector<std::pair<std::size_t, std::size_t>>;
    for (int lg = kMinLog; lg <= max_lg; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        const bool exact = lg <= 11;
        const Shapes shapes = lg <= 16 ? Shapes{{len / 2, len / 2}, {len / 2 + 1, len / 2 - 1}, {len - 1, 2},
                                                {len - len / 4, len / 4 + 1}, {len / 4 + 1, len / 4 + 1},
                                                {len / 8 + 3, len / 8 - 2}, {len / 2 - 17, len / 4 + 9}}
                                       : Shapes{{len / 2, len / 2}, {len - len / 4, len / 4 + 1}, {len / 2 - 17, len / 4 + 9}};
        for (const auto& [n, m] : shapes) {
            const auto f = random_poly(n, lg % 2), g = random_poly(m, lg % 2);
            std::vector<u32> product;
            if (exact) {
                product = schoolbook(f, g);
            } else {  // checked against evaluation, then the reference for its cuts
                Convolution conv(n, m);
                std::copy(f.begin(), f.end(), conv.a());
                std::copy(g.begin(), g.end(), conv.b());
                const u32* c = conv.template multiply<V>();
                product.assign(c, c + conv.size());
                bool ok = true;
                for (int i = 0; i < 2; ++i) {
                    const u32 x = random_below(P);
                    ok &= evaluate(product, x) == mul(evaluate(f, x), evaluate(g, x));
                }
                expect(ok, name, lg);
            }
            const std::size_t size = n + m - 1, offset = lg % 3 == 0 ? 0 : lg % 3 == 1 ? 1 : 16 + lg;
            if (lg <= 16) test_spans<V>(name, f, g, product, {size, len / 2, len / 2 + 1, len / 4 + 3, 9, size + 3}, offset);
            else test_spans<V>(name, f, g, product, {size, len / 2 + 7}, offset);
        }
    }
    test_spans_overlap<V>(name, 40, 30);
    test_spans_overlap<V>(name, 1000, 700);
}

// The product of f and g: schoolbook below 2^12 words, else multiply<V> checked by evaluation.
template <class V>
inline std::vector<u32> reference_product(const char* name, const std::vector<u32>& f, const std::vector<u32>& g) {
    if (f.size() + g.size() <= 4096) return schoolbook(f, g);
    std::vector<u32> c(f.size() + g.size() - 1);
    multiply<V>(f, g, c);
    for (int i = 0; i < 2; ++i) {
        const u32 x = random_below(P);
        expect(evaluate(c, x) == mul(evaluate(f, x), evaluate(g, x)), name, f.size());
    }
    return c;
}

// f.operator()<V>(name, max_lg) for every variant this machine runs: scalar, intrinsics<S> for
// every simd struct (Scalar is slow, so shorter), avx2.
template <class F>
inline void for_each_variant(F f) {
    f.template operator()<variant::scalar>("scalar", kScalarMaxLog);
    fps::simd::for_each(fps::simd::All{}, [&]<class S> {
        const std::string name = std::string("intrinsics ") + S::name;
        f.template operator()<variant::intrinsics<S>>(name.c_str(), S::lanes == 1 ? kScalarMaxLog : kVectorMaxLog);
    });
#ifdef __AVX2__
    f.template operator()<variant::avx2>("avx2", kVectorMaxLog);
#endif
}

inline int finish() {
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

