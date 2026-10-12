// Tests for backend/ntt. Every variant this machine runs: tables, transforms against the leaf form
// computed directly and against the scalar model, products against schoolbook multiplication and,
// for long ones, evaluation at random points. AVX2: each asm kernel against a scalar model of its
// butterfly, with inputs at the edges of its lazy ranges. Convolution's memory: kept per thread,
// the caller's, several threads. multiply(f, g, out): spans of any alignment, cut and padded
// outputs, overlap. The tuned variant with every candidate per op and length, measure(), wisdom.
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
#include <utility>
#include <vector>

namespace {

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

std::mt19937_64 rng(2026);
int failures = 0;

void expect(bool ok, const char* what, u64 detail = 0) {
    if (ok) return;
    if (++failures <= 20) std::printf("FAIL: %s (%llu)\n", what, static_cast<unsigned long long>(detail));
}

u32 random_below(u64 bound) { return u32(rng() % bound); }

u32 mul(u32 x, u32 y) { return u32(u64(x) * y % P); }
u32 add(u32 x, u32 y) { return u32((u64(x) + y) % P); }
u32 sub(u32 x, u32 y) { return u32((u64(x) % P + P - y % P) % P); }
u32 power(u32 x, u64 e) {
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

std::vector<u32> random_poly(std::size_t n, int kind) {
    std::vector<u32> p(n);
    for (auto& x : p) x = kind == 0 ? random_below(P) : kind == 1 ? P - 1 : kind == 2 ? 0 : random_below(10);
    return p;
}

// r(k) from its definition: the product of 3^((P - 1) / 2^(j + 2)) over the set bits j of k.
u32 root(u64 k) {
    u32 r = 1;
    for (int j = 0; k >> j; ++j)
        if (k >> j & 1) r = mul(r, power(3, (P - 1) >> (j + 2)));
    return r;
}

// Tables against the definition: built within the static prefix and past it; the static tables.
template <class V>
void test_tables(const char* name) {
    for (int lg : {kMinLog, kMinLog + 1, 10, detail::kStaticLog, detail::kStaticLog + 2}) {
        const std::size_t count = (std::size_t(1) << lg) / 16;
        std::vector<Words> t;
        for (int inverse = 0; inverse < 2; ++inverse) {
            t.emplace_back(detail::table_words(lg));
            V::build_table(t.back().begin(), count, inverse);
        }
        for (std::size_t k = 0; k < count; ++k) {
            const u32* e = t[0].begin() + detail::slot(k);
            const u32* ie = t[1].begin() + detail::slot(k);
            expect(e[0] == root(k), name, k);
            expect(mul(e[0], ie[0]) == 1, name, k);
            for (const u32* x : {e, ie}) expect(x[8] == u32((u64(x[0]) << 32) / P), name, k);
            if (k < detail::kStaticCount) {
                for (int inverse = 0; inverse < 2; ++inverse)
                    for (int w : {0, 8})
                        expect(detail::kStaticTables.words[inverse][detail::slot(k) + w] == t[inverse].begin()[detail::slot(k) + w],
                               "static tables", k);
            }
        }
    }
}

// f mod (x^8 - w_v) for every leaf v, computed directly: w_v = (-1)^v r(v / 2).
std::vector<u32> leaf_form(const std::vector<u32>& f) {
    std::vector<u32> out(f.size());
    for (std::size_t v = 0; v < f.size() / 8; ++v) {
        const u32 w = v % 2 ? sub(0, root(v / 2)) : root(v / 2);
        u32 wi = 1;
        for (std::size_t i = 0; i < f.size(); i += 8, wi = mul(wi, w))
            for (int l = 0; l < 8; ++l) out[8 * v + l] = add(out[8 * v + l], mul(f[i + l], wi));
    }
    return out;
}

std::vector<u32> run_forward(const Transform& t, const std::vector<u32>& f, auto variant) {
    Words w(f.size());
    std::copy(f.begin(), f.end(), w.begin());
    t.forward<decltype(variant)>(w.begin());
    return {w.begin(), w.begin() + f.size()};
}

std::vector<u32> run_inverse(const Transform& t, const std::vector<u32>& f, auto variant) {
    Words w(f.size());
    std::copy(f.begin(), f.end(), w.begin());
    t.inverse<decltype(variant)>(w.begin());
    return {w.begin(), w.begin() + f.size()};
}

// Forward against the leaf form (small lengths) and the scalar model; inverse undoes it.
template <class V>
void test_transforms(const char* name, int max_lg) {
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
        }
    }
}

std::vector<u32> schoolbook(const std::vector<u32>& a, const std::vector<u32>& b) {
    if (a.empty() || b.empty()) return {};
    std::vector<u64> c(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = (c[i + j] + u64(a[i]) * b[j]) % P;
    return {c.begin(), c.end()};
}

u32 evaluate(const std::vector<u32>& p, u32 x) {
    u32 r = 0;
    for (std::size_t i = p.size(); i--;) r = add(mul(r, x), p[i]);
    return r;
}

template <class V>
void test_product(const char* name, std::size_t n, std::size_t m, int kind, bool exact) {
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
void test_products(const char* name, int max_lg) {
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
void test_upper_halves_unread(const char* name, std::size_t n, std::size_t m) {
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
std::vector<u32> cut(const std::vector<u32>& product, std::size_t count) {
    std::vector<u32> c(count);
    std::copy_n(product.begin(), std::min(count, product.size()), c.begin());
    return c;
}

// multiply(f, g, out) for out of each count, against `product` (f g): f, g and out each `offset`
// words or more into a page and ending their allocation (ASan sees any access past them), out full
// of garbage; f and g unchanged.
template <class V>
void test_spans(const char* name, const std::vector<u32>& f, const std::vector<u32>& g,
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
void test_spans_overlap(const char* name, std::size_t n, std::size_t m) {
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
void test_multiply_spans(const char* name, int max_lg) {
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
std::vector<u32> reference_product(const char* name, const std::vector<u32>& f, const std::vector<u32>& g) {
    if (f.size() + g.size() <= 4096) return schoolbook(f, g);
    std::vector<u32> c(f.size() + g.size() - 1);
    multiply<V>(f, g, c);
    for (int i = 0; i < 2; ++i) {
        const u32 x = random_below(P);
        expect(evaluate(c, x) == mul(evaluate(f, x), evaluate(g, x)), name, f.size());
    }
    return c;
}

// square(f, out) against `product` (f^2), as test_spans; and out = f.
template <class V>
void test_square(const char* name, const std::vector<u32>& f, const std::vector<u32>& product,
                 std::initializer_list<std::size_t> counts, std::size_t offset) {
    PageWords fw(f.size(), offset, 0);
    std::copy(f.begin(), f.end(), fw.begin());
    for (const std::size_t count : counts) {
        PageWords out(count, offset + 3, 0);
        for (u32& w : out.span()) w = u32(rng());
        square<V>(fw.span(), out.span());
        bool ok = std::ranges::equal(out.span(), cut(product, count)) && std::ranges::equal(fw.span(), f);
        expect(ok, name, f.size() * 1000 + count % 1000);
    }
    std::vector<u32> in_place(f);
    square<V>(in_place, in_place);
    expect(in_place == cut(product, f.size()), name, f.size());
}

// square(f, out): every size to 40, then shapes around each length's levels.
template <class V>
void test_squares(const char* name, int max_lg) {
    for (std::size_t n = 0; n <= 40; ++n) {
        const auto f = random_poly(n, n % 4);
        const std::size_t size = n ? 2 * n - 1 : 0;
        test_square<V>(name, f, schoolbook(f, f), {0, 1, n, size, size + 5}, n % 4);
    }
    for (int lg = kMinLog; lg <= max_lg; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        const std::vector<std::size_t> sizes = lg <= 16 ? std::vector<std::size_t>{len / 2, len / 2 - 1, len / 4 + 1, len / 8 + 3}
                                                        : std::vector<std::size_t>{len / 2, len / 4 + 1};
        for (const std::size_t n : sizes) {
            const auto f = random_poly(n, lg % 2);
            const std::size_t size = 2 * n - 1, offset = lg % 3 == 0 ? 0 : lg % 3 == 1 ? 1 : 16 + lg;
            const auto product = reference_product<V>(name, f, f);
            if (lg <= 16) test_square<V>(name, f, product, {size, n, n / 2 + 1, 9, size + 3}, offset);
            else test_square<V>(name, f, product, {size, n + 7}, offset);
        }
    }
}

// middle_product(f, g, out) for out of each count, against `product` (f g): out[i] = product[m - 1
// + i]. Spans as test_spans; f is cut to the words the call may read (ASan sees any read past
// them). Then out at f and at g.
template <class V>
void test_middle(const char* name, const std::vector<u32>& f, const std::vector<u32>& g,
                 const std::vector<u32>& product, std::initializer_list<std::size_t> counts, std::size_t offset) {
    const std::size_t n = f.size(), m = g.size();
    for (const std::size_t count : counts) {
        const std::size_t read = m - 1 + std::max<std::size_t>(count, 1);
        PageWords fw(read, offset, 0), gw(m, offset + 5, 0), out(count, offset + 3, 0);
        std::copy_n(f.begin(), read, fw.begin());
        std::copy(g.begin(), g.end(), gw.begin());
        for (u32& w : out.span()) w = u32(rng());
        middle_product<V>(fw.span(), gw.span(), out.span());
        bool ok = std::ranges::equal(out.span(), std::span(product).subspan(m - 1, count));
        ok &= std::ranges::equal(gw.span(), g) && std::ranges::equal(fw.span(), std::span(f).first(read));
        expect(ok, name, (n * 100000 + m) * 1000 + count % 1000);
    }
    const std::size_t count = n - m + 1;
    std::vector<u32> both(f);
    both.insert(both.end(), g.begin(), g.end());
    const auto want = std::span(product).subspan(m - 1, count);
    middle_product<V>(std::span(both).first(n), std::span(both).subspan(n, m), std::span(both).first(count));
    expect(std::ranges::equal(std::span(both).first(count), want), name, 1);
    std::copy(f.begin(), f.end(), both.begin());  // out over the end of f and all of g
    middle_product<V>(std::span(both).first(n), std::span(both).subspan(n, m), std::span(both).subspan(n + m - count, count));
    expect(std::ranges::equal(std::span(both).subspan(n + m - count, count), want), name, 2);
}

// middle_product(f, g, out): every n <= 30 and m <= n, then shapes around each length: f of the
// whole length and one word past it, g of 1, half and all of f; the errors.
template <class V>
void test_middles(const char* name, int max_lg) {
    for (std::size_t n = 1; n <= 30; ++n)
        for (std::size_t m = 1; m <= n; ++m) {
            const auto f = random_poly(n, (n + m) % 4), g = random_poly(m, (n + m) % 4);
            const std::size_t all = n - m + 1;
            test_middle<V>(name, f, g, schoolbook(f, g), {0, 1, all, all / 2}, (n + m) % 4);
        }
    for (int lg = kMinLog; lg <= std::min(max_lg, 23); ++lg) {  // the reference product to 2^25
        const std::size_t len = std::size_t(1) << lg;
        using Shapes = std::vector<std::pair<std::size_t, std::size_t>>;
        const Shapes shapes = lg <= 16 ? Shapes{{len, len / 2}, {len, 1}, {len, len}, {len + 1, len / 2}, {len - 1, len / 4 + 3},
                                                {len, len - 1}, {len / 2 + 1, len / 2}}
                                       : Shapes{{len, len / 2}, {len + 1, len / 4}, {len - 3, 5}};
        for (const auto& [n, m] : shapes) {
            const auto f = random_poly(n, lg % 2), g = random_poly(m, lg % 2);
            const auto product = reference_product<V>(name, f, g);
            const std::size_t all = n - m + 1, offset = lg % 3 == 0 ? 0 : lg % 3 == 1 ? 1 : 16 + lg;
            if (lg <= 16) test_middle<V>(name, f, g, product, {all, all / 2 + 1, 1, all - 1 ? all - 1 : 1}, offset);
            else test_middle<V>(name, f, g, product, {all}, offset);
        }
    }
    const std::vector<u32> f(10, 1), g(4, 1);
    std::vector<u32> out(8);
    const auto throws = [&](std::span<const u32> a, std::span<const u32> b, std::span<u32> o) {
        try {
            middle_product<V>(a, b, o);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    expect(throws(f, {}, {}) && throws(g, f, {}) && throws(f, g, out) && !throws(f, g, std::span(out).first(7)),
           name, 0);
}

// a in the caller's memory at offsets of 0 to 1008 words in a page, full of garbage: the product
// comes back in it, its padding is not written.
template <class V>
void test_caller_memory(const char* name, std::size_t n, std::size_t m) {
    const auto f = random_poly(n, 0), g = random_poly(m, 0);
    const std::size_t len = Convolution::length(n, m);
    const auto product = schoolbook(f, g);
    for (const std::size_t offset : {0, 16, 32, 48, 512, 1008}) {
        PageWords a(len, offset);
        for (std::size_t i = 0; i < len + kPadding; ++i) a.begin()[i] = u32(rng());
        const std::vector<u32> padding(a.begin() + len, a.begin() + len + kPadding);
        std::copy(f.begin(), f.end(), a.begin());  // before constructing
        Convolution conv(n, m, a.begin());
        std::copy(g.begin(), g.end(), conv.b());
        const u32* c = conv.template multiply<V>();
        bool ok = c == a.begin() && conv.a() == a.begin() && conv.length() == len;
        ok &= std::vector<u32>(c, c + conv.size()) == product;
        ok &= std::vector<u32>(a.begin() + len, a.begin() + len + kPadding) == padding;
        expect(ok, name, offset * 100000000 + n);
    }
}

template <class V>
void test_variant(const char* name, int max_lg) {
    test_tables<V>(name);
    test_transforms<V>(name, max_lg);
    test_products<V>(name, max_lg);
    for (int lg = kMinLog; lg <= 11; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        test_upper_halves_unread<V>(name, len / 2, len / 2);
        test_upper_halves_unread<V>(name, len / 2 - 3, len / 4 + 9);
        test_caller_memory<V>(name, len / 2, len / 2);
        test_caller_memory<V>(name, len - 9, 10);
        test_caller_memory<V>(name, len / 4 + 1, len / 4 + 1);
    }
    test_caller_memory<V>(name, 0, 5);
    test_multiply_spans<V>(name, max_lg);
    test_squares<V>(name, max_lg);
    test_middles<V>(name, max_lg);
}

// Products on one thread, inputs from its own generator: the number of wrong ones. Lengths cross
// 2^kStaticLog, so threads build their own tables at the same time.
int thread_products(unsigned seed) {
    std::mt19937_64 local(seed);
    const auto poly = [&](std::size_t n) {
        std::vector<u32> p(n);
        for (u32& x : p) x = u32(local() % P);
        return p;
    };
    int wrong = 0;
    for (int round = 0; round < 24; ++round) {
        const bool exact = round % 3 != 2;
        const std::size_t n = exact ? 1 + local() % 1200 : (std::size_t(1) << 16) + local() % 70000;
        const std::size_t m = 1 + local() % n;
        const auto f = poly(n), g = poly(m);
        Convolution conv(n, m);
        std::copy(f.begin(), f.end(), conv.a());
        std::copy(g.begin(), g.end(), conv.b());
        const u32* c = conv.multiply();
        const std::vector<u32> h(c, c + conv.size());
        if (exact) {
            wrong += h != schoolbook(f, g);
        } else {
            const u32 x = u32(local() % P);
            wrong += evaluate(h, x) != mul(evaluate(f, x), evaluate(g, x));
        }
    }
    return wrong;
}

// The memory kept per thread: two Convolutions alive at once, shorter tables after longer ones,
// threads, and the errors. (Every other product test reuses regions of other sizes.)
void test_memory() {
    {
        const auto f = random_poly(300, 0), g = random_poly(200, 0), u = random_poly(40, 0), v = random_poly(50, 0);
        Convolution outer(300, 200);
        std::copy(f.begin(), f.end(), outer.a());
        std::copy(g.begin(), g.end(), outer.b());
        {
            Convolution inner(40, 50);
            std::copy(u.begin(), u.end(), inner.a());
            std::copy(v.begin(), v.end(), inner.b());
            const u32* c = inner.multiply();
            expect(std::vector<u32>(c, c + 89) == schoolbook(u, v), "memory: inner product");
        }
        const u32* c = outer.multiply();
        expect(std::vector<u32>(c, c + 499) == schoolbook(f, g), "memory: outer product");
    }
    for (const int lg : {19, 17, 18, 17})
        test_product<SelectedMultiply>("memory: tables", std::size_t(1) << (lg - 1), 77, 0, false);
    std::array<int, 4> wrong{};
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < wrong.size(); ++t) threads.emplace_back([t, &wrong] { wrong[t] = thread_products(t); });
    for (auto& thread : threads) thread.join();
    for (const int w : wrong) expect(w == 0, "memory: threads", u64(w));

    bool threw = false;
    try {
        PageWords a(64, 4);  // 16 bytes past a page boundary
        Convolution conv(10, 10, a.begin());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "memory: a not 64-byte aligned");
    threw = false;
    try {
        Convolution conv(std::size_t(1) << kMaxLog, 2);
    } catch (const std::length_error&) {
        threw = true;
    }
    expect(threw, "memory: product too long");
    // multiply(f, g, out), factors of 2^(kMaxLog - 1) + 1 words: throws before writing out.
    std::vector<u32> big((std::size_t(1) << kMaxLog) + 1, 1);
    const std::span<const u32> half = std::span(big).first(big.size() / 2 + 1);
    threw = false;
    try {
        multiply(half, half, std::span(big));
    } catch (const std::length_error&) {
        threw = true;
    }
    expect(threw && std::ranges::all_of(big, [](u32 w) { return w == 1; }), "memory: multiply too long");
}

// Random values below bound, the largest ones in the first words.
void fill(u32* p, std::size_t count, u64 bound) {
    for (std::size_t i = 0; i < count; ++i) p[i] = random_below(bound);
    for (std::size_t i = 0; i < count && i < 4; ++i) p[i] = u32(bound - 1 - i);
}

// Forward butterfly of one lane: a + x c + y (b + x d), ..., mod P.
std::array<u32, 4> forward_model(const std::array<u32, 4>& f, u32 x, u32 y, u32 z) {
    const u32 A = add(f[0], mul(x, f[2])), C = sub(f[0], mul(x, f[2]));
    const u32 B = add(f[1], mul(x, f[3])), D = sub(f[1], mul(x, f[3]));
    return {add(A, mul(y, B)), sub(A, mul(y, B)), add(C, mul(z, D)), sub(C, mul(z, D))};
}

std::array<u32, 4> inverse_model(const std::array<u32, 4>& f, u32 x, u32 y, u32 z) {
    const u32 ab = add(f[0], f[1]), cd = add(f[2], f[3]);
    const u32 amb = mul(y, sub(f[0], f[1])), cmd = mul(z, sub(f[2], f[3]));
    return {add(ab, cd), add(amb, cmd), mul(x, sub(ab, cd)), mul(x, sub(amb, cmd))};
}

// Twiddle tables for transforms of length 2^lg.
struct Tables {
    explicit Tables(int lg) : forward_(detail::table_words(lg)), inverse_(detail::table_words(lg)) {
        Native::build_table(forward_.begin(), (std::size_t(1) << lg) / 16, false);
        Native::build_table(inverse_.begin(), (std::size_t(1) << lg) / 16, true);
    }
    const u32* r(std::size_t k) { return forward_.begin() + detail::slot(k); }  // value; quotient at [8]
    const u32* ir(std::size_t k) { return inverse_.begin() + detail::slot(k); }
    const u32* r() { return forward_.begin(); }
    const u32* ir() { return inverse_.begin(); }

private:
    Words forward_, inverse_;
};

constexpr int kTableLog = 12;
constexpr std::size_t kGroups = (std::size_t(1) << kTableLog) / 32;  // groups k with r(2k + 1) in the table

// intrinsics<S>: each kernel against the lane models, inputs at the edges of their ranges.
template <class S>
struct IntrinsicsKernels {
    using K = detail::intrinsics::Ntt<S>;
    using Leaf = typename K::Leaf;

    // Runs kernel on count groups of quarters of q words from group first, inputs below in_bound;
    // checks every lane against model(f, group) and outputs below out_bound.
    template <class Kernel, class Model>
    static void check(const std::string& name, std::size_t q, std::size_t count, u64 in_bound, u64 out_bound,
                      Kernel kernel, Model model) {
        Words v(4 * q * count);
        fill(v.begin(), 4 * q * count, in_bound);
        const std::vector<u32> in(v.begin(), v.begin() + 4 * q * count);
        kernel(v.begin());
        for (std::size_t g = 0; g < count; ++g)
            for (std::size_t j = 0; j < q; ++j) {
                std::array<u32, 4> f;
                for (std::size_t t = 0; t < 4; ++t) f[t] = in[4 * q * g + j + q * t];
                const auto expected = model(f, g);
                for (std::size_t t = 0; t < 4; ++t) {
                    const u32 got = v.begin()[4 * q * g + j + q * t];
                    expect(got < out_bound && got % P == expected[t], name.c_str(), j);
                }
            }
    }

    static void run(const std::string& name) {
        Tables t(kTableLog);
        const auto fm = [&](std::size_t k) {
            return [&t, k](const std::array<u32, 4>& f, std::size_t g) {
                return forward_model(f, *t.r(k + g), t.r(2 * (k + g))[0], t.r(2 * (k + g))[1]);
            };
        };
        const auto im = [&](std::size_t k) {
            return [&t, k](const std::array<u32, 4>& f, std::size_t g) {
                return inverse_model(f, *t.ir(k + g), t.ir(2 * (k + g))[0], t.ir(2 * (k + g))[1]);
            };
        };
        // Levels: count groups from k (0: an identity group first), two columns per group on zmm.
        for (std::size_t q : {32, 128}) {
            for (int round = 0; round < 8; ++round) {
                const std::size_t count = 1 + random_below(5), k = round == 0 ? 0 : 1 + random_below(kGroups - count);
                check(name + " forward_groups", q, count, 4 * u64(P), 4 * u64(P),
                      [&](u32* f) { K::forward_groups(f, q, k, count, t.r()); }, fm(k));
                check(name + " inverse_groups", q, count, 2 * u64(P), 2 * u64(P),
                      [&](u32* f) { K::inverse_groups(f, q, k, count, t.ir()); }, im(k));
            }
        }
        // 2^15: quarters long enough for the passes that prefetch (detail::intrinsics::kPrefetchMin).
        for (std::size_t q : {32, 128, 1 << 15}) {
            for (int round = 0; round < (q > 128 ? 2 : 8); ++round) {
                const std::size_t k = round == 0 ? 0 : 1 + random_below(kGroups - 1);  // 0: identity groups
                check(name + " forward_group", q, 1, 4 * u64(P), 4 * u64(P),
                      [&](u32* f) { K::forward_group(f, q, k, t.r()); }, fm(k));
                check(name + " forward_group, in < 2P", q, 1, 2 * u64(P), 4 * u64(P),
                      [&](u32* f) { K::template forward_group<2>(f, q, k, t.r()); }, fm(k));
                check(name + " forward_group, in < P", q, 1, P, 4 * u64(P),
                      [&](u32* f) { K::template forward_group<1>(f, q, k, t.r()); }, fm(k));
                check(name + " inverse_group", q, 1, 2 * u64(P), 2 * u64(P),
                      [&](u32* f) { K::inverse_group(f, q, k, t.ir()); }, im(k));
                // The top groups that take the scale: k < 4.
                const u32 s = round == 1 ? P - 1 : 1 + random_below(P - 1);
                const auto scaled = detail::intrinsics::scaled_block(s);
                check(name + " inverse_group_scaled", q, 1, 2 * u64(P), 2 * u64(P),
                      [&](u32* f) { K::inverse_group_scaled(f, q, k % 4, t.ir(), scaled.words); },
                      [&](const std::array<u32, 4>& f, std::size_t g) {
                          auto e = im(k % 4)(f, g);
                          for (u32& x : e) x = mul(x, s);
                          return e;
                      });
            }
        }
        for (int round = 0; round < 30; ++round) {
            const std::size_t count = 1 + random_below(4), k = round == 0 ? 0 : random_below(kGroups - count);
            check(name + " forward bottom", 8, count, 4 * u64(P), P,
                  [&](u32* f) { Leaf::forward(f, count, k, t.r()); }, fm(k));
            check(name + " inverse bottom", 8, count, 2 * u64(P), 2 * u64(P),
                  [&](u32* f) { Leaf::inverse(f, count, k, t.ir()); }, im(k));
        }
        test_multiply(name + " leaf products", t, 1, [&](u32* a, u32* b, std::size_t count, std::size_t k) {
            Leaf::multiply(a, b, count, k, t.r(), t.ir());
        });
        test_multiply(name + " leaf squares", t, 1, [&](u32* a, u32* b, std::size_t count, std::size_t k) {
            Leaf::template multiply<true>(a, b, count, k, t.r(), t.ir());
        }, true);
        if constexpr (S::lanes == 16) {  // pairs of groups on S
            using W = detail::intrinsics::WideBottom<S>;
            for (int round = 0; round < 30; ++round) {
                const std::size_t count = 2 + 2 * random_below(2), k = 2 * random_below(kGroups / 2 - 2);
                check(name + " wide forward bottom", 8, count, 4 * u64(P), P,
                      [&](u32* f) { W::forward(f, count, k, t.r()); }, fm(k));
                check(name + " wide inverse bottom", 8, count, 2 * u64(P), 2 * u64(P),
                      [&](u32* f) { W::inverse(f, count, k, t.ir()); }, im(k));
            }
            test_multiply(name + " wide leaf products", t, 2, [&](u32* a, u32* b, std::size_t count, std::size_t k) {
                W::multiply(a, b, count, k, t.r(), t.ir());
            });
            test_multiply(name + " wide leaf squares", t, 2, [&](u32* a, u32* b, std::size_t count, std::size_t k) {
                W::template multiply<true>(a, b, count, k, t.r(), t.ir());
            }, true);
        }
    }

    // run(a, b, count, first): Leaf::multiply or a variant of it on groups of 32 words, count and
    // first multiples of step: forward butterfly, products of the leaves mod x^8 - w (with the
    // Montgomery factor 2^-32), inverse butterfly.
    // square: b = a, which run must not read (a times a).
    template <class Run>
    static void test_multiply(const std::string& name, Tables& t, std::size_t step, Run run, bool square = false) {
        const u32 r_inverse = power(u32((u64(1) << 32) % P), P - 2);
        for (int round = 0; round < 50; ++round) {
            const std::size_t count = step * (1 + random_below(4 / step)), k0 = step * random_below(kGroups / step - 4);
            Words a(32 * count), b(32 * count);
            fill(a.begin(), 32 * count, 4 * u64(P));
            fill(b.begin(), 32 * count, 4 * u64(P));
            if (square) std::copy_n(a.begin(), 32 * count, b.begin());
            const std::vector<u32> a0(a.begin(), a.begin() + 32 * count), b0(b.begin(), b.begin() + 32 * count);
            if (square) std::fill_n(b.begin(), 32 * count, u32(rng()));
            run(a.begin(), b.begin(), count, k0);
            for (std::size_t j = 0; j < count; ++j) {
                const std::size_t k = k0 + j;
                const u32 x = *t.r(k), y = t.r(2 * k)[0], z = t.r(2 * k)[1];
                const std::array<u32, 4> w{y, P - y, z, P - z};
                u32 fa[4][8], fb[4][8], c[4][8];
                for (int l = 0; l < 8; ++l) {
                    std::array<u32, 4> xa, xb;
                    for (int s = 0; s < 4; ++s) xa[s] = a0[32 * j + 8 * s + l], xb[s] = b0[32 * j + 8 * s + l];
                    const auto ya = forward_model(xa, x, y, z), yb = forward_model(xb, x, y, z);
                    for (int s = 0; s < 4; ++s) fa[s][l] = ya[s], fb[s][l] = yb[s];
                }
                for (int s = 0; s < 4; ++s)
                    for (int l = 0; l < 8; ++l) {
                        u32 sum = 0;
                        for (int i = 0; i < 8; ++i)
                            sum = add(sum, mul(fb[s][i], l >= i ? fa[s][l - i] : mul(w[s], fa[s][l - i + 8])));
                        c[s][l] = mul(sum, r_inverse);
                    }
                for (int l = 0; l < 8; ++l) {
                    const auto expected = inverse_model({c[0][l], c[1][l], c[2][l], c[3][l]}, *t.ir(k),
                                                        t.ir(2 * k)[0], t.ir(2 * k)[1]);
                    for (int s = 0; s < 4; ++s) {
                        const u32 got = a.begin()[32 * j + 8 * s + l];
                        expect(got < 2 * P && got % P == expected[s], name.c_str(), 32 * j + 8 * s + l);
                    }
                }
            }
        }
    }
};

// A table with candidate (op + lg) % kCandidates: neighbouring ops and lengths differ.
Choices mixed_choices() {
    Choices choices{};
    for (int op = 0; op < kOps; ++op)
        for (int i = 0; i < kLengths; ++i) choices[op][i] = std::uint8_t((op + i) % kCandidates);
    return choices;
}

// The tuned variant (select.hpp) with every candidate in every place, then a mixed table up to 2^17
// (tables past 2^16 are built by tuned::build_table). Each candidate is also tested on its own.
void test_tuned() {
    for (int v = 0; v < kCandidates; ++v) {
        Choices same{};
        for (auto& row : same) row.fill(std::uint8_t(v));
        variants.set(same);
        const std::string name = "tuned, all " + variant_names()[v];
        test_transforms<variant::tuned>(name.c_str(), 10);
        test_products<variant::tuned>(name.c_str(), 10);
        test_multiply_spans<variant::tuned>(name.c_str(), 10);
    }
    variants.set(mixed_choices());
    test_transforms<variant::tuned>("tuned, mixed", 17);
    test_products<variant::tuned>("tuned, mixed", 17);
    test_multiply_spans<variant::tuned>("tuned, mixed", 17);
    for (int op = 0; op < kOps; ++op)
        for (int lg = kMinLog; lg <= kMaxLog; ++lg)
            expect(choice(Op(op), lg) == (kCandidates == 1 ? 0 : (op + lg - kMinLog) % kCandidates), "tuned: choice", lg);
    variants.set(kDefaults);
    expect(valid(kDefaults) && variants.get() == kDefaults, "tuned: compiled defaults");
}

// measure() at a small budget: valid choices; past max_lg start's, or the choice at max_lg where it
// changed; a margin above any difference keeps start. The wisdom text; "ntt.variant" through
// tune::save and load, and tune::run().
void test_measure() {
    const Choices mixed = mixed_choices();
    const Budget quick = {8, 1, 2e3, 0};
    const Choices measured = measure(mixed, quick);
    expect(valid(measured), "measure: valid");
    for (int op = 0; op < kOps; ++op) {
        const int last = quick.max_lg - kMinLog;
        const bool moved = measured[op][last] != mixed[op][last];
        for (int i = last + 1; i < kLengths; ++i)
            expect(measured[op][i] == (moved ? measured[op][last] : mixed[op][i]), "measure: past max_lg", i);
    }
    expect(measure(mixed, {7, 1, 2e3, 1e9}) == mixed, "measure: a huge margin keeps start");
    std::vector<Times> log;
    measure(kDefaults, {7, 1, 2e3, 0.03}, &log);
    expect(log.size() == (kCandidates > 1 ? std::size_t(2 * kOps) : 0u), "measure: log", log.size());
    for (const Times& t : log)
        for (const auto& ns : t.ns) expect(ns.size() == std::size_t(kCandidates) && ns[0] > 0, "measure: log times");

    expect(parse(text(mixed)) == mixed, "wisdom: text round trip");
    const std::string good = text(mixed);
    expect(!parse(std::to_string(kSignature ^ 1) + good.substr(good.find(' '))), "wisdom: another list's signature");
    expect(!parse(good.substr(0, good.rfind(' '))), "wisdom: a value short");
    expect(!parse(good.substr(0, good.rfind(' ')) + " " + std::to_string(kCandidates)), "wisdom: no such candidate");
    const std::string path = "fps-ntt-test-wisdom-" + std::to_string(std::random_device{}());
    variants.set(mixed);
    expect(fps::tune::save(path), "wisdom: save");
    variants.set(kDefaults);
    expect(fps::tune::load(path) && variants.get() == mixed, "wisdom: load");
    std::remove(path.c_str());
    variants.set(kDefaults);
    fps::tune::run();
    expect(valid(variants.get()), "tune::run: valid");
    variants.set(kDefaults);
}

// Every simd struct this machine runs; Scalar is slow, so shorter.
void test_intrinsics() {
    fps::simd::for_each(fps::simd::All{}, []<class S> {
        const std::string name = std::string("intrinsics ") + S::name;
        IntrinsicsKernels<S>::run(name);
        test_variant<variant::intrinsics<S>>(name.c_str(), S::lanes == 1 ? kScalarMaxLog : kVectorMaxLog);
    });
}

#ifdef __AVX2__

namespace impl = detail::avx2;
using Vec = impl::Vec;

// Variant avx2 with each bottom stage at every length.
template <class Bottom>
struct avx2_bottom : variant::avx2 {
    static void multiply(u32* a, u32* b, std::size_t n, std::size_t m, int lg, const u32* r, const u32* ir) {
        const std::size_t len = std::size_t(1) << lg;
        impl::product<Bottom>(a, b, 2 * n <= len, 2 * m <= len, lg, r, ir);
    }
    static void multiply(const detail::Product& p, const u32* r, const u32* ir) { impl::product<Bottom>(p, r, ir); }
};
using avx2_group = avx2_bottom<impl::GroupBottom>;
using avx2_pair = avx2_bottom<impl::PairBottom>;

// Variant avx2 with each square bottom stage at every length.
template <class Bottom>
struct avx2_square_bottom : variant::avx2 {
    static void square(const detail::Product& p, const u32* r, const u32* ir) { impl::square<Bottom>(p, r, ir); }
};
using avx2_group_square = avx2_square_bottom<impl::GroupSquareBottom>;
using avx2_pair_square = avx2_square_bottom<impl::PairSquareBottom>;

void test_bottoms() {
    for (int lg = kMinLog; lg <= kMaxLog; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        const bool exact = lg <= 11;
        for (const auto& [n, m] : {std::pair{len / 2, len / 2}, {len / 2, std::size_t(2)}, {len / 4 + 1, len / 4 + 1},
                                   {len - len / 4, len / 4 + 1}, {len / 4 + 1, len - 1 - len / 4}}) {
            test_product<avx2_group>("avx2 one-group bottom", n, m, 0, exact);
            test_product<avx2_pair>("avx2 two-group bottom", n, m, 0, exact);
            if (lg <= 12) {  // from spans, against the products above
                const auto f = random_poly(n, 0), g = random_poly(m, 0), product = schoolbook(f, g);
                test_spans<avx2_group>("avx2 one-group bottom, spans", f, g, product, {n + m - 1, len / 2 + 3}, 1);
                test_spans<avx2_pair>("avx2 two-group bottom, spans", f, g, product, {n + m - 1, len / 2 + 3}, 1);
            }
        }
        for (const std::size_t n : {len / 2, len / 4 + 1, len / 2 - 3}) {
            if (lg > 20) break;
            const auto f = random_poly(n, 0);
            const auto product = lg <= 12 ? schoolbook(f, f) : reference_product<variant::avx2>("avx2 square", f, f);
            test_square<avx2_group_square>("avx2 one-group square bottom", f, product, {2 * n - 1, len / 2 + 3}, 1);
            test_square<avx2_pair_square>("avx2 two-group square bottom", f, product, {2 * n - 1, len / 2 + 3}, 1);
        }
    }
}

Vec* vec(Words& w) { return reinterpret_cast<Vec*>(w.begin()); }

// Runs kernel on 4h vectors with inputs below in_bound; checks every lane against model and
// outputs below out_bound.
template <class Kernel, class Model>
void test_loop(const char* name, std::size_t h, u64 in_bound, u64 out_bound, Kernel kernel, Model model) {
    Words v(32 * h);
    fill(v.begin(), 32 * h, in_bound);
    const std::vector<u32> in(v.begin(), v.begin() + 32 * h);
    kernel(vec(v), h);
    for (std::size_t j = 0; j < 8 * h; ++j) {
        std::array<u32, 4> f;
        for (std::size_t t = 0; t < 4; ++t) f[t] = in[j + 8 * h * t];
        const auto expected = model(f);
        for (std::size_t t = 0; t < 4; ++t) {
            const u32 got = v.begin()[j + 8 * h * t];
            expect(got < out_bound && got % P == expected[t], name, j);
        }
    }
}

void test_loops() {
    namespace k = impl::kernels;
    Tables t(kTableLog);
    for (std::size_t h : {2, 4, 16}) {
        for (int round = 0; round < 8; ++round) {
            const std::size_t g = 1 + random_below(kGroups - 1);
            const u32 *x = t.r(g), *y = t.r(2 * g), *ix = t.ir(g), *iy = t.ir(2 * g);
            test_loop("forward", h, 4 * u64(P), 4 * u64(P), [&](Vec* f, std::size_t n) { k::forward(f, n, x, y); },
                      [&](const std::array<u32, 4>& f) { return forward_model(f, *x, y[0], y[1]); });
            test_loop("inverse", h, 2 * u64(P), 2 * u64(P), [&](Vec* f, std::size_t n) { k::inverse(f, n, ix, iy); },
                      [&](const std::array<u32, 4>& f) { return inverse_model(f, *ix, iy[0], iy[1]); });
            test_loop("forward_identity", h, 4 * u64(P), 4 * u64(P),
                      [&](Vec* f, std::size_t n) { k::forward_identity(f, n, t.r()); },
                      [&](const std::array<u32, 4>& f) { return forward_model(f, 1, 1, *t.r(1)); });
            test_loop("forward_top", h, P, 4 * u64(P),
                      [&](Vec* f, std::size_t n) { k::forward_top(f, n, t.r()); },
                      [&](const std::array<u32, 4>& f) { return forward_model(f, 1, 1, *t.r(1)); });
            test_loop("forward_top_from", h, P, 4 * u64(P),
                      [&](Vec* f, std::size_t n) { k::forward_top_from(reinterpret_cast<u32*>(f), f, n, n, t.r()); },
                      [&](const std::array<u32, 4>& f) { return forward_model(f, 1, 1, *t.r(1)); });
            test_loop("forward_top3_from", h, P, 4 * u64(P),
                      [&](Vec* f, std::size_t n) { k::forward_top3_from(reinterpret_cast<u32*>(f), f, n, n, t.r()); },
                      [&](std::array<u32, 4> f) { return f[3] = 0, forward_model(f, 1, 1, *t.r(1)); });
            test_loop("inverse_identity", h, 2 * u64(P), 2 * u64(P),
                      [&](Vec* f, std::size_t n) { k::inverse_identity(f, n, t.ir()); },
                      [&](const std::array<u32, 4>& f) { return inverse_model(f, 1, 1, *t.ir(1)); });
            // forward_pair: forward() on two arrays.
            Words a(32 * h), b(32 * h);
            fill(a.begin(), 32 * h, 4 * u64(P));
            fill(b.begin(), 32 * h, 4 * u64(P));
            Words a1 = a, b1 = b;
            k::forward_pair(vec(a), vec(b), h, x, y);
            k::forward(vec(a1), h, x, y);
            k::forward(vec(b1), h, x, y);
            for (std::size_t i = 0; i < 32 * h; ++i)
                expect(a.begin()[i] == a1.begin()[i] && b.begin()[i] == b1.begin()[i], "forward_pair", i);
        }
        // scale_radix2: s (f0 + f1), s (f0 - f1), canonical.
        Words v(16 * h);
        fill(v.begin(), 16 * h, 2 * u64(P));
        const std::vector<u32> in(v.begin(), v.begin() + 16 * h);
        const u32 s = random_below(P);
        impl::scale_radix2(vec(v), h, s);
        for (std::size_t j = 0; j < 8 * h; ++j) {
            expect(v.begin()[j] == mul(s, add(in[j], in[j + 8 * h])), "scale_radix2 sum", j);
            expect(v.begin()[j + 8 * h] == mul(s, sub(in[j], in[j + 8 * h])), "scale_radix2 difference", j);
        }
        // inverse_radix4: s inverse_identity(f), canonical.
        Words w(32 * h);
        fill(w.begin(), 32 * h, 2 * u64(P));
        const std::vector<u32> top(w.begin(), w.begin() + 32 * h);
        impl::inverse_radix4(vec(w), h, t.ir(), s);
        for (std::size_t j = 0; j < 8 * h; ++j) {
            std::array<u32, 4> f;
            for (std::size_t q = 0; q < 4; ++q) f[q] = top[j + 8 * h * q];
            const auto expected = inverse_model(f, 1, 1, *t.ir(1));
            for (std::size_t q = 0; q < 4; ++q) expect(w.begin()[j + 8 * h * q] == mul(s, expected[q]), "inverse_radix4", j);
        }
    }
}

// A product's first and last levels in caller memory against the same levels in place. First: a
// source of n words 4 bytes past alignment against its zero-padded copy, equal mod P and < 4P.
// Last: count words to an output 4 bytes past alignment, ending its allocation, against the
// in-place output (canonical); inputs < 2P. Block sizes 4 and 16 vectors.
void test_caller_levels() {
    Tables t(kTableLog);
    const u32 s = random_below(P);
    for (const std::size_t h : {4, 16}) {
        struct Level {
            const char* name;
            std::size_t words;  // the level's span: 2, 4 or 8 blocks of 8h words
            std::function<void(const u32*, std::size_t, u32*)> first;
            std::function<void(u32*, u32*, std::size_t)> last;
        };
        const Level levels[] = {
            {"radix-2 level", 16 * h, [&](const u32* f, std::size_t n, u32* to) { impl::forward_radix2_from(f, n, to, h); },
             [&](u32* f, u32* out, std::size_t count) { impl::scale_radix2_out(f, h, out, count, s); }},
            {"radix-4 level", 32 * h, [&](const u32* f, std::size_t n, u32* to) { impl::forward_radix4_from(f, n, to, h, t.r()); },
             [&](u32* f, u32* out, std::size_t count) { impl::inverse_radix4_out(f, h, out, count, t.ir(), s); }},
            {"radix-8 level", 64 * h, [&](const u32* f, std::size_t n, u32* to) { impl::forward_radix8_from(f, n, to, h, t.r()); },
             [&](u32* f, u32* out, std::size_t count) { impl::inverse_radix8_out(f, h, out, count, t.ir(), s); }},
        };
        for (const Level& level : levels) {
            const std::size_t w = level.words;
            const std::size_t sizes[] = {w, w - 1, w - w / 4, w - w / 4 + 5, w / 2, w / 2 - 9, 8 * h + 1, 13, 1};
            for (const std::size_t n : sizes) {
                PageWords source(n, 1, 0);
                fill(source.begin(), n, P);
                Words padded(w), from(w);
                std::copy_n(source.begin(), n, padded.begin());
                level.first(padded.begin(), w, padded.begin());
                level.first(source.begin(), n, from.begin());
                for (std::size_t i = 0; i < w; ++i)
                    expect(from.begin()[i] < 4 * P && from.begin()[i] % P == padded.begin()[i] % P, level.name, n);
            }
            Words in(w);
            fill(in.begin(), w, 2 * u64(P));
            Words whole = in;
            level.last(whole.begin(), whole.begin(), w);
            for (const std::size_t count : {w, w - 1, w - 8, w / 2 + 3, w / 2, 8 * h - 1, std::size_t(9), std::size_t(1)}) {
                Words work = in;
                PageWords out(count, 1, 0);
                level.last(work.begin(), out.begin(), count);
                expect(std::equal(out.begin(), out.begin() + count, whole.begin()), level.name, count);
            }
        }
    }
}

// forward_bottom and inverse_bottom: groups of 4 vectors (h = 1), each with its own twiddles.
void test_bottom_levels() {
    Tables t(kTableLog);
    for (int round = 0; round < 50; ++round) {
        const std::size_t count = 2 + 2 * random_below(2), first = 2 * random_below(kGroups / 2 - 2);
        for (int inverse = 0; inverse < 2; ++inverse) {
            const u64 bound = inverse ? u64(P) : 4 * u64(P);  // inverse: leaf form in
            Words v(32 * count);
            fill(v.begin(), 32 * count, bound);
            const std::vector<u32> in(v.begin(), v.begin() + 32 * count);
            if (inverse) impl::inverse_bottom(vec(v), count, first, t.ir());
            else impl::forward_bottom(vec(v), count, first, t.r());
            for (std::size_t g = 0; g < count; ++g) {
                const std::size_t k = first + g;
                for (int l = 0; l < 8; ++l) {
                    std::array<u32, 4> f;
                    for (int s = 0; s < 4; ++s) f[s] = in[32 * g + 8 * s + l];
                    const auto expected = inverse ? inverse_model(f, *t.ir(k), t.ir(2 * k)[0], t.ir(2 * k)[1])
                                                  : forward_model(f, *t.r(k), t.r(2 * k)[0], t.r(2 * k)[1]);
                    for (int s = 0; s < 4; ++s) {
                        const u32 got = v.begin()[32 * g + 8 * s + l];
                        expect(got < (inverse ? 2 * P : P) && got % P == expected[s],
                               inverse ? "inverse_bottom" : "forward_bottom", k);
                    }
                }
            }
        }
    }
}

// The product bottom stage (GroupBottom) on batches of four vectors: forward butterfly with
// h = 1, products of the leaves mod x^8 - w (with the Montgomery factor 2^-32), inverse butterfly.
void test_group_bottom() {
    Tables t(kTableLog);
    const u32 r_inverse = power(u32((u64(1) << 32) % P), P - 2);
    for (int round = 0; round < 50; ++round) {
        const std::size_t batches = 1 + random_below(4), k0 = random_below(kGroups - batches);
        Words a(32 * batches), b(32 * batches);
        fill(a.begin(), 32 * batches, 4 * u64(P));
        fill(b.begin(), 32 * batches, 4 * u64(P));
        const std::vector<u32> a0(a.begin(), a.begin() + 32 * batches), b0(b.begin(), b.begin() + 32 * batches);
        impl::GroupBottom::run(vec(a), vec(b), 4 * batches, k0, t.r(), t.ir());
        for (std::size_t j = 0; j < batches; ++j) {
            const std::size_t k = k0 + j;
            const u32 x = *t.r(k), y = t.r(2 * k)[0], z = t.r(2 * k)[1];
            const std::array<u32, 4> w{y, P - y, z, P - z};
            u32 fa[4][8], fb[4][8], c[4][8];
            for (int l = 0; l < 8; ++l) {
                std::array<u32, 4> xa, xb;
                for (int s = 0; s < 4; ++s) xa[s] = a0[32 * j + 8 * s + l], xb[s] = b0[32 * j + 8 * s + l];
                const auto ya = forward_model(xa, x, y, z), yb = forward_model(xb, x, y, z);
                for (int s = 0; s < 4; ++s) fa[s][l] = ya[s], fb[s][l] = yb[s];
            }
            for (int s = 0; s < 4; ++s)
                for (int l = 0; l < 8; ++l) {
                    u32 sum = 0;
                    for (int i = 0; i < 8; ++i)
                        sum = add(sum, mul(fb[s][i], l >= i ? fa[s][l - i] : mul(w[s], fa[s][l - i + 8])));
                    c[s][l] = mul(sum, r_inverse);
                }
            for (int l = 0; l < 8; ++l) {
                const auto expected = inverse_model({c[0][l], c[1][l], c[2][l], c[3][l]}, *t.ir(k), t.ir(2 * k)[0],
                                                    t.ir(2 * k)[1]);
                for (int s = 0; s < 4; ++s) {
                    const u32 got = a.begin()[32 * j + 8 * s + l];
                    expect(got < 2 * P && got % P == expected[s], "group bottom", 32 * j + 8 * s + l);
                }
            }
        }
    }
}

void test_avx2() {
    test_loops();
    test_caller_levels();
    test_bottom_levels();
    test_group_bottom();
    test_bottoms();
    test_variant<variant::avx2>("avx2", kVectorMaxLog);
}

#endif

}  // namespace

int main() {
    test_memory();
    test_tuned();
    test_measure();
    test_variant<variant::scalar>("scalar", kScalarMaxLog);
    test_intrinsics();
#ifdef __AVX2__
    test_avx2();
#endif
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
