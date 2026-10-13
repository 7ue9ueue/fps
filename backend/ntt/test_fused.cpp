// Tests for Transform's fused ops (issue #112): forward and forward_upper from caller memory,
// forward_product, cyclic_product and inverse_product, on every variant this machine runs and the
// tuned one. References: cyclic convolution by schoolbook (short lengths) and the scalar variant's
// separate ops (forward of a zero-padded copy, leaf products, inverse). Sources of every length
// around 0, L/2 and L, both output halves, negated outputs, out over in, spans at odd addresses,
// words past each output untouched, the argument checks.
#include "backend/ntt/ntt.hpp"

#include <algorithm>
#include <cstdio>
#include <new>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using namespace fps::ntt;

constexpr u32 P = kModulus;
#if __has_feature(address_sanitizer)
constexpr int kMaxTestLog = 16;
#else
constexpr int kMaxTestLog = 18;
#endif
constexpr int kThoroughMaxLog = 12;  // past it, fewer source and output sizes
constexpr int kScalarMaxLog = 14;
constexpr int kSchoolbookMaxLog = 11;
constexpr u32 kSentinel = 0xdeadbeef;

std::mt19937_64 rng(112);
int failures = 0;

void expect(bool ok, const std::string& what, u64 detail = 0) {
    if (ok) return;
    if (++failures <= 30) std::printf("FAIL: %s (%llu)\n", what.c_str(), static_cast<unsigned long long>(detail));
}


u32 add(u32 x, u32 y) { return u32((u64(x) + y) % P); }
u32 neg(u32 x) { return x ? P - x : 0; }

// count words, 32-byte aligned, then kPadding words (ASan sees reads past them); the padding holds
// kSentinel, which no op may change.
struct Words {
    struct alignas(32) Lanes {
        u32 lane[8];
    };
    explicit Words(std::size_t count) : size(count), data((count + kPadding) / 8) {
        std::fill(begin() + count, begin() + count + kPadding, kSentinel);
    }
    u32* begin() { return data.data()->lane; }
    bool padding_kept() { return std::all_of(begin() + size, begin() + size + kPadding, [](u32 x) { return x == kSentinel; }); }
    std::size_t size;
    std::vector<Lanes> data;
};

std::vector<u32> random_poly(std::size_t n, int kind = 0) {
    std::vector<u32> p(n);
    for (auto& x : p) x = kind == 0 ? u32(rng() % P) : kind == 1 ? P - 1 : u32(rng() % 3);
    return p;
}

// Caller memory at an odd address: words [3, 3 + n) of a buffer whose other words hold kSentinel.
struct Span {
    explicit Span(std::size_t n) : buffer(n + 8, kSentinel), n(n) {}
    explicit Span(const std::vector<u32>& v) : Span(v.size()) { std::copy(v.begin(), v.end(), buffer.begin() + 3); }
    u32* data() { return buffer.data() + 3; }
    std::span<u32> span() { return {data(), n}; }
    std::vector<u32> words() const { return {buffer.begin() + 3, buffer.begin() + 3 + std::ptrdiff_t(n)}; }
    bool edges_kept() const {
        return std::all_of(buffer.begin(), buffer.begin() + 3, [](u32 x) { return x == kSentinel; }) &&
               std::all_of(buffer.begin() + 3 + std::ptrdiff_t(n), buffer.end(), [](u32 x) { return x == kSentinel; });
    }
    std::vector<u32> buffer;
    std::size_t n;
};

// Leaf forms the ops read (B, the terms): 64-byte aligned, as Transform's arrays must be (the avx2
// asm loads them aligned), kPadding words readable past the end.
template <class T>
struct Aligned {
    using value_type = T;
    Aligned() = default;
    template <class U>
    Aligned(const Aligned<U>&) {}
    T* allocate(std::size_t n) { return static_cast<T*>(::operator new((n + kPadding) * sizeof(T), std::align_val_t(64))); }
    void deallocate(T* p, std::size_t) { ::operator delete(p, std::align_val_t(64)); }
    bool operator==(const Aligned&) const { return true; }
};
using Form = std::vector<u32, Aligned<u32>>;

// The scalar model's leaf form of in, zero-padded to length() words.
Form leaves(const Transform& t, const std::vector<u32>& in) {
    Words w(t.length());
    std::fill(std::copy(in.begin(), in.end(), w.begin()), w.begin() + t.length(), 0u);
    t.forward<variant::scalar>(w.begin());
    return {w.begin(), w.begin() + t.length()};
}

std::vector<u32> coefficients(const Transform& t, Form F) {
    Words w(t.length());
    std::copy(F.begin(), F.end(), w.begin());
    t.inverse<variant::scalar>(w.begin());
    return {w.begin(), w.begin() + t.length()};
}

Form leaf_product(const Transform& t, const Form& a, const Form& b) {
    Form out(t.length());
    t.multiply<variant::scalar>(a.data(), b.data(), out.data());
    return out;
}

// in b mod (x^L - 1) by schoolbook.
std::vector<u32> cyclic(const std::vector<u32>& in, const std::vector<u32>& b, std::size_t len) {
    std::vector<u64> c(len);
    for (std::size_t i = 0; i < in.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[(i + j) % len] = (c[(i + j) % len] + u64(in[i]) * b[j]) % P;
    return {c.begin(), c.end()};
}

// out as Output o picks it from the coefficients c.
std::vector<u32> pick(const std::vector<u32>& c, std::size_t count, Output o) {
    std::vector<u32> out(count);
    for (std::size_t i = 0; i < count; ++i) out[i] = o.negate ? neg(c[o.offset + i]) : c[o.offset + i];
    return out;
}

// Source lengths around 0, L/2 and L, and a random one.
std::vector<std::size_t> source_sizes(std::size_t len) {
    std::vector<std::size_t> s = {0, 1, 7, 8, 9, len / 2 - 1, len / 2, len / 2 + 1, len - 1, len, 1 + rng() % len};
    if (len > (std::size_t(1) << kThoroughMaxLog)) s = {1, len / 2 - 1, len / 2, len, 1 + rng() % len};
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
    return s;
}

template <class V>
void test_forwards(const std::string& name, const Transform& t) {
    const std::size_t len = t.length();
    const std::vector<u32> b = random_poly(len);
    const Form B = leaves(t, b);
    for (std::size_t n : source_sizes(len)) {
        for (int kind : {0, 1}) {
            const std::vector<u32> in = random_poly(n, kind);
            Span source(in);
            const Form expected = leaves(t, in);
            Words F(len);
            t.forward<V>(std::span<const u32>(source.span()), F.begin());
            expect(std::equal(expected.begin(), expected.end(), F.begin()) && F.padding_kept(), name + " forward(in, F)", n);
            Words G(len);
            t.forward_product<V>(std::span<const u32>(source.span()), B.data(), G.begin());
            const Form product = leaf_product(t, expected, B);
            expect(std::equal(product.begin(), product.end(), G.begin()) && G.padding_kept(), name + " forward_product", n);
            expect(source.words() == in && source.edges_kept(), name + " forward: source kept", n);
            if (t.length() > (std::size_t(1) << kMinLog) && n <= len / 2) {
                Words H(len / 2);
                t.forward_upper<V>(std::span<const u32>(source.span()), H.begin());
                expect(std::equal(expected.begin() + std::ptrdiff_t(len / 2), expected.end(), H.begin()) && H.padding_kept(),
                       name + " forward_upper(in, H)", n);
            }
        }
    }
}

template <class V>
void test_products(const std::string& name, const Transform& t, int lg) {
    const std::size_t len = t.length();
    const std::vector<u32> b = random_poly(len), b2 = random_poly(len), a2 = random_poly(len), a3 = random_poly(len);
    const Form B = leaves(t, b), B2 = leaves(t, b2), A2 = leaves(t, a2), A3 = leaves(t, a3);
    for (std::size_t n : source_sizes(len)) {
        const std::vector<u32> in = random_poly(n, n % 3 == 0 ? 1 : 0);
        const std::vector<u32> full = lg <= kSchoolbookMaxLog ? cyclic(in, b, len) : coefficients(t, leaf_product(t, leaves(t, in), B));
        const Form A = leaves(t, in);
        // The sum A B + A2 B2 + A3 (a term alone) - checked against the cyclic product where the
        // other terms are zero.
        Form sum = leaf_product(t, A, B), second = leaf_product(t, A2, B2);
        for (std::size_t i = 0; i < len; ++i) sum[i] = add(add(sum[i], second[i]), A3[i]);
        const std::vector<u32> sum_coefficients = coefficients(t, sum);
        for (std::size_t offset : {std::size_t(0), len / 2}) {
            for (bool negate : {false, true}) {
                const Output o{offset, negate};
                for (std::size_t count : {std::size_t(0), std::size_t(1), std::size_t(5), len / 2 - 3, len / 2, len - offset}) {
                    if (count > len - offset || (lg > kThoroughMaxLog && count < len / 2 - 3)) continue;
                    const std::string what = name + " offset " + std::to_string(offset) + (negate ? " negated" : "") + " count " +
                                             std::to_string(count);
                    Words work(len);
                    Span source(in), out(count);
                    t.cyclic_product<V>(std::span<const u32>(source.span()), B.data(), work.begin(), out.span(), o);
                    expect(out.words() == pick(full, count, o) && out.edges_kept() && work.padding_kept(), what + " cyclic_product", n);
                    expect(source.words() == in, what + " cyclic_product: source kept", n);
                    // In place: out over in.
                    if (count <= n) {
                        Span both(in);
                        t.cyclic_product<V>(std::span<const u32>(both.span()), B.data(), work.begin(),
                                            std::span<u32>(both.data(), count), o);
                        std::vector<u32> expected = pick(full, count, o);
                        expected.insert(expected.end(), in.begin() + std::ptrdiff_t(count), in.end());
                        expect(both.words() == expected && both.edges_kept(), what + " cyclic_product in place", n);
                    }
                    // inverse_product: one term, then three (one alone).
                    const Term one[] = {{A.data(), B.data()}};
                    Span out1(count);
                    t.inverse_product<V>(one, work.begin(), out1.span(), o);
                    expect(out1.words() == pick(full, count, o) && out1.edges_kept(), what + " inverse_product, 1 term", n);
                    const Term three[] = {{A.data(), B.data()}, {A2.data(), B2.data()}, {A3.data(), nullptr}};
                    Span out3(count);
                    t.inverse_product<V>(three, work.begin(), out3.span(), o);
                    expect(out3.words() == pick(sum_coefficients, count, o) && out3.edges_kept(), what + " inverse_product, 3 terms", n);
                }
            }
        }
    }
    // A term alone first, then products; four terms.
    const std::vector<u32> in = random_poly(len);
    const Form A = leaves(t, in);
    const Term four[] = {{A3.data(), nullptr}, {A.data(), B.data()}, {A2.data(), B2.data()}, {B.data(), B2.data()}};
    Form sum = A3;
    for (const auto& [a, c] : {std::pair(A, B), std::pair(A2, B2), std::pair(B, B2)}) {
        const Form p = leaf_product(t, a, c);
        for (std::size_t i = 0; i < len; ++i) sum[i] = add(sum[i], p[i]);
    }
    Words work(len);
    Span out(len);
    t.inverse_product<V>(four, work.begin(), out.span());
    expect(out.words() == coefficients(t, sum), name + " inverse_product, 4 terms");
}

template <class V>
void test_variant(const std::string& name, int max_lg) {
    for (int lg = kMinLog; lg <= max_lg; ++lg) {
        const Transform t(lg);
        test_forwards<V>(name + " lg " + std::to_string(lg), t);
        test_products<V>(name + " lg " + std::to_string(lg), t, lg);
    }
}

void test_checks() {
    const Transform t(8);
    std::vector<u32> in(257), work(256 + kPadding), out(129), B(256);
    const auto throws = [](auto f, const char* what) {
        bool thrown = false;
        try {
            f();
        } catch (const std::logic_error&) {
            thrown = true;
        }
        expect(thrown, what);
    };
    throws([&] { t.forward(std::span<const u32>(in), work.data()); }, "forward: source past the length");
    throws([&] { t.forward_upper(std::span<const u32>(in.data(), 129), work.data()); }, "forward_upper: source past half");
    throws([&] { t.cyclic_product(std::span<const u32>(in.data(), 4), B.data(), work.data(), std::span<u32>(out), {.offset = 128}); },
           "cyclic_product: output past the length");
    throws([&] { t.cyclic_product(std::span<const u32>(in.data(), 4), B.data(), work.data(), std::span<u32>(out.data(), 4), {.offset = 64}); },
           "cyclic_product: offset not 0 or L/2");
    throws([&] { t.inverse_product(std::span<const Term>(), work.data(), std::span<u32>(out.data(), 4)); }, "inverse_product: no term");
    const Transform shortest(kMinLog);
    throws([&] { shortest.forward_upper(std::span<const u32>(in.data(), 4), work.data()); }, "forward_upper at kMinLog");
}

}  // namespace

int main() {
    test_checks();
    test_variant<variant::scalar>("scalar", kScalarMaxLog);
    fps::simd::for_each(fps::simd::All{}, []<class S> {
        test_variant<variant::intrinsics<S>>(std::string("intrinsics ") + S::name, S::lanes == 1 ? kScalarMaxLog : kMaxTestLog);
    });
#ifdef __AVX2__
    test_variant<variant::avx2>("avx2", kMaxTestLog);
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    test_variant<variant::avx512>("avx512", kMaxTestLog);
#endif
    test_variant<variant::tuned>("tuned", kMaxTestLog);
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
