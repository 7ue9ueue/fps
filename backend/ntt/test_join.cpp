// Tests for the joined single-group bottoms at 2^6 and 2^7 (Ntt::kJoin; issue #176 for zmm, #151
// for neon): there every subtree below the top level is one group, and the top levels run all the
// subtrees' bottoms in one call (2 groups from group 0, 2 from group 2 for forward_upper at 2^6,
// 4 from group 0), so the pair kernels take them. Every op that joins, on each variant this
// machine runs (the joined ones and the others), against the scalar variant word for word:
// transforms in place and in the values form, forward_upper and inverse_upper, inverses into other
// memory, Convolution products (sparse and dense), products, squares and middle products from
// spans, the fused ops (forward from spans, forward_product, forward_upper from spans,
// cyclic_product with both halves and negated, inverse_product of 1 to 3 terms), the hooked ops.
// Inputs random, all P - 1, zeros and mixed edges; 2^8 as well, the first length with no join.
#include "backend/ntt/test_util.hpp"

#include <string>

namespace {

constexpr int kMaxTestLog = 8;

// n words of kind 0 random, 1 all P - 1, 2 zeros, 3 mixed (P - 1, 0, random).
std::vector<u32> words(std::size_t n, int kind) {
    std::vector<u32> p(n);
    for (std::size_t i = 0; i < n; ++i) {
        const u32 edge = i % 3 == 0 ? P - 1 : i % 3 == 1 ? 0 : random_below(P);
        p[i] = kind == 0 ? random_below(P) : kind == 1 ? P - 1 : kind == 2 ? 0 : edge;
    }
    return p;
}

// An array for Transform (32-byte aligned, kPadding after) holding v, and its first n words back.
struct Array {
    explicit Array(const std::vector<u32>& v, std::size_t size = 0) : w(std::max(size, v.size())), n(std::max(size, v.size())) {
        std::fill(w.begin(), w.begin() + n + kPadding, 0u);
        std::copy(v.begin(), v.end(), w.begin());
    }
    u32* begin() { return w.begin(); }
    std::vector<u32> get(std::size_t count) { return {w.begin(), w.begin() + count}; }
    Words w;
    std::size_t n;
};

template <class V>
void test_transforms(const std::string& name, const Transform& t, int kind) {
    const std::size_t len = t.length(), half = len / 2;
    const auto f = words(len, kind);
    const auto in_place = [&](auto variant, auto op, const std::vector<u32>& x, std::size_t count) {
        Array a(x, count);
        op(decltype(variant){}, a.begin());
        return a.get(count);
    };
    const auto both = [&](const char* what, auto op, const std::vector<u32>& x, std::size_t count) {
        expect(in_place(V{}, op, x, count) == in_place(variant::scalar{}, op, x, count), (name + what).c_str(), kind);
    };
    both(" forward", [&]<class W>(W, u32* p) { t.forward<W>(p); }, f, len);
    both(" inverse", [&]<class W>(W, u32* p) { t.inverse<W>(p); }, f, len);
    both(" forward values", [&]<class W>(W, u32* p) { t.forward<W>(p, Form::values); }, f, len);
    both(" inverse values", [&]<class W>(W, u32* p) { t.inverse<W>(p, Form::values); }, f, len);
    if (t.length() > (std::size_t(1) << kMinLog)) {
        const std::vector<u32> h(f.begin(), f.begin() + half);
        both(" forward_upper", [&]<class W>(W, u32* p) { t.forward_upper<W>(p); }, h, half);
        both(" inverse_upper", [&]<class W>(W, u32* p) { t.inverse_upper<W>(p); }, h, half);
    }
    // Inverses into other memory: leaves and values, whole, cut, at an offset, added.
    for (const Form form : {Form::leaves, Form::values}) {
        Array F(f, len);
        for (const auto& [count, c] : {std::pair{len, Coefficients{}}, std::pair{len / 2 + 3, Coefficients{}},
                                       std::pair{std::size_t(5), Coefficients{.offset = 7}},
                                       std::pair{len / 2, Coefficients{.offset = len / 2, .add = true}}}) {
            std::vector<u32> x = words(count, 0), y = x;
            t.inverse<V>(F.begin(), std::span(x), form, c);
            t.inverse<variant::scalar>(F.begin(), std::span(y), form, c);
            expect(x == y, (name + " inverse into other memory").c_str(), count);
        }
    }
}

template <class V>
void test_products(const std::string& name, int lg, int kind) {
    const std::size_t len = std::size_t(1) << lg;
    for (const auto& [n, m] : {std::pair{len / 2, len / 2}, std::pair{len - len / 4, len / 4 + 1},
                               std::pair{len / 2 + 1, len / 2 - 1}, std::pair{len - 1, std::size_t(2)},
                               std::pair{len / 4 + 1, len / 4 + 1}}) {
        if (detail::log_length(n, m) != lg) continue;
        const auto f = words(n, kind), g = words(m, (kind + 1) % 4), product = schoolbook(f, g);
        Convolution conv(n, m);
        std::copy(f.begin(), f.end(), conv.a());
        std::copy(g.begin(), g.end(), conv.b());
        const u32* c = conv.template multiply<V>();
        expect(std::vector<u32>(c, c + n + m - 1) == product, (name + " Convolution").c_str(), n * 1000 + m);
        for (const std::size_t count : {n + m - 1, n, n / 2 + 1}) {
            std::vector<u32> out(count);
            multiply<V>(f, g, out);
            expect(out == cut(product, count), (name + " product").c_str(), n * 1000 + count);
        }
        std::vector<u32> sq(2 * n - 1);
        square<V>(f, sq);
        expect(sq == schoolbook(f, f), (name + " square").c_str(), n);
    }
    // Middle products of a transform of 2^lg: f of len words, g of len / 2.
    const auto f = words(len, kind), g = words(len / 2, (kind + 2) % 4), full = schoolbook(f, g);
    std::vector<u32> out(len / 2 + 1);
    middle_product<V>(f, g, out);
    expect(std::equal(out.begin(), out.end(), full.begin() + (g.size() - 1)), (name + " middle_product").c_str(), lg);
}

// The fused and hooked ops, against the scalar variant's.
template <class V>
void test_fused(const std::string& name, const Transform& t, int kind) {
    const std::size_t len = t.length(), half = len / 2;
    Array B(words(len, (kind + 1) % 4)), B2(words(len, kind));
    for (const std::size_t n : {len, half + 1, half, std::size_t(3)}) {
        const auto in = words(n, kind);
        Array x({}, len), y({}, len);
        t.forward<V>(std::span(in), x.begin());
        t.forward<variant::scalar>(std::span(in), y.begin());
        expect(x.get(len) == y.get(len), (name + " forward from a span").c_str(), n);
        t.forward_product<V>(std::span(in), B.begin(), x.begin());
        t.forward_product<variant::scalar>(std::span(in), B.begin(), y.begin());
        expect(x.get(len) == y.get(len), (name + " forward_product").c_str(), n);
        if (n <= half && len > (std::size_t(1) << kMinLog)) {
            t.forward_upper<V>(std::span(in), x.begin());
            t.forward_upper<variant::scalar>(std::span(in), y.begin());
            expect(x.get(half) == y.get(half), (name + " forward_upper from a span").c_str(), n);
        }
        for (const Output o : {Output{}, Output{.offset = half}, Output{.negate = true}}) {
            const std::size_t count = o.offset ? half : n == len ? len : half + 5;
            std::vector<u32> p(count), q(count);
            t.cyclic_product<V>(std::span(in), B.begin(), x.begin(), std::span(p), o);
            t.cyclic_product<variant::scalar>(std::span(in), B.begin(), y.begin(), std::span(q), o);
            expect(p == q, (name + " cyclic_product").c_str(), n * 10 + o.offset + o.negate);
        }
        // forward_hooked: the hook negates its leaves; inverse_hooked: it writes B's.
        const auto negate = [](u32* leaves, std::size_t, std::size_t count) {
            for (std::size_t i = 0; i < 8 * count; ++i) leaves[i] = leaves[i] ? P - leaves[i] : 0;
        };
        t.forward_hooked<V>(std::span(in), x.begin(), negate);
        t.forward_hooked<variant::scalar>(std::span(in), y.begin(), negate);
        expect(x.get(len) == y.get(len), (name + " forward_hooked").c_str(), n);
    }
    const auto write_b = [&](u32* leaves, std::size_t first, std::size_t count) {
        std::copy_n(B.begin() + 8 * first, 8 * count, leaves);
    };
    for (const Output o : {Output{}, Output{.offset = half, .negate = true}}) {
        Array x({}, len), y({}, len);
        std::vector<u32> p(half), q(half);
        t.inverse_hooked<V>(write_b, x.begin(), std::span(p), o);
        t.inverse_hooked<variant::scalar>(write_b, y.begin(), std::span(q), o);
        expect(p == q, (name + " inverse_hooked").c_str(), o.offset);
    }
    Array A(words(len, kind)), A2(words(len, (kind + 3) % 4));
    const Term one[] = {{A.begin(), B.begin()}};
    const Term plain[] = {{A.begin(), nullptr}};
    const Term three[] = {{A.begin(), B.begin()}, {A2.begin(), B2.begin()}, {B2.begin(), nullptr}};
    for (const std::span<const Term> terms : {std::span<const Term>(one), std::span<const Term>(plain), std::span<const Term>(three)}) {
        for (const Output o : {Output{}, Output{.offset = half, .negate = true}}) {
            Array x({}, len), y({}, len);
            const std::size_t count = o.offset ? half : len - 3;
            std::vector<u32> p(count), q(count);
            t.inverse_product<V>(terms, x.begin(), std::span(p), o);
            t.inverse_product<variant::scalar>(terms, y.begin(), std::span(q), o);
            expect(p == q, (name + " inverse_product").c_str(), terms.size() * 10 + o.negate);
        }
    }
}

template <class V>
void test_variant(const char* name) {
    for (int lg = kMinLog; lg <= kMaxTestLog; ++lg) {
        const Transform t(lg);
        const std::string at = std::string(name) + " lg " + std::to_string(lg);
        for (int kind = 0; kind < 4; ++kind) {
            test_transforms<V>(at, t, kind);
            test_products<V>(at, lg, kind);
            test_fused<V>(at, t, kind);
        }
    }
}

}  // namespace

int main() {
    for_each_variant([]<class V>(const char* name, int) { test_variant<V>(name); });
    test_variant<variant::tuned>("tuned");
    return finish();
}
