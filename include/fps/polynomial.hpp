// Polynomials modulo a prime P: fps::Polynomial<P> and its exact arithmetic. v1 compiles for
// P = 998244353 only (DESIGN.md, Moduli).
//
//   using Poly = fps::Polynomial<998244353>;
//   const Poly x = Poly::x();
//   Poly f = 1 - 4 * x, g{1, 2, 3};             // coefficients c[0] + c[1] x + ...
//   Poly(n), Poly(span), Poly::x(k)             // n zeros; a copy of the span; x^k
//   f.size(), f.degree()                        // stored length; last nonzero index, -1 for zero
//   f[i], f.coeff(i), f.resize(n), f.span()     // coeff(i) is 0 past the end
//   f.truncate(n), f.reverse(n), f(c)           // f mod x^n; x^(n-1) f(1/x) mod x^n; f at c
//   f + g, f - g, -f, f * g                     // exact; also +=, -=, *=
//   f + c, c + f, f - c, c - f, f * c, c * f    // c an integer of at most 64 bits, taken mod P
//
// Coefficients are canonical, in [0, P), in and out: the constructors throw std::domain_error on
// one >= P. size() counts trailing zeros and nothing trims: f + g and f - g have
// max(f.size(), g.size()) coefficients, f + c and f - c max(f.size(), 1), f * c f.size(), and
// f * g f.size() + g.size() - 1 (0 if either is empty).
//
// The operators allocate their result and write it in one pass. +=, -= and *= c work in place
// and allocate nothing, except += and -= with a longer g, which build the result in one pass too.
//
// f * g with n and m coefficients: a factor with at most one nonzero term (c x^k) makes it a
// shift, O(n + m); a product the backend's small::multiply does faster than the NTT on this target
// runs it (small::beats_ntt(n, m), tuned per target: on the timed targets min(n, m) up to 33-59
// next to a factor of 2^10 coefficients, 66-110 next to one of 2^20, or n = m up to 45-101); the
// rest runs ntt::multiply, which needs n + m - 1 <= 2^25 (else std::length_error). f * f (the same
// polynomial on both sides) is a square: ntt::square on the NTT path, one forward transform fewer,
// and small::multiply up to n = 41-78. Both write into the result's own memory, so nothing is
// copied out. The same paths run mul, square, middle_product and raw::mul (fps/products.hpp).
#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "backend/modarith/modarith.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/simd/select.hpp"
#include "backend/small/small.hpp"
#include "fps/modular.hpp"
#include "fps/words.hpp"

namespace fps {

template <std::uint32_t P>
class Polynomial;

namespace detail {

// A scalar for the operators: an integer type of at most 64 bits, so residue() is exact.
template <class T>
concept Integer = std::integral<T> && sizeof(T) <= 8;

// c mod P.
template <std::uint32_t P, Integer T>
constexpr std::uint32_t residue(T c) {
    if constexpr (std::is_signed_v<T>) {
        const std::int64_t r = std::int64_t(c) % std::int64_t(P);
        return std::uint32_t(r < 0 ? r + std::int64_t(P) : r);
    } else {
        return std::uint32_t(std::uint64_t(c) % P);
    }
}

// out[i] = op(a[i], b[i]) for i < n, as op.template operator()<S>(x, y) on vectors of
// simd::Native, then on single words (simd::Scalar). out may be a or b.
template <class Op>
void lanewise(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t* out, std::size_t n, Op op) {
    using S = simd::Native;
    std::size_t i = 0;
    for (; i + S::lanes <= n; i += S::lanes)
        S::store(out + i, op.template operator()<S>(S::load(a + i), S::load(b + i)));
    for (; i < n; ++i) out[i] = op.template operator()<simd::Scalar>(a[i], b[i]);
}

// The term c x^k of a polynomial with at most one nonzero coefficient; {0, 0} for zero.
struct Term {
    std::size_t k;
    std::uint32_t c;
};

inline std::optional<Term> single_term(std::span<const std::uint32_t> f) {
    Term t{0, 0};
    for (std::size_t i = 0; i < f.size(); ++i) {
        if (f[i] == 0) continue;
        if (t.c != 0) return std::nullopt;
        t = {i, f[i]};
    }
    return t;
}

// n coefficients, unwritten: for the library's functions that write all of them.
template <std::uint32_t P>
Polynomial<P> unwritten_polynomial(std::size_t n);

// The products of polynomials (f * g here; mul, square, middle_product and raw::mul in
// fps/products.hpp), defined after Polynomial.
template <std::uint32_t P>
struct Products;

}  // namespace detail

template <std::uint32_t P>
class Polynomial {
    static_assert(P == ntt::kModulus, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");

public:
    Polynomial() = default;
    explicit Polynomial(std::size_t n) : c_(n) {}
    Polynomial(std::initializer_list<std::uint32_t> c) : Polynomial(std::span(c.begin(), c.size())) {}
    explicit Polynomial(std::span<const std::uint32_t> c) : Polynomial(c.size(), kUnwritten) {
        std::ranges::copy(c, data());
        if (std::ranges::any_of(c, [](std::uint32_t v) { return v >= P; }))
            throw std::domain_error("fps::Polynomial: coefficient >= P");
    }

    static Polynomial x(std::size_t k = 1) {
        Polynomial f(k + 1);
        f[k] = 1;
        return f;
    }

    std::size_t size() const { return c_.size(); }

    int degree() const {
        std::size_t d = size();
        while (d > 0 && data()[d - 1] == 0) --d;
        return int(d) - 1;
    }

    std::uint32_t& operator[](std::size_t i) { return data()[i]; }
    std::uint32_t operator[](std::size_t i) const { return data()[i]; }
    std::uint32_t coeff(std::size_t i) const { return i < size() ? data()[i] : 0; }
    void resize(std::size_t n) { c_.resize(n); }
    std::span<std::uint32_t> span() { return {data(), size()}; }
    std::span<const std::uint32_t> span() const { return {data(), size()}; }

    Polynomial truncate(std::size_t n) const {
        Polynomial g(n, kUnwritten);
        const std::size_t k = std::min(n, size());
        std::copy_n(data(), k, g.data());
        std::fill(g.data() + k, g.data() + n, 0u);
        return g;
    }

    // g[i] = f[n - 1 - i].
    Polynomial reverse(std::size_t n) const {
        Polynomial g(n, kUnwritten);
        const std::size_t k = std::min(n, size());
        std::fill_n(g.data(), n - k, 0u);
        std::reverse_copy(data(), data() + k, g.data() + (n - k));
        return g;
    }

    // Horner's rule; any c < 2^32.
    std::uint32_t operator()(std::uint32_t c) const {
        std::uint32_t r = 0;
        for (std::size_t i = size(); i-- > 0;) r = Word::add(mod_mul<P>(r, c), data()[i]);
        return r;
    }

    // A longer g needs new memory: the sum is built there in one pass.
    Polynomial& operator+=(const Polynomial& g) {
        if (g.size() > size()) return *this = std::as_const(*this) + g;
        add(data(), g.data(), data(), g.size());
        return *this;
    }

    Polynomial& operator-=(const Polynomial& g) {
        if (g.size() > size()) return *this = std::as_const(*this) - g;
        subtract(data(), g.data(), data(), g.size());
        return *this;
    }

    Polynomial& operator*=(const Polynomial& g) { return *this = *this * g; }

    template <detail::Integer T>
    Polynomial& operator+=(T c) {
        if (size() == 0) resize(1);
        data()[0] = Word::add(data()[0], detail::residue<P>(c));
        return *this;
    }

    template <detail::Integer T>
    Polynomial& operator-=(T c) {
        if (size() == 0) resize(1);
        data()[0] = Word::sub(data()[0], detail::residue<P>(c));
        return *this;
    }

    template <detail::Integer T>
    Polynomial& operator*=(T c) {
        scale(data(), detail::residue<P>(c), data(), size());
        return *this;
    }

    friend Polynomial operator+(const Polynomial& f, const Polynomial& g) {
        const Polynomial& longer = f.size() >= g.size() ? f : g;
        const std::size_t n = std::min(f.size(), g.size());
        Polynomial h(longer.size(), kUnwritten);
        add(f.data(), g.data(), h.data(), n);
        std::copy(longer.data() + n, longer.data() + longer.size(), h.data() + n);
        return h;
    }

    friend Polynomial operator+(Polynomial&& f, const Polynomial& g) {
        f += g;
        return std::move(f);
    }

    friend Polynomial operator-(const Polynomial& f, const Polynomial& g) {
        const std::size_t n = std::min(f.size(), g.size());
        Polynomial h(std::max(f.size(), g.size()), kUnwritten);
        subtract(f.data(), g.data(), h.data(), n);
        std::copy(f.data() + n, f.data() + f.size(), h.data() + n);  // f is the longer
        negate(g.data() + n, h.data() + n, g.size() - n);            // g is the longer
        return h;
    }

    friend Polynomial operator-(Polynomial&& f, const Polynomial& g) {
        f -= g;
        return std::move(f);
    }

    friend Polynomial operator-(const Polynomial& f) {
        Polynomial h(f.size(), kUnwritten);
        negate(f.data(), h.data(), f.size());
        return h;
    }

    friend Polynomial operator*(const Polynomial& f, const Polynomial& g) {
        return detail::Products<P>::multiply(f.span(), g.span(), f.size() && g.size() ? f.size() + g.size() - 1 : 0);
    }

    // f + c, c + f and f - c change one coefficient: the copy of f is their one pass.
    template <detail::Integer T>
    friend Polynomial operator+(Polynomial f, T c) {
        f += c;
        return f;
    }

    template <detail::Integer T>
    friend Polynomial operator+(T c, Polynomial f) {
        f += c;
        return f;
    }

    template <detail::Integer T>
    friend Polynomial operator-(Polynomial f, T c) {
        f -= c;
        return f;
    }

    template <detail::Integer T>
    friend Polynomial operator-(T c, const Polynomial& f) {
        Polynomial h = -f;
        h += c;
        return h;
    }

    template <detail::Integer T>
    friend Polynomial operator*(const Polynomial& f, T c) {
        Polynomial h(f.size(), kUnwritten);
        scale(f.data(), detail::residue<P>(c), h.data(), f.size());
        return h;
    }

    template <detail::Integer T>
    friend Polynomial operator*(T c, const Polynomial& f) {
        return f * c;
    }

private:
    template <class S>
    using Ops = modarith::Ops<S, P>;
    using Word = Ops<simd::Scalar>;
    static constexpr detail::Words::Unwritten kUnwritten{};

    template <std::uint32_t Q>
    friend Polynomial<Q> detail::unwritten_polynomial(std::size_t n);
    friend struct detail::Products<P>;

    Polynomial(std::size_t n, detail::Words::Unwritten u) : c_(n, u) {}
    Polynomial(std::size_t n, std::size_t capacity, detail::Words::Unwritten u) : c_(n, capacity, u) {}

    std::uint32_t* data() { return c_.data(); }
    const std::uint32_t* data() const { return c_.data(); }

    // The streams, for i < n; out may be a or b.
    static void add(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t* out, std::size_t n) {
        detail::lanewise(a, b, out, n, []<class S>(typename S::V x, typename S::V y) { return Ops<S>::add(x, y); });
    }

    static void subtract(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t* out, std::size_t n) {
        detail::lanewise(a, b, out, n, []<class S>(typename S::V x, typename S::V y) { return Ops<S>::sub(x, y); });
    }

    static void negate(const std::uint32_t* a, std::uint32_t* out, std::size_t n) {
        detail::lanewise(a, a, out, n,
                         []<class S>(typename S::V x, typename S::V) { return Ops<S>::sub(S::broadcast(0), x); });
    }

    // out[i] = c a[i] by Shoup's multiplication.
    static void scale(const std::uint32_t* a, std::uint32_t c, std::uint32_t* out, std::size_t n) {
        const std::uint32_t q = Word::shoup_quotient(c);
        detail::lanewise(a, a, out, n, [c, q]<class S>(typename S::V x, typename S::V) {
            return Ops<S>::template reduce<P>(Ops<S>::shoup(x, S::broadcast(c), S::broadcast(q)));
        });
    }

    detail::Words c_;
};

template <std::uint32_t P>
Polynomial<P> detail::unwritten_polynomial(std::size_t n) {
    return Polynomial<P>(n, Polynomial<P>::kUnwritten);
}

namespace detail {

// a b mod x^n for f * g, mul and square, the middle product, and raw::mul. Factors are cut to n
// coefficients first. A factor with at most one nonzero term (c x^k) makes a shift, O(n); a
// product small::beats_ntt gives to the backend's small::multiply runs it (a square or a middle
// product: small_square, small_middle); the rest runs ntt::multiply (ntt::square,
// ntt::middle_product), which writes into the caller's memory.
template <std::uint32_t P>
struct Products {
    using Poly = Polynomial<P>;
    using Span = std::span<const std::uint32_t>;
    using Out = std::span<std::uint32_t>;

    // n coefficients. Both kernels write into the result's own memory: no copy out. a and b the same
    // words (f * f, mul(f, f, n)) make a square.
    static Poly multiply(Span a, Span b, std::size_t n) {
        a = a.first(std::min(a.size(), n)), b = b.first(std::min(b.size(), n));
        if (a.data() == b.data() && a.size() == b.size()) return square(a, n);
        if (a.empty() || b.empty()) return Poly(n);
        if (const auto t = single_term(b)) return shifted(a, *t, n);
        if (const auto t = single_term(a)) return shifted(b, *t, n);
        if (small::beats_ntt(a.size(), b.size())) return small_product(a, b, n);
        Poly h(n, Poly::kUnwritten);
        ntt::multiply(a, b, h.span());
        return h;
    }

    // a^2 mod x^n, a.size() <= n; square(a) all 2 a.size() - 1 coefficients.
    static Poly square(Span a, std::size_t n) {
        if (a.empty()) return Poly(n);
        if (const auto t = single_term(a)) return shifted(a, *t, n);
        if (small_square(a.size())) return small_product(a, a, n);
        Poly h(n, Poly::kUnwritten);
        ntt::square(a, h.span());
        return h;
    }

    static Poly square(Span a) { return square(a, a.empty() ? 0 : 2 * a.size() - 1); }

    // out = a b mod x^out.size(); out apart from a and b. a and b the same words make a square.
    static void multiply_into(Span a, Span b, Out out) {
        const std::size_t n = out.size();
        a = a.first(std::min(a.size(), n)), b = b.first(std::min(b.size(), n));
        if (a.empty() || b.empty()) return void(std::ranges::fill(out, 0u));
        const bool same = a.data() == b.data() && a.size() == b.size();
        if (const auto t = single_term(b)) return shift(a, *t, out);
        if (!same)
            if (const auto t = single_term(a)) return shift(b, *t, out);
        const std::size_t size = a.size() + b.size() - 1;
        if (same ? small_square(a.size()) : small::beats_ntt(a.size(), b.size())) {
            if (n < size) return low_product(a, b, out);
            small::multiply<P>(a, b, out.first(size));
            return void(std::fill(out.begin() + size, out.end(), 0u));
        }
        if (same) ntt::square(a, out);
        else ntt::multiply(a, b, out);
    }

    // Coefficients m - 1 .. n - 1 of f g, for n = f.size() >= m = g.size() >= 1: ntt::middle_product
    // into the result (a cyclic product of length >= n), or all of f g by small::multiply in the
    // result's own memory, then the n - m + 1 coefficients moved down.
    static Poly middle(Span f, Span g) {
        const std::size_t n = f.size(), m = g.size(), count = n - m + 1;
        if (!small_middle(n, m)) {
            Poly h(count, Poly::kUnwritten);
            ntt::middle_product(f, g, h.span());
            return h;
        }
        Poly h(count, n + m - 1, Poly::kUnwritten);
        small::multiply<P>(f, g, {h.data(), n + m - 1});
        std::memmove(h.data(), h.data() + (m - 1), count * sizeof(std::uint32_t));
        return h;
    }

    // Whether small::multiply beats the NTT path, by small's cost model against the NTT's time at
    // the length the NTT runs (backend/small: model_ns, ntt_ns, the NTT's tuned times). beats_ntt
    // takes ntt::multiply at n + m - 1 words; these take the NTT ops the paths above run instead:
    // - a^2 for a of n >= 1 words: ntt::square at 2 n - 1, one forward transform fewer than a
    //   product, kSquareShare of its time;
    // - the middle product of f of n >= g of m >= 1 words: ntt::middle_product, a cyclic product of
    //   length >= n (as a product of n words), while small::multiply computes all n + m - 1.
    static bool small_square(std::size_t n) {
        return small::model_ns(small::kModel, n, n) < kSquareShare * small::ntt_ns(2 * n - 1);
    }

    static bool small_middle(std::size_t n, std::size_t m) {
        return small::model_ns(small::kModel, n, m) < small::ntt_ns(n);
    }

    // ntt::square's time over ntt::multiply's at the same length where squares cross small: 0.74-0.92
    // at n = 10-160 on the timed targets (frontend/notes/271.md).
    static constexpr double kSquareShare = 0.85;

private:
    // a b mod x^n by small::multiply, which writes all of a b: past n too, into spare words of the
    // result.
    static Poly small_product(Span a, Span b, std::size_t n) {
        const std::size_t size = a.size() + b.size() - 1;
        Poly h = n >= size ? Poly(n, Poly::kUnwritten) : Poly(n, size, Poly::kUnwritten);
        small::multiply<P>(a, b, {h.data(), size});
        if (n > size) std::fill(h.data() + size, h.data() + n, 0u);
        return h;
    }

    static Poly shifted(Span f, Term t, std::size_t n) {
        Poly h(n, Poly::kUnwritten);
        shift(f, t, h.span());
        return h;
    }

    // out = c x^k f mod x^out.size() for the term t = c x^k.
    static void shift(Span f, Term t, Out out) {
        const std::size_t n = out.size(), k = std::min(t.k, n), count = std::min(f.size(), n - k);
        std::uint32_t* const middle = out.data() + k;
        std::fill(out.data(), middle, 0u);
        if (t.c == 1) std::copy_n(f.data(), count, middle);
        else Poly::scale(f.data(), t.c, middle, count);
        std::fill(middle + count, out.data() + n, 0u);
    }

    // out = a b mod x^out.size() for out shorter than a b, which small::multiply cannot write
    // (backend issue #37): the plain loop, in blocks of 64 outputs. 64-bit sums: a reduced sum
    // plus 16 products below P^2 stays below 2^64 for P < 2^30.
    static void low_product(Span a, Span b, Out out) {
        constexpr std::size_t kBlock = 64;
        std::uint64_t sum[kBlock];
        for (std::size_t lo = 0; lo < out.size(); lo += kBlock) {
            const std::size_t hi = std::min(out.size(), lo + kBlock);
            std::fill(sum, sum + (hi - lo), 0);
            const std::size_t first = lo < a.size() ? 0 : lo - a.size() + 1, last = std::min(b.size(), hi);
            for (std::size_t j = first; j < last; ++j) {
                const std::size_t begin = lo > j ? lo - j : 0, end = std::min(a.size(), hi - j);
                for (std::size_t i = begin; i < end; ++i) sum[i + j - lo] += std::uint64_t(a[i]) * b[j];
                if ((j - first) % 16 == 15)
                    for (std::size_t k = 0; k < hi - lo; ++k) sum[k] %= P;
            }
            for (std::size_t k = 0; k < hi - lo; ++k) out[lo + k] = std::uint32_t(sum[k] % P);
        }
    }
};

}  // namespace detail

}  // namespace fps
