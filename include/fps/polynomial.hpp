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
// The operators allocate their result and write it in one pass; +=, -= and *= c work in place and
// allocate nothing (+= and -= only to grow).
//
// f * g with n and m coefficients: a factor with at most one nonzero term (c x^k) makes it a
// shift, O(n + m); min(n, m) <= 32 or n m < 3 2^16 runs schoolbook; the rest runs the backend's
// ntt::Convolution, which needs n + m - 1 <= 2^25 (else std::length_error). The Convolution keeps
// the product in its own memory, so f * g copies it out: 0.6% to 6% over the kernel on x86 from
// 2^10 coefficients (frontend/notes.md). The fast path is fps::ntt::Convolution itself, reading
// the product from multiply(); once backend issue #38 lets it write into caller memory, the fast
// path is fps::raw::mul (DESIGN.md, Raw functions).
#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "backend/modarith/modarith.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/simd/select.hpp"
#include "fps/modular.hpp"

namespace fps {

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

// Coefficient storage: size() words on the heap, none reserved. Unlike std::vector it can
// allocate without writing, so an operator writes its result in one pass, and it copies with
// memcpy (std::vector with such an allocator copies word by word under GCC 15).
class Words {
public:
    struct Unwritten {};

    Words() = default;
    explicit Words(std::size_t n) : Words(n, Unwritten{}) { std::fill_n(data(), n, 0u); }
    Words(std::size_t n, Unwritten)
        : words_(n ? std::make_unique_for_overwrite<std::uint32_t[]>(n) : nullptr), size_(n) {}
    Words(const Words& w) : Words(w.size_, Unwritten{}) { std::copy_n(w.data(), size_, data()); }
    Words(Words&& w) noexcept : words_(std::move(w.words_)), size_(std::exchange(w.size_, 0)) {}
    Words& operator=(const Words& w) { return *this = Words(w); }
    Words& operator=(Words&& w) noexcept {
        words_ = std::move(w.words_);
        size_ = std::exchange(w.size_, 0);
        return *this;
    }

    std::size_t size() const { return size_; }
    std::uint32_t* data() { return words_.get(); }
    const std::uint32_t* data() const { return words_.get(); }

    // Words past the old size are zero.
    void resize(std::size_t n) {
        if (n <= size_) return void(size_ = n);
        Words w(n, Unwritten{});
        std::copy_n(data(), size_, w.data());
        std::fill(w.data() + size_, w.data() + n, 0u);
        *this = std::move(w);
    }

private:
    std::unique_ptr<std::uint32_t[]> words_;
    std::size_t size_ = 0;
};

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

// Whether a product of n and m coefficients runs schoolbook rather than ntt::Convolution. On Zen 3
// schoolbook costs 0.24 ns per term; a Convolution 51 us at any short length (a fresh mapping per
// call) plus 3 to 4.5 ns per coefficient of its length. They meet near 443 x 443 (3 2^16 terms)
// and, for long factors, between m = 32 and m = 48 (frontend/notes.md).
constexpr bool schoolbook_wins(std::size_t n, std::size_t m) {
    return std::min(n, m) <= 32 || n * m < 3 * (std::size_t(1) << 16);
}

// out = a b for out.size() = a.size() + b.size() - 1, in blocks of 64 outputs. 64-bit sums: a
// reduced sum plus 16 products below P^2 stays below 2^64 for P < 2^30.
template <std::uint32_t P>
void schoolbook(std::span<const std::uint32_t> a, std::span<const std::uint32_t> b, std::span<std::uint32_t> out) {
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

    Polynomial& operator+=(const Polynomial& g) {
        if (g.size() > size()) resize(g.size());
        add(data(), g.data(), data(), g.size());
        return *this;
    }

    Polynomial& operator-=(const Polynomial& g) {
        if (g.size() > size()) resize(g.size());
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
        if (f.size() == 0 || g.size() == 0) return {};
        if (const auto t = detail::single_term(g.span())) return shifted(f, *t, g.size());
        if (const auto t = detail::single_term(f.span())) return shifted(g, *t, f.size());
        if (detail::schoolbook_wins(f.size(), g.size())) {
            Polynomial h(f.size() + g.size() - 1, kUnwritten);
            detail::schoolbook<P>(f.span(), g.span(), h.span());
            return h;
        }
        return convolution(f, g);
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

    Polynomial(std::size_t n, detail::Words::Unwritten u) : c_(n, u) {}

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

    // f t for the term t of a factor of m coefficients: f.size() + m - 1 coefficients.
    static Polynomial shifted(const Polynomial& f, detail::Term t, std::size_t m) {
        Polynomial h(f.size() + m - 1, kUnwritten);
        std::uint32_t* const middle = h.data() + t.k;
        std::fill(h.data(), middle, 0u);
        if (t.c == 1) std::copy_n(f.data(), f.size(), middle);
        else scale(f.data(), t.c, middle, f.size());
        std::fill(middle + f.size(), h.data() + h.size(), 0u);
        return h;
    }

    static Polynomial convolution(const Polynomial& f, const Polynomial& g) {
        ntt::Convolution conv(f.size(), g.size());
        std::copy_n(f.data(), f.size(), conv.a());
        std::copy_n(g.data(), g.size(), conv.b());
        const std::uint32_t* product = conv.multiply();
        Polynomial h(conv.size(), kUnwritten);
        std::copy_n(product, conv.size(), h.data());
        return h;
    }

    detail::Words c_;
};

}  // namespace fps
