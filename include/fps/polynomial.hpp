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
//   f + c, c + f, f - c, c - f, f * c, c * f    // c any integer, taken mod P; also +=, -=, *=
//
// Coefficients are canonical, in [0, P), in and out: the constructors throw std::domain_error on
// one >= P. size() counts trailing zeros and nothing trims: f + g and f - g have
// max(f.size(), g.size()) coefficients, f + c and f - c max(f.size(), 1), f * c f.size(), and
// f * g f.size() + g.size() - 1 (0 if either is empty).
//
// f * g with n and m coefficients: a factor with at most one nonzero term (c x^k) makes it a
// shift, O(n + m); min(n, m) <= 32 or n m < 2^18 runs schoolbook; the rest runs the backend's
// ntt::Convolution, which needs n + m - 1 <= 2^25 (else std::length_error) and leaves the product
// in its own memory, so f * g copies it out. Every call allocates its result.
#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "backend/modarith/modarith.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/simd/select.hpp"
#include "fps/modular.hpp"

namespace fps {

namespace detail {

// c mod P for an integer c of at most 64 bits.
template <std::uint32_t P, std::integral T>
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

// Whether a product of n and m coefficients runs schoolbook rather than ntt::Convolution. On Zen 3
// schoolbook costs 0.24 ns per term; a Convolution 51 us at any short length (a fresh mapping per
// call) plus 3 to 4.5 ns per coefficient of its length (frontend/notes.md).
constexpr bool schoolbook_wins(std::size_t n, std::size_t m) {
    return std::min(n, m) <= 32 || n * m < (std::size_t(1) << 18);
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
    Polynomial(std::initializer_list<std::uint32_t> c) : c_(c) { check_canonical(); }
    explicit Polynomial(std::span<const std::uint32_t> c) : c_(c.begin(), c.end()) { check_canonical(); }

    static Polynomial x(std::size_t k = 1) {
        Polynomial f(k + 1);
        f.c_[k] = 1;
        return f;
    }

    std::size_t size() const { return c_.size(); }

    int degree() const {
        std::size_t d = c_.size();
        while (d > 0 && c_[d - 1] == 0) --d;
        return int(d) - 1;
    }

    std::uint32_t& operator[](std::size_t i) { return c_[i]; }
    std::uint32_t operator[](std::size_t i) const { return c_[i]; }
    std::uint32_t coeff(std::size_t i) const { return i < c_.size() ? c_[i] : 0; }
    void resize(std::size_t n) { c_.resize(n); }
    std::span<std::uint32_t> span() { return c_; }
    std::span<const std::uint32_t> span() const { return c_; }

    Polynomial truncate(std::size_t n) const {
        Polynomial g;
        g.c_.assign(c_.begin(), c_.begin() + std::ptrdiff_t(std::min(n, size())));
        g.c_.resize(n);
        return g;
    }

    // g[i] = f[n - 1 - i].
    Polynomial reverse(std::size_t n) const {
        Polynomial g(n);
        const std::size_t k = std::min(n, size());
        std::reverse_copy(c_.begin(), c_.begin() + std::ptrdiff_t(k), g.c_.begin() + std::ptrdiff_t(n - k));
        return g;
    }

    // Horner's rule; any c < 2^32.
    std::uint32_t operator()(std::uint32_t c) const {
        std::uint32_t r = 0;
        for (std::size_t i = c_.size(); i-- > 0;) r = Word::add(mod_mul<P>(r, c), c_[i]);
        return r;
    }

    Polynomial& operator+=(const Polynomial& g) {
        if (g.size() > size()) c_.resize(g.size());
        detail::lanewise(c_.data(), g.c_.data(), c_.data(), g.size(),
                         []<class S>(typename S::V x, typename S::V y) { return Ops<S>::add(x, y); });
        return *this;
    }

    Polynomial& operator-=(const Polynomial& g) {
        if (g.size() > size()) c_.resize(g.size());
        detail::lanewise(c_.data(), g.c_.data(), c_.data(), g.size(),
                         []<class S>(typename S::V x, typename S::V y) { return Ops<S>::sub(x, y); });
        return *this;
    }

    Polynomial& operator*=(const Polynomial& g) { return *this = *this * g; }

    template <std::integral T>
    Polynomial& operator+=(T c) {
        if (c_.empty()) c_.resize(1);
        c_[0] = Word::add(c_[0], detail::residue<P>(c));
        return *this;
    }

    template <std::integral T>
    Polynomial& operator-=(T c) {
        if (c_.empty()) c_.resize(1);
        c_[0] = Word::sub(c_[0], detail::residue<P>(c));
        return *this;
    }

    template <std::integral T>
    Polynomial& operator*=(T c) {
        scale(c_.data(), detail::residue<P>(c), c_.data(), size());
        return *this;
    }

    friend Polynomial operator+(Polynomial f, const Polynomial& g) {
        f += g;
        return f;
    }

    friend Polynomial operator-(Polynomial f, const Polynomial& g) {
        f -= g;
        return f;
    }

    friend Polynomial operator-(Polynomial f) {
        detail::lanewise(f.c_.data(), f.c_.data(), f.c_.data(), f.size(),
                         []<class S>(typename S::V x, typename S::V) { return Ops<S>::sub(S::broadcast(0), x); });
        return f;
    }

    friend Polynomial operator*(const Polynomial& f, const Polynomial& g) {
        if (f.c_.empty() || g.c_.empty()) return {};
        if (const auto t = detail::single_term(g.span())) return shifted(f, *t, g.size());
        if (const auto t = detail::single_term(f.span())) return shifted(g, *t, f.size());
        if (detail::schoolbook_wins(f.size(), g.size())) {
            Polynomial h(f.size() + g.size() - 1);
            detail::schoolbook<P>(f.c_, g.c_, h.c_);
            return h;
        }
        return convolution(f, g);
    }

    template <std::integral T>
    friend Polynomial operator+(Polynomial f, T c) {
        f += c;
        return f;
    }

    template <std::integral T>
    friend Polynomial operator+(T c, Polynomial f) {
        f += c;
        return f;
    }

    template <std::integral T>
    friend Polynomial operator-(Polynomial f, T c) {
        f -= c;
        return f;
    }

    template <std::integral T>
    friend Polynomial operator-(T c, Polynomial f) {
        f = -std::move(f);
        f += c;
        return f;
    }

    template <std::integral T>
    friend Polynomial operator*(Polynomial f, T c) {
        f *= c;
        return f;
    }

    template <std::integral T>
    friend Polynomial operator*(T c, Polynomial f) {
        f *= c;
        return f;
    }

private:
    template <class S>
    using Ops = modarith::Ops<S, P>;
    using Word = Ops<simd::Scalar>;

    void check_canonical() const {
        if (std::ranges::any_of(c_, [](std::uint32_t c) { return c >= P; }))
            throw std::domain_error("fps::Polynomial: coefficient >= P");
    }

    // out[i] = c a[i] for i < n, by Shoup's multiplication; out may be a.
    static void scale(const std::uint32_t* a, std::uint32_t c, std::uint32_t* out, std::size_t n) {
        const std::uint32_t q = Word::shoup_quotient(c);
        detail::lanewise(a, a, out, n, [c, q]<class S>(typename S::V x, typename S::V) {
            return Ops<S>::template reduce<P>(Ops<S>::shoup(x, S::broadcast(c), S::broadcast(q)));
        });
    }

    // f t for the term t of a factor of m coefficients: f.size() + m - 1 coefficients.
    static Polynomial shifted(const Polynomial& f, detail::Term t, std::size_t m) {
        Polynomial h(f.size() + m - 1);
        if (t.c == 1) std::ranges::copy(f.c_, h.c_.begin() + std::ptrdiff_t(t.k));
        else scale(f.c_.data(), t.c, h.c_.data() + t.k, f.size());
        return h;
    }

    static Polynomial convolution(const Polynomial& f, const Polynomial& g) {
        ntt::Convolution conv(f.size(), g.size());
        std::ranges::copy(f.c_, conv.a());
        std::ranges::copy(g.c_, conv.b());
        const std::uint32_t* product = conv.multiply();
        Polynomial h;
        h.c_.assign(product, product + conv.size());
        return h;
    }

    std::vector<std::uint32_t> c_;
};

}  // namespace fps
