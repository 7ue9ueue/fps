// Rational series modulo P = 998244353 (DESIGN.md, Fraction): fps::Fraction<P>, the infinite
// series p / q, stored as the two polynomials.
//
//   using Frac = fps::Fraction<998244353>;
//   Frac fib(x, 1 - x - x * x);          // p / q: 0, 1, 1, 2, 3, 5, 8, ...; q[0] != 0
//   Frac a = f;                          // f / 1, for a polynomial f
//   a[k]                                 // coefficient k, any k < 2^64 (Bostan-Mori)
//   a.take(n)                            // the first n coefficients
//   a.numerator(), a.denominator()       // p, q
//   a + b, a - b, -a, c * a, a * c       // termwise; c an integer of at most 64 bits, taken mod P
//   a * b, f * a, a * f                  // product of series (convolution of the sequences)
//   a / f, a / b                         // quotient of series; f[0] != 0, b's numerator [0] != 0
//   a / (1 - x)                          // prefix sums: coefficient k is a_0 + ... + a_k
//   a * Poly::x(k)                       // the sequence moved k places later
//   fps::derivative(a)                   // coefficient k is (k + 1) a_(k+1)
//
// A q with q[0] = 0 (an empty q too) throws std::domain_error, from the constructor, and so does
// a / f or a / b with f[0] = 0 or b's numerator [0] = 0: their denominators are q f and q p_b.
//
// Exact and eager: an operation multiplies numerators and denominators by f * g
// (fps/polynomial.hpp), so sizes add and nothing is reduced (fib + fib has denominator q^2).
// a + b and a - b cost three products, a * b and a / b two, a / f and f * a one, derivative(a)
// three (one a square); -a and c * a are O(d) streams. a * b and a + b run within 2% of the same
// ntt::multiply calls by hand from 2^10 coefficients (frontend/notes/23.md).
//
// a[k]: fps::bostan_mori::coefficient (backend/bostan_mori, Bostan and Mori) on p and q as
// they are, one call: p(x) q(-x) = U_0(x^2) + x U_1(x^2), q(x) q(-x) = V(x^2), and
// [x^k] p / q = [x^(k / 2)] U_(k mod 2) / V, with p and q cut to k + 1 coefficients. Schoolbook
// steps while p and q are short (76 to 96 words by target), steps in the NTT's values form beyond:
// O(d log d log k), d the larger of p.size() and q.size(). Nothing is cached: each [k] is a full
// computation; for many small indices, call take(n) once. k = 10^18: 0.8 to 1.2 us for
// Fibonacci; d = 10^5: 21 to 59 ms, d = 2^20: 0.36 to 1.06 s (Zen 5 to Neoverse V2;
// frontend/notes/191.md). The kernel keeps its arrays per thread and allocates only for a call
// longer than the thread's earlier ones.
//
// take(n) = mul(p, inverse(q, n), n) (fps/products.hpp, fps/series.hpp): inverse runs a linear
// recurrence, O(n t), for a q with t <= 16 nonzero terms past q[0], Newton iteration otherwise.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "backend/bostan_mori/bostan_mori.hpp"
#include "fps/polynomial.hpp"
#include "fps/products.hpp"
#include "fps/series.hpp"

namespace fps {

template <std::uint32_t P>
class Fraction {
public:
    Fraction(Polynomial<P> p, Polynomial<P> q) : p_(std::move(p)), q_(std::move(q)) {
        if (q_.coeff(0) == 0) throw std::domain_error("fps::Fraction: q[0] = 0");
    }

    Fraction(Polynomial<P> p) : p_(std::move(p)), q_{1} {}

    const Polynomial<P>& numerator() const { return p_; }
    const Polynomial<P>& denominator() const { return q_; }

    Polynomial<P> take(std::size_t n) const { return mul(p_, inverse(q_, n), n); }

    std::uint32_t operator[](std::uint64_t k) const { return bostan_mori::coefficient(p_.span(), q_.span(), k); }

    // The second product is freed before the third is allocated, which can then reuse its memory.
    friend Fraction operator+(const Fraction& a, const Fraction& b) {
        Polynomial<P> p = a.p_ * b.q_;
        p += b.p_ * a.q_;
        return {std::move(p), a.q_ * b.q_};
    }

    friend Fraction operator-(const Fraction& a, const Fraction& b) {
        Polynomial<P> p = a.p_ * b.q_;
        p -= b.p_ * a.q_;
        return {std::move(p), a.q_ * b.q_};
    }

    friend Fraction operator-(const Fraction& a) { return {-a.p_, a.q_}; }
    friend Fraction operator*(const Fraction& a, const Fraction& b) { return {a.p_ * b.p_, a.q_ * b.q_}; }

    friend Fraction operator/(const Fraction& a, const Fraction& b) {
        if (b.p_.coeff(0) == 0) throw std::domain_error("fps::Fraction: a / b with b[0] = 0");
        return {a.p_ * b.q_, a.q_ * b.p_};
    }

    template <detail::Integer T>
    friend Fraction operator*(T c, const Fraction& a) {
        return {c * a.p_, a.q_};
    }

    template <detail::Integer T>
    friend Fraction operator*(const Fraction& a, T c) {
        return {c * a.p_, a.q_};
    }

private:
    Polynomial<P> p_, q_;
};

// (p / q)' = (p' q - p q') / q^2.
template <std::uint32_t P>
Fraction<P> derivative(const Fraction<P>& a) {
    const Polynomial<P>& p = a.numerator();
    const Polynomial<P>& q = a.denominator();
    Polynomial<P> numerator = derivative(p) * q;
    numerator -= p * derivative(q);
    return {std::move(numerator), square(q)};
}

}  // namespace fps
