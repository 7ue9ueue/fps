// Algebra of polynomials modulo P = 998244353 (DESIGN.md, Functions: Algebra).
//
//   fps::gcd(f, g)           // monic gcd: deg + 1 coefficients; gcd(0, 0) = 0 (no coefficients)
//   fps::inverse_mod(f, m)   // std::optional: h with f h = 1 mod m, deg m coefficients; m != 0
//   fps::roots(f)            // the distinct roots in [0, P), increasing; f != 0
//   fps::factor(f)           // monic irreducible factors with multiplicities; f != 0, deg f < P
//
// f and g may end in zeros (size() counts them; nothing here reads past the degree). A broken
// precondition throws std::domain_error. inverse_mod returns std::nullopt when gcd(f, m) != 1; for
// a constant m every residue is 0, so it returns the zero polynomial (no coefficients). factor
// returns the empty list for a constant f and drops the leading coefficient: f = lc(f) times the
// product of the factors to their multiplicities. Factors are ordered by degree, then by their
// coefficients from the constant term up (lexicographic); results do not depend on the random
// choices below.
//
// Paths, all plain (backend issue #103 moves lc-opt's half-gcd for gcd and inverse_mod; #122 and
// #123 its root finding and factorization):
// - gcd: Euclid on f % g (fps/division.hpp), O(n^2) for degree n: up to min(deg f, deg g) + 1
//   remainders; for random inputs every quotient after the first has degree 1, a division by the
//   recurrence and a short product.
// - inverse_mod: extended Euclid on divmod, carrying only the cofactor of f.
// - roots: f = x^z u (0 is a root for z > 0); the product r of (x - a) over the nonzero roots a is
//   gcd(u, x^(P-1) - 1); r splits by Cantor and Zassenhaus: gcd(r, b^((P-1)/2) - 1) for random b.
// - factor: Yun's square-free decomposition (it needs deg f < P, so that f' = 0 only for a
//   constant f), then per square-free part s the distinct-degree split gcd(s, x^(P^d) - x) for
//   d = 1, 2, ... up to deg s / 2, then Cantor and Zassenhaus for each degree d: b^((P^d - 1)/2) - 1.
// Powers modulo m are square and multiply on f * g and f % m: 30 squarings for an exponent near P.
// A xorshift generator with a fixed seed draws b, so the run time is the same on every call.
// Costs on the four timed targets (frontend/notes/25.md), ms: gcd and inverse_mod of two random
// polynomials of degree 2^10 2.5-4.0, 2^12 38-60 (2.2-3.6 ns per n^2); roots of 2^10 distinct linear
// factors 20-31; factor of a random polynomial of degree 64 1.1-1.5, 256 34-56.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fps/division.hpp"
#include "fps/modular.hpp"
#include "fps/polynomial.hpp"
#include "fps/products.hpp"
#include "fps/series.hpp"

namespace fps {

namespace detail::algebra {

// f without its trailing zeros (resize to a smaller size keeps the memory, no copy).
template <std::uint32_t P>
Polynomial<P> trimmed(Polynomial<P> f) {
    f.resize(std::size_t(f.degree() + 1));
    return f;
}

// f trimmed and divided by its leading coefficient; 0 stays 0.
template <std::uint32_t P>
Polynomial<P> monic(Polynomial<P> f) {
    f = trimmed(std::move(f));
    if (f.size() > 0 && f[f.size() - 1] != 1) f *= mod_inverse<P>(f[f.size() - 1]);
    return f;
}

// a^e mod m, m monic of degree >= 1: deg m coefficients.
template <std::uint32_t P>
Polynomial<P> pow_mod(Polynomial<P> a, std::uint64_t e, const Polynomial<P>& m) {
    Polynomial<P> r = Polynomial<P>{1} % m;
    for (a = a % m; e; e >>= 1) {
        if (e & 1) r = (r * a) % m;
        if (e > 1) a = square(a) % m;
    }
    return r;
}

// Fixed-seed xorshift for the random splittings.
class Random {
public:
    std::uint32_t next(std::uint32_t bound) {
        state_ ^= state_ << 13, state_ ^= state_ >> 7, state_ ^= state_ << 17;
        return std::uint32_t(state_ % bound);
    }

private:
    std::uint64_t state_ = 0x9e3779b97f4a7c15;
};

template <std::uint32_t P>
using Factors = std::vector<std::pair<Polynomial<P>, int>>;

// The factors of f, monic, square-free, a product of irreducibles of degree d, into out with
// multiplicity e. b^((P^d - 1)/2) = (b b^P ... b^(P^(d-1)))^((P-1)/2) is +-1 modulo each factor
// for b coprime to f, each sign with probability about 1/2.
template <std::uint32_t P>
void split(const Polynomial<P>& f, std::size_t d, int e, Factors<P>& out, Random& random) {
    const std::size_t n = std::size_t(f.degree());
    if (n == d) return void(out.emplace_back(f, e));
    for (;;) {
        Polynomial<P> b(n);
        for (std::uint32_t& c : b.span()) c = random.next(P);
        Polynomial<P> t = b, norm = b;
        for (std::size_t j = 1; j < d; ++j) t = pow_mod(t, P, f), norm = (norm * t) % f;
        const Polynomial<P> g = gcd(f, pow_mod(norm, (P - 1) / 2, f) - 1);
        const std::size_t k = std::size_t(g.degree());
        if (k == 0 || k == n) continue;
        split(g, d, e, out, random);
        split(f / g, d, e, out, random);
        return;
    }
}

// The irreducible factors of s, monic and square-free, into out with multiplicity e.
template <std::uint32_t P>
void factor_square_free(Polynomial<P> s, int e, Factors<P>& out, Random& random) {
    const Polynomial<P> x = Polynomial<P>::x();
    Polynomial<P> h = x % s;  // x^(P^d) mod s
    for (std::size_t d = 1; s.degree() >= int(2 * d); ++d) {
        h = pow_mod(std::move(h), P, s);
        const Polynomial<P> part = gcd(s, h - x);  // the factors of degree d
        if (part.degree() == 0) continue;
        split(part, d, e, out, random);
        s = trimmed(s / part);
        h = h % s;
    }
    if (s.degree() > 0) out.emplace_back(std::move(s), e);
}

}  // namespace detail::algebra

template <std::uint32_t P>
Polynomial<P> gcd(const Polynomial<P>& f, const Polynomial<P>& g) {
    using namespace detail::algebra;
    Polynomial<P> a = trimmed(f), b = trimmed(g);
    while (b.size() > 0) a = std::exchange(b, trimmed(a % b));
    return monic(std::move(a));
}

// Invariant: r0 = s0 f and r1 = s1 f mod m, deg s0 < deg s1 < deg m.
template <std::uint32_t P>
std::optional<Polynomial<P>> inverse_mod(const Polynomial<P>& f, const Polynomial<P>& m) {
    using namespace detail::algebra;
    const int d = m.degree();
    if (d < 0) throw std::domain_error("fps::inverse_mod: m = 0");
    if (d == 0) return Polynomial<P>();
    Polynomial<P> r0 = trimmed(m), r1 = trimmed(f % r0), s0, s1{1};
    while (r1.size() > 0) {
        auto [q, r] = divmod(r0, r1);
        r0 = std::exchange(r1, trimmed(std::move(r)));
        s0 = std::exchange(s1, trimmed(s0 - q * s1));
    }
    if (r0.size() != 1) return std::nullopt;
    return s0.truncate(std::size_t(d)) * mod_inverse<P>(r0[0]);
}

template <std::uint32_t P>
std::vector<std::uint32_t> roots(const Polynomial<P>& f) {
    using namespace detail::algebra;
    if (f.degree() < 0) throw std::domain_error("fps::roots: f = 0");
    Polynomial<P> u = monic(f);
    std::vector<std::uint32_t> found;
    const std::size_t z = std::size_t(std::ranges::find_if(u.span(), [](std::uint32_t c) { return c != 0; }) -
                                      u.span().begin());
    if (z > 0) {
        found.push_back(0);
        u = u / Polynomial<P>::x(z);
    }
    if (u.degree() > 0) {
        const Polynomial<P> r = gcd(u, pow_mod(Polynomial<P>::x(), P - 1, u) - 1);
        Factors<P> linear;
        Random random;
        if (r.degree() > 0) split(r, 1, 1, linear, random);
        for (const auto& [l, e] : linear) found.push_back(l[0] == 0 ? 0 : P - l[0]);
    }
    std::ranges::sort(found);
    return found;
}

template <std::uint32_t P>
std::vector<std::pair<Polynomial<P>, int>> factor(const Polynomial<P>& f) {
    using namespace detail::algebra;
    const int n = f.degree();
    if (n < 0) throw std::domain_error("fps::factor: f = 0");
    if (std::uint64_t(n) >= P) throw std::domain_error("fps::factor: deg f >= P");
    Factors<P> out;
    Random random;
    // Yun: g = prod s_i^i, with w = prod_{j >= i} s_j and c = prod_{j > i} s_j^(j - i) at step i.
    const Polynomial<P> g = monic(f);
    Polynomial<P> c = gcd(g, derivative(g)), w = trimmed(g / c);
    for (int i = 1; w.degree() > 0; ++i) {
        Polynomial<P> y = gcd(w, c);
        factor_square_free(trimmed(w / y), i, out, random);
        c = trimmed(c / y), w = std::move(y);
    }
    std::ranges::sort(out, [](const auto& a, const auto& b) {
        if (a.first.size() != b.first.size()) return a.first.size() < b.first.size();
        return std::ranges::lexicographical_compare(a.first.span(), b.first.span());
    });
    return out;
}

}  // namespace fps
