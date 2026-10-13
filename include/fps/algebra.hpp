// Algebra of polynomials modulo P = 998244353 (DESIGN.md, Functions: Algebra).
//
//   fps::gcd(f, g)           // monic gcd: deg + 1 coefficients; gcd(0, 0) = 0 (no coefficients)
//   fps::inverse_mod(f, m)   // std::optional: h with f h = 1 mod m, deg m coefficients; m != 0
//   fps::roots(f)            // the distinct roots in [0, P), increasing; f != 0, deg f < 2^24
//   fps::factor(f)           // monic irreducible factors with multiplicities; f != 0, deg f < P
//
// f and g may end in zeros (size() counts them; nothing here reads past the degree). A broken
// precondition throws std::domain_error, a degree past the kernels' limit std::length_error
// (below). inverse_mod returns std::nullopt when gcd(f, m) != 1; for
// a constant m every residue is 0, so it returns the zero polynomial (no coefficients). factor
// returns the empty list for a constant f and drops the leading coefficient: f = lc(f) times the
// product of the factors to their multiplicities. Factors are ordered by degree, then by their
// coefficients from the constant term up (lexicographic); results do not depend on the random
// choices below.
//
// Paths (factor is plain; backend issue #123 moves lc-opt's factorization):
// - gcd, inverse_mod: one call each into the backend's half-gcd, fps::half_gcd (backend/gcd/gcd.hpp:
//   lc-opt's, #103), O(n log^2 n), on f and g trimmed to their degrees (m too: the kernel needs its
//   top word nonzero). An input of degree 2^25 or more (the transforms' limit; trailing zeros do
//   not count) is first reduced modulo the other, by division, if the other is within the limit:
//   gcd(f, x + 1) works for any f. Both past it, or m past it in inverse_mod: std::length_error.
// - roots: one call into the backend's tangent Graeffe root finder, fps::root_finding::roots
//   (backend/roots/roots.hpp: lc-opt's, #122), O(d log d) for d = deg f, more with many repeated
//   or colliding roots (up to about d^2 / 300). Its transforms of 2 bit_ceil(d + 1) words need
//   deg f <= 2^24 - 1 (trailing zeros do not count; a factor x^z counts): std::length_error past
//   it, before any work. Its workspace, kept per thread, is 32 to 45 d words.
// - factor: Yun's square-free decomposition (it needs deg f < P, so that f' = 0 only for a
//   constant f), then per square-free part s the distinct-degree split gcd(s, x^(P^d) - x) for
//   d = 1, 2, ... up to deg s / 2, then Cantor and Zassenhaus for each degree d: b^((P^d - 1)/2) - 1.
// Powers modulo m are square and multiply on f * g and f % m: 30 squarings for an exponent near P.
// A xorshift generator with a fixed seed draws b, so the run time is the same on every call.
// Costs on the four timed targets (frontend/notes/168.md, 192.md), ns per degree: gcd and
// inverse_mod of two random polynomials of degree 2^12 82-235, 2^16 137-385 (within 1% of the
// kernel's from 2^8 on); roots of a random polynomial of degree 2^12 ROOTS_RANDOM, of 2^12 distinct
// linear factors ROOTS_SPLIT; ms: factor of a random polynomial of degree 256 16-29.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "backend/gcd/gcd.hpp"
#include "backend/roots/roots.hpp"
#include "fps/division.hpp"
#include "fps/modular.hpp"
#include "fps/polynomial.hpp"
#include "fps/products.hpp"
#include "fps/series.hpp"

namespace fps {

namespace detail::algebra {

// The half-gcd's transforms take inputs of at most 2^25 words past trailing zeros.
inline constexpr std::size_t kHalfGcdMax = std::size_t(1) << ntt::kMaxLog;

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

// The kernel's out has as many words as the gcd can have: the shorter trimmed size if both are
// nonzero, else the nonzero one's. One input past the kernel's limit next to one within it is
// first reduced: gcd(f, g) = gcd(f mod g, g), gcd(f, 0) = monic(f) (division has no such limit).
template <std::uint32_t P>
Polynomial<P> gcd(const Polynomial<P>& f, const Polynomial<P>& g) {
    using detail::algebra::kHalfGcdMax;
    const std::size_t nf = std::size_t(f.degree() + 1), ng = std::size_t(g.degree() + 1);
    if (nf > kHalfGcdMax && ng <= kHalfGcdMax) return ng ? gcd(f % g, g) : detail::algebra::monic(f);
    if (ng > kHalfGcdMax && nf <= kHalfGcdMax) return nf ? gcd(f, g % f) : detail::algebra::monic(g);
    Polynomial<P> out = detail::unwritten_polynomial<P>(nf && ng ? std::min(nf, ng) : std::max(nf, ng));
    out.resize(half_gcd::gcd(f.span().first(nf), g.span().first(ng), out.span()));
    return out;
}

// m trimmed for the kernel (it needs m's top word nonzero); f trimmed, so that only its degree
// counts toward the transforms' limit, and reduced mod m first if only f is past it.
template <std::uint32_t P>
std::optional<Polynomial<P>> inverse_mod(const Polynomial<P>& f, const Polynomial<P>& m) {
    const int d = m.degree();
    if (d < 0) throw std::domain_error("fps::inverse_mod: m = 0");
    using detail::algebra::kHalfGcdMax;
    if (std::size_t(f.degree() + 1) > kHalfGcdMax && std::size_t(d) + 1 <= kHalfGcdMax)
        return inverse_mod(f % m, m);
    Polynomial<P> h = detail::unwritten_polynomial<P>(std::size_t(d));
    const std::span<const std::uint32_t> a = f.span().first(std::size_t(f.degree() + 1));
    if (half_gcd::inverse_mod(a, m.span().first(std::size_t(d) + 1), h.span()) < 0) return std::nullopt;
    return h;
}

// out of deg f words, as the kernel asks, trimmed to the roots found (the capacity too: a random
// f has about one root). f = 0 is checked here, so the kernel's std::invalid_argument never
// escapes, and the degree before out is allocated.
template <std::uint32_t P>
std::vector<std::uint32_t> roots(const Polynomial<P>& f) {
    const int d = f.degree();
    if (d < 0) throw std::domain_error("fps::roots: f = 0");
    if (std::size_t(d) > root_finding::kMaxDegree) throw std::length_error("fps::roots: deg f >= 2^24");
    std::vector<std::uint32_t> out(std::size_t(d), 0);
    out.resize(root_finding::roots(f.span().first(std::size_t(d) + 1), out));
    out.shrink_to_fit();
    return out;
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
