// Products of polynomials modulo P = 998244353 (DESIGN.md, Functions: Products; Raw functions).
//
//   fps::mul(f, g, n)                  // f g mod x^n: n coefficients
//   fps::square(f)                     // f^2: 2 f.size() - 1 coefficients (0 if f is empty)
//   fps::product(fs)                   // f_0 f_1 ... (1 if fs is empty), a span or vector of them
//   fps::middle_product(f, g)          // coefficients m - 1 .. n - 1 of f g, n = f.size() >= m = g.size()
//   fps::raw::mul<P>(a, b, out)        // out = a b mod x^out.size(), into the caller's memory
//
// mul, square, middle_product, product and raw::mul run the paths of f * g (fps/polynomial.hpp):
// a shift for a factor with one nonzero term, small::multiply where small::beats_ntt says it is
// faster, else ntt::multiply, which reads the factors and writes the product cut to the output's
// size in the caller's memory. Nothing is copied in or out. The NTT path throws std::length_error
// past 2^25 coefficients (factors cut to n for mul); a short factor next to a longer one runs
// small::multiply and does not throw.
//
// mul cuts both factors to n coefficients and pads the result with zeros past their product.
// small::multiply writes all of the cut product, so on that path the result keeps up to 2 n - 1
// words (fps/words.hpp); a copy is tight.
//
// square(f) costs what f * f costs: the backend has no squaring kernel yet, which would save one of
// the three transforms (backend issue #55).
//
// middle_product(f, g)[i] = sum_{j < m} f[i + j] g[m - 1 - j] for i <= n - m: the coefficients of
// f g whose sums use all of g. With g reversed it is the transposed product, sum_j f[i + j] g[j].
// It throws std::domain_error unless n >= m >= 1. It computes f g mod x^n, then moves the
// n - m + 1 coefficients down: 0.3% to 2.3% over the kernel call on the NTT path
// (frontend/notes.md). A cyclic product of length >= n, with the result at out[0], would need a
// transform of length n, not n + m - 1, and no move (backend issue #55).
//
// product(fs) runs the backend's product_tree::product into the result's memory (a tree with one
// transform of each length per node; small factors 8 at a time in lanes). An empty factor makes
// the product empty, as with f * g. 2^18 linear factors: product_tree/notes.md has the times.
//
// raw::mul<P>(a, b, out): coefficients < P in, canonical out; out must not overlap a or b
// (std::domain_error). It allocates nothing: ntt::multiply runs in this thread's region, kept
// across calls. small::multiply writes all of a b, so an out shorter than that runs a plain loop
// of the frontend instead: 2.1-3.0x the time of small::multiply's whole product at 16 x 16 and
// 24 x 24 on x86 (backend issue #37). Where that matters, call small::multiply into an out of
// a.size() + b.size() - 1 words.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "backend/product_tree/product_tree.hpp"
#include "fps/polynomial.hpp"

namespace fps {

template <std::uint32_t P>
Polynomial<P> mul(const Polynomial<P>& f, const Polynomial<P>& g, std::size_t n) {
    return detail::Products<P>::multiply(f.span(), g.span(), n);
}

template <std::uint32_t P>
Polynomial<P> square(const Polynomial<P>& f) {
    return detail::Products<P>::multiply(f.span(), f.span(), f.size() ? 2 * f.size() - 1 : 0);
}

template <std::uint32_t P>
Polynomial<P> product(std::span<const Polynomial<P>> fs) {
    static_assert(P == ntt::kModulus, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");
    const auto words = [](const Polynomial<P>& f) { return f.span(); };
    Polynomial<P> h = detail::unwritten_polynomial<P>(product_tree::size(fs, words));
    product_tree::product(fs, h.span(), words);
    return h;
}

template <std::uint32_t P>
Polynomial<P> product(const std::vector<Polynomial<P>>& fs) {
    return product(std::span<const Polynomial<P>>(fs));
}

template <std::uint32_t P>
Polynomial<P> middle_product(const Polynomial<P>& f, const Polynomial<P>& g) {
    if (g.size() == 0 || f.size() < g.size())
        throw std::domain_error("fps::middle_product: needs f.size() >= g.size() >= 1");
    return detail::Products<P>::middle(f.span(), g.span());
}

namespace raw {

template <std::uint32_t P>
void mul(std::span<const std::uint32_t> a, std::span<const std::uint32_t> b, std::span<std::uint32_t> out) {
    static_assert(P == ntt::kModulus, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");
    const auto overlaps = [out](std::span<const std::uint32_t> x) {
        return !x.empty() && !out.empty() && x.data() < out.data() + out.size() && out.data() < x.data() + x.size();
    };
    if (overlaps(a) || overlaps(b)) throw std::domain_error("fps::raw::mul: out overlaps a or b");
    detail::Products<P>::multiply_into(a, b, out);
}

}  // namespace raw

}  // namespace fps
