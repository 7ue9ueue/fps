// Values at points, interpolation and the Newton basis modulo P = 998244353 (DESIGN.md, Functions:
// Evaluation and interpolation), on any span of points.
//
//   fps::evaluate(f, points)              // f(points[i]) for each i: points.size() values
//   fps::interpolate<P>(points, values)   // the f with f.size() = m and f(points[i]) = values[i]
//   fps::to_newton_basis(f, points)       // c with f = sum_k c[k] prod_(i < k) (x - points[i])
//
// Points and values are canonical, in [0, P); one >= P throws std::domain_error. evaluate takes
// any points, repeats too, and any number of them. interpolate needs as many values as points,
// all points distinct (else std::domain_error: a repeat makes some M'(x_i) zero, which the kernel
// divides by), and returns m = points.size() coefficients (an empty polynomial for m = 0); 2^25
// or more points throw std::length_error, before any work. to_newton_basis returns f.size() = n
// coefficients and reads the first n - 1 points (the last basis polynomial's); fewer throws
// std::domain_error, and n >= 2^25 throws std::length_error, before any work.
//
// Each is one call into the backend's kernel, fps::evaluation::{evaluate, interpolate,
// to_newton_basis} (backend/evaluation/evaluation.hpp: lc-opt's transposed descent on point trees
// with kept transforms, direct O(n m) paths for small sizes; evaluate splits f and the points into
// chunks of 2^24, so it has no size limit). The API checks the preconditions the kernel leaves to
// its caller (words < P) and throws DESIGN.md's error types where the kernel throws
// std::invalid_argument (sizes), then writes the kernel's result into the returned memory: nothing
// is copied. Times and the kernel's memory (one per-thread region, kept across calls):
// backend/evaluation/notes.md; API against kernel: frontend/notes/139.md.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "backend/evaluation/evaluation.hpp"
#include "fps/polynomial.hpp"

namespace fps {

namespace detail::points {

inline constexpr std::size_t kMaxPoints = std::size_t(1) << ntt::kMaxLog;  // 2^25, excluded

inline void check_canonical(std::span<const std::uint32_t> a, std::uint32_t p, const char* what) {
    if (std::ranges::any_of(a, [p](std::uint32_t v) { return v >= p; })) throw std::domain_error(what);
}

}  // namespace detail::points

template <std::uint32_t P>
std::vector<std::uint32_t> evaluate(const Polynomial<P>& f, std::span<const std::uint32_t> points) {
    static_assert(P == ntt::kModulus, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");
    detail::points::check_canonical(points, P, "fps::evaluate: point >= P");
    std::vector<std::uint32_t> values(points.size());
    evaluation::evaluate(f.span(), points, values);
    return values;
}

template <std::uint32_t P>
Polynomial<P> interpolate(std::span<const std::uint32_t> points, std::span<const std::uint32_t> values) {
    static_assert(P == ntt::kModulus, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");
    const std::size_t m = points.size();
    if (values.size() != m) throw std::domain_error("fps::interpolate: points and values differ in size");
    if (m >= detail::points::kMaxPoints) throw std::length_error("fps::interpolate: 2^25 or more points");
    detail::points::check_canonical(points, P, "fps::interpolate: point >= P");
    detail::points::check_canonical(values, P, "fps::interpolate: value >= P");
    Polynomial<P> f = detail::unwritten_polynomial<P>(m);
    evaluation::interpolate(points, values, f.span());  // std::domain_error on a repeated point
    return f;
}

template <std::uint32_t P>
Polynomial<P> to_newton_basis(const Polynomial<P>& f, std::span<const std::uint32_t> points) {
    static_assert(P == ntt::kModulus, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");
    const std::size_t n = f.size();
    if (n >= detail::points::kMaxPoints) throw std::length_error("fps::to_newton_basis: f.size() >= 2^25");
    if (n > 1 && points.size() < n - 1) throw std::domain_error("fps::to_newton_basis: needs f.size() - 1 points");
    const auto read = points.first(n > 0 ? n - 1 : 0);
    detail::points::check_canonical(read, P, "fps::to_newton_basis: point >= P");
    Polynomial<P> c = detail::unwritten_polynomial<P>(n);
    evaluation::to_newton_basis(f.span(), read, c.span());
    return c;
}

}  // namespace fps
