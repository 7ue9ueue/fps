// Products of short polynomials modulo a prime P, 2^12 < P < 2^30 (backend kernel).
//
//   fps::small::multiply<P>(a, b, out);  // out = a b, out.size() = a.size() + b.size() - 1
//   fps::small::beats_ntt(n, m)          // whether multiply() is faster than ntt::Convolution
//
// multiply: coefficients below P in and out. a or b may be empty (then out is too). out must not
// overlap a or b. No allocation (about 9 KiB of stack). No checks. Any sizes work, but the cost is
// O(a.size() b.size()). select.hpp picks the variant per target; both are public:
//   model::multiply<P>(a, n, b, m, out)          the O(n m) reference (model.hpp)
//   Schoolbook<S, R, P>::run(a, n, b, m, out)    n >= m >= 1; blocks of R vectors (schoolbook.hpp)
//
// beats_ntt(n, m), for factors of n and m coefficients: true where multiply() is expected to run
// faster than an ntt::Convolution with the product in the caller's memory, on this target. A cost
// model fitted to measurements (select.hpp). On the timed targets: min(n, m) up to 38-130 next to
// a factor of about 1000 coefficients, 53-175 next to one of 16000 (the higher ends where the
// product just passes a power of two), and n = m up to 25-104. It compares against the NTT on
// main when it was measured; re-measure it when either kernel gets faster (notes.md). Lengths past
// 2^25 count the NTT at twice the time per doubling, so huge products still go to the NTT (which
// throws) and a short factor of any length to multiply().
// Design and measurements: notes.md.
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "backend/small/model.hpp"
#include "backend/small/schoolbook.hpp"
#include "backend/small/select.hpp"

namespace fps::small {

template <u32 P>
void multiply(std::span<const u32> a, std::span<const u32> b, std::span<u32> out) {
    if (a.empty() || b.empty()) return;
    if (a.size() < b.size()) std::swap(a, b);
    Schoolbook<Isa, kVectors, P>::run(a.data(), a.size(), b.data(), b.size(), out.data());
}

inline bool beats_ntt(std::size_t n, std::size_t m) {
    if (n == 0 || m == 0) return true;
    const int lg = std::max(6, int(std::bit_width(n + m - 2)));
    const double x = double(n), y = double(m), s = double(std::min(n, m));
    const double small = kCosts.term * x * y + kCosts.word * (x + y) + kCosts.call + kCosts.square * s * s;
    const double ntt = lg <= 25 ? kCosts.ntt[lg - 6] : std::ldexp(kCosts.ntt[19], lg - 25);
    return small < ntt;
}

}  // namespace fps::small
