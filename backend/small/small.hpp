// Products of short polynomials modulo a prime P, 2^12 < P < 2^30 (backend kernel).
//
//   fps::small::multiply<P>(a, b, out);  // out = a b, out.size() = a.size() + b.size() - 1
//   fps::small::beats_ntt(n, m)          // whether multiply() is faster than ntt::Convolution
//
// multiply: coefficients below P in and out. a or b may be empty (then out is too). out must not
// overlap a or b. No allocation (about 9 KiB of stack). No checks. Any sizes work, but the cost is
// O(a.size() b.size()). The variant is a tuned choice per target and shape class ("small.variant",
// select.hpp): one table lookup per call, then a chain of compares on the variant's index. The
// variants are public, as the fast path for callers that pick their own:
//   model::multiply<P>(a, n, b, m, out)                the O(n m) reference (model.hpp)
//   Schoolbook<S, R, P, U, Solo, T>::run(a, n, b, m, out)  n >= m >= 1 (schoolbook.hpp)
//   run_variant<P>(v, a, n, b, m, out)                 candidate v of select.hpp's list, n >= m >= 1
//
// beats_ntt(n, m), for factors of n and m coefficients: true where multiply() is expected to run
// faster than ntt::multiply into the caller's memory, on this target: a cost model of multiply()
// against the NTT's time at the product's length N = n + m - 1 (ntt_ns: measured at 2^lg and four
// lengths in each (2^(lg-1), 2^lg), linear between; the NTT runs ragged and truncated plans there,
// issue #212), two acquire loads per call. Both are compiled defaults measured on the target's
// machine by tools/tune.py; the NTT times are tuned choices ("small.ntt", "small.ragged",
// tune.hpp) that tune::run() measures on this machine and tune::save() / load() keep, after
// "small.variant". Rerun tools/tune.py small when either kernel gets faster. Where it picks
// multiply() on the timed targets: notes.md (Cutoff against the NTT). Lengths past 2^25 count the
// NTT at twice the time per doubling, so huge products still go to the NTT (which throws) and a
// short factor of any length to multiply().
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

template <class V, u32 P>
using SchoolbookOf = Schoolbook<typename V::Isa, V::vectors, P, V::unroll, V::solo, V::tiny>;

// A chain of compares on v, inlined into the caller: a lambda with captures here cost 6 ns per call
// on Zen 3 (its closure went through the stack, and the call was not inlined; notes.md).
template <u32 P, class... V>
[[gnu::always_inline]] inline void run_candidate(Candidates<V...>, int v, const u32* a, std::size_t n, const u32* b,
                                                 std::size_t m, u32* out) {
    int i = 0;
    (void)((i++ == v && (SchoolbookOf<V, P>::run(a, n, b, m, out), true)) || ...);
}

template <u32 P>
[[gnu::always_inline]] inline void run_variant(int v, const u32* a, std::size_t n, const u32* b, std::size_t m,
                                               u32* out) {
    run_candidate<P>(Selected{}, v, a, n, b, m, out);
}

template <u32 P>
void multiply(std::span<const u32> a, std::span<const u32> b, std::span<u32> out) {
    if (a.empty() || b.empty()) return;
    if (a.size() < b.size()) std::swap(a, b);
    const std::size_t n = a.size(), m = b.size();
    run_variant<P>(variants.get()[shape_class(n, m)], a.data(), n, b.data(), m, out.data());
}

// The model's ns for multiply() on factors of n and m coefficients (select.hpp).
inline double model_ns(const Model& model, std::size_t n, std::size_t m) {
    const double x = double(n), y = double(m), s = double(std::min(n, m));
    return model.term * x * y + model.word * (x + y) + model.call + model.square * s * s;
}

// The NTT's time for a product of N >= 1 words, in the model's ns (select.hpp): in (2^(lg-1), 2^lg],
// linear between the measured lengths ntt_point(lg, j), with 2^(lg-1) + 1 taken as 2^(lg-1); up to
// 2^5 that of 33; past 2^25 twice per doubling.
inline double ntt_ns(std::size_t N, const std::array<double, 20>& top,
                     const std::array<double, 20 * kRaggedPoints>& low) {
    const int lg = std::max(6, int(std::bit_width(N - 1)));
    if (lg > 25) return std::ldexp(top[19], lg - 25);
    const std::size_t half = std::size_t(1) << (lg - 1);
    const double u = N > half ? double(N - half) * kRaggedPoints / double(half) : 0;  // in [0, 4]
    const int j = std::min(int(u), kRaggedPoints - 1);
    const double lo = low[kRaggedPoints * (lg - 6) + j];
    const double hi = j + 1 < kRaggedPoints ? low[kRaggedPoints * (lg - 6) + j + 1] : top[lg - 6];
    return lo + (u - j) * (hi - lo);
}
inline double ntt_ns(std::size_t N) { return ntt_ns(N, ntt_times.get(), ragged_times.get()); }

inline bool beats_ntt(std::size_t n, std::size_t m) {
    if (n == 0 || m == 0) return true;
    return model_ns(kModel, n, m) < ntt_ns(n + m - 1);
}

}  // namespace fps::small
