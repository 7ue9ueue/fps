// The NTT variant per op and transform length on this target, a tuned choice (backend/tune): one of
// the target's candidates (Selected) per op and lg.
//
//   Op: forward, inverse (Transform), multiply (Convolution::multiply, factors in place), product
//   (multiply(f, g, out), factors from spans and the product into out; also square(f, out) and
//   middle_product(f, g, out), which have no row of their own)
//   variants.get()[op][lg - kMinLog]   the candidate: an index into Selected
//   kDefaults                          compiled defaults: tools/tune.py ntt (tune.cpp, tuned.hpp)
//   variant::tuned                     the selected variant: one table read per call (SelectedForward ...)
//
// Variants: scalar (every target), intrinsics<S> (every simd struct S the target has), avx2
// (__AVX2__). backend/ntt/tune.hpp measures the table on this machine. -DFPS_NTT_FORWARD=<variant>
// (and _INVERSE, _MULTIPLY, the last for both products) forces one, e.g.
// -DFPS_NTT_FORWARD='intrinsics<fps::simd::Avx512>'. Measurements: notes.md.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "backend/ntt/intrinsics.hpp"
#include "backend/ntt/scalar.hpp"
#include "backend/ntt/tuned.hpp"
#include "backend/simd/select.hpp"
#include "backend/tune/tune.hpp"
#ifdef __AVX2__
#include "backend/ntt/avx2.hpp"
#endif

namespace fps::ntt {

// Native: the variant that builds the tables of ntt::Transform (every variant builds the same).
#if defined(__AVX2__) && !(defined(__AVX512F__) && defined(__AVX512BW__))
using Native = variant::avx2;
#else
using Native = variant::intrinsics<fps::simd::Native>;
#endif

template <class... V>
struct Candidates {};

// Per target, the variants that are fastest at some op and length on our machines (notes.md). With
// AVX-512: the asm, intrinsics on ymm (32 registers) and on zmm. With AVX2 alone the asm: on Zen 3
// intrinsics<Avx2> won only forward at 2^8 (2.7%), and its code in the binary, even unused, made
// f * g at 2^6 22-31% slower there (code placement). Elsewhere intrinsics on the widest struct.
// A target with one candidate reads no table.
#if defined(__AVX512F__) && defined(__AVX512BW__)  // avx512, zen5
using Selected = Candidates<variant::avx2, variant::intrinsics<simd::Avx512Vl>, variant::intrinsics<simd::Avx512>>;
#elif defined(__AVX2__)
using Selected = Candidates<variant::avx2>;
#else
using Selected = Candidates<variant::intrinsics<simd::Native>>;
#endif

inline constexpr int kCandidates = []<class... V>(Candidates<V...>) { return int(sizeof...(V)); }(Selected{});

enum Op { kForward, kInverse, kMultiply, kProduct };
inline constexpr int kOps = 4;
inline constexpr int kLengths = kMaxLog - kMinLog + 1;

using Choices = std::array<std::array<std::uint8_t, kLengths>, kOps>;

// "avx2" or "intrinsics_<struct>" (the bench's names), and a hash of the list that tuned.hpp
// records, so a table measured for another list does not compile.
template <class V>
struct Name;
#ifdef __AVX2__
template <>
struct Name<variant::avx2> {
    static constexpr std::string_view prefix = "avx2", isa = "";
};
#endif
template <class S>
struct Name<variant::intrinsics<S>> {
    static constexpr std::string_view prefix = "intrinsics_", isa = S::name;
};
template <class V>
std::string variant_name() {
    return std::string(Name<V>::prefix) + std::string(Name<V>::isa);
}
template <class... V>
constexpr std::uint32_t signature(Candidates<V...>) {
    tune::Hash h;
    (h.add(Name<V>::prefix).add(Name<V>::isa).add(";"), ...);
    return h.value >> 1;  // fits an int
}
inline constexpr std::uint32_t kSignature = signature(Selected{});

constexpr bool valid(const Choices& choices) {
    return std::ranges::all_of(choices, [](const auto& row) {
        return std::ranges::all_of(row, [](std::uint8_t v) { return v < kCandidates; });
    });
}

inline constexpr int kTuned = int(tune::kTarget);
static_assert(tuned::signature[kTuned] == kSignature, "tuned.hpp was measured for other candidates: run tools/tune.py ntt");
inline constexpr Choices kDefaults = [] {
    Choices choices{};
    for (int i = 0; i < kLengths; ++i) {
        choices[kForward][i] = std::uint8_t(tuned::forward[kTuned][i]);
        choices[kInverse][i] = std::uint8_t(tuned::inverse[kTuned][i]);
        choices[kMultiply][i] = std::uint8_t(tuned::multiply[kTuned][i]);
        choices[kProduct][i] = std::uint8_t(tuned::product[kTuned][i]);
    }
    return choices;
}();
static_assert(valid(kDefaults));

inline constinit tune::Table<kDefaults> variants;

// The candidate for op at 2^lg: 0 without a table read where there is one candidate.
[[gnu::always_inline]] constexpr int choice(Op op, int lg) {
    if constexpr (kCandidates == 1) return 0;
    else return variants.get()[op][lg - kMinLog];
}

namespace detail {

// Candidate v of the list for each op: a chain of compares on v, inlined into the caller. A lambda
// with captures in this place cost small's products 6 ns per call on Zen 3 (backend/tune/notes.md).
template <class... V>
[[gnu::always_inline]] inline void run_forward(Candidates<V...>, int v, std::uint32_t* f, int lg, const std::uint32_t* r) {
    int i = 0;
    (void)((i++ == v && (V::forward(f, lg, r), true)) || ...);
}
template <class... V>
[[gnu::always_inline]] inline void run_forward_upper(Candidates<V...>, int v, std::uint32_t* f, int lg, const std::uint32_t* r) {
    int i = 0;
    (void)((i++ == v && (V::forward_upper(f, lg, r), true)) || ...);
}
template <class... V>
[[gnu::always_inline]] inline void run_inverse(Candidates<V...>, int v, std::uint32_t* f, int lg, const std::uint32_t* ir) {
    int i = 0;
    (void)((i++ == v && (V::inverse(f, lg, ir), true)) || ...);
}
template <class... V>
[[gnu::always_inline]] inline void run_multiply(Candidates<V...>, int v, std::uint32_t* a, std::uint32_t* b,
                                                std::size_t n, std::size_t m, int lg, const std::uint32_t* r,
                                                const std::uint32_t* ir) {
    int i = 0;
    (void)((i++ == v && (V::multiply(a, b, n, m, lg, r, ir), true)) || ...);
}
template <class... V>
[[gnu::always_inline]] inline void run_product(Candidates<V...>, int v, const Product& p, const std::uint32_t* r,
                                               const std::uint32_t* ir) {
    int i = 0;
    (void)((i++ == v && (V::multiply(p, r, ir), true)) || ...);
}
template <class... V>
[[gnu::always_inline]] inline void run_square(Candidates<V...>, int v, const Product& p, const std::uint32_t* r,
                                              const std::uint32_t* ir) {
    int i = 0;
    (void)((i++ == v && (V::square(p, r, ir), true)) || ...);
}

// Transform's leaf-form ops (ntt.hpp): the values form by the forward's choice, the inverse
// from other memory (or of the values form) by the inverse's.
template <class... V>
[[gnu::always_inline]] inline void run_forward_values(Candidates<V...>, int v, std::uint32_t* f, int lg,
                                                      const std::uint32_t* r, const std::uint32_t* rows) {
    int i = 0;
    (void)((i++ == v && (V::forward_values(f, lg, r, rows), true)) || ...);
}
template <class... V>
[[gnu::always_inline]] inline void run_inverse_from(Candidates<V...>, int v, const std::uint32_t* from, std::uint32_t* f,
                                                    std::uint32_t* out, std::size_t count, int lg,
                                                    const std::uint32_t* ir, const std::uint32_t* rows) {
    int i = 0;
    (void)((i++ == v && (V::inverse(from, f, out, count, lg, ir, rows), true)) || ...);
}

}  // namespace detail

namespace variant {

// The table's candidate per op and length.
struct tuned {
    static void build_table(std::uint32_t* t, std::size_t count, bool inverse) { Native::build_table(t, count, inverse); }

    static void forward(std::uint32_t* f, int lg, const std::uint32_t* roots) {
        detail::run_forward(Selected{}, choice(kForward, lg), f, lg, roots);
    }

    // The forward's candidate at the length it runs, 2^lg (its table is the one of 2^(lg + 1)).
    static void forward_upper(std::uint32_t* f, int lg, const std::uint32_t* roots) {
        detail::run_forward_upper(Selected{}, choice(kForward, lg), f, lg, roots);
    }

    static void inverse(std::uint32_t* f, int lg, const std::uint32_t* inverse_roots) {
        detail::run_inverse(Selected{}, choice(kInverse, lg), f, lg, inverse_roots);
    }

    static void multiply(std::uint32_t* a, std::uint32_t* b, std::size_t n, std::size_t m, int lg,
                         const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
        detail::run_multiply(Selected{}, choice(kMultiply, lg), a, b, n, m, lg, roots, inverse_roots);
    }

    static void multiply(const detail::Product& p, const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
        detail::run_product(Selected{}, choice(kProduct, p.lg), p, roots, inverse_roots);
    }

    // The product row's candidate (ntt::square).
    static void square(const detail::Product& p, const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
        detail::run_square(Selected{}, choice(kProduct, p.lg), p, roots, inverse_roots);
    }

    // Transform's leaf-form ops: leaf products have no row in the table (Native).
    static void leaf_products(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t* out, int lg,
                              const std::uint32_t* roots, bool accumulate) {
        Native::leaf_products(a, b, out, lg, roots, accumulate);
    }

    static void forward_values(std::uint32_t* f, int lg, const std::uint32_t* roots, const std::uint32_t* rows) {
        detail::run_forward_values(Selected{}, choice(kForward, lg), f, lg, roots, rows);
    }

    static void inverse(const std::uint32_t* from, std::uint32_t* f, std::uint32_t* out, std::size_t count, int lg,
                        const std::uint32_t* inverse_roots, const std::uint32_t* rows) {
        detail::run_inverse_from(Selected{}, choice(kInverse, lg), from, f, out, count, lg, inverse_roots, rows);
    }
};

}  // namespace variant

#ifdef FPS_NTT_FORWARD
using SelectedForward = variant::FPS_NTT_FORWARD;
#else
using SelectedForward = variant::tuned;
#endif

#ifdef FPS_NTT_INVERSE
using SelectedInverse = variant::FPS_NTT_INVERSE;
#else
using SelectedInverse = variant::tuned;
#endif

#ifdef FPS_NTT_MULTIPLY
using SelectedMultiply = variant::FPS_NTT_MULTIPLY;
#else
using SelectedMultiply = variant::tuned;
#endif

}  // namespace fps::ntt
