// Number-theoretic transforms and polynomial products modulo P = 998244353 (backend kernel).
// Variants: scalar (every target), intrinsics<S> (every target, on a backend/simd struct) and avx2
// (lc-opt's, asm scheduled for Zen 3); select.hpp picks one per op. Linux or macOS. Design and measurements: notes.md.
//
//   fps::ntt::Convolution conv(n, m);          // factors of n and m coefficients
//   std::copy(f, f + n, conv.a());             // all n coefficients, each < P
//   std::copy(g, g + m, conv.b());
//   const std::uint32_t* c = conv.multiply();  // the n + m - 1 coefficients of f g, in conv.a()
//
//   // The product in the caller's memory: no copy out. a: 64-byte aligned, at least
//   // Convolution::length(n, m) + kPadding words; f may be written before or after constructing.
//   fps::ntt::Convolution into(n, m, a);
//   std::copy(f, f + n, a);
//   std::copy(g, g + m, into.b());
//   into.multiply();                           // a[0, n + m - 1) = f g; the rest of a is scratch
//
//   fps::ntt::Transform t(lg);                 // twiddle tables for the length L = 2^lg
//   t.forward(f);                              // f: L coefficients -> leaf form
//   t.inverse(f);                              // leaf form -> L coefficients
//
// Memory of a Convolution: each thread keeps the largest region a Convolution of it used and hands
// it to the next one, so a short product costs about what multiply() costs (a 32 x 32 product on
// Zen 3: 0.3 us, not 52 us; notes.md). Twiddle tables past 2^16 too: built once per thread for the
// longest length so far. A second Convolution alive on the same thread maps its own region. A
// thread frees what it keeps when it exits.
//
// The caller's a: pick its address so the copy into it is fast. memcpy into an address 0 to 48
// bytes past the source's, modulo 4 KiB, is 4-20x slower on Zen 3 (4K aliasing, issue #38); 64
// bytes or more past is fast. b is placed past a as in a Convolution's own memory.
//
// Leaf form. The transform splits x^L - 1 into L / 8 leaves x^8 - w_v: forward() replaces f by
// its residues, f mod (x^8 - w_v) in words [8v, 8v + 8). w_v = (-1)^v r(v / 2), where r(k) is the
// product of 3^((P - 1) / 2^(j + 2)) over the set bits j of k. Multiplying two leaf forms leaf by
// leaf mod x^8 - w_v, then inverse(), gives the cyclic product mod x^L - 1.
//
// Arrays for Transform: 32-byte aligned, L + kPadding words (the padding is read, never written),
// values < P in and out. No checks on these calls. Lengths: 2^kMinLog .. 2^kMaxLog.
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

#include "backend/ntt/common.hpp"
#include "backend/ntt/memory.hpp"
#include "backend/ntt/select.hpp"

namespace fps::ntt {

// The product of a of n and b of m coefficients, n + m - 1 <= 2^kMaxLog. a and b: length() +
// kPadding words each, 64-byte aligned (the padding is read, never written); a in this thread's
// region or the caller's memory, b in the region. Tables up to 2^detail::kStaticLog are the
// compile-time ones, longer ones this thread's (memory.hpp).
class Convolution {
public:
    Convolution(std::size_t n, std::size_t m) : Convolution(n, m, nullptr) {}

    // a: the caller's, 64-byte aligned, length(n, m) + kPadding words; null for this thread's region.
    Convolution(std::size_t n, std::size_t m, std::uint32_t* a) : n_(n), m_(m), lg_(log_length(n, m)) {
        if (lg_ > kMaxLog) throw std::length_error("fps::ntt::Convolution: product longer than 2^kMaxLog");
        if (reinterpret_cast<std::uintptr_t>(a) % 64)
            throw std::invalid_argument("fps::ntt::Convolution: a is not 64-byte aligned");
        const std::size_t len = length(), span = len + kPadding;
        region_ = detail::take_region(sizeof(std::uint32_t) * 2 * span + 4096);
        auto* start = reinterpret_cast<std::uint32_t*>(region_.start);
        a_ = a ? a : start;
        // b at a + span modulo 4 KiB, as in the region: equal offsets of a and b in different sets.
        b_ = start + (a ? reinterpret_cast<std::uintptr_t>(a_ + span) % 4096 / sizeof(std::uint32_t) : span);
        if (size()) zero_unwritten(a_, n_), zero_unwritten(b_, m_);
    }

    ~Convolution() { detail::give_region(region_); }

    Convolution(const Convolution&) = delete;
    Convolution& operator=(const Convolution&) = delete;

    // The transform length for factors of n and m coefficients: 2^lg >= n + m - 1, lg >= kMinLog.
    static std::size_t length(std::size_t n, std::size_t m) { return std::size_t(1) << log_length(n, m); }

    // The factors: write all n and m coefficients, each < P. Past them, up to length(), the
    // constructor wrote zeros where the product reads.
    std::uint32_t* a() { return a_; }
    std::uint32_t* b() { return b_; }
    std::size_t length() const { return std::size_t(1) << lg_; }
    // The number of coefficients of the product: n + m - 1, or 0 if a factor is empty.
    std::size_t size() const { return n_ && m_ ? n_ + m_ - 1 : 0; }

    // The size() coefficients of a * b, canonical, in a(); b() is destroyed. Single use.
    template <class Variant = SelectedMultiply>
    const std::uint32_t* multiply() {
        if (size() == 0) return a_;
        const auto [roots, inverse_roots] = lg_ <= detail::kStaticLog
            ? std::pair(detail::kStaticTables.words[0], detail::kStaticTables.words[1])
            : detail::thread_tables<Variant>(lg_);
        Variant::multiply(a_, b_, n_, m_, lg_, roots, inverse_roots);
        return a_;
    }

private:
    static int log_length(std::size_t n, std::size_t m) {
        return n && m ? std::max(kMinLog, int(std::bit_width(n + m - 2))) : kMinLog;
    }

    // Zeros where the variants read past a factor's n coefficients: [n, len / 2) if n <= len / 2
    // (they do not read the upper half then), else [n, len).
    void zero_unwritten(std::uint32_t* f, std::size_t n) const {
        const std::size_t len = length();
        std::fill(f + n, f + (2 * n <= len ? len / 2 : len), 0u);
    }

    std::size_t n_, m_;
    int lg_;
    detail::Region region_;
    std::uint32_t *a_, *b_;
};

// Forward and inverse transforms of length 2^lg, kMinLog <= lg <= kMaxLog, with their tables.
class Transform {
public:
    explicit Transform(int lg) : lg_(checked(lg)), tables_(new Line[2 * lines(lg)]) {
        Native::build_table(roots(), length() / 16, false);
        Native::build_table(inverse_roots(), length() / 16, true);
    }

    std::size_t length() const { return std::size_t(1) << lg_; }

    // f -> its leaf form.
    template <class Variant = SelectedForward>
    void forward(std::uint32_t* f) const { Variant::forward(f, lg_, roots()); }

    // Leaf form -> f; exactly undoes forward().
    template <class Variant = SelectedInverse>
    void inverse(std::uint32_t* f) const { Variant::inverse(f, lg_, inverse_roots()); }

private:
    struct alignas(64) Line {
        std::uint32_t words[16];
    };
    static int checked(int lg) {
        if (lg < kMinLog || lg > kMaxLog) throw std::length_error("fps::ntt::Transform: lg out of range");
        return lg;
    }
    static std::size_t lines(int lg) { return (detail::table_words(lg) + 15) / 16; }

    std::uint32_t* roots() const { return tables_[0].words; }
    std::uint32_t* inverse_roots() const { return tables_[lines(lg_)].words; }

    int lg_;
    std::unique_ptr<Line[]> tables_;
};

}  // namespace fps::ntt
