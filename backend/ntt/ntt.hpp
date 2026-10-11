// Number-theoretic transforms and polynomial products modulo P = 998244353 (backend kernel).
// Variants: scalar (every target), intrinsics<S> (every target, on a backend/simd struct) and avx2
// (lc-opt's, asm scheduled for Zen 3); select.hpp picks one per op. Linux or macOS. Design and measurements: notes.md.
//
//   fps::ntt::Convolution conv(n, m);          // factors of n and m coefficients
//   std::copy(f, f + n, conv.a());             // coefficients < P
//   std::copy(g, g + m, conv.b());
//   const std::uint32_t* c = conv.multiply();  // the n + m - 1 coefficients of f g
//
//   fps::ntt::Transform t(lg);                 // twiddle tables for the length L = 2^lg
//   t.forward(f);                              // f: L coefficients -> leaf form
//   t.inverse(f);                              // leaf form -> L coefficients
//
// Leaf form. The transform splits x^L - 1 into L / 8 leaves x^8 - w_v: forward() replaces f by
// its residues, f mod (x^8 - w_v) in words [8v, 8v + 8). w_v = (-1)^v r(v / 2), where r(k) is the
// product of 3^((P - 1) / 2^(j + 2)) over the set bits j of k. Multiplying two leaf forms leaf by
// leaf mod x^8 - w_v, then inverse(), gives the cyclic product mod x^L - 1.
//
// Arrays for Transform: 32-byte aligned, L + kPadding words (the padding is read, never written),
// values < P in and out. No checks on these calls. Lengths: 2^kMinLog .. 2^kMaxLog.
#pragma once

#include <sys/mman.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>

#include "backend/ntt/common.hpp"
#include "backend/ntt/select.hpp"

namespace fps::ntt {

// The product of a of n and b of m coefficients, n + m - 1 <= 2^kMaxLog. Owns its memory: a, b
// and the twiddle tables in one mapping, 2 MiB aligned, in huge pages where the system allows.
// Lengths up to 2^detail::kStaticLog use the compile-time tables instead.
class Convolution {
public:
    Convolution(std::size_t n, std::size_t m) : n_(n), m_(m), lg_(log_length(n, m)) {
        if (lg_ > kMaxLog) throw std::length_error("fps::ntt::Convolution: product longer than 2^kMaxLog");
        const std::size_t len = length(), words = 2 * (len + kPadding) + 2 * owned_table_words(lg_);
        constexpr std::size_t kHuge = std::size_t(1) << 21;
        bytes_ = (words * sizeof(std::uint32_t) + kHuge - 1) / kHuge * kHuge + kHuge;
        region_ = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (region_ == MAP_FAILED) throw std::bad_alloc();
        const std::uintptr_t aligned = (reinterpret_cast<std::uintptr_t>(region_) + kHuge - 1) & ~(kHuge - 1);
        a_ = reinterpret_cast<std::uint32_t*>(aligned);
#ifdef MADV_HUGEPAGE
        ::madvise(a_, bytes_ - kHuge, MADV_HUGEPAGE);
#endif
        b_ = a_ + len + kPadding;  // a different cache set from a at equal offsets
        if (owned_table_words(lg_)) tables_ = b_ + len + kPadding;
    }

    ~Convolution() { ::munmap(region_, bytes_); }

    Convolution(const Convolution&) = delete;
    Convolution& operator=(const Convolution&) = delete;

    // The factors, zero-filled: write n and m coefficients < P. length() words each.
    std::uint32_t* a() { return a_; }
    std::uint32_t* b() { return b_; }
    // The transform length 2^lg >= size(), lg >= kMinLog.
    std::size_t length() const { return std::size_t(1) << lg_; }
    // The number of coefficients of the product: n + m - 1, or 0 if a factor is empty.
    std::size_t size() const { return n_ && m_ ? n_ + m_ - 1 : 0; }

    // The size() coefficients of a * b, canonical, in a(); b() is destroyed. Single use: a()
    // and b() must hold the factors with zeros up to length().
    template <class Variant = SelectedMultiply>
    const std::uint32_t* multiply() {
        if (size() == 0) return a_;
        const std::uint32_t *roots = detail::kStaticTables.words[0], *inverse_roots = detail::kStaticTables.words[1];
        if (tables_) {
            roots = tables_, inverse_roots = tables_ + owned_table_words(lg_);
            Variant::build_table(tables_, length() / 16, false);
            Variant::build_table(tables_ + owned_table_words(lg_), length() / 16, true);
        }
        Variant::multiply(a_, b_, n_, m_, lg_, roots, inverse_roots);
        return a_;
    }

private:
    static int log_length(std::size_t n, std::size_t m) {
        return n && m ? std::max(kMinLog, int(std::bit_width(n + m - 2))) : kMinLog;
    }
    static std::size_t owned_table_words(int lg) { return lg > detail::kStaticLog ? detail::table_words(lg) : 0; }

    std::size_t n_, m_;
    int lg_;
    void* region_;
    std::size_t bytes_;
    std::uint32_t *a_, *b_, *tables_ = nullptr;  // tables_: forward, then inverse; null if static
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
