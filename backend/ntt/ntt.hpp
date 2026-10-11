// Number-theoretic transforms and polynomial products modulo P = 998244353 (backend kernel).
// Variants: scalar (every target), intrinsics<S> (every target, on a backend/simd struct) and avx2
// (lc-opt's, asm scheduled for Zen 3); select.hpp picks one per op. Linux or macOS. Design and measurements: notes.md.
//
//   fps::ntt::multiply(f, g, out);             // out = f g mod x^out.size(); spans of words < P
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
// multiply(f, g, out): the product from and into the caller's memory, with nothing copied: the
// first transform level reads f and g, the last writes out, the levels between run in this
// thread's region (below). out gets f g cut to out.size() words, zeros past its last coefficient;
// only f and g mod x^out.size() are read, so a cut product may use a shorter transform. Spans of
// any alignment; out may overlap f or g. Throws std::length_error, before writing out, if the
// product of f and g cut to out.size() words is longer than 2^kMaxLog. Same binary, 2^9 to 2^22
// words (notes.md, issue #38 round 2): against a Convolution with f and g copied in and the product
// copied out, 0.93-1.00 of the time on Zen 3, Emerald Rapids, Zen 5 and Neoverse V2; against f
// copied into the caller's a of Convolution(n, m, a), 0.98-1.04, with out of n + m - 1 words, not
// length + kPadding; against Convolution::multiply() alone, 1.00-1.12. At 2^6-2^8 up to 10% slower
// than the copying forms. out of half the transform length (f g mod x^n for factors of n words):
// 0.96-1.00 of the time of the whole product, 0.89-0.97 at 2^6-2^8.
//
// Memory of a product: each thread keeps the largest region a product of it used and hands it to
// the next one. Constructing, multiplying and destroying a 32 x 32 Convolution takes 110 ns on Zen
// 3, against 106 ns for multiply() alone and 56 us when each Convolution mapped fresh memory
// (notes.md). Twiddle tables past 2^16 too: one pair per thread, built for the longest length so
// far. A second Convolution alive on the same thread maps its own region. A thread keeps 8 bytes
// per transform word of its longest product, and 1 byte per word of tables, until it exits.
//
// The caller's a: b is placed as far past it, modulo 4 KiB, as in a Convolution's own memory, so
// multiply() runs at least as fast. A memcpy into a ran at full speed on all four timing machines
// with a 64 or 128 bytes before its source, or 256 bytes or more past it, a multiple of 64 bytes,
// modulo 4 KiB; other offsets were up to 1.7x slower (notes.md).
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
#include <span>
#include <stdexcept>
#include <utility>

#include "backend/ntt/common.hpp"
#include "backend/ntt/memory.hpp"
#include "backend/ntt/select.hpp"

namespace fps::ntt {

namespace detail {

// log2 of the transform length for factors of n and m coefficients: 2^lg >= n + m - 1, lg >= kMinLog.
inline int log_length(std::size_t n, std::size_t m) {
    return n && m ? std::max(kMinLog, int(std::bit_width(n + m - 2))) : kMinLog;
}

// Bytes of a region for a and b at length 2^lg: length + kPadding words each, and 4 KiB to place
// a (multiply) or b (Convolution with the caller's a).
inline std::size_t region_bytes(int lg) { return sizeof(std::uint32_t) * 2 * ((std::size_t(1) << lg) + kPadding) + 4096; }

// The offset of a in a region (bytes, a multiple of 64 below 4096) for a product from f and g into
// out, with b span bytes after a: a's page offset at least 256 bytes from f's and out's, b's from
// g's. A pass between arrays 0 to 192 bytes apart modulo 4 KiB ran up to 1.7x slower (4K aliasing,
// notes.md, issue #38 rounds 1 and 2).
inline std::size_t place(const void* f, const void* g, const void* out, std::size_t span) {
    const auto offset = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 4096; };
    const auto apart = [](std::size_t x, std::size_t y) { return (x - y + 256) % 4096 >= 512; };
    for (std::size_t a = 0; a < 4096; a += 64)
        if (apart(a, offset(f)) && apart(a, offset(out)) && apart((a + span) % 4096, offset(g))) return a;
    return 0;  // not reached: the three conditions exclude at most 3 x 512 of the 4096 bytes
}

// The forward and inverse tables for length 2^lg: the compile-time ones up to 2^kStaticLog, then
// this thread's (memory.hpp).
template <class Variant>
std::pair<const std::uint32_t*, const std::uint32_t*> tables(int lg) {
    if (lg <= kStaticLog) return {kStaticTables.words[0], kStaticTables.words[1]};
    return thread_tables<Variant>(lg);
}

}  // namespace detail

// out = f g mod x^out.size(), from and into the caller's memory (above). f, g: words < P.
template <class Variant = SelectedMultiply>
void multiply(std::span<const std::uint32_t> f, std::span<const std::uint32_t> g, std::span<std::uint32_t> out) {
    const std::size_t n = std::min(f.size(), out.size()), m = std::min(g.size(), out.size());
    const std::size_t count = n && m ? std::min(n + m - 1, out.size()) : 0;
    if (count) {
        const int lg = detail::log_length(n, m);
        if (lg > kMaxLog) throw std::length_error("fps::ntt::multiply: product longer than 2^kMaxLog");
        const auto [roots, inverse_roots] = detail::tables<Variant>(lg);
        const detail::Region region = detail::take_region(detail::region_bytes(lg));
        const std::size_t span = (std::size_t(1) << lg) + kPadding;
        auto* a = reinterpret_cast<std::uint32_t*>(
            region.start + detail::place(f.data(), g.data(), out.data(), span * sizeof(std::uint32_t)));
        std::uint32_t* b = a + span;
        Variant::multiply(detail::Product{f.data(), n, g.data(), m, a, b, out.data(), count, lg}, roots, inverse_roots);
        detail::give_region(region);
    }
    std::fill(out.begin() + count, out.end(), 0u);
}

// The product of a of n and b of m coefficients, n + m - 1 <= 2^kMaxLog. a and b: length() +
// kPadding words each, 64-byte aligned (the padding is read, never written); a in this thread's
// region or the caller's memory, b in the region. Tables up to 2^detail::kStaticLog are the
// compile-time ones, longer ones this thread's (memory.hpp).
class Convolution {
public:
    Convolution(std::size_t n, std::size_t m) : Convolution(n, m, nullptr) {}

    // a: the caller's, 64-byte aligned, length(n, m) + kPadding words; null for this thread's region.
    Convolution(std::size_t n, std::size_t m, std::uint32_t* a) : n_(n), m_(m), lg_(detail::log_length(n, m)) {
        if (lg_ > kMaxLog) throw std::length_error("fps::ntt::Convolution: product longer than 2^kMaxLog");
        if (reinterpret_cast<std::uintptr_t>(a) % 64)
            throw std::invalid_argument("fps::ntt::Convolution: a is not 64-byte aligned");
        const std::size_t span = length() + kPadding;
        region_ = detail::take_region(detail::region_bytes(lg_));
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
    static std::size_t length(std::size_t n, std::size_t m) { return std::size_t(1) << detail::log_length(n, m); }

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
        const auto [roots, inverse_roots] = detail::tables<Variant>(lg_);
        Variant::multiply(a_, b_, n_, m_, lg_, roots, inverse_roots);
        return a_;
    }

private:
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
