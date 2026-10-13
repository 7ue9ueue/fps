// Number-theoretic transforms and polynomial products modulo P = 998244353 (backend kernel).
// Variants: scalar (every target), intrinsics<S> (every target, on a backend/simd struct), avx2
// (lc-opt's, asm scheduled for Zen 3) and neon (arm: intrinsics<Neon> with asm generated for
// Neoverse V2). select.hpp picks one per op and length from a tuned table
// (one table read per call; backend/ntt/tune.hpp measures it, tune::run() and wisdom keep it); the
// ops' Variant template argument calls one directly. Linux or macOS. Design and measurements: notes.md.
//
//   fps::ntt::multiply(f, g, out);             // out = f g mod x^out.size(); spans of words < P
//   fps::ntt::square(f, out);                  // out = f^2 mod x^out.size()
//   fps::ntt::middle_product(f, g, out);       // out[i] = (f g)[g.size() - 1 + i]
//
//   fps::ntt::Convolution conv(n, m);         // factors of n and m coefficients
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
//   t.forward_upper(h);                        // h: L / 2 words, p mod (x^(L/2) + 1) -> the upper
//                                              // half of p's leaf form (transform doubling)
//   t.inverse_upper(h);                        // forward_upper undone: the upper half of a leaf
//                                              // form -> p mod (x^(L/2) + 1)
//   t.multiply(a, b, out);                     // out = a b leaf by leaf; out may be a or b
//   t.multiply_add(acc, a, b);                 // acc += a b leaf by leaf; acc may be a or b
//   t.inverse(F, out);                         // out = the first out.size() coefficients; F kept
//   t.inverse(F, out, Form::leaves, {.offset = k, .add = true});   // out[i] += coefficient k + i
//   t.forward(f, Form::values);                // f -> values form (L <= 2^23); inverse(F, Form::values),
//                                              // inverse(F, out, Form::values) undo it
//   t.forward_upper(h, Form::values);          // the upper half of p's values form (doubling)
//
//   // Fused ops for Newton steps: sources and outputs in caller memory, one pass each.
//   t.forward(std::span(g), G);                // G = leaf form of g (g.size() <= L, zero past it)
//   t.forward_product(q, G, X);                // X = leaf form of q, times G leaf by leaf
//   t.forward_upper(r, X + L / 2);             // the upper half of r's leaf form (r.size() <= L / 2)
//   t.cyclic_product(f, G, work, e, {.offset = L / 2});  // e[i] = (f g mod (x^L - 1))[L/2 + i]
//   t.cyclic_product(e, G, work, e, {.negate = true});   // e = -(e g mod (x^L - 1))[0, e.size())
//   const fps::ntt::Term terms[] = {{A, B}, {C, D}};
//   t.inverse_product(terms, work, out);       // out = coefficients of A B + C D
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
// square(f, out): multiply(f, f, out) with one forward transform, not two. Same contract.
//
// middle_product(f, g, out): out[i] = (f g)[m - 1 + i] = sum_{j < m} f[i + j] g[m - 1 - j] for
// n = f.size() >= m = g.size() >= 1 and out.size() <= n - m + 1 (else std::invalid_argument), the
// sums that use all of g. Reads f[0, m - 1 + out.size()) only. A cyclic product of length 2^lg >=
// m - 1 + out.size(), not >= n + m - 1: its wrapped terms land below word m - 1. g is copied into
// this thread's region turned by x^-(m - 1) (zeros between), so the result starts at word 0 and
// the last level writes it to out. out may overlap f or g. Throws std::length_error, before
// writing out, past 2^kMaxLog.
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
// multiply(a, b, out) and multiply_add: leaf products of two leaf forms (canonical out), each an
// 8 x 8 product in 64-bit sums, reduced exactly (x86: hi (2^32 mod P) + lo) or by Montgomery and
// a Shoup product by 2^32. Zen 3: 0.93-0.97 ns per word at every length (notes.md).
//
// Values form (Form::values, L <= 2^kValuesMaxLog = 2^23, as 2^lg must divide P - 1): each leaf
// taken on to its values at the 8 roots of x^8 - w_v, so a product of two values forms is one
// modular product per word. Words in blocks of 64 (8 leaves): word 64g + 8i + t is leaf 8g + t at
// rho z^bitrev3(i), rho = r(4(8g + t)) an 8th root of w_v, z = r(2) of order 8 (common.hpp). Per
// leaf a twist by rho^k and an 8-point transform, run on each subtree of the transform while it is
// in cache (after its bottom level forward, before it inverse).
//
// forward_upper(h): transform doubling. The leaf form of p (deg p < L) is the leaf form of length
// L / 2 of p mod (x^(L/2) - 1), then the upper half, from h = p mod (x^(L/2) + 1): the subtree of
// group 1 under a radix-2 top level, or of groups 2 and 3 under a radix-4 one (one radix-2 level
// with r(1) first). So a product tree's node of length L / 2 doubles to its parent's length with
// one transform of L / 2 words and no twist pass (backend/product_tree). The variant is the
// forward's at L / 2.
//
// inverse_upper(h): forward_upper undone, with the inverse's scale: the subtrees' inverse levels,
// then (radix-4 top) the radix-2 level with r(1)^-1 and the scale, or (radix-2 top) a pass for the
// scale. With the lower half's inverse it gives p mod (x^(L/2) -+ 1) from p's leaf form at half
// the cost of an inverse of length L each (backend/evaluation's descent halves a node's transform).
//
// inverse(F, out, ...): into other memory, F kept. Its subtrees copy (or join) their leaves from F
// into this thread's region before their bottom level, and the top level writes out: no copy of
// the whole of F in, nor of the coefficients out (with an offset or add: one pass out). Against
// a copy in, the inverse in place and a copy out: 0.84-0.97 from 2^11 on Zen 3 and Emerald Rapids
// (Zen 5 1.08 at 2^22). To 2^10, where that was slower, F is copied into a stack buffer.
//
// Arrays for Transform: 32-byte aligned, L + kPadding words (the padding is read, never written),
// values < P in and out. No checks on these calls but the ranges of inverse(F, out, ...) and the
// values form's length. Lengths: 2^kMinLog .. 2^kMaxLog. The ops with a Form or a span run the
// variant the tuned table picks for forward or inverse at that length (select.hpp); multiply and
// multiply_add run Native.
//
// Fused ops (issue #112, lc-opt's transform layer): the leaf forms they read or write (F, B, the
// terms, work) are arrays as above; sources and outputs are spans of any alignment. forward(in, F)
// reads in from its first level, skipping the zero half of a source of at most L/2 words. A cyclic
// product runs the forward transform of in, the leaf products with B in each subtree's bottom stage
// (registers, not a pass of their own) and the inverse transform depth first, so each subtree is
// in cache once; inverse_product does the same for an inverse of leaf products. Their top level
// writes only the half asked for, negated through its scale. Zen 3, bench_fused.cpp: a cyclic
// product took 0.83-0.89 of the time of the steps it replaces (copy in, forward, multiply, inverse
// in place, copy out) from 2^12 on, 0.98 at 2^8 (notes/112-r1.md).
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

// out[i] = out[i] + x[i] mod P for n words, both canonical, any alignment.
inline void add_words(std::uint32_t* out, const std::uint32_t* x, std::size_t n) {
    using S = simd::Native;
    using M = modarith::Ops<S, kModulus>;
    std::size_t i = 0;
    for (; i + S::lanes <= n; i += S::lanes) S::store(out + i, M::add(S::load(out + i), S::load(x + i)));
    for (; i < n; ++i) {
        const std::uint32_t s = out[i] + x[i];
        out[i] = s >= kP ? s - kP : s;
    }
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

// out = f^2 mod x^out.size(), from and into the caller's memory (above). f: words < P.
template <class Variant = SelectedMultiply>
void square(std::span<const std::uint32_t> f, std::span<std::uint32_t> out) {
    const std::size_t n = std::min(f.size(), out.size());
    const std::size_t count = n ? std::min(2 * n - 1, out.size()) : 0;
    if (count) {
        const int lg = detail::log_length(n, n);
        if (lg > kMaxLog) throw std::length_error("fps::ntt::square: square longer than 2^kMaxLog");
        const auto [roots, inverse_roots] = detail::tables<Variant>(lg);
        const detail::Region region = detail::take_region(detail::region_bytes(lg));
        const std::size_t span = (std::size_t(1) << lg) + kPadding;
        auto* a = reinterpret_cast<std::uint32_t*>(
            region.start + detail::place(f.data(), f.data(), out.data(), span * sizeof(std::uint32_t)));
        Variant::square(detail::Product{f.data(), n, f.data(), n, a, a, out.data(), count, lg}, roots, inverse_roots);
        detail::give_region(region);
    }
    std::fill(out.begin() + count, out.end(), 0u);
}

// out[i] = (f g)[g.size() - 1 + i], the middle product (above). f, g: words < P.
template <class Variant = SelectedMultiply>
void middle_product(std::span<const std::uint32_t> f, std::span<const std::uint32_t> g, std::span<std::uint32_t> out) {
    const std::size_t m = g.size(), count = out.size();
    if (m == 0 || m > f.size() || count > f.size() - m + 1)
        throw std::invalid_argument("fps::ntt::middle_product: needs f.size() >= g.size() >= 1, out.size() <= "
                                    "f.size() - g.size() + 1");
    if (count == 0) return;
    const std::size_t n = m - 1 + count;  // the words of f read
    const int lg = std::max(kMinLog, int(std::bit_width(n - 1)));
    if (lg > kMaxLog) throw std::length_error("fps::ntt::middle_product: longer than 2^kMaxLog");
    const auto [roots, inverse_roots] = detail::tables<Variant>(lg);
    const detail::Region region = detail::take_region(detail::region_bytes(lg));
    const std::size_t len = std::size_t(1) << lg, span = len + kPadding;
    auto* a = reinterpret_cast<std::uint32_t*>(
        region.start + detail::place(f.data(), g.data(), out.data(), span * sizeof(std::uint32_t)));
    std::uint32_t* b = a + span;
    // b = g x^-(m - 1) mod x^len - 1: g[m - 1] at word 0, g[0, m - 1) at the top, zeros between.
    b[0] = g[m - 1];
    std::fill(b + 1, b + len - (m - 1), 0u);
    std::copy_n(g.data(), m - 1, b + len - (m - 1));
    Variant::multiply(detail::Product{f.data(), n, b, len, a, b, out.data(), count, lg}, roots, inverse_roots);
    detail::give_region(region);
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

// The forms of a transform: the leaf form, or the values form (lengths up to 2^kValuesMaxLog).
enum class Form { leaves, values };

// The coefficients an inverse into other memory writes (Transform::inverse), and how.
struct Coefficients {
    std::size_t offset = 0;  // out[i] is coefficient offset + i
    bool add = false;        // out[i] += it, mod P (out canonical); else out[i] = it
};

// The coefficients a fused op writes (Transform::cyclic_product, inverse_product).
struct Output {
    std::size_t offset = 0;  // out[i] is coefficient offset + i: 0 or length() / 2
    bool negate = false;     // out[i] = -it
};

// A term of Transform::inverse_product: the leaf products of the leaf forms a and b, or a alone.
using Term = detail::LeafTerm;

// Forward and inverse transforms of length 2^lg, kMinLog <= lg <= kMaxLog, with their tables.
class Transform {
public:
    explicit Transform(int lg) : lg_(checked(lg)), tables_(new Line[2 * lines(lg) + value_lines(lg)]) {
        Native::build_table(roots(), length() / 16, false);
        Native::build_table(inverse_roots(), length() / 16, true);
        if (lg <= kValuesMaxLog) detail::build_value_rows(value_rows(), lg);
    }

    std::size_t length() const { return std::size_t(1) << lg_; }

    // f -> its leaf form.
    template <class Variant = SelectedForward>
    void forward(std::uint32_t* f) const { Variant::forward(f, lg_, roots()); }

    // Leaf form -> f; exactly undoes forward().
    template <class Variant = SelectedInverse>
    void inverse(std::uint32_t* f) const { Variant::inverse(f, lg_, inverse_roots()); }

    // f -> its leaf form or values form. Throws std::length_error for the values form past
    // 2^kValuesMaxLog.
    template <class Variant = SelectedForward>
    void forward(std::uint32_t* f, Form form) const {
        if (form == Form::leaves) return Variant::forward(f, lg_, roots());
        Variant::forward_values(f, lg_, roots(), checked_rows());
    }

    // h: length() / 2 words (32-byte aligned, kPadding after), the coefficients of p mod
    // (x^(L/2) + 1), canonical -> leaves L / 16 .. L / 8 - 1 of p's leaf form. lg > kMinLog.
    template <class Variant = SelectedForward>
    void forward_upper(std::uint32_t* h) const {
        if (lg_ == kMinLog) throw std::length_error("fps::ntt::Transform::forward_upper: needs lg > kMinLog");
        Variant::forward_upper(h, lg_ - 1, roots());
    }

    // forward_upper(h), then (Form::values) those leaves on to their values: the upper half of p's
    // values form, so a values form doubles too (backend/gcd). The split is intrinsics'
    // (Values<Native>; its words are every variant's). Throws std::length_error past 2^kValuesMaxLog.
    template <class Variant = SelectedForward>
    void forward_upper(std::uint32_t* h, Form form) const {
        if (form == Form::leaves) return forward_upper<Variant>(h);
        const std::uint32_t* rows = checked_rows();
        forward_upper<Variant>(h);
        const std::size_t blocks = length() / 128;  // of 64 words in each half
        detail::intrinsics::Values<detail::intrinsics::Ntt<simd::Native>::N>::split(h, h, blocks, blocks, rows);
    }

    // forward_upper undone: h (length() / 2 words, 32-byte aligned, kPadding after), leaves L / 16
    // .. L / 8 - 1 of p's leaf form -> the coefficients of p mod (x^(L/2) + 1), canonical. lg >
    // kMinLog. The variant is the inverse's at L / 2.
    template <class Variant = SelectedInverse>
    void inverse_upper(std::uint32_t* h) const {
        if (lg_ == kMinLog) throw std::length_error("fps::ntt::Transform::inverse_upper: needs lg > kMinLog");
        Variant::inverse_upper(h, lg_ - 1, inverse_roots());
    }

    // F in that form -> its coefficients, in place.
    template <class Variant = SelectedInverse>
    void inverse(std::uint32_t* F, Form form) const {
        if (form == Form::leaves) return Variant::inverse(F, lg_, inverse_roots());
        Variant::inverse(F, F, F, length(), lg_, inverse_roots(), checked_rows());
    }

    // out[i] = coefficient c.offset + i of the polynomial whose leaf or values form is F, or
    // out[i] += it (c.add); F kept. out: any memory apart from F, any alignment, no padding;
    // c.offset + out.size() <= length(), else std::out_of_range. The work runs in this thread's
    // region (Convolution's memory), on the stack to 2^kStackLog; the top level writes out when c
    // is {}. With an offset or add, one pass more from the work array.
    template <class Variant = SelectedInverse>
    void inverse(const std::uint32_t* F, std::span<std::uint32_t> out, Form form = Form::leaves, Coefficients c = {}) const {
        const std::size_t len = length();
        if (c.offset > len || out.size() > len - c.offset)
            throw std::out_of_range("fps::ntt::Transform::inverse: coefficients past the length");
        const std::uint32_t* rows = form == Form::values ? checked_rows() : nullptr;
        if (out.empty()) return;
        const auto run = [&](std::uint32_t* f) {
            if (c.offset == 0 && !c.add) return Variant::inverse(F, f, out.data(), out.size(), lg_, inverse_roots(), rows);
            Variant::inverse(F, f, f, len, lg_, inverse_roots(), rows);
            if (c.add) detail::add_words(out.data(), f + c.offset, out.size());
            else std::copy_n(f + c.offset, out.size(), out.data());
        };
        if (lg_ <= kStackLog) {  // F copied whole, the inverse in place: faster to 2^10 (notes.md)
            // f 256 bytes or more from F's and out's page offsets (place()): of 4 offsets 1 KiB
            // apart, each of the two excludes at most one.
            alignas(64) std::uint32_t buffer[(std::size_t(1) << kStackLog) + kPadding + 768];
            const auto apart = [&buffer](std::size_t a, const void* p) {
                const std::size_t x = (reinterpret_cast<std::uintptr_t>(buffer) + a) % 4096, y = reinterpret_cast<std::uintptr_t>(p) % 4096;
                return (x - y + 256) % 4096 >= 512;
            };
            std::size_t a = 0;
            while (!apart(a, F) || !apart(a, out.data())) a += 1024;
            std::uint32_t* f = buffer + a / sizeof(std::uint32_t);
            std::copy_n(F, len, f);
            std::fill_n(f + len, kPadding, 0u);
            if (rows) Variant::inverse(f, f, f, len, lg_, inverse_roots(), rows);
            else Variant::inverse(f, lg_, inverse_roots());
            if (c.add) detail::add_words(out.data(), f + c.offset, out.size());
            else std::copy_n(f + c.offset, out.size(), out.data());
            return;
        }
        const detail::Region region = detail::take_region(sizeof(std::uint32_t) * (len + kPadding) + 4096);
        run(reinterpret_cast<std::uint32_t*>(region.start + detail::place(F, F, out.data(), 0)));
        detail::give_region(region);
    }

    // out = a b leaf by leaf (leaf forms, mod x^8 - w_v); out may be a or b.
    template <class Variant = Native>
    void multiply(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t* out) const {
        Variant::leaf_products(a, b, out, lg_, roots(), false);
    }

    // acc += a b leaf by leaf; acc may be a or b.
    template <class Variant = Native>
    void multiply_add(std::uint32_t* acc, const std::uint32_t* a, const std::uint32_t* b) const {
        Variant::leaf_products(a, b, acc, lg_, roots(), true);
    }

    // Fused ops (Newton steps): sources in caller memory, products and inverses in one pass.

    // F = the leaf form of in (in.size() <= length() words, zero past them): the top level reads in,
    // and skips the zero half of an in of at most length() / 2 words. F apart from in.
    template <class Variant = SelectedForward>
    void forward(std::span<const std::uint32_t> in, std::uint32_t* F) const {
        Variant::forward_from(in.data(), checked_source(in, length()), F, lg_, roots());
    }

    // F = the leaf form of in times B, leaf by leaf: forward(in, F) then multiply(F, B, F) in one
    // pass. F apart from in and B.
    template <class Variant = SelectedForward>
    void forward_product(std::span<const std::uint32_t> in, const std::uint32_t* B, std::uint32_t* F) const {
        Variant::forward_product(in.data(), checked_source(in, length()), B, F, lg_, roots());
    }

    // H (length() / 2 words) = forward_upper of h = in, zero past it (in.size() <= length() / 2):
    // the upper half of the leaf form of in. lg > kMinLog. H apart from in.
    template <class Variant = SelectedForward>
    void forward_upper(std::span<const std::uint32_t> in, std::uint32_t* H) const {
        if (lg_ == kMinLog) throw std::length_error("fps::ntt::Transform::forward_upper: needs lg > kMinLog");
        Variant::forward_upper_from(in.data(), checked_source(in, length() / 2), H, lg_ - 1, roots());
    }

    // out[i] = coefficient o.offset + i of in b mod (x^L - 1), negated if o.negate, for B the leaf
    // form of b: the forward transform of in, the leaf products and the inverse transform in one
    // depth-first pass, each subtree in cache. work: length() + kPadding words, apart from in, B and
    // out; out may overlap in (in is read before out is written). o.offset: 0 or length() / 2,
    // o.offset + out.size() <= length(), else std::out_of_range.
    template <class Variant = SelectedMultiply>
    void cyclic_product(std::span<const std::uint32_t> in, const std::uint32_t* B, std::uint32_t* work,
                        std::span<std::uint32_t> out, Output o = {}) const {
        const std::size_t n = checked_source(in, length());
        const unsigned output = checked_output(out, o);
        if (out.empty()) return;
        Variant::cyclic_product(in.data(), n, B, work, out.data(), out.size(), lg_, roots(), inverse_roots(), output);
    }

    // out[i] = coefficient o.offset + i of the polynomial whose leaf form is the sum over the terms
    // of a b leaf by leaf (of a, if b is null), negated if o.negate: the leaf products in each
    // subtree of the inverse transform, in cache. 1 to 4 terms (else std::invalid_argument); work,
    // o: as cyclic_product; out apart from the terms.
    template <class Variant = SelectedInverse>
    void inverse_product(std::span<const Term> terms, std::uint32_t* work, std::span<std::uint32_t> out, Output o = {}) const {
        if (terms.empty() || terms.size() > std::size_t(detail::kMaxTerms))
            throw std::invalid_argument("fps::ntt::Transform::inverse_product: 1 to 4 terms");
        const unsigned output = checked_output(out, o);
        if (out.empty()) return;
        Variant::inverse_product(terms.data(), int(terms.size()), work, out.data(), out.size(), lg_, roots(),
                                 inverse_roots(), output);
    }

private:
    static std::size_t checked_source(std::span<const std::uint32_t> in, std::size_t most) {
        if (in.size() > most) throw std::out_of_range("fps::ntt::Transform: source longer than the transform");
        return in.size();
    }
    // The variants' output flags (common.hpp): kUpperHalf for o.offset = length() / 2, kNegated.
    unsigned checked_output(std::span<std::uint32_t> out, Output o) const {
        const std::size_t len = length();
        if ((o.offset != 0 && o.offset != len / 2) || out.size() > len - o.offset)
            throw std::out_of_range("fps::ntt::Transform: output offset not 0 or length() / 2, or past the length");
        return (o.offset ? detail::kUpperHalf : 0u) | (o.negate ? detail::kNegated : 0u);
    }

    struct alignas(64) Line {
        std::uint32_t words[16];
    };
    static int checked(int lg) {
        if (lg < kMinLog || lg > kMaxLog) throw std::length_error("fps::ntt::Transform: lg out of range");
        return lg;
    }
    static constexpr int kStackLog = 10;  // inverse(F, out, ...) on the stack, F copied, up to 2^kStackLog
    static std::size_t lines(int lg) { return (detail::table_words(lg) + 15) / 16; }
    static std::size_t value_lines(int lg) { return lg <= kValuesMaxLog ? 2 * detail::value_row_count(lg) : 0; }

    std::uint32_t* roots() const { return tables_[0].words; }
    std::uint32_t* inverse_roots() const { return tables_[lines(lg_)].words; }
    std::uint32_t* value_rows() const { return tables_[2 * lines(lg_)].words; }
    const std::uint32_t* checked_rows() const {
        if (lg_ > kValuesMaxLog) throw std::length_error("fps::ntt::Transform: the values form needs lg <= 23");
        return value_rows();
    }

    int lg_;
    std::unique_ptr<Line[]> tables_;
};

}  // namespace fps::ntt
