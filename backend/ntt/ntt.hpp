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
//   fps::ntt::Transform u(lg2, t);             // lg2 <= lg, on t's tables (t outlives u)
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
//   t.forward_upper(std::span(h), H, Form::values);  // the same from h in caller memory (backend/gcd)
//   t.inverse(F, work, out, Form::values);     // out = the first out.size() coefficients, through the
//                                              // caller's work (F may be work)
//
//   // Fused ops for Newton steps: sources and outputs in caller memory, one pass each.
//   t.forward(std::span(g), G);                // G = leaf form of g (g.size() <= L, zero past it)
//   t.forward_product(q, G, X);                // X = leaf form of q, times G leaf by leaf
//   t.forward_upper(r, X + L / 2);             // the upper half of r's leaf form (r.size() <= L / 2)
//   t.cyclic_product(f, G, work, e, {.offset = L / 2});  // e[i] = (f g mod (x^L - 1))[L/2 + i]
//   t.cyclic_product(e, G, work, e, {.negate = true});   // e = -(e g mod (x^L - 1))[0, e.size())
//   const fps::ntt::Term terms[] = {{A, B}, {C, D}};
//   t.inverse_product(terms, work, out);       // out = coefficients of A B + C D
//   t.toeplitz_product(A1, A2, A3, B0, B1, work, out);  // out = coefficients of A2 B0 + A1 B1;
//                                              // A3 = A3 B0 + A2 B1: 3 leaf products, not 4
//   t.forward(std::span(g), V, Form::values);  // V = values form of g (V apart from g)
//   t.to_values(G, V);                         // V = the values form of the leaf form G (a pass)
//   t.inverse_product(terms, work, out, {}, Form::values);  // terms of values forms: one
//                                              // Montgomery product per word and term
//
//   // Hooked ops: the caller's pass on each run of leaves in cache; pruned levels (issue #149).
//   t.forward_hooked(in, F, hook, zero);       // F = leaf form of in; hook(F + 8 i, i, count) on runs
//   t.inverse_hooked(hook, work, out, o, unused);  // hook writes the leaves (times 2^-32), then inverse
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
// Ragged lengths (issue #155, ragged.hpp): a product of N = n + m - 1 words with L < N <= 2L (a
// middle product: of the m - 1 + out.size() words of f it reads) runs on about N words of
// transforms, not 2L: c = A + (x^L - 1) q, A = c mod (x^L - 1) a cyclic product of length L, and q,
// the top N - L words of c, from a cyclic product of length s mod (x^s - 3^s) (twisted to y^s - 1)
// and a product of the top words of f and g; middle products run the transpose. The plan per
// length and r / L is a tuned choice ("ntt.ragged"); the plain product at 2L stays one of its
// options, and powers of two run it. Against the plain products, same run, on Zen 3, Emerald
// Rapids, Zen 5 and Neoverse V2 (notes/155-r1.md): N = 2^20 + 1 0.49-0.51x the time, 1.25 2^20
// 0.66-0.69x, 1.5 2^20 0.80-0.82x; ns per word of N 1.00-1.05, 1.10-1.13 and 1.12-1.15 of a product
// of 2^20. multiply and square have a third form (truncated.hpp, notes/155-r2.md): the transform of
// 2L cut to the first N words of its leaf form (rounded up to nodes of 128 words), the variants'
// group products on its nodes and generic levels above them; it wins most of (1.25, 1.94) 2^k from
// 2^12 on: 1.75 2^20 0.90-0.92x the time of the plain product (1.05-1.14 per word of 2^20's, was
// 1.17-1.24), 1.625 2^20 0.90-0.96x, 1.875 2^20 0.95-0.98x. Middle products have one too (round 3,
// truncated.hpp's Middle products, notes/155-r3.md): v extended past its N words so that it vanishes
// on the nodes past the cut, then the group products and an inverse pruned to them, writing words
// [m - 1, N); its plan is the table's second half (choose_middle). Against the plain middle product,
// same binary (notes/155-r3.md): 1.25 2^20 0.63-0.66x, 1.5 2^20 0.73-0.77x, 1.75 2^20 0.87-0.91x,
// 1.875 2^20 0.93-0.97x; 4.4-4.6 ns per word on Zen 3 from 1.25 to 1.875 2^20.
// Convolution::length and Transform stay powers of two: callers size memory and Newton steps by
// them.
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
//
// The fused ops in the values form (issue #251): forward(in, V, Form::values) splits each run of
// leaves in cache after the bottom level (forward_hooked), and inverse_product(terms, ..., Form::
// values) sums the terms' Montgomery products and joins each run before its bottom level
// (inverse_hooked, the 2^-32 and the join's 8 in the top level's scale). Sums of several products
// of kept transforms (series' blocked division) then cost a product per word and term, not a leaf
// product; at 2^19 on Zen 5, ns per word: a forward 0.46 leaves, 0.65 values; inverse_product of
// 1 / 2 / 4 terms 1.01 / 1.46 / 2.24 leaves, 0.74 / 0.80 / 0.97 values (notes/251-r1.md).
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
#include "backend/ntt/ragged.hpp"
#include "backend/ntt/select.hpp"
#include "backend/ntt/truncated.hpp"

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
// out, with b span bytes after a: the middle of the widest gap between the offsets of f, of out
// and of g minus span, modulo `period`, so a lies as far as it can from f and out, and b from g.
// Arrays 0 to 192 bytes apart modulo 4 KiB ran a pass up to 1.7x slower (4K aliasing, issue #38);
// on Zen 3 a product at 2^14-2^22 ran 3-5% faster with a 1-3 KiB from f and out than 256-512
// bytes, the offset the rule before #236 (the first 256 bytes from each) gave (notes/236-r1.md).
inline std::size_t place(const void* f, const void* g, const void* out, std::size_t span, std::size_t period = 4096) {
    const std::size_t mask = period - 1;  // period: a power of two (masks, not divisions: 2^6 products)
    const auto at = [mask](const void* p) { return std::size_t(reinterpret_cast<std::uintptr_t>(p)) & mask; };
    std::size_t p[3] = {at(f), at(out), (at(g) - span) & mask};
    if (p[0] > p[1]) std::swap(p[0], p[1]);
    if (p[1] > p[2]) std::swap(p[1], p[2]);
    if (p[0] > p[1]) std::swap(p[0], p[1]);
    std::size_t start = p[2], gap = p[0] + period - p[2];
    if (p[1] - p[0] > gap) start = p[0], gap = p[1] - p[0];
    if (p[2] - p[1] > gap) start = p[1], gap = p[2] - p[1];
    return ((start + gap / 2) & mask) / 64 * 64;
}

// place()'s period for a transform of 2^lg words: the stride of its top level's blocks (2^lg / 8
// words at even lg, 2^lg / 4 at odd) where that is 1 to 2 KiB, so a's blocks fall between f's
// (2^12, blocks 2 KiB apart: 1.05x slower with a 2 KiB from f), else 4 KiB.
inline std::size_t place_period(int lg) {
    const std::size_t stride = sizeof(std::uint32_t) << (lg - (lg % 2 == 0 ? 3 : 2));
    return stride >= 1024 && stride < 4096 ? stride : 4096;
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

// The variant's product or square of a Product.
template <class Variant, bool kSquare>
[[gnu::always_inline]] inline void run_product(const Product& p, const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
    if constexpr (kSquare) Variant::square(p, roots, inverse_roots);
    else Variant::multiply(p, roots, inverse_roots);
}

// The log of the tables a plan needs: a truncated plan's of 2L.
constexpr int table_log(const ragged::Plan& p) { return p.lg + (p.truncated ? 1 : 0); }

// ragged_into's placement of an array of span = 2^k + kPadding words in work at `at` (64-byte
// aligned) or less than place()'s period past it: far from x, y and z modulo the period. place()'s
// offset is from a base aligned to the period, so `at`'s own offset is taken off (before #261 it
// was added: the top plan's middle product at 2^14 + 2^9 put a 64-128 bytes past f and out modulo
// 4 KiB, and ran 1.05x slower on Zen 3).
inline std::uint32_t* placed(std::uint32_t* at, const void* x, const void* y, const void* z, std::size_t span) {
    const std::size_t period = place_period(std::countr_zero(span - kPadding));
    const std::size_t target = place(x, y, z, span * sizeof(std::uint32_t), period);
    const std::size_t now = std::size_t(reinterpret_cast<std::uintptr_t>(at)) & (period - 1);
    return at + ((target - now) & (period - 1)) / sizeof(std::uint32_t);
}

// c = A + (x^L - 1) q's A = c mod (x^L - 1), the variant's product of length L of f and g (each
// folded mod x^L - 1 first if longer), to out[0, L); work: 3 (L + kPadding) + 1024 words.
template <class Variant, bool kSquare>
void cyclic_into(const std::uint32_t* f, std::size_t n, const std::uint32_t* g, std::size_t m, std::uint32_t* out,
                 std::uint32_t* work, int lg, const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
    const std::size_t L = std::size_t(1) << lg;
    const std::uint32_t *ff = f, *gg = g;
    std::size_t nn = n, mm = m;
    std::uint32_t* next = work;
    if (n > L) ragged::fold(f, n, L, next), ff = next, nn = L, next += L + kPadding;
    if (m > L) ragged::fold(g, m, L, next), gg = next, mm = L, next += L + kPadding;
    std::uint32_t* a = placed(next, ff, gg, out, L + kPadding);
    run_product<Variant, kSquare>(Product{ff, nn, kSquare ? ff : gg, kSquare ? nn : mm, a, kSquare ? a : a + L + kPadding,
                                          out, L, lg},
                                  roots, inverse_roots);
}

// count <= N words of f g (f^2 if kSquare), N = n + m - 1, by a truncated plan into out
// (truncated.hpp): f and g to the leaf form of length 2L cut to shape(N)'s nodes, the group
// products, the inverse into out. work: ragged::truncated_words(L); tables of 2L.
template <class Variant, bool kSquare>
void truncated_into(const std::uint32_t* f, std::size_t n, const std::uint32_t* g, std::size_t m, std::uint32_t* out,
                    std::size_t count, std::uint32_t* work, const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
    const truncated::Shape s = truncated::shape(n + m - 1);
    const std::size_t span = (std::size_t(2) << s.lg) + kPadding;
    std::uint32_t* const a = placed(work, f, g, out, span);
    std::uint32_t* const b = a + span;
    truncated::forward(f, n, a, s, roots);
    if (!kSquare) truncated::forward(g, m, b, s, roots);
    for (int i = 0; i < s.fulls; ++i) {
        const truncated::Node& node = s.full[i];
        if constexpr (kSquare) Variant::group_square(a + node.at, node.size / 8, node.k, roots, inverse_roots);
        else Variant::group_product(a + node.at, b + node.at, node.size / 8, node.k, roots, inverse_roots);
    }
    truncated::inverse(a, out, count, s, roots, inverse_roots);
}

// f g (f^2 if kSquare: g = f) by plan p into out, N = n + m - 1 words of memory apart from f and g
// (ragged.hpp). work: 64-byte aligned, ragged::split_words(2^p.lg) words (p.split),
// ragged::truncated_words(2^p.lg) (p.truncated) or 2 (2^p.lg + kPadding) + 1024; tables for
// 2^table_log(p) at least.
template <class Variant, bool kSquare>
void ragged_into(const std::uint32_t* f, std::size_t n, const std::uint32_t* g, std::size_t m, std::uint32_t* out,
                 std::uint32_t* work, const ragged::Plan& p, const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
    const std::size_t N = n + m - 1, L = std::size_t(1) << p.lg;
    if (p.truncated) return truncated_into<Variant, kSquare>(f, n, g, m, out, N, work, roots, inverse_roots);
    if (!p.split) {
        std::uint32_t* a = placed(work, f, g, out, L + kPadding);
        run_product<Variant, kSquare>(Product{f, n, g, m, a, kSquare ? a : a + L + kPadding, out, N, p.lg}, roots,
                                      inverse_roots);
        return;
    }
    const std::size_t r = N - L, s = p.s, t = p.t;
    if (t) {  // T, the top t words of c, from the top t words of f and g, to out[L + s, N)
        const std::size_t nt = std::min(n, t), mt = std::min(m, t), count = nt + mt - 1;
        ragged_into<Variant, kSquare>(f + (n - nt), nt, g + (m - mt), mt, out, work, ragged::choose_product(count), roots,
                                      inverse_roots);
        std::copy(out + (count - t), out + count, out + L + s);
    }
    cyclic_into<Variant, kSquare>(f, n, g, m, out, work, p.lg, roots, inverse_roots);  // A = c mod (x^L - 1)
    std::uint32_t* C = work;  // the twisted piece: f(z y) g(z y) mod (y^s - 1)
    if (s) {
        std::uint32_t *fs = work, *gs = work + s + kPadding;
        ragged::twist_fold(f, n, s, fs);
        if (!kSquare) ragged::twist_fold(g, m, s, gs);
        std::uint32_t* a = placed(gs + s + kPadding, fs, gs, fs, s + kPadding);
        run_product<Variant, kSquare>(Product{fs, s, kSquare ? fs : gs, s, a, kSquare ? a : a + s + kPadding, fs, s,
                                              std::countr_zero(s)},
                                      roots, inverse_roots);
    }
    ragged::combine(out, L, r, s, t, C);
}

// Words of work middle_into needs for g of m words and count words out (v of m - 1 + count words),
// by plan p.
inline std::size_t middle_words(std::size_t m, std::size_t count, const ragged::Plan& p) {
    const std::size_t len = std::size_t(1) << p.lg;
    const auto block = [](std::size_t w) { return (w + kPadding + 15) / 16 * 16; };
    if (p.truncated) return ragged::truncated_words(len);
    if (!p.split) return 2 * block(len) + 1024;
    const std::size_t s = p.s, t = p.t, nt = std::min(count, t), mt = std::min(m, t);
    const std::size_t arrays = (s ? block(len) + block(s) : 0) + (t ? block(nt + mt - 1) + block(nt) : 0);
    std::size_t more = 2 * block(len) + 1024;
    if (s) more = std::max(more, 3 * block(s) + 1024);
    if (t) more = std::max(more, middle_words(mt, nt, ragged::choose_middle(nt + mt - 1)));
    return arrays + more;
}

// The middle product (middle_into) by a truncated plan of n = m - 1 + count words (truncated.hpp,
// Middle products): v extended to p, p and g to the leaf form of length 2L cut to shape(n)'s nodes,
// the group products, and q's words [m - 1, n) into out. v and g are read before out is written,
// so out may overlap them. work: ragged::truncated_words(L); tables of 2L.
template <class Variant>
void truncated_middle_into(const std::uint32_t* v, std::size_t n, const std::uint32_t* g, std::size_t m, std::uint32_t* out,
                           std::uint32_t* work, const std::uint32_t* roots, const std::uint32_t* inverse_roots) {
    const truncated::Shape s = truncated::shape(n);
    const std::size_t span = (std::size_t(2) << s.lg) + kPadding;
    std::uint32_t* const a = placed(work, v, g, out, span);
    std::uint32_t* const b = a + span;
    truncated::co_forward(v, a, s, roots, inverse_roots);
    truncated::forward(g, m, b, s, roots);
    for (int i = 0; i < s.fulls; ++i) {
        const truncated::Node& node = s.full[i];
        Variant::group_product(a + node.at, b + node.at, node.size / 8, node.k, roots, inverse_roots);
    }
    truncated::pruned_inverse(a, out, m - 1, n, s, roots, inverse_roots);
}

// out[i] = sum_{j < m} v[i + j] g[m - 1 - j] for i < count, v of n = m - 1 + count words: the
// middle product by plan p (ragged.hpp). out apart from v and g; work: middle_words(m, count, p)
// words, 64-byte aligned; tables for 2^table_log(p) at least.
template <class Variant>
void middle_into(const std::uint32_t* v, std::size_t n, const std::uint32_t* g, std::size_t m, std::uint32_t* out,
                 std::size_t count, std::uint32_t* work, const ragged::Plan& p, const std::uint32_t* roots,
                 const std::uint32_t* inverse_roots) {
    const std::size_t L = std::size_t(1) << p.lg;
    const auto block = [](std::size_t w) { return (w + kPadding + 15) / 16 * 16; };
    if (p.truncated) return truncated_middle_into<Variant>(v, n, g, m, out, work, roots, inverse_roots);
    if (!p.split) {  // a cyclic product of length L >= n: its wrapped terms land below word m - 1
        std::uint32_t* a = placed(work, v, g, out, L + kPadding);
        std::uint32_t* b = a + L + kPadding;
        ragged::stage_turned(g, m, L, b);
        Variant::multiply(Product{v, n, b, L, a, b, out, count, p.lg}, roots, inverse_roots);
        return;
    }
    const std::size_t r = n - L, s = p.s, t = p.t, nt = std::min(count, t), mt = std::min(m, t), top = nt + mt - 1;
    std::uint32_t* Abar = s ? work : nullptr;  // s = 0: v[0, L) itself
    std::uint32_t* Cbar = work + (s ? block(L) : 0);
    std::uint32_t* Pbar = Cbar + (s ? block(s) : 0);  // [0 .. 0, Tbar], top words
    std::uint32_t* above = Pbar + (t ? block(top) : 0);  // the top product's out, nt words
    std::uint32_t* rest = above + (t ? block(nt) : 0);
    if (t) std::fill_n(Pbar, top - t, 0u);
    ragged::split_transposed(v, L, r, s, t, Abar, Cbar, t ? Pbar + (top - t) : Pbar);  // t = 0: top wraps
    {  // out[i] = (Abar g x^-(m - 1) mod (x^L - 1))[i mod L]
        const std::uint32_t* A = s ? Abar : v;
        std::uint32_t* a = placed(rest, A, g, out, L + kPadding);  // b far from g: stage_turned copies it
        std::uint32_t* b = a + L + kPadding;
        ragged::stage_turned(g, m, L, b);
        Variant::multiply(Product{A, L, b, L, a, b, out, std::min(count, L), p.lg}, roots, inverse_roots);
        if (count > L) std::copy_n(out, count - L, out + L);
    }
    if (s) {  // += z^i Z[(i + m - 1) mod s], Z = Cbar (z^(m - 1) g(y / z) mod (y^s - 1))
        std::uint32_t* turned = rest;
        ragged::twist_fold(g, m, s, turned, ragged::inverse(ragged::kZeta), power(ragged::kZeta, std::uint32_t((m - 1) % (kP - 1))));
        std::uint32_t* a = placed(turned + block(s), Cbar, turned, Cbar, s + kPadding);
        Variant::multiply(Product{Cbar, s, turned, s, a, a + s + kPadding, Cbar, s, std::countr_zero(s)}, roots, inverse_roots);
        ragged::unfold_add(Cbar, s, m - 1, out, count);
    }
    if (t) {  // += the middle product of [0 .. 0, Tbar] and g[0, mt) in the top nt words
        middle_into<Variant>(Pbar, top, g, mt, above, nt, rest, ragged::choose_middle(top), roots, inverse_roots);
        add_words(out + (count - nt), above, nt);
    }
}

// middle_product(f, g, out) by a split or truncated plan: f of m - 1 + count words read, out of
// count words. Out of line, as ragged_product.
template <class Variant>
[[gnu::noinline]] void ragged_middle(const std::uint32_t* f, const std::uint32_t* g, std::size_t m, std::uint32_t* out,
                                     std::size_t count, const ragged::Plan& p) {
    const std::size_t n = m - 1 + count;
    const auto apart = [out, count](const std::uint32_t* x, std::size_t len) {
        const auto o = reinterpret_cast<std::uintptr_t>(out), b = reinterpret_cast<std::uintptr_t>(x);
        return o + count * sizeof(std::uint32_t) <= b || b + len * sizeof(std::uint32_t) <= o;
    };
    // out itself when apart from f and g (a truncated plan reads them before it writes out); else the
    // region, then copied.
    const bool direct = p.truncated || (apart(f, n) && apart(g, m));
    const std::size_t own = direct ? 0 : (count + kPadding + 15) / 16 * 16;
    const auto [roots, inverse_roots] = tables<Variant>(table_log(p));
    const Region region = take_region(sizeof(std::uint32_t) * (own + middle_words(m, count, p)));
    auto* start = reinterpret_cast<std::uint32_t*>(region.start);
    std::uint32_t* c = direct ? out : start;
    middle_into<Variant>(f, n, g, m, c, count, start + own, p, roots, inverse_roots);
    if (!direct) std::copy_n(c, count, out);
    give_region(region);
}

// multiply(f, g, out) or square(f, out) by a split or truncated plan (ragged.hpp): count words of f
// g to out. Out of line: inlined into multiply(), it cost the frontend's products at 2^10-2^11
// (plain plans) 23-36% on Zen 5 (code layout, notes/155-r1.md).
template <class Variant, bool kSquare>
[[gnu::noinline]] void ragged_product(const std::uint32_t* f, std::size_t n, const std::uint32_t* g, std::size_t m,
                                      std::uint32_t* out, std::size_t count, const ragged::Plan& p) {
    const std::size_t N = n + m - 1;
    // The words written to out: a split plan writes all of c, a truncated one count words.
    const std::size_t span = p.truncated ? count : N;
    const auto apart = [span, out](const std::uint32_t* x, std::size_t len) {
        const auto o = reinterpret_cast<std::uintptr_t>(out), b = reinterpret_cast<std::uintptr_t>(x);
        return o + span * sizeof(std::uint32_t) <= b || b + len * sizeof(std::uint32_t) <= o;
    };
    // out itself when it takes those words and is apart from f and g; else c in the region, then copied.
    const bool direct = (p.truncated || count == N) && apart(f, n) && apart(g, m);
    const std::size_t own = direct ? 0 : (N + kPadding + 15) / 16 * 16;
    const auto [roots, inverse_roots] = tables<Variant>(table_log(p));
    const Region region = take_region(sizeof(std::uint32_t) * (own + ragged::work_words(p)));
    auto* start = reinterpret_cast<std::uint32_t*>(region.start);
    std::uint32_t* c = direct ? out : start;
    if (p.truncated) truncated_into<Variant, kSquare>(f, n, g, m, c, direct ? count : N, start + own, roots, inverse_roots);
    else ragged_into<Variant, kSquare>(f, n, g, m, c, start + own, p, roots, inverse_roots);
    if (!direct) std::copy_n(c, count, out);
    give_region(region);
}

// count words of f g (f^2 if kSquare, with g = f) to out by plan p: the plain product at 2^p.lg
// (the first level reads f and g, the last writes out) or a split (ragged_product). n, m >= 1,
// count <= n + m - 1.
template <class Variant, bool kSquare>
void product(const std::uint32_t* f, std::size_t n, const std::uint32_t* g, std::size_t m, std::uint32_t* out,
             std::size_t count, const ragged::Plan& p) {
    if (p.split || p.truncated) return ragged_product<Variant, kSquare>(f, n, g, m, out, count, p);
    const auto [roots, inverse_roots] = tables<Variant>(p.lg);
    const Region region = take_region(region_bytes(p.lg));
    const std::size_t span = (std::size_t(1) << p.lg) + kPadding;
    const std::size_t at = place(f, g, out, span * sizeof(std::uint32_t), place_period(p.lg));
    auto* a = reinterpret_cast<std::uint32_t*>(region.start + at);
    run_product<Variant, kSquare>(Product{f, n, g, m, a, kSquare ? a : a + span, out, count, p.lg}, roots, inverse_roots);
    give_region(region);
}

// out[i] = (f g)[m - 1 + i] for i < count (f: m - 1 + count words read, g: m) by plan p: the plain
// middle product, a cyclic product of length 2^p.lg >= m - 1 + count, or a split or truncated one
// (ragged_middle).
template <class Variant>
void middle(const std::uint32_t* f, const std::uint32_t* g, std::size_t m, std::uint32_t* out, std::size_t count,
            const ragged::Plan& p) {
    if (p.split || p.truncated) return ragged_middle<Variant>(f, g, m, out, count, p);
    const auto [roots, inverse_roots] = tables<Variant>(p.lg);
    const Region region = take_region(region_bytes(p.lg));
    const std::size_t len = std::size_t(1) << p.lg, span = len + kPadding;
    const std::size_t at = place(f, g, out, span * sizeof(std::uint32_t), place_period(p.lg));
    auto* a = reinterpret_cast<std::uint32_t*>(region.start + at);
    std::uint32_t* b = a + span;
    // b = g x^-(m - 1) mod x^len - 1: g[m - 1] at word 0, g[0, m - 1) at the top, zeros between.
    b[0] = g[m - 1];
    std::fill(b + 1, b + len - (m - 1), 0u);
    std::copy_n(g, m - 1, b + len - (m - 1));
    Variant::multiply(Product{f, m - 1 + count, b, len, a, b, out, count, p.lg}, roots, inverse_roots);
    give_region(region);
}

}  // namespace detail

// out = f g mod x^out.size(), from and into the caller's memory (above). f, g: words < P.
template <class Variant = SelectedMultiply>
void multiply(std::span<const std::uint32_t> f, std::span<const std::uint32_t> g, std::span<std::uint32_t> out) {
    const std::size_t n = std::min(f.size(), out.size()), m = std::min(g.size(), out.size());
    const std::size_t count = n && m ? std::min(n + m - 1, out.size()) : 0;
    if (count) {
        if (detail::log_length(n, m) > kMaxLog) throw std::length_error("fps::ntt::multiply: product longer than 2^kMaxLog");
        detail::product<Variant, false>(f.data(), n, g.data(), m, out.data(), count, detail::ragged::choose_product(n + m - 1));
    }
    std::fill(out.begin() + count, out.end(), 0u);
}

// out = f^2 mod x^out.size(), from and into the caller's memory (above). f: words < P.
template <class Variant = SelectedMultiply>
void square(std::span<const std::uint32_t> f, std::span<std::uint32_t> out) {
    const std::size_t n = std::min(f.size(), out.size());
    const std::size_t count = n ? std::min(2 * n - 1, out.size()) : 0;
    if (count) {
        if (detail::log_length(n, n) > kMaxLog) throw std::length_error("fps::ntt::square: square longer than 2^kMaxLog");
        detail::product<Variant, true>(f.data(), n, f.data(), n, out.data(), count, detail::ragged::choose_product(2 * n - 1));
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
    if (int(std::bit_width(n - 1)) > kMaxLog) throw std::length_error("fps::ntt::middle_product: longer than 2^kMaxLog");
    detail::middle<Variant>(f.data(), g.data(), m, out.data(), count, detail::ragged::choose_middle(n));
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

// A term of Transform::inverse_product: the leaf products of the leaf forms a and b, or a alone;
// it takes up to kMaxTerms of them.
using Term = detail::LeafTerm;
inline constexpr int kMaxTerms = detail::kMaxTerms;

// Forward and inverse transforms of length 2^lg, kMinLog <= lg <= kMaxLog, with their tables.
class Transform {
public:
    explicit Transform(int lg)
        : lg_(checked(lg)), tables_lg_(lg), tables_(new Line[2 * lines(lg) + value_lines(lg)]), lines_(tables_.get()) {
        Native::build_table(roots(), length() / 16, false);
        Native::build_table(inverse_roots(), length() / 16, true);
        if (lg <= kValuesMaxLog) detail::build_value_rows(value_rows(), lg);
    }

    // A transform of length 2^lg on the tables of `tables` (lg <= its, else std::invalid_argument),
    // which must outlive it: the twiddle tables and the value rows are prefix compatible across
    // lengths, so one longest transform serves every shorter one (backend/product_tree keeps one
    // per thread). Its values form needs tables' length <= 2^kValuesMaxLog.
    Transform(int lg, const Transform& tables) : lg_(checked(lg)), tables_lg_(tables.tables_lg_), lines_(tables.lines_) {
        if (lg > tables.lg_) throw std::invalid_argument("fps::ntt::Transform: tables of a shorter transform");
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
    // (ValuesFor<Native>; its words are every variant's). Throws std::length_error past 2^kValuesMaxLog.
    template <class Variant = SelectedForward>
    void forward_upper(std::uint32_t* h, Form form) const {
        if (form == Form::leaves) return forward_upper<Variant>(h);
        const std::uint32_t* rows = checked_rows();
        forward_upper<Variant>(h);
        const std::size_t blocks = length() / 128;  // of 64 words in each half
        detail::intrinsics::ValuesFor<simd::Native>::split(h, h, blocks, blocks, rows);
    }

    // H (length() / 2 words) = forward_upper(h, form) of h = in, zero past it (in.size() <= length()
    // / 2): the upper half of the leaf or values form of in, read from in by the first level (no copy
    // into H). lg > kMinLog; H apart from in. The values form's split is forward_upper(h, form)'s.
    template <class Variant = SelectedForward>
    void forward_upper(std::span<const std::uint32_t> in, std::uint32_t* H, Form form) const {
        forward_upper<Variant>(in, H);
        if (form == Form::leaves) return;
        const std::size_t blocks = length() / 128;
        detail::intrinsics::ValuesFor<simd::Native>::split(H, H, blocks, blocks, checked_rows());
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
        run(reinterpret_cast<std::uint32_t*>(region.start + detail::place(F, F, out.data(), 0, detail::place_period(lg_))));
        detail::give_region(region);
    }

    // out = the first out.size() coefficients of the polynomial whose leaf or values form is F, as
    // inverse(F, out, form), through the caller's work (length() + kPadding words, 32-byte aligned):
    // the subtrees copy or join their leaves from F into work, the top level writes out. F may be
    // work (then F is not kept); out apart from F and work, any alignment, out.size() <= length().
    template <class Variant = SelectedInverse>
    void inverse(const std::uint32_t* F, std::uint32_t* work, std::span<std::uint32_t> out, Form form) const {
        if (out.size() > length()) throw std::out_of_range("fps::ntt::Transform::inverse: coefficients past the length");
        const std::uint32_t* rows = form == Form::values ? checked_rows() : nullptr;
        if (out.empty()) return;
        Variant::inverse(F, work, out.data(), out.size(), lg_, inverse_roots(), rows);
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
    // and skips the zero half of an in of at most length() / 2 words. F apart from in, or in place:
    // in at F, in.size() <= length() / 2 (F's upper half is not read).
    template <class Variant = SelectedForward>
    void forward(std::span<const std::uint32_t> in, std::uint32_t* F) const {
        Variant::forward_from(source(in, F), checked_source(in, length()), F, lg_, roots());
    }

    // The same in that form: Form::values splits each run of leaves after the bottom level, F apart
    // from in (std::length_error past 2^kValuesMaxLog).
    template <class Variant = SelectedForward>
    void forward(std::span<const std::uint32_t> in, std::uint32_t* F, Form form) const {
        if (form == Form::leaves) return forward<Variant>(in, F);
        const std::uint32_t* rows = checked_rows();
        const auto split = [rows] [[gnu::always_inline]] (std::uint32_t* leaves, std::size_t first, std::size_t count) {
            detail::intrinsics::ValuesFor<simd::Native>::split(leaves, leaves, count / 8, first / 8, rows);
        };
        Variant::forward_hooked(source(in, F), checked_source(in, length()), F, lg_, roots(), 0, split);
    }

    // V = the values form of the leaf form F, a pass over both (V may be F). std::length_error past
    // 2^kValuesMaxLog.
    void to_values(const std::uint32_t* F, std::uint32_t* V) const {
        detail::intrinsics::ValuesFor<simd::Native>::split(F, V, length() / 64, 0, checked_rows());
    }

    // F = the leaf form of in times B, leaf by leaf: forward(in, F) then multiply(F, B, F) in one
    // pass. F apart from in and B.
    template <class Variant = SelectedForward>
    void forward_product(std::span<const std::uint32_t> in, const std::uint32_t* B, std::uint32_t* F) const {
        Variant::forward_product(source(in, F), checked_source(in, length()), B, F, lg_, roots());
    }

    // H (length() / 2 words) = forward_upper of h = in, zero past it (in.size() <= length() / 2):
    // the upper half of the leaf form of in. lg > kMinLog. H apart from in.
    template <class Variant = SelectedForward>
    void forward_upper(std::span<const std::uint32_t> in, std::uint32_t* H) const {
        if (lg_ == kMinLog) throw std::length_error("fps::ntt::Transform::forward_upper: needs lg > kMinLog");
        Variant::forward_upper_from(source(in, H), checked_source(in, length() / 2), H, lg_ - 1, roots());
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
        Variant::cyclic_product(source(in, work), n, B, work, out.data(), out.size(), lg_, roots(), inverse_roots(),
                                output);
    }

    // out[i] = coefficient o.offset + i of the polynomial whose leaf form is the sum over the terms
    // of a b leaf by leaf (of a, if b is null), negated if o.negate: the leaf products in each
    // subtree of the inverse transform, in cache. 1 to 8 terms (else std::invalid_argument); work,
    // o: as cyclic_product; out apart from the terms. 5 to 8 products in one call took 0.79-0.91 of
    // the time of the split forms callers used before (leaf products in memory and 4 terms, or two
    // inverses and an add) on Zen 3, 2^12-2^20 (notes/143-r1.md).
    // Form::values: the terms are values forms (a b per word; a alone if b is null), summed and
    // joined in each run of leaves in cache before its bottom level (std::length_error past
    // 2^kValuesMaxLog).
    template <class Variant = SelectedInverse>
    void inverse_product(std::span<const Term> terms, std::uint32_t* work, std::span<std::uint32_t> out, Output o = {},
                         Form form = Form::leaves) const {
        if (terms.empty() || terms.size() > std::size_t(detail::kMaxTerms))
            throw std::invalid_argument("fps::ntt::Transform::inverse_product: 1 to 8 terms");
        const unsigned output = checked_output(out, o);
        const std::uint32_t* rows = form == Form::values ? checked_rows() : nullptr;
        if (out.empty()) return;
        if (!rows)
            return Variant::inverse_product(terms.data(), int(terms.size()), work, out.data(), out.size(), lg_, roots(),
                                            inverse_roots(), output);
        const Term* t = terms.data();
        const int n = int(terms.size());
        const auto join = [t, n, rows] [[gnu::always_inline]] (std::uint32_t* leaves, std::size_t first, std::size_t count) {
            detail::intrinsics::ValuesTerms<simd::Native>::sum(t, n, 8 * first, 8 * count, leaves);
            detail::intrinsics::ValuesFor<simd::Native>::template join<true>(leaves, leaves, count / 8, first / 8, rows);
        };
        Variant::inverse_hooked(join, work, out.data(), out.size(), lg_, roots(), inverse_roots(),
                                output | detail::kJoined, 0);
    }

    // out[i] = coefficient o.offset + i of the polynomial whose leaf form is A2 B0 + A1 B1, negated if
    // o.negate, and A3 = A3 B0 + A2 B1 leaf by leaf, by three leaf products instead of four (lc-opt's
    // Toeplitz sharing in log's blocked division): m1 = A2 (B0 + B1), m2 = (A1 - A2) B1, m3 = (A3 -
    // A2) B0; A2 B0 + A1 B1 = m1 + m2 and A3 B0 + A2 B1 = m1 + m3. A1, A2, A3, B0, B1: leaf forms;
    // work, o: as cyclic_product; out apart from the leaf forms. A3 then is a leaf form as any other
    // (a term of inverse_product; avx2 runs a product and a term alone in one asm bottom). With
    // inverse_product of A1 B2 and A3 after it (a division's blocks 2 and 3): 0.89-0.91 of the time
    // of the inverse_products of 2 and 3 products it replaces (Zen 3, 2^16-2^20, notes/87-r2.md).
    template <class Variant = SelectedInverse>
    void toeplitz_product(const std::uint32_t* A1, const std::uint32_t* A2, std::uint32_t* A3, const std::uint32_t* B0,
                          const std::uint32_t* B1, std::uint32_t* work, std::span<std::uint32_t> out, Output o = {}) const {
        const unsigned output = checked_output(out, o);
        if (out.empty()) return;
        Variant::toeplitz_product(detail::Toeplitz{A1, A2, A3, B0, B1}, work, out.data(), out.size(), lg_, roots(),
                                  inverse_roots(), output);
    }

    // Hooked ops (issue #149): the caller's pass over each run of leaves while it is in cache, and
    // pruned levels for Kronecker layouts. hook(leaves, first, count) gets the leaves [first, first +
    // count) of the leaf form at `leaves` (u32*, 8 count words, 32-byte aligned); count >= 4 a power of
    // two, first a multiple of count. A column bit (a power of two of words; 0: none) skips the half
    // of the columns above it: the levels above that bit of the index run on the words i with (i &
    // bit) == 0 only. Not a power of two: std::invalid_argument.

    // F = the leaf form of in (as forward(in, F)); hook(F + 8 first, first, count) after the bottom
    // level of each run, which may read and change those leaves (canonical); the runs come in
    // increasing order of first, so a hook may write F's words below 8 first. in[i] counts as 0 where
    // i & zero is not 0: it must be 0 there, except that with 32 <= zero < length() / 4 those words
    // are not used (detail::kUnreadWidth) and may hold anything. F apart from in, or in place (in at
    // F, any in.size(); issue #210).
    template <class Variant = SelectedForward, class Hook>
    void forward_hooked(std::span<const std::uint32_t> in, std::uint32_t* F, Hook hook, std::size_t zero = 0) const {
        Variant::forward_hooked(source(in, F), checked_source(in, length()), F, lg_, roots(), checked_bit(zero), hook);
    }

    // out[i] = coefficient o.offset + i of the polynomial whose leaf form times 2^-32 (as Montgomery
    // products leave it) the hook writes: hook(work + 8 first, first, count) writes those leaves,
    // canonical, then the inverse levels of the run. out[i] is unspecified where (o.offset + i) &
    // unused is not 0. work, o: as cyclic_product. out is written after the hook's last call, so the
    // hook may read out's memory (composition: P's leaf form in the array that gets R; issue #210).
    template <class Variant = SelectedInverse, class Hook>
    void inverse_hooked(Hook hook, std::uint32_t* work, std::span<std::uint32_t> out, Output o = {},
                        std::size_t unused = 0) const {
        const unsigned output = checked_output(out, o);
        const std::size_t width = checked_bit(unused);
        if (out.empty()) return;
        Variant::inverse_hooked(hook, work, out.data(), out.size(), lg_, roots(), inverse_roots(), output, width);
    }

    // Doubling for Kronecker layouts (issue #149 round 2, lc-opt's Doubling): block j of `block`
    // words of a leaf form of f (block a power of two, 32 <= block <= length() / 8) is the leaf
    // form of f mod (x^block - r(j)^2) by the levels within the block, the same in every length (the
    // lower half of a leaf form of length 2L is the one of length L). A layout of rows of `block`
    // words, x below block / 2, doubles: its lower half by truncate_blocks, its upper half by
    // join_blocks and forward_upper_hooked, without the levels above the blocks of the lower half.

    // For each block j of the leaf form F: c_j = (f mod (x^block - r(j)^2)) mod x^(block / 2), times
    // block / 8; copy[j block + i] = c_j's coefficient i for i < block / 2 (canonical; copy's other
    // words unspecified) and F's block becomes c_j's leaf form. hook(F + 8 first, first, count) on
    // runs of the new leaves, as forward_hooked's. copy: length() words, apart from F.
    template <class Variant = SelectedForward, class Hook>
    void truncate_blocks(std::uint32_t* F, std::size_t block, std::uint32_t* copy, Hook hook) const {
        Variant::truncate_blocks(F, checked_block(block), copy, lg_, roots(), inverse_roots(), hook);
    }

    // copy as truncate_blocks writes it (block j's first block / 2 words: a polynomial c_j) -> in
    // place, the polynomial c of degree below length() with c mod (x^block - r(j)^2) = c_j for every
    // j and coefficient i zero where (i mod block) >= block / 2, times scale: copy[i] for (i mod
    // block) < block / 2, canonical; the other words unspecified.
    template <class Variant = SelectedInverse>
    void join_blocks(std::uint32_t* copy, std::size_t block, std::uint32_t scale) const {
        Variant::join_blocks(copy, checked_block(block), scale, lg_, inverse_roots());
    }

    // forward_upper(H) with a hook and a column bit: H (length() / 2 words) = p mod (x^(L/2) + 1) ->
    // the upper half of p's leaf form; hook(H + 8 i, length() / 16 + i, count) on runs of its
    // leaves (their indices in the whole leaf form). H[i] counts as 0 where i & zero is not 0, and is
    // not used there when 16 <= zero < length() / 8 (the first level skips those words, or zeroes
    // them where a whole group would read them). lg > kMinLog.
    template <class Variant = SelectedForward, class Hook>
    void forward_upper_hooked(std::uint32_t* H, Hook hook, std::size_t zero = 0) const {
        if (lg_ == kMinLog) throw std::length_error("fps::ntt::Transform::forward_upper_hooked: needs lg > kMinLog");
        Variant::forward_upper_hooked(H, lg_ - 1, roots(), checked_bit(zero), hook);
    }

private:
    std::size_t checked_block(std::size_t block) const {
        if (block < 32 || block > length() / 8 || (block & (block - 1)))
            throw std::invalid_argument("fps::ntt::Transform: block not a power of two in [32, length() / 8]");
        return block;
    }
    static std::size_t checked_bit(std::size_t bit) {
        if (bit & (bit - 1)) throw std::invalid_argument("fps::ntt::Transform: column bit not a power of two");
        return bit;
    }
    // An empty source's words: at the array written (any pointer reads none; an empty span's may be
    // null, and the variants form pointers from it).
    static const std::uint32_t* source(std::span<const std::uint32_t> in, const std::uint32_t* at) {
        return in.empty() ? at : in.data();
    }
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

    // The tables of length 2^tables_lg_ at lines_: owned (tables_) or another transform's.
    std::uint32_t* roots() const { return lines_[0].words; }
    std::uint32_t* inverse_roots() const { return lines_[lines(tables_lg_)].words; }
    std::uint32_t* value_rows() const { return lines_[2 * lines(tables_lg_)].words; }
    const std::uint32_t* checked_rows() const {
        if (tables_lg_ > kValuesMaxLog) throw std::length_error("fps::ntt::Transform: the values form needs lg <= 23");
        return value_rows();
    }

    int lg_, tables_lg_;
    std::unique_ptr<Line[]> tables_;  // null if on another transform's tables
    Line* lines_;
};

}  // namespace fps::ntt
