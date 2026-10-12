// Polynomials in value form: fps::Values<P>. A product of two value forms is value by value, O(L),
// so a factor used in many products is converted once; a sum of products needs one conversion back.
//
//   auto B = fps::to_values(b, L);                  // L a power of two, b.size() <= L
//   Poly c = fps::to_coefficients(fps::to_values(a, L) * B);   // a b mod (x^L - 1): L coefficients
//   fps::to_values(b, L, fps::Layout::pointwise)    // the value-by-value layout
//   fps::Values<P> Z(L, layout), E;                 // zero; empty (L = 0, for containers)
//   A.size()                                        // L
//   A + B, A - B, A * B, c * A, A * c               // also +=, -=, *=; c an integer, taken mod P
//   A.multiply_add(B, C)                            // A += B C; A may be B or C
//   fps::raw::to_values<P>(f, A)                    // into an existing A, f.size() <= A.size()
//   fps::raw::to_coefficients<P>(A, out, offset)    // out[i] = coefficient offset + i
//   fps::raw::add_coefficients<P>(A, out, offset)   // out[i] += coefficient offset + i
//
// to_coefficients(A * B) is a b mod (x^L - 1): coefficient L + i adds onto i. For the exact
// product, L > deg a + deg b. The words are opaque. Layouts (DESIGN.md, Values):
// - automatic: the backend's leaf form (backend/ntt/ntt.hpp): residues mod x^8 - w_v, 8 words
//   each; a product is an 8 x 8 product per leaf.
// - pointwise: each leaf taken on to its values at the 8 roots of x^8 - w_v, so a product is one
//   modular product per word. Lengths above 2^23 have no such roots (P - 1 = 119 2^23) and keep
//   the leaf form.
// Lengths L < 64 are held at length 64, below the backend's shortest transform: x^L - 1 divides
// x^64 - 1, so every operation is exact mod x^64 - 1 and the conversion back folds mod x^L - 1.
//
// Errors: L not a power of two, f.size() > L, operands of different L or layout, and a raw range
// past L throw std::domain_error; L > 2^25 (the backend's longest transform) throws
// std::length_error. Coefficients for raw::to_values are unchecked: they must be below P.
//
// Memory: a Values holds max(L, 64) + 16 words, 64-byte aligned. The tables of each length are
// built on first use and kept: ntt::Transform's (about L / 4 words) and the leaf roots (up to
// 5 L / 16 words). raw::to_coefficients and raw::add_coefficients transform a copy of A in a
// per-thread scratch, allocated at a thread's longest L and then reused; the raw functions
// allocate nothing else.
// out never overlaps A, whose words are private.
//
// Cost against the backend (frontend/notes.md): to_values and raw::add_coefficients run as fast as
// ntt::Transform by hand. to_coefficients(A) copies A into the scratch, runs the inverse there and
// copies the coefficients out: one copy more than the kernel, 5% to 19% on Zen 3 from 2^10 to
// 2^22, as the backend has no inverse into other memory (backend issue #49). An rvalue A, as in
// to_coefficients(A * B), is transformed in its own memory: no copy in. The leaf products and the
// pointwise layout's conversions have no backend kernel yet (#49): they are plain scalar loops
// here, about 7 ns per coefficient on Zen 3 for A * B in the automatic layout (5 forward
// transforms) and 8 to 10 ns for each pointwise conversion. Products in the pointwise layout are a
// modarith stream, about 0.2 ns per coefficient.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "backend/modarith/modarith.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/simd/select.hpp"
#include "fps/modular.hpp"
#include "fps/polynomial.hpp"

namespace fps {

enum class Layout { automatic, pointwise };

namespace detail::values {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
inline constexpr u32 kP = ntt::kModulus;

inline constexpr int kMinLog = ntt::kMinLog;   // shorter L are held at 2^kMinLog
inline constexpr int kSplitMaxLog = 23;        // the pointwise layout needs 2^lg | P - 1

// 64-byte aligned words.
struct AlignedDelete {
    void operator()(u32* p) const { ::operator delete(p, std::align_val_t(64)); }
};
using Buffer = std::unique_ptr<u32[], AlignedDelete>;

// n words and ntt::kPadding more, zero (the transforms read the padding, never write it).
inline Buffer allocate(std::size_t n) {
    Buffer b(static_cast<u32*>(::operator new((n + ntt::kPadding) * sizeof(u32), std::align_val_t(64))));
    std::fill_n(b.get() + n, ntt::kPadding, 0u);
    return b;
}

// The internal log length for L: max(kMinLog, log2 L).
inline int log_length(std::size_t L) {
    if (!std::has_single_bit(L)) throw std::domain_error("fps::Values: L is not a power of two");
    if (L > (std::size_t(1) << ntt::kMaxLog)) throw std::length_error("fps::Values: L > 2^25");
    return std::max(kMinLog, int(std::countr_zero(L)));
}

// Roots of unity of order 2^e, e <= 23, and their inverses; 3 generates the multiplicative group.
struct Roots {
    std::array<u32, kSplitMaxLog + 1> root, inverse;
};
inline constexpr Roots kRoots = [] {
    Roots t{};
    for (int e = 0; e <= kSplitMaxLog; ++e) {
        t.root[e] = mod_pow<kP>(3, (kP - 1) >> e);
        t.inverse[e] = mod_inverse<kP>(t.root[e]);
    }
    return t;
}();

// The tables for transforms of length 2^lg, built once per lg and then only read.
// Leaf v is mod x^8 - w_v, w_v = (-1)^v r(v / 2), r(k) the product of the roots of order 2^(j + 2)
// over the set bits j of k (backend/ntt/ntt.hpp, "Leaf form"). rho_v is an 8th root of w_v: the
// same product with the roots of order 2^(j + 5), times one of order 16 (an 8th root of -1) for odd v.
struct Plan {
    explicit Plan(int lg) : transform(lg), root(leaves(lg) / 2) {
        const Roots& r = kRoots;
        root[0] = 1;
        for (std::size_t k = 1; k < root.size(); ++k)
            root[k] = mod_mul<kP>(root[k & (k - 1)], r.root[std::countr_zero(k) + 2]);
        if (lg > kSplitMaxLog) return;
        rho.resize(leaves(lg)), inverse_rho.resize(leaves(lg));
        rho[0] = inverse_rho[0] = 1;
        for (std::size_t k = 1; k < root.size(); ++k) {
            const int j = std::countr_zero(k) + 5;
            rho[2 * k] = mod_mul<kP>(rho[2 * (k & (k - 1))], r.root[j]);
            inverse_rho[2 * k] = mod_mul<kP>(inverse_rho[2 * (k & (k - 1))], r.inverse[j]);
        }
        for (std::size_t k = 0; k < root.size(); ++k) {
            rho[2 * k + 1] = mod_mul<kP>(rho[2 * k], r.root[4]);
            inverse_rho[2 * k + 1] = mod_mul<kP>(inverse_rho[2 * k], r.inverse[4]);
        }
    }

    static std::size_t leaves(int lg) { return (std::size_t(1) << lg) / 8; }
    u32 leaf_modulus(std::size_t v) const { return v % 2 ? kP - root[v / 2] : root[v / 2]; }

    ntt::Transform transform;
    std::vector<u32> root;              // r(k), k < leaves / 2
    std::vector<u32> rho, inverse_rho;  // per leaf; empty above kSplitMaxLog
};

inline const Plan& plan(int lg) {
    static std::array<std::once_flag, ntt::kMaxLog + 1> once;
    static std::array<std::unique_ptr<const Plan>, ntt::kMaxLog + 1> plans;
    std::call_once(once[std::size_t(lg)], [lg] { plans[std::size_t(lg)] = std::make_unique<const Plan>(lg); });
    return *plans[std::size_t(lg)];
}

// A per-thread array of at least n words, 64-byte aligned and padded.
inline u32* scratch(std::size_t n) {
    thread_local Buffer buffer;
    thread_local std::size_t capacity = 0;
    if (n > capacity) buffer = allocate(n), capacity = n;
    return buffer.get();
}

// The streams on n words, on simd::Native by modarith; out may be a or b.
inline void add_words(const u32* a, const u32* b, u32* out, std::size_t n) {
    lanewise(a, b, out, n, []<class S>(typename S::V x, typename S::V y) { return modarith::Ops<S, kP>::add(x, y); });
}

inline void subtract_words(const u32* a, const u32* b, u32* out, std::size_t n) {
    lanewise(a, b, out, n, []<class S>(typename S::V x, typename S::V y) { return modarith::Ops<S, kP>::sub(x, y); });
}

// out[i] = c a[i] by Shoup's multiplication.
inline void scale_words(const u32* a, u32 c, u32* out, std::size_t n) {
    const u32 q = modarith::Ops<simd::Scalar, kP>::shoup_quotient(c);
    lanewise(a, a, out, n, [c, q]<class S>(typename S::V x, typename S::V) {
        using M = modarith::Ops<S, kP>;
        return M::template reduce<kP>(M::shoup(x, S::broadcast(c), S::broadcast(q)));
    });
}

inline u32 add(u32 x, u32 y) {
    const u32 s = x + y;  // < 2P < 2^31
    return s >= kP ? s - kP : s;
}
inline u32 sub(u32 x, u32 y) { return x >= y ? x - y : x + kP - y; }

// The 8th roots of unity z^k and their inverses, k < 4, and 1/8.
struct Eighths {
    std::array<u32, 4> z, inverse_z;
    u32 inverse_eight;
};
inline constexpr Eighths kEighths = [] {
    Eighths t{};
    t.z[0] = t.inverse_z[0] = 1;
    for (int k = 1; k < 4; ++k) {
        t.z[k] = mod_mul<kP>(t.z[k - 1], kRoots.root[3]);
        t.inverse_z[k] = mod_mul<kP>(t.inverse_z[k - 1], kRoots.inverse[3]);
    }
    t.inverse_eight = mod_inverse<kP>(8);
    return t;
}();

// c mod x^8 - rho^8 -> its values at rho z^i, i < 8, in bit-reversed order of i: the twist by
// rho^k, then an 8-point transform (decimation in frequency).
inline void split_leaf(u32* c, u32 rho) {
    const Eighths& e = kEighths;
    u32 power = rho;
    for (int k = 1; k < 8; ++k) c[k] = mod_mul<kP>(c[k], power), power = mod_mul<kP>(power, rho);
    for (int half = 4; half >= 1; half /= 2)
        for (int b = 0; b < 8; b += 2 * half)
            for (int k = 0; k < half; ++k) {
                const u32 x = c[b + k], y = c[b + k + half];
                c[b + k] = add(x, y), c[b + k + half] = mod_mul<kP>(sub(x, y), e.z[k * 4 / half]);
            }
}

// split_leaf undone; inverse_rho = 1 / rho.
inline void join_leaf(u32* c, u32 inverse_rho) {
    const Eighths& e = kEighths;
    for (int half = 1; half <= 4; half *= 2)
        for (int b = 0; b < 8; b += 2 * half)
            for (int k = 0; k < half; ++k) {
                const u32 x = c[b + k], y = mod_mul<kP>(c[b + k + half], e.inverse_z[k * 4 / half]);
                c[b + k] = add(x, y), c[b + k + half] = sub(x, y);
            }
    u32 power = e.inverse_eight;  // the 8-point transform's gain
    for (int k = 0; k < 8; ++k) c[k] = mod_mul<kP>(c[k], power), power = mod_mul<kP>(power, inverse_rho);
}

// out = a b mod x^8 - w, 8 words each; out may be a or b.
inline void leaf_product(const u32* a, const u32* b, u32 w, u32* out) {
    u32 c[8];
    for (int l = 0; l < 8; ++l) {
        u64 low = 0, high = 0;  // each < 8 P^2 < 2^63
        for (int i = 0; i <= l; ++i) low += u64(a[i]) * b[l - i];
        for (int i = l + 1; i < 8; ++i) high += u64(a[i]) * b[l + 8 - i];
        c[l] = u32((low % kP + high % kP * w) % kP);  // < P + P^2 < 2^64 before the last %
    }
    std::copy_n(c, 8, out);
}

// The form of 2^lg canonical coefficients in w, in place; split: the pointwise layout.
inline void forward(u32* w, int lg, bool split) {
    const Plan& p = plan(lg);
    p.transform.forward(w);
    if (split)
        for (std::size_t v = 0; v < Plan::leaves(lg); ++v) split_leaf(w + 8 * v, p.rho[v]);
}

inline void inverse(u32* w, int lg, bool split) {
    const Plan& p = plan(lg);
    if (split)
        for (std::size_t v = 0; v < Plan::leaves(lg); ++v) join_leaf(w + 8 * v, p.inverse_rho[v]);
    p.transform.inverse(w);
}

// a b mod P for a, b < P, canonical, by Barrett's reduction.
template <class S>
typename S::V product(typename S::V a, typename S::V b) {
    using M = modarith::Ops<S, kP>;
    const typename S::V r = M::barrett(a, b);  // < kBarrettBound, 2P or 3P
    if constexpr (M::kBarrettBound == 2 * kP) return M::template reduce<kP>(r);
    else return M::template reduce<kP>(M::template reduce<2 * kP>(r));
}

// out = a b, value by value; out may be a or b. 2^lg >= 64 is a multiple of the vector lanes.
inline void multiply(const u32* a, const u32* b, u32* out, int lg, bool split) {
    using S = simd::Native;
    const std::size_t n = std::size_t(1) << lg;
    if (split) {
        for (std::size_t i = 0; i < n; i += S::lanes) S::store(out + i, product<S>(S::load(a + i), S::load(b + i)));
        return;
    }
    const Plan& p = plan(lg);
    for (std::size_t v = 0; v < n / 8; ++v) leaf_product(a + 8 * v, b + 8 * v, p.leaf_modulus(v), out + 8 * v);
}

// acc += a b; acc may be a or b.
inline void multiply_add(u32* acc, const u32* a, const u32* b, int lg, bool split) {
    using S = simd::Native;
    const std::size_t n = std::size_t(1) << lg;
    if (split) {
        for (std::size_t i = 0; i < n; i += S::lanes)
            S::store(acc + i, modarith::Ops<S, kP>::add(S::load(acc + i), product<S>(S::load(a + i), S::load(b + i))));
        return;
    }
    const Plan& p = plan(lg);
    u32 c[8];
    for (std::size_t v = 0; v < n / 8; ++v) {
        leaf_product(a + 8 * v, b + 8 * v, p.leaf_modulus(v), c);
        for (int l = 0; l < 8; ++l) acc[8 * v + l] = add(acc[8 * v + l], c[l]);
    }
}

// Coefficients [offset, offset + count) of the 2^lg coefficients in s, mod x^L - 1, L <= 2^lg.
// count words to out; add: out[i] += instead.
inline void read_coefficients(const u32* s, int lg, std::size_t L, u32* out, std::size_t offset, std::size_t count,
                              bool add_to) {
    const std::size_t n = std::size_t(1) << lg;
    if (L == n) {
        if (add_to) add_words(out, s + offset, out, count);
        else std::copy_n(s + offset, count, out);
        return;
    }
    for (std::size_t i = 0; i < count; ++i) {
        u32 c = 0;
        for (std::size_t j = offset + i; j < n; j += L) c = add(c, s[j]);
        out[i] = add_to ? add(out[i], c) : c;
    }
}

}  // namespace detail::values

template <std::uint32_t P>
class Values;

namespace raw {
template <std::uint32_t P>
void to_values(std::span<const std::uint32_t> f, Values<P>& A);
template <std::uint32_t P>
void to_coefficients(const Values<P>& A, std::span<std::uint32_t> out, std::size_t offset);
template <std::uint32_t P>
void add_coefficients(const Values<P>& A, std::span<std::uint32_t> out, std::size_t offset);
}  // namespace raw

template <std::uint32_t P>
Values<P> to_values(const Polynomial<P>& f, std::size_t L, Layout layout = Layout::automatic);
template <std::uint32_t P>
Polynomial<P> to_coefficients(Values<P>&& A);

template <std::uint32_t P>
class Values {
    static_assert(P == ntt::kModulus, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");

public:
    Values() = default;
    explicit Values(std::size_t L, Layout layout = Layout::automatic) : Values(checked(L), layout, kUnwritten) {
        std::fill_n(data(), words(), 0u);
    }
    Values(const Values& A) : Values(A.size_, A.layout_, kUnwritten) { std::copy_n(A.data(), words(), data()); }
    Values(Values&& A) noexcept
        : size_(std::exchange(A.size_, 0)), lg_(A.lg_), layout_(A.layout_), words_(std::move(A.words_)) {}
    Values& operator=(const Values& A) { return *this = Values(A); }
    Values& operator=(Values&& A) noexcept {
        size_ = std::exchange(A.size_, 0), lg_ = A.lg_, layout_ = A.layout_, words_ = std::move(A.words_);
        return *this;
    }

    std::size_t size() const { return size_; }

    Values& operator+=(const Values& B) {
        check(B);
        detail::values::add_words(data(), B.data(), data(), words());
        return *this;
    }

    Values& operator-=(const Values& B) {
        check(B);
        detail::values::subtract_words(data(), B.data(), data(), words());
        return *this;
    }

    Values& operator*=(const Values& B) {
        check(B);
        if (size_) detail::values::multiply(data(), B.data(), data(), lg_, split());
        return *this;
    }

    template <detail::Integer T>
    Values& operator*=(T c) {
        detail::values::scale_words(data(), detail::residue<P>(c), data(), words());
        return *this;
    }

    // *this += a b.
    void multiply_add(const Values& a, const Values& b) {
        check(a), check(b);
        if (size_) detail::values::multiply_add(data(), a.data(), b.data(), lg_, split());
    }

    // Each operator writes its result in one pass into new memory; an rvalue left operand works in place.
    friend Values operator+(const Values& A, const Values& B) {
        A.check(B);
        Values C(A.size_, A.layout_, kUnwritten);
        detail::values::add_words(A.data(), B.data(), C.data(), C.words());
        return C;
    }
    friend Values operator+(Values&& A, const Values& B) { return std::move(A += B); }

    friend Values operator-(const Values& A, const Values& B) {
        A.check(B);
        Values C(A.size_, A.layout_, kUnwritten);
        detail::values::subtract_words(A.data(), B.data(), C.data(), C.words());
        return C;
    }
    friend Values operator-(Values&& A, const Values& B) { return std::move(A -= B); }

    friend Values operator*(const Values& A, const Values& B) {
        A.check(B);
        Values C(A.size_, A.layout_, kUnwritten);
        if (C.size_) detail::values::multiply(A.data(), B.data(), C.data(), C.lg_, C.split());
        return C;
    }
    friend Values operator*(Values&& A, const Values& B) { return std::move(A *= B); }

    template <detail::Integer T>
    friend Values operator*(T c, const Values& A) {
        Values C(A.size_, A.layout_, kUnwritten);
        detail::values::scale_words(A.data(), detail::residue<P>(c), C.data(), C.words());
        return C;
    }
    template <detail::Integer T>
    friend Values operator*(const Values& A, T c) {
        return c * A;
    }

private:
    struct Unwritten {};
    static constexpr Unwritten kUnwritten{};

    static std::size_t checked(std::size_t L) {
        detail::values::log_length(L);
        return L;
    }

    // L = 0 only for the result of an operation on empty operands.
    Values(std::size_t L, Layout layout, Unwritten)
        : size_(L), lg_(L ? detail::values::log_length(L) : 0), layout_(layout),
          words_(L ? detail::values::allocate(words()) : nullptr) {}

    // The words held: 2^lg_, or 0 if empty.
    std::size_t words() const { return size_ ? std::size_t(1) << lg_ : 0; }
    std::uint32_t* data() { return words_.get(); }
    const std::uint32_t* data() const { return words_.get(); }
    bool split() const { return layout_ == Layout::pointwise && lg_ <= detail::values::kSplitMaxLog; }

    void check(const Values& B) const {
        if (B.size_ != size_ || B.layout_ != layout_)
            throw std::domain_error("fps::Values: operands differ in L or layout");
    }

    // Coefficients [offset, offset + out.size()) of the polynomial in s, which holds A's words
    // transformed back.
    void read(const std::uint32_t* s, std::span<std::uint32_t> out, std::size_t offset, bool add_to) const {
        detail::values::read_coefficients(s, lg_, size_, out.data(), offset, out.size(), add_to);
    }

    void read_copy(std::span<std::uint32_t> out, std::size_t offset, bool add_to) const {
        if (offset > size_ || out.size() > size_ - offset)
            throw std::domain_error("fps::raw: coefficient range past L");
        if (out.empty()) return;
        std::uint32_t* s = detail::values::scratch(words());
        std::copy_n(data(), words(), s);
        detail::values::inverse(s, lg_, split());
        read(s, out, offset, add_to);
    }

    friend void raw::to_values<P>(std::span<const std::uint32_t> f, Values& A);
    friend void raw::to_coefficients<P>(const Values& A, std::span<std::uint32_t> out, std::size_t offset);
    friend void raw::add_coefficients<P>(const Values& A, std::span<std::uint32_t> out, std::size_t offset);
    friend Values to_values<P>(const Polynomial<P>& f, std::size_t L, Layout layout);
    friend Polynomial<P> to_coefficients<P>(Values&& A);

    std::size_t size_ = 0;  // L
    int lg_ = 0;            // log2 of the words held: max(6, log2 L)
    Layout layout_ = Layout::automatic;
    detail::values::Buffer words_;
};

namespace raw {

// A = the value form of f, f.size() <= A.size(); f canonical (unchecked).
template <std::uint32_t P>
void to_values(std::span<const std::uint32_t> f, Values<P>& A) {
    if (f.size() > A.size()) throw std::domain_error("fps::raw::to_values: f.size() > L");
    if (A.size() == 0) return;
    std::copy(f.begin(), f.end(), A.data());
    std::fill(A.data() + f.size(), A.data() + A.words(), 0u);
    detail::values::forward(A.data(), A.lg_, A.split());
}

// out[i] = coefficient offset + i of A's polynomial mod (x^L - 1), offset + out.size() <= L.
template <std::uint32_t P>
void to_coefficients(const Values<P>& A, std::span<std::uint32_t> out, std::size_t offset) {
    A.read_copy(out, offset, false);
}

// out[i] += coefficient offset + i, offset + out.size() <= L; out canonical.
template <std::uint32_t P>
void add_coefficients(const Values<P>& A, std::span<std::uint32_t> out, std::size_t offset) {
    A.read_copy(out, offset, true);
}

}  // namespace raw

// The value form of f at length L, f.size() <= L.
template <std::uint32_t P>
Values<P> to_values(const Polynomial<P>& f, std::size_t L, Layout layout) {
    if (f.size() > L) throw std::domain_error("fps::to_values: f.size() > L");
    Values<P> A(Values<P>::checked(L), layout, Values<P>::kUnwritten);
    raw::to_values<P>(f.span(), A);
    return A;
}

// The L coefficients of A's polynomial mod (x^L - 1).
template <std::uint32_t P>
Polynomial<P> to_coefficients(const Values<P>& A) {
    Polynomial<P> f = detail::unwritten_polynomial<P>(A.size());
    raw::to_coefficients<P>(A, f.span(), 0);
    return f;
}

// The same, transforming A's own words: A is left empty.
template <std::uint32_t P>
Polynomial<P> to_coefficients(Values<P>&& A) {
    Values<P> B = std::move(A);
    Polynomial<P> f = detail::unwritten_polynomial<P>(B.size());
    if (B.size() == 0) return f;
    detail::values::inverse(B.data(), B.lg_, B.split());
    B.read(B.data(), f.span(), 0, false);
    return f;
}

}  // namespace fps
