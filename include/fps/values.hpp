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
// built on first use and kept: ntt::Transform's (about 5 L / 16 words with the values form's).
// to_coefficients and the raw functions run the backend's inverse into other memory, which works
// in the thread's NTT region (kept per thread, ntt.hpp); for L < 64 a per-thread scratch of 64
// words takes the coefficients before they are folded. Nothing else is allocated.
// out never overlaps A, whose words are private.
//
// Cost: every operation but the pointwise products is one backend call (ntt::Transform: forward,
// inverse into other memory, the leaf products, the values form; frontend/notes.md has the
// numbers). Products in the pointwise layout are a modarith stream here.
// In a program that also runs f * g on avx2, GCC 15.2 compiles the NTT product differently: f * g
// of 32 x 32 coefficients ran 25% slower on Zen 3, 64 x 64 4% (backend issue #81).
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>

#include "backend/modarith/modarith.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/simd/select.hpp"
#include "fps/modular.hpp"
#include "fps/polynomial.hpp"

namespace fps {

enum class Layout { automatic, pointwise };

namespace detail::values {

using u32 = std::uint32_t;
inline constexpr u32 kP = ntt::kModulus;

inline constexpr int kMinLog = ntt::kMinLog;   // shorter L are held at 2^kMinLog
inline constexpr int kSplitMaxLog = ntt::kValuesMaxLog;  // the pointwise layout needs 2^lg | P - 1

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

// The transform of each length (backend/ntt/ntt.hpp), built once and then only read.
inline const ntt::Transform& transform(int lg) {
    static std::array<std::once_flag, ntt::kMaxLog + 1> once;
    static std::array<std::unique_ptr<const ntt::Transform>, ntt::kMaxLog + 1> transforms;
    std::call_once(once[std::size_t(lg)], [lg] { transforms[std::size_t(lg)] = std::make_unique<const ntt::Transform>(lg); });
    return *transforms[std::size_t(lg)];
}

// The backend's form for a layout: split, the values form; else the leaf form.
inline ntt::Form form(bool split) { return split ? ntt::Form::values : ntt::Form::leaves; }

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

// The form of 2^lg canonical coefficients in w, in place; split: the pointwise layout.
inline void forward(u32* w, int lg, bool split) {
    if (split) return transform(lg).forward(w, ntt::Form::values);
    transform(lg).forward(w);
}

// a b mod P for a, b < P, canonical, by Barrett's reduction.
template <class S>
typename S::V product(typename S::V a, typename S::V b) {
    using M = modarith::Ops<S, kP>;
    const typename S::V r = M::barrett(a, b);  // < kBarrettBound, 2P or 3P
    if constexpr (M::kBarrettBound == 2 * kP) return M::template reduce<kP>(r);
    else return M::template reduce<kP>(M::template reduce<2 * kP>(r));
}

// out = a b, value by value (pointwise) or leaf by leaf (the backend's leaf products); out may be
// a or b. 2^lg >= 64 is a multiple of the vector lanes.
inline void multiply(const u32* a, const u32* b, u32* out, int lg, bool split) {
    using S = simd::Native;
    const std::size_t n = std::size_t(1) << lg;
    if (split) {
        for (std::size_t i = 0; i < n; i += S::lanes) S::store(out + i, product<S>(S::load(a + i), S::load(b + i)));
        return;
    }
    transform(lg).multiply(a, b, out);
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
    transform(lg).multiply_add(acc, a, b);
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
    ntt::Form form() const { return detail::values::form(split()); }

    void check(const Values& B) const {
        if (B.size_ != size_ || B.layout_ != layout_)
            throw std::domain_error("fps::Values: operands differ in L or layout");
    }

    // Coefficients [offset, offset + out.size()) of the polynomial in s, which holds A's words
    // transformed back.
    void read(const std::uint32_t* s, std::span<std::uint32_t> out, std::size_t offset, bool add_to) const {
        detail::values::read_coefficients(s, lg_, size_, out.data(), offset, out.size(), add_to);
    }

    // Coefficients [offset, offset + out.size()) into out (or added to it): the backend's inverse
    // into other memory, A kept. L < 64: the 64 coefficients into the scratch, then folded.
    void read_copy(std::span<std::uint32_t> out, std::size_t offset, bool add_to) const {
        if (offset > size_ || out.size() > size_ - offset)
            throw std::domain_error("fps::raw: coefficient range past L");
        if (out.empty()) return;
        const ntt::Transform& t = detail::values::transform(lg_);
        if (size_ == words()) return t.inverse(data(), out, form(), {offset, add_to});
        std::uint32_t* s = detail::values::scratch(words());
        t.inverse(data(), std::span(s, words()), form());
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

// The same; A is left empty (its memory freed).
template <std::uint32_t P>
Polynomial<P> to_coefficients(Values<P>&& A) {
    const Values<P> B = std::move(A);
    return to_coefficients(B);
}

}  // namespace fps
