// Products modulo a monic g of degree m by NTT (backend/ntt), for large m (in place of the reduction
// table and the multiplication matrices where poly.hpp's ntt_products() says so):
//
//   Products P(g, m);           // g: m + 1 canonical words, g[m] = 1; m >= 2
//   P.multiply(a, b, out);      // out[0, m) = a b mod g; a, b: m words; out may be a or b
//   P.square(a, out);           // a^2 mod g
//   Multiplier X(P, r);         // r: m canonical words (deg r < m)
//   X.step(q, out);             // out[0, m) = q r mod g; out may be q
//   Powers Y(P, r);             // r^1 .. r^8 mod g and their quotients
//   Y.steps(q, out, stride, last);  // out + (t - 1) stride = q r^t mod g for t = 1 .. 7, last = q r^8;
//                               // apart from q
//
// The words are multiplied as given: Montgomery a and b (x 2^32) give (a b mod g) 2^32 too high (the
// caller takes it out), a canonical r keeps q's form (so the rows q, q r, q r^2, ... of the
// Berlekamp matrix stay Montgomery).
//
// Lengths L = 2 H, H = bit_ceil(m) >= m (at least 2^kMinLog): products of 2m - 1 words at L,
// remainders (degree < m <= H) at H. Barrett for polynomials (exact): for deg A <= 2m - 1 and mu =
// floor(x^(2m-1) / g), floor(A / g) = floor(A1 mu / x^(m-1)), A1 = A div x^m: with x^(2m-1) = mu g
// + rho, A / g = A1 mu / x^(m-1) + (terms of negative degree). So a quotient is one cyclic product at
// L of A1 by x^(H-m+1) mu, its words [H, H + m): no wrap, as the degree stays below L. Then A mod g
// = A - floor(A / g) g exactly, of degree < m <= H: the inverse at H of A's leaf form there (the
// first half of its leaf form at L) minus the quotient's times g's.
// - multiply: p = a b at L, its words [m, 2m - 1) are A1, then the quotient and the remainder.
// - step: Shoup's precomputed quotient for a fixed factor, for polynomials: floor(q r / g) =
//   floor(q r' / x^m) for r' = floor(r x^m / g) (r' from mu as above, one cyclic product), by the
//   same argument. One cyclic product at L of q by x^(H-m) r', then q r - floor(q r / g) g at H:
//   about 3.5 transforms of length L per step, against m^2 products by the multiplication matrix.
// - steps: q's transform at L once for 8 factors r^t: per product the quotient by one
//   inverse_product at L (q's leaf form times r'_t's), the remainder at H: about 2.1 transforms of
//   length L per row. The 8 factors' setup is about 30 such (7 products, 8 cyclic products).
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <vector>

#include "backend/division/division.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/series/series.hpp"

namespace fps::factorization::detail {

// 64-byte aligned words, freed with it.
class Words {
public:
    explicit Words(std::size_t n)
        : p_(static_cast<std::uint32_t*>(::operator new[](n * sizeof(std::uint32_t), std::align_val_t(64)))) {
        std::fill(p_, p_ + n, 0u);
    }
    Words(const Words&) = delete;
    Words& operator=(const Words&) = delete;
    ~Words() { ::operator delete[](p_, std::align_val_t(64)); }
    std::uint32_t* data() const { return p_; }

private:
    std::uint32_t* p_;
};

class Products {
public:
    Products(const std::uint32_t* g, int m)
        : m_(m), lg_(std::max(ntt::kMinLog, int(std::bit_width(unsigned(m) - 1)))), H_(std::size_t(1) << lg_),
          big_(series::detail::transform(lg_ + 1)), half_(series::detail::transform(lg_)),
          memory_(7 * span(2 * H_)), MU_(at(0)), NG_(at(1)), FA_(at(2)), FB_(at(3)), work_(at(4)), QH_(at(5)),
          words_(at(6)) {
        const std::size_t n = std::size_t(m);
        // mu = floor(x^(2m-1) / g), m words, to x^(H-m+1) mu at L.
        std::vector<std::uint32_t> f(2 * n, 0u), mu(n);
        f.back() = 1;
        division::divide(f, std::span(g, n + 1), mu, {});
        std::fill(words_, words_ + 2 * H_, 0u);
        std::copy(mu.begin(), mu.end(), words_ + (H_ - n + 1));
        big_.forward(std::span<const std::uint32_t>(words_, H_ + 1), MU_);
        // -g mod (x^H - 1) at H.
        std::fill(words_, words_ + H_, 0u);
        for (std::size_t i = 0; i <= n; ++i) {
            std::uint32_t& w = words_[i % H_];
            w = w + g[i] >= ntt::kModulus ? w + g[i] - ntt::kModulus : w + g[i];
        }
        for (std::size_t i = 0; i < H_; ++i) words_[i] = words_[i] ? ntt::kModulus - words_[i] : 0u;
        half_.forward(std::span<const std::uint32_t>(words_, H_), NG_);
    }
    Products(const Products&) = delete;
    Products& operator=(const Products&) = delete;

    int degree() const { return m_; }

    void multiply(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t* out) {
        const std::size_t n = std::size_t(m_);
        big_.forward(std::span(a, n), FA_);
        if (a == b) {
            big_.multiply(FA_, FA_, FA_);
        } else {
            big_.forward(std::span(b, n), FB_);
            big_.multiply(FA_, FB_, FA_);
        }
        std::uint32_t* p1 = words_;
        big_.inverse(FA_, std::span(p1, n - 1), ntt::Form::leaves, {.offset = n});
        std::uint32_t* quotient = words_ + span(n);
        quotient_of(p1, n - 1, quotient, n - 1);
        remainder(FA_, nullptr, quotient, out);
    }
    void square(const std::uint32_t* a, std::uint32_t* out) { multiply(a, a, out); }

private:
    friend class Multiplier;
    friend class Powers;

    // Words of an array of n words and its padding, kept 320 bytes apart modulo 4 KiB.
    static constexpr std::size_t span(std::size_t n) { return series::detail::footprint(n + ntt::kPadding); }
    std::uint32_t* at(int i) const { return memory_.data() + std::size_t(i) * span(2 * H_); }

    // out[0, count) = floor(A / g) for A div x^m = a1 (words <= m, deg A <= 2m - 1; count <= m).
    void quotient_of(const std::uint32_t* a1, std::size_t words, std::uint32_t* out, std::size_t count) {
        big_.cyclic_product(std::span(a1, words), MU_, work_, std::span(out, count), {.offset = H_});
    }

    // out[0, m) = (A B - quotient g) at H, for A (B) a leaf form at H or longer (B null: A alone).
    void remainder(const std::uint32_t* A, const std::uint32_t* B, const std::uint32_t* quotient, std::uint32_t* out) {
        half_.forward(std::span(quotient, std::size_t(m_) - 1), QH_);
        const ntt::Term terms[] = {{A, B}, {QH_, NG_}};
        half_.inverse_product(terms, work_, std::span(out, std::size_t(m_)));
    }

    int m_, lg_;
    std::size_t H_;
    const ntt::Transform &big_, &half_;
    Words memory_;
    std::uint32_t *MU_, *NG_, *FA_, *FB_, *work_, *QH_, *words_;
};

class Multiplier {
public:
    Multiplier(Products& P, const std::uint32_t* r) : P_(P), memory_(2 * Products::span(2 * P.H_)) {
        RS_ = memory_.data(), R_ = RS_ + Products::span(2 * P.H_);
        prepare(P, r, RS_, R_);
    }
    Multiplier(const Multiplier&) = delete;
    Multiplier& operator=(const Multiplier&) = delete;

    void step(const std::uint32_t* q, std::uint32_t* out) {
        Products& P = P_;
        const std::size_t n = std::size_t(P.m_);
        std::uint32_t* quotient = P.words_;
        P.big_.cyclic_product(std::span(q, n), RS_, P.work_, std::span(quotient, n - 1), {.offset = P.H_});
        P.half_.forward(std::span(q, n), P.FA_);
        P.remainder(P.FA_, R_, quotient, out);
    }

    // RS = the leaf form at L of x^(H-m) r', r' = floor(r x^m / g) (r of m words: A1 = r); R = r's at H.
    static void prepare(Products& P, const std::uint32_t* r, std::uint32_t* RS, std::uint32_t* R) {
        const std::size_t n = std::size_t(P.m_), H = P.H_;
        std::uint32_t* words = P.words_;
        std::fill(words, words + H, 0u);
        P.quotient_of(r, n, words + (H - n), n);
        P.big_.forward(std::span<const std::uint32_t>(words, H), RS);
        P.half_.forward(std::span(r, n), R);
    }

private:
    Products& P_;
    Words memory_;
    std::uint32_t *RS_, *R_;
};

class Powers {
public:
    static constexpr int kSteps = 8;

    Powers(Products& P, const std::uint32_t* r) : P_(P), memory_(std::size_t(2 * kSteps + 2) * Products::span(2 * P.H_)) {
        const std::size_t n = std::size_t(P.m_), s = Products::span(2 * P.H_);
        F_ = memory_.data(), quotient_ = F_ + s;
        std::vector<std::uint32_t> power(r, r + n), next(n);
        for (int t = 0; t < kSteps; ++t) {
            RS_[t] = quotient_ + s + std::size_t(2 * t) * s, R_[t] = RS_[t] + s;
            if (t > 0) P.multiply(power.data(), r, next.data()), power.swap(next);
            Multiplier::prepare(P, power.data(), RS_[t], R_[t]);
        }
    }
    Powers(const Powers&) = delete;
    Powers& operator=(const Powers&) = delete;

    void steps(const std::uint32_t* q, std::uint32_t* out, std::ptrdiff_t stride, std::uint32_t* last) {
        Products& P = P_;
        const std::size_t n = std::size_t(P.m_);
        P.big_.forward(std::span(q, n), F_);
        for (int t = 0; t < kSteps; ++t) {
            const ntt::Term product[] = {{F_, RS_[t]}};
            P.big_.inverse_product(product, P.work_, std::span(quotient_, n - 1), {.offset = P.H_});
            P.remainder(F_, R_[t], quotient_, t + 1 < kSteps ? out + t * stride : last);
        }
    }

private:
    Products& P_;
    Words memory_;
    std::uint32_t *F_, *quotient_;
    std::array<std::uint32_t*, kSteps> RS_{}, R_{};
};

}  // namespace fps::factorization::detail
