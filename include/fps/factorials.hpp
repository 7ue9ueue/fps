// Factorials modulo P: tables of i! and 1 / i!, and binomials from them.
//
//   fps::Factorials<P> fact(n);   // tables for i < n; n <= P, else std::domain_error
//   fact.size()                   // n
//   fact[i]                       // i!, i < size()
//   fact.inverse(i)               // 1 / i!, i < size()
//   fact.binomial(n, k)           // C(n, k) for n < size(); 0 if k > n
//
// P = 998244353 only in v1 (DESIGN.md, Moduli); other P fail to compile. Indices are not checked,
// as in std::vector's operator[]. The tables are const once built, so any number of threads may
// read them. Building takes O(n) time and 8 n bytes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "backend/factorials/factorials.hpp"
#include "fps/modular.hpp"

namespace fps {

namespace detail {

// fact[i] = i! and inverse[i] = 1 / i! for i < n = fact.size() = inverse.size(); any prime
// P < 2^32, n <= P. The backend kernel for 2^12 < P < 2^30 (backend/factorials), else its scalar
// model: two chains of n - 1 products each and one inverse.
template <std::uint32_t P>
void factorial_tables(std::span<std::uint32_t> fact, std::span<std::uint32_t> inverse) {
    if constexpr (P > (1u << 12) && P < (1u << 30)) factorials::tables<P>(fact, inverse);
    else factorials::model::tables<P>(fact.data(), inverse.data(), fact.size());
}

}  // namespace detail

template <std::uint32_t P>
class Factorials {
    static_assert(P == 998244353, "fps: v1 supports P = 998244353 only (DESIGN.md, Moduli)");

public:
    explicit Factorials(std::size_t n) : fact_(checked(n)), inverse_(n) {
        detail::factorial_tables<P>(fact_, inverse_);
    }

    std::size_t size() const { return fact_.size(); }
    std::uint32_t operator[](std::size_t i) const { return fact_[i]; }
    std::uint32_t inverse(std::size_t i) const { return inverse_[i]; }

    std::uint32_t binomial(std::size_t n, std::size_t k) const {
        return k > n ? 0 : mod_mul<P>(fact_[n], mod_mul<P>(inverse_[k], inverse_[n - k]));
    }

private:
    // i! for i < P is a product of nonzero residues, so it has an inverse; P! = 0 has none.
    // Checked before the tables are allocated.
    static std::size_t checked(std::size_t n) {
        if (n > P) throw std::domain_error("fps::Factorials: n > P, and P! = 0 has no inverse");
        return n;
    }

    std::vector<std::uint32_t> fact_, inverse_;  // i!, 1 / i!; values in [0, P)
};

}  // namespace fps
