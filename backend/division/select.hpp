// The division's cost model: ns per word of the ops it is built from, per transform length, on this
// target (a tuned choice, "division.costs"; tune.hpp measures it, tools/tune.py division writes the
// compiled defaults in tuned.hpp). division.hpp's Plan reads it once per call.
//
//   costs.get()          // Costs: the active values, kCostDefaults until tune::run() or load()
//   Cost(costs.get())    // accessors in ns: forward(lg), cyclic(lg), product(lg, terms), ...
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>

#include "backend/division/tuned.hpp"
#include "backend/tune/tune.hpp"

namespace fps::division {

// Per transform length 2^lg, lg = kFirstLg .. kFirstLg + kLgs - 1 (ntt::kMinLog .. kMaxLog), ns per word:
//   forward   Transform::forward(in, F) from a source of L/2 words
//   cyclic    Transform::cyclic_product of a source of L/2 words, the lower half out
//   product   Transform::inverse_product of one leaf product, the upper half out
//   term      each further leaf product of inverse_product (sums of 3 against 1, halved)
//   leaf      Transform::multiply_add, a leaf product in memory (sums past 4 terms)
//   inverse   series::inverse of n = 2^lg coefficients, per coefficient
// then scalars: pass (ns per word of an element pass: a fold, a reversed sum), long_term and
// long_step (long division of k d terms: k d long_term + k long_step).
inline constexpr int kFirstLg = 6, kLgs = 20, kRows = 6;
enum Row { kForward, kCyclic, kProduct, kTerm, kLeaf, kInverse };
inline constexpr std::size_t kPass = kRows * kLgs, kLongTerm = kPass + 1, kLongStep = kPass + 2, kCostCount = kPass + 3;
using Costs = std::array<double, kCostCount>;

inline constexpr Costs kCostDefaults = [] {
    const int t = int(tune::kTarget);
    Costs c{};
    const double* const rows[kRows] = {tuned::forward[t], tuned::cyclic[t], tuned::product[t],
                                       tuned::term[t],    tuned::leaf[t],   tuned::inverse[t]};
    for (int r = 0; r < kRows; ++r)
        for (int i = 0; i < kLgs; ++i) c[std::size_t(r * kLgs + i)] = rows[r][i];
    c[kPass] = tuned::pass[t];
    c[kLongTerm] = tuned::long_term[t];
    c[kLongStep] = tuned::long_step[t];
    return c;
}();
inline constinit tune::Table<kCostDefaults> costs;

// The model in ns, from a Costs.
class Cost {
public:
    explicit Cost(const Costs& c) : c_(c) {}

    double forward(int lg) const { return at(kForward, lg); }
    double cyclic(int lg) const { return at(kCyclic, lg); }
    // inverse_product of `terms` leaf products; past 4, the rest as leaf products in memory first.
    double product(int lg, std::size_t terms) const {
        const double extra = double(std::min<std::size_t>(terms, 4) - 1) * at(kTerm, lg);
        return at(kProduct, lg) + extra + (terms > 4 ? double(terms - 3) * at(kLeaf, lg) : 0.0);
    }
    // series::inverse of n coefficients: linear between the powers of two around n.
    double inverse(std::size_t n) const {
        if (n <= 1) return 0;
        const std::size_t hi = std::bit_ceil(n);
        const int lg = std::bit_width(hi) - 1;
        const double top = at(kInverse, lg);
        if (hi == n || hi <= (std::size_t(1) << kFirstLg)) return top * double(n) / double(hi);
        const double low = at(kInverse, lg - 1);
        return low + (top - low) * double(n - hi / 2) / double(hi / 2);
    }
    double pass(std::size_t words) const { return c_[kPass] * double(words); }
    double dot(std::size_t terms) const { return c_[kLongTerm] * double(terms); }  // as long division's
    double long_division(std::size_t k, std::size_t d) const {
        return c_[kLongTerm] * double(k) * double(d) + c_[kLongStep] * double(k);
    }

private:
    // ns of row r's op at length 2^lg; below kFirstLg the first length's; past the table, the
    // last length's per word, doubled per doubling (ntt::kMaxLog is the table's end).
    double at(Row r, int lg) const {
        const int i = std::clamp(lg - kFirstLg, 0, kLgs - 1);
        return c_[std::size_t(r * kLgs + i)] * std::ldexp(1.0, std::max(lg, kFirstLg));
    }
    const Costs& c_;
};

}  // namespace fps::division
