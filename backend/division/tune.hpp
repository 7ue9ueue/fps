// Run-time measurement of the division's cost model ("division.costs", select.hpp), registered with
// backend/tune: tune::run() measures it on this machine, tune::save() / load() keep it.
//
//   division::measure_costs(start, budget)   // Costs: each op timed per length up to 2^budget.max_lg
//
// Per length 2^lg: Transform::forward from a source of L/2 words, cyclic_product of L/2 words (the
// lower half out), inverse_product of 1 and of 3 leaf products (the upper half out; term = half
// the difference), multiply_add, and series::inverse of L coefficients, each the median of
// budget.samples samples of at least budget.min_sample_ns, timed in turn; ns per word. Past
// max_lg: start's values times each row's ratio new / start at max_lg. pass: reversed<true> on 2^14
// words. Long division: k = 4096 by d = 1 and k = 128 by d = 1024, solved for long_term and
// long_step. A value within budget.margin of start's keeps start's. tools/tune.py division runs
// tune.cpp, which measures to 2^25.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string_view>
#include <vector>

#include "backend/division/division.hpp"
#include "backend/tune/tune.hpp"

namespace fps::division {

struct Budget {
    int max_lg;            // measured up to 2^max_lg, at most 25
    int samples;           // per op and length
    double min_sample_ns;  // each sample at least this long
    double margin;         // keep start's value when within it
};

// Up to 2^18 at run time (2^25 would take about 1 GB and 20 s), 5 samples of 200 us.
inline constexpr Budget kRunTime = {18, 5, 2e5, 0.03};

namespace detail {

struct alignas(64) Line {
    u32 words[16];
};

// An aligned array of n + ntt::kPadding words, random below P.
struct Array {
    std::vector<Line> lines;
    explicit Array(std::size_t n, unsigned seed) : lines((n + ntt::kPadding + 15) / 16) {
        std::mt19937 rng(seed);
        for (Line& l : lines)
            for (u32& x : l.words) x = rng() % kP;
    }
    u32* data() { return lines.data()->words; }
};

inline void kept(const void* p) { asm volatile("" : : "r"(p) : "memory"); }

// The rows' ns per word at length 2^lg.
inline std::array<double, kRows> time_length(int lg, const Budget& budget) {
    const ntt::Transform& t = transform(lg);
    const std::size_t len = t.length(), half = len / 2;
    Array src(len, 1), A(len, 2), B(len, 3), C(len, 4), D(len, 5), E(len, 6), F(len, 7), work(len, 8), out(len, 9);
    const std::span<const u32> in(src.data(), half);
    const std::span<u32> o(out.data(), half);
    const ntt::Term one[] = {{A.data(), B.data()}};
    const ntt::Term three[] = {{A.data(), B.data()}, {C.data(), D.data()}, {E.data(), F.data()}};
    const auto times = tune::time(
        budget.samples, budget.min_sample_ns, [&] { t.forward(in, F.data()), kept(F.data()); },
        [&] { t.cyclic_product(in, B.data(), work.data(), o), kept(o.data()); },
        [&] { t.inverse_product(one, work.data(), o, {.offset = half}), kept(o.data()); },
        [&] { t.inverse_product(three, work.data(), o, {.offset = half}), kept(o.data()); },
        [&] { t.multiply_add(work.data(), A.data(), B.data()), kept(work.data()); });
    // series::inverse of len coefficients, f[0] != 0
    Array f(len, 10), g(len, 11);
    f.data()[0] = 3;
    std::vector<u32> scratch(series::inverse_scratch(len));
    const auto inverse = tune::time(budget.samples, budget.min_sample_ns, [&] {
        series::inverse(std::span<const u32>(f.data(), len), std::span<u32>(g.data(), len), scratch);
        kept(g.data());
    });
    const double words = double(len);
    return {times[0] / words, times[1] / words, times[2] / words, std::max(0.0, times[3] - times[2]) / 2 / words,
            times[4] / words, inverse[0] / words};
}

// pass, long_term, long_step.
inline std::array<double, 3> time_scalars(const Budget& budget) {
    constexpr std::size_t kPassWords = std::size_t(1) << 14;
    Array a(kPassWords, 12), b(kPassWords, 13);
    const double pass = tune::time(budget.samples, budget.min_sample_ns, [&] {
        reversed<true>(a.data() + kPassWords - 1, b.data(), kPassWords);
        kept(b.data());
    })[0] / double(kPassWords);
    const auto long_ns = [&](std::size_t k, std::size_t d) {
        Array f(k + d, 14), g(d + 1, 15), q(k, 16), rest(k + d, 17);
        g.data()[d] = 1;
        return tune::time(budget.samples, budget.min_sample_ns, [&] {
            long_division(std::span<const u32>(f.data(), k + d), std::span<const u32>(g.data(), d + 1),
                          std::span<u32>(q.data(), k), {}, rest.data());
            kept(q.data());
        })[0] / double(k);
    };
    const double thin = long_ns(4096, 1), wide = long_ns(128, 1024);  // per step: step + d term
    const double term = std::max(0.0, (wide - thin) / 1023);
    return {pass, term, std::max(0.0, thin - term)};
}

}  // namespace detail

inline Costs measure_costs(const Costs& start, const Budget& budget) {
    Costs c = start;
    const auto keep = [&](std::size_t i, double value) {
        if (std::abs(value / start[i] - 1) > budget.margin) c[i] = value;
    };
    std::array<double, kRows> ratio{};
    for (int lg = kFirstLg; lg <= std::min(budget.max_lg, kFirstLg + kLgs - 1); ++lg) {
        const auto row = detail::time_length(lg, budget);
        for (int r = 0; r < kRows; ++r) {
            const std::size_t i = std::size_t(r * kLgs + lg - kFirstLg);
            keep(i, row[std::size_t(r)]);
            ratio[std::size_t(r)] = c[i] / start[i];
        }
    }
    for (int lg = budget.max_lg + 1; lg < kFirstLg + kLgs; ++lg)
        for (int r = 0; r < kRows; ++r) {
            const std::size_t i = std::size_t(r * kLgs + lg - kFirstLg);
            c[i] = start[i] * ratio[std::size_t(r)];
        }
    const auto scalars = detail::time_scalars(budget);
    keep(kPass, scalars[0]), keep(kLongTerm, scalars[1]), keep(kLongStep, scalars[2]);
    return c;
}

#ifndef FPS_NO_TUNER

inline const bool kCostsRegistered = tune::add({
    "division.costs",
    [] { costs.set(measure_costs(kCostDefaults, kRunTime)); },
    [] { return tune::text(costs.get()); },
    [](std::string_view text, bool apply) {
        const auto values = tune::parse<Costs>(text);
        if (!values || !std::ranges::all_of(*values, [](double t) { return t >= 0 && std::isfinite(t); })) return false;
        if (apply) costs.set(*values);
        return true;
    },
});

#endif

}  // namespace fps::division
