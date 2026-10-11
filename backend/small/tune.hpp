// Measures beats_ntt's NTT times (select.hpp) on this machine, and registers them with
// backend/tune as the choice "small.ntt": tune::run() measures, tune::save() / load() keep them.
//
//   small::measure_ntt(model, start, budget)   // NTT times for lg = 6 .. 25 (kNttDefaults' form)
//
// For each lg up to budget.max_lg, the probe is the n x m with n + m - 1 = 3 2^(lg - 2) (the middle
// of the products that transform at 2^lg), n >= m, where the model's time first reaches start's
// NTT time. There multiply() and an ntt::Convolution (the product in the caller's memory) are timed
// in turn, and the NTT time becomes the model's time there times NTT / multiply as measured: so
// beats_ntt switches where the measured times cross, along the model's shape. If that moves the
// crossing by more than 25%, a second probe goes there. Past max_lg: start's times scaled by the
// ratio at max_lg. Values within budget.margin of start keep start's.
//
// kRunTime, the budget of tune::run(): its cost and the noise behind the margin are in
// backend/tune/notes.md. tools/tune.py runs tune.cpp, which fits the model and measures every lg.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "backend/ntt/ntt.hpp"
#include "backend/small/small.hpp"
#include "backend/tune/tune.hpp"

namespace fps::small {

struct Budget {
    int max_lg;            // measured up to 2^max_lg, at most 25
    int samples;           // per probe and candidate
    double min_sample_ns;  // each sample at least this long
    double margin;         // keep start's value when within it
};

// Samples of 200 us: at 50 us, Emerald Rapids measured the NTT up to 16% slower at lg 6-9 than
// with 1 ms samples; at 200 us within 2%. A margin of 5%: one run's median moved 2-5% on our
// machines (backend/tune/notes.md).
inline constexpr Budget kRunTime = {16, 5, 2e5, 0.05};

namespace detail {

// The probe for lg: n + m - 1 = 3 2^(lg - 2), n >= m, the least m with model_ns(n, m) >= ntt
// (the model grows with m there), or m = (n + m) / 2 if none.
inline std::pair<std::size_t, std::size_t> probe(const Model& model, int lg, double ntt) {
    const std::size_t length = std::size_t(3) << (lg - 2);
    std::size_t lo = 1, hi = (length + 1) / 2;
    while (lo < hi) {
        const std::size_t m = (lo + hi) / 2;
        if (model_ns(model, length + 1 - m, m) < ntt) lo = m + 1;
        else hi = m;
    }
    return {length + 1 - lo, lo};
}

// ns of multiply() and of the Convolution with the product in the same memory, timed in turn.
inline std::array<double, 2> time_shape(std::size_t n, std::size_t m, const Budget& budget) {
    constexpr u32 P = 998244353;
    struct alignas(64) Line {
        u32 words[16];
    };
    std::mt19937 rng(unsigned(n * 31 + m));
    std::vector<u32> a(n), b(m);
    for (u32& x : a) x = rng() % P;
    for (u32& x : b) x = rng() % P;
    std::vector<Line> lines((std::max(n + m - 1, ntt::Convolution::length(n, m) + ntt::kPadding) + 15) / 16);
    u32* const out = lines.data()->words;
    const auto kept = [](const u32* p) { asm volatile("" : : "r"(p) : "memory"); };
    return tune::time(
        budget.samples, budget.min_sample_ns,
        [&] {
            multiply<P>(a, b, std::span(out, n + m - 1));
            kept(out);
        },
        [&] {
            ntt::Convolution conv(n, m, out);
            std::ranges::copy(a, out);
            std::ranges::copy(b, conv.b());
            conv.multiply();
            kept(out);
        });
}

}  // namespace detail

inline std::array<double, 20> measure_ntt(const Model& model, const std::array<double, 20>& start,
                                          const Budget& budget) {
    std::array<double, 20> times = start;
    double ratio = 1;
    for (int lg = 6; lg <= budget.max_lg; ++lg) {
        double t = start[lg - 6];
        for (int pass = 0; pass < 2; ++pass) {
            const auto [n, m] = detail::probe(model, lg, t);
            const auto [small, ntt] = detail::time_shape(n, m, budget);
            const double next = model_ns(model, n, m) * ntt / small;
            const bool near = std::abs(next / t - 1) <= 0.25;
            t = next;
            if (near) break;
        }
        ratio = t / start[lg - 6];
        if (std::abs(ratio - 1) > budget.margin) times[lg - 6] = t;
    }
    if (std::abs(ratio - 1) > budget.margin)
        for (int lg = budget.max_lg + 1; lg <= 25; ++lg) times[lg - 6] = start[lg - 6] * ratio;
    return times;
}

#ifndef FPS_NO_TUNER

inline const bool kNttRegistered = tune::add({
    "small.ntt",
    [] { ntt_times.set(measure_ntt(kModel, kNttDefaults, kRunTime)); },
    [] { return tune::text(ntt_times.get()); },
    [](std::string_view text, bool apply) {
        const auto times = tune::parse<std::array<double, 20>>(text);
        if (!times || !std::ranges::all_of(*times, [](double t) { return t > 0 && std::isfinite(t); }))
            return false;
        if (apply) ntt_times.set(*times);
        return true;
    },
});

#endif

}  // namespace fps::small
