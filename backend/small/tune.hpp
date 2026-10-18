// Measures small's tuned choices on this machine and registers them with backend/tune, in this
// order: the variant per shape class ("small.variant") and beats_ntt's NTT times ("small.ntt" at
// powers of two, "small.ragged" between them), which time multiply() with the variants in place.
// tune::run() measures, tune::save() / load() keep them.
//
//   small::measure_variants(start, budget[, log])  // Choices: per class the fastest, else start's
//   small::points(c, dense)                        // the shapes class c is timed at
//   small::text(choices), parse(text)              // wisdom: kSignature, then the variant per class
//   small::variant_names()                         // the candidates' names, e.g. "avx512.r8.u2"
//   small::measure_ntt(model, start, budget)       // NTT times for lg = 6 .. 25 (kNttDefaults' form)
//   small::measure_ragged(model, start, budget)    // the same between powers of two (kRaggedDefaults')
//
// measure_variants: each class with m up to 2^budget.max_octave is timed at points(c, dense): m at
// the octave's end and at 3/4 of it (dense: also just past its start), n / m 1 / 3 / 6 / 16 by the
// ratio class (dense: 1 and 1.5 / 2 and 3 / 4 and 6 / 8, 16 and 64, at most 2^23 terms). Every
// candidate is timed in turn at each point (tune::time_each); the class takes the candidate with
// the least geometric mean of time / fastest, unless start's is within budget.margin of it. Larger
// octaves keep start's.
//
// measure_ntt: for each lg up to budget.max_lg, the probe is the n x m with n + m - 1 = 2^lg
// (select.hpp's ntt_point; measure_ragged: each of its points in (2^(lg-1), 2^lg)), n >= m, where
// the model's time first reaches start's NTT time. There multiply() and ntt::multiply (into the
// caller's memory, as f * g runs it) are timed in turn, and the NTT time becomes the model's time
// there times NTT / multiply as measured: so beats_ntt switches where the measured times cross,
// along the model's shape. If that moves the crossing by more than 25%, a second probe goes there.
// Past max_lg: start's times scaled by the ratio at max_lg (per point). Values within budget.margin
// of start keep start's.
//
// kVariantRunTime and kRunTime, the budgets of tune::run(): their costs and the noise behind the
// margins are in backend/tune/notes.md. tools/tune.py runs tune.cpp, which measures every class
// densely, fits the model and measures every lg.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "backend/ntt/ntt.hpp"
#include "backend/ntt/tune.hpp"  // registers "ntt.variant" before "small.ntt", which times the NTT
#include "backend/small/small.hpp"
#include "backend/tune/tune.hpp"

namespace fps::small {

struct VariantBudget {
    int max_octave;        // classes with m up to 2^max_octave; 10: every class
    bool dense;            // more points per class (points())
    int samples;           // per point and candidate
    double min_sample_ns;  // each sample at least this long
    double margin;         // keep start's variant when within it
};

// Classes with m up to 256 (beats_ntt sends larger m to the NTT on every timed target), samples of
// 10 us, margin 3% (backend/tune/notes.md).
inline constexpr VariantBudget kVariantRunTime = {8, false, 3, 1e4, 0.03};

// The shapes n x m (n >= m) class c is timed at.
inline std::vector<std::pair<std::size_t, std::size_t>> points(int c, bool dense) {
    static constexpr double kSparse[kRatios][3] = {{1}, {3}, {6}, {16}};
    static constexpr double kDense[kRatios][3] = {{1, 1.5}, {2, 3}, {4, 6}, {8, 16, 64}};
    const std::size_t end = std::size_t(1) << (c / kRatios);
    std::vector<std::size_t> ms = {end, std::max<std::size_t>(1, 3 * end / 4)};
    if (dense) ms.push_back(end / 2 + 1);
    std::vector<std::pair<std::size_t, std::size_t>> shapes;
    for (const std::size_t m : ms)
        for (const double r : (dense ? kDense : kSparse)[c % kRatios]) {
            const std::size_t n = std::max(m, std::size_t(std::lround(r * double(m))));
            const bool large = dense && n * m > (std::size_t(1) << 23) && !shapes.empty();
            if (r > 0 && !large && shape_class(n, m) == c && std::ranges::find(shapes, std::pair(n, m)) == shapes.end())
                shapes.push_back({n, m});
        }
    return shapes;
}

// What measure_variants() timed at one class: the points and ns[point][candidate].
struct ClassTimes {
    int c;
    std::vector<std::pair<std::size_t, std::size_t>> shapes;
    std::vector<std::vector<double>> ns;
};

namespace detail {

// ns per call of every candidate on n x m, modulo P = 998244353, into a 64-byte aligned output.
inline std::vector<double> time_variants(std::size_t n, std::size_t m, const VariantBudget& budget) {
    constexpr u32 P = 998244353;
    std::mt19937 rng(unsigned(n * 31 + m));
    std::vector<u32> a(n), b(m);
    for (u32& x : a) x = rng() % P;
    for (u32& x : b) x = rng() % P;
    struct alignas(64) Line {
        u32 words[16];
    };
    std::vector<Line> lines((n + m - 1 + 15) / 16);
    u32* const out = lines.data()->words;
    std::vector<int> ids(kCandidates);
    for (int v = 0; v < kCandidates; ++v) ids[v] = v;
    return tune::time_each(budget.samples, budget.min_sample_ns, ids, [&](int v) {
        run_variant<P>(v, a.data(), n, b.data(), m, out);
        asm volatile("" : : "r"(out) : "memory");
    });
}

}  // namespace detail

inline Choices measure_variants(const Choices& start, const VariantBudget& budget,
                                std::vector<ClassTimes>* log = nullptr) {
    Choices choices = start;
    for (int c = 0; c < kClasses && c / kRatios <= budget.max_octave; ++c) {
        ClassTimes times{c, points(c, budget.dense), {}};
        std::vector<double> score(kCandidates);
        for (const auto& [n, m] : times.shapes) {
            const std::vector<double> t = detail::time_variants(n, m, budget);
            const double fastest = *std::ranges::min_element(t);
            for (int v = 0; v < kCandidates; ++v) score[v] += std::log(t[v] / fastest) / double(times.shapes.size());
            times.ns.push_back(t);
        }
        const int best = int(std::ranges::min_element(score) - score.begin());
        if (score[start[c]] > score[best] + std::log1p(budget.margin)) choices[c] = std::uint8_t(best);
        if (log) log->push_back(std::move(times));
    }
    return choices;
}

inline std::vector<std::string> variant_names() {
    return []<class... V>(Candidates<V...>) { return std::vector<std::string>{variant_name<V>()...}; }(Selected{});
}

// Wisdom text: kSignature, then the variant of every class.
inline std::string text(const Choices& choices) {
    std::array<unsigned, kClasses + 1> values{kSignature};
    for (int c = 0; c < kClasses; ++c) values[c + 1] = choices[c];
    return tune::text(values);
}

// The choices of a wisdom text: nothing unless it has this list's signature and every variant exists.
inline std::optional<Choices> parse(std::string_view s) {
    const auto values = tune::parse<std::array<unsigned, kClasses + 1>>(s);
    if (!values || (*values)[0] != kSignature) return std::nullopt;
    Choices choices{};
    for (int c = 0; c < kClasses; ++c) {
        if ((*values)[c + 1] >= unsigned(kCandidates)) return std::nullopt;
        choices[c] = std::uint8_t((*values)[c + 1]);
    }
    return choices;
}

struct Budget {
    int max_lg;            // measured up to 2^max_lg, at most 25
    int samples;           // per probe and candidate
    double min_sample_ns;  // each sample at least this long
    double margin;         // keep start's value when within it
};

// Samples of 200 us: at 50 us, Emerald Rapids measured the NTT up to 16% slower at lg 6-9 than
// with 1 ms samples; at 200 us within 2%. A margin of 5%: one run's median moved 2-5% on our
// machines (backend/tune/notes.md). The ragged points to 2^12 (4 per lg; notes/212-r1.md).
inline constexpr Budget kRunTime = {16, 5, 2e5, 0.05};
inline constexpr Budget kRaggedRunTime = {12, 5, 2e5, 0.05};

namespace detail {

// The probe for a product of `length` words: n + m - 1 = length, n >= m, the least m with
// model_ns(n, m) >= ntt (the model grows with m there), or m = (n + m) / 2 if none.
inline std::pair<std::size_t, std::size_t> probe(const Model& model, std::size_t length, double ntt) {
    std::size_t lo = 1, hi = (length + 1) / 2;
    while (lo < hi) {
        const std::size_t m = (lo + hi) / 2;
        if (model_ns(model, length + 1 - m, m) < ntt) lo = m + 1;
        else hi = m;
    }
    return {length + 1 - lo, lo};
}

// ns of multiply() and of ntt::multiply with the product in the same memory, timed in turn.
inline std::array<double, 2> time_shape(std::size_t n, std::size_t m, const Budget& budget) {
    constexpr u32 P = 998244353;
    struct alignas(64) Line {
        u32 words[16];
    };
    std::mt19937 rng(unsigned(n * 31 + m));
    std::vector<u32> a(n), b(m);
    for (u32& x : a) x = rng() % P;
    for (u32& x : b) x = rng() % P;
    std::vector<Line> lines((n + m - 1 + 15) / 16);
    u32* const out = lines.data()->words;
    const auto kept = [](const u32* p) { asm volatile("" : : "r"(p) : "memory"); };
    return tune::time(
        budget.samples, budget.min_sample_ns,
        [&] {
            multiply<P>(a, b, std::span(out, n + m - 1));
            kept(out);
        },
        [&] {
            ntt::multiply(a, b, std::span(out, n + m - 1));
            kept(out);
        });
}

// The NTT's time at length ntt_point(lg, points[i]) for lg = 6 .. 25, entry K (lg - 6) + i
// (measure_ntt, above).
template <std::size_t K>
std::array<double, 20 * K> measure_points(const Model& model, const std::array<double, 20 * K>& start,
                                          const Budget& budget, const std::array<int, K>& points) {
    std::array<double, 20 * K> times = start;
    std::array<double, K> ratio;
    ratio.fill(1);
    for (int lg = 6; lg <= budget.max_lg; ++lg)
        for (std::size_t i = 0; i < K; ++i) {
            const std::size_t e = K * std::size_t(lg - 6) + i;
            double t = start[e];
            for (int pass = 0; pass < 2; ++pass) {
                const auto [n, m] = probe(model, ntt_point(lg, points[i]), t);
                const auto [small, ntt] = time_shape(n, m, budget);
                const double next = model_ns(model, n, m) * ntt / small;
                const bool near = std::abs(next / t - 1) <= 0.25;
                t = next;
                if (near) break;
            }
            ratio[i] = t / start[e];
            if (std::abs(ratio[i] - 1) > budget.margin) times[e] = t;
        }
    for (std::size_t i = 0; i < K; ++i)
        if (std::abs(ratio[i] - 1) > budget.margin)
            for (int lg = budget.max_lg + 1; lg <= 25; ++lg)
                times[K * std::size_t(lg - 6) + i] = start[K * std::size_t(lg - 6) + i] * ratio[i];
    return times;
}

}  // namespace detail

inline std::array<double, 20> measure_ntt(const Model& model, const std::array<double, 20>& start,
                                          const Budget& budget) {
    return detail::measure_points<1>(model, start, budget, {kRaggedPoints});
}

inline std::array<double, 20 * kRaggedPoints> measure_ragged(const Model& model,
                                                             const std::array<double, 20 * kRaggedPoints>& start,
                                                             const Budget& budget) {
    return detail::measure_points<kRaggedPoints>(model, start, budget, {0, 1, 2, 3});
}

#ifndef FPS_NO_TUNER

// Registered first: tune::run() measures in this order, and measure_ntt times multiply().
inline const bool kVariantRegistered = tune::add({
    "small.variant",
    [] { variants.set(measure_variants(kDefaults, kVariantRunTime)); },
    [] { return text(variants.get()); },
    [](std::string_view s, bool apply) {
        const auto choices = parse(s);
        if (choices && apply) variants.set(*choices);
        return choices.has_value();
    },
});

// The NTT times of a wisdom text: nothing unless every value is positive and finite.
template <std::size_t K>
std::optional<std::array<double, K>> parse_times(std::string_view text) {
    const auto times = tune::parse<std::array<double, K>>(text);
    if (!times || !std::ranges::all_of(*times, [](double t) { return t > 0 && std::isfinite(t); }))
        return std::nullopt;
    return times;
}

inline const bool kNttRegistered = tune::add({
    "small.ntt",
    [] { ntt_times.set(measure_ntt(kModel, kNttDefaults, kRunTime)); },
    [] { return tune::text(ntt_times.get()); },
    [](std::string_view text, bool apply) {
        const auto times = parse_times<20>(text);
        if (times && apply) ntt_times.set(*times);
        return times.has_value();
    },
});

inline const bool kRaggedRegistered = tune::add({
    "small.ragged",
    [] { ragged_times.set(measure_ragged(kModel, kRaggedDefaults, kRaggedRunTime)); },
    [] { return tune::text(ragged_times.get()); },
    [](std::string_view text, bool apply) {
        const auto times = parse_times<20 * kRaggedPoints>(text);
        if (times && apply) ragged_times.set(*times);
        return times.has_value();
    },
});

#endif

}  // namespace fps::small
