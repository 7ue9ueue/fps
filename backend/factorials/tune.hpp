// Measures the factorials variant per size class (select.hpp) on this machine, and registers it
// with backend/tune as the choice "factorials.variant": tune::run() measures, tune::save() / load()
// keep it.
//
//   factorials::measure(start, budget[, log])  // Choices: per class the fastest variant, else start's
//   factorials::points(c, count)               // the sizes a dense class is timed at
//   factorials::text(choices), parse(text)     // wisdom: kSignature, then the variant per class
//   factorials::variant_names()                // "loop", then "<isa>.<K>.<step>" per candidate
//
// Dense part: a class (e, e + w] ending at most 2^dense_lg is timed at `points` sizes:
// e + j w / points + 1 for 0 < j < points (just past a round size: the most lanes' slack) and e + w
// (none). At each, every variant that runs at the class's least n and was not dropped is timed in
// turn (tune::time_each, two fresh std::vectors); the class takes the variant with the least
// geometric mean of time / fastest over its points, unless start's is within budget.margin of it.
// A variant more than 2x the fastest, and narrower than it (fewer lanes; the loop has 1), is
// dropped for the larger sizes (wider variants only gain with n), except where start has it.
// Sparse part: one size per octave, 2^k for dense_lg < k <= sparse_lg, decides the octave's
// classes (2^(k-1), 2^k], each keeping start's variant when within budget.sparse_margin there;
// 2^sparse_lg also decides every larger class. One size cannot place a change inside an octave,
// so this margin is wider: it corrects a table made on another CPU (notes.md).
// kRunTime, the budget of tune::run(): its cost and decisions are in backend/tune/notes.md.
// tools/tune.py runs tune.cpp, which measures every class to 2^24 at a larger budget.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "backend/factorials/factorials.hpp"
#include "backend/tune/tune.hpp"

namespace fps::factorials {

struct Budget {
    int dense_lg;          // every class ending at most 2^dense_lg
    int sparse_lg;         // then one size per octave up to 2^sparse_lg
    int points;            // sizes per class in the dense part
    int samples;           // per size and variant
    double min_sample_ns;  // each sample at least this long
    double margin;         // keep start's variant when within it (dense part)
    double sparse_margin;  // the same in the sparse part
};

// Dense through 2^17 (Zen 5 changes variant inside (2^16, 2^17]; past it a variant wins whole
// octaves on our machines), samples of 10 us (20 us decided within 0.1%), margins 3% and 5%:
// 37-90 ms per run (backend/tune/notes.md).
inline constexpr Budget kRunTime = {17, 20, 2, 3, 1e4, 0.03, 0.05};

// The lanes of each variant (W = K S::lanes; the loop 1).
inline constexpr auto kWidths = []<class... V>(Candidates<V...>) {
    return std::array<int, sizeof...(V) + 1>{1, V::vectors * V::Isa::lanes...};
}(Selected{});

inline std::vector<std::size_t> points(int c, int count) {
    const std::size_t end = class_end(c), e = class_start(c) - 1, width = end - e;
    std::vector<std::size_t> n;
    for (int j = 1; j < count; ++j) {
        const std::size_t m = e + j * width / count + 1;
        if (m < end && (n.empty() || n.back() < m)) n.push_back(m);
    }
    n.push_back(end);
    return n;
}

// What measure() timed at one class: the variants, the points, and ns[point][variant].
struct ClassTimes {
    int c;
    std::vector<int> ids;
    std::vector<std::size_t> n;
    std::vector<std::vector<double>> ns;
};

// ns per call of each variant in ids at size n, on fresh tables of P = 998244353.
inline std::vector<double> time_size(std::size_t n, const std::vector<int>& ids, const Budget& budget) {
    constexpr u32 P = 998244353;
    std::vector<u32> fact(n), inverse(n);
    return tune::time_each(budget.samples, budget.min_sample_ns, ids, [&](int v) {
        run_variant<P>(v, fact.data(), inverse.data(), n);
        asm volatile("" : : "r"(fact.data()), "r"(inverse.data()) : "memory");
    });
}

inline Choices measure(const Choices& start, const Budget& budget, std::vector<ClassTimes>* log = nullptr) {
    Choices choices = start;
    std::vector<bool> dropped(kMinSizes.size());
    // The variants to time for classes [first, last): not dropped, or start's at one of them.
    const auto variants_for = [&](int first, int last) {
        std::vector<int> ids;
        for (int v = 0; v < int(kMinSizes.size()); ++v) {
            bool timed = !dropped[v];
            for (int c = first; c < last; ++c) timed |= v == start[c];
            if (timed && kMinSizes[v] <= class_start(first)) ids.push_back(v);
        }
        return ids;
    };
    // score[i]: mean log(time / fastest) of ids[i]. Sets class c; drops the slow narrow variants.
    const auto decide = [&](int c, const std::vector<int>& ids, const std::vector<double>& score, double margin) {
        const std::size_t best = std::ranges::min_element(score) - score.begin();
        const auto kept = std::ranges::find(ids, int(start[c]));
        if (kept == ids.end() || score[kept - ids.begin()] > score[best] + std::log1p(margin))
            choices[c] = std::uint8_t(ids[best]);
        for (std::size_t i = 0; i < ids.size(); ++i)
            if (score[i] > std::log(2.0) && kWidths[ids[i]] < kWidths[ids[best]]) dropped[ids[i]] = true;
    };
    const auto scores = [](const std::vector<std::vector<double>>& times) {
        std::vector<double> score(times[0].size());
        for (const std::vector<double>& t : times) {
            const double fastest = *std::ranges::min_element(t);
            for (std::size_t i = 0; i < t.size(); ++i) score[i] += std::log(t[i] / fastest) / double(times.size());
        }
        return score;
    };
    int c = 0;
    for (; c < kClasses && class_end(c) <= (std::size_t(1) << budget.dense_lg); ++c) {
        const std::vector<int> ids = variants_for(c, c + 1);
        const std::vector<std::size_t> n = points(c, budget.points);
        std::vector<std::vector<double>> times;
        for (const std::size_t m : n) times.push_back(time_size(m, ids, budget));
        if (log) log->push_back({c, ids, n, times});
        decide(c, ids, scores(times), budget.margin);
    }
    for (int k = budget.dense_lg + 1; k <= budget.sparse_lg && c < kClasses; ++k) {
        const std::size_t n = std::size_t(1) << k;
        const int last = k == budget.sparse_lg ? kClasses : size_class(n) + 1;
        const std::vector<int> ids = variants_for(c, last);
        const std::vector<std::vector<double>> times = {time_size(n, ids, budget)};
        if (log) log->push_back({size_class(n), ids, {n}, times});
        const std::vector<double> score = scores(times);
        for (; c < last; ++c) decide(c, ids, score, budget.sparse_margin);
    }
    return choices;
}

// "loop", then "<isa>.<K>.<step>" per candidate, e.g. "avx512.1.shoup".
inline std::vector<std::string> variant_names() {
    return []<class... V>(Candidates<V...>) {
        return std::vector<std::string>{"loop", (std::string(V::Isa::name) + "." + std::to_string(V::vectors) + "." +
                                                 std::string(step_name<V>))...};
    }(Selected{});
}

// Wisdom text: kSignature, then the variant of every class.
inline std::string text(const Choices& choices) {
    std::array<unsigned, kClasses + 1> values{kSignature};
    for (int c = 0; c < kClasses; ++c) values[c + 1] = choices[c];
    return tune::text(values);
}

// The choices of a wisdom text: nothing unless it has this list's signature and is valid.
inline std::optional<Choices> parse(std::string_view s) {
    const auto values = tune::parse<std::array<unsigned, kClasses + 1>>(s);
    if (!values || (*values)[0] != kSignature) return std::nullopt;
    Choices choices{};
    for (int c = 0; c < kClasses; ++c) {
        if ((*values)[c + 1] >= kMinSizes.size()) return std::nullopt;
        choices[c] = std::uint8_t((*values)[c + 1]);
    }
    if (!valid(choices)) return std::nullopt;
    return choices;
}

#ifndef FPS_NO_TUNER

inline const bool kVariantRegistered = tune::add({
    "factorials.variant",
    [] { variants.set(measure(kDefaults, kRunTime)); },
    [] { return text(variants.get()); },
    [](std::string_view s, bool apply) {
        const auto choices = parse(s);
        if (choices && apply) variants.set(*choices);
        return choices.has_value();
    },
});

#endif

}  // namespace fps::factorials
