// Measures the NTT variant per op and length (select.hpp) on this machine and registers it with
// backend/tune as the choice "ntt.variant": tune::run() measures, tune::save() / load() keep it.
// backend/small/tune.hpp includes this header, so "ntt.variant" is measured before "small.ntt",
// which times the NTT.
//
//   ntt::measure(start, budget[, log])   // Choices: per op and lg the fastest candidate, else start's
//   ntt::text(choices), parse(text)      // wisdom: kSignature, then the candidate per op and lg
//   ntt::variant_names()                 // the candidates' names, e.g. "intrinsics_avx512"
//   ntt::op_name(op)                     // "forward", "inverse", "multiply", "product"
//
// measure: for each op and lg up to budget.max_lg, every candidate is timed in turn (tune::time_each)
// the way the library runs it, at the op's points: forward and inverse on a Transform and an array
// kept across calls; multiply on one Convolution, which takes this thread's kept region (memory.hpp),
// with factors of L/2 and L/2 and of 3L/4 and L/4 + 1 coefficients; product by multiply(f, g, out)
// on the same two shapes, out the whole product. The candidate with the least geometric mean of
// time / fastest over the points wins, unless start's is within budget.margin of it. Past max_lg
// each op keeps start's, unless its choice at max_lg changed: then every longer length takes the new
// one (a table made on another CPU).
// kRunTime, the budget of tune::run(): its cost and decisions are in backend/tune/notes.md.
// tools/tune.py runs tune.cpp, which measures every length at a larger budget.
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
#include <vector>

#include "backend/ntt/ntt.hpp"
#include "backend/tune/tune.hpp"

namespace fps::ntt {

struct Budget {
    int max_lg;            // lengths 2^kMinLog .. 2^max_lg measured
    int samples;           // per point and candidate
    double min_sample_ns;  // each sample at least this long
    double margin;         // keep start's candidate when within it
};

// Lengths to 2^12, 5 samples of 100 us, margin 3%: 55-57 ms on Emerald Rapids and Zen 5. Samples
// of 10 us cost 4 ms, but one run in five on Emerald Rapids changed 7-8 entries and lost 0.7% on
// its own machine (backend/tune/notes.md).
inline constexpr Budget kRunTime = {12, 5, 1e5, 0.03};

inline constexpr std::string_view op_name(Op op) {
    constexpr std::string_view names[kOps] = {"forward", "inverse", "multiply", "product"};
    return names[op];
}

// What measure() timed at one op and length: ns[point][candidate].
struct Times {
    Op op;
    int lg;
    std::vector<std::vector<double>> ns;
};

namespace detail {

// f.template operator()<V>() for candidate v of the list. For measurements only: the library's
// dispatch is run_forward and its siblings (select.hpp).
template <class... V, class F>
void with_candidate(Candidates<V...>, int v, F f) {
    int i = 0;
    (void)((i++ == v && (f.template operator()<V>(), true)) || ...);
}

struct alignas(64) Line {
    std::uint32_t words[16];
};

// count random words < P, 64-byte aligned, then kPadding words.
inline std::vector<Line> random_lines(std::size_t count) {
    std::vector<Line> lines((count + kPadding + 15) / 16);
    std::mt19937 rng{unsigned(count)};
    for (Line& line : lines)
        for (std::uint32_t& w : line.words) w = rng() % kModulus;
    return lines;
}

template <class Run>
std::vector<double> time_candidates(const Budget& budget, Run run) {
    std::vector<int> ids(kCandidates);
    for (int v = 0; v < kCandidates; ++v) ids[v] = v;
    return tune::time_each(budget.samples, budget.min_sample_ns, ids, [&](int v) { with_candidate(Selected{}, v, run); });
}

// ns[point][candidate] for op at 2^lg.
inline std::vector<std::vector<double>> time_op(Op op, int lg, const Budget& budget) {
    const std::size_t len = std::size_t(1) << lg;
    const auto kept = [](const void* p) { asm volatile("" : : "r"(p) : "memory"); };
    std::vector<std::vector<double>> ns;
    if (op == kForward || op == kInverse) {
        const Transform t(lg);
        std::vector<Line> f = random_lines(len);
        std::uint32_t* const data = f.data()->words;
        ns.push_back(time_candidates(budget, [&]<class V> {
            if (op == kForward) t.forward<V>(data);
            else t.inverse<V>(data);
            kept(data);
        }));
        return ns;
    }
    for (const auto& [n, m] : {std::pair{len / 2, len / 2}, std::pair{len - len / 4, len / 4 + 1}}) {
        const std::vector<Line> a = random_lines(n), b = random_lines(m);
        if (op == kMultiply) {
            Convolution conv(n, m);
            std::copy_n(a.data()->words, n, conv.a());
            std::copy_n(b.data()->words, m, conv.b());
            ns.push_back(time_candidates(budget, [&]<class V> { kept(conv.multiply<V>()); }));
        } else {
            std::vector<std::uint32_t> out(n + m - 1);
            const std::span<const std::uint32_t> f(a.data()->words, n), g(b.data()->words, m);
            ns.push_back(time_candidates(budget, [&]<class V> {
                multiply<V>(f, g, out);
                kept(out.data());
            }));
        }
    }
    return ns;
}

}  // namespace detail

inline Choices measure(const Choices& start, const Budget& budget, std::vector<Times>* log = nullptr) {
    Choices choices = start;
    if (kCandidates == 1) return choices;
    for (int op = 0; op < kOps; ++op) {
        for (int lg = kMinLog; lg <= budget.max_lg; ++lg) {
            const auto ns = detail::time_op(Op(op), lg, budget);
            std::vector<double> score(kCandidates);
            for (const std::vector<double>& t : ns) {
                const double fastest = *std::ranges::min_element(t);
                for (int v = 0; v < kCandidates; ++v) score[v] += std::log(t[v] / fastest) / double(ns.size());
            }
            const int best = int(std::ranges::min_element(score) - score.begin()), i = lg - kMinLog;
            if (score[start[op][i]] > score[best] + std::log1p(budget.margin)) choices[op][i] = std::uint8_t(best);
            if (log) log->push_back({Op(op), lg, ns});
        }
        const int last = budget.max_lg - kMinLog;
        if (choices[op][last] != start[op][last])
            std::fill(choices[op].begin() + last + 1, choices[op].end(), choices[op][last]);
    }
    return choices;
}

inline std::vector<std::string> variant_names() {
    return []<class... V>(Candidates<V...>) { return std::vector<std::string>{variant_name<V>()...}; }(Selected{});
}

// Wisdom text: kSignature, then the candidate per op (forward, inverse, multiply, product) and lg.
inline std::string text(const Choices& choices) {
    std::array<unsigned, kOps * kLengths + 1> values{kSignature};
    for (int op = 0; op < kOps; ++op)
        for (int i = 0; i < kLengths; ++i) values[1 + op * kLengths + i] = choices[op][i];
    return tune::text(values);
}

// The choices of a wisdom text: nothing unless it has this list's signature and every candidate exists.
inline std::optional<Choices> parse(std::string_view s) {
    const auto values = tune::parse<std::array<unsigned, kOps * kLengths + 1>>(s);
    if (!values || (*values)[0] != kSignature) return std::nullopt;
    Choices choices{};
    for (int op = 0; op < kOps; ++op)
        for (int i = 0; i < kLengths; ++i) {
            const unsigned v = (*values)[1 + op * kLengths + i];
            if (v >= unsigned(kCandidates)) return std::nullopt;
            choices[op][i] = std::uint8_t(v);
        }
    return choices;
}

#ifndef FPS_NO_TUNER

inline const bool kVariantRegistered = tune::add({
    "ntt.variant",
    [] { variants.set(measure(kDefaults, kRunTime)); },
    [] { return text(variants.get()); },
    [](std::string_view s, bool apply) {
        const auto choices = parse(s);
        if (choices && apply) variants.set(*choices);
        return choices.has_value();
    },
});

#endif

}  // namespace fps::ntt
