// tools/tune.py factorials: measures the variant per size class (select.hpp) on this machine and
// prints it as compiled defaults for tuned.hpp ("field <name> <type> <values>"), with diagnostics
// ("# ...").
//   1. measure() (tune.hpp) on every class ending at most 2^24: 4 points per class, median of 7,
//      samples of at least 1 ms. The compiled table's variant stays unless another is more than 2%
//      faster (a rerun does not flip choices by noise). Classes above 2^24 take the last measured
//      class's.
//   2. Per class the variants' geometric mean of time / fastest; then, over the measured points,
//      the points the compiled table and the new one send the slower way (> 2%), against the
//      fastest variant there.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "backend/factorials/tune.hpp"

namespace {

using namespace fps::factorials;

// Points where the table's variant is more than 2% slower than the fastest; the worst ratio; the
// geometric mean of chosen / fastest over all points.
void misses(const char* name, const Choices& choices, const std::vector<ClassTimes>& log) {
    int count = 0, total = 0;
    double worst = 1, log_sum = 0;
    for (const ClassTimes& t : log)
        for (std::size_t p = 0; p < t.n.size(); ++p) {
            const auto chosen = std::ranges::find(t.ids, int(choices[t.c]));
            if (chosen == t.ids.end()) continue;  // dropped: counted where it was timed
            const double ratio = t.ns[p][chosen - t.ids.begin()] / *std::ranges::min_element(t.ns[p]);
            ++total;
            log_sum += std::log(ratio);
            worst = std::max(worst, ratio);
            count += ratio > 1.02;
        }
    std::printf("# %s: %d of %d points more than 2%% slower than the fastest, worst %.3fx, geomean %.4f\n", name, count,
                total, worst, std::exp(log_sum / std::max(total, 1)));
}

}  // namespace

int main() {
    std::printf("# cpu %s\n# target %s\n", fps::tune::cpu_name().c_str(), std::string(fps::tune::target_name()).c_str());
    const auto start_time = std::chrono::steady_clock::now();
    const std::vector<std::string> names = variant_names();
    constexpr int kMaxLg = 24;
    std::vector<ClassTimes> log;
    Choices choices = measure(kDefaults, {kMaxLg, kMaxLg, 4, 7, 1e6, 0.02, 0.02}, &log);
    const int last = log.back().c;
    for (int c = last + 1; c < kClasses; ++c) choices[c] = choices[last];

    for (const ClassTimes& t : log) {
        std::printf("# class %d (%zu, %zu]: %s (compiled %s)", t.c, class_start(t.c) - 1, class_end(t.c),
                    names[choices[t.c]].c_str(), names[kDefaults[t.c]].c_str());
        for (std::size_t i = 0; i < t.ids.size(); ++i) {
            double score = 0;
            for (std::size_t p = 0; p < t.n.size(); ++p)
                score += std::log(t.ns[p][i] / *std::ranges::min_element(t.ns[p])) / double(t.n.size());
            std::printf(" %s %.3f", names[t.ids[i]].c_str(), std::exp(score));
        }
        std::printf("\n");
        for (std::size_t p = 0; p < t.n.size(); ++p) {
            std::printf("#   n %zu:", t.n[p]);
            for (std::size_t i = 0; i < t.ids.size(); ++i) std::printf(" %.1f", t.ns[p][i]);
            std::printf("\n");
        }
    }
    misses("compiled", kDefaults, log);
    misses("new", choices, log);
    std::printf("# seconds %.1f\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count());

    std::printf("field signature int %u\n", kSignature);
    std::printf("field variant int");
    for (const std::uint8_t v : choices) std::printf(" %d", v);
    std::printf("\n");
    return 0;
}
