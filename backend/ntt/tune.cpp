// tools/tune.py ntt: measures the NTT variant per op and length (select.hpp) on this machine and
// prints it as compiled defaults for tuned.hpp ("field <name> <type> <values>"), with diagnostics
// ("# ...").
//   1. measure() (tune.hpp) at every length 2^6 .. 2^25: median of 7, samples of at least 1 ms. The
//      compiled table's candidate stays unless another is more than 1% faster (a rerun does not flip
//      choices by noise).
//   2. Per op and lg the candidates' geometric mean of time / fastest and their times per point;
//      then the points the compiled and the new table send to a candidate more than 2% slower than
//      the fastest there.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "backend/ntt/tune.hpp"

namespace {

using namespace fps::ntt;

void misses(const char* name, const Choices& choices, const std::vector<Times>& log) {
    int count = 0, total = 0;
    double worst = 1, log_sum = 0;
    for (const Times& t : log)
        for (const std::vector<double>& ns : t.ns) {
            const double ratio = ns[choices[t.op][t.lg - kMinLog]] / *std::ranges::min_element(ns);
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
    std::vector<Times> log;
    const Choices choices = measure(kDefaults, {kMaxLog, 7, 1e6, 0.01}, &log);
    for (const Times& t : log) {
        const int i = t.lg - kMinLog;
        std::printf("# %s %d: %s (compiled %s)", std::string(op_name(t.op)).c_str(), t.lg, names[choices[t.op][i]].c_str(),
                    names[kDefaults[t.op][i]].c_str());
        for (int v = 0; v < kCandidates; ++v) {
            double score = 0;
            for (const std::vector<double>& ns : t.ns)
                score += std::log(ns[v] / *std::ranges::min_element(ns)) / double(t.ns.size());
            std::printf(" %s %.3f", names[v].c_str(), std::exp(score));
        }
        std::printf("\n");
        for (const std::vector<double>& ns : t.ns) {
            std::printf("#  ");
            for (const double x : ns) std::printf(" %.1f", x);
            std::printf("\n");
        }
    }
    if (kCandidates > 1) {
        misses("compiled", kDefaults, log);
        misses("new", choices, log);
    }
    std::printf("# seconds %.1f\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count());

    std::printf("field signature int %u\n", kSignature);
    for (int op = 0; op < kOps; ++op) {
        std::printf("field %s int", std::string(op_name(Op(op))).c_str());
        for (const std::uint8_t v : choices[op]) std::printf(" %d", v);
        std::printf("\n");
    }
    return 0;
}
