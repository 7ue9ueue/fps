// tools/tune.py ntt: measures the NTT variant per op and length (select.hpp) on this machine and
// prints it as compiled defaults for tuned.hpp ("field <name> <type> <values>"), with diagnostics
// ("# ...").
//   1. measure() (tune.hpp) at every length 2^6 .. 2^25: median of 7, samples of at least 1 ms. The
//      compiled table's candidate stays unless another is more than 1% faster (a rerun does not flip
//      choices by noise).
//   2. Per op and lg the candidates' geometric mean of time / fastest and their times per point;
//      then the points the compiled and the new table send to a candidate more than 2% slower than
//      the fastest there.
//   3. measure_ragged(): the plan kinds per main length and bucket of r / L (ragged.hpp; "truncated/
//      floor": products truncated, the others floor; then the middle products' kind), every main
//      length (scalar and sse: to 2^14), 5 samples of at least 1 ms, margin 1%; the times per kind,
//      products then middle products (at the bucket's middle / its first length). Fields ragged and
//      middle: the two tables of Kinds.
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

    // The ragged plans: every main length on targets with a timing machine; scalar and sse to 2^14.
    const bool slow = fps::tune::kTarget == fps::tune::Target::scalar || fps::tune::kTarget == fps::tune::Target::sse;
    std::vector<RaggedTimes> ragged_log;
    const auto kinds = measure_ragged(detail::ragged::kKindDefaults, {slow ? 14 : kMaxLog - 1, 5, 1e6, 0.01}, &ragged_log);
    constexpr const char* kind_names[] = {"plain", "top", "floor", "ceil", "truncated"};
    const auto entry_name = [&](int v) {
        const int product = detail::ragged::product_kind(v), other = detail::ragged::other_kind(v);
        return std::string(kind_names[product]) + (product == other ? "" : std::string("/") + kind_names[other]);
    };
    for (const RaggedTimes& t : ragged_log) {
        const int i = (t.lg - kMinLog) * detail::ragged::kBuckets + t.bucket - 1, im = detail::ragged::kTable + i;
        std::printf("# ragged %d %d: %s (compiled %s)", t.lg, t.bucket, entry_name(kinds[i]).c_str(),
                    entry_name(detail::ragged::kKindDefaults[i]).c_str());
        for (std::size_t k = 0; k < t.kinds.size(); ++k) std::printf(" %s %.1f", kind_names[t.kinds[k]], t.ns[k]);
        std::printf("\n# middle %d %d: %s (compiled %s)", t.lg, t.bucket, kind_names[detail::ragged::middle_kind(kinds[im])],
                    kind_names[detail::ragged::middle_kind(detail::ragged::kKindDefaults[im])]);
        for (std::size_t k = 0; k < t.kinds.size(); ++k)
            std::printf(" %s %.1f/%.1f", kind_names[t.kinds[k]], t.middle[k], t.first[k]);  // the middle, the first length
        std::printf("\n");
    }
    std::printf("# seconds %.1f\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count());

    std::printf("field signature int %u\n", kSignature);
    for (int op = 0; op < kOps; ++op) {
        std::printf("field %s int", std::string(op_name(Op(op))).c_str());
        for (const std::uint8_t v : choices[op]) std::printf(" %d", v);
        std::printf("\n");
    }
    std::printf("field ragged int");
    for (int i = 0; i < detail::ragged::kTable; ++i) std::printf(" %d", kinds[i]);
    std::printf("\nfield middle int");
    for (int i = 0; i < detail::ragged::kTable; ++i) std::printf(" %d", kinds[detail::ragged::kTable + i]);
    std::printf("\n");
    return 0;
}
