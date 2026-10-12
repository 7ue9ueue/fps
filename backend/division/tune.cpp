// tools/tune.py division: measures the division's cost model on this machine (tune.hpp's
// measure_costs up to 2^25, 7 samples of >= 1 ms, a 2% margin against the compiled table) and prints
// it as compiled defaults for tuned.hpp ("field <name> double <values>"), with diagnostics ("# ...").
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <string>

#include "backend/division/tune.hpp"

int main() {
    using namespace fps::division;
    const auto start_time = std::chrono::steady_clock::now();
    std::printf("# cpu %s\n# target %s\n", fps::tune::cpu_name().c_str(), std::string(fps::tune::target_name()).c_str());
    const Costs c = measure_costs(kCostDefaults, {25, 7, 1e6, 0.02});
    std::printf("# seconds %.1f\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count());
    const char* rows[kRows] = {"forward", "cyclic", "product", "term", "leaf", "inverse"};
    for (int r = 0; r < kRows; ++r) {
        std::printf("field %s double", rows[r]);
        for (int i = 0; i < kLgs; ++i) std::printf(" %.4g", c[std::size_t(r * kLgs + i)]);
        std::printf("\n");
    }
    std::printf("field pass double %.4g\nfield long_term double %.4g\nfield long_step double %.4g\n", c[kPass],
                c[kLongTerm], c[kLongStep]);
}
