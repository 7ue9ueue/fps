// Times the frontend's API modulo P = 998244353; tools/kernel.py runs it. Items are coefficients
// or table entries.
//   "<op>/<lg>"          through the API, on 2^lg items (gated)
//   "<op>@kernel/<lg>"   the backend call the API wraps, on the same inputs
// Factorials wraps no backend kernel yet (frontend/notes.md), so its cases have no @kernel pair.
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "backend/bench.hpp"
#include "fps/fps.hpp"

namespace {

constexpr std::uint32_t P = 998244353;

// Builds the tables: 2^lg products, one inverse, 2^lg products, two vectors of 2^lg words.
void factorials(bench::Suite& suite) {
    for (int lg : {10, 14, 18, 22}) {
        const std::size_t n = std::size_t(1) << lg;
        suite.add("factorials/" + std::to_string(lg), [n] {
            return [n] { bench::keep(fps::Factorials<P>(n)); };
        }, n);
    }
}

// 2^lg binomials C(m, k), k <= m < 2^lg random, from tables of 2^lg (512 KiB at lg = 16).
void binomial(bench::Suite& suite) {
    const int lg = 16;
    const std::size_t n = std::size_t(1) << lg;
    suite.add("binomial/" + std::to_string(lg), [n] {
        std::mt19937_64 rng(1);
        std::vector<std::pair<std::size_t, std::size_t>> queries(n);
        for (auto& [m, k] : queries) m = rng() % n, k = rng() % (m + 1);
        return [fact = fps::Factorials<P>(n), queries] {
            std::uint32_t sum = 0;
            for (const auto& [m, k] : queries) sum += fact.binomial(m, k);
            bench::keep(sum);
        };
    }, n);
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    factorials(suite);
    binomial(suite);
    return suite.run();
}
