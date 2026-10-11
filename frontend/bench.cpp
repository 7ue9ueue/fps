// Times the frontend's API modulo P = 998244353; tools/kernel.py runs it. Items are coefficients
// or table entries.
//   "<op>/<lg>"          through the API, on 2^lg items (gated)
//   "<op>@kernel/<lg>"   the backend call the API wraps, on the same inputs
// Factorials wraps no backend kernel yet (frontend/notes.md), so its cases have no @kernel pair.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "backend/bench.hpp"
#include "fps/fps.hpp"

namespace {

constexpr std::uint32_t P = 998244353;
using Poly = fps::Polynomial<P>;

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

std::shared_ptr<const Poly> random_poly(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<std::uint32_t> c(n);
    for (std::uint32_t& x : c) x = rng() % P;
    return std::make_shared<const Poly>(std::span<const std::uint32_t>(c));
}

// f * g for f and g of 2^lg / 2 coefficients: schoolbook below lg = 10, then ntt::Convolution.
// The kernel: construct an ntt::Convolution, copy the factors in, multiply. The scalar NTT takes
// seconds at 2^22, so on scalar targets the cases stop at 2^18.
void multiply(bench::Suite& suite) {
    const int max_lg = std::is_same_v<fps::ntt::SelectedMultiply, fps::ntt::variant::scalar> ? 18 : 22;
    for (int lg = 6; lg <= max_lg; ++lg) {
        const std::size_t n = std::size_t(1) << (lg - 1);
        const auto f = random_poly(n, 2 * lg), g = random_poly(n, 2 * lg + 1);
        const std::string size = "/" + std::to_string(lg);
        suite.add("multiply" + size, [f, g] {
            return [f, g] {
                const Poly h = *f * *g;
                bench::keep(h);
            };
        }, 2 * n);
        suite.add("multiply@kernel" + size, [f, g] {
            return [f, g] {
                fps::ntt::Convolution conv(f->size(), g->size());
                std::ranges::copy(f->span(), conv.a());
                std::ranges::copy(g->span(), conv.b());
                bench::keep(*conv.multiply());
            };
        }, 2 * n);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    factorials(suite);
    binomial(suite);
    multiply(suite);
    return suite.run();
}
