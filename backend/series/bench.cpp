// Times backend/series; tools/kernel.py runs it. Items are coefficients of the result.
//   "<op>/<lg>"              inverse, log, exp of n = 2^(lg-1) random coefficients into kept memory
//   "<op>_ragged/<lg>"       n = 2^(lg-1) + 2^(lg-3): the last step of exp at half length
//   "multiply@ntt/<lg>"      the unit: ntt::multiply of two factors of n into n words (not gated)
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/series/series.hpp"

namespace {

constexpr std::uint32_t P = 998244353;
using Vec = std::vector<std::uint32_t>;

Vec random_vec(std::size_t n, std::uint32_t first, unsigned seed) {
    std::mt19937 rng(seed);
    Vec a(n);
    for (auto& x : a) x = rng() % P;
    if (n) a[0] = first;
    return a;
}

template <class Op>
void add(bench::Suite& suite, const std::string& name, std::size_t n, std::uint32_t first, Op op) {
    suite.add(name, [n, first, op] {
        return [f = random_vec(n, first, unsigned(n)), out = Vec(n), op]() mutable {
            op(f, out);
            bench::keep(out);
        };
    }, n);
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    const auto inverse = [](const Vec& f, Vec& out) { fps::series::inverse(f, out); };
    const auto log = [](const Vec& f, Vec& out) { fps::series::log(f, out); };
    const auto exp = [](const Vec& f, Vec& out) { fps::series::exp(f, out); };
    for (int lg : {7, 11, 14, 18, 22}) {
        const std::size_t n = std::size_t(1) << (lg - 1), ragged = n + n / 4;
        const std::string size = "/" + std::to_string(lg);
        add(suite, "inverse" + size, n, 5, inverse);
        add(suite, "log" + size, n, 1, log);
        add(suite, "exp" + size, n, 0, exp);
        if (lg >= 11) {
            add(suite, "inverse_ragged" + size, ragged, 5, inverse);
            add(suite, "log_ragged" + size, ragged, 1, log);
            add(suite, "exp_ragged" + size, ragged, 0, exp);
        }
        suite.add("multiply@ntt" + size, [n, lg] {
            return [f = random_vec(n, 1, unsigned(lg)), g = random_vec(n, 2, unsigned(lg + 1)), out = Vec(n)]() mutable {
                fps::ntt::multiply(f, g, out);
                bench::keep(out);
            };
        }, n);
    }
    return suite.run();
}
