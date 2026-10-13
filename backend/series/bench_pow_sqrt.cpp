// Times backend/series pow and sqrt (issue #107), a binary of its own (bench.cpp's code generation
// stays as it was); tools/kernel.py runs it. Items are coefficients of the result.
//   "<op>/<lg>"              pow, sqrt of n = 2^(lg-1) random coefficients into kept memory (pow:
//                            f[0] = 5, a random exponent and constant; sqrt: f[0] = 4, root 2)
//   "<op>_ragged/<lg>"       n = 2^(lg-1) + 2^(lg-3): pow's last exp step without block 3, sqrt's
//                            last step with one cyclic product
// Against lc-opt and the frontend's pow and sqrt in the same run: notes/107-r1.md.
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "backend/bench.hpp"
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
    const auto pow = [](const Vec& f, Vec& out) { fps::series::pow(f, 0x2545f491u % P, 0x3c6ef372u % P, out); };
    const auto sqrt = [](const Vec& f, Vec& out) { fps::series::sqrt(f, 2, out); };
    for (int lg : {7, 11, 14, 18, 22}) {
        const std::size_t n = std::size_t(1) << (lg - 1), ragged = n + n / 4;
        const std::string size = "/" + std::to_string(lg);
        add(suite, "pow" + size, n, 5, pow);
        add(suite, "sqrt" + size, n, 4, sqrt);
        if (lg >= 11) {
            add(suite, "pow_ragged" + size, ragged, 5, pow);
            add(suite, "sqrt_ragged" + size, ragged, 4, sqrt);
        }
    }
    return suite.run();
}
