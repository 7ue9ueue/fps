// Times backend/series' inverse, log and pow at the powers of two the other benches skip (issue
// #251), a binary of its own; tools/kernel.py runs it. With bench.cpp, bench_pow_sqrt.cpp and
// bench_ragged.cpp every n = 2^12 .. 2^21 is timed for these three ops.
//   "<op>/<lg>"   n = 2^(lg-1) (bench.cpp's names), lg = 15, 16, 19, 20; op: inverse, log, pow
//                 (f[0] = 5, 1, 5; pow's exponent and constant of bench_pow_sqrt.cpp), of random
//                 f into kept memory
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
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
    using Op = void (*)(const Vec&, Vec&);
    const std::pair<const char*, std::pair<std::uint32_t, Op>> ops[] = {
        {"inverse", {5, [](const Vec& f, Vec& out) { fps::series::inverse(f, out); }}},
        {"log", {1, [](const Vec& f, Vec& out) { fps::series::log(f, out); }}},
        {"pow", {5, [](const Vec& f, Vec& out) { fps::series::pow(f, 0x2545f491u % P, 0x3c6ef372u % P, out); }}},
    };
    for (const auto& [op, spec] : ops)
        for (const int lg : {15, 16, 19, 20})
            add(suite, std::string(op) + "/" + std::to_string(lg), std::size_t(1) << (lg - 1), spec.first, spec.second);
    return suite.run();
}
