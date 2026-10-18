// Times backend/series at ragged n (issue #214), a binary of its own; tools/kernel.py runs it. Items
// are coefficients of the result, so ns per item shows the staircase at powers of two.
//   "<op>/<k>+1", "<op>/<k>.25", "<op>/<k>.5", "<op>/<k>.75"
//                       n = 2^k + 1, 1.25, 1.5, 1.75 2^k (BACKEND.md's ragged names: k is the power
//                       of two below n), k = 12, 16, 20
//   "<op>/<k + 1>"      n = 2^k, the power of two below them (bench.cpp's names: <op>/<lg> has n =
//                       2^(lg-1)); <op>/22 there is n = 2^21
//   "<op>/6e5"          n = 600000 (1.14 2^19; the frontend's inverse/6e5, backend/ntt issue #155)
// op: inverse, log, exp (f[0] = 5, 1, 0), sqrt (f[0] = 4, root 2) and pow (f[0] = 5, the exponent
// and constant of bench_pow_sqrt.cpp), of random f into kept memory.
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
        {"exp", {0, [](const Vec& f, Vec& out) { fps::series::exp(f, out); }}},
        {"sqrt", {4, [](const Vec& f, Vec& out) { fps::series::sqrt(f, 2, out); }}},
        {"pow", {5, [](const Vec& f, Vec& out) { fps::series::pow(f, 0x2545f491u % P, 0x3c6ef372u % P, out); }}},
    };
    std::vector<std::pair<std::string, std::size_t>> sizes;
    for (const int k : {12, 16, 20}) {
        const std::size_t L = std::size_t(1) << k;
        const std::string lg = std::to_string(k);
        sizes.insert(sizes.end(), {{std::to_string(k + 1), L},
                                   {lg + "+1", L + 1},
                                   {lg + ".25", L + L / 4},
                                   {lg + ".5", L + L / 2},
                                   {lg + ".75", 2 * L - L / 4}});
    }
    sizes.emplace_back("6e5", 600000);
    for (const auto& [op, spec] : ops)
        for (const auto& [size, n] : sizes) add(suite, std::string(op) + "/" + size, n, spec.first, spec.second);
    return suite.run();
}
