// Times fps/points.hpp modulo P = 998244353; tools/kernel.py runs it, a binary of its own. Items
// are points.
//   "<op>/<lg>"          through the API (gated): f of n = 2^lg coefficients, 2^lg random points
//   "<op>@kernel/<lg>"   backend/evaluation on the same inputs into a kept output
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "fps/fps.hpp"

namespace {

constexpr std::uint32_t P = 998244353;
using Poly = fps::Polynomial<P>;
using Vec = std::vector<std::uint32_t>;

Vec random_vec(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    Vec a(n);
    for (std::uint32_t& x : a) x = rng() % P;
    return a;
}

// n distinct random points, in random order.
Vec distinct_points(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    Vec x;
    while (x.size() < n) {
        while (x.size() < n) x.push_back(rng() % P);
        std::ranges::sort(x);
        x.erase(std::unique(x.begin(), x.end()), x.end());
    }
    std::ranges::shuffle(x, rng);
    return x;
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    for (int lg : {10, 14, 17}) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        const unsigned seed = unsigned(11 * lg);
        suite.add("evaluate" + size, [n, seed] {
            return [f = Poly(std::span<const std::uint32_t>(random_vec(n, seed))), x = random_vec(n, seed + 1)] {
                bench::keep(fps::evaluate(f, x));
            };
        }, n);
        suite.add("interpolate" + size, [n, seed] {
            return [x = distinct_points(n, seed + 2), y = random_vec(n, seed + 3)] {
                bench::keep(fps::interpolate<P>(x, y));
            };
        }, n);
        suite.add("newton" + size, [n, seed] {
            return [f = Poly(std::span<const std::uint32_t>(random_vec(n, seed + 5))), x = random_vec(n, seed + 6)] {
                bench::keep(fps::to_newton_basis(f, x));
            };
        }, n);
        suite.add("evaluate@kernel" + size, [n, seed] {
            return [f = random_vec(n, seed), x = random_vec(n, seed + 1), out = Vec(n)]() mutable {
                fps::evaluation::evaluate(f, x, out);
                bench::keep(out);
            };
        }, n);
        suite.add("interpolate@kernel" + size, [n, seed] {
            return [x = distinct_points(n, seed + 2), y = random_vec(n, seed + 3), out = Vec(n)]() mutable {
                fps::evaluation::interpolate(x, y, out);
                bench::keep(out);
            };
        }, n);
        suite.add("newton@kernel" + size, [n, seed] {
            return [f = random_vec(n, seed + 5), x = random_vec(n, seed + 6), out = Vec(n)]() mutable {
                fps::evaluation::to_newton_basis(f, std::span<const std::uint32_t>(x).first(x.size() - 1), out);
                bench::keep(out);
            };
        }, n);
    }
    return suite.run();
}
