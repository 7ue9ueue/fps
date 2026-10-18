// Times the small/NTT choice of fps/products.hpp's square and middle_product near where it switches
// on the timed targets (issue #271), and f * f, a square since #271. Each API case has the two
// paths it chooses between on the same inputs, into new memory as the API's result:
//   "square/<n>x<n>"                 fps::square, f of n coefficients (gated)
//   "square@small/<n>x<n>"           small::multiply(f, f)
//   "square@ntt/<n>x<n>"             ntt::square
//   "middle_product/<n>x<m>"         fps::middle_product, f of n and g of m coefficients (gated)
//   "middle_product@small/<n>x<m>"   small::multiply(f, g), then the n - m + 1 words moved down
//   "middle_product@ntt/<n>x<m>"     ntt::middle_product
//   "multiply_self/<lg>"             f * f, f of 2^lg / 2 coefficients (gated)
//   "multiply_self@kernel/<lg>"      ntt::square, the backend call f * f runs
// The shapes: where #271's choice differs from beats_ntt's on some timed target (before #271 both
// went by small::beats_ntt(n, m): ntt::multiply's time at n + m - 1 words), and long factors.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "backend/bench.hpp"
#include "fps/fps.hpp"

namespace {

constexpr std::uint32_t P = 998244353;
using Poly = fps::Polynomial<P>;
using Words = fps::detail::Words;

Poly random_poly(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<std::uint32_t> c(n);
    for (std::uint32_t& x : c) x = rng() % P;
    return Poly(std::span<const std::uint32_t>(c));
}

std::string shape(std::size_t n, std::size_t m) { return "/" + std::to_string(n) + "x" + std::to_string(m); }

void squares(bench::Suite& suite) {
    for (const std::size_t n : {32, 44, 54, 56, 60, 72, 74, 80, 84, 88, 96, 100}) {
        const std::string size = shape(n, n);
        const unsigned seed = unsigned(3 * n);
        suite.add("square" + size, [n, seed] {
            return [f = random_poly(n, seed)] { bench::keep(fps::square(f)); };
        }, n * n);
        suite.add("square@small" + size, [n, seed] {
            return [f = random_poly(n, seed), n] {
                Words h(2 * n - 1, Words::Unwritten{});
                fps::small::multiply<P>(f.span(), f.span(), {h.data(), 2 * n - 1});
                bench::keep(h);
            };
        }, n * n);
        suite.add("square@ntt" + size, [n, seed] {
            return [f = random_poly(n, seed), n] {
                Words h(2 * n - 1, Words::Unwritten{});
                fps::ntt::square(f.span(), {h.data(), 2 * n - 1});
                bench::keep(h);
            };
        }, n * n);
    }
}

void middle_products(bench::Suite& suite) {
    constexpr std::size_t k20 = std::size_t(1) << 20;
    for (const auto& [n, m] : std::vector<std::pair<std::size_t, std::size_t>>{
             {40, 40}, {50, 40}, {56, 56}, {60, 48}, {64, 32}, {72, 72}, {96, 96}, {112, 56}, {120, 60}, {168, 28},
             {192, 64}, {256, 64}, {336, 56}, {1024, 48}, {4096, 56}, {k20, 64}, {k20, 96}, {k20 - 48, 96}}) {
        const std::string size = shape(n, m);
        const unsigned seed = unsigned(n + 5 * m);
        const std::size_t count = n - m + 1;
        suite.add("middle_product" + size, [n, m, seed] {
            return [f = random_poly(n, seed), g = random_poly(m, seed + 1)] { bench::keep(fps::middle_product(f, g)); };
        }, n * m);
        suite.add("middle_product@small" + size, [n, m, seed, count] {
            return [f = random_poly(n, seed), g = random_poly(m, seed + 1), n, m, count] {
                Words h(count, n + m - 1, Words::Unwritten{});
                fps::small::multiply<P>(f.span(), g.span(), {h.data(), n + m - 1});
                std::memmove(h.data(), h.data() + (m - 1), count * sizeof(std::uint32_t));
                bench::keep(h);
            };
        }, n * m);
        suite.add("middle_product@ntt" + size, [n, m, seed, count] {
            return [f = random_poly(n, seed), g = random_poly(m, seed + 1), count] {
                Words h(count, Words::Unwritten{});
                fps::ntt::middle_product(f.span(), g.span(), {h.data(), count});
                bench::keep(h);
            };
        }, n * m);
    }
}

void self_products(bench::Suite& suite) {
    for (const int lg : {11, 20}) {
        const std::size_t n = std::size_t(1) << (lg - 1);
        const unsigned seed = unsigned(7 * lg);
        suite.add("multiply_self/" + std::to_string(lg), [n, seed] {
            return [f = random_poly(n, seed)] {
                const Poly h = f * f;
                bench::keep(h);
            };
        }, 2 * n);
        suite.add("multiply_self@kernel/" + std::to_string(lg), [n, seed] {
            return [f = random_poly(n, seed), n] {
                Words h(2 * n - 1, Words::Unwritten{});
                fps::ntt::square(f.span(), {h.data(), 2 * n - 1});
                bench::keep(h);
            };
        }, 2 * n);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    squares(suite);
    middle_products(suite);
    self_products(suite);
    return suite.run();
}
