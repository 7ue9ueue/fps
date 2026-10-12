// Times fps/division.hpp modulo P = 998244353; tools/kernel.py runs it, a binary of its own. Items
// are coefficients of f.
//   "<op>/<lg>"   through the API (gated)
// The backend has no division kernel yet (backend issue #102 moves lc-opt's), so there are no
// "<op>@kernel" pairs: the general path runs fps/series.hpp's inverse and two products
// (raw::mul), whose own pair is mul / mul@kernel in bench.cpp. frontend/notes.md gives each case
// in units of mul@kernel/<lg> (ntt::multiply of two factors of n into n words) from the same run.
#include <cstddef>
#include <cstdint>
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

// n random coefficients, the last one nonzero.
Poly random_poly(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<std::uint32_t> a(n);
    for (std::uint32_t& x : a) x = rng() % P;
    if (n && a.back() == 0) a.back() = 1;
    return Poly(std::span<const std::uint32_t>(a));
}

// f of n = 2^lg coefficients by g of n / 2 (a quotient of n / 2 + 1 coefficients, a remainder of
// n / 2 - 1), and divmod_long by g of n / 64 (a long quotient), as backend issue #102 times them:
// both on the Newton path.
void general(bench::Suite& suite) {
    for (int lg : {11, 14, 18, 22}) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        const unsigned seed = unsigned(5 * lg);
        const auto inputs = [n, seed](std::size_t m) {
            return std::pair{random_poly(n, seed), random_poly(m, seed + 1)};
        };
        suite.add("divmod" + size, [inputs, n] {
            return [fg = inputs(n / 2)] { bench::keep(fps::divmod(fg.first, fg.second)); };
        }, n);
        suite.add("divide" + size, [inputs, n] {
            return [fg = inputs(n / 2)] { bench::keep(fg.first / fg.second); };
        }, n);
        suite.add("modulo" + size, [inputs, n] {
            return [fg = inputs(n / 2)] { bench::keep(fg.first % fg.second); };
        }, n);
        if (lg >= 14)
            suite.add("divmod_long" + size, [inputs, n] {
                return [fg = inputs(n / 64)] { bench::keep(fps::divmod(fg.first, fg.second)); };
            }, n);
    }
}

// f of n = 2^lg coefficients by x^3 + 2 x + 5 (the recurrence, a remainder of 3), and by x^(n/2)
// (a shift: f / x^k, f % x^k).
void special(bench::Suite& suite) {
    const Poly x = Poly::x(), g = x * x * x + 2 * x + 5;
    for (int lg : {14, 18, 22}) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        const unsigned seed = unsigned(7 * lg);
        suite.add("divmod_short" + size, [n, seed, g] {
            return [f = random_poly(n, seed), g] { bench::keep(fps::divmod(f, g)); };
        }, n);
        suite.add("divide_monomial" + size, [n, seed] {
            return [f = random_poly(n, seed), m = Poly::x(n / 2)] { bench::keep(f / m); };
        }, n);
        suite.add("modulo_monomial" + size, [n, seed] {
            return [f = random_poly(n, seed), m = Poly::x(n / 2)] { bench::keep(f % m); };
        }, n);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    special(suite);
    general(suite);
    return suite.run();
}
