// Times backend/division; tools/kernel.py runs it. Items are coefficients of f, n = 2^lg.
//   "divmod/<lg>"          q and r for g of n / 2 words (a quotient of n / 2 + 1)
//   "divide/<lg>"          the same, q alone
//   "divmod_long/<lg>"     g of n / 64 words (a long quotient, a short divisor)
//   "divmod_short/<lg>"    g of 9 words
//   "n_max_04"             f of 500000 words by g of 277012 (lc-opt's judge test; windows plans)
//   "divmod@frontend/<lg>" fps::divmod (include/fps/division.hpp) on the divmod shape (not gated)
//   "multiply@ntt/<lg>"    the unit: ntt::multiply of two factors of n into n words (not gated)
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "backend/division/division.hpp"
#include "backend/ntt/ntt.hpp"
#include "fps/division.hpp"

namespace {

constexpr std::uint32_t P = 998244353;
using Vec = std::vector<std::uint32_t>;

// n random words, the last one nonzero.
Vec random_vec(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    Vec a(n);
    for (auto& x : a) x = rng() % P;
    if (n && a.back() == 0) a.back() = 1;
    return a;
}

void add(bench::Suite& suite, const std::string& name, std::size_t n, std::size_t m, bool remainder) {
    suite.add(name, [n, m, remainder] {
        return [f = random_vec(n, unsigned(n + m)), g = random_vec(m, unsigned(m)),
                q = Vec(fps::division::quotient_size(n, m)), r = Vec(remainder ? m - 1 : 0)]() mutable {
            fps::division::divide(f, g, q, r);
            bench::keep(q), bench::keep(r);
        };
    }, n);
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    for (int lg : {7, 11, 14, 18, 22}) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        add(suite, "divmod" + size, n, n / 2, true);
        add(suite, "divide" + size, n, n / 2, false);
        if (lg >= 11) add(suite, "divmod_long" + size, n, n / 64, true);
        if (lg >= 11) add(suite, "divmod_short" + size, n, 9, true);
        suite.add("divmod@frontend" + size, [n] {
            using Poly = fps::Polynomial<P>;
            const Vec f = random_vec(n, unsigned(n + n / 2)), g = random_vec(n / 2, unsigned(n / 2));
            return [f = Poly(std::span<const std::uint32_t>(f)), g = Poly(std::span<const std::uint32_t>(g))] {
                bench::keep(fps::divmod(f, g));
            };
        }, n);
        suite.add("multiply@ntt" + size, [n, lg] {
            return [f = random_vec(n / 2, unsigned(lg)), g = random_vec(n / 2, unsigned(lg + 1)), out = Vec(n / 2)]() mutable {
                fps::ntt::multiply(f, g, out);
                bench::keep(out);
            };
        }, n);
    }
    // lc-opt's slowest judge shape (division_of_polynomials n_max_04): windows of up to 8 terms.
    add(suite, "n_max_04", 500000, 277012, true);
    return suite.run();
}
