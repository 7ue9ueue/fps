// Times fps/series.hpp modulo P = 998244353; tools/kernel.py runs it, a binary of its own. Items
// are coefficients of the result.
//   "<op>/<lg>"   through the API (gated)
// The backend has no series kernel yet, so there are no "<op>@kernel" pairs: the Newton paths
// run on ntt::multiply (raw::mul), whose own pair is mul / mul@kernel in bench.cpp, and the
// recurrences and calculus are frontend loops. frontend/notes.md gives each case in units of
// mul@kernel/<lg> (ntt::multiply of two factors of n into n words) from the same run.
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

// n random coefficients with f[0] = c.
Poly random_poly(std::size_t n, std::uint32_t c, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<std::uint32_t> a(n);
    for (std::uint32_t& x : a) x = rng() % P;
    if (n) a[0] = c;
    return Poly(std::span<const std::uint32_t>(a));
}

// derivative and integral of 2^lg coefficients.
void calculus(bench::Suite& suite) {
    for (int lg : {10, 14, 18, 22}) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        suite.add("derivative" + size, [n, lg] {
            return [f = random_poly(n, 1, unsigned(lg))] { bench::keep(fps::derivative(f)); };
        }, n);
        suite.add("integral" + size, [n, lg] {
            return [f = random_poly(n, 1, unsigned(lg))] { bench::keep(fps::integral(f)); };
        }, n);
    }
}

// inverse, log and exp of n = 2^(lg-1) random coefficients, as mul/<lg> in bench.cpp; raw_exp
// into a kept out.
void newton(bench::Suite& suite) {
    for (int lg : {11, 14, 18, 22}) {
        const std::size_t n = std::size_t(1) << (lg - 1);
        const std::string size = "/" + std::to_string(lg);
        const unsigned seed = unsigned(3 * lg);
        suite.add("inverse" + size, [n, seed] {
            return [f = random_poly(n, 5, seed)] { bench::keep(fps::inverse(f)); };
        }, n);
        suite.add("log" + size, [n, seed] {
            return [f = random_poly(n, 1, seed)] { bench::keep(fps::log(f)); };
        }, n);
        suite.add("exp" + size, [n, seed] {
            return [f = random_poly(n, 0, seed)] { bench::keep(fps::exp(f)); };
        }, n);
        if (lg == 14 || lg == 18)
            suite.add("raw_exp" + size, [n, seed] {
                return [f = random_poly(n, 0, seed), out = Poly(n)]() mutable {
                    fps::raw::exp<P>(f.span(), out.span());
                    bench::keep(out);
                };
            }, n);
    }
}

// The recurrences, on 2^lg coefficients: inverse(1 - x - x^2), log(1 - x - x^2), exp(x + x^2 / 2
// + 3 x^5).
void sparse(bench::Suite& suite) {
    const Poly x = Poly::x();
    const Poly q = 1 - x - x * x, e = x + fps::mod_inverse<P>(2) * x * x + 3 * Poly::x(5);
    for (int lg : {14, 18, 22}) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        suite.add("inverse_sparse" + size, [n, q] { return [n, q] { bench::keep(fps::inverse(q, n)); }; }, n);
        suite.add("log_sparse" + size, [n, q] { return [n, q] { bench::keep(fps::log(q, n)); }; }, n);
        suite.add("exp_sparse" + size, [n, e] { return [n, e] { bench::keep(fps::exp(e, n)); }; }, n);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    calculus(suite);
    sparse(suite);
    newton(suite);
    return suite.run();
}
