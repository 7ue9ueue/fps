// Times fps/series.hpp modulo P = 998244353; tools/kernel.py runs it, a binary of its own. Items
// are coefficients of the result.
//   "<op>/<lg>"          through the API (gated)
//   "<op>@kernel/<lg>"   inverse, log, exp, pow, sqrt: backend/series on the same input into a
//                        kept out (pow: e mod P and f[0]^e, as the API passes them)
// The recurrences and calculus are frontend loops; "<op>_sparse@kernel/<lg>" times the backend on
// the recurrences' inputs (pow, sqrt), the other side of the frontend's choice of path. frontend/notes.md gives the Newton cases in
// units of mul@kernel/<lg> (ntt::multiply of two factors of n into n words) from the same run.
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
constexpr std::uint64_t kExponent = 0x9e3779b97f4a7c15;

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

// inverse, log, exp, pow (e a random 64-bit exponent) and sqrt of n = 2^(lg-1) random
// coefficients, as mul/<lg> in bench.cpp; raw_exp into a kept out.
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
        suite.add("pow" + size, [n, seed] {
            return [f = random_poly(n, 5, seed)] { bench::keep(fps::pow(f, kExponent)); };
        }, n);
        suite.add("sqrt" + size, [n, seed] {
            return [f = random_poly(n, 4, seed)] { bench::keep(fps::sqrt(f)); };
        }, n);
        const auto kernel = [&suite, n, seed, size](const std::string& op, std::uint32_t c, auto call) {
            suite.add(op + "@kernel" + size, [n, seed, c, call] {
                return [f = random_poly(n, c, seed), out = Poly(n), call]() mutable {
                    call(f.span(), out.span());
                    bench::keep(out);
                };
            }, n);
        };
        kernel("inverse", 5, [](std::span<const std::uint32_t> f, std::span<std::uint32_t> o) { fps::series::inverse(f, o); });
        kernel("log", 1, [](std::span<const std::uint32_t> f, std::span<std::uint32_t> o) { fps::series::log(f, o); });
        kernel("exp", 0, [](std::span<const std::uint32_t> f, std::span<std::uint32_t> o) { fps::series::exp(f, o); });
        kernel("pow", 5, [](std::span<const std::uint32_t> f, std::span<std::uint32_t> o) {
            fps::series::pow(f, std::uint32_t(kExponent % P), fps::mod_pow<P>(5, kExponent), o);
        });
        kernel("sqrt", 4, [](std::span<const std::uint32_t> f, std::span<std::uint32_t> o) { fps::series::sqrt(f, 2, o); });
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
// + 3 x^5), pow(1 - x - x^2, e), sqrt(1 - x - x^2).
void sparse(bench::Suite& suite) {
    const Poly x = Poly::x();
    const Poly q = 1 - x - x * x, e = x + fps::mod_inverse<P>(2) * x * x + 3 * Poly::x(5);
    for (int lg : {14, 18, 22}) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        suite.add("inverse_sparse" + size, [n, q] { return [n, q] { bench::keep(fps::inverse(q, n)); }; }, n);
        suite.add("log_sparse" + size, [n, q] { return [n, q] { bench::keep(fps::log(q, n)); }; }, n);
        suite.add("exp_sparse" + size, [n, e] { return [n, e] { bench::keep(fps::exp(e, n)); }; }, n);
        suite.add("pow_sparse" + size, [n, q] { return [n, q] { bench::keep(fps::pow(q, kExponent, n)); }; }, n);
        suite.add("sqrt_sparse" + size, [n, q] { return [n, q] { bench::keep(fps::sqrt(q, n)); }; }, n);
        suite.add("pow_sparse@kernel" + size, [n, q] {
            return [q, out = Poly(n)]() mutable {
                fps::series::pow(q.span(), std::uint32_t(kExponent % P), 1, out.span());
                bench::keep(out);
            };
        }, n);
        suite.add("sqrt_sparse@kernel" + size, [n, q] {
            return [q, out = Poly(n)]() mutable {
                fps::series::sqrt(q.span(), 1, out.span());
                bench::keep(out);
            };
        }, n);
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
