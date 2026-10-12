// Times backend/gcd; tools/kernel.py runs it. Items are coefficients of f, n = 2^lg; f and g random
// of n words each (coprime with probability ~1 - n / P).
//   "gcd/<lg>"                   gcd(f, g) (1 here)
//   "inverse_mod/<lg>"           1 / f mod g
//   "gcd@frontend/<lg>"          fps::gcd (include/fps/algebra.hpp, Euclid) on the same (not gated)
//   "inverse_mod@frontend/<lg>"  fps::inverse_mod on the same (not gated)
//   "multiply@ntt/<lg>"          the unit: ntt::multiply of two factors of n into n words (not gated)
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "backend/gcd/gcd.hpp"
#include "backend/ntt/ntt.hpp"
#include "fps/algebra.hpp"

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

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    for (int lg = 6; lg <= 20; lg += 2) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        suite.add("gcd" + size, [n] {
            return [f = random_vec(n, unsigned(n)), g = random_vec(n, unsigned(n + 1)), out = Vec(n)]() mutable {
                bench::keep(fps::gcd::gcd(f, g, out));
                bench::keep(out);
            };
        }, n);
        suite.add("inverse_mod" + size, [n] {
            return [f = random_vec(n, unsigned(n)), g = random_vec(n, unsigned(n + 1)), h = Vec(n - 1)]() mutable {
                bench::keep(fps::gcd::inverse_mod(f, g, h));
                bench::keep(h);
            };
        }, n);
        if (lg <= 12) {
            using Poly = fps::Polynomial<P>;
            suite.add("gcd@frontend" + size, [n] {
                const Vec f = random_vec(n, unsigned(n)), g = random_vec(n, unsigned(n + 1));
                return [f = Poly(std::span<const std::uint32_t>(f)), g = Poly(std::span<const std::uint32_t>(g))] {
                    bench::keep(fps::gcd(f, g));
                };
            }, n);
            suite.add("inverse_mod@frontend" + size, [n] {
                const Vec f = random_vec(n, unsigned(n)), g = random_vec(n, unsigned(n + 1));
                return [f = Poly(std::span<const std::uint32_t>(f)), g = Poly(std::span<const std::uint32_t>(g))] {
                    bench::keep(fps::inverse_mod(f, g));
                };
            }, n);
        }
        suite.add("multiply@ntt" + size, [n, lg] {
            return [f = random_vec(n, unsigned(lg)), g = random_vec(n, unsigned(lg + 1)), out = Vec(n)]() mutable {
                fps::ntt::multiply(f, g, out);
                bench::keep(out);
            };
        }, n);
    }
    return suite.run();
}
