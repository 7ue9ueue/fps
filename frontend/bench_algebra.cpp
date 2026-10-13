// Times fps/algebra.hpp modulo P = 998244353; tools/kernel.py runs it, a binary of its own. Items
// are the degree of the input.
//   "<op>/<lg>"          through the API (gated)
//   "<op>@kernel/<lg>"   gcd, inverse_mod: fps::half_gcd (backend/gcd); roots, roots_split,
//                        roots_repeated: fps::root_finding::roots (backend/roots); on the same
//                        inputs, into kept memory
//   "<op>@api/<lg>"      through the API, outside the gate's verdict: roots_split at 2^20, 1-5 s a
//                        call, timed for reports only (it would add minutes to every gate)
// factor has no kernel yet (backend issue #123 moves lc-opt's): it runs on gcd, f % g and f * g.
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "backend/roots/roots.hpp"
#include "fps/fps.hpp"

namespace {

constexpr std::uint32_t P = 998244353;
using Poly = fps::Polynomial<P>;

// Degree exactly n: n + 1 random coefficients, the last one nonzero.
Poly random_poly(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<std::uint32_t> a(n + 1);
    for (std::uint32_t& x : a) x = rng() % P;
    if (a.back() == 0) a.back() = 1;
    return Poly(std::span<const std::uint32_t>(a));
}

// The product of n factors x - r for random r: n roots to split off.
Poly split_poly(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<Poly> linear;
    for (std::size_t i = 0; i < n; ++i) linear.push_back(Poly{P - 1 - rng() % (P - 1), 1});
    return fps::product(linear);
}

// n linear factors over n / 4 random values, times a random polynomial of degree n / 2.
Poly repeated_poly(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<std::uint32_t> pool(n / 4);
    for (std::uint32_t& r : pool) r = rng() % P;
    std::vector<Poly> linear;
    for (std::size_t i = 0; i < n; ++i) linear.push_back(Poly{(P - pool[rng() % pool.size()]) % P, 1});
    return fps::product(linear) * random_poly(n / 2, seed + 1);
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    // gcd and inverse_mod of two random polynomials of degree n: coprime, n steps of Euclid.
    for (int lg : {6, 8, 10, 12, 14, 16}) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        const unsigned seed = unsigned(3 * lg);
        suite.add("gcd" + size, [n, seed] {
            return [f = random_poly(n, seed), g = random_poly(n, seed + 1)] { bench::keep(fps::gcd(f, g)); };
        }, n);
        suite.add("gcd@kernel" + size, [n, seed] {
            return [f = random_poly(n, seed), g = random_poly(n, seed + 1), out = std::vector<std::uint32_t>(n + 1)]() mutable {
                bench::keep(fps::half_gcd::gcd(f.span(), g.span(), out));
                bench::keep(out);
            };
        }, n);
        suite.add("inverse_mod" + size, [n, seed] {
            return [f = random_poly(n, seed), m = random_poly(n, seed + 1)] { bench::keep(fps::inverse_mod(f, m)); };
        }, n);
        suite.add("inverse_mod@kernel" + size, [n, seed] {
            return [f = random_poly(n, seed), m = random_poly(n, seed + 1), h = std::vector<std::uint32_t>(n)]() mutable {
                bench::keep(fps::half_gcd::inverse_mod(f.span(), m.span(), h));
                bench::keep(h);
            };
        }, n);
    }
    // roots: of a random polynomial (about one root), of n distinct linear factors (every root
    // found), and of n linear factors over n / 4 values times a random polynomial of degree n / 2
    // (repeated roots: the kernel's climbs, up to about d^2 / 300 more: to 2^12 only).
    for (int lg : {6, 8, 10, 12, 20}) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        const auto add = [&](const std::string& name, auto make) {
            const std::string api = name == "roots_split" && lg == 20 ? name + "@api" : name;
            suite.add(api + size, [make] { return [f = make()] { bench::keep(fps::roots(f)); }; }, n);
            suite.add(name + "@kernel" + size, [make] {
                Poly f = make();
                std::vector<std::uint32_t> out(f.size());
                return [f = std::move(f), out = std::move(out)]() mutable {
                    bench::keep(fps::root_finding::roots(f.span(), out));
                    bench::keep(out);
                };
            }, n);
        };
        add("roots", [n, lg] { return random_poly(n, unsigned(5 * lg)); });
        add("roots_split", [n, lg] { return split_poly(n, unsigned(7 * lg)); });
        if (lg <= 12) add("roots_repeated", [n, lg] { return repeated_poly(n, unsigned(13 * lg)); });
    }
    // factor of a random polynomial: distinct degrees up to n / 2.
    for (int lg : {4, 6, 8}) {
        const std::size_t n = std::size_t(1) << lg;
        suite.add("factor/" + std::to_string(lg), [n, lg] {
            return [f = random_poly(n, unsigned(11 * lg))] { bench::keep(fps::factor(f)); };
        }, n);
    }
    return suite.run();
}
