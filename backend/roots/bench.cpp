// Times backend/roots; tools/kernel.py runs it. Items are the degree n = 2^lg of f.
//   "roots/<lg>"              random f of degree n (few roots: about 1 on average)
//   "roots_split/<lg>"        n distinct random linear factors
//   "roots_repeated/<lg>"     n linear factors over n / 4 values times n / 2 junk words (lc-opt's
//                             max_random with repeated roots: many multiple zeros climb)
//   "roots@frontend/<lg>",    fps::roots (include/fps/algebra.hpp, Cantor and Zassenhaus) on the
//   "roots_split@frontend/<lg>"  same, to 2^10 (not gated)
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/roots/roots.hpp"
#include "fps/algebra.hpp"

namespace {

constexpr std::uint32_t P = 998244353;
using Vec = std::vector<std::uint32_t>;

Vec random_vec(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    Vec a(n);
    for (auto& x : a) x = rng() % P;
    if (n && a.back() == 0) a.back() = 1;
    return a;
}

Vec multiply(const Vec& a, const Vec& b) {
    Vec c(a.size() + b.size() - 1);
    fps::ntt::multiply(a, b, c);
    return c;
}

// The product of (x - a) over roots, by halves.
Vec from_roots(std::span<const std::uint32_t> roots) {
    if (roots.size() == 1) return {roots[0] ? P - roots[0] : 0, 1};
    const std::size_t h = roots.size() / 2;
    return multiply(from_roots(roots.first(h)), from_roots(roots.subspan(h)));
}

// n distinct nonzero random roots' product.
Vec split(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    Vec r(n);
    for (std::size_t i = 0; i < n; ++i) r[i] = 1 + rng() % (P - 1);  // repeats have probability ~n^2 / P
    return from_roots(r);
}

// n linear factors over n / 4 values, times a random polynomial of degree n / 2.
Vec repeated(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    const Vec pool = random_vec(n / 4, seed + 1);
    Vec r(n);
    for (auto& x : r) x = pool[rng() % pool.size()];
    return multiply(from_roots(r), random_vec(n / 2 + 1, seed + 2));
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    using Poly = fps::Polynomial<P>;
    for (int lg = 6; lg <= 14; lg += 2) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        const auto add = [&](const std::string& name, Vec f) {
            suite.add(name + size, [f] {
                return [f, out = Vec(f.size())]() mutable {
                    bench::keep(fps::root_finding::roots(f, out));
                    bench::keep(out);
                };
            }, n);
            if (lg > 10 || name == "roots_repeated") return;
            suite.add(name + "@frontend" + size, [f] {
                return [f = Poly(std::span<const std::uint32_t>(f))] { bench::keep(fps::roots(f)); };
            }, n);
        };
        add("roots", random_vec(n + 1, unsigned(n)));
        add("roots_split", split(n, unsigned(n)));
        add("roots_repeated", repeated(n, unsigned(n)));
    }
    return suite.run();
}
