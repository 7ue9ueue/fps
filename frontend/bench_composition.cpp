// Times fps/composition.hpp modulo P = 998244353; tools/kernel.py runs it, a binary of its own.
// Items are coefficients of the result, n = 2^(lg-1) as mul/<lg> in bench.cpp.
//   "<op>/<lg>"          through the API (gated)
//   "<op>@kernel/<lg>"   the backend call it wraps, on the same inputs:
//     compose_linear@kernel   fps::taylor_shift::shift on the same f and g[0] into kept memory: what
//     taylor_shift@kernel     compose runs before the scaling by g[1]^k (compose_linear, g[1] != 1;
//                             taylor_shift is compose(f, g[0] + x), the shift alone)
//     compose@kernel         fps::composition::compose on the same f and g into kept memory; it
//                             reads g[0] as 0, so it times what compose runs after its Taylor shift
//     compose_g0@kernel       the same with g[0] = 0, where compose is the kernel call alone
//     compositional_inverse@kernel   fps::composition::compositional_inverse into kept memory
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "backend/composition/composition.hpp"
#include "backend/taylor_shift/taylor_shift.hpp"
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

// The kernel's Taylor shift of f by a into kept memory.
auto shift_kernel(std::size_t n, unsigned seed, std::uint32_t a) {
    return [f = random_poly(n, 7, seed), a, out = std::vector<std::uint32_t>(n)]() mutable {
        fps::taylor_shift::shift(f.span(), a, out);
        bench::keep(out);
    };
}

// compose(f, a + b x): the Taylor shift and the scaling by b^k; compose(f, a + x): the shift alone.
// Their kernel: the Taylor shift.
void linear(bench::Suite& suite) {
    constexpr std::uint32_t a = 123456789, b = 987654321;
    for (int lg : {11, 14, 18, 22}) {
        const std::size_t n = std::size_t(1) << (lg - 1);
        const std::string size = "/" + std::to_string(lg);
        const unsigned seed = unsigned(5 * lg);
        suite.add("compose_linear" + size, [n, seed] {
            return [f = random_poly(n, 7, seed), g = Poly{a, b}] { bench::keep(fps::compose(f, g)); };
        }, n);
        suite.add("compose_linear@kernel" + size, [n, seed] { return shift_kernel(n, seed, a); }, n);
        suite.add("taylor_shift" + size, [n, seed] {
            return [f = random_poly(n, 7, seed), g = Poly{a, 1}] { bench::keep(fps::compose(f, g)); };
        }, n);
        suite.add("taylor_shift@kernel" + size, [n, seed] { return shift_kernel(n, seed, a); }, n);
    }
}

// The kernel's compose into kept memory.
auto compose_kernel(std::size_t n, unsigned seed, std::uint32_t g0) {
    return [f = random_poly(n, 3, seed), g = random_poly(n, g0, seed + 1), out = std::vector<std::uint32_t>(n)]() mutable {
        fps::composition::compose(f.span(), g.span(), out);
        bench::keep(out);
    };
}

// compose(f, g) with g[0] != 0 (Taylor shift, then the kernel) and g[0] = 0 (the kernel alone);
// compositional_inverse(f).
void general(bench::Suite& suite) {
    for (int lg : {11, 14, 18}) {
        const std::size_t n = std::size_t(1) << (lg - 1);
        const std::string size = "/" + std::to_string(lg);
        const unsigned seed = unsigned(7 * lg);
        suite.add("compose" + size, [n, seed] {
            return [f = random_poly(n, 3, seed), g = random_poly(n, 5, seed + 1)] { bench::keep(fps::compose(f, g)); };
        }, n);
        suite.add("compose@kernel" + size, [n, seed] { return compose_kernel(n, seed, 5); }, n);
        suite.add("compose_g0" + size, [n, seed] {
            return [f = random_poly(n, 3, seed), g = random_poly(n, 0, seed + 1)] { bench::keep(fps::compose(f, g)); };
        }, n);
        suite.add("compose_g0@kernel" + size, [n, seed] { return compose_kernel(n, seed, 0); }, n);
        suite.add("compositional_inverse" + size, [n, seed] {
            Poly f = random_poly(n, 0, seed);
            f[1] = 2;
            return [f] { bench::keep(fps::compositional_inverse(f)); };
        }, n);
        suite.add("compositional_inverse@kernel" + size, [n, seed] {
            Poly f = random_poly(n, 0, seed);
            f[1] = 2;
            return [f, out = std::vector<std::uint32_t>(n)]() mutable {
                fps::composition::compositional_inverse(f.span(), out);
                bench::keep(out);
            };
        }, n);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    linear(suite);
    general(suite);
    return suite.run();
}
