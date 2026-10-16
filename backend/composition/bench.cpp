// Times backend/composition; tools/kernel.py runs it. Items are coefficients of the result, n =
// 2^(lg-1) random coefficients (g[0] = 0, f[1] != 0), lg 6 to 18 and 21.
//   "compose/<lg>", "power_projection/<lg>", "compositional_inverse/<lg>"
//   "mul@ntt/<lg>"   the unit: ntt::multiply of two factors of n into n words (not gated)
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "backend/composition/composition.hpp"
#include "backend/ntt/ntt.hpp"

namespace {

constexpr std::uint32_t P = 998244353;
using Vec = std::vector<std::uint32_t>;

Vec random_vec(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    Vec a(n);
    for (auto& x : a) x = rng() % P;
    if (n) a[0] = 0;
    if (n > 1) a[1] = a[1] ? a[1] : 1;
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    namespace c = fps::composition;
    for (int lg : {6, 8, 10, 12, 14, 16, 18, 21}) {  // 21: n = 2^20 (issue #149 round 3)
        const std::size_t n = std::size_t(1) << (lg - 1);
        const std::string size = "/" + std::to_string(lg);
        suite.add("compose" + size, [n, lg] {
            return [f = random_vec(n, unsigned(lg)), g = random_vec(n, unsigned(lg + 1)), out = Vec(n)]() mutable {
                c::compose(f, g, out);
                bench::keep(out);
            };
        }, n);
        suite.add("power_projection" + size, [n, lg] {
            return [g = random_vec(n, unsigned(lg + 2)), out = Vec(n)]() mutable {
                c::power_projection(g, out);
                bench::keep(out);
            };
        }, n);
        suite.add("compositional_inverse" + size, [n, lg] {
            return [f = random_vec(n, unsigned(lg + 3)), out = Vec(n)]() mutable {
                c::compositional_inverse(f, out);
                bench::keep(out);
            };
        }, n);
        suite.add("mul@ntt" + size, [n, lg] {
            return [f = random_vec(n, unsigned(lg + 4)), g = random_vec(n, unsigned(lg + 5)), out = Vec(n)]() mutable {
                fps::ntt::multiply(f, g, out);
                bench::keep(out);
            };
        }, n);
    }
    return suite.run();
}
