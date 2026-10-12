// Benchmarks of backend/chirp. Items: words.
//   chirp/<lg>                  2^lg terms into an array
//   multiply_chirp/<lg>         2^lg words times the terms, in place
//   chirp@k<K>/<lg>             chirp with Terms<Wide, K>, K in {1, 2, 4, 8}, lg in {10, 16, 20}
//   products@scalar/<lg>        prefix products of 2^lg geometric factors and their inverses (scan.hpp),
//   products@lanes<K>/<lg>      in scalar chains or in lanes, K in {1, 2, 4}, lg in {10, 16, 20}
//   evaluate_geometric/<lg>     f of 2^lg coefficients at 2^lg points a r^i
//   interpolate_geometric/<lg>  2^lg values at a r^i -> coefficients
//   interpolate_geometric@<products>/<lg>   the same with ntt::multiply or ntt::Transform products
//   extrapolate/<lg>            2^lg samples f(0 .. n - 1) -> f(c + k), k < 2^lg, c past the samples
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "backend/chirp/chirp.hpp"
#include "backend/chirp/geometric.hpp"
#include "backend/chirp/shift.hpp"

namespace {

using u32 = std::uint32_t;
using fps::chirp::kP;

std::vector<u32> random_words(std::size_t n, unsigned seed) {
    std::mt19937 rng{seed};
    std::vector<u32> f(n);
    for (u32& x : f) x = u32(rng() % kP);
    return f;
}

constexpr u32 kC = 123456789, kS = 987654321, kQ = 31415926;

template <int K>
void add_variant(bench::Suite& suite, int lg) {
    const std::size_t n = std::size_t(1) << lg;
    suite.add("chirp@k" + std::to_string(K) + "/" + std::to_string(lg), [n] {
        return [out = std::vector<u32>(n)]() mutable {
            fps::chirp::chirp<fps::chirp::Wide, K>(kC, kS, kQ, out);
            bench::keep(out);
        };
    }, n);
}

template <class Run>
void add_products(bench::Suite& suite, const std::string& name, int lg, Run run) {
    const std::size_t n = std::size_t(1) << lg;
    suite.add(name + "/" + std::to_string(lg), [n, run] {
        return [prefix = std::vector<u32>(n + 1), inverse = std::vector<u32>(n + 1), n, run]() mutable {
            run(fps::chirp::detail::Geometric(kQ, kP - 1, kS), n, prefix.data(), 1, inverse.data(), 1);
            bench::keep(prefix);
            bench::keep(inverse);
        };
    }, n);
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    for (int lg = 6; lg <= 22; ++lg) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        suite.add("chirp" + size, [n] {
            return [out = std::vector<u32>(n)]() mutable {
                fps::chirp::chirp(kC, kS, kQ, out);
                bench::keep(out);
            };
        }, n);
        suite.add("multiply_chirp" + size, [n, lg] {
            return [f = random_words(n, unsigned(lg))]() mutable {
                fps::chirp::multiply_chirp(kC, kS, kQ, f);
                bench::keep(f);
            };
        }, n);
    }
    for (int lg : {10, 12, 14, 16, 20}) {
        add_variant<1>(suite, lg);
        add_variant<2>(suite, lg);
        add_variant<4>(suite, lg);
        add_variant<8>(suite, lg);
        using fps::chirp::Wide;
        namespace d = fps::chirp::detail;
        add_products(suite, "products@scalar", lg, [](auto&&... a) { d::products_scalar(a...); });
        add_products(suite, "products@lanes1", lg, [](auto&&... a) { d::products_lanes<Wide, 1>(a...); });
        add_products(suite, "products@lanes2", lg, [](auto&&... a) { d::products_lanes<Wide, 2>(a...); });
        add_products(suite, "products@lanes4", lg, [](auto&&... a) { d::products_lanes<Wide, 4>(a...); });
    }
    for (int lg = 6; lg <= 22; lg += 2) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        suite.add("evaluate_geometric" + size, [n, lg] {
            return [f = random_words(n, unsigned(lg)), out = std::vector<u32>(n)]() mutable {
                fps::chirp::evaluate_geometric(f, kC, kS, out);
                bench::keep(out);
            };
        }, n);
        suite.add("interpolate_geometric" + size, [n, lg] {
            return [y = random_words(n, unsigned(lg)), out = std::vector<u32>(n)]() mutable {
                fps::chirp::interpolate_geometric(kC, kS, y, out);
                bench::keep(out);
            };
        }, n);
        suite.add("interpolate_geometric@transform" + size, [n, lg] {
            return [y = random_words(n, unsigned(lg)), out = std::vector<u32>(n)]() mutable {
                fps::chirp::interpolate_geometric<fps::chirp::Products::transform>(kC, kS, y, out);
                bench::keep(out);
            };
        }, n);
        suite.add("interpolate_geometric@multiply" + size, [n, lg] {
            return [y = random_words(n, unsigned(lg)), out = std::vector<u32>(n)]() mutable {
                fps::chirp::interpolate_geometric<fps::chirp::Products::multiply>(kC, kS, y, out);
                bench::keep(out);
            };
        }, n);
        suite.add("extrapolate" + size, [n, lg] {
            return [y = random_words(n, unsigned(lg)), out = std::vector<u32>(n)]() mutable {
                fps::chirp::extrapolate(y, kC, out);
                bench::keep(out);
            };
        }, n);
    }
    return suite.run();
}
