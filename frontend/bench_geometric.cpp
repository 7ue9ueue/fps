// Times fps/geometric.hpp modulo P = 998244353; tools/kernel.py runs it, a binary of its own.
// Items are points (values). n = m = L = 2^lg, random inputs.
//   "<op>/<lg>"          through the API (gated)
//   "<op>@kernel/<lg>"   the backend call it wraps (backend/chirp; dft, idft: backend/ntt), on the
//                        same inputs, into new unwritten memory; the API returns a std::vector
//                        (evaluate, extrapolate, dft), whose zero fill is the difference
//   "raw_<op>/<lg>", "raw_<op>@kernel/<lg>"   raw::dft, raw::idft and the kernel, into a buffer
//                        kept across calls
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "fps/fps.hpp"

namespace {

constexpr std::uint32_t P = 998244353;
using Poly = fps::Polynomial<P>;
using Vec = std::vector<std::uint32_t>;
using Words = std::unique_ptr<std::uint32_t[]>;

Vec random_vec(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    Vec a(n);
    for (std::uint32_t& x : a) x = rng() % P;
    return a;
}

std::uint32_t random_word(unsigned seed) { return std::uint32_t(std::mt19937(seed)() % (P - 1)) + 1; }

Words unwritten(std::size_t n) { return std::make_unique_for_overwrite<std::uint32_t[]>(n); }

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    for (int lg : {10, 14, 18}) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        const unsigned seed = unsigned(27 * lg);
        const std::uint32_t a = random_word(seed), r = random_word(seed + 1);
        const std::uint32_t c = std::uint32_t(n) + random_word(seed + 2) % (P - 3 * std::uint32_t(n));

        suite.add("evaluate_geometric" + size, [n, seed, a, r] {
            return [f = Poly(std::span<const std::uint32_t>(random_vec(n, seed))), a, r, n] {
                bench::keep(fps::evaluate(f, fps::geometric(a, r, n)));
            };
        }, n);
        suite.add("evaluate_geometric@kernel" + size, [n, seed, a, r] {
            return [f = random_vec(n, seed), a, r, n] {
                Words out = unwritten(n);
                fps::chirp::evaluate_geometric(f, a, r, {out.get(), n});
                bench::keep(out);
            };
        }, n);

        suite.add("interpolate_geometric" + size, [n, seed, a, r] {
            return [y = random_vec(n, seed + 3), a, r, n] {
                bench::keep(fps::interpolate<P>(fps::geometric(a, r, n), y));
            };
        }, n);
        suite.add("interpolate_geometric@kernel" + size, [n, seed, a, r] {
            return [y = random_vec(n, seed + 3), a, r, n] {
                Words out = unwritten(n);
                fps::chirp::interpolate_geometric(a, r, y, {out.get(), n});
                bench::keep(out);
            };
        }, n);

        suite.add("extrapolate" + size, [n, seed, c] {
            return [v = random_vec(n, seed + 4), c, n] { bench::keep(fps::extrapolate<P>(v, c, n)); };
        }, n);
        suite.add("extrapolate@kernel" + size, [n, seed, c] {
            return [v = random_vec(n, seed + 4), c, n] {
                Words out = unwritten(n);
                fps::chirp::extrapolate(v, c, {out.get(), n});
                bench::keep(out);
            };
        }, n);
    }
    // dft and idft of L = 2^lg: one call to ntt::dft or ntt::idft (backend/ntt/dft.hpp). raw_dft and
    // raw_idft write into a buffer kept across calls; so do their @kernel cases.
    for (int lg : {10, 14, 18, 20}) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        const unsigned seed = unsigned(27 * lg);
        suite.add("dft" + size, [n, seed] {
            return [f = Poly(std::span<const std::uint32_t>(random_vec(n, seed + 5))), n] { bench::keep(fps::dft(f, n)); };
        }, n);
        suite.add("dft@kernel" + size, [n, seed] {
            return [f = random_vec(n, seed + 5), n] {
                Words out = unwritten(n);
                fps::ntt::dft(f, {out.get(), n});
                bench::keep(out);
            };
        }, n);
        suite.add("raw_dft" + size, [n, seed] {
            return [f = random_vec(n, seed + 5), out = Vec(n)]() mutable {
                fps::raw::dft<P>(f, out);
                bench::keep(out);
            };
        }, n);
        suite.add("raw_dft@kernel" + size, [n, seed] {
            return [f = random_vec(n, seed + 5), out = Vec(n)]() mutable {
                fps::ntt::dft(f, out);
                bench::keep(out);
            };
        }, n);

        suite.add("idft" + size, [n, seed] {
            return [v = random_vec(n, seed + 6)] { bench::keep(fps::idft<P>(v)); };
        }, n);
        suite.add("idft@kernel" + size, [n, seed] {
            return [v = random_vec(n, seed + 6), n] {
                Words out = unwritten(n);
                fps::ntt::idft(v, {out.get(), n});
                bench::keep(out);
            };
        }, n);
        suite.add("raw_idft" + size, [n, seed] {
            return [v = random_vec(n, seed + 6), out = Vec(n)]() mutable {
                fps::raw::idft<P>(v, out);
                bench::keep(out);
            };
        }, n);
        suite.add("raw_idft@kernel" + size, [n, seed] {
            return [v = random_vec(n, seed + 6), out = Vec(n)]() mutable {
                fps::ntt::idft(v, out);
                bench::keep(out);
            };
        }, n);
    }
    return suite.run();
}
