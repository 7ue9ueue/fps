// Times fps/division.hpp modulo P = 998244353; tools/kernel.py runs it, a binary of its own. Items
// are coefficients of f.
//   "<op>/<lg>"          through the API (gated)
//   "<op>@kernel/<lg>"   fps::division::divide (backend/division) on the same inputs, into a kept q
//                        and r (none for divide*); for the sparse and monomial cases, what the
//                        kernel would cost there
// ops: divmod (q and r), divide (f / g), modulo (f % g); shapes in the comments below.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "backend/division/division.hpp"
#include "fps/fps.hpp"

namespace {

constexpr std::uint32_t P = 998244353;
using Poly = fps::Polynomial<P>;
using Vec = std::vector<std::uint32_t>;

// n random coefficients, the last one nonzero.
Vec random_vec(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    Vec a(n);
    for (std::uint32_t& x : a) x = rng() % P;
    if (n && a.back() == 0) a.back() = 1;
    return a;
}

Poly poly(const Vec& a) { return Poly(std::span<const std::uint32_t>(a)); }

enum class Op { divmod, divide, modulo };

// "<name>/<lg>" through the API and "<name>@kernel/<lg>" into kept memory, f by g (g[m - 1] != 0).
void add(bench::Suite& suite, const std::string& name, int lg, Op op, std::function<Vec()> f,
         std::function<Vec()> g) {
    const std::string size = "/" + std::to_string(lg);
    const std::size_t n = std::size_t(1) << lg;
    suite.add(name + size, [op, f, g] {
        return [op, f = poly(f()), g = poly(g())] {
            if (op == Op::divmod) bench::keep(fps::divmod(f, g));
            if (op == Op::divide) bench::keep(f / g);
            if (op == Op::modulo) bench::keep(f % g);
        };
    }, n);
    suite.add(name + "@kernel" + size, [op, f, g] {
        Vec a = f(), b = g();
        Vec q(fps::division::quotient_size(a.size(), b.size())), r(op == Op::divide ? 0 : b.size() - 1);
        return [a = std::move(a), b = std::move(b), q = std::move(q), r = std::move(r)]() mutable {
            fps::division::divide(a, b, q, r);
            bench::keep(q), bench::keep(r);
        };
    }, n);
}

// f of n = 2^lg coefficients by g of n / 2 (a quotient of n / 2 + 1 coefficients, a remainder of
// n / 2 - 1), and divmod_long by g of n / 64 (a long quotient), the shapes of backend/division.
void general(bench::Suite& suite) {
    for (int lg : {11, 14, 18, 22}) {
        const std::size_t n = std::size_t(1) << lg;
        const unsigned seed = unsigned(5 * lg);
        const auto f = [n, seed] { return random_vec(n, seed); };
        const auto half = [n, seed] { return random_vec(n / 2, seed + 1); };
        add(suite, "divmod", lg, Op::divmod, f, half);
        add(suite, "divide", lg, Op::divide, f, half);
        add(suite, "modulo", lg, Op::modulo, f, half);
        if (lg >= 14) add(suite, "divmod_long", lg, Op::divmod, f, [n, seed] { return random_vec(n / 64, seed + 1); });
    }
}

// f of n = 2^lg coefficients by divisors whose reversal has few terms (the frontend's recurrence
// before #147): x^3 + 2 x + 5 (2 terms), g of 17 random words (16 terms), g of n - 16 (a quotient
// of 17), all in the kernel; x^(n/2) - x - 1, x^(n/2) plus 16 terms at random below n/2 - 68,
// (the sparse path); x^d plus t = 2 or 16 terms whose highest is gap below d, around the sparse
// path's bound (gap >= 8 (t + 1)); and by x^(n/2) (the monomial path: f / x^k, f % x^k).
void special(bench::Suite& suite) {
    for (int lg : {14, 18, 22}) {
        const std::size_t n = std::size_t(1) << lg;
        const unsigned seed = unsigned(7 * lg);
        const auto f = [n, seed] { return random_vec(n, seed); };
        add(suite, "divmod_short", lg, Op::divmod, f, [] { return Vec{5, 2, 0, 1}; });
        add(suite, "divmod_taps16", lg, Op::divmod, f, [seed] { return random_vec(17, seed + 1); });
        add(suite, "divmod_sparse", lg, Op::divmod, f, [n] {
            Vec g(n / 2 + 1);
            g[0] = g[1] = P - 1, g[n / 2] = 1;
            return g;
        });
        add(suite, "divmod_sparse16", lg, Op::divmod, f, [n, seed] {
            std::mt19937 rng(seed + 2);
            Vec g(n / 2 + 1);
            g[n / 2] = 1;
            for (int t = 0; t < 16; ++t) g[rng() % (n / 2 - 68)] = 1 + rng() % (P - 1);
            return g;
        });
        // x^d plus t terms x^(d-gap) and below, d = gap + t - 1: around the sparse path's bound.
        for (const auto& [t, gap] : {std::pair<std::size_t, std::size_t>{2, 12}, {2, 24}, {2, 48}, {16, 68}, {16, 136}, {16, 272}})
            add(suite, "divmod_t" + std::to_string(t) + "_gap" + std::to_string(gap), lg, Op::divmod, f, [t, gap, seed] {
                const Vec a = random_vec(t, seed + 3);
                Vec g(gap + t);
                for (std::size_t j = 0; j < t; ++j) g[j] = a[j] ? a[j] : 1;
                g.back() = 1;
                return g;
            });
        add(suite, "divmod_top", lg, Op::divmod, f, [n, seed] { return random_vec(n - 16, seed + 1); });
        const auto monomial = [n] {
            Vec g(n / 2 + 1);
            g[n / 2] = 1;
            return g;
        };
        add(suite, "divide_monomial", lg, Op::divide, f, monomial);
        add(suite, "modulo_monomial", lg, Op::modulo, f, monomial);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    special(suite);
    general(suite);
    return suite.run();
}
