// Times fps/fraction.hpp modulo P = 998244353; tools/kernel.py runs it, a binary of its own.
//   "<op>/<size>"          through the API (gated)
//   "<op>@kernel/<size>"   the backend calls the API makes, by hand, on the same inputs
//   "<op>@api/<size>"      through the API, outside the gate's verdict: [k] at d = 2^20, 0.4-1.1 s
//                          a call, timed for reports only (it would add minutes to every gate)
// a * b and a + b are products of polynomials, so they have kernel pairs: ntt::multiply into new
// memory, as the results hold them. [k] is one bostan_mori::coefficient call: its pair calls the
// kernel on the same spans. take(n) runs inverse and mul, with no kernel of its own;
// frontend/notes/23.md gives it in units of mul@kernel (bench.cpp) from the same run. Sizes stop
// at 2^18 to keep a gate's run short, except [k]: d = 10^5 (DESIGN.md's cost example) and 2^20.
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "backend/bench.hpp"
#include "backend/bostan_mori/bostan_mori.hpp"
#include "fps/fps.hpp"

namespace {

constexpr std::uint32_t P = 998244353;
using Poly = fps::Polynomial<P>;
using Frac = fps::Fraction<P>;
using Words = fps::detail::Words;

template <class S>
using Ops = fps::modarith::Ops<S, P>;

// n random coefficients with f[0] = c (if n > 0).
Poly random_poly(std::size_t n, std::uint32_t c, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<std::uint32_t> a(n);
    for (std::uint32_t& x : a) x = rng() % P;
    if (n) a[0] = c;
    return Poly(std::span<const std::uint32_t>(a));
}

Words ntt_product(const Poly& f, const Poly& g) {
    const std::size_t size = f.size() + g.size() - 1;
    Words out(size, Words::Unwritten{});
    fps::ntt::multiply(f.span(), g.span(), {out.data(), size});
    return out;
}

// a[k] at k = 10^18 (60 bits: 60 steps), p of d and q of d + 1 coefficients; d = 2 is
// Fibonacci, d = 16 and 64 run the kernel's schoolbook steps, d >= 1024 its values form (2^22
// words at d = 2^20).
void coefficient(bench::Suite& suite) {
    constexpr std::uint64_t k = 1'000'000'000'000'000'000ULL;
    for (const std::size_t d : {2, 16, 64, 1024, 16384, 100000, 1 << 20}) {
        const Frac a = d == 2 ? Frac(Poly{0, 1}, Poly{1, P - 1, P - 1})
                              : Frac(random_poly(d, 3, unsigned(d)), random_poly(d + 1, 1, unsigned(d) + 1));
        const std::string size = "/" + std::to_string(d);
        suite.add(std::string(d == 1 << 20 ? "fraction_at@api" : "fraction_at") + size, [a] { return [a] { bench::keep(a[k]); }; }, d);
        suite.add("fraction_at@kernel" + size, [a] {
            return [a] {
                bench::keep(fps::bostan_mori::coefficient(a.numerator().span(), a.denominator().span(), k));
            };
        }, d);
    }
}

// take(n), n = 2^(lg-1), as mul/<lg>: p and q of n random coefficients (inverse by Newton, then
// mul); and Fibonacci, n = 2^lg (inverse by the recurrence, then a shift).
void take(bench::Suite& suite) {
    for (int lg : {11, 14, 18}) {
        const std::size_t n = std::size_t(1) << (lg - 1);
        suite.add("fraction_take/" + std::to_string(lg), [n, lg] {
            return [a = Frac(random_poly(n, 3, unsigned(lg)), random_poly(n, 1, unsigned(lg) + 1)), n] {
                bench::keep(a.take(n));
            };
        }, n);
    }
    for (int lg : {14, 18}) {
        const std::size_t n = std::size_t(1) << lg;
        suite.add("fraction_take_sparse/" + std::to_string(lg), [n] {
            return [a = Frac(Poly{0, 1}, Poly{1, P - 1, P - 1}), n] { bench::keep(a.take(n)); };
        }, n);
    }
}

// a * b and a + b, every numerator and denominator of n = 2^(lg-1) coefficients, as
// multiply/<lg>: two and three products of n x n, against ntt::multiply into new memory (and the
// modarith add of a + b's numerator, in place, as the API's rvalue sum).
void operations(bench::Suite& suite) {
    for (int lg : {11, 14, 18}) {
        const std::size_t n = std::size_t(1) << (lg - 1);
        const std::string size = "/" + std::to_string(lg);
        const unsigned seed = unsigned(7 * lg);
        const auto fractions = [n, seed] {
            return std::pair{Frac(random_poly(n, 3, seed), random_poly(n, 1, seed + 1)),
                             Frac(random_poly(n, 5, seed + 2), random_poly(n, 2, seed + 3))};
        };
        suite.add("fraction_mul" + size, [fractions] {
            return [f = fractions()] { bench::keep(f.first * f.second); };
        }, 2 * n);
        suite.add("fraction_mul@kernel" + size, [fractions] {
            return [f = fractions()] {
                const auto& [a, b] = f;
                Words p = ntt_product(a.numerator(), b.numerator()), q = ntt_product(a.denominator(), b.denominator());
                bench::keep(p);
                bench::keep(q);
            };
        }, 2 * n);
        suite.add("fraction_add" + size, [fractions] {
            return [f = fractions()] { bench::keep(f.first + f.second); };
        }, 2 * n);
        suite.add("fraction_add@kernel" + size, [fractions] {
            return [f = fractions()] {
                const auto& [a, b] = f;
                Words p = ntt_product(a.numerator(), b.denominator());
                {
                    const Words r = ntt_product(b.numerator(), a.denominator());
                    fps::detail::lanewise(p.data(), r.data(), p.data(), p.size(),
                                          []<class S>(typename S::V x, typename S::V y) { return Ops<S>::add(x, y); });
                }
                Words q = ntt_product(a.denominator(), b.denominator());
                bench::keep(p);
                bench::keep(q);
            };
        }, 2 * n);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    take(suite);
    operations(suite);
    coefficient(suite);
    return suite.run();
}
