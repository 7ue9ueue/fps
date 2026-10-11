// Times the frontend's API modulo P = 998244353; tools/kernel.py runs it. Items are coefficients
// or table entries.
//   "<op>/<lg>"          through the API, on 2^lg items (gated)
//   "<op>@kernel/<lg>"   the backend call the API wraps, on the same inputs
// Cases run in the order they are added. Each makes its inputs in its setup, so they live only
// while it runs; the products, which map fresh memory per call, run last.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "backend/bench.hpp"
#include "fps/fps.hpp"

namespace {

constexpr std::uint32_t P = 998244353;
using Poly = fps::Polynomial<P>;

template <class S>
using Ops = fps::modarith::Ops<S, P>;

// Builds the tables of 2^lg entries. The @kernel case allocates the same two vectors as
// Factorials (zero-filled, fresh pages) and calls the backend's factorials::tables.
void factorials(bench::Suite& suite) {
    for (int lg : {10, 14, 18, 22}) {
        const std::size_t n = std::size_t(1) << lg;
        suite.add("factorials/" + std::to_string(lg), [n] {
            return [n] { bench::keep(fps::Factorials<P>(n)); };
        }, n);
        suite.add("factorials@kernel/" + std::to_string(lg), [n] {
            return [n] {
                std::vector<std::uint32_t> fact(n), inverse(n);
                fps::factorials::tables<P>(fact, inverse);
                bench::keep(fact);
                bench::keep(inverse);
            };
        }, n);
    }
}

// 2^lg binomials C(m, k), k <= m < 2^lg random, from tables of 2^lg (512 KiB at lg = 16).
void binomial(bench::Suite& suite) {
    const int lg = 16;
    const std::size_t n = std::size_t(1) << lg;
    suite.add("binomial/" + std::to_string(lg), [n] {
        std::mt19937_64 rng(1);
        std::vector<std::pair<std::size_t, std::size_t>> queries(n);
        for (auto& [m, k] : queries) m = rng() % n, k = rng() % (m + 1);
        return [fact = fps::Factorials<P>(n), queries] {
            std::uint32_t sum = 0;
            for (const auto& [m, k] : queries) sum += fact.binomial(m, k);
            bench::keep(sum);
        };
    }, n);
}

Poly random_poly(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<std::uint32_t> c(n);
    for (std::uint32_t& x : c) x = rng() % P;
    return Poly(std::span<const std::uint32_t>(c));
}

// The modarith stream an element-wise operator wraps: out[i] = op(a[i], b[i]) on simd::Native
// vectors, then single words, into new memory, as the operator's result is.
template <class Op>
std::unique_ptr<std::uint32_t[]> stream(const Poly& a, const Poly& b, Op op) {
    using S = fps::simd::Native;
    const std::size_t n = a.size();
    auto out = std::make_unique_for_overwrite<std::uint32_t[]>(n);
    const std::uint32_t *x = a.span().data(), *y = b.span().data();
    std::size_t i = 0;
    for (; i + S::lanes <= n; i += S::lanes)
        S::store(out.get() + i, op.template operator()<S>(S::load(x + i), S::load(y + i)));
    for (; i < n; ++i) out[i] = op.template operator()<fps::simd::Scalar>(x[i], y[i]);
    return out;
}

// f + g, f - g, -f and f * c on 2^lg coefficients, each against its stream.
template <class Api, class Kernel>
void elementwise(bench::Suite& suite, const std::string& op, Api api, Kernel kernel) {
    for (int lg : {10, 14, 18, 22}) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        suite.add(op + size, [n, lg, api] {
            return [f = random_poly(n, 2 * lg), g = random_poly(n, 2 * lg + 1), api] {
                const Poly h = api(f, g);
                bench::keep(h);
            };
        }, n);
        suite.add(op + "@kernel" + size, [n, lg, kernel] {
            return [f = random_poly(n, 2 * lg), g = random_poly(n, 2 * lg + 1), kernel] {
                bench::keep(stream(f, g, kernel)[0]);
            };
        }, n);
    }
}

void elementwise(bench::Suite& suite) {
    constexpr std::uint32_t c = 123456789, q = Ops<fps::simd::Scalar>::shoup_quotient(c);
    elementwise(suite, "add", [](const Poly& f, const Poly& g) { return f + g; },
                []<class S>(typename S::V x, typename S::V y) { return Ops<S>::add(x, y); });
    elementwise(suite, "sub", [](const Poly& f, const Poly& g) { return f - g; },
                []<class S>(typename S::V x, typename S::V y) { return Ops<S>::sub(x, y); });
    elementwise(suite, "neg", [](const Poly& f, const Poly&) { return -f; },
                []<class S>(typename S::V x, typename S::V) { return Ops<S>::sub(S::broadcast(0), x); });
    elementwise(suite, "scale", [](const Poly& f, const Poly&) { return f * c; },
                []<class S>(typename S::V x, typename S::V) {
                    return Ops<S>::template reduce<P>(Ops<S>::shoup(x, S::broadcast(c), S::broadcast(q)));
                });
}

// f * g for f and g of n coefficients against ntt::Convolution on the same factors (construct,
// copy them in, multiply). "<lg>": n = 2^lg / 2, schoolbook below lg = 10. "480x480": a shape the
// old cutoff (n m < 2^18) ran schoolbook; it runs the NTT now.
void multiply(bench::Suite& suite) {
    std::vector<std::pair<std::string, std::size_t>> shapes;
    for (int lg = 6; lg <= 22; ++lg) shapes.emplace_back(std::to_string(lg), std::size_t(1) << (lg - 1));
    shapes.emplace_back("480x480", 480);
    for (const auto& [size, n] : shapes) {
        const unsigned seed = unsigned(n);
        suite.add("multiply/" + size, [n, seed] {
            return [f = random_poly(n, seed), g = random_poly(n, seed + 1)] {
                const Poly h = f * g;
                bench::keep(h);
            };
        }, 2 * n);
        suite.add("multiply@kernel/" + size, [n, seed] {
            return [f = random_poly(n, seed), g = random_poly(n, seed + 1)] {
                fps::ntt::Convolution conv(f.size(), g.size());
                std::ranges::copy(f.span(), conv.a());
                std::ranges::copy(g.span(), conv.b());
                bench::keep(*conv.multiply());
            };
        }, 2 * n);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    factorials(suite);
    binomial(suite);
    elementwise(suite);
    multiply(suite);
    return suite.run();
}
