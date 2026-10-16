// Times the frontend's API modulo P = 998244353; tools/kernel.py runs it. Items are coefficients
// or table entries.
//   "<op>/<lg>"          through the API, on 2^lg items (gated)
//   "<op>@kernel/<lg>"   the backend call the API wraps, on the same inputs
// Cases run in the order they are added. Each makes its inputs in its setup, so they live only
// while it runs. The products run last: ntt::multiply keeps its memory per thread.
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

// Builds the tables of 2^lg entries. The @kernel case allocates two unwritten tables, as
// Factorials does, and calls the backend's factorials::tables.
void factorials(bench::Suite& suite) {
    for (int lg : {10, 14, 18, 22}) {
        const std::size_t n = std::size_t(1) << lg;
        suite.add("factorials/" + std::to_string(lg), [n] {
            return [n] { bench::keep(fps::Factorials<P>(n)); };
        }, n);
        suite.add("factorials@kernel/" + std::to_string(lg), [n] {
            return [n] {
                auto fact = std::make_unique_for_overwrite<std::uint32_t[]>(n);
                auto inverse = std::make_unique_for_overwrite<std::uint32_t[]>(n);
                fps::factorials::tables<P>({fact.get(), n}, {inverse.get(), n});
                bench::keep(fact[n - 1]);
                bench::keep(inverse[n - 1]);
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

// The kernel case of a product: ntt::multiply(f, g, out) into new memory of size words, as the
// API's result holds it: f g mod x^size.
fps::detail::Words ntt_product(const Poly& f, const Poly& g, std::size_t size) {
    fps::detail::Words out(size, fps::detail::Words::Unwritten{});
    fps::ntt::multiply(f.span(), g.span(), {out.data(), size});
    return out;
}

// The kernel cases of square and middle_product: ntt::square and ntt::middle_product into new
// memory, as the API's result holds it.
fps::detail::Words ntt_square(const Poly& f) {
    fps::detail::Words out(2 * f.size() - 1, fps::detail::Words::Unwritten{});
    fps::ntt::square(f.span(), {out.data(), 2 * f.size() - 1});
    return out;
}

fps::detail::Words ntt_middle(const Poly& f, const Poly& g) {
    const std::size_t count = f.size() - g.size() + 1;
    fps::detail::Words out(count, fps::detail::Words::Unwritten{});
    fps::ntt::middle_product(f.span(), g.span(), {out.data(), count});
    return out;
}

// f * g for f and g of n coefficients against ntt::multiply on the same factors into new memory
// (ntt_product). "<lg>": n = 2^lg / 2. "480x480": a shape an old cutoff (n m < 2^18) ran
// schoolbook. "6e5": n = 600000, a product of ragged length (1.2e6 - 1 words, backend issue #155). Products small::beats_ntt gives to small::multiply (e.g. lg = 6 on most targets)
// are still timed against ntt::multiply here; short_products() times them against
// small::multiply.
void multiply(bench::Suite& suite) {
    std::vector<std::pair<std::string, std::size_t>> shapes;
    for (int lg = 6; lg <= 22; ++lg) shapes.emplace_back(std::to_string(lg), std::size_t(1) << (lg - 1));
    shapes.emplace_back("480x480", 480);
    shapes.emplace_back("6e5", 600000);
    for (const auto& [size, n] : shapes) {
        const unsigned seed = unsigned(n);
        suite.add("multiply/" + size, [n, seed] {
            return [f = random_poly(n, seed), g = random_poly(n, seed + 1)] {
                const Poly h = f * g;
                bench::keep(h);
            };
        }, 2 * n);
        suite.add("multiply@kernel/" + size, [n, seed] {
            return [f = random_poly(n, seed), g = random_poly(n, seed + 1), n] {
                bench::keep(ntt_product(f, g, 2 * n - 1));
            };
        }, 2 * n);
    }
}

// fps/products.hpp on transform length L = 2^lg, against the backend call on the same factors:
// - mul(f, g, n), f and g of n = L / 2: ntt_product of n words;
// - square(f), f of L / 2: ntt::square (ntt_square);
// - middle_product(f, g), f of L / 2, g of L / 4: ntt::middle_product, a transform of L / 2
//   (ntt_middle);
// - raw::mul into a kept out of L / 2 words: ntt::multiply into the same out.
void products(bench::Suite& suite) {
    for (int lg : {11, 14, 18, 22}) {
        const std::size_t n = std::size_t(1) << (lg - 1), L = 2 * n;
        const std::string size = "/" + std::to_string(lg);
        const unsigned seed = unsigned(5 * lg);
        suite.add("mul" + size, [n, seed] {
            return [f = random_poly(n, seed), g = random_poly(n, seed + 1), n] { bench::keep(fps::mul(f, g, n)); };
        }, L);
        suite.add("mul@kernel" + size, [n, seed] {
            return [f = random_poly(n, seed), g = random_poly(n, seed + 1), n] { bench::keep(ntt_product(f, g, n)); };
        }, L);
        suite.add("square" + size, [n, seed] {
            return [f = random_poly(n, seed)] { bench::keep(fps::square(f)); };
        }, L);
        suite.add("square@kernel" + size, [n, seed] {
            return [f = random_poly(n, seed)] { bench::keep(ntt_square(f)); };
        }, L);
        suite.add("middle_product" + size, [n, seed] {
            return [f = random_poly(n, seed), g = random_poly(n / 2, seed + 1)] {
                bench::keep(fps::middle_product(f, g));
            };
        }, L);
        suite.add("middle_product@kernel" + size, [n, seed] {
            return [f = random_poly(n, seed), g = random_poly(n / 2, seed + 1)] { bench::keep(ntt_middle(f, g)); };
        }, L);
        suite.add("raw_mul" + size, [n, seed] {
            return [f = random_poly(n, seed), g = random_poly(n, seed + 1), out = Poly(n)]() mutable {
                fps::raw::mul<P>(f.span(), g.span(), out.span());
                bench::keep(out);
            };
        }, L);
        suite.add("raw_mul@kernel" + size, [n, seed] {
            return [f = random_poly(n, seed), g = random_poly(n, seed + 1), out = Poly(n)]() mutable {
                fps::ntt::multiply(f.span(), g.span(), out.span());
                bench::keep(out);
            };
        }, L);
    }
}

// raw::mul of two short factors of n into n words: small::multiply writes all 2 n - 1, so the API
// runs its own truncated loop (backend issue #37). The kernel case: small::multiply into a kept
// array of 2 n - 1 words. "<n>x<n>".
void short_raw_products(bench::Suite& suite) {
    for (const std::size_t n : {std::size_t(16), std::size_t(24)}) {
        const std::string size = "/" + std::to_string(n) + "x" + std::to_string(n);
        const unsigned seed = unsigned(7 * n);
        suite.add("raw_mul" + size, [n, seed] {
            return [f = random_poly(n, seed), g = random_poly(n, seed + 1), out = Poly(n)]() mutable {
                fps::raw::mul<P>(f.span(), g.span(), out.span());
                bench::keep(out);
            };
        }, n * n);
        suite.add("raw_mul@kernel" + size, [n, seed] {
            return [f = random_poly(n, seed), g = random_poly(n, seed + 1), out = Poly(2 * n - 1)]() mutable {
                fps::small::multiply<P>(f.span(), g.span(), out.span());
                bench::keep(out);
            };
        }, n * n);
    }
}

// product(fs) of 2^lg factors x - r_i, against product_tree::product on spans of the same factors
// (built in the setup) into new memory, as the API's result holds it.
void product(bench::Suite& suite) {
    for (int lg : {10, 14, 18}) {
        const std::size_t count = std::size_t(1) << lg;
        const auto factors = [count, lg] {
            std::mt19937 rng{unsigned(lg)};
            std::vector<Poly> fs;
            for (std::size_t i = 0; i < count; ++i) fs.push_back(Poly{std::uint32_t(rng() % P), 1});
            return fs;
        };
        suite.add("product/" + std::to_string(lg), [factors] {
            return [fs = factors()] { bench::keep(fps::product(fs)); };
        }, count);
        suite.add("product@kernel/" + std::to_string(lg), [factors] {
            const std::vector<Poly> fs = factors();
            std::vector<std::span<const std::uint32_t>> spans(fs.size());
            for (std::size_t i = 0; i < fs.size(); ++i) spans[i] = fs[i].span();
            return [fs, spans] {
                const std::size_t size = fps::product_tree::size(spans);
                fps::detail::Words out(size, fps::detail::Words::Unwritten{});
                fps::product_tree::product(spans, {out.data(), size});
                bench::keep(out);
            };
        }, count);
    }
}

// f * g on shapes small::beats_ntt gives to small::multiply on every timed target, against
// small::multiply into new memory, as f * g writes its result: "<n>x<m>".
void short_products(bench::Suite& suite) {
    for (const auto& [n, m] : std::vector<std::pair<std::size_t, std::size_t>>{{16, 16}, {4096, 4}, {4096, 16}}) {
        const std::string size = "/" + std::to_string(n) + "x" + std::to_string(m);
        const unsigned seed = unsigned(n + m);
        suite.add("multiply" + size, [n, m, seed] {
            return [f = random_poly(n, seed), g = random_poly(m, seed + 1)] {
                const Poly h = f * g;
                bench::keep(h);
            };
        }, n * m);
        suite.add("multiply@kernel" + size, [n, m, seed] {
            return [f = random_poly(n, seed), g = random_poly(m, seed + 1)] {
                auto h = std::make_unique_for_overwrite<std::uint32_t[]>(f.size() + g.size() - 1);
                fps::small::multiply<P>(f.span(), g.span(), std::span(h.get(), f.size() + g.size() - 1));
                bench::keep(h[0]);
            };
        }, n * m);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    factorials(suite);
    binomial(suite);
    elementwise(suite);
    short_products(suite);
    multiply(suite);
    products(suite);
    short_raw_products(suite);
    product(suite);
    return suite.run();
}
