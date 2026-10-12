// Times the frontend's API modulo P = 998244353; tools/kernel.py runs it. Items are coefficients
// or table entries.
//   "<op>/<lg>"          through the API, on 2^lg items (gated)
//   "<op>@kernel/<lg>"   the backend call the API wraps, on the same inputs
// The products of value forms and the pointwise layout wrap no backend kernel yet
// (frontend/notes.md), so their cases have no @kernel pair. Cases run in the order they are added.
// Each makes its inputs in its setup, so they live only while it runs. Value forms run last: the
// library keeps their tables and scratch, and a Convolution keeps its memory per thread, so cases
// after them would run in another state than on a main without them.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
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

// f * g for f and g of n coefficients against ntt::Convolution on the same factors (construct,
// copy them in, multiply). "<lg>": n = 2^lg / 2. "480x480": a shape an old cutoff (n m < 2^18) ran
// schoolbook. Products small::beats_ntt gives to small::multiply (e.g. lg = 6 on most targets)
// are still timed against the Convolution here; short_products() times them against
// small::multiply.
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

// The leaf form by the backend directly: L words and ntt::kPadding more, 64-byte aligned.
struct Leaves {
    struct Delete {
        void operator()(std::uint32_t* p) const { ::operator delete(p, std::align_val_t(64)); }
    };
    std::unique_ptr<std::uint32_t[], Delete> words;

    explicit Leaves(std::size_t L)
        : words(static_cast<std::uint32_t*>(::operator new((L + fps::ntt::kPadding) * 4, std::align_val_t(64)))) {
        std::fill_n(words.get() + L, fps::ntt::kPadding, 0u);
    }
    // f, zeros up to L, then the forward transform.
    Leaves(const Poly& f, std::size_t L, const fps::ntt::Transform& t) : Leaves(L) {
        std::ranges::copy(f.span(), words.get());
        std::fill(words.get() + f.size(), words.get() + L, 0u);
        t.forward(words.get());
    }
};

// fps::Values against ntt::Transform on L = 2^lg words. to_values: f of L / 2 coefficients (an
// exact product's factor) into new memory; the kernel builds its tables in setup, as the API
// keeps them per length. to_coefficients: a value form of L coefficients back into new memory;
// the kernel copies the leaf form and runs the inverse in place. add_coefficients: the upper half
// added into the same kept array, ABC222 H's step; the kernel copies into a kept array, runs the
// inverse and adds with modarith. The products and the pointwise layout's conversions have no
// backend kernel (frontend/notes.md): no @kernel pair.
void values(bench::Suite& suite) {
    for (int lg : {10, 14, 18, 22}) {
        const std::size_t L = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        const unsigned seed = unsigned(3 * lg);
        suite.add("to_values" + size, [L, seed] {
            return [f = random_poly(L / 2, seed), L] { bench::keep(fps::to_values(f, L)); };
        }, L);
        suite.add("to_values@kernel" + size, [L, lg, seed] {
            return [f = random_poly(L / 2, seed), L, t = std::make_shared<fps::ntt::Transform>(lg)] {
                bench::keep(Leaves(f, L, *t));
            };
        }, L);
        suite.add("to_coefficients" + size, [L, seed] {
            return [A = fps::to_values(random_poly(L, seed), L)] { bench::keep(fps::to_coefficients(A)); };
        }, L);
        suite.add("to_coefficients@kernel" + size, [L, lg, seed] {
            auto t = std::make_shared<fps::ntt::Transform>(lg);
            auto A = std::make_shared<Leaves>(random_poly(L, seed), L, *t);
            return [A, L, t] {
                Leaves C(L);
                std::copy_n(A->words.get(), L, C.words.get());
                t->inverse(C.words.get());
                bench::keep(C);
            };
        }, L);
        suite.add("add_coefficients" + size, [L, seed] {
            return [A = fps::to_values(random_poly(L, seed), L), h = random_poly(L / 2, seed + 1), L]() mutable {
                fps::raw::add_coefficients<P>(A, h.span(), L / 2);
                bench::keep(h);
            };
        }, L);
        suite.add("add_coefficients@kernel" + size, [L, lg, seed] {
            auto t = std::make_shared<fps::ntt::Transform>(lg);
            auto A = std::make_shared<Leaves>(random_poly(L, seed), L, *t);
            auto C = std::make_shared<Leaves>(L);
            return [A, C, h = random_poly(L / 2, seed + 1), L, t]() mutable {
                std::copy_n(A->words.get(), L, C->words.get());
                t->inverse(C->words.get());
                std::uint32_t* out = h.span().data();
                fps::detail::lanewise(out, C->words.get() + L / 2, out, L / 2,
                                      []<class S>(typename S::V x, typename S::V y) { return Ops<S>::add(x, y); });
                bench::keep(h);
            };
        }, L);
        for (const auto layout : {fps::Layout::automatic, fps::Layout::pointwise}) {
            const std::string suffix = layout == fps::Layout::pointwise ? "_pointwise" : "";
            suite.add("values_product" + suffix + size, [L, seed, layout] {
                return [A = fps::to_values(random_poly(L / 2, seed), L, layout),
                        B = fps::to_values(random_poly(L / 2, seed + 1), L, layout)] { bench::keep(A * B); };
            }, L);
        }
        suite.add("to_values_pointwise" + size, [L, seed] {
            return [f = random_poly(L / 2, seed), L] { bench::keep(fps::to_values(f, L, fps::Layout::pointwise)); };
        }, L);
        suite.add("to_coefficients_pointwise" + size, [L, seed] {
            return [A = fps::to_values(random_poly(L, seed), L, fps::Layout::pointwise)] {
                bench::keep(fps::to_coefficients(A));
            };
        }, L);
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
    values(suite);
    return suite.run();
}
