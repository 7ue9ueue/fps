// Times fps::Values modulo P = 998244353; tools/kernel.py runs it next to bench.cpp, with the same
// case names ("<op>/<lg>" through the API, "<op>@kernel/<lg>" the backend call it wraps).
// Products in the pointwise layout are the frontend's own stream: no @kernel pair.
//
// A binary of its own: with these cases in bench.cpp, its f * g ran 25% slower at 32 x 32
// coefficients on Zen 3 (multiply/6). Not the NTT's inlining: with the product's functions
// compiled identically it stayed (issue #81, backend/ntt/notes.md); products of 2^6-2^8 move by
// up to 25% with the program's code layout. Its own binary keeps bench.cpp laid out as on main.
// The library keeps the tables and scratch of value forms, so cases here run in that state.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "fps/fps.hpp"

namespace {

constexpr std::uint32_t P = 998244353;
using Poly = fps::Polynomial<P>;

template <class S>
using Ops = fps::modarith::Ops<S, P>;

Poly random_poly(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<std::uint32_t> c(n);
    for (std::uint32_t& x : c) x = rng() % P;
    return Poly(std::span<const std::uint32_t>(c));
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
    // f, zeros up to L, then the forward transform into the leaf or values form.
    Leaves(const Poly& f, std::size_t L, const fps::ntt::Transform& t, fps::ntt::Form form = fps::ntt::Form::leaves)
        : Leaves(L) {
        std::ranges::copy(f.span(), words.get());
        std::fill(words.get() + f.size(), words.get() + L, 0u);
        if (form == fps::ntt::Form::leaves) t.forward(words.get());
        else t.forward(words.get(), form);
    }
};

// fps::Values against ntt::Transform on L = 2^lg words. to_values: f of L / 2 coefficients (an
// exact product's factor) into new memory; the kernel builds its tables in setup, as the API
// keeps them per length. to_coefficients: a value form of L coefficients back into new memory, by
// the inverse into other memory (@copy: what the API did before it, a copy of the leaf form, the
// inverse in place, a copy out). add_coefficients: the upper half added into the same kept array,
// ABC222 H's step. values_product: A * B into new memory, leaf products.
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
                t->inverse(A->words.get(), std::span(C.words.get(), L));
                bench::keep(C);
            };
        }, L);
        suite.add("to_coefficients@copy" + size, [L, lg, seed] {
            auto t = std::make_shared<fps::ntt::Transform>(lg);
            auto A = std::make_shared<Leaves>(random_poly(L, seed), L, *t);
            return [A, L, t] {
                Leaves C(L), D(L);
                std::copy_n(A->words.get(), L, C.words.get());
                t->inverse(C.words.get());
                std::copy_n(C.words.get(), L, D.words.get());
                bench::keep(D);
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
            return [A, h = random_poly(L / 2, seed + 1), L, t]() mutable {
                t->inverse(A->words.get(), h.span(), fps::ntt::Form::leaves, {L / 2, true});
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
        suite.add("values_product@kernel" + size, [L, lg, seed] {
            auto t = std::make_shared<fps::ntt::Transform>(lg);
            auto A = std::make_shared<Leaves>(random_poly(L / 2, seed), L, *t);
            auto B = std::make_shared<Leaves>(random_poly(L / 2, seed + 1), L, *t);
            return [A, B, L, t] {
                Leaves C(L);
                t->multiply(A->words.get(), B->words.get(), C.words.get());
                bench::keep(C);
            };
        }, L);
        if (lg > fps::ntt::kValuesMaxLog) continue;
        suite.add("to_values_pointwise" + size, [L, seed] {
            return [f = random_poly(L / 2, seed), L] { bench::keep(fps::to_values(f, L, fps::Layout::pointwise)); };
        }, L);
        suite.add("to_values_pointwise@kernel" + size, [L, lg, seed] {
            return [f = random_poly(L / 2, seed), L, t = std::make_shared<fps::ntt::Transform>(lg)] {
                bench::keep(Leaves(f, L, *t, fps::ntt::Form::values));
            };
        }, L);
        suite.add("to_coefficients_pointwise" + size, [L, seed] {
            return [A = fps::to_values(random_poly(L, seed), L, fps::Layout::pointwise)] {
                bench::keep(fps::to_coefficients(A));
            };
        }, L);
        suite.add("to_coefficients_pointwise@kernel" + size, [L, lg, seed] {
            auto t = std::make_shared<fps::ntt::Transform>(lg);
            auto A = std::make_shared<Leaves>(random_poly(L, seed), L, *t, fps::ntt::Form::values);
            return [A, L, t] {
                Leaves C(L);
                t->inverse(A->words.get(), std::span(C.words.get(), L), fps::ntt::Form::values);
                bench::keep(C);
            };
        }, L);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    values(suite);
    return suite.run();
}
