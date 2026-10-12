// Times fps::Values modulo P = 998244353; tools/kernel.py runs it next to bench.cpp, with the same
// case names ("<op>/<lg>" through the API, "<op>@kernel/<lg>" the backend call it wraps). The
// products of value forms and the pointwise layout wrap no backend kernel yet (frontend/notes.md),
// so their cases have no @kernel pair.
//
// A binary of its own: with ntt::Transform's forward and inverse in the program, GCC 15.2 inlines
// the avx2 Convolution differently, and bench.cpp's f * g ran 25% slower at 32 x 32 coefficients on
// Zen 3 (multiply/6; frontend/notes.md). Its own binary keeps bench.cpp's cases as main compiles
// them. The library keeps the tables and scratch of value forms, so cases here run in that state.
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
    values(suite);
    return suite.run();
}
