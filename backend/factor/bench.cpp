// Times backend/factor; tools/kernel.py runs it. Items are the degree n = 2^lg of f.
//   "factor/<lg>"               random f of degree n (a few factors of distinct degrees)
//   "factor_split/<lg>"         n / 2 random irreducible quadratics (lc-opt's same_degree: Berlekamp
//                               with k = n / 2, the split by the minimal polynomial)
//   "<case>@avx512vl/<lg>"      AVX-512 targets: on ymm (Avx512Vl) instead of zmm (not gated)
//   "<case>@frontend/<lg>"      fps::factor (include/fps/algebra.hpp: distinct-degree split and
//                               Cantor-Zassenhaus) on the same f, factor to 2^8, factor_split to
//                               2^6 (not gated)
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "backend/factor/factor.hpp"
#include "backend/simd/select.hpp"
#include "fps/algebra.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = 998244353;

u32 power(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = u32(u64(a) * a % P))
        if (e & 1) r = u32(u64(r) * a % P);
    return r;
}

Vec multiply(const Vec& a, const Vec& b) {
    Vec c(a.size() + b.size() - 1, 0);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = u32((c[i + j] + u64(a[i]) * b[j]) % P);
    return c;
}

Vec random_poly(std::size_t n, unsigned seed) {  // degree n
    std::mt19937 rng(seed);
    Vec a(n + 1);
    for (u32& x : a) x = rng() % P;
    if (a.back() == 0) a.back() = 1;
    return a;
}

// k random monic irreducible quadratics x^2 + b x + c (b^2 - 4c not a square).
Vec quadratics(std::size_t k, unsigned seed) {
    std::mt19937 rng(seed);
    Vec f = {1};
    for (std::size_t i = 0; i < k; ++i) {
        u32 b, c;
        do {
            b = rng() % P, c = rng() % P;
        } while (power(u32((u64(b) * b + 4 * u64(P - c)) % P), (P - 1) / 2) != P - 1);
        f = multiply(f, {c, b, 1});
    }
    return f;
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    namespace fz = fps::factorization;
    using Poly = fps::Polynomial<P>;
    for (int lg = 4; lg <= 8; ++lg) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        const auto add = [&](const std::string& name, const Vec& f, bool frontend) {
            suite.add(name + size, [f] { return [f] { bench::keep(fz::factor(f)); }; }, n);
#if defined(__AVX512F__) && defined(__AVX512VL__)
            suite.add(name + "@avx512vl" + size, [f] {
                return [f] { bench::keep(fz::factor_as<fps::simd::Avx512Vl>(f)); };
            }, n);
#endif
            if (frontend)
                suite.add(name + "@frontend" + size, [f] {
                    return [f = Poly(std::span<const u32>(f))] { bench::keep(fps::factor(f)); };
                }, n);
        };
        add("factor", random_poly(n, unsigned(11 * lg)), true);
        add("factor_split", quadratics(n / 2, unsigned(13 * lg)), lg <= 6);
    }
    return suite.run();
}
