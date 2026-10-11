// Times the factorials tables modulo P = 998244353 into buffers allocated once (warm, no page
// faults); items are table positions n (each gives two words).
//   "tables/<lg>"              the selected variant (gated)
//   "tables@loop/<lg>"         the scalar model, the frontend's loop before this kernel
//   "tables@<isa>.<K>/<lg>"    Chains on simd::Native (and Avx512Vl on AVX-512) with K vectors
//                              per chain
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "backend/factorials/factorials.hpp"
#include "backend/simd/select.hpp"

namespace {

using fps::factorials::u32;

constexpr u32 P = 998244353;

template <class F>
void add(bench::Suite& suite, const std::string& name, int lg, F f) {
    const std::size_t n = std::size_t(1) << lg;
    suite.add(name + "/" + std::to_string(lg), [n, f] {
        return [fact = std::vector<u32>(n), inverse = std::vector<u32>(n), n, f]() mutable {
            f(fact.data(), inverse.data(), n);
            bench::keep(fact[0]);
            bench::keep(inverse[0]);
        };
    }, n);
}

template <class S, int K>
void chains(bench::Suite& suite) {
    using C = fps::factorials::Chains<S, K, P>;
    for (int lg : {10, 14, 18, 22})
        add(suite, std::string("tables@") + S::name + "." + std::to_string(K), lg,
            [](u32* fact, u32* inverse, std::size_t n) { C::run(fact, inverse, n); });
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    for (int lg = 6; lg <= 22; lg += 2)
        add(suite, "tables", lg, [](u32* fact, u32* inverse, std::size_t n) {
            fps::factorials::tables<P>(std::span(fact, n), std::span(inverse, n));
        });
    for (int lg : {6, 10, 14, 18, 22})
        add(suite, "tables@loop", lg, [](u32* fact, u32* inverse, std::size_t n) {
            fps::factorials::model::tables<P>(fact, inverse, n);
        });
    chains<fps::simd::Native, 2>(suite);
    chains<fps::simd::Native, 4>(suite);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    chains<fps::simd::Avx512Vl, 2>(suite);
#endif
    return suite.run();
}
