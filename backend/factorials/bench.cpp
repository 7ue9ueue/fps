// Times the factorials tables modulo P = 998244353 into buffers allocated once (warm, no page
// faults); items are table positions n (each gives two words).
//   "tables/<lg>"              the selected variant (gated)
//   "tables@loop/<lg>"         the scalar model, the frontend's loop before this kernel
//   "tables@<isa>.<K>.<step>[.prefetch]/<lg>"
//                              each candidate's Chains (select.hpp), at lg 6, 10, 14, 18, 22
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

template <class V>
void candidate(bench::Suite& suite) {
    using C = fps::factorials::ChainsOf<V, P>;
    const std::string name = std::string("tables@") + V::Isa::name + "." + std::to_string(V::vectors) + "." +
                             std::string(fps::factorials::step_name<V>) +
                             std::string(fps::factorials::prefetch_name<V>);
    for (int lg : {6, 10, 14, 18, 22})
        add(suite, name, lg, [](u32* fact, u32* inverse, std::size_t n) { C::run(fact, inverse, n); });
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
    [&]<class... V>(fps::factorials::Candidates<V...>) { (candidate<V>(suite), ...); }(fps::factorials::Selected{});
    return suite.run();
}
