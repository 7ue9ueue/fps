// Times the composed simd products on arrays of 2048 words (24 KiB with the output: in L1 on
// every target); items are words.
//   "<op>/2048"          on simd::Native (gated)
//   "<op>@<isa>/2048"    on each struct this target can run
// out[i] = op(a[i], b[i]); a, b random below 2^30 (so mul_shr<29> fits in 32 bits).
// Instruction latencies and throughputs in cycles: probe.cpp.
#include <cstdint>
#include <string>

#include "backend/bench.hpp"
#include "backend/simd/select.hpp"
#include "backend/simd/stream.hpp"

namespace {

using fps::simd::u32;

constexpr std::size_t kWords = 2048;

template <class S, class Op>
void stream(bench::Suite& suite, const std::string& name, Op op) {
    fps::simd::add_stream<S>(suite, name, kWords, 1u << 30, op);
}

template <class S>
void ops(bench::Suite& suite, const std::string& suffix) {
    using V = typename S::V;
    stream<S>(suite, "mul_lo" + suffix, [](V a, V b) { return S::mul_lo(a, b); });
    stream<S>(suite, "mulhi" + suffix, [](V a, V b) { return S::mulhi(a, b); });
    stream<S>(suite, "mul_shr" + suffix, [](V a, V b) { return S::template mul_shr<29>(a, b); });
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    ops<fps::simd::Native>(suite, "");
    fps::simd::for_each(fps::simd::All{}, [&]<class S> { ops<S>(suite, std::string("@") + S::name); });
    return suite.run();
}
