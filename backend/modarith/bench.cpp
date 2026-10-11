// Times modarith operations modulo P = 998244353 on arrays of 2048 words (24 KiB with the output:
// in L1 on every target); items are words.
//   "<op>/2048"                 the selected variant on simd::Native (gated)
//   "<op>@<variant>/2048"       one variant on simd::Native
//   "<op>@<variant>.ymm/2048"   one variant on Avx512Vl, on AVX-512 targets
//   "<op>_chain@<variant>/1024" 1024 dependent operations on one vector: latency per item
// Streams: montgomery and barrett read a and b, shoup reads x (w is a constant); each writes out.
#include <cstdint>
#include <string>

#include "backend/bench.hpp"
#include "backend/modarith/modarith.hpp"
#include "backend/simd/select.hpp"
#include "backend/simd/stream.hpp"

namespace {

using fps::modarith::Barrett;
using fps::modarith::Montgomery;
using fps::modarith::Shoup;
using fps::modarith::u32;

constexpr u32 P = 998244353;
constexpr std::size_t kWords = 2048, kChain = 1024;
constexpr u32 kW = 123456789;  // the constant factor of shoup

template <class S, class Op>
void stream(bench::Suite& suite, const std::string& name, Op op) {
    fps::simd::add_stream<S>(suite, name, kWords, P, op);
}

// x = op(x, c), kChain times; x and c below P at the start, x below 2P after each step. c is
// state of the timed call, so the compiler cannot fold products of constants.
template <class S, class Op>
void chain(bench::Suite& suite, const std::string& name, Op op) {
    suite.add(name + "/" + std::to_string(kChain), [op] {
        return [x = S::broadcast(12345), c = S::broadcast(kW), op]() mutable {
            for (std::size_t i = 0; i < kChain; ++i) x = op(x, c);
            bench::keep(x);
        };
    }, kChain);
}

template <class S>
void variants(bench::Suite& suite, const std::string& suffix) {
    using O = fps::modarith::Ops<S, P>;
    using V = typename S::V;
    const auto w = S::broadcast(kW), q = S::broadcast(O::shoup_quotient(kW));
    const auto montgomery = [&]<Montgomery v>(const char* name) {
        if constexpr (fps::modarith::has<S>(v)) {
            const auto op = [](V a, V b) { return O::template montgomery_as<v>(a, b); };
            stream<S>(suite, std::string("montgomery@") + name + suffix, op);
            chain<S>(suite, std::string("montgomery_chain@") + name + suffix, op);
        }
    };
    montgomery.template operator()<Montgomery::wide>("wide");
    montgomery.template operator()<Montgomery::mulhi>("mulhi");
    montgomery.template operator()<Montgomery::doubling>("doubling");
    const auto shoup = [&]<Shoup v>(const char* name) {
        if constexpr (fps::modarith::has<S>(v)) {
            const auto op = [w, q](V x, V) { return O::template shoup_as<v>(x, w, q); };
            stream<S>(suite, std::string("shoup@") + name + suffix, op);
            chain<S>(suite, std::string("shoup_chain@") + name + suffix, op);
        }
    };
    shoup.template operator()<Shoup::mulhi>("mulhi");
    shoup.template operator()<Shoup::wide>("wide");
    if constexpr (requires(V x) { O::shoup_doubling(x, x, x); }) {
        const auto q31 = S::broadcast(O::shoup_quotient31(kW));
        const auto op = [w, q31](V x, V) { return O::shoup_doubling(x, w, q31); };
        stream<S>(suite, "shoup@doubling" + suffix, op);
        chain<S>(suite, "shoup_chain@doubling" + suffix, op);
    }
    if constexpr (requires(V x) { O::montgomery52(x, x); }) {
        const auto mul = [](V a, V b) { return O::montgomery52(a, b); };  // lambdas inline; pointers may not
        stream<S>(suite, "montgomery@ifma" + suffix, mul);
        chain<S>(suite, "montgomery_chain@ifma" + suffix, mul);
        const auto w64 = S::broadcast64(kW), q52 = S::broadcast64(O::shoup_quotient52(kW));
        const auto op = [w64, q52](V x, V) { return O::shoup52(x, w64, q52); };
        stream<S>(suite, "shoup@ifma" + suffix, op);
        chain<S>(suite, "shoup_chain@ifma" + suffix, op);
    }
    // Barrett needs a < P: the chain reduces first (one sub and one min more per step).
    const auto barrett = [&]<Barrett v>(const char* name) {
        if constexpr (fps::modarith::has<S>(v)) {
            stream<S>(suite, std::string("barrett@") + name + suffix,
                      [](V a, V b) { return O::template barrett_as<v>(a, b); });
            chain<S>(suite, std::string("barrett_chain@") + name + suffix,
                     [](V x, V c) { return O::template barrett_as<v>(O::template reduce<P>(x), c); });
        }
    };
    barrett.template operator()<Barrett::mulhi>("mulhi");
    barrett.template operator()<Barrett::wide>("wide");
    barrett.template operator()<Barrett::split>("split");
}

}  // namespace

int main(int argc, char** argv) {
    using S = fps::simd::Native;
    using O = fps::modarith::Ops<S, P>;
    using V = S::V;
    bench::Suite suite(argc, argv);
    const auto w = S::broadcast(kW), q = S::broadcast(O::shoup_quotient(kW));
    stream<S>(suite, "add", [](V a, V b) { return O::add(a, b); });
    stream<S>(suite, "sub", [](V a, V b) { return O::sub(a, b); });
    stream<S>(suite, "montgomery", [](V a, V b) { return O::montgomery(a, b); });
    stream<S>(suite, "shoup", [w, q](V x, V) { return O::shoup(x, w, q); });
    stream<S>(suite, "barrett", [](V a, V b) { return O::barrett(a, b); });
    variants<S>(suite, "");
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    variants<fps::simd::Avx512Vl>(suite, ".ymm");
#endif
    return suite.run();
}
