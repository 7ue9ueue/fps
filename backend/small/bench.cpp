// Times the small kernel modulo P = 998244353 on factors of n and m coefficients into an output
// allocated once; items are terms n m.
//   "multiply/<n>x<m>"                  the selected variant (gated)
//   "multiply@<variant>/<n>x<m>"        each candidate of select.hpp's list, e.g. avx512.r8.u2
//                                       (run_variant)
//   "multiply@ntt/<n>x<m>"              ntt::Convolution on the same factors with the product in
//                                       the same output: construct, copy them in, multiply
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "backend/bench.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/small/small.hpp"

namespace {

using fps::small::u32;

constexpr u32 P = 998244353;

struct Shape {
    std::size_t n, m;
    std::string name() const { return std::to_string(n) + "x" + std::to_string(m); }
};

const std::vector<Shape> kShapes = {
    {8, 8},      {16, 16},    {32, 32},     {48, 48},     {64, 64},     {96, 96},     {128, 128},  {192, 192},
    {256, 256},  {384, 384},  {512, 512},   {4096, 4},    {4096, 8},    {4096, 16},   {4096, 32},  {4096, 64},
    {4096, 128}, {16384, 32}, {16384, 64}, {16384, 128}, {131072, 32}, {131072, 64},
};

std::vector<u32> random_vec(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<u32> a(n);
    for (u32& x : a) x = rng() % P;
    return a;
}

using Factors = std::vector<u32>;

// The output: 64-byte aligned, room for the Convolution's length and padding.
struct Output {
    struct alignas(64) Line {
        u32 words[16];
    };
    explicit Output(const Shape& s)
        : lines((std::max(s.n + s.m - 1, fps::ntt::Convolution::length(s.n, s.m) + fps::ntt::kPadding) + 15) / 16) {}
    u32* data() { return lines.data()->words; }
    std::vector<Line> lines;
};

template <class F>
void add(bench::Suite& suite, const std::string& op, const Shape& s, F f) {
    suite.add(op + "/" + s.name(), [s, f] {
        return [a = random_vec(s.n, unsigned(s.n)), b = random_vec(s.m, unsigned(s.m) + 1), out = Output(s),
                f]() mutable {
            f(a, b, out.data());
            bench::keep(out.data()[0]);
        };
    }, s.n * s.m);
}

template <class... V>
void candidates(bench::Suite& suite, const Shape& s, fps::small::Candidates<V...>) {
    (add(suite, "multiply@" + fps::small::variant_name<V>(), s,
         [](const Factors& a, const Factors& b, u32* out) {
             fps::small::SchoolbookOf<V, P>::run(a.data(), a.size(), b.data(), b.size(), out);
         }),
     ...);
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    for (const Shape& s : kShapes)
        add(suite, "multiply", s, [](const Factors& a, const Factors& b, u32* out) {
            fps::small::multiply<P>(a, b, std::span(out, a.size() + b.size() - 1));
        });
    for (const Shape& s : kShapes) candidates(suite, s, fps::small::Selected{});
    for (const Shape& s : kShapes)
        add(suite, "multiply@ntt", s, [](const Factors& a, const Factors& b, u32* out) {
            fps::ntt::Convolution conv(a.size(), b.size(), out);
            std::ranges::copy(a, out);
            std::ranges::copy(b, conv.b());
            conv.multiply();
        });
    return suite.run();
}
