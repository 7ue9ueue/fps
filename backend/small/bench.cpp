// Times the small kernel modulo P = 998244353 on factors of n and m coefficients into an output
// allocated once; items are terms n m.
//   "multiply/<n>x<m>"                  the selected variant (gated)
//   "multiply@<variant>/<n>x<m>"        each candidate of select.hpp's list, e.g. avx512.r8.u2
//                                       (run_variant)
//   "multiply@ntt/<n>x<m>"              ntt::multiply on the same factors into the same output, as
//                                       f * g runs it on the NTT path (ragged and truncated plans)
//   "product/<n>x<m>"                   small::multiply where small::beats_ntt(n, m), else
//                                       ntt::multiply: f * g's choice (gated), at shapes near the
//                                       crossover, many just past a power of two (issue #212)
//   "multiply@small/<n>x<m>"            small::multiply at those shapes
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
    bool operator==(const Shape&) const = default;
    std::string name() const { return std::to_string(n) + "x" + std::to_string(m); }
};

const std::vector<Shape> kShapes = {
    {8, 8},      {16, 16},    {32, 32},     {48, 48},     {64, 64},     {96, 96},     {128, 128},  {192, 192},
    {256, 256},  {384, 384},  {512, 512},   {4096, 4},    {4096, 8},    {4096, 16},   {4096, 32},  {4096, 64},
    {4096, 128}, {16384, 32}, {16384, 64}, {16384, 128}, {131072, 32}, {131072, 64},
};

// Near the crossover against the NTT on the timed targets: balanced factors whose product passes
// 2^6, 2^7, 2^8; 2^k x m (n + m - 1 just past 2^k); n + m - 1 = 2^k + 1 and within (2^k, 2^(k+1));
// just below 2^k (the plain product at 2^k, controls).
const std::vector<Shape> kCrossShapes = {
    {24, 24}, {32, 32}, {33, 33}, {40, 40}, {48, 48}, {56, 56}, {64, 64}, {65, 65}, {80, 80}, {96, 96},
    {112, 112}, {129, 129}, {160, 160}, {256, 16}, {256, 32}, {256, 48}, {256, 64}, {320, 32}, {384, 48},
    {640, 48}, {768, 64}, {512, 24}, {512, 48}, {512, 64}, {512, 96}, {1024, 24}, {1024, 32}, {1024, 48},
    {1024, 64}, {1024, 80}, {1024, 96}, {1024, 112}, {1000, 25}, {1000, 50}, {1000, 75}, {1000, 100},
    {1280, 48}, {1280, 96}, {1536, 48}, {1536, 96}, {1792, 64}, {4096, 48}, {4096, 96}, {4067, 30}, {4033, 64},
    {4000, 97}, {16384, 64}, {16384, 96}, {16384, 128}, {16384, 160}, {16000, 100}, {16000, 160}, {16289, 96},
    {1048576, 64}, {1048576, 96}, {1048576, 128}, {1048576, 160}, {1048481, 96},
};

std::vector<u32> random_vec(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<u32> a(n);
    for (u32& x : a) x = rng() % P;
    return a;
}

using Factors = std::vector<u32>;

// The output: 64-byte aligned, n + m - 1 words.
struct Output {
    struct alignas(64) Line {
        u32 words[16];
    };
    explicit Output(const Shape& s) : lines((s.n + s.m - 1 + 15) / 16) {}
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
    const auto small = [](const Factors& a, const Factors& b, u32* out) {
        fps::small::multiply<P>(a, b, std::span(out, a.size() + b.size() - 1));
    };
    const auto ntt = [](const Factors& a, const Factors& b, u32* out) {
        fps::ntt::multiply(a, b, std::span(out, a.size() + b.size() - 1));
    };
    for (const Shape& s : kShapes) add(suite, "multiply@ntt", s, ntt);
    for (const Shape& s : kCrossShapes)
        add(suite, "product", s, [small, ntt](const Factors& a, const Factors& b, u32* out) {
            if (fps::small::beats_ntt(a.size(), b.size())) small(a, b, out);
            else ntt(a, b, out);
        });
    for (const Shape& s : kCrossShapes) add(suite, "multiply@small", s, small);
    for (const Shape& s : kCrossShapes)
        if (std::ranges::find(kShapes, s) == kShapes.end()) add(suite, "multiply@ntt", s, ntt);
    return suite.run();
}
