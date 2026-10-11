// Tests the small kernel against the O(n m) model: every Schoolbook<S, R, P, U, Solo> this machine
// can run (every simd struct; R = 1, 2, 4, 6, 8; R = 4, U = 2; R = 6, U = 4; R = 8, U = 2; Solo
// with R = 4, U = 1 and R = 6, U = 4; six primes) and every candidate of select.hpp's list
// (run_variant) on all shapes up to 40 x 40, term counts at the bounds of the 64-bit sums, 0, 1,
// powers of two and their neighbours up to 1025, panels (m > 1024) and long factors; multiply()
// with its argument orders and empty factors, and with each candidate set for every shape class.
// Inputs random, or making the largest sums or sums near them. Words past the product must stay
// untouched. Then beats_ntt on the compiled defaults, measure_ntt and measure_variants (tune.hpp),
// and both choices through wisdom. Exits nonzero on a failure.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "backend/simd/select.hpp"
#include "backend/small/small.hpp"
#include "backend/small/tune.hpp"

namespace {

using fps::small::u32;
using fps::small::u64;
using Vec = std::vector<u32>;

constexpr u32 kGuard = 0xdeadbeef;
constexpr std::size_t kGuardWords = 16;
int failures = 0;
std::mt19937_64 rng(20261011);

void fail(const std::string& what) {
    if (++failures <= 20) std::printf("FAIL %s\n", what.c_str());
}

u32 power(u32 x, u64 e, u32 p) {
    u64 r = 1;
    for (; e; e >>= 1, x = u32(u64(x) * x % p))
        if (e & 1) r = r * x % p;
    return u32(r);
}

// kind 0: random below P; 1: all P - 1; 2: random, a quarter of them P - 1 and a quarter 0;
// 3: random in [P - 1024, P), so that sums near their bounds take many values. With montgomery,
// kinds 1 and 3 give values x 2^-32 instead: the kernel multiplies by b 2^32 (its Montgomery
// form), so those factors reach the largest sums.
template <u32 P>
Vec random_vec(std::size_t n, int kind, bool montgomery = false) {
    const u32 unscale = power(power(2, 32, P), P - 2, P);
    Vec a(n);
    for (u32& x : a) {
        const u64 r = rng();
        x = kind == 1 ? P - 1 : kind == 2 && r % 4 == 0 ? P - 1 : kind == 2 && r % 4 == 1 ? 0 : u32(r % P);
        if (kind == 3) x = P - 1 - u32(r % 1024);
        if (montgomery && (kind == 1 || kind == 3)) x = u32(u64(x) * unscale % P);
    }
    return a;
}

template <u32 P>
Vec model(const Vec& a, const Vec& b) {
    Vec c(a.empty() || b.empty() ? 0 : a.size() + b.size() - 1);
    fps::small::model::multiply<P>(a.data(), a.size(), b.data(), b.size(), c.data());
    return c;
}

// f(a, n, b, m, out) on an output with guard words after it, against want.
template <class F>
void check(const std::string& name, const Vec& a, const Vec& b, const Vec& want, F f) {
    Vec out(want.size() + kGuardWords, kGuard);
    f(a.data(), a.size(), b.data(), b.size(), out.data());
    const std::string where = name + " " + std::to_string(a.size()) + "x" + std::to_string(b.size());
    for (std::size_t k = 0; k < want.size(); ++k)
        if (out[k] != want[k])
            return fail(where + " k=" + std::to_string(k) + ": " + std::to_string(out[k]) + ", want " +
                        std::to_string(want[k]));
    for (std::size_t k = want.size(); k < out.size(); ++k)
        if (out[k] != kGuard) return fail(where + ": wrote past the product");
}

// Every Schoolbook<S, R, P, U, Solo> on one pair of factors, n >= m: R = 1, 2, 4, 6, 8 with U = 1,
// and unrolled loops and vector-at-a-time edges. The target's candidates: candidates().
template <class S, u32 P>
void schoolbooks(const Vec& a, const Vec& b, const Vec& want) {
    const auto one = [&]<int R, int U = 1, bool Solo = false> {
        check(std::string("Schoolbook<") + S::name + ", " + std::to_string(R) + ", " + std::to_string(P) + ", " +
                  std::to_string(U) + ", " + (Solo ? "true" : "false") + ">",
              a, b, want, [](const u32* x, std::size_t n, const u32* y, std::size_t m, u32* out) {
                  fps::small::Schoolbook<S, R, P, U, Solo>::run(x, n, y, m, out);
              });
    };
    one.template operator()<1>();
    one.template operator()<2>();
    one.template operator()<4>();
    one.template operator()<6>();
    one.template operator()<8>();
    one.template operator()<4, 2>();
    one.template operator()<6, 4>();
    one.template operator()<8, 2>();
    one.template operator()<4, 1, true>();
    one.template operator()<6, 4, true>();
}

// Every candidate of select.hpp's list through run_variant, n >= m.
template <u32 P>
void candidates(const Vec& a, const Vec& b, const Vec& want) {
    for (int v = 0; v < fps::small::kCandidates; ++v)
        check("run_variant " + fps::small::variant_names()[v] + " P=" + std::to_string(P), a, b, want,
              [v](const u32* x, std::size_t n, const u32* y, std::size_t m, u32* out) {
                  fps::small::run_variant<P>(v, x, n, y, m, out);
              });
}

template <u32 P>
void all_structs(std::size_t n, std::size_t m, int kind) {
    const Vec a = random_vec<P>(n, kind), b = random_vec<P>(m, kind, true), want = model<P>(a, b);
    fps::simd::for_each(fps::simd::All{}, [&]<class S> { schoolbooks<S, P>(a, b, want); });
    candidates<P>(a, b, want);
}

template <u32 P>
void small_shapes() {
    for (std::size_t n = 1; n <= 40; ++n)
        for (std::size_t m = 1; m <= n; ++m)
            for (int kind : {0, 1}) all_structs<P>(n, m, kind);
}

const std::vector<std::size_t> kEdges = {1,  2,   3,   4,   5,   7,   8,   9,   15,  16,   17,   31,   32,  33,
                                         63, 64,  65,  127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025};

void edge_shapes() {
    constexpr u32 P = 998244353;
    for (std::size_t n : kEdges)
        for (std::size_t m : kEdges)
            if (m <= n) all_structs<P>(n, m, int(rng() % 3));
}

// Term counts at the bounds of the sums (Sums<P>): around the first fold, the last step from zero
// and from a fold, and two folds; long a, so most blocks run every term.
template <u32 P>
void bounds() {
    using B = fps::small::Sums<P>;
    for (std::size_t m : {B::kLastFromZero, B::kLastFromZero + 1, B::kFold, B::kFold + 1,
                          B::kFold + B::kLastFromFold, B::kFold + B::kLastFromFold + 1, 2 * B::kFold,
                          2 * B::kFold + B::kLastFromFold + 1})
        for (int kind : {1, 3}) all_structs<P>(400, m, kind);
}

// Panels of b (m > 1024) and long factors.
template <u32 P>
void long_shapes() {
    for (const auto& [n, m] : std::vector<std::pair<std::size_t, std::size_t>>{
             {1025, 1025}, {2049, 1500}, {3000, 3000}, {5000, 2049}, {100000, 1}, {100000, 3}, {100003, 32}})
        for (int kind : {1, 2}) all_structs<P>(n, m, kind);
}

// multiply(), both orders, through spans; empty factors leave out alone. Then with each candidate
// set for every class, on shapes of every class.
void selected() {
    constexpr u32 P = 998244353;
    const auto run = [](const u32* a, std::size_t n, const u32* b, std::size_t m, u32* out) {
        fps::small::multiply<P>(std::span(a, n), std::span(b, m), std::span(out, n && m ? n + m - 1 : 0));
    };
    for (std::size_t n : kEdges)
        for (std::size_t m : {std::size_t(0), std::size_t(1), std::size_t(5), std::size_t(33), std::size_t(700)}) {
            const Vec a = random_vec<P>(n, 2), b = random_vec<P>(m, 2);
            check("multiply", a, b, model<P>(a, b), run);
            check("multiply", b, a, model<P>(b, a), run);
        }
    check("multiply", Vec{}, Vec{}, Vec{}, run);

    using namespace fps::small;
    std::vector<int> seen(kClasses);
    for (int v = 0; v < kCandidates; ++v) {
        Choices forced;
        forced.fill(std::uint8_t(v));
        variants.set(forced);
        for (std::size_t m : {1, 2, 3, 6, 13, 30, 47, 100, 200, 300, 700, 1100})
            for (std::size_t n : {m, m + m / 2, 3 * m, 5 * m, 9 * m}) {
                ++seen[shape_class(n, m)];
                const Vec a = random_vec<P>(n, 2), b = random_vec<P>(m, 2);
                check("multiply, variant " + variant_names()[v], a, b, model<P>(a, b), run);
            }
    }
    variants.set(kDefaults);
    if (std::ranges::count(seen, 0)) fail("selected: a class without a shape");
}

// beats_ntt: short products small, long ones the NTT, past 2^25 too.
void cutoff() {
    using fps::small::beats_ntt;
    const std::size_t big = std::size_t(1) << 20;
    if (!beats_ntt(0, 5) || !beats_ntt(1, 1) || !beats_ntt(8, 8) || !beats_ntt(big, 2) || !beats_ntt(2, big))
        fail("beats_ntt: a short product goes to the NTT");
    if (beats_ntt(big, big) || beats_ntt(1024, 1024) || beats_ntt(big, 1024) || beats_ntt(std::size_t(1) << 26, 1 << 26))
        fail("beats_ntt: a long product goes to small");
    if (!beats_ntt(std::size_t(1) << 27, 1)) fail("beats_ntt: x^k past 2^25 goes to the NTT");
}

// measure_ntt with a small budget: times positive; past max_lg the start times the ratio at
// max_lg; a margin above every change keeps the start. Then the choice "small.ntt" through wisdom.
void tuned() {
    using namespace fps::small;
    const auto times = measure_ntt(kModel, kNttDefaults, {8, 1, 1e4, 0});
    for (const double t : times)
        if (!(t > 0 && std::isfinite(t))) fail("measure_ntt: a time of " + std::to_string(t));
    const double ratio = times[8 - 6] / kNttDefaults[8 - 6];
    for (int lg = 9; lg <= 25; ++lg)
        if (std::abs(times[lg - 6] / (kNttDefaults[lg - 6] * ratio) - 1) > 1e-12)
            fail("measure_ntt: lg " + std::to_string(lg) + " is not the default times the ratio at max_lg");
    if (measure_ntt(kModel, kNttDefaults, {8, 1, 1e4, 1e9}) != kNttDefaults) fail("measure_ntt: changed within the margin");

    const std::string path =
        (std::filesystem::temp_directory_path() / ("fps-small-test-" + std::to_string(::getpid()))).string();
    ntt_times.set(times);
    if (!fps::tune::save(path)) fail("save");
    ntt_times.set(kNttDefaults);
    if (!fps::tune::load(path) || ntt_times.get() != times) fail("small.ntt did not load back from wisdom");
    std::ifstream in(path);
    bool found = false;
    for (std::string line; std::getline(in, line);) found |= line == "small.ntt " + fps::tune::text(times);
    if (!found) fail("no small.ntt line in the wisdom");
    std::filesystem::remove(path);
    ntt_times.set(kNttDefaults);
}

// Shape classes and their points; measure_variants at a small budget (valid; classes above
// max_octave keep start's; a margin above every change keeps start); the wisdom text; and
// "small.variant" through tune::save / load, before "small.ntt".
void tuned_variants() {
    using namespace fps::small;
    if (shape_class(1, 1) != 0 || shape_class(7, 4) != 2 * kRatios || shape_class(8, 4) != 2 * kRatios + 1 ||
        shape_class(31, 4) != 2 * kRatios + 2 || shape_class(32, 4) != 2 * kRatios + 3 ||
        shape_class(512, 512) != 9 * kRatios || shape_class(513, 513) != 10 * kRatios ||
        shape_class(std::size_t(1) << 30, 5000) != kClasses - 1)
        fail("shape_class");
    for (int c = 0; c < kClasses; ++c)
        for (const bool dense : {false, true}) {
            const auto shapes = points(c, dense);
            if (shapes.empty()) fail("points: none for class " + std::to_string(c));
            for (const auto& [n, m] : shapes)
                if (n < m || shape_class(n, m) != c) fail("points: a shape outside class " + std::to_string(c));
        }

    std::vector<ClassTimes> log;
    const Choices measured = measure_variants(kDefaults, {3, false, 1, 2e3, 0}, &log);
    if (!valid(measured) || log.size() != 4 * kRatios) fail("measure_variants: not a valid table, or classes missed");
    for (int c = 4 * kRatios; c < kClasses; ++c)
        if (measured[c] != kDefaults[c]) fail("measure_variants: changed a class above max_octave");
    for (const ClassTimes& t : log) {
        const std::vector<double>& ns = t.ns[0];
        if (t.ns.size() != t.shapes.size() || int(ns.size()) != kCandidates || !std::ranges::all_of(ns, [](double x) {
                return x > 0 && std::isfinite(x);
            }))
            fail("measure_variants: times of class " + std::to_string(t.c));
    }
    if (measure_variants(kDefaults, {3, false, 1, 2e3, 1e9}) != kDefaults) fail("measure_variants: changed within the margin");

    if (parse(text(measured)) != measured) fail("variant text does not read back");
    std::string bad = text(measured);
    bad[0] = bad[0] == '1' ? '2' : '1';
    if (parse(bad)) fail("parse took another candidate list's signature");
    Choices too_large = measured;
    too_large[5] = std::uint8_t(kCandidates);
    if (parse(text(too_large))) fail("parse took a variant that does not exist");

    const std::string path =
        (std::filesystem::temp_directory_path() / ("fps-small-test-v-" + std::to_string(::getpid()))).string();
    Choices other = kDefaults;
    other[7] = std::uint8_t((other[7] + 1) % kCandidates);
    variants.set(other);
    if (!fps::tune::save(path)) fail("save");
    variants.set(kDefaults);
    if (!fps::tune::load(path) || variants.get() != other) fail("small.variant did not load back from wisdom");
    std::ifstream in(path);
    int variant_line = -1, ntt_line = -1, i = 0;
    for (std::string line; std::getline(in, line); ++i) {
        if (line == "small.variant " + text(other)) variant_line = i;
        if (line.starts_with("small.ntt ")) ntt_line = i;
    }
    if (variant_line < 0 || ntt_line < variant_line) fail("wisdom: no small.variant line, or not before small.ntt");
    std::filesystem::remove(path);
    variants.set(kDefaults);
}

}  // namespace

int main() {
    // The model on a hand example: (1 + 2x)(3 + x + (P - 1) x^2) = 3 + 7x + (P + 1) x^2 + (2P - 2) x^3.
    constexpr u32 P = 998244353;
    if (model<P>({1, 2}, {3, 1, P - 1}) != Vec{3, 7, 1, P - 2}) fail("model");
    small_shapes<4099>();
    small_shapes<65537>();
    small_shapes<469762049>();
    small_shapes<998244353>();
    small_shapes<1000000007>();
    small_shapes<1073741789>();  // the largest prime below 2^30
    bounds<469762049>();
    bounds<998244353>();
    bounds<1000000007>();
    bounds<1073741789>();
    edge_shapes();
    long_shapes<998244353>();
    long_shapes<1073741789>();
    selected();
    cutoff();
    tuned();
    tuned_variants();
    std::printf("%s\n", failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
