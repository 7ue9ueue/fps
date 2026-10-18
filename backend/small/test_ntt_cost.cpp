// Tests beats_ntt's NTT times at the lengths the NTT runs (issue #212): ntt_point, ntt_ns's
// interpolation (the values at the points, linear between them, the ends of each length class, past
// 2^25), beats_ntt against ntt_ns on the compiled defaults and on set tables, measure_ragged at a
// small budget, and "small.ragged" through wisdom. Exits nonzero on a failure.
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

#include "backend/small/small.hpp"
#include "backend/small/tune.hpp"

namespace {

using namespace fps::small;
using Top = std::array<double, 20>;
using Low = std::array<double, 20 * kRaggedPoints>;

int failures = 0;

void fail(const std::string& what) {
    std::printf("FAIL %s\n", what.c_str());
    ++failures;
}

bool near(double x, double y) { return std::abs(x - y) <= 1e-9 * std::abs(y); }

// Tables whose values are distinct and grow with N: point j of lg is 2^(lg-3) (4 + j), 2^lg's is 2^lg.
void tables(Top& top, Low& low) {
    for (int lg = 6; lg <= 25; ++lg) {
        for (int j = 0; j < kRaggedPoints; ++j) low[kRaggedPoints * (lg - 6) + j] = std::ldexp(4 + j, lg - 3);
        top[lg - 6] = std::ldexp(1, lg);
    }
}

void points() {
    for (int lg = 6; lg <= 25; ++lg) {
        const std::size_t half = std::size_t(1) << (lg - 1);
        if (ntt_point(lg, 0) != half + 1 || ntt_point(lg, 1) != half + half / 4 || ntt_point(lg, 2) != 3 * half / 2 ||
            ntt_point(lg, 3) != half + 3 * half / 4 || ntt_point(lg, kRaggedPoints) != 2 * half)
            fail("ntt_point at lg " + std::to_string(lg));
        for (int j = 0; j <= kRaggedPoints; ++j)
            if (std::max(6, int(std::bit_width(ntt_point(lg, j) - 1))) != lg)
                fail("ntt_point outside its class at lg " + std::to_string(lg));
    }
}

void interpolation() {
    Top top;
    Low low;
    tables(top, low);
    for (int lg = 6; lg <= 25; ++lg) {
        const std::size_t half = std::size_t(1) << (lg - 1);
        if (!near(ntt_ns(2 * half, top, low), top[lg - 6])) fail("ntt_ns at 2^lg, lg " + std::to_string(lg));
        for (int j = 1; j < kRaggedPoints; ++j)
            if (!near(ntt_ns(ntt_point(lg, j), top, low), low[kRaggedPoints * (lg - 6) + j]))
                fail("ntt_ns at point " + std::to_string(j) + ", lg " + std::to_string(lg));
        // 2^(lg-1) + 1 counts as 2^(lg-1): point 0 plus 1 / (half / 4) of the step to point 1.
        const double step = low[kRaggedPoints * (lg - 6) + 1] - low[kRaggedPoints * (lg - 6)];
        if (!near(ntt_ns(half + 1, top, low), low[kRaggedPoints * (lg - 6)] + step * 4 / double(half)))
            fail("ntt_ns at 2^(lg-1) + 1, lg " + std::to_string(lg));
        // Linear: the middle of two points is the mean of their values.
        if (half >= 64) {
            const std::size_t a = ntt_point(lg, 2), b = ntt_point(lg, 3);
            if (!near(ntt_ns((a + b) / 2, top, low), (low[kRaggedPoints * (lg - 6) + 2] + low[kRaggedPoints * (lg - 6) + 3]) / 2))
                fail("ntt_ns between points, lg " + std::to_string(lg));
            const std::size_t c = ntt_point(lg, 3);
            if (!near(ntt_ns((c + 2 * half) / 2, top, low), (low[kRaggedPoints * (lg - 6) + 3] + top[lg - 6]) / 2))
                fail("ntt_ns between the last point and 2^lg, lg " + std::to_string(lg));
        }
    }
    // Every N up to 2^14 and around each power of two: increasing (these tables grow), and in
    // [point 0, 2^lg's] of its class.
    double last = 0;
    for (std::size_t N = 1; N <= (std::size_t(1) << 14); ++N) {
        const double t = ntt_ns(N, top, low);
        if (t < last) fail("ntt_ns decreases at " + std::to_string(N));
        last = t;
    }
    for (int lg = 7; lg <= 25; ++lg)
        for (std::size_t N = (std::size_t(1) << (lg - 1)) - 3; N <= (std::size_t(1) << (lg - 1)) + 3; ++N) {
            const int c = std::max(6, int(std::bit_width(N - 1)));
            const double t = ntt_ns(N, top, low);
            if (t < low[kRaggedPoints * (c - 6)] || t > top[c - 6]) fail("ntt_ns outside its class at " + std::to_string(N));
        }
    // Up to 2^5: point 0 of 2^6. Past 2^25: twice per doubling.
    if (!near(ntt_ns(1, top, low), low[0]) || !near(ntt_ns(32, top, low), low[0])) fail("ntt_ns up to 2^5");
    if (!near(ntt_ns((std::size_t(1) << 25) + 1, top, low), 2 * top[19]) ||
        !near(ntt_ns(std::size_t(1) << 27, top, low), 4 * top[19]))
        fail("ntt_ns past 2^25");
    // A staircase (every point of lg at 2^lg's value, main's model) is flat in each class.
    Low flat;
    for (int i = 0; i < 20 * kRaggedPoints; ++i) flat[i] = top[i / kRaggedPoints];
    for (int lg = 6; lg <= 25; ++lg)
        for (const std::size_t N : {ntt_point(lg, 0), ntt_point(lg, 2) + 7, std::size_t(1) << lg})
            if (!near(ntt_ns(N, top, flat), top[lg - 6])) fail("ntt_ns on a staircase at " + std::to_string(N));
}

// beats_ntt is model_ns < ntt_ns, on the compiled defaults and on set tables.
void decisions() {
    const auto check = [](const char* what) {
        for (std::size_t n : {1, 2, 7, 32, 33, 64, 65, 100, 255, 256, 257, 1000, 1024, 1025, 4096, 4097, 100000})
            for (std::size_t m : {1, 2, 5, 16, 31, 32, 33, 64, 100, 128, 129, 1024})
                if (beats_ntt(n, m) != (model_ns(kModel, n, m) < ntt_ns(n + m - 1)) || beats_ntt(n, m) != beats_ntt(m, n))
                    fail(std::string("beats_ntt against ntt_ns, ") + what + ", " + std::to_string(n) + "x" +
                         std::to_string(m));
    };
    check("compiled");
    Top top;
    Low low;
    tables(top, low);
    ntt_times.set(top);
    ragged_times.set(low);
    check("set");
    if (ntt_ns(1000) != ntt_ns(1000, top, low)) fail("ntt_ns does not read the set tables");
    ntt_times.set(kNttDefaults);
    ragged_times.set(kRaggedDefaults);
    if (!beats_ntt(0, 5) || !beats_ntt(8, 8) || beats_ntt(1 << 20, 1 << 20) || !beats_ntt(std::size_t(1) << 27, 1))
        fail("beats_ntt on the compiled defaults");
    for (const double t : kRaggedDefaults)
        if (!(t > 0 && std::isfinite(t))) fail("kRaggedDefaults: a time of " + std::to_string(t));
}

// measure_ragged at a small budget: times positive; past max_lg the start times the point's ratio
// at max_lg; a margin above every change keeps the start. Then "small.ragged" through wisdom.
void wisdom() {
    const Low times = measure_ragged(kModel, kRaggedDefaults, {8, 1, 1e4, 0});
    for (const double t : times)
        if (!(t > 0 && std::isfinite(t))) fail("measure_ragged: a time of " + std::to_string(t));
    for (int j = 0; j < kRaggedPoints; ++j) {
        const auto at = [j](int lg) { return kRaggedPoints * (lg - 6) + j; };
        const double ratio = times[at(8)] / kRaggedDefaults[at(8)];
        for (int lg = 9; lg <= 25; ++lg)
            if (!near(times[at(lg)], kRaggedDefaults[at(lg)] * ratio))
                fail("measure_ragged: point " + std::to_string(j) + " of lg " + std::to_string(lg) +
                     " is not the default times the ratio at max_lg");
    }
    if (measure_ragged(kModel, kRaggedDefaults, {7, 1, 1e4, 1e9}) != kRaggedDefaults)
        fail("measure_ragged: changed within the margin");

    const std::string path =
        (std::filesystem::temp_directory_path() / ("fps-small-ntt-cost-" + std::to_string(::getpid()))).string();
    ragged_times.set(times);
    if (!fps::tune::save(path)) fail("save");
    ragged_times.set(kRaggedDefaults);
    if (!fps::tune::load(path) || ragged_times.get() != times) fail("small.ragged did not load back from wisdom");
    std::ifstream in(path);
    bool found = false;
    for (std::string line; std::getline(in, line);) found |= line == "small.ragged " + fps::tune::text(times);
    if (!found) fail("no small.ragged line in the wisdom");
    std::filesystem::remove(path);
    ragged_times.set(kRaggedDefaults);
    if (parse_times<20 * kRaggedPoints>(fps::tune::text(times)) != times) fail("parse_times");
    Low bad = times;
    bad[3] = 0;
    if (parse_times<20 * kRaggedPoints>(fps::tune::text(bad))) fail("parse_times took a time of 0");
}

}  // namespace

int main() {
    points();
    interpolation();
    decisions();
    wisdom();
    std::printf("%s\n", failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
