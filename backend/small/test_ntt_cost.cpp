// Tests beats_ntt's NTT times at the lengths the NTT runs (issue #212): ntt_point, ntt_ns's
// interpolation (the values at the points, linear between them, the ends of each length class, past
// 2^25), beats_ntt against ntt_ns on the compiled defaults and with "small.ntt" set (the points
// between powers of two follow it), and measure_ragged at a small budget. Exits nonzero on a
// failure.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>

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

// Tables whose values are distinct and grow with N: 2^lg's time is 2^lg, point j's (4 + j) / 8 of it.
void tables(Top& top, Low& ratio) {
    for (int lg = 6; lg <= 25; ++lg) {
        for (int j = 0; j < kRaggedPoints; ++j) ratio[kRaggedPoints * (lg - 6) + j] = (4 + j) / 8.0;
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
    Low ratio;
    tables(top, ratio);
    for (int lg = 6; lg <= 25; ++lg) {
        const std::size_t half = std::size_t(1) << (lg - 1);
        const double* const p = ratio.data() + kRaggedPoints * (lg - 6);
        const double t = top[lg - 6];
        if (!near(ntt_ns(2 * half, top, ratio), t)) fail("ntt_ns at 2^lg, lg " + std::to_string(lg));
        for (int j = 1; j < kRaggedPoints; ++j)
            if (!near(ntt_ns(ntt_point(lg, j), top, ratio), t * p[j]))
                fail("ntt_ns at point " + std::to_string(j) + ", lg " + std::to_string(lg));
        // 2^(lg-1) + 1 counts as 2^(lg-1): point 0 plus 1 / (half / 4) of the step to point 1.
        if (!near(ntt_ns(half + 1, top, ratio), t * (p[0] + (p[1] - p[0]) * 4 / double(half))))
            fail("ntt_ns at 2^(lg-1) + 1, lg " + std::to_string(lg));
        // Linear: the middle of two points is the mean of their values.
        if (half >= 64) {
            const std::size_t a = ntt_point(lg, 2), b = ntt_point(lg, 3);
            if (!near(ntt_ns((a + b) / 2, top, ratio), t * (p[2] + p[3]) / 2))
                fail("ntt_ns between points, lg " + std::to_string(lg));
            if (!near(ntt_ns((b + 2 * half) / 2, top, ratio), t * (p[3] + 1) / 2))
                fail("ntt_ns between the last point and 2^lg, lg " + std::to_string(lg));
        }
    }
    // Every N up to 2^14: non-decreasing (these tables grow); around each power of two, within
    // [point 0, 2^lg's] of its class.
    double last = 0;
    for (std::size_t N = 1; N <= (std::size_t(1) << 14); ++N) {
        const double t = ntt_ns(N, top, ratio);
        if (t < last) fail("ntt_ns decreases at " + std::to_string(N));
        last = t;
    }
    for (int lg = 7; lg <= 25; ++lg)
        for (std::size_t N = (std::size_t(1) << (lg - 1)) - 3; N <= (std::size_t(1) << (lg - 1)) + 3; ++N) {
            const int c = std::max(6, int(std::bit_width(N - 1)));
            const double t = ntt_ns(N, top, ratio);
            if (t < top[c - 6] * ratio[kRaggedPoints * (c - 6)] || t > top[c - 6])
                fail("ntt_ns outside its class at " + std::to_string(N));
        }
    // Up to 2^5: point 0 of 2^6. Past 2^25: twice per doubling.
    if (!near(ntt_ns(1, top, ratio), top[0] * ratio[0]) || !near(ntt_ns(32, top, ratio), top[0] * ratio[0]))
        fail("ntt_ns up to 2^5");
    if (!near(ntt_ns((std::size_t(1) << 25) + 1, top, ratio), 2 * top[19]) ||
        !near(ntt_ns(std::size_t(1) << 27, top, ratio), 4 * top[19]))
        fail("ntt_ns past 2^25");
    // Ratios of 1 (main's staircase): flat in each class.
    Low flat;
    flat.fill(1);
    for (int lg = 6; lg <= 25; ++lg)
        for (const std::size_t N : {ntt_point(lg, 0), ntt_point(lg, 2) + 7, std::size_t(1) << lg})
            if (!near(ntt_ns(N, top, flat), top[lg - 6])) fail("ntt_ns on a staircase at " + std::to_string(N));
}

// beats_ntt is model_ns < ntt_ns, on the compiled defaults and with "small.ntt" set; the points
// between powers of two scale with it.
void decisions() {
    const auto check = [](const char* what) {
        for (std::size_t n : {1, 2, 7, 32, 33, 64, 65, 100, 255, 256, 257, 1000, 1024, 1025, 4096, 4097, 100000})
            for (std::size_t m : {1, 2, 5, 16, 31, 32, 33, 64, 100, 128, 129, 1024})
                if (beats_ntt(n, m) != (model_ns(kModel, n, m) < ntt_ns(n + m - 1)) || beats_ntt(n, m) != beats_ntt(m, n))
                    fail(std::string("beats_ntt against ntt_ns, ") + what + ", " + std::to_string(n) + "x" +
                         std::to_string(m));
    };
    check("compiled");
    for (const double r : kRagged)
        if (!(r > 0 && std::isfinite(r))) fail("kRagged: a ratio of " + std::to_string(r));
    Top doubled = kNttDefaults;
    for (double& t : doubled) t *= 2;
    ntt_times.set(doubled);
    check("set");
    for (const std::size_t N : {1, 40, 100, 129, 1000, 1025, 3000, 100000, 1 << 25})
        if (!near(ntt_ns(N), 2 * ntt_ns(N, kNttDefaults))) fail("ntt_ns does not follow small.ntt at " + std::to_string(N));
    ntt_times.set(kNttDefaults);
    if (!beats_ntt(0, 5) || !beats_ntt(8, 8) || beats_ntt(1 << 20, 1 << 20) || !beats_ntt(std::size_t(1) << 27, 1))
        fail("beats_ntt on the compiled defaults");
}

// measure_ragged at a small budget: times positive; past max_lg the start times the point's ratio
// at max_lg; a margin above every change keeps the start.
void measured() {
    Low start;
    for (int e = 0; e < 20 * kRaggedPoints; ++e) start[e] = kRagged[e] * kNttDefaults[e / kRaggedPoints];
    const Low times = measure_ragged(kModel, start, {8, 1, 1e4, 0});
    for (const double t : times)
        if (!(t > 0 && std::isfinite(t))) fail("measure_ragged: a time of " + std::to_string(t));
    for (int j = 0; j < kRaggedPoints; ++j) {
        const auto at = [j](int lg) { return kRaggedPoints * (lg - 6) + j; };
        const double ratio = times[at(8)] / start[at(8)];
        for (int lg = 9; lg <= 25; ++lg)
            if (!near(times[at(lg)], start[at(lg)] * ratio))
                fail("measure_ragged: point " + std::to_string(j) + " of lg " + std::to_string(lg) +
                     " is not the start times the ratio at max_lg");
    }
    if (measure_ragged(kModel, start, {7, 1, 1e4, 1e9}) != start) fail("measure_ragged: changed within the margin");
}

}  // namespace

int main() {
    points();
    interpolation();
    decisions();
    measured();
    std::printf("%s\n", failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
