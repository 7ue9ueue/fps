// tools/tune.py small: measures the variant per shape class and beats_ntt's costs on this machine
// and prints them as compiled defaults for tuned.hpp ("field <name> <type> <values>"), with
// diagnostics ("# ...").
//   0. measure_variants (tune.hpp) on every class, at the dense points (median of 7, samples >= 1
//      ms). The compiled table's variant stays unless another is more than 1% faster (a rerun does
//      not flip choices by noise). The new table is set before the next steps.
//   1. Sweep: lg = 6 .. 17; n + m - 1 = 2^(lg - 1) + 1, 3 2^(lg - 2) and 2^lg; m = 1 .. 1024,
//      m <= n. multiply() and the Convolution timed in turn (median of 5, samples >= 1 ms).
//   2. The model (select.hpp) fitted to the swept shapes within 0.4x-2.5x of the NTT, least
//      squares on relative error, coefficients >= 0.
//   3. measure_ntt (tune.hpp) with that model at every lg up to 25 (median of 7, >= 1 ms, no margin),
//      starting from the sweep's NTT times (past 17: the compiled defaults, scaled).
//   4. On the swept shapes, the shapes each table sends the slower way, new against compiled.
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "backend/small/tune.hpp"

namespace {

using fps::small::Model;

struct Shape {
    std::size_t n, m;
    int lg;
    double small, ntt;
};

int lg_of(std::size_t n, std::size_t m) { return std::max(6, int(std::bit_width(n + m - 2))); }

std::vector<Shape> sweep() {
    const fps::small::Budget budget = {17, 5, 1e6, 0};
    std::vector<Shape> shapes;
    for (int lg = 6; lg <= 17; ++lg)
        for (const std::size_t length : {(std::size_t(1) << (lg - 1)) + 1, std::size_t(3) << (lg - 2), std::size_t(1) << lg})
            for (const std::size_t m : {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024}) {
                const std::size_t n = length + 1 - m;
                if (m > n) break;
                const auto [small, ntt] = fps::small::detail::time_shape(n, m, budget);
                shapes.push_back({n, m, lg, small, ntt});
            }
    return shapes;
}

// Least squares on relative error over the shapes near the crossing; a negative coefficient is
// fixed at 0 and the rest fitted again.
Model fit(const std::vector<Shape>& shapes) {
    std::array<bool, 4> free = {true, true, true, true};
    std::array<double, 4> c{};
    for (int round = 0; round < 4; ++round) {
        std::array<std::array<double, 5>, 4> a{};  // normal equations, right-hand side in column 4
        for (const Shape& s : shapes) {
            if (s.small < 0.4 * s.ntt || s.small > 2.5 * s.ntt) continue;
            const double x = double(s.n), y = double(s.m), q = double(std::min(s.n, s.m));
            const std::array<double, 4> row = {x * y / s.small, (x + y) / s.small, 1 / s.small, q * q / s.small};
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) a[i][j] += row[i] * row[j];
                a[i][4] += row[i];
            }
        }
        for (int i = 0; i < 4; ++i)
            if (!free[i]) {
                a[i].fill(0);
                for (auto& r : a) r[i] = 0;
                a[i][i] = 1;
            }
        for (int i = 0; i < 4; ++i) {  // Gauss-Jordan, partial pivoting
            int pivot = i;
            for (int r = i + 1; r < 4; ++r)
                if (std::abs(a[r][i]) > std::abs(a[pivot][i])) pivot = r;
            std::swap(a[i], a[pivot]);
            for (int r = 0; r < 4; ++r)
                if (r != i) {
                    const double f = a[r][i] / a[i][i];
                    for (int j = i; j < 5; ++j) a[r][j] -= f * a[i][j];
                }
        }
        for (int i = 0; i < 4; ++i) c[i] = a[i][4] / a[i][i];
        const int worst = int(std::ranges::min_element(c) - c.begin());
        if (c[worst] >= 0) break;
        free[worst] = false;
    }
    return {c[0], c[1], c[2], c[3]};
}

bool picks_small(const Model& model, const std::array<double, 20>& ntt, std::size_t n, std::size_t m) {
    return fps::small::model_ns(model, n, m) < ntt[lg_of(n, m) - 6];
}

// Shapes the table sends the slower way, and the loss: chosen / faster time.
void misses(const char* name, const Model& model, const std::array<double, 20>& ntt, const std::vector<Shape>& shapes) {
    int count = 0;
    double worst = 1, log_sum = 0;
    for (const Shape& s : shapes) {
        const double chosen = picks_small(model, ntt, s.n, s.m) ? s.small : s.ntt;
        const double loss = chosen / std::min(s.small, s.ntt);
        log_sum += std::log(loss);
        if (loss > 1) {
            ++count;
            worst = std::max(worst, loss);
            std::printf("#   %s miss %zux%zu: small %.0f ntt %.0f\n", name, s.n, s.m, s.small, s.ntt);
        }
    }
    std::printf("# %s: %d of %zu shapes the slower way, worst %.3fx, geomean over all %.4f\n", name, count,
                shapes.size(), worst, std::exp(log_sum / double(shapes.size())));
}

// Step 0: the variant per class, with each class's scores (geometric mean of time / fastest) and
// times per point; then the points the compiled and the new table send to a variant more than 2%
// slower than the fastest there.
fps::small::Choices measure_table() {
    using namespace fps::small;
    const std::vector<std::string> names = variant_names();
    std::vector<ClassTimes> log;
    const Choices choices = measure_variants(kDefaults, {kOctaves - 1, true, 7, 1e6, 0.01}, &log);
    for (const ClassTimes& t : log) {
        std::printf("# class %d (octave %d, ratio %d): %s (compiled %s)", t.c, t.c / kRatios, t.c % kRatios,
                    names[choices[t.c]].c_str(), names[kDefaults[t.c]].c_str());
        for (int v = 0; v < kCandidates; ++v) {
            double score = 0;
            for (const std::vector<double>& ns : t.ns)
                score += std::log(ns[v] / *std::ranges::min_element(ns)) / double(t.ns.size());
            std::printf(" %s %.3f", names[v].c_str(), std::exp(score));
        }
        std::printf("\n");
        for (std::size_t p = 0; p < t.shapes.size(); ++p) {
            std::printf("#   %zux%zu:", t.shapes[p].first, t.shapes[p].second);
            for (const double ns : t.ns[p]) std::printf(" %.1f", ns);
            std::printf("\n");
        }
    }
    for (const auto& [name, table] : {std::pair{"compiled", kDefaults}, std::pair{"new", choices}}) {
        int count = 0, total = 0;
        double worst = 1, log_sum = 0;
        for (const ClassTimes& t : log)
            for (const std::vector<double>& ns : t.ns) {
                const double ratio = ns[table[t.c]] / *std::ranges::min_element(ns);
                ++total;
                log_sum += std::log(ratio);
                worst = std::max(worst, ratio);
                count += ratio > 1.02;
            }
        std::printf("# variants, %s: %d of %d points more than 2%% slower than the fastest, worst %.3fx, geomean %.4f\n",
                    name, count, total, worst, std::exp(log_sum / total));
    }
    return choices;
}

void print_field(const char* name, const double* values, int count) {
    std::printf("field %s double", name);
    for (int i = 0; i < count; ++i) std::printf(" %.4g", values[i]);
    std::printf("\n");
}

}  // namespace

int main() {
    std::printf("# cpu %s\n# target %s\n", fps::tune::cpu_name().c_str(), std::string(fps::tune::target_name()).c_str());
    const auto start_time = std::chrono::steady_clock::now();
    const fps::small::Choices choices = measure_table();
    fps::small::variants.set(choices);
    const std::vector<Shape> shapes = sweep();
    for (const Shape& s : shapes) std::printf("# shape %zux%zu lg %d small %.1f ntt %.1f\n", s.n, s.m, s.lg, s.small, s.ntt);

    const Model model = fit(shapes);
    std::vector<double> errors;
    for (const Shape& s : shapes)
        if (s.small >= 0.4 * s.ntt && s.small <= 2.5 * s.ntt)
            errors.push_back(std::abs(fps::small::model_ns(model, s.n, s.m) / s.small - 1));
    std::ranges::sort(errors);
    std::printf("# fit on %zu shapes: relative error median %.3f, 90%% %.3f, max %.3f\n", errors.size(),
                errors[errors.size() / 2], errors[errors.size() * 9 / 10], errors.back());

    std::array<double, 20> start = fps::small::kNttDefaults;
    for (int lg = 6; lg <= 17; ++lg) {
        std::vector<double> ntt;
        for (const Shape& s : shapes)
            if (s.lg == lg) ntt.push_back(s.ntt);
        std::ranges::sort(ntt);
        start[lg - 6] = ntt[ntt.size() / 2];
    }
    for (int lg = 18; lg <= 25; ++lg) start[lg - 6] = fps::small::kNttDefaults[lg - 6] * start[11] / fps::small::kNttDefaults[11];
    const std::array<double, 20> ntt = fps::small::measure_ntt(model, start, {25, 7, 1e6, 0});
    for (int lg = 6; lg <= 25; ++lg) {
        const auto [n, m] = fps::small::detail::probe(model, lg, ntt[lg - 6]);
        std::printf("# lg %d: ntt %.0f (sweep or start %.0f), crossing at %zux%zu\n", lg, ntt[lg - 6], start[lg - 6], n, m);
    }

    misses("compiled", fps::small::kModel, fps::small::kNttDefaults, shapes);
    misses("new", model, ntt, shapes);
    std::printf("# seconds %.1f\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count());

    std::printf("field signature int %u\nfield variant int", fps::small::kSignature);
    for (const std::uint8_t v : choices) std::printf(" %d", v);
    std::printf("\n");
    print_field("term", &model.term, 1);
    print_field("word", &model.word, 1);
    print_field("call", &model.call, 1);
    print_field("square", &model.square, 1);
    print_field("ntt", ntt.data(), 20);
    return 0;
}
