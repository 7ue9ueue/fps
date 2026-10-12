// Benchmarks of backend/evaluation, n = m = 2^lg. Items: points.
//   evaluate/<lg>          f of 2^lg coefficients at 2^lg points (the selected path)
//   interpolate/<lg>       the f through 2^lg distinct points
//   newton/<lg>            to_newton_basis of f of 2^lg coefficients
//   <op>@tree/<lg>         the tree path alone (evaluate, interpolate), at small sizes
//   evaluate@horner/<lg>   Horner's rule alone, at small sizes
#include <algorithm>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "backend/evaluation/evaluation.hpp"

namespace {

using u32 = std::uint32_t;
constexpr u32 P = 998244353;

std::vector<u32> random_words(std::size_t n, unsigned seed) {
    std::mt19937 rng{seed};
    std::vector<u32> v(n);
    for (u32& x : v) x = u32(rng() % P);
    return v;
}

std::vector<u32> distinct(std::size_t n, unsigned seed) {
    std::vector<u32> v = random_words(n + n / 4 + 16, seed);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    v.resize(n);
    std::shuffle(v.begin(), v.end(), std::mt19937{seed});
    return v;
}

struct Data {
    std::vector<u32> f, points, out;
    explicit Data(std::size_t n) : f(random_words(n, unsigned(n))), points(distinct(n, unsigned(n) + 1)), out(n) {}
};

using Op = void (*)(std::span<const u32>, std::span<const u32>, std::span<u32>);

void add(bench::Suite& suite, const std::string& name, int lg, Op op) {
    const std::size_t n = std::size_t(1) << lg;
    suite.add(name + "/" + std::to_string(lg), [n, op] {
        return [d = Data(n), op]() mutable {
            op(d.f, d.points, d.out);
            bench::keep(d.out);
        };
    }, n);
}

void evaluate(std::span<const u32> f, std::span<const u32> points, std::span<u32> out) { fps::evaluation::evaluate(f, points, out); }
void evaluate_tree(std::span<const u32> f, std::span<const u32> points, std::span<u32> out) {
    fps::evaluation::detail::evaluate_tree(f, points, out);
}
void evaluate_horner(std::span<const u32> f, std::span<const u32> points, std::span<u32> out) {
    fps::evaluation::detail::horner<fps::evaluation::detail::Wide>(f.data(), f.size(), points.data(), points.size(), out.data());
}
// The f through (points[i], f[i]).
void interpolate(std::span<const u32> f, std::span<const u32> points, std::span<u32> out) { fps::evaluation::interpolate(points, f, out); }
void interpolate_tree(std::span<const u32> f, std::span<const u32> points, std::span<u32> out) {
    fps::evaluation::detail::interpolate_tree(points, f, out);
}
void newton(std::span<const u32> f, std::span<const u32> points, std::span<u32> out) { fps::evaluation::to_newton_basis(f, points, out); }

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    for (int lg : {6, 8, 10, 12, 14, 16, 17, 18, 20}) {
        add(suite, "evaluate", lg, evaluate);
        add(suite, "interpolate", lg, interpolate);
        add(suite, "newton", lg, newton);
    }
    for (int lg : {6, 7, 8, 9, 10, 11, 12, 13, 14}) {
        add(suite, "evaluate@tree", lg, evaluate_tree);
        add(suite, "evaluate@horner", lg, evaluate_horner);
        if (lg <= 8) add(suite, "interpolate@tree", lg, interpolate_tree);
    }
    return suite.run();
}
