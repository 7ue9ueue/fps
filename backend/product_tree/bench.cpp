// Benchmarks of product_tree::product. Items: factors.
//   product/<lg>            2^lg factors x - r (the frontend's product/<lg>), lg 4 .. 20
//   product@standard/<lg>   the same with the standard tree alone (no lanes), lg <= 18
//   product@lanes/<lg>      the same with every factor in lanes
//   product_mixed/<lg>      2^lg factors of degrees 0 .. 8 (a quarter constants), sum about 3.4 2^lg
//   product_few/<lg>        8 factors of degree 2^lg / 8 - 1 and 2^lg / 8: the standard tree's top
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "backend/bench.hpp"
#include "backend/product_tree/product_tree.hpp"

namespace {

using u32 = std::uint32_t;
using Mode = fps::product_tree::detail::Mode;
constexpr u32 P = 998244353;

struct Case {
    std::vector<std::vector<u32>> fs;
    std::vector<std::span<const u32>> spans;
    std::vector<u32> out;

    explicit Case(std::vector<std::vector<u32>> factors) : fs(std::move(factors)), spans(fs.begin(), fs.end()) {
        out.resize(fps::product_tree::size(spans));
    }
};

std::vector<std::vector<u32>> linear(std::size_t count, unsigned seed) {
    std::mt19937 rng{seed};
    std::vector<std::vector<u32>> fs;
    for (std::size_t i = 0; i < count; ++i) fs.push_back({u32(rng() % P), 1});
    return fs;
}

std::vector<std::vector<u32>> mixed(std::size_t count, unsigned seed) {
    std::mt19937 rng{seed};
    std::vector<std::vector<u32>> fs;
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t d = rng() % 4 == 0 ? 0 : 1 + rng() % 8;
        std::vector<u32> f(d + 1);
        for (u32& x : f) x = u32(rng() % (P - 1)) + 1;
        fs.push_back(f);
    }
    return fs;
}

std::vector<std::vector<u32>> few(int lg, unsigned seed) {
    std::mt19937 rng{seed};
    std::vector<std::vector<u32>> fs;
    for (int i = 0; i < 8; ++i) {
        std::vector<u32> f((std::size_t(1) << lg) / 8 + (i % 2));
        for (u32& x : f) x = u32(rng() % P);
        fs.push_back(f);
    }
    return fs;
}

template <class Make>
void add(bench::Suite& suite, const std::string& name, Make make, Mode mode, std::size_t items) {
    suite.add(name, [make, mode] {
        return [c = Case(make()), mode]() mutable {
            fps::product_tree::detail::product(c.spans, c.out, mode);
            bench::keep(c.out);
        };
    }, items);
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    for (int lg : {4, 6, 8, 10, 14, 18, 20}) {
        const std::size_t count = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        add(suite, "product" + size, [=] { return linear(count, unsigned(lg)); }, Mode::automatic, count);
        if (lg <= 18) {
            add(suite, "product@standard" + size, [=] { return linear(count, unsigned(lg)); }, Mode::standard, count);
            add(suite, "product@lanes" + size, [=] { return linear(count, unsigned(lg)); }, Mode::lanes, count);
        }
        if (lg >= 10) add(suite, "product_mixed" + size, [=] { return mixed(count, unsigned(lg)); }, Mode::automatic, count);
    }
    for (int lg : {12, 16, 20}) add(suite, "product_few/" + std::to_string(lg), [=] { return few(lg, unsigned(lg)); }, Mode::automatic, 8);
    return suite.run();
}
