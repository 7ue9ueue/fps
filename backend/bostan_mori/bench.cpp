// Times backend/bostan_mori; tools/kernel.py runs it. Coefficient k = 10^18 (60 steps) of p / q,
// p of d and q of d + 1 random words (d = 2: Fibonacci, 1 / (1 - x - x^2) times x), as the
// frontend's fraction_at/<d>. Items: d.
//   "coefficient/<d>"              the selected path (gated)
//   "coefficient@<variant>/<d>"    one variant: schoolbook, values (no schoolbook), products (the
//                                  frontend's loop of two ntt::multiply per step)
//   "multiply@ntt/<lg>"            the unit: ntt::multiply of two factors of 2^(lg-1) into 2^lg - 1
//                                  words (the frontend's multiply@kernel/<lg>)
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "backend/bostan_mori/bostan_mori.hpp"
#include "backend/ntt/ntt.hpp"

namespace {

constexpr std::uint32_t P = 998244353;
constexpr std::uint64_t kIndex = 1'000'000'000'000'000'000ULL;
using Vec = std::vector<std::uint32_t>;
namespace bm = fps::bostan_mori;

Vec random_vec(std::size_t n, std::uint32_t first, unsigned seed) {
    std::mt19937 rng(seed);
    Vec a(n);
    for (auto& x : a) x = rng() % P;
    if (n) a[0] = first;
    return a;
}

struct Fraction {
    Vec p, q;
};
Fraction fraction(std::size_t d) {
    if (d == 2) return {{0, 1}, {1, P - 1, P - 1}};
    return {random_vec(d, 3, unsigned(d)), random_vec(d + 1, 1, unsigned(d) + 1)};
}

template <class Op>
void add(bench::Suite& suite, const std::string& name, std::size_t d, Op op) {
    suite.add(name + "/" + std::to_string(d), [d, op] {
        return [f = fraction(d), op] { bench::keep(op(f.p, f.q, kIndex)); };
    }, d);
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    const auto selected = [](const Vec& p, const Vec& q, std::uint64_t k) { return bm::coefficient(p, q, k); };
    const auto schoolbook = [](const Vec& p, const Vec& q, std::uint64_t k) { return bm::variant::schoolbook(p, q, k); };
    const auto values = [](const Vec& p, const Vec& q, std::uint64_t k) { return bm::variant::values(p, q, k, 0); };
    const auto products = [](const Vec& p, const Vec& q, std::uint64_t k) { return bm::variant::products(p, q, k); };
    for (std::size_t d : {2, 16, 64, 1024, 16384, 100000}) add(suite, "coefficient", d, selected);
    for (std::size_t d : {16, 32, 48, 64, 96, 128}) {
        add(suite, "coefficient@schoolbook", d, schoolbook);
        add(suite, "coefficient@values", d, values);
    }
    for (std::size_t d : {16, 1024, 100000}) add(suite, "coefficient@products", d, products);
    for (int lg : {11, 18}) {
        const std::size_t n = std::size_t(1) << (lg - 1);
        suite.add("multiply@ntt/" + std::to_string(lg), [n, lg] {
            return [f = random_vec(n, 1, unsigned(lg)), g = random_vec(n, 2, unsigned(lg + 1)), out = Vec(2 * n - 1)]() mutable {
                fps::ntt::multiply(f, g, out);
                bench::keep(out);
            };
        }, 2 * n);
    }
    return suite.run();
}
