// Benchmarks of backend/taylor_shift. Items: coefficients (n).
//   shift/<lg>               f of n = 2^lg coefficients, all n of f(x + c) (the selected path)
//   shift@transform/<lg>     its product alone: ntt's in-place product of length 2n of two factors
//                            of n words (the floor of the transform's part)
//   shift@plain/<lg>         the frontend's algorithm before this kernel: factorials::tables,
//                            scalar weights, ntt::middle_product, a scalar output loop
#include <bit>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "backend/factorials/factorials.hpp"
#include "backend/ntt/ntt.hpp"
#include "backend/taylor_shift/taylor_shift.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using fps::taylor_shift::kP;

std::vector<u32> random_words(std::size_t n, unsigned seed) {
    std::mt19937 rng{seed};
    std::vector<u32> f(n);
    for (u32& x : f) x = u32(rng() % kP);
    return f;
}

constexpr u32 kC = 123456789;

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % kP); }

// The frontend's taylor_shift before this kernel (include/fps/composition.hpp at 3fd62d5).
void plain(std::span<const u32> f, u32 a, std::span<u32> out) {
    const std::size_t m = f.size(), t = out.size();
    std::vector<u32> fact(m), inverse(m), weighted(m + t - 1, 0), e(m), h(t);
    fps::factorials::tables<kP>(fact, inverse);
    for (std::size_t i = 0; i < m; ++i) weighted[i] = mul(f[i], fact[i]);
    u32 power = 1;
    for (std::size_t j = 0; j < m; ++j, power = mul(power, a)) e[m - 1 - j] = mul(power, inverse[j]);
    fps::ntt::middle_product(weighted, e, h);
    for (std::size_t k = 0; k < t; ++k) out[k] = mul(h[k], inverse[k]);
}

// The shift's product alone: factors of n = L/2 words in place (a, b as shift_lanes lays them out),
// each call on the previous call's output (any words < P serve).
struct Transforms {
    using Product = fps::ntt::SelectedMultiply;
    std::size_t n;
    int lg;
    std::vector<u32> storage;
    u32 *a, *b;
    explicit Transforms(std::size_t n_) : n(n_), lg(std::bit_width(2 * n_ - 1)), storage(4 * n_ + 2 * fps::ntt::kPadding + 16) {
        a = storage.data() + (16 - reinterpret_cast<std::uintptr_t>(storage.data()) / 4 % 16) % 16;
        b = a + 2 * n + fps::ntt::kPadding;
        const std::vector<u32> x = random_words(n, 1), y = random_words(n, 2);
        std::copy(x.begin(), x.end(), a), std::copy(y.begin(), y.end(), b);
    }
    void run() {
        const auto [roots, inverse_roots] = fps::ntt::detail::tables<Product>(lg);
        Product::multiply(a, b, n, n, lg, roots, inverse_roots);
    }
};

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    for (int lg = 6; lg <= 22; ++lg) {
        const std::size_t n = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        suite.add("shift" + size, [n, lg] {
            return [f = random_words(n, unsigned(lg)), out = std::vector<u32>(n)]() mutable {
                fps::taylor_shift::shift(f, kC, out);
                bench::keep(out);
            };
        }, n);
        if (lg % 2 == 0 && lg >= 8) {
            suite.add("shift@transform" + size, [n] {
                return [x = std::make_shared<Transforms>(n)]() mutable {
                    x->run();
                    bench::keep(x->a);
                };
            }, n);
            suite.add("shift@plain" + size, [n, lg] {
                return [f = random_words(n, unsigned(lg)), out = std::vector<u32>(n)]() mutable {
                    plain(f, kC, out);
                    bench::keep(out);
                };
            }, n);
        }
    }
    return suite.run();
}
