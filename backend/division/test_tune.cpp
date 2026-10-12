// The cost model's measurement (tune.hpp) on a short budget: positive finite values, the
// registration's text and parse round trip, and divisions under the measured table still right.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "backend/division/tune.hpp"

int main() {
    using namespace fps::division;
    int failures = 0;
    const Costs c = measure_costs(kCostDefaults, {9, 3, 2e4, 0});
    for (std::size_t i = 0; i < c.size(); ++i)
        if (!(c[i] >= 0 && std::isfinite(c[i]))) std::printf("FAIL cost %zu = %g\n", i, c[i]), ++failures;
    costs.set(c);
    const auto parsed = fps::tune::parse<Costs>(fps::tune::text(costs.get()));
    if (!parsed || *parsed != c) std::printf("FAIL text and parse\n"), ++failures;
    // f = q g + r at a point, for a few shapes under the measured table.
    std::mt19937 rng(7);
    const auto eval = [](const std::vector<u32>& a, u32 x) {
        u32 y = 0;
        for (std::size_t i = a.size(); i-- > 0;) y = u32((u64(y) * x + a[i]) % kP);
        return y;
    };
    for (const auto& [n, m] : {std::pair<std::size_t, std::size_t>{5000, 2500}, {5000, 40}, {3000, 2900}, {70000, 1000}}) {
        std::vector<u32> f(n), g(m), q(quotient_size(n, m)), r(m - 1);
        for (u32& x : f) x = rng() % kP;
        for (u32& x : g) x = 1 + rng() % (kP - 1);
        divide(f, g, q, r);
        const u32 x = rng() % kP;
        if (eval(f, x) != (u64(eval(q, x)) * eval(g, x) + eval(r, x)) % kP)
            std::printf("FAIL n=%zu m=%zu\n", n, m), ++failures;
    }
    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
