// Tests the small kernel with -DFPS_NO_TUNER: beats_ntt and multiply() read their compiled
// defaults, as constants. Exits nonzero on a failure.
#define FPS_NO_TUNER
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "backend/small/model.hpp"
#include "backend/small/small.hpp"
#include "backend/small/tune.hpp"

static_assert(fps::small::ntt_times.get()[19] == fps::small::kNttDefaults[19]);
static_assert(fps::small::ragged_times.get()[79] == fps::small::kRaggedDefaults[79]);
static_assert(fps::small::variants.get()[fps::small::shape_class(64, 64)] ==
              fps::small::kDefaults[fps::small::shape_class(64, 64)]);

int main() {
    using fps::small::beats_ntt;
    constexpr std::uint32_t P = 998244353;
    const std::size_t big = std::size_t(1) << 20;
    bool ok = beats_ntt(8, 8) && beats_ntt(big, 2) && !beats_ntt(1024, 1024) && !beats_ntt(big, big);
    for (std::size_t n : {1, 7, 64, 300})
        for (std::size_t m : {1, 5, 64}) {
            std::vector<std::uint32_t> a(n), b(m), out(n + m - 1), want(n + m - 1);
            for (std::size_t i = 0; i < n; ++i) a[i] = std::uint32_t((i * 2654435761u) % P);
            for (std::size_t j = 0; j < m; ++j) b[j] = std::uint32_t((j * 40503u + 7) % P);
            fps::small::multiply<P>(a, b, out);
            fps::small::model::multiply<P>(a.data(), n, b.data(), m, want.data());
            ok &= out == want;
        }
    std::printf("%s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
