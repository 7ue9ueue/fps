// Tests backend/ntt with -DFPS_NO_TUNER: the variant table is its compiled defaults, as constants,
// and products and transforms run through it. Exits nonzero on a failure.
#define FPS_NO_TUNER
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "backend/ntt/ntt.hpp"
#include "backend/ntt/tune.hpp"

static_assert(fps::ntt::variants.get() == fps::ntt::kDefaults);
static_assert(fps::ntt::choice(fps::ntt::kProduct, 25) == fps::ntt::kDefaults[fps::ntt::kProduct][19]);

int main() {
    using u32 = std::uint32_t;
    constexpr u32 P = fps::ntt::kModulus;
    bool ok = true;
    for (const std::size_t n : {1, 33, 100, 700})
        for (const std::size_t m : {1, 64, 300}) {
            std::vector<u32> f(n), g(m), out(n + m - 1), want(n + m - 1);
            for (std::size_t i = 0; i < n; ++i) f[i] = u32((i * 2654435761u + 3) % P);
            for (std::size_t j = 0; j < m; ++j) g[j] = u32((j * 40503u + 7) % P);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < m; ++j) want[i + j] = u32((want[i + j] + std::uint64_t(f[i]) * g[j]) % P);
            fps::ntt::multiply(f, g, out);
            ok &= out == want;
        }
    struct alignas(32) Lanes {
        u32 lane[8];
    };
    for (int lg = fps::ntt::kMinLog; lg <= 12; ++lg) {
        const fps::ntt::Transform t(lg);
        std::vector<Lanes> lanes((t.length() + fps::ntt::kPadding) / 8);  // 32-byte aligned (ntt.hpp)
        u32* const f = lanes.data()->lane;
        for (std::size_t i = 0; i < t.length(); ++i) f[i] = u32((i * 977 + lg) % P);
        const std::vector<u32> before(f, f + t.length());
        t.forward(f);
        t.inverse(f);
        ok &= std::vector<u32>(f, f + t.length()) == before;
    }
    std::printf("%s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
