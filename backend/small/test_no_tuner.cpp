// Tests the small kernel with -DFPS_NO_TUNER: beats_ntt reads its compiled defaults, as constants.
// Exits nonzero on a failure.
#define FPS_NO_TUNER
#include <cstddef>
#include <cstdio>

#include "backend/small/small.hpp"
#include "backend/small/tune.hpp"

static_assert(fps::small::ntt_times.get()[19] == fps::small::kNttDefaults[19]);

int main() {
    using fps::small::beats_ntt;
    const std::size_t big = std::size_t(1) << 20;
    const bool ok = beats_ntt(8, 8) && beats_ntt(big, 2) && !beats_ntt(1024, 1024) && !beats_ntt(big, big);
    std::printf("%s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
