// Tests the factorials kernel with -DFPS_NO_TUNER: the variant table is the compiled defaults, as
// constants, and tables() reads it. Exits nonzero on a failure.
#define FPS_NO_TUNER
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

#include "backend/factorials/factorials.hpp"
#include "backend/factorials/tune.hpp"

namespace fx = fps::factorials;

static_assert(fx::variants.get() == fx::kDefaults);
static_assert(fx::variants.get()[fx::size_class(1 << 20)] == fx::kDefaults[fx::size_class(1 << 20)]);

int main() {
    bool ok = true;
    for (const std::size_t n : {0, 5, 64, 1000, 1 << 16}) {
        std::vector<fx::u32> fact(n), inverse(n), want_fact(n), want_inverse(n);
        fx::tables<998244353>(std::span(fact), std::span(inverse));
        fx::model::tables<998244353>(want_fact.data(), want_inverse.data(), n);
        ok &= fact == want_fact && inverse == want_inverse;
    }
    std::printf("%s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
