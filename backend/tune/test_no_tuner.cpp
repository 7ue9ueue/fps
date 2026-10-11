// Tests backend/tune with -DFPS_NO_TUNER: a table is its compiled defaults, as constants, and has
// no set(). Exits nonzero on a failure.
#define FPS_NO_TUNER
#include <array>
#include <cstdio>

#include "backend/tune/tune.hpp"

namespace {

inline constexpr std::array<int, 3> kDefaults = {3, 1, 4};
inline constinit fps::tune::Table<kDefaults> table;

static_assert(table.get()[2] == 4, "the defaults are constants");
template <class T>
concept Settable = requires(T& t) { t.set(kDefaults); };
static_assert(!Settable<decltype(table)>, "no set() without the tuner");

}  // namespace

int main() {
    const bool ok = table.get() == kDefaults && fps::tune::parse<std::array<int, 3>>("3 1 4") == kDefaults;
    std::printf("%s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
