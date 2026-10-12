// Tests fps/tune.hpp with -DFPS_NO_TUNER: tune_all() does nothing, save_wisdom() writes nothing and
// returns false, load_wisdom() returns false for any file; f * g and Factorials
// still match O(n^2) references on the compiled defaults. Exits nonzero on a failure.
#define FPS_NO_TUNER
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include <unistd.h>

#include "fps/fps.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
constexpr u32 P = 998244353;
using Poly = fps::Polynomial<P>;

static_assert(fps::small::ntt_times.get() == fps::small::kNttDefaults, "the choices are constants");

// f * g against the O(n m) product, on both sides of beats_ntt's cutoffs and an NTT length.
bool products() {
    bool ok = true;
    for (std::size_t n : {0, 1, 7, 64, 300, 2048})
        for (std::size_t m : {0, 1, 5, 64, 2048}) {
            std::vector<u32> a(n), b(m), want(n && m ? n + m - 1 : 0);
            for (std::size_t i = 0; i < n; ++i) a[i] = u32(i * 2654435761u % P);
            for (std::size_t j = 0; j < m; ++j) b[j] = u32((j * 40503u + 7) % P);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < m; ++j) want[i + j] = u32((want[i + j] + u64(a[i]) * b[j]) % P);
            const Poly h = Poly(std::span<const u32>(a)) * Poly(std::span<const u32>(b));
            ok &= std::vector<u32>(h.span().begin(), h.span().end()) == want;
        }
    return ok;
}

bool factorials() {
    const std::size_t n = 70000;
    const fps::Factorials<P> fact(n);
    u64 f = 1;
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i) {
        if (i) f = f * i % P;
        ok &= fact[i] == f;
    }
    return ok;
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("fps-test-tune-no-tuner-" + std::to_string(getpid()));
    fs::create_directories(dir);
    const fs::path path = dir / "wisdom";

    fps::tune_all();
    bool ok = !fps::save_wisdom(path.string()) && !fs::exists(path);
    // A wisdom header for this target (no CPU name: the tuner that reads it is compiled out).
    std::ofstream(path) << "fps wisdom 1\ncpu unknown\ntarget " << fps::tune::target_name() << '\n';
    ok &= !fps::load_wisdom(path.string()) && !fps::load_wisdom((dir / "missing").string());
    ok &= products() && factorials();
    fs::remove_all(dir);
    std::printf("%s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
