// Benches for backend/ntt, per transform length 2^lg; items per call: 2^lg coefficients.
//   forward/<lg>, inverse/<lg>   Transform::forward and inverse, in place, tables built.
//   multiply/<lg>                Convolution::multiply, factors of 2^lg / 2 coefficients each.
//   multiply_dense/<lg>          factors of 3 2^lg / 4 and 2^lg / 4 + 1 coefficients.
//   <op>@<variant>/<lg>          one variant: scalar, intrinsics_<simd struct>, avx2 (if not selected).
// Each multiply call builds its tables, as a single-use Convolution does. Repeated calls reuse
// the last output as input: the timing does not depend on the values.
#include "backend/ntt/ntt.hpp"

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "backend/bench.hpp"

namespace {

using u32 = std::uint32_t;
using namespace fps::ntt;

// The scalar variant takes seconds at 2^25; its benches stop here.
constexpr int kScalarMaxLog = 18;
// Variants not selected: up to 2^22 (64 MiB for a multiply), to keep the gate's 21 rounds short.
constexpr int kVariantMaxLog = 22;

struct Words {
    struct alignas(64) Line {
        u32 words[16];
    };
    explicit Words(std::size_t count) : lines((count + kPadding + 15) / 16) {
        std::mt19937 rng(static_cast<unsigned>(count));
        for (auto& line : lines)
            for (u32& w : line.words) w = rng() % kModulus;
    }
    u32* begin() { return lines.data()->words; }
    std::vector<Line> lines;
};

template <class V>
constexpr bool selected_everywhere =
    std::is_same_v<V, SelectedForward> && std::is_same_v<V, SelectedInverse> && std::is_same_v<V, SelectedMultiply>;

template <class Forward, class Inverse, class Multiply>
void add_cases(bench::Suite& suite, const std::string& suffix, int max_lg) {
    for (int lg = kMinLog; lg <= max_lg; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        suite.add("forward" + suffix + size, [lg] {
            return [t = std::make_shared<Transform>(lg), f = std::make_shared<Words>(std::size_t(1) << lg)] {
                t->template forward<Forward>(f->begin());
                bench::keep(*f->begin());
            };
        }, len);
        suite.add("inverse" + suffix + size, [lg] {
            return [t = std::make_shared<Transform>(lg), f = std::make_shared<Words>(std::size_t(1) << lg)] {
                t->template inverse<Inverse>(f->begin());
                bench::keep(*f->begin());
            };
        }, len);
        for (const auto& [name, n, m] : {std::tuple{"multiply", len / 2, len / 2},
                                         std::tuple{"multiply_dense", len - len / 4, len / 4 + 1}}) {
            suite.add(name + suffix + size, [n, m] {
                auto conv = std::make_shared<Convolution>(n, m);
                const Words a(n), b(m);
                std::copy(a.lines.data()->words, a.lines.data()->words + n, conv->a());
                std::copy(b.lines.data()->words, b.lines.data()->words + m, conv->b());
                return [conv] { bench::keep(*conv->template multiply<Multiply>()); };
            }, len);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    const bool native_scalar = std::is_same_v<Native, variant::scalar>;
    add_cases<SelectedForward, SelectedInverse, SelectedMultiply>(suite, "", native_scalar ? kScalarMaxLog : kMaxLog);
    if (!native_scalar) add_cases<variant::scalar, variant::scalar, variant::scalar>(suite, "@scalar", 12);
    // intrinsics<S> for the vector structs worth timing here (8 lanes and more, or the native one;
    // not Avx2 where Avx512Vl runs the same code on 32 registers), and avx2, unless selected for
    // every op.
    fps::simd::for_each(fps::simd::All{}, [&]<class S> {
        using V = variant::intrinsics<S>;
        constexpr bool worth = S::lanes > 1 && (S::lanes >= 8 || std::is_same_v<S, fps::simd::Native>) &&
                               !(S::lanes == 8 && S::registers < fps::simd::Native::registers);
        if constexpr (worth && !selected_everywhere<V>)
            add_cases<V, V, V>(suite, std::string("@intrinsics_") + S::name, kVariantMaxLog);
    });
#ifdef __AVX2__
    if constexpr (!selected_everywhere<variant::avx2>)
        add_cases<variant::avx2, variant::avx2, variant::avx2>(suite, "@avx2", kVariantMaxLog);
#endif
    return suite.run();
}
