// Benches for backend/ntt, per transform length 2^lg; items per call: 2^lg coefficients.
//   forward/<lg>, inverse/<lg>   Transform::forward and inverse, in place, tables built.
//   multiply/<lg>                Convolution::multiply, factors of 2^lg / 2 coefficients each.
//   multiply_dense/<lg>          factors of 3 2^lg / 4 and 2^lg / 4 + 1 coefficients.
//   <op>@<variant>/<lg>          one variant: scalar, intrinsics_<simd struct>, avx2 (unless the
//                                target's one candidate, select.hpp), avx512.
//   convolution/<lg>             a Convolution of 2^lg / 2 and 2^lg / 2 coefficients constructed,
//                                multiplied and destroyed, to 2^22: construction against multiply/<lg>.
//   convolution_into/<lg>        the same with a in the caller's memory, 64 bytes past a page boundary.
//   product/<lg>                 multiply(f, g, out), f and g of 2^lg / 2 words, out of 2^lg - 1, in
//                                caller memory 64 bytes past a page boundary, to 2^22.
//   product_cut/<lg>             the same, out of 2^lg / 2 words: f g mod x^(2^lg / 2).
//   product@copy/<lg>            what product/<lg> replaces: a Convolution, f and g copied in, the
//                                product copied out (fps::raw::mul before issue #38 round 2).
//   product@into/<lg>            f copied into the caller's a, Convolution(n, n, a), g copied in.
//   square/<lg>                  square(f, out), f of 2^lg / 2 words, out of 2^lg - 1, as product.
//   square@multiply/<lg>         multiply(f, f, out) on the same spans: what square/<lg> replaces.
//   middle_product/<lg>          middle_product(f, g, out), f of 2^lg words, g of 2^lg / 2, out of
//                                2^lg / 2 + 1: a transform of 2^lg.
//   middle_product@multiply/<lg> what it replaces (fps::middle_product before issue #55):
//                                multiply(f, g, out) into 2^lg words (a transform of 2^(lg + 1)),
//                                then 2^lg / 2 + 1 words moved down.
//   <op>/<N>                     ragged lengths (issue #155) for product, product_cut, square and
//                                middle_product: N = 2^k + 1, 1.25, 1.5, 1.625, 1.75, 1.875 2^k and
//                                2^(k + 1) - 1 for k = 8, 12, 16, 20, 22 (named k+1, k.25, k.5,
//                                k.625, k.75, k.875, (k+1)-1), and 5e5, 1e6, 1.2e6. Items per call: N. product: f
//                                of N / 2 + 1 words, g of the rest, out of N = n + m - 1;
//                                product_cut: out of N / 2 + 1; square: f of (N + 1) / 2;
//                                middle_product: f of N, g of N / 2, out of N - N / 2 + 1.
//   <op>/<k>+L/<d>               product, square and middle_product of N = 2^k + 2^k / d words, d =
//                                128, 32, 16, and of 2^k + 1, for k = 12 .. 22 (issue #261).
// Tables past 2^16 are built by a thread's first call, then kept (ntt.hpp, Memory). Repeated calls
// reuse the last output as input, and convolution cases write no factors: the timing does not
// depend on the values.
#include "backend/ntt/ntt.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <new>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "backend/bench.hpp"

namespace {

using u32 = std::uint32_t;
using namespace fps::ntt;

// The scalar variant takes seconds at 2^25; its benches stop here.
constexpr int kScalarMaxLog = 18;
// Variants not selected and convolution cases: up to 2^22 (64 MiB for a multiply), to keep the
// gate's 21 rounds short.
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

// count words 64 bytes past a page boundary, then kPadding words.
class CallerWords {
public:
    explicit CallerWords(std::size_t count)
        : words_(static_cast<u32*>(::operator new((kOffset + count + kPadding) * sizeof(u32), std::align_val_t(4096)))) {}
    ~CallerWords() { ::operator delete(words_, std::align_val_t(4096)); }
    CallerWords(const CallerWords&) = delete;
    CallerWords& operator=(const CallerWords&) = delete;
    u32* begin() { return words_ + kOffset; }

private:
    static constexpr std::size_t kOffset = 16;
    u32* words_;
};

void add_convolution_cases(bench::Suite& suite) {
    for (int lg = kMinLog; lg <= kVariantMaxLog; ++lg) {
        const std::size_t n = (std::size_t(1) << lg) / 2;
        suite.add("convolution/" + std::to_string(lg), [n] {
            return [n] {
                Convolution conv(n, n);
                bench::keep(*conv.multiply());
            };
        }, 2 * n);
        suite.add("convolution_into/" + std::to_string(lg), [n] {
            return [n, a = std::make_shared<CallerWords>(Convolution::length(n, n))] {
                Convolution conv(n, n, a->begin());
                bench::keep(*conv.multiply());
            };
        }, 2 * n);
    }
}

// Factors f and g of n words and an output of count words, each in caller memory.
struct Spans {
    Spans(std::size_t n, std::size_t count) : f(n), g(n), out(count), n(n), count(count) {
        const Words w(n);
        std::copy_n(w.lines.data()->words, n, f.begin());
        std::copy_n(w.lines.data()->words, n, g.begin());
    }
    std::span<const u32> fs() { return {f.begin(), n}; }
    std::span<const u32> gs() { return {g.begin(), n}; }
    std::span<u32> outs() { return {out.begin(), count}; }
    CallerWords f, g, out;
    std::size_t n, count;
};

void add_product_cases(bench::Suite& suite) {
    for (int lg = kMinLog; lg <= kVariantMaxLog; ++lg) {
        const std::size_t n = (std::size_t(1) << lg) / 2;
        const std::string size = "/" + std::to_string(lg);
        for (const auto& [name, count] : {std::pair{"product", 2 * n - 1}, std::pair{"product_cut", n}}) {
            suite.add(name + size, [n, count] {
                return [s = std::make_shared<Spans>(n, count)] {
                    multiply(s->fs(), s->gs(), s->outs());
                    bench::keep(s->out.begin()[0]);
                };
            }, 2 * n);
        }
        suite.add("product@copy" + size, [n] {
            return [s = std::make_shared<Spans>(n, 2 * n - 1)] {
                Convolution conv(s->n, s->n);
                std::ranges::copy(s->fs(), conv.a());
                std::ranges::copy(s->gs(), conv.b());
                const u32* c = conv.multiply();
                std::copy_n(c, conv.size(), s->out.begin());
                bench::keep(s->out.begin()[0]);
            };
        }, 2 * n);
        suite.add("square" + size, [n] {
            return [s = std::make_shared<Spans>(n, 2 * n - 1)] {
                square(s->fs(), s->outs());
                bench::keep(s->out.begin()[0]);
            };
        }, 2 * n);
        suite.add("square@multiply" + size, [n] {
            return [s = std::make_shared<Spans>(n, 2 * n - 1)] {
                multiply(s->fs(), s->fs(), s->outs());
                bench::keep(s->out.begin()[0]);
            };
        }, 2 * n);
        // f of 2n words, g the first n of the second factor.
        suite.add("middle_product" + size, [n] {
            return [s = std::make_shared<Spans>(2 * n, n + 1)] {
                middle_product(s->fs(), s->gs().first(s->n / 2), s->outs());
                bench::keep(s->out.begin()[0]);
            };
        }, 2 * n);
        suite.add("middle_product@multiply" + size, [n] {
            return [s = std::make_shared<Spans>(2 * n, 2 * n)] {
                const std::size_t m = s->n / 2;
                multiply(s->fs(), s->gs().first(m), s->outs());
                std::memmove(s->out.begin(), s->out.begin() + (m - 1), (m + 1) * sizeof(u32));
                bench::keep(s->out.begin()[0]);
            };
        }, 2 * n);
        suite.add("product@into" + size, [n] {
            return [s = std::make_shared<Spans>(n, Convolution::length(n, n))] {
                std::ranges::copy(s->fs(), s->out.begin());
                Convolution conv(s->n, s->n, s->out.begin());
                std::ranges::copy(s->gs(), conv.b());
                bench::keep(*conv.multiply());
            };
        }, 2 * n);
    }
}

// The ragged lengths N and their names (issue #155).
std::vector<std::pair<std::string, std::size_t>> ragged_lengths() {
    std::vector<std::pair<std::string, std::size_t>> lengths;
    for (const int k : {8, 12, 16, 20, 22}) {
        const std::size_t len = std::size_t(1) << k;
        const std::string lg = std::to_string(k);
        lengths.insert(lengths.end(), {{lg + "+1", len + 1}, {lg + ".25", len + len / 4}, {lg + ".5", len + len / 2},
                                       {lg + ".625", len + 5 * len / 8}, {lg + ".75", 2 * len - len / 4},
                                       {lg + ".875", 2 * len - len / 8}, {std::to_string(k + 1) + "-1", 2 * len - 1}});
    }
    lengths.insert(lengths.end(), {{"5e5", 500000}, {"1e6", 1000000}, {"1.2e6", 1200000}});
    return lengths;
}

void add_ragged_cases(bench::Suite& suite) {
    for (const auto& [size, N] : ragged_lengths()) {
        const std::size_t n = N / 2 + 1, m = N - n + 1;
        for (const auto& [name, count] : {std::pair{"product/", N}, std::pair{"product_cut/", n}}) {
            suite.add(name + size, [n, m, count] {
                return [f = std::make_shared<Spans>(n, count), m] {
                    multiply(f->fs(), f->gs().first(m), f->outs());
                    bench::keep(f->out.begin()[0]);
                };
            }, N);
        }
        suite.add("square/" + size, [N] {
            const std::size_t h = (N + 1) / 2;
            return [s = std::make_shared<Spans>(h, 2 * h - 1)] {
                square(s->fs(), s->outs());
                bench::keep(s->out.begin()[0]);
            };
        }, N);
        suite.add("middle_product/" + size, [N] {
            return [s = std::make_shared<Spans>(N, N - N / 2 + 1), m = N / 2] {
                middle_product(s->fs(), s->gs().first(m), s->outs());
                bench::keep(s->out.begin()[0]);
            };
        }, N);
    }
}

// Just past a power of two (issue #261): N = 2^k + 1 and 2^k + 2^k / d for d = 128, 32, 16 (named
// k+1, k+L/128, k+L/32, k+L/16), k = 12 .. 22, for product, square and middle_product, as
// add_ragged_cases (k+1 there for k = 12, 16, 20, 22).
void add_bucket1_cases(bench::Suite& suite) {
    for (int k = 12; k <= 22; ++k) {
        const std::size_t L = std::size_t(1) << k;
        const std::string lg = std::to_string(k);
        for (const auto& [size, N] : {std::pair{lg + "+1", L + 1}, std::pair{lg + "+L/128", L + L / 128},
                                      std::pair{lg + "+L/32", L + L / 32}, std::pair{lg + "+L/16", L + L / 16}}) {
            if (size.ends_with("+1") && (k % 4 == 0 || k == 22)) continue;  // in add_ragged_cases
            const std::size_t n = N / 2 + 1, m = N - n + 1;
            suite.add("product/" + size, [n, m, N] {
                return [f = std::make_shared<Spans>(n, N), m] {
                    multiply(f->fs(), f->gs().first(m), f->outs());
                    bench::keep(f->out.begin()[0]);
                };
            }, N);
            suite.add("square/" + size, [N] {
                const std::size_t h = (N + 1) / 2;
                return [s = std::make_shared<Spans>(h, 2 * h - 1)] {
                    square(s->fs(), s->outs());
                    bench::keep(s->out.begin()[0]);
                };
            }, N);
            suite.add("middle_product/" + size, [N] {
                return [s = std::make_shared<Spans>(N, N - N / 2 + 1), m = N / 2] {
                    middle_product(s->fs(), s->gs().first(m), s->outs());
                    bench::keep(s->out.begin()[0]);
                };
            }, N);
        }
    }
}

template <class V>
constexpr bool only_candidate =std::is_same_v<Selected, Candidates<V>>;

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
    // not Avx2 where Avx512Vl runs the same code on 32 registers), and avx2, unless it is the
    // target's one candidate (select.hpp).
    fps::simd::for_each(fps::simd::All{}, [&]<class S> {
        using V = variant::intrinsics<S>;
        constexpr bool worth = S::lanes > 1 && (S::lanes >= 8 || std::is_same_v<S, fps::simd::Native>) &&
                               !(S::lanes == 8 && S::registers < fps::simd::Native::registers);
        if constexpr (worth && !only_candidate<V>)
            add_cases<V, V, V>(suite, std::string("@intrinsics_") + S::name, kVariantMaxLog);
    });
#ifdef __AVX2__
    if constexpr (!only_candidate<variant::avx2>)
        add_cases<variant::avx2, variant::avx2, variant::avx2>(suite, "@avx2", kVariantMaxLog);
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    add_cases<variant::avx512, variant::avx512, variant::avx512>(suite, "@avx512", kVariantMaxLog);
#endif
    add_convolution_cases(suite);
    add_product_cases(suite);
    add_ragged_cases(suite);
    add_bucket1_cases(suite);
    return suite.run();
}
