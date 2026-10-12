// Benches for ntt::Transform's leaf-form ops (ntt.hpp), per transform length 2^lg; items per call:
// 2^lg words. A binary of its own: in bench.cpp's binary, these kernels moved its short products by
// code placement alone (Zen 3 product/6 1.12, convolution/7 1.09 against main; notes.md, #49).
//   leaf_product/<lg>, leaf_product_add/<lg>   Transform::multiply (out = a) and multiply_add.
//   forward_values/<lg>, inverse_values/<lg>   the values form, in place, to 2^23.
//   inverse_to/<lg>                            inverse(F, out): out of 2^lg words in caller memory,
//                                              64 bytes past a page boundary; F kept.
//   inverse_to@copy/<lg>                       what it replaces: F copied, the inverse in place,
//                                              the coefficients copied out.
//   <op>@<variant>/<lg>                        one variant: intrinsics_<simd struct>, avx2 (those
//                                              not Native, these ops' default), at 2^8, 2^12,
//                                              2^16 and 2^20.
#include "backend/ntt/ntt.hpp"

#include <algorithm>
#include <memory>
#include <new>
#include <random>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "backend/bench.hpp"

namespace {

using u32 = std::uint32_t;
using namespace fps::ntt;

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

// The ops by variant V at the lengths 2^lg; with "@copy" (Native only) what inverse(F, out)
// replaces.
template <class V>
void add_leaf_cases(bench::Suite& suite, const std::string& suffix, const std::vector<int>& lgs) {
    for (const int lg : lgs) {
        const std::size_t len = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        const auto words = [len] { return std::make_shared<Words>(len); };
        suite.add("leaf_product" + suffix + size, [lg, words] {
            return [t = std::make_shared<Transform>(lg), a = words(), b = words()] {
                t->template multiply<V>(a->begin(), b->begin(), a->begin());
                bench::keep(*a->begin());
            };
        }, len);
        suite.add("leaf_product_add" + suffix + size, [lg, words, len] {
            return [t = std::make_shared<Transform>(lg), a = words(), b = std::make_shared<Words>(len + 1), c = words()] {
                t->template multiply_add<V>(c->begin(), a->begin(), b->begin());
                bench::keep(*c->begin());
            };
        }, len);
        suite.add("forward_values" + suffix + size, [lg, words] {
            return [t = std::make_shared<Transform>(lg), f = words()] {
                t->template forward<V>(f->begin(), Form::values);
                bench::keep(*f->begin());
            };
        }, len);
        suite.add("inverse_values" + suffix + size, [lg, words] {
            return [t = std::make_shared<Transform>(lg), f = words()] {
                t->template inverse<V>(f->begin(), Form::values);
                bench::keep(*f->begin());
            };
        }, len);
        suite.add("inverse_to" + suffix + size, [lg, words, len] {
            return [t = std::make_shared<Transform>(lg), f = words(), out = std::make_shared<CallerWords>(len), len] {
                t->template inverse<V>(f->begin(), std::span(out->begin(), len));
                bench::keep(*out->begin());
            };
        }, len);
        if (suffix.empty()) {
            suite.add("inverse_to@copy" + size, [lg, words, len] {
                return [t = std::make_shared<Transform>(lg), f = words(), g = words(), out = std::make_shared<CallerWords>(len),
                        len] {
                    std::copy_n(f->begin(), len, g->begin());
                    t->inverse(g->begin());
                    std::copy_n(g->begin(), len, out->begin());
                    bench::keep(*out->begin());
                };
            }, len);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    const bool native_scalar = std::is_same_v<Native, variant::scalar>;
    std::vector<int> lgs;
    for (int lg = kMinLog; lg <= kValuesMaxLog; ++lg) lgs.push_back(lg);
    add_leaf_cases<Native>(suite, "", native_scalar ? std::vector<int>{6, 8, 10, 12, 14, 16, 18} : lgs);
    fps::simd::for_each(fps::simd::All{}, [&]<class S> {
        using V = variant::intrinsics<S>;
        if constexpr (S::lanes > 1 && !std::is_same_v<V, Native>)
            add_leaf_cases<V>(suite, std::string("@intrinsics_") + S::name, {8, 12, 16, 20});
    });
#ifdef __AVX2__
    if constexpr (!std::is_same_v<variant::avx2, Native>) add_leaf_cases<variant::avx2>(suite, "@avx2", {8, 12, 16, 20});
#endif
    return suite.run();
}
