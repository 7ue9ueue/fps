// Benches for ntt::dft and ntt::idft (dft.hpp, issue #138), per length L = 2^lg, lg 10 .. 22; items
// per call: L words. f of L random words; out a std::vector kept across calls (the caller's
// memory, 16 bytes past a line as malloc gives it).
//   dft/<lg>, idft/<lg>              the natural-order transforms
//   dft@chirp/<lg>, idft@chirp/<lg>  what they replace: chirp::evaluate_geometric at w (at w^-1,
//                                    then times 1 / L), the frontend's path before (#27)
//   dft@forward/<lg>                 dft's forward alone: f to the values form, no permutation
//   dft@permute/<lg>, idft@permute/<lg>  the permutation alone, values form to natural order and
//                                    back, the values side 1 KiB past a page
//   <op>@direct/<lg>                 the op with the values side not staged (Order<6, 14, kNever, 14>)
//   <op>@r16/<lg>                    the op with tiles of 16 blocks (Order<4, 14, 14, 14>)
//   copy/<lg>                        L words copied, for scale
//   to_values/<lg>                   the frontend's to_values (the issue's yardstick): f of L / 2
//                                    words into new memory (64-byte aligned, zero past f), the
//                                    leaf form in place there
#include "backend/ntt/dft.hpp"

#include <algorithm>
#include <memory>
#include <new>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/bench.hpp"
#include "backend/chirp/chirp.hpp"
#include "backend/chirp/geometric.hpp"

namespace {

using u32 = std::uint32_t;
using namespace fps::ntt;
namespace dd = fps::ntt::detail::dft;
constexpr u32 P = kModulus;

std::vector<u32> random_words(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<u32> a(n);
    for (u32& x : a) x = rng() % P;
    return a;
}

u32 power(u32 x, u32 e) { return detail::power(x, e); }

struct AlignedDelete {
    void operator()(u32* p) const { ::operator delete(p, std::align_val_t(64)); }
};
using Buffer = std::unique_ptr<u32[], AlignedDelete>;

Buffer unwritten(std::size_t n) { return Buffer(static_cast<u32*>(::operator new(n * sizeof(u32), std::align_val_t(64)))); }

Buffer zeros(std::size_t n) {
    Buffer b = unwritten(n);
    std::fill_n(b.get(), n, 0u);
    return b;
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    for (int lg = 10; lg <= 22; ++lg) {
        const std::size_t L = std::size_t(1) << lg;
        const std::string size = "/" + std::to_string(lg);
        const unsigned seed = unsigned(138 * lg);
        const u32 w = power(3, (P - 1) >> lg), wi = power(w, P - 2), inverse_L = power(u32(L), P - 2);
        const auto words = [L, seed](unsigned s) { return std::make_shared<std::vector<u32>>(random_words(L, seed + s)); };

        // The ops with permutation O, under these names.
        const auto ops = [&]<class O>(const std::string& dft_name, const std::string& idft_name) {
            suite.add(dft_name + size, [words] {
                return [f = words(0), out = words(1)] {
                    dd::run_dft<SelectedForward, O>(*f, *out);
                    bench::keep(*out);
                };
            }, L);
            suite.add(idft_name + size, [words] {
                return [y = words(2), out = words(1)] {
                    dd::run_idft<SelectedInverse, O>(*y, *out);
                    bench::keep(*out);
                };
            }, L);
        };
        ops.template operator()<dd::Order<>>("dft", "idft");
        suite.add("dft@chirp" + size, [words, w] {
            return [f = words(0), out = words(1), w] {
                fps::chirp::evaluate_geometric(*f, 1, w, *out);
                bench::keep(*out);
            };
        }, L);
        suite.add("idft@chirp" + size, [words, wi, inverse_L] {
            return [y = words(2), out = words(1), wi, inverse_L] {
                fps::chirp::evaluate_geometric(*y, 1, wi, *out);
                fps::chirp::multiply_chirp(inverse_L, 1, 1, *out);
                bench::keep(*out);
            };
        }, L);
        suite.add("dft@forward" + size, [words, lg, L] {
            return [f = words(0), work = zeros(L + kPadding), lg] {
                dd::forward<SelectedForward>(*f, work.get(), lg, detail::tables<SelectedForward>(lg).first,
                                             dd::value_rows(lg));
                bench::keep(work);
            };
        }, L);
        suite.add("dft@permute" + size, [words, lg, L] {
            return [f = zeros(L + 1024), out = words(1), lg] {
                dd::Order<>::run<false>(f.get() + 256, out->data(), lg);
                bench::keep(*out);
            };
        }, L);
        suite.add("idft@permute" + size, [words, lg, L] {
            return [y = words(2), out = zeros(L + 1024), lg] {
                dd::Order<>::run<true>(y->data(), out.get() + 256, lg);
                bench::keep(out);
            };
        }, L);
        ops.template operator()<dd::Order<6, 14, dd::kNever, 14>>("dft@direct", "idft@direct");
        ops.template operator()<dd::Order<4, 14, 14, 14>>("dft@r16", "idft@r16");
        suite.add("copy" + size, [words] {
            return [f = words(0), out = words(1)] {
                std::copy(f->begin(), f->end(), out->begin());
                bench::keep(*out);
            };
        }, L);
        suite.add("to_values" + size, [words, lg, L] {
            return [f = words(0), t = std::make_shared<Transform>(lg), L] {
                Buffer a = unwritten(L + kPadding);
                std::copy_n(f->data(), L / 2, a.get());
                std::fill(a.get() + L / 2, a.get() + L + kPadding, 0u);
                t->forward(a.get());
                bench::keep(a);
            };
        }, L);
    }
    return suite.run();
}
