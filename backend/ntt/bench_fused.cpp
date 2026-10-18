// Benches for ntt::Transform's fused ops (ntt.hpp, issue #112), per transform length 2^lg; items
// per call: 2^lg words. A binary of its own, as bench_leaves.cpp. Sources and outputs in caller
// memory (std::vector), leaf forms and the work array in 64-byte aligned arrays.
//   forward_from_half/<lg>, forward_from/<lg>   forward(in, F), in of 2^lg / 2 and 2^lg words.
//   forward_product/<lg>                        forward_product(in, B, F), in of 2^lg words.
//   cyclic_product/<lg>                         cyclic_product(in, B, work, out): in of 2^lg / 2
//                                               words, the lower half out (a Newton step's).
//   cyclic_product_upper/<lg>                   in of 2^lg words, the upper half out.
//   inverse_product/<lg>, inverse_product3/<lg> inverse_product: 1 and 3 terms, the upper half out;
//   inverse_product5/<lg>, inverse_product8/<lg> 5 and 8 terms (issue #143).
//   toeplitz_product/<lg>                       toeplitz_product of 5 leaf forms, the upper half out
//                                               (issue #218; log's block 2).
//   multiply@steps/<lg>                         what cyclic_product replaces: in copied and zeroed,
//                                               forward, multiply, inverse in place, half copied out.
#include "backend/ntt/ntt.hpp"

#include <algorithm>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/bench.hpp"

namespace {

using u32 = std::uint32_t;
using namespace fps::ntt;

struct Words {
    struct alignas(64) Line {
        u32 words[16];
    };
    explicit Words(std::size_t count, unsigned seed) : lines((count + kPadding + 15) / 16) {
        std::mt19937 rng(seed);
        for (auto& line : lines)
            for (u32& w : line.words) w = rng() % kModulus;
    }
    u32* begin() { return lines.data()->words; }
    std::vector<Line> lines;
};

std::vector<u32> random_words(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<u32> v(n);
    for (u32& w : v) w = rng() % kModulus;
    return v;
}

struct Case {
    explicit Case(int lg)
        : t(lg), len(t.length()), in(random_words(len, 1)), out(len), B(len, 2), C(len, 3), D(len, 4), F(len, 5), work(len, 6) {}
    Transform t;
    std::size_t len;
    std::vector<u32> in, out;
    Words B, C, D, F, work;
};

template <class Op>
void add(bench::Suite& suite, const std::string& name, int lg, Op op) {
    suite.add(name + "/" + std::to_string(lg), [lg, op] {
        return [c = std::make_shared<Case>(lg), op]() mutable {
            op(*c);
            bench::keep(c->out);
        };
    }, std::size_t(1) << lg);
}

// inverse_product<k>/<lg>: k products of 2k leaf forms of their own, the upper half out.
void add_sum(bench::Suite& suite, int k, int lg) {
    suite.add("inverse_product" + std::to_string(k) + "/" + std::to_string(lg), [k, lg] {
        struct Sum {
            Transform t;
            std::vector<Words> forms;
            std::vector<Term> terms;
            Words work;
            std::vector<u32> out;
        };
        const std::size_t len = std::size_t(1) << lg;
        auto c = std::make_shared<Sum>(Sum{Transform(lg), {}, {}, Words(len, 6), std::vector<u32>(len / 2)});
        for (int i = 0; i < 2 * k; ++i) c->forms.emplace_back(c->t.length(), unsigned(10 + i));
        for (int i = 0; i < k; ++i) c->terms.push_back({c->forms[2 * i].begin(), c->forms[2 * i + 1].begin()});
        return [c] {
            c->t.inverse_product(c->terms, c->work.begin(), std::span<u32>(c->out), {.offset = c->t.length() / 2});
            bench::keep(c->out);
        };
    }, std::size_t(1) << lg);
}

// toeplitz_product/<lg>: A1, A2, A3, B0, B1 of their own (A3 rewritten each call), the upper half out.
void add_toeplitz(bench::Suite& suite, int lg) {
    suite.add("toeplitz_product/" + std::to_string(lg), [lg] {
        struct Toep {
            Transform t;
            std::vector<Words> forms;
            Words work;
            std::vector<u32> out;
        };
        const std::size_t len = std::size_t(1) << lg;
        auto c = std::make_shared<Toep>(Toep{Transform(lg), {}, Words(len, 6), std::vector<u32>(len / 2)});
        for (int i = 0; i < 5; ++i) c->forms.emplace_back(len, unsigned(20 + i));
        return [c] {
            auto& f = c->forms;
            c->t.toeplitz_product(f[0].begin(), f[1].begin(), f[2].begin(), f[3].begin(), f[4].begin(), c->work.begin(),
                                  std::span<u32>(c->out), {.offset = c->t.length() / 2});
            bench::keep(c->out);
        };
    }, std::size_t(1) << lg);
}

}  // namespace

int main(int argc, char** argv) {
    bench::Suite suite(argc, argv);
    for (int lg : {8, 12, 16, 20}) {
        add(suite, "forward_from_half", lg, [](Case& c) { c.t.forward(std::span<const u32>(c.in.data(), c.len / 2), c.F.begin()); });
        add(suite, "forward_from", lg, [](Case& c) { c.t.forward(std::span<const u32>(c.in), c.F.begin()); });
        add(suite, "forward_product", lg, [](Case& c) { c.t.forward_product(std::span<const u32>(c.in), c.B.begin(), c.F.begin()); });
        add(suite, "cyclic_product", lg, [](Case& c) {
            c.t.cyclic_product(std::span<const u32>(c.in.data(), c.len / 2), c.B.begin(), c.work.begin(),
                               std::span<u32>(c.out.data(), c.len / 2));
        });
        add(suite, "cyclic_product_upper", lg, [](Case& c) {
            c.t.cyclic_product(std::span<const u32>(c.in), c.B.begin(), c.work.begin(), std::span<u32>(c.out.data(), c.len / 2),
                               {.offset = c.len / 2});
        });
        add(suite, "inverse_product", lg, [](Case& c) {
            const Term terms[] = {{c.B.begin(), c.C.begin()}};
            c.t.inverse_product(terms, c.work.begin(), std::span<u32>(c.out.data(), c.len / 2), {.offset = c.len / 2});
        });
        add(suite, "inverse_product3", lg, [](Case& c) {
            const Term terms[] = {{c.B.begin(), c.C.begin()}, {c.C.begin(), c.D.begin()}, {c.D.begin(), c.B.begin()}};
            c.t.inverse_product(terms, c.work.begin(), std::span<u32>(c.out.data(), c.len / 2), {.offset = c.len / 2});
        });
        for (int k : {5, 8}) add_sum(suite, k, lg);
        add_toeplitz(suite, lg);
        add(suite, "multiply@steps", lg, [](Case& c) {
            u32* w = c.work.begin();
            std::fill(std::copy_n(c.in.data(), c.len / 2, w), w + c.len, 0u);
            c.t.forward(w);
            c.t.multiply(w, c.B.begin(), w);
            c.t.inverse(w);
            std::copy_n(w, c.len / 2, c.out.data());
        });
    }
    return suite.run();
}
