// Tests every modarith operation and variant, on every simd struct this machine can runs, against
// the scalar model (model.hpp). Inputs are random below each tested bound, with the edges
// (0, 1, bound - 1) often; the bounds include each contract's limits (e.g. a b just below 2^32 P).
// Every lane must be below the model's bound and congruent to its value. Exits nonzero on a failure.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "backend/modarith/model.hpp"
#include "backend/simd/select.hpp"

namespace {

using fps::modarith::Barrett;
using fps::modarith::Montgomery;
using fps::modarith::Shoup;
using fps::modarith::u32;
using fps::modarith::u64;
using fps::modarith::model::Lazy;

constexpr int kTrials = 300;
constexpr u64 kTwo31 = u64(1) << 31, kTwo32 = u64(1) << 32;

std::mt19937_64 rng(20261010);
int failures = 0;

u64 draw(u64 bound) {
    const u64 r = rng();
    switch (r % 8) {
        case 0: return bound - 1;
        case 1: return (r >> 3) % std::min<u64>(bound, 2);
        default: return (r >> 3) % bound;
    }
}

template <class S, u32 P>
struct Tester {
    using O = fps::modarith::Ops<S, P>;
    using Model = fps::modarith::model::Model<P>;
    using V = typename S::V;
    using Words = std::array<u32, S::lanes>;

    static Words random(u64 bound) {
        Words w;
        for (u32& x : w) x = u32(draw(bound));
        return w;
    }

    static void fail(const std::string& op, const std::string& what) {
        if (++failures <= 20) std::printf("FAIL %s P=%u %s: %s\n", S::name, P, op.c_str(), what.c_str());
    }

    // Lane i of got against expect(i), a Lazy from the model.
    template <class Expect>
    static void check(const std::string& op, V got, Expect expect) {
        Words out;
        S::store(out.data(), got);
        for (int i = 0; i < S::lanes; ++i) {
            const Lazy e = expect(i);
            if (out[i] >= e.bound || out[i] % P != e.value % P) {
                fail(op, "lane " + std::to_string(i) + " = " + std::to_string(out[i]) + ", want " +
                             std::to_string(e.value % P) + " mod P, below " + std::to_string(e.bound));
                return;
            }
        }
    }

    // vector(a, b) against model(a, b) for a < A, b < B.
    template <class Vector, class ModelOp>
    static void binary(const std::string& op, u64 A, u64 B, Vector vector, ModelOp model) {
        const std::string name = op + " (bounds " + std::to_string(A) + ", " + std::to_string(B) + ")";
        for (int t = 0; t < kTrials; ++t) {
            const Words a = random(A), b = random(B);
            check(name, vector(S::load(a.data()), S::load(b.data())),
                  [&](int i) { return model(Lazy{a[i], A}, Lazy{b[i], B}); });
        }
    }

    // Largest B with (A - 1) (B - 1) < limit, at most cap.
    static u64 partner(u64 A, u64 limit, u64 cap) {
        return std::min<u64>(cap, u64(((unsigned __int128)limit - 1) / (A - 1) + 1));
    }

    static void montgomery_variants() {
        const u64 bounds[][2] = {{P, P},
                                 {2 * u64(P), 2 * u64(P)},
                                 {std::min(4 * u64(P), kTwo31), P},
                                 {kTwo31, partner(kTwo31, kTwo32 * P, kTwo31)},
                                 {4 * u64(P), partner(4 * u64(P), kTwo32 * P, kTwo31)}};
        const auto test = [&]<Montgomery v>(const char* name) {
            if constexpr (fps::modarith::has<S>(v))
                for (const auto& [A, B] : bounds)
                    if (A <= kTwo31)
                        binary(name, A, B, [](V a, V b) { return O::template montgomery_as<v>(a, b); },
                               Model::montgomery);
        };
        test.template operator()<Montgomery::wide>("montgomery@wide");
        test.template operator()<Montgomery::mulhi>("montgomery@mulhi");
        test.template operator()<Montgomery::doubling>("montgomery@doubling");
        binary("montgomery", P, P, O::montgomery, Model::montgomery);
    }

    // x < X times per-lane w < P, with the variant's quotients.
    template <class Vector, class ModelOp, class Quotient>
    static void shoup_like(const std::string& op, u64 X, Vector vector, ModelOp model, Quotient quotient) {
        const std::string name = op + " (bound " + std::to_string(X) + ")";
        for (int t = 0; t < kTrials; ++t) {
            const Words x = random(X), w = random(P);
            Words q;
            for (int i = 0; i < S::lanes; ++i) q[i] = quotient(w[i]);
            check(name, vector(S::load(x.data()), S::load(w.data()), S::load(q.data())),
                  [&](int i) { return model(Lazy{x[i], X}, w[i]); });
        }
    }

    static void shoup_variants() {
        const u64 bounds[] = {P, 4 * u64(P), kTwo31, kTwo32};
        const auto test = [&]<Shoup v>(const char* name) {
            if constexpr (fps::modarith::has<S>(v))
                for (u64 X : bounds)
                    shoup_like(name, X, [](V x, V w, V q) { return O::template shoup_as<v>(x, w, q); },
                               Model::shoup, O::shoup_quotient);
        };
        test.template operator()<Shoup::mulhi>("shoup@mulhi");
        test.template operator()<Shoup::wide>("shoup@wide");
        shoup_like("shoup", kTwo32, O::shoup, Model::shoup, O::shoup_quotient);
        if constexpr (requires(V x) { O::shoup_doubling(x, x, x); })
            for (u64 X : {u64(P), std::min(4 * u64(P), kTwo31), kTwo31})
                shoup_like("shoup_doubling", X, O::shoup_doubling, Model::shoup_doubling, O::shoup_quotient31);
    }

    static void ifma() {
        if constexpr (requires(V x) { O::montgomery52(x, x); }) {
            for (u64 A : {u64(P), 4 * u64(P), kTwo32}) binary("montgomery52", A, A, O::montgomery52, Model::montgomery52);
            for (u64 X : {u64(P), kTwo32}) {
                for (int t = 0; t < kTrials; ++t) {  // w, q52 in 64-bit lanes
                    const Words x = random(X), w32 = random(P);
                    std::array<u64, S::lanes / 2> w, q;
                    for (int i = 0; i < S::lanes / 2; ++i) w[i] = w32[i], q[i] = O::shoup_quotient52(w32[i]);
                    const V got = O::shoup52(S::load(x.data()), S::load(reinterpret_cast<const u32*>(w.data())),
                                             S::load(reinterpret_cast<const u32*>(q.data())));
                    check("shoup52 (bound " + std::to_string(X) + ")", got,
                          [&](int i) { return Model::shoup52(Lazy{x[i], X}, u32(w[i / 2])); });
                }
            }
        }
    }

    static void run() {
        binary("add", P, P, O::add, Model::add);
        binary("sub", P, P, O::sub, Model::sub);
        binary("reduce<P>", 2 * u64(P), 1, [](V x, V) { return O::template reduce<P>(x); },
               [](Lazy x, Lazy) { return Model::template reduce<P>(x); });
        binary("reduce<2P>", 4 * u64(P), 1, [](V x, V) { return O::template reduce<2 * P>(x); },
               [](Lazy x, Lazy) { return Model::template reduce<2 * P>(x); });
        if constexpr (u64(P) * 4 <= kTwo31)
            binary("reduce<4P>", 8 * u64(P), 1, [](V x, V) { return O::template reduce<4 * P>(x); },
                   [](Lazy x, Lazy) { return Model::template reduce<4 * P>(x); });
        binary("sub_lazy<P>", 2 * u64(P), u64(P) + 1, O::template sub_lazy<P>, Model::template sub_lazy<P>);
        binary("sub_lazy<2P>", 2 * u64(P), 2 * u64(P) + 1, O::template sub_lazy<2 * P>,
               Model::template sub_lazy<2 * P>);
        montgomery_variants();
        shoup_variants();
        binary("barrett", P, P, O::barrett, Model::barrett);
        if constexpr (fps::modarith::has<S>(Barrett::wide))
            binary("barrett@wide", P, P, [](V a, V b) { return O::template barrett_as<Barrett::wide>(a, b); },
                   Model::barrett);
        binary("barrett@mulhi", P, P, [](V a, V b) { return O::template barrett_as<Barrett::mulhi>(a, b); },
               Model::barrett);
        binary("barrett@split", P, P, [](V a, V b) { return O::template barrett_as<Barrett::split>(a, b); },
               Model::barrett);
        ifma();
    }
};

template <class S>
void test_isa() {
    const int before = failures;
    Tester<S, 4099>::run();
    Tester<S, 65537>::run();
    Tester<S, 268435459>::run();  // Barrett bound 3P
    Tester<S, 469762049>::run();
    Tester<S, 536870923>::run();  // Barrett bound 3P, just above 2^29
    Tester<S, 998244353>::run();
    Tester<S, 1000000007>::run();
    Tester<S, 1073741789>::run();  // largest prime below 2^30
    std::printf("%-9s %s\n", S::name, failures == before ? "ok" : "FAIL");
}

}  // namespace

int main() {
    try {
        fps::simd::for_each(fps::simd::All{}, []<class S> { test_isa<S>(); });
    } catch (const std::exception& e) {
        std::printf("FAIL test inputs: %s\n", e.what());
        return 1;
    }
    return failures ? 1 : 0;
}
