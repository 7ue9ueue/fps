// Tests of backend/composition's round-3 leaf passes (issue #149 round 3): every simd struct's runs
// (Shoup weights, the projection's and level 1's leaves out of line, the product's leaves one at a
// time) against the scalar ones on random leaves and on leaves at the top of their range; the CRT
// without the 1/2 (Crt::twice(), lazy sums) as twice the plain one; f reversed with a scale. The
// doubled levels' factors are checked through compose and power_projection (test_doubling.cpp).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "backend/composition/composition.hpp"
#include "backend/simd/select.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = 998244353;
namespace c = fps::composition;
namespace d = fps::composition::detail;

int failures = 0;

void check(bool ok, const char* what, int line, u64 a = 0, u64 b = 0) {
    if (ok) return;
    if (++failures <= 20)
        std::printf("FAIL line %d: %s (%llu, %llu)\n", line, what, (unsigned long long)a, (unsigned long long)b);
}
#define CHECK(cond, ...) check(bool(cond), #cond, __LINE__ __VA_OPT__(, ) __VA_ARGS__)

std::mt19937_64 rng(1493);

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }

// kind 0: random words; 1: all P - 1; 2: P - 1 and 0 alternating; 3: random among 0, 1, P - 1.
Vec leaves(std::size_t n, int kind) {
    Vec a(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (kind == 0) a[i] = u32(rng() % P);
        else if (kind == 1) a[i] = P - 1;
        else if (kind == 2) a[i] = i % 2 ? 0 : P - 1;
        else a[i] = std::array<u32, 3>{0, 1, P - 1}[rng() % 3];
    }
    return a;
}

// The runs of every simd struct against Pairs<Scalar>, from pair 0 and from pair 8.
void test_runs() {
    const std::size_t m = 512, pairs = m / 4;
    const auto& t = d::pair_tables(pairs);
    using Ref = d::Pairs<fps::simd::Scalar>;
    for (int kind = 0; kind < 4; ++kind) {
        const Vec F = leaves(4 * m, kind), F2 = leaves(4 * m, kind), H = leaves(2 * m, kind), H2 = leaves(2 * m, kind);
        const Vec F1 = leaves(m, kind), Fq = leaves(m, kind), Fp = leaves(m, kind);
        Vec rG(2 * m), rGt(2 * m), rV(2 * m), rW(2 * m), rVt(2 * m), rWt(2 * m), rR(4 * m), rR2(4 * m);
        Ref::graeffe_run(F.data(), rG.data(), 0, pairs, t);
        Ref::graeffe_run(F.data(), rGt.data(), 0, pairs, t, d::PairTables::twice());
        Ref::projection_run(F.data(), F2.data(), rV.data(), rW.data(), 0, pairs, t);
        Ref::projection_run(F.data(), F2.data(), rVt.data(), rWt.data(), 0, pairs, t, d::PairTables::twice());
        Ref::product_run(H.data(), F.data(), rR.data(), 0, pairs, t);
        {
            const u32* const fp[2] = {H.data(), H2.data()};
            const u32* const fq[2] = {F.data(), F2.data()};
            Ref::product_run<2>(fp, fq, rR2.data(), 0, pairs, t);
        }
        std::vector<Vec> rL(5, Vec(m / 2)), rL3(3, Vec(m / 2));
        {
            u32* out[5] = {rL[0].data(), rL[1].data(), rL[2].data(), rL[3].data(), rL[4].data()};
            Ref::level1_run(F1.data(), Fq.data(), Fp.data(), out, 0, m / 16, t);
            u32* out3[5] = {rL3[0].data(), rL3[1].data(), rL3[2].data(), nullptr, nullptr};
            Ref::level1_run(F1.data(), Fq.data(), nullptr, out3, 0, m / 16, t);
        }
        // Twice: 2 times the plain CRT's leaves.
        bool twice = true;
        for (std::size_t i = 0; i < 2 * m; ++i)
            twice = twice && rGt[i] == mul(2, rG[i]) && rVt[i] == mul(2, rV[i]) && rWt[i] == mul(2, rW[i]);
        CHECK(twice, kind);
        CHECK(rV == rG, kind);
        for (int i = 0; i < 3; ++i) CHECK(rL3[std::size_t(i)] == rL[std::size_t(i)], kind, i);
        fps::simd::for_each(fps::simd::All{}, [&]<class S> {
            using N = d::Pairs<d::Narrow<S>>;
            for (std::size_t p0 : {std::size_t(0), std::size_t(8)}) {
                const std::size_t count = pairs - p0;
                Vec G(2 * m), Gt(2 * m), V(2 * m), W(2 * m), Vt(2 * m), Wt(2 * m), R(4 * m), R2(4 * m);
                N::graeffe_run(F.data() + 16 * p0, G.data() + 8 * p0, p0, count, t);
                N::graeffe_run(F.data() + 16 * p0, Gt.data() + 8 * p0, p0, count, t, d::PairTables::twice());
                N::projection_run(F.data() + 16 * p0, F2.data() + 16 * p0, V.data() + 8 * p0, W.data() + 8 * p0, p0, count, t);
                N::projection_run(F.data() + 16 * p0, F2.data() + 16 * p0, Vt.data() + 8 * p0, Wt.data() + 8 * p0, p0, count, t,
                                  d::PairTables::twice());
                N::product_run(H.data() + 8 * p0, F.data() + 16 * p0, R.data() + 16 * p0, p0, count, t);
                {
                    const u32* const fp[2] = {H.data() + 8 * p0, H2.data() + 8 * p0};
                    const u32* const fq[2] = {F.data() + 16 * p0, F2.data() + 16 * p0};
                    N::template product_run<2>(fp, fq, R2.data() + 16 * p0, p0, count, t);
                }
                const auto tail = [p0](const Vec& x, const Vec& y, std::size_t per) {
                    return std::equal(x.begin() + std::ptrdiff_t(per * p0), x.end(), y.begin() + std::ptrdiff_t(per * p0));
                };
                CHECK(tail(G, rG, 8) && tail(Gt, rGt, 8), S::lanes, kind);
                CHECK(tail(V, rV, 8) && tail(W, rW, 8) && tail(Vt, rVt, 8) && tail(Wt, rWt, 8), S::lanes, kind);
                CHECK(tail(R, rR, 16) && tail(R2, rR2, 16), S::lanes, kind);
            }
            if (m / 16 % N::W == 0) {
                std::vector<Vec> L(5, Vec(m / 2)), L3(3, Vec(m / 2));
                u32* out[5] = {L[0].data(), L[1].data(), L[2].data(), L[3].data(), L[4].data()};
                N::level1_run(F1.data(), Fq.data(), Fp.data(), out, 0, m / 16, t);
                u32* out3[5] = {L3[0].data(), L3[1].data(), L3[2].data(), nullptr, nullptr};
                N::level1_run(F1.data(), Fq.data(), nullptr, out3, 0, m / 16, t);
                CHECK(L == rL && L3 == rL3, S::lanes, kind);
            }
        });
    }
}

// The twice runs in place (V over Q's leaves, W over P's), in runs of 32 pairs from the start, as
// the doubled levels' hooks write.
void test_in_place() {
    const std::size_t m = 512, pairs = m / 4;
    const auto& t = d::pair_tables(pairs);
    const Vec F = leaves(4 * m, 0), F2 = leaves(4 * m, 0);
    fps::simd::for_each(fps::simd::All{}, [&]<class S> {
        using N = d::Pairs<d::Narrow<S>>;
        Vec V(2 * m), W(2 * m), G(2 * m);
        N::projection_run(F.data(), F2.data(), V.data(), W.data(), 0, pairs, t, d::PairTables::twice());
        N::graeffe_run(F.data(), G.data(), 0, pairs, t, d::PairTables::twice());
        Vec Q = F, Pp = F2, Qg = F;
        for (std::size_t p0 = 0; p0 < pairs; p0 += 32) {
            N::projection_run(Q.data() + 16 * p0, Pp.data() + 16 * p0, Q.data() + 8 * p0, Pp.data() + 8 * p0, p0, 32, t,
                              d::PairTables::twice());
            N::graeffe_run(Qg.data() + 16 * p0, Qg.data() + 8 * p0, p0, 32, t, d::PairTables::twice());
        }
        CHECK(std::equal(V.begin(), V.end(), Q.begin()) && std::equal(W.begin(), W.end(), Pp.begin()), S::lanes);
        CHECK(std::equal(G.begin(), G.end(), Qg.begin()), S::lanes);
    });
}

// reverse_words with and without a scale, every count to 40 and a few longer.
void test_reverse() {
    for (std::size_t n = 0; n <= 300; n += n < 40 ? 1 : 37) {
        const Vec f = leaves(n, 0);
        for (u32 scale : {u32(1), u32(2), P - 1, u32(rng() % P)}) {
            Vec out(n + 3, 7);
            d::reverse_words(f.data(), n, out.data(), scale);
            bool ok = out[n] == 7 && out[n + 1] == 7;
            for (std::size_t i = 0; i < n; ++i) ok = ok && out[i] == mul(f[n - 1 - i], scale);
            CHECK(ok, n, scale);
        }
    }
}

// doubled_factor against its definition; compose at n = m (from 512 on: every doubled level, with
// their factors) against the same at n + 1 (m twice as long: other levels, other factors), and
// power_projection by the transposition: sum_k f_k a_k = [x^(n-1)] f(g).
void test_factors() {
    const u32 c0 = u32(rng() % (P - 1)) + 1;
    for (std::size_t L : {std::size_t(16), std::size_t(512), std::size_t(1) << 15}) {
        const u32 want = mul(mul(mul(c0, c0), d::power((P + 1) / 2, 31)), u32(L / 8));
        CHECK(d::doubled_factor(c0, L) == want, L);
    }
    for (std::size_t n : {std::size_t(512), std::size_t(1024), std::size_t(4096), std::size_t(16384)}) {
        const Vec f = leaves(n, 0), g = leaves(n, 0);
        Vec h(n), h2(n + 1), a(n);
        c::compose(f, g, h);
        c::compose(f, g, h2);
        CHECK(std::equal(h.begin(), h.end(), h2.begin()), n);
        c::power_projection(g, a);
        u64 dot = 0;
        for (std::size_t k = 0; k < n; ++k) dot = (dot + u64(f[k]) * a[k]) % P;
        CHECK(dot == h[n - 1], n);
    }
}

}  // namespace

int main() {
    test_runs();
    test_in_place();
    test_reverse();
    test_factors();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
