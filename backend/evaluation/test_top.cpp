// Tests for the point tree's top in the lanes (issue #136 round 2): pairs_to_lanes and
// lane_pair_sums on every simd struct the machine runs against forward transforms of cyclic
// products computed apart; the twist tables and roots; combine_alternating, halves_of_residues,
// lane_windows against scalar loops; the lanes transforms past twisted_from (the twisted forms that
// run at m8 = 2^25) against ntt's; and the trees with the twisted forms at small sizes (evaluate
// against Horner, interpolate at its points, the Newton basis expanded back, both at C = L and C <
// L). Exits nonzero on a failure.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend/evaluation/evaluation.hpp"

namespace {

namespace ev = fps::evaluation::detail;
namespace pt = fps::product_tree::detail;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = 998244353;

std::mt19937_64 rng(1362);
int failures = 0, checks = 0;

void expect(bool ok, const std::string& what, u64 detail = 0) {
    ++checks;
    if (!ok && ++failures <= 20) std::printf("FAIL: %s (%llu)\n", what.c_str(), static_cast<unsigned long long>(detail));
}

u32 mul(u32 x, u32 y) { return u32(u64(x) * y % P); }
u32 add(u32 x, u32 y) { return u32((u64(x) + y) % P); }
u32 sub(u32 x, u32 y) { return u32((u64(x) + P - y) % P); }
u32 power(u32 x, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, x = mul(x, x))
        if (e & 1) r = mul(r, x);
    return r;
}

// 64-byte aligned words with kPadding after, filled with `fill`.
struct Words {
    std::vector<u32> memory;
    u32* data;
    explicit Words(std::size_t n, u32 fill = 0) : memory(n + fps::ntt::kPadding + 16, fill) {
        data = memory.data() + (16 - reinterpret_cast<std::uintptr_t>(memory.data()) / 4 % 16) % 16;
    }
};

// a b mod (x^n - 1), schoolbook.
Vec cyclic(const Vec& a, const Vec& b, std::size_t n) {
    Vec c(n);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[(i + j) % n] = u32((c[(i + j) % n] + u64(a[i]) * b[j]) % P);
    return c;
}

Vec random_words(std::size_t n, int kind) {
    Vec v(n);
    for (u32& x : v) x = kind == 1 ? P - 1 : kind == 2 && rng() % 2 ? 0 : u32(rng() % P);
    return v;
}

// The lanes transform of length T of 8 polynomials (T coefficients each), 8 T words.
Words lanes_transform(const std::vector<Vec>& lanes, std::size_t T) {
    Words v(8 * T);
    for (std::size_t j = 0; j < T; ++j)
        for (int l = 0; l < 8; ++l) v.data[8 * j + l] = lanes[l][j];
    pt::forward(v.data, 8 * T);
    return v;
}

// pairs_to_lanes<L, kRightOnes>: pair states s_j = forward_T(a_j), lanes = forward(b_l); out lane l
// = forward of a_(l/2) b_(l^1) 2^-32 mod (X^T - 1), or a_(l/2) itself in odd lanes (kRightOnes).
template <class L, bool kRightOnes>
void test_pairs_to_lanes(const char* name, std::size_t T, int kind) {
    const std::string where = std::string("pairs_to_lanes<") + name + (kRightOnes ? ", ones" : "") + "> T " + std::to_string(T) + " kind " +
                              std::to_string(kind);
    std::vector<Vec> a(4), b(8);
    for (Vec& x : a) x = random_words(T, kind);
    for (Vec& x : b) x = random_words(T, kind);
    std::vector<Words> s;
    for (int j = 0; j < 4; ++j) {
        s.emplace_back(T);
        std::copy(a[j].begin(), a[j].end(), s[j].data);
        pt::forward(s[j].data, T);
    }
    const Words lanes = lanes_transform(b, T);
    Words out(8 * T + 64, 0xdeadbeefu);
    const u32* states[4] = {s[0].data, s[1].data, s[2].data, s[3].data};
    ev::pairs_to_lanes<L, kRightOnes>(states, lanes.data, T, out.data, ev::twist_factors<false>(T / 8));
    std::vector<Vec> want(8);
    for (int l = 0; l < 8; ++l) {
        if (kRightOnes && l % 2) {
            want[l] = a[l / 2];
            continue;
        }
        want[l] = cyclic(a[l / 2], b[l ^ 1], T);
        for (u32& x : want[l]) x = mul(x, pt::kRInverse);
    }
    const Words expected = lanes_transform(want, T);
    std::size_t bad = 0;
    for (std::size_t i = 0; i < 8 * T; ++i) bad += out.data[i] != expected.data[i];
    expect(bad == 0, where, bad);
    bool guard = true;
    for (std::size_t i = 8 * T; i < 8 * T + 64; ++i) guard &= out.data[i] == 0xdeadbeefu;
    expect(guard, where + ": wrote past the lanes");
}

// lane_pair_sums<L>: out[j] = forward_T(R_2j Q_2j+1 + R_2j+1 Q_2j, times 2^-32, mod X^T - 1).
template <class L>
void test_lane_pair_sums(const char* name, std::size_t T, int kind) {
    const std::string where = std::string("lane_pair_sums<") + name + "> T " + std::to_string(T) + " kind " + std::to_string(kind);
    std::vector<Vec> r(8), q(8);
    for (Vec& x : r) x = random_words(T, kind);
    for (Vec& x : q) x = random_words(T, kind);
    const Words rt = lanes_transform(r, T), qt = lanes_transform(q, T);
    std::vector<Words> outs;
    for (int j = 0; j < 4; ++j) outs.emplace_back(T + 64, 0xdeadbeefu);
    u32* out[4];
    for (int j = 0; j < 4; ++j) out[j] = j == 1 ? nullptr : outs[j].data;  // pair 1 not written
    ev::lane_pair_sums<L>(rt.data, qt.data, T, out, ev::twist_factors<true>(T / 8));
    for (int j = 0; j < 4; ++j) {
        if (!out[j]) {
            bool untouched = true;
            for (std::size_t i = 0; i < T + 64; ++i) untouched &= outs[j].data[i] == 0xdeadbeefu;
            expect(untouched, where + ": a null pair written");
            continue;
        }
        const Vec x = cyclic(r[2 * j], q[2 * j + 1], T), y = cyclic(r[2 * j + 1], q[2 * j], T);
        Words want(T);
        for (std::size_t i = 0; i < T; ++i) want.data[i] = mul(add(x[i], y[i]), pt::kRInverse);
        pt::forward(want.data, T);
        std::size_t bad = 0;
        for (std::size_t i = 0; i < T; ++i) bad += out[j][i] != want.data[i];
        expect(bad == 0, where + ": pair " + std::to_string(j), bad);
        bool guard = true;
        for (std::size_t i = T; i < T + 64; ++i) guard &= out[j][i] == 0xdeadbeefu;
        expect(guard, where + ": wrote past the pair");
    }
}

template <class L>
void test_structs(const char* name) {
    for (const std::size_t T : {64, 128, 256, 1024})
        for (int kind = 0; kind < 3; ++kind) {
            test_pairs_to_lanes<L, false>(name, T, kind);
            test_pairs_to_lanes<L, true>(name, T, kind);
            test_lane_pair_sums<L>(name, T, kind);
        }
    // lane_windows: even lanes [T/2, T), odd lanes [T/2 - shift, T - shift), times u[l] 2^-32.
    for (const std::size_t T : {64, 256})
        for (const std::size_t shift : {std::size_t(1), T / 4, T / 2 - 3, T / 2}) {
            const Vec x = random_words(8 * T, 0), u = random_words(8, 0);
            Words in(8 * T), out(4 * T + 64, 0xdeadbeefu);
            std::copy(x.begin(), x.end(), in.data);
            alignas(64) u32 f[8];
            std::copy(u.begin(), u.end(), f);
            ev::lane_windows<L>(in.data, T, shift, f, out.data);
            bool ok = true;
            for (std::size_t j = 0; j < T / 2; ++j)
                for (int l = 0; l < 8; ++l) {
                    const std::size_t from = l % 2 ? T / 2 - shift + j : T / 2 + j;
                    ok &= out.data[8 * j + l] == pt::montgomery(x[8 * from + l], u[l]);
                }
            for (std::size_t i = 4 * T; i < 4 * T + 64; ++i) ok &= out.data[i] == 0xdeadbeefu;
            expect(ok, std::string("lane_windows<") + name + ">", shift);
            // In place, as newton_top runs it.
            ev::lane_windows<L>(in.data, T, shift, f, in.data);
            expect(std::equal(in.data, in.data + 4 * T, out.data), std::string("lane_windows in place <") + name + ">", shift);
        }
}

template <class S>
void test_passes(const char* name) {
    for (const std::size_t h : {std::size_t(16), std::size_t(64), std::size_t(1024)}) {
        const Vec x = random_words(2 * h, 0);
        Words f(2 * h), g(2 * h);
        std::copy(x.begin(), x.end(), f.data);
        std::copy(x.begin(), x.end(), g.data);
        ev::combine_alternating<S>(f.data, h);
        bool ok = true;
        for (std::size_t i = 0; i < h; ++i) ok &= f.data[i] == (i % 2 ? add(x[i], x[h + i]) : sub(x[i], x[h + i]));
        expect(ok, std::string("combine_alternating<") + name + ">", h);
        ev::halves_of_residues<S>(g.data, h);
        const u32 half = (P + 1) / 2;
        ok = true;
        for (std::size_t i = 0; i < h; ++i)
            ok &= g.data[i] == mul(add(x[i], x[h + i]), half) && g.data[h + i] == mul(sub(x[i], x[h + i]), half);
        expect(ok, std::string("halves_of_residues<") + name + ">", h);
    }
}

void test_tables() {
    for (int j = 0; j < 21; ++j) expect(ev::kTopRoots[0][j] == fps::ntt::detail::kRoots[0][j], "kTopRoots agree with ntt's", j);
    for (int j = 0; j < 22; ++j) {
        const u32 r = ev::kTopRoots[0][j];
        expect(power(r, u64(1) << (j + 2)) == 1 && power(r, u64(1) << (j + 1)) == P - 1, "kTopRoots order", j);
        expect(mul(r, ev::kTopRoots[1][j]) == 1, "kTopRoots inverses", j);
    }
    // twist_factors: rho_u^k 2^32 with rho_u = r(4u); the inverse table rho_u^-k 2^32 / 8, as
    // product_tree's untwist_factors; prefix compatible.
    const Vec small(ev::twist_factors<false>(4), ev::twist_factors<false>(4) + 32);
    const u32* t = ev::twist_factors<false>(1 << 12);
    const u32* ti = ev::twist_factors<true>(1 << 12);
    expect(std::equal(small.begin(), small.end(), t), "twist_factors prefix");
    expect(std::equal(ti, ti + 8 * (1 << 12), pt::untwist_factors(1 << 12)), "twist_factors<true> = untwist_factors");
    bool ok = true;
    for (std::size_t u : {std::size_t(0), std::size_t(1), std::size_t(5), std::size_t(4095)}) {
        u32 rho = 1;
        for (int j = 0; j < 20; ++j)
            if ((4 * u) >> j & 1) rho = mul(rho, fps::ntt::detail::kRoots[0][j]);
        for (u32 k = 0; k < 8; ++k) ok &= t[8 * u + k] == mul(power(rho, k), pt::kR);
    }
    expect(ok, "twist_factors values");
}

// The lanes transforms past twisted_from against ntt's own, words of 2^11 .. 2^13.
void test_twisted() {
    for (const std::size_t words : {std::size_t(1) << 11, std::size_t(1) << 12, std::size_t(1) << 13}) {
        const Vec x = random_words(words, 0);
        Words a(words), b(words);
        // forward_upper, in place and from a source
        std::copy_n(x.begin(), words / 2, a.data);
        std::copy_n(x.begin(), words / 2, b.data);
        ev::twisted_from = std::size_t(1) << fps::ntt::kMaxLog;
        ev::lanes_forward_upper(a.data, words);
        ev::twisted_from = words / 2;
        ev::lanes_forward_upper(b.data, words);
        expect(std::equal(a.data, a.data + words / 2, b.data), "twisted forward_upper", words);
        Words c(words);
        ev::lanes_forward_upper(x.data(), c.data, words);
        expect(std::equal(a.data, a.data + words / 2, c.data), "twisted forward_upper from a source", words);
        // inverse_upper undoes it
        ev::lanes_inverse_upper(b.data, words);
        expect(std::equal(b.data, b.data + words / 2, x.begin()), "twisted inverse_upper", words);
        // the whole inverse
        std::copy(x.begin(), x.end(), a.data);
        std::copy(x.begin(), x.end(), b.data);
        ev::twisted_from = std::size_t(1) << fps::ntt::kMaxLog;
        ev::lanes_inverse(a.data, words);
        ev::twisted_from = words / 2;
        ev::lanes_inverse(b.data, words);
        expect(std::equal(a.data, a.data + words, b.data), "twisted inverse", words);
    }
    ev::twisted_from = std::size_t(1) << fps::ntt::kMaxLog;
}

u32 horner(std::span<const u32> f, u32 x) {
    u32 r = 0;
    for (std::size_t i = f.size(); i-- > 0;) r = add(mul(r, x), f[i]);
    return r;
}

// The trees with the lanes' transforms at 16 L words twisted (as at m8 = 2^25), at sizes with C =
// L and C < L (the Newton basis' windows then do not halve).
void test_trees_twisted() {
    for (const std::size_t m : {136, 256, 500, 1024, 1200, 4096}) {
        const std::size_t slots = ev::padded(m) / 8;
        ev::twisted_from = 8 * ev::lane_length(slots);  // the 16 L words ones
        const Vec f = random_words(m, 0);
        Vec points(m);
        for (std::size_t i = 0; i < m; ++i) points[i] = u32((u64(i) * 7919 + 13) % P);
        Vec values(m);
        ev::workspace_spilled = false;
        ev::evaluate_tree(f, points, values);
        bool ok = !ev::workspace_spilled;
        for (std::size_t i = 0; i < m; ++i) ok &= values[i] == horner(f, points[i]);
        expect(ok, "evaluate_tree twisted", m);
        Vec g(m);
        ev::workspace_spilled = false;
        ev::interpolate_tree(points, values, g);
        expect(!ev::workspace_spilled && g == f, "interpolate_tree twisted", m);
        // The Newton basis: f = sum_k c[k] prod_(i < k) (x - p_i), at random x.
        Vec c(m);
        ev::workspace_spilled = false;
        ev::newton_tree(f, points, c);
        ok = !ev::workspace_spilled;
        for (int t = 0; t < 8; ++t) {
            const u32 x = u32(rng() % P);
            u32 v = 0;
            for (std::size_t k = m; k-- > 0;) v = add(mul(v, sub(x, points[k])), c[k]);
            ok &= v == horner(f, x);
        }
        expect(ok, "newton_tree twisted", m);
    }
    ev::twisted_from = std::size_t(1) << fps::ntt::kMaxLog;
}

}  // namespace

int main() {
    ev::workspace_exact = true;
    test_tables();
    test_structs<fps::simd::Scalar>("Scalar");
    test_structs<fps::product_tree::Lanes>("Lanes");
#if defined(__SSE4_1__)
    test_structs<fps::simd::Sse>("Sse");
#endif
#if defined(__AVX2__)
    test_structs<fps::simd::Avx2>("Avx2");
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
    test_structs<fps::simd::Neon>("Neon");
#endif
    test_passes<fps::simd::Scalar>("Scalar");
    test_passes<fps::product_tree::Wide>("Wide");
    test_twisted();
    test_trees_twisted();
    std::printf("%s evaluation top: %d checks, %d failures\n", failures ? "FAIL" : "pass", checks, failures);
    return failures ? 1 : 0;
}
