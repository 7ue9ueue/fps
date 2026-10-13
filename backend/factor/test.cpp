// Tests of backend/factor, on every simd struct the target runs: the dense kernels (combine,
// lincomb, multiply, transposes) against scalar Montgomery products, including sums of P - 1 at
// every fold; the ring (gcd, products modulo g on the table and in registers, powers, square roots)
// against plain arithmetic; the elimination (kernel dimension against the rank, A v = 0); and the
// factorization: products of known irreducibles with multiplicities against the exact sorted list,
// random f (and lc-opt's test shapes: equal degrees, consecutive degrees, all linear, irreducible
// binomials, x^n - 1) by Rabin's irreducibility test and the product, the repair paths of the
// split forced (Factorizer<S, true>), and the argument checks. tools/kernel.py builds it at -O2
// and with ASan/UBSan.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backend/factor/factor.hpp"
#include "backend/simd/select.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
namespace fz = fps::factorization;
namespace fd = fz::detail;
constexpr u32 P = fz::kP;

int failures = 0, checks = 0;
std::mt19937_64 rng(123);

void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok && ++failures <= 20) std::printf("FAIL %s\n", what.c_str()), std::fflush(stdout);
}

// ---- Plain arithmetic (coefficients in [0, P), trimmed vectors, {} = 0) -------------------------

u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 add(u32 a, u32 b) { return u32((u64(a) + b) % P); }
u32 sub(u32 a, u32 b) { return a >= b ? a - b : a + P - b; }
u32 pw(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}
u32 inv(u32 a) { return pw(a, P - 2); }
u32 rand_word() { return u32(rng() % P); }
u32 rand_nonzero() { return 1 + u32(rng() % (P - 1)); }

void trim(Vec& a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
}
Vec pmul(const Vec& a, const Vec& b) {
    if (a.empty() || b.empty()) return {};
    Vec c(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = add(c[i + j], mul(a[i], b[j]));
    trim(c);
    return c;
}
Vec psub(Vec a, const Vec& b) {
    a.resize(std::max(a.size(), b.size()));
    for (std::size_t i = 0; i < b.size(); ++i) a[i] = sub(a[i], b[i]);
    trim(a);
    return a;
}
// a = q b + r (b nonzero, trimmed).
void pdivmod(Vec a, const Vec& b, Vec* q, Vec* r) {
    trim(a);
    const std::size_t m = b.size();
    const u32 il = inv(b.back());
    Vec quot(a.size() >= m ? a.size() - m + 1 : 0);
    for (std::size_t i = a.size(); i-- >= m;) {
        const u32 c = mul(a[i], il);
        quot[i - m + 1] = c;
        if (c)
            for (std::size_t j = 0; j < m; ++j) a[i - m + 1 + j] = sub(a[i - m + 1 + j], mul(c, b[j]));
    }
    trim(a), trim(quot);
    if (q) *q = quot;
    if (r) *r = a;
}
Vec pmod(const Vec& a, const Vec& b) {
    Vec r;
    pdivmod(a, b, nullptr, &r);
    return r;
}
Vec monic(Vec a) {
    trim(a);
    if (a.empty()) return a;
    const u32 il = inv(a.back());
    for (u32& x : a) x = mul(x, il);
    return a;
}
Vec pgcd(Vec a, Vec b) {
    trim(a), trim(b);
    while (!b.empty()) {
        Vec r = pmod(a, b);
        a = std::move(b), b = std::move(r);
    }
    return monic(a);
}
Vec powmod(Vec base, u64 e, const Vec& m) {
    Vec r = pmod({1}, m);
    base = pmod(base, m);
    for (; e; e >>= 1, base = pmod(pmul(base, base), m))
        if (e & 1) r = pmod(pmul(r, base), m);
    return r;
}
Vec random_poly(int degree) {  // exactly this degree
    Vec a(std::size_t(degree) + 1);
    for (u32& x : a) x = rand_word();
    if (a.back() == 0) a.back() = 1;
    return a;
}
Vec random_monic(int degree) {
    Vec a = random_poly(degree);
    a.back() = 1;
    return a;
}

// Rabin: f monic of degree d is irreducible iff x^(P^d) = x mod f and gcd(x^(P^(d/q)) - x, f) = 1
// for every prime q | d. x^(P^i) by the matrix of the Frobenius (rows x^(P j) mod f).
bool irreducible(const Vec& f) {
    const int d = int(f.size()) - 1;
    if (d <= 1) return d == 1;
    const Vec x = {0, 1}, xp = powmod(x, P, f);
    std::vector<Vec> rows(static_cast<std::size_t>(d));
    rows[0] = {1};
    for (int j = 1; j < d; ++j) rows[j] = pmod(pmul(rows[std::size_t(j) - 1], xp), f);
    auto frobenius = [&](const Vec& h) {
        Vec r(std::size_t(d), 0);
        for (std::size_t j = 0; j < h.size(); ++j)
            for (std::size_t i = 0; i < rows[j].size(); ++i) r[i] = add(r[i], mul(h[j], rows[j][i]));
        trim(r);
        return r;
    };
    std::vector<Vec> h(std::size_t(d) + 1);
    h[0] = x;
    for (int i = 1; i <= d; ++i) h[i] = frobenius(h[std::size_t(i) - 1]);
    if (h[d] != x) return false;
    for (int q = 2, rest = d; q <= rest; ++q) {
        if (rest % q) continue;
        while (rest % q == 0) rest /= q;
        if (pgcd(psub(h[d / q], x), f).size() > 1) return false;
    }
    return true;
}

Vec random_irreducible(int degree) {
    for (;;) {
        Vec f = random_monic(degree);
        if (irreducible(f)) return f;
    }
}

// ---- Kernels ------------------------------------------------------------------------------------

template <class S>
void test_kernels() {
    const std::string name = S::name;
    constexpr int W = S::lanes;
    for (int trial = 0; trial < 300; ++trial) {
        const int len = trial < 80 ? 1 + trial : 1 + int(rng() % 140);
        const int count = std::vector<int>{0, 1, 2, 16, 17, 18, 34, 35, 52, 100}[trial % 10];
        const bool extreme = trial % 7 == 3, with_init = trial % 3 != 0;
        const int stride = fd::round_up(len, 16) + 16 * int(trial % 2);
        auto word = [&] { return extreme ? P - 1 : rand_word(); };
        Vec base(std::size_t(std::max(count, 1)) * stride + 16, 0), init(std::size_t(stride) + 16, 0), coef(std::size_t(count) + 1);
        for (int s = 0; s < count; ++s)
            for (int t = 0; t < len; ++t) base[std::size_t(s) * stride + t] = word();
        for (int t = 0; t < len; ++t) init[t] = word();
        for (u32& c : coef) c = word();
        const u32 scale = word();
        Vec out(std::size_t(stride) + 16, 12345);
        fz::combine_scaled<S>(out.data(), len, with_init ? init.data() : nullptr, scale, coef.data(), count, base.data(), stride);
        bool ok = true;
        for (int t = 0; t < fd::round_up(len, W); ++t) {
            u32 e = with_init ? fd::mul(scale, init[t]) : 0;
            for (int s = 0; s < count; ++s) e = fd::add(e, fd::mul(coef[s], base[std::size_t(s) * stride + t]));
            ok &= out[t] == e;
        }
        check(ok, name + " combine_scaled len " + std::to_string(len) + " count " + std::to_string(count));
        // lincomb, in place.
        Vec x(std::size_t(stride) + 16, 0), y(std::size_t(stride) + 16, 0);
        for (int t = 0; t < len; ++t) x[t] = word(), y[t] = word();
        const u32 a = word(), b = word();
        Vec expect(std::size_t(fd::round_up(len, W)));
        for (std::size_t t = 0; t < expect.size(); ++t) expect[t] = fd::add(fd::mul(a, x[t]), fd::mul(b, y[t]));
        Vec x0 = x;
        fz::lincomb<S>(x.data(), x.data(), a, y.data(), b, len);
        check(std::equal(expect.begin(), expect.end(), x.begin()), name + " lincomb len " + std::to_string(len));
        // axpy, in place.
        for (std::size_t t = 0; t < expect.size(); ++t) expect[t] = fd::add(x0[t], fd::mul(b, y[t]));
        fz::axpy<S>(x0.data(), x0.data(), b, y.data(), len);
        check(std::equal(expect.begin(), expect.end(), x0.begin()), name + " axpy len " + std::to_string(len));
    }
    // times_x: y = x a mod g, g monic of degree m.
    for (int m = 1; m <= 70; ++m) {
        Vec a(std::size_t(fd::round_up(m, 16)) + 16, 0), g(a.size(), 0), y(a.size(), 4242);
        for (int t = 0; t < m; ++t) a[t] = m % 5 == 0 ? P - 1 : rand_word(), g[t] = rand_word();
        g[m] = fd::kOne;
        fz::times_x<S>(y.data(), a.data(), g.data(), m);
        bool ok = true;
        for (int t = 0; t < fd::round_up(m, W); ++t) {
            const u32 e = fd::sub(t >= 1 && t <= m ? a[t - 1] : 0, fd::mul(a[m - 1], g[t]));
            ok &= y[t] == e;
        }
        check(ok, name + " times_x m " + std::to_string(m));
    }
    // combine_rows: 1 to 9 rows, in place or from other rows, init null or not.
    for (int trial = 0; trial < 200; ++trial) {
        const int rows = 1 + trial % 9, len = 1 + int(rng() % 140);
        const int count = std::vector<int>{0, 1, 5, 16, 17, 18, 40, 100}[trial % 8];
        const bool extreme = trial % 7 == 3, in_place = trial % 3 == 1, with_init = trial % 3 != 0;
        const int stride = fd::round_up(len, 16) + 16 * int(trial % 2);
        // Coefficients by rows (cstride, step 1) or by columns (step sstride, rows 1 apart).
        const bool by_columns = trial % 4 == 2;
        const int cstride = by_columns ? 1 : count + int(trial % 5), sstride = by_columns ? rows + int(trial % 3) : 1;
        auto word = [&] { return extreme ? P - 1 : rand_word(); };
        Vec base(std::size_t(std::max(count, 1)) * stride + 16, 0), coef(std::size_t(rows) * cstride + std::size_t(count) * sstride + 1, 0);
        Vec init(std::size_t(rows) * stride + 16, 0), out(std::size_t(rows) * stride + 16, 999);
        for (int s = 0; s < count; ++s)
            for (int t = 0; t < len; ++t) base[std::size_t(s) * stride + t] = word();
        for (int r = 0; r < rows; ++r) {
            for (int t = 0; t < len; ++t) init[std::size_t(r) * stride + t] = word();
            for (int s = 0; s < count; ++s) coef[std::size_t(r) * cstride + std::size_t(s) * sstride] = word();
        }
        const u32 scale = word();
        const Vec init0 = init;
        u32* target = in_place ? init.data() : out.data();
        fz::combine_rows<S>(target, stride, rows, len, with_init ? init.data() : nullptr, stride, scale, coef.data(), cstride,
                            sstride, count, base.data(), stride);
        bool ok = true;
        for (int r = 0; r < rows; ++r)
            for (int t = 0; t < fd::round_up(len, W); ++t) {
                u32 e = with_init ? fd::mul(scale, init0[std::size_t(r) * stride + t]) : 0;
                for (int s = 0; s < count; ++s)
                    e = fd::add(e, fd::mul(coef[std::size_t(r) * cstride + std::size_t(s) * sstride], base[std::size_t(s) * stride + t]));
                ok &= target[std::size_t(r) * stride + t] == e;
            }
        check(ok, name + " combine_rows rows " + std::to_string(rows) + " len " + std::to_string(len) + " count " +
                      std::to_string(count));
    }
    // Products: every size pair to 40, then random ones to 300.
    for (int trial = 0; trial < 1800; ++trial) {
        const int na = trial < 1600 ? 1 + trial % 40 : 1 + int(rng() % 300);
        const int nb = trial < 1600 ? 1 + trial / 40 : 1 + int(rng() % 300);
        const bool extreme = trial % 5 == 4;
        Vec a(static_cast<std::size_t>(na)), b(static_cast<std::size_t>(nb));
        for (u32& w : a) w = extreme ? P - 1 : rand_word();
        for (u32& w : b) w = extreme ? P - 1 : rand_word();
        const int len = na + nb - 1;
        Vec out(std::size_t(fd::round_up(len, 16)) + 16, 777), scratch(std::size_t(fz::multiply_scratch<S>(nb)));
        fz::multiply<S>(a.data(), na, b.data(), nb, out.data(), scratch.data());
        bool ok = true;
        for (int t = 0; t < fd::round_up(len, W); ++t) {
            u32 e = 0;
            for (int i = std::max(0, t - nb + 1); i <= std::min(t, na - 1); ++i) e = fd::add(e, fd::mul(a[i], b[t - i]));
            ok &= out[t] == e;
        }
        check(ok, name + " multiply " + std::to_string(na) + " x " + std::to_string(nb));
    }
    // Transposes of 8 rows.
    for (int rows : {1, 7, 8, 9, 16, 33, 100}) {
        const int stride = fd::round_up(rows, 16), ld = fd::round_up(rows + 8, 16);
        Vec src(std::size_t(8) * stride), dst(std::size_t(rows) * ld, 0);
        for (u32& w : src) w = rand_word();
        fz::transpose_rows<S>(dst.data(), ld, rows, 8, src.data(), stride);
        bool ok = true;
        for (int i = 0; i < rows; ++i)
            for (int q = 0; q < 8; ++q) ok &= dst[std::size_t(i) * ld + 8 + q] == src[std::size_t(q) * stride + i];
        check(ok, name + " transpose_rows " + std::to_string(rows));
    }
}

// ---- Ring ---------------------------------------------------------------------------------------

Vec to_plain(const fd::Poly& a) {
    Vec r(std::size_t(a.n));
    for (int i = 0; i < a.n; ++i) r[i] = fd::from_montgomery(a.c[i]);
    return r;
}
template <class S>
fd::Poly to_poly(fd::Ring<S>& R, const Vec& a) {
    fd::Poly p = R.zero();
    for (std::size_t i = 0; i < a.size(); ++i) p.c[i] = fd::to_montgomery(a[i]);
    p.set_size(int(a.size()));
    return p;
}

template <class S>
void test_ring() {
    const std::string name = S::name;
    for (int limit : {9, 40, 130, 300}) {
        fd::Ring<S> R(limit);
        for (int trial = 0; trial < 60; ++trial) {
            const int m = 1 + int(rng() % std::min(limit - 1, trial < 30 ? 9 : limit - 1));
            const Vec g = random_monic(m);
            // gcd with a planted common factor.
            const int dc = int(rng() % (m + 1));
            const Vec common = random_poly(dc);
            const int da = int(rng() % std::max(1, limit - dc)), db = int(rng() % std::max(1, limit - dc));
            Vec a = pmul(common, random_poly(da)), b = pmul(common, random_poly(db));
            if (trial % 9 == 0) b = {};
            if (int(a.size()) <= limit && int(b.size()) <= limit) {
                const Vec expect = pgcd(a, b);
                check(to_plain(R.gcd(to_poly(R, a), to_poly(R, b))) == expect, name + " gcd");
                check(to_plain(R.gcd(to_poly(R, b), to_poly(R, a))) == expect, name + " gcd swapped");
                Vec q, r;
                pdivmod(a, g, &q, &r);
                fd::Poly fq = R.zero(), fr = R.zero();
                R.divide(to_poly(R, a), to_poly(R, g), &fq, &fr);
                check(to_plain(fq) == q && to_plain(fr) == r, name + " divide");
            }
            if (m < 2) continue;
            const fd::Modulus M = R.modulus(to_poly(R, g));
            const Vec u = pmod(random_poly(m - 1), g), v = pmod(random_poly(int(rng() % m)), g);
            check(to_plain(R.multiply_mod(to_poly(R, u), to_poly(R, v), M)) == pmod(pmul(u, v), g),
                  name + " multiply_mod m " + std::to_string(m));
            const u64 e = trial % 3 ? (P - 1) / 2 : 1 + rng() % 1000;
            check(to_plain(R.power_of_x(e, M)) == powmod({0, 1}, e, g), name + " power_of_x m " + std::to_string(m));
            const u32 c = rand_word();
            check(to_plain(R.power_of_linear(fd::to_montgomery(c), e, M)) == powmod({c, 1}, e, g),
                  name + " power_of_linear m " + std::to_string(m));
            if (2 * m <= limit) {
                const Vec h = random_monic(1 + int(rng() % m));
                check(to_plain(R.multiply_remainder(to_poly(R, u), to_poly(R, v), to_poly(R, h))) == pmod(pmul(u, v), h),
                      name + " multiply_remainder");
            }
        }
        // Pool: Polys taken and given back in any order come back zeroed.
        std::vector<fd::Poly> held;
        for (int i = 0; i < 50; ++i) {
            fd::Poly p = R.zero();
            bool clean = true;
            for (int j = 0; j < R.pool().cap(); ++j) clean &= p.c[j] == 0;
            check(clean, name + " pool buffer zeroed");
            const int n = 1 + int(rng() % (limit - 1));
            for (int j = 0; j < n; ++j) p.c[j] = 1 + u32(j);
            p.n = n;
            if (i % 3) held.push_back(std::move(p));
            if (i % 7 == 0 && !held.empty()) held.erase(held.begin() + std::ptrdiff_t(rng() % held.size()));
        }
    }
    for (int i = 0; i < 200; ++i) {
        const u32 r = rand_word(), a = mul(r, r);
        const u32 s = fd::from_montgomery(fd::Ring<S>::sqrt(fd::to_montgomery(a)));
        check(mul(s, s) == a, name + " sqrt");
    }
}

// ---- Elimination --------------------------------------------------------------------------------

int rank_of(std::vector<Vec> a, int cols) {
    int rank = 0;
    for (int c = 0; c < cols && rank < int(a.size()); ++c) {
        int p = rank;
        while (p < int(a.size()) && a[p][c] == 0) ++p;
        if (p == int(a.size())) continue;
        std::swap(a[p], a[rank]);
        const u32 il = inv(a[rank][c]);
        for (int i = 0; i < int(a.size()); ++i) {
            if (i == rank || a[i][c] == 0) continue;
            const u32 f = mul(a[i][c], il);
            for (int j = c; j < cols; ++j)
                a[i][j] = sub(a[i][j], mul(f, a[rank][j]));
        }
        ++rank;
    }
    return rank;
}

template <class S>
void test_elimination() {
    const std::string name = S::name;
    for (int trial = 0; trial < 60; ++trial) {
        const int rows = 1 + int(rng() % (trial < 40 ? 40 : 130)), cols = 1 + int(rng() % (trial < 40 ? 40 : 130));
        const int r = int(rng() % (std::min(rows, cols) + 1));
        // A = B C (rank <= r), some columns zeroed, some rows repeated.
        std::vector<Vec> B(rows, Vec(r, 0)), C(r, Vec(cols, 0));
        for (auto& row : B)
            for (u32& w : row) w = rand_word();
        for (auto& row : C)
            for (u32& w : row) w = trial % 4 == 0 && rng() % 3 == 0 ? 0 : rand_word();
        std::vector<Vec> plain(rows, Vec(cols, 0));
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                for (int t = 0; t < r; ++t) plain[i][j] = add(plain[i][j], mul(B[i][t], C[t][j]));
        fd::Matrix A(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) A.row(i)[j] = fd::to_montgomery(plain[i][j]);
        const fd::Echelon E = fd::eliminate<S>(A);
        const int k = E.dimension();
        check(k == cols - rank_of(plain, cols), name + " kernel dimension");
        if (k == 0) continue;
        const int count = 3, stride = fd::round_up(cols, 16);
        Vec values(std::size_t(count) * std::size_t(k)), out(std::size_t(count) * std::size_t(stride));
        for (u32& w : values) w = fd::to_montgomery(rand_word());
        E.solve<S>(A, values.data(), count, out.data(), stride);
        bool ok = true;
        for (int j = 0; j < count; ++j) {
            const u32* v = out.data() + std::size_t(j) * stride;
            for (int i = 0; i < k; ++i) ok &= v[E.free[i]] == values[j * k + i];
            for (int i = 0; i < rows; ++i) {
                u32 s = 0;
                for (int c = 0; c < cols; ++c) s = add(s, mul(plain[i][c], fd::from_montgomery(v[c])));
                ok &= s == 0;
            }
        }
        check(ok, name + " kernel vectors " + std::to_string(rows) + " x " + std::to_string(cols));
    }
}

// ---- Factorization ------------------------------------------------------------------------------

using Factors = std::vector<std::pair<Vec, u32>>;

Factors run_plain(const std::vector<fz::Factor>& fs) {
    Factors r;
    for (const auto& f : fs) r.emplace_back(f.coefficients, f.multiplicity);
    return r;
}

// fs is the factorization of f: monic irreducible distinct factors, sorted, product = monic(f).
bool valid(const Vec& f0, const Factors& fs) {
    Vec f = monic(f0), product = {1};
    for (std::size_t i = 0; i < fs.size(); ++i) {
        const auto& [g, e] = fs[i];
        if (g.size() < 2 || g.back() != 1 || e == 0 || !irreducible(g)) return false;
        if (i > 0) {
            const Vec& h = fs[i - 1].first;
            if (h.size() > g.size() || (h.size() == g.size() && !std::ranges::lexicographical_compare(h, g))) return false;
        }
        for (u32 j = 0; j < e; ++j) product = pmul(product, g);
    }
    return product == f;
}

// Expected factorization of the product of factors^multiplicities: equal factors merged, sorted.
Factors expected(const Factors& parts) {
    std::map<std::pair<std::size_t, Vec>, u32> merged;
    for (const auto& [g, e] : parts) merged[{g.size(), g}] += e;
    Factors r;
    for (const auto& [key, e] : merged) r.emplace_back(key.second, e);
    return r;
}

Vec product_of(const Factors& parts, u32 lead) {
    Vec f = {lead};
    for (const auto& [g, e] : parts)
        for (u32 j = 0; j < e; ++j) f = pmul(f, g);
    return f;
}

template <class S, bool Collide = false>
Factors factor_with(const Vec& f) {
    if constexpr (Collide) {
        Vec g = f;
        trim(g);
        if (g.size() <= 1) return {};
        return run_plain(fd::Factorizer<S, true>(int(g.size()) - 1).run(g));
    } else {
        return run_plain(fz::factor_as<S>(f));
    }
}

template <class S, bool Collide>
void test_known(const std::string& name, const std::vector<Vec>& pool, int cases, int max_degree) {
    for (int trial = 0; trial < cases; ++trial) {
        Factors parts;
        int degree = 0;
        const int target = 1 + int(rng() % max_degree);
        while (degree < target) {
            const Vec& g = pool[rng() % pool.size()];
            const u32 e = 1 + u32(rng() % (trial % 3 == 0 ? 4 : 1));
            if (degree + int(g.size() - 1) * int(e) > max_degree) break;
            parts.emplace_back(g, e);
            degree += int(g.size() - 1) * int(e);
        }
        Vec f = product_of(parts, rand_nonzero());
        if (trial % 5 == 0) f.resize(f.size() + 3, 0);  // trailing zeros
        const Factors got = factor_with<S, Collide>(f);
        check(got == expected(parts), name + (Collide ? " collide" : "") + " known product, degree " + std::to_string(degree));
    }
}

template <class S>
void test_factor() {
    const std::string name = S::name;
    // Irreducibles: x, linear, quadratics x^2 - z for a non-square z, random of degree 3 to 12.
    std::vector<Vec> pool = {{0, 1}};
    for (int i = 0; i < 12; ++i) pool.push_back({rand_word(), 1});
    for (int i = 0; i < 6; ++i) {
        u32 z = rand_nonzero();
        while (pw(z, (P - 1) / 2) == 1) z = rand_nonzero();
        pool.push_back({P - z, 0, 1});
    }
    for (int d = 3; d <= 12; ++d) pool.push_back(random_irreducible(d));
    test_known<S, false>(name, pool, 120, 100);
    test_known<S, true>(name, pool, 60, 100);
    // Only degrees 2 and above (no roots): Berlekamp with many factors.
    std::vector<Vec> high(pool.begin() + 13, pool.end());
    test_known<S, false>(name, high, 40, 120);
    test_known<S, true>(name, high, 40, 120);

    // Random f, degrees 0 to 120, some with square factors, then a few long ones.
    for (int trial = 0; trial < 160; ++trial) {
        const int d = trial < 121 ? trial : 121 + int(rng() % 40);
        Vec f = random_poly(d);
        if (trial % 6 == 5 && d <= 60) f = pmul(pmul(f, f), {1, 1});
        if (trial % 7 == 6) f.resize(f.size() + 2, 0);
        check(valid(f, factor_with<S>(f)), name + " random f, degree " + std::to_string(d));
    }
    for (int d : {200, 256, 300}) {
        const Vec f = random_poly(d);
        check(valid(f, factor_with<S>(f)), name + " random f, degree " + std::to_string(d));
    }

    // lc-opt's shapes. Equal degrees: k factors of degree d.
    for (auto [k, d] : std::vector<std::pair<int, int>>{{44, 2}, {33, 3}, {20, 5}, {8, 12}, {2, 50}, {4, 25}}) {
        Factors parts;
        for (int i = 0; i < k; ++i) parts.emplace_back(random_irreducible(d), 1);
        const Factors expect = expected(parts);
        check(factor_with<S>(product_of(parts, 1)) == expect, name + " equal degrees " + std::to_string(d));
        check(factor_with<S, true>(product_of(parts, 1)) == expect, name + " collide equal degrees " + std::to_string(d));
    }
    {  // Consecutive degrees 1 to 13 (91), and 1, 3, 7, 15, 31.
        Factors parts;
        for (int d = 1; d <= 13; ++d) parts.emplace_back(random_irreducible(d), 1);
        check(factor_with<S>(product_of(parts, 1)) == expected(parts), name + " consecutive degrees");
        parts.clear();
        for (int d : {1, 3, 7, 15, 31}) parts.emplace_back(random_irreducible(d), 1);
        check(factor_with<S>(product_of(parts, 1)) == expected(parts), name + " sparse degrees");
    }
    {  // All linear: 100 distinct roots; few distinct roots, repeated; 0 among them.
        Factors parts;
        for (int i = 0; i < 100; ++i) parts.emplace_back(Vec{rand_word(), 1}, 1);
        check(factor_with<S>(product_of(parts, 1)) == expected(parts), name + " 100 linear factors");
        parts.clear();
        for (int i = 0; i < 6; ++i) parts.emplace_back(Vec{i ? rand_word() : 0, 1}, 1 + u32(rng() % 20));
        check(factor_with<S>(product_of(parts, 1)) == expected(parts), name + " repeated roots");
        // Roots r c^i, c = 3^(2^23) (order 119): every 2-power character x^((P-1)/2^k) agrees on
        // them, so the first split (at x, no shift) cannot separate them; and 0 with them.
        for (int count : {2, 3, 9, 40}) {
            parts.clear();
            const u32 c = pw(3, u64(1) << 23), r = rand_nonzero();
            for (int i = 0; i < count; ++i) parts.emplace_back(Vec{P - mul(r, pw(c, u64(i))), 1}, 1);
            if (count == 9) parts.emplace_back(Vec{0, 1}, 1);
            for (const Vec& g : {random_irreducible(3), random_irreducible(5)}) parts.emplace_back(g, 1);
            check(factor_with<S>(product_of(parts, 1)) == expected(parts), name + " roots with equal characters");
        }
    }
    // Irreducible binomials: x^n - 3 for n with every prime factor dividing P - 1 = 2^23 7 17
    // (3 is a primitive root), shifted: (x + b)^n - 3.
    for (int n : {2, 7, 17, 49, 56, 64, 68, 98}) {
        Vec f = {1};
        const u32 b = rand_word();
        for (int i = 0; i < n; ++i) f = pmul(f, {b, 1});
        f[0] = sub(f[0], 3);
        check(factor_with<S>(f) == Factors{{f, 1}}, name + " binomial degree " + std::to_string(n));
        check(factor_with<S>(pmul(f, f)) == Factors{{f, 2}}, name + " binomial squared " + std::to_string(n));
    }
    {  // x^n, x^n - 1 for n | P - 1 (n linear factors), (x - a)^50 (x - b)^49, x^4 + 1.
        for (int n = 1; n <= 9; ++n) {
            Vec f(std::size_t(n) + 1, 0);
            f.back() = 5;
            check(factor_with<S>(f) == Factors{{{0, 1}, u32(n)}}, name + " x^n");
        }
        for (int n : {2, 4, 8, 64, 256}) {
            Vec f(std::size_t(n) + 1, 0);
            f[0] = P - 1, f.back() = 1;
            const Factors got = factor_with<S>(f);
            bool ok = int(got.size()) == n;
            for (const auto& [g, e] : got) ok &= g.size() == 2 && e == 1 && pw(P - g[0], u64(n)) == 1;
            check(ok, name + " x^n - 1, n = " + std::to_string(n));
        }
        Factors parts = {{{rand_word(), 1}, 50}, {{rand_word(), 1}, 49}};
        check(factor_with<S>(product_of(parts, 7)) == expected(parts), name + " high multiplicities");
        check(valid({1, 0, 0, 0, 1}, factor_with<S>({1, 0, 0, 0, 1})), name + " x^4 + 1");
    }
    // Constants, errors.
    check(factor_with<S>({5}).empty() && factor_with<S>({5, 0, 0}).empty(), name + " constant");
    for (const Vec& zero : {Vec{}, Vec{0, 0, 0}}) {
        bool thrown = false;
        try {
            fz::factor_as<S>(zero);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        check(thrown, name + " f = 0 throws");
    }
    {
        Vec f(std::size_t(fz::kMaxDegree) + 2, 0);
        f.back() = 1;
        bool thrown = false;
        try {
            fz::factor_as<S>(f);
        } catch (const std::length_error&) {
            thrown = true;
        }
        check(thrown, name + " deg f > kMaxDegree throws");
    }
}

}  // namespace

int main() {
    fps::simd::for_each(fps::simd::All{}, []<class S> {
        test_kernels<S>();
        test_ring<S>();
        test_elimination<S>();
        test_factor<S>();
        std::printf("%s: %d checks, %d failures so far\n", S::name, checks, failures);
        std::fflush(stdout);
    });
    {  // The API runs the native struct.
        const Vec f = random_poly(80);
        const auto got = fz::factor(f);
        check(valid(f, run_plain(got)), "factor (native)");
    }
    std::printf("%s: %d checks, %d failures\n", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
