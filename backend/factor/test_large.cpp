// Tests of backend/factor's paths for large degrees (#200), on every simd struct the target runs:
// products modulo g by NTT (large.hpp: Products, Multiplier, Powers) against plain arithmetic at
// degrees around powers of two; the ring on NTT moduli (a Ring with NTT products from degree 2:
// multiply_mod, power_of_x, power_of_linear); the blocked elimination with 32 and 64 columns,
// short tiles, the columns copied out or read in place (eliminate_blocks), by the kernel's
// dimension and A v = 0; factorizations with the NTT path from degree 9 (Factorizer<S, Collide,
// 9>) against planted factors and against the table path, and at the defaults on planted products
// of degree 300 to 700. tools/kernel.py builds it at -O2 and with ASan/UBSan.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <span>
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
std::mt19937_64 rng(200);

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
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] == 0) continue;
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = add(c[i + j], mul(a[i], b[j]));
    }
    trim(c);
    return c;
}
Vec psub(Vec a, const Vec& b) {
    a.resize(std::max(a.size(), b.size()));
    for (std::size_t i = 0; i < b.size(); ++i) a[i] = sub(a[i], b[i]);
    trim(a);
    return a;
}
// a mod b for b monic (trimmed).
Vec pmod(Vec a, const Vec& b) {
    trim(a);
    const std::size_t m = b.size();
    for (std::size_t i = a.size(); i-- >= m;) {
        const u32 c = a[i];
        if (c)
            for (std::size_t j = 0; j < m; ++j) a[i - m + 1 + j] = sub(a[i - m + 1 + j], mul(c, b[j]));
    }
    trim(a);
    return a;
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
        Vec r = pmod(a, monic(b));
        a = monic(b), b = std::move(r);
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
Vec random_below(int m) {  // degree < m, m words
    Vec a(static_cast<std::size_t>(m));
    for (u32& x : a) x = rng() % 7 == 0 ? P - 1 : rand_word();
    return a;
}
Vec padded(Vec a, std::size_t n) {
    a.resize(n, 0);
    return a;
}

// Rabin (small degrees): x^(P^d) = x mod f and gcd(x^(P^(d/q)) - x, f) = 1 for primes q | d.
bool irreducible(const Vec& f) {
    const int d = int(f.size()) - 1;
    if (d <= 1) return d == 1;
    const Vec x = {0, 1};
    std::vector<Vec> h(std::size_t(d) + 1);
    h[0] = x;
    for (int i = 1; i <= d; ++i) h[i] = powmod(h[i - 1], P, f);
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
// (x + b)^n - 3: irreducible for n whose prime factors divide P - 1 = 2^23 7 17 (3 a generator).
Vec binomial(int n) {
    Vec f = {1};
    const u32 b = rand_word();
    Vec power = {b, 1};
    for (int e = n; e; e >>= 1, power = pmul(power, power))
        if (e & 1) f = pmul(f, power);
    f[0] = sub(f[0], 3);
    return f;
}

using Factors = std::vector<std::pair<Vec, u32>>;

Factors run_plain(const std::vector<fz::Factor>& fs) {
    Factors r;
    for (const auto& f : fs) r.emplace_back(f.coefficients, f.multiplicity);
    return r;
}
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

// ---- Products by NTT ------------------------------------------------------------------------------

void test_products() {
    for (int m : {2, 3, 7, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 300, 511, 512, 513}) {
        for (int trial = 0; trial < 3; ++trial) {
            Vec g = random_monic(m);
            if (trial == 1) std::fill(g.begin(), g.end() - 1, P - 1);
            fd::Products products(g.data(), m);
            const Vec a = random_below(m), b = random_below(m);
            Vec out(static_cast<std::size_t>(m));
            const std::string at = " m " + std::to_string(m);
            products.multiply(a.data(), b.data(), out.data());
            check(out == padded(pmod(pmul(a, b), g), std::size_t(m)), "Products multiply" + at);
            products.square(a.data(), out.data());
            check(out == padded(pmod(pmul(a, a), g), std::size_t(m)), "Products square" + at);
            Vec in_place = a;
            products.multiply(in_place.data(), b.data(), in_place.data());
            check(in_place == padded(pmod(pmul(a, b), g), std::size_t(m)), "Products multiply in place" + at);
            // Steps by r: q, q r, q r^2, q r^3 (the last in place).
            const Vec r = trial == 2 ? padded(powmod({0, 1}, P, g), std::size_t(m)) : random_below(m);
            fd::Multiplier X(products, r.data());
            Vec q = a, expect = a;
            for (int step = 0; step < 3; ++step) {
                Vec next(static_cast<std::size_t>(m));
                if (step < 2) X.step(q.data(), next.data());
                else X.step(q.data(), q.data()), next = q;
                expect = padded(pmod(pmul(expect, r), g), std::size_t(m));
                check(next == expect, "Multiplier step " + std::to_string(step) + at);
                q = next;
            }
            // Powers: q r^t for t = 1 .. 8 from one transform of q, rows at a stride, the last apart.
            fd::Powers Y(products, r.data());
            const std::size_t stride = std::size_t(m) + 5;
            Vec rows(7 * stride, 0), last(static_cast<std::size_t>(m)), power = a;
            Y.steps(a.data(), rows.data(), std::ptrdiff_t(stride), last.data());
            bool ok = true;
            for (int t = 1; t <= 8; ++t) {
                power = padded(pmod(pmul(power, r), g), std::size_t(m));
                const u32* got = t < 8 ? rows.data() + std::size_t(t - 1) * stride : last.data();
                ok &= std::equal(power.begin(), power.end(), got);
            }
            check(ok, "Powers steps" + at);
        }
    }
}

// ---- The ring on NTT moduli -----------------------------------------------------------------------

template <class S>
fd::Poly to_poly(fd::Ring<S>& R, const Vec& a) {
    fd::Poly p = R.zero();
    for (std::size_t i = 0; i < a.size(); ++i) p.c[i] = fd::to_montgomery(a[i]);
    p.set_size(int(a.size()));
    return p;
}
Vec to_plain(const fd::Poly& p) {
    Vec a(std::size_t(p.n));
    for (int i = 0; i < p.n; ++i) a[std::size_t(i)] = fd::from_montgomery(p.c[i]);
    return a;
}

template <class S>
void test_ring() {
    const std::string name = S::name;
    fd::Ring<S> R(301, 2);
    for (int trial = 0; trial < 40; ++trial) {
        const int m = trial < 20 ? 2 + trial : 2 + int(rng() % 299);
        const Vec g = random_monic(m);
        const fd::Modulus M = R.modulus(to_poly(R, g));
        const std::string at = " m " + std::to_string(m);
        check(M.ntt != nullptr && M.table.empty(), name + " NTT modulus" + at);
        Vec u = pmod(random_poly(m - 1), g), v = pmod(random_poly(int(rng() % m)), g);
        check(to_plain(R.multiply_mod(to_poly(R, u), to_poly(R, v), M)) == pmod(pmul(u, v), g),
              name + " NTT multiply_mod" + at);
        check(to_plain(R.multiply_mod(to_poly(R, u), to_poly(R, u), M)) == pmod(pmul(u, u), g),
              name + " NTT multiply_mod square" + at);
        const u64 e = trial % 3 ? (P - 1) / 2 : 1 + rng() % 1000;
        check(to_plain(R.power_of_x(e, M)) == powmod({0, 1}, e, g), name + " NTT power_of_x" + at);
        const u32 c = rand_word();
        check(to_plain(R.power_of_linear(fd::to_montgomery(c), e, M)) == powmod({c, 1}, e, g),
              name + " NTT power_of_linear" + at);
    }
}

// ---- Wide elimination -----------------------------------------------------------------------------

// rows x cols of rank r (with high probability): r random rows, the others sums of two earlier
// ones, some columns zero; shuffled.
std::vector<Vec> low_rank(int rows, int cols, int r) {
    std::vector<Vec> a(std::size_t(rows), Vec(std::size_t(cols), 0));
    std::vector<char> zero(std::size_t(cols), 0);
    for (int j = 0; j < cols; ++j) zero[j] = rng() % 9 == 0;
    for (int i = 0; i < rows; ++i) {
        if (i < r) {
            for (int j = 0; j < cols; ++j) a[i][j] = zero[j] ? 0 : rand_word();
        } else if (r > 0) {  // the sum of two earlier rows
            const std::size_t x = rng() % std::size_t(i), y = rng() % std::size_t(i);
            for (int j = 0; j < cols; ++j) a[i][j] = add(a[x][j], a[y][j]);
        }
    }
    std::shuffle(a.begin(), a.end(), rng);
    return a;
}

template <class S, int kColumns, bool kGather = true>
void check_elimination(const std::string& name, const std::vector<Vec>& plain, int cols, int tile, int expect_dim) {
    const int rows = int(plain.size());
    fd::Matrix A(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) A.row(i)[j] = fd::to_montgomery(plain[i][j]);
    const fd::Echelon E =
        kColumns ? fd::eliminate_blocks<S, (kColumns ? kColumns : 16), kGather>(A, tile) : fd::eliminate<S>(A);
    const int k = E.dimension();
    const std::string at = " " + std::to_string(rows) + " x " + std::to_string(cols) + ", blocks of " +
                           std::to_string(kColumns) + ", tile " + std::to_string(tile) + (kGather ? "" : ", in place");
    check(k == expect_dim, name + " wide kernel dimension" + at);
    if (k == 0) return;
    const int count = 2, stride = fd::round_up(cols, 16);
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
    check(ok, name + " wide kernel vectors" + at);
}

template <class S>
void test_elimination() {
    const std::string name = S::name;
    for (int trial = 0; trial < 24; ++trial) {
        const int rows = 1 + int(rng() % 200), cols = 1 + int(rng() % 200);
        const int r = int(rng() % (std::min(rows, cols) + 1));
        const std::vector<Vec> a = low_rank(rows, cols, r);
        int nonzero = 0;  // columns not all zero bound the rank too
        for (int j = 0; j < cols; ++j) {
            bool any = false;
            for (int i = 0; i < rows; ++i) any |= a[i][j] != 0;
            nonzero += any;
        }
        const int dim = cols - std::min(r, nonzero);
        check_elimination<S, 64>(name, a, cols, 16 * (1 + trial % 3), dim);
        check_elimination<S, 32>(name, a, cols, 32, dim);
        check_elimination<S, 32, false>(name, a, cols, 48, dim);
    }
    {  // Longer rows than tiles, rows and columns past 256.
        const int rows = 300, cols = 333, r = 290;
        const std::vector<Vec> a = low_rank(rows, cols, r);
        int nonzero = 0;
        for (int j = 0; j < cols; ++j) {
            bool any = false;
            for (int i = 0; i < rows; ++i) any |= a[i][j] != 0;
            nonzero += any;
        }
        check_elimination<S, 64>(name, a, cols, 64, cols - std::min(r, nonzero));
        check_elimination<S, 0>(name, a, cols, 0, cols - std::min(r, nonzero));
    }
}

// ---- Factorizations -------------------------------------------------------------------------------

template <class S, bool Collide>
Factors factor_ntt(Vec f) {
    trim(f);
    if (f.size() <= 1) return {};
    return run_plain(fd::Factorizer<S, Collide, 9>(int(f.size()) - 1).run(f));
}
template <class S>
Factors factor_table(Vec f) {
    trim(f);
    if (f.size() <= 1) return {};
    return run_plain(fd::Factorizer<S, false, 1 << 30>(int(f.size()) - 1).run(f));  // no NTT moduli
}

template <class S>
void test_factor() {
    const std::string name = S::name;
    std::vector<Vec> pool = {{0, 1}};
    for (int i = 0; i < 10; ++i) pool.push_back({rand_word(), 1});
    for (int d = 2; d <= 12; ++d) pool.push_back(random_irreducible(d));
    for (int n : {14, 16, 28, 32, 49}) pool.push_back(binomial(n));
    // Planted products to degree 160 with the NTT path from degree 9, and the repair paths.
    for (int trial = 0; trial < 24; ++trial) {
        Factors parts;
        int degree = 0;
        const int target = 10 + int(rng() % 150);
        while (degree < target) {
            const Vec& g = pool[rng() % pool.size()];
            const u32 e = 1 + u32(rng() % (trial % 3 == 0 ? 3 : 1));
            if (degree + int(g.size() - 1) * int(e) > target + 20) break;
            parts.emplace_back(g, e);
            degree += int(g.size() - 1) * int(e);
        }
        const Vec f = product_of(parts, rand_nonzero());
        const std::string at = ", degree " + std::to_string(degree);
        check(factor_ntt<S, false>(f) == expected(parts), name + " NTT path known product" + at);
        if (trial % 2 == 0) check(factor_ntt<S, true>(f) == expected(parts), name + " NTT path collide" + at);
    }
    // k equal-degree factors (the split's powers by NTT).
    for (auto [k, d] : std::vector<std::pair<int, int>>{{44, 2}, {20, 5}, {6, 16}}) {
        Factors parts;
        for (int i = 0; i < k; ++i) parts.emplace_back(d == 16 ? binomial(16) : random_irreducible(d), 1);
        const Vec f = product_of(parts, 1);
        check(factor_ntt<S, false>(f) == expected(parts), name + " NTT path equal degrees " + std::to_string(d));
        check(factor_ntt<S, true>(f) == expected(parts), name + " NTT path collide equal degrees " + std::to_string(d));
    }
    // Random f: the NTT path and the table path give the same list.
    for (int d : {9, 10, 31, 64, 65, 100, 128, 129, 200}) {
        const Vec f = random_poly(d);
        check(factor_ntt<S, false>(f) == factor_table<S>(f), name + " NTT and table paths, random degree " + std::to_string(d));
    }
    // Defaults (NTT products from m = 120 to 220): (x + b)^512 - 3 (Berlekamp at m = 512 with k = 1 alone), with smaller
    // irreducibles, roots and squares.
    {
        Factors parts = {{binomial(512), 1}};
        check(run_plain(fz::factor_as<S>(product_of(parts, 3))) == expected(parts), name + " irreducible of degree 512");
        parts.emplace_back(binomial(112), 1);
        for (int d : {2, 3, 5, 7, 11, 12}) parts.emplace_back(random_irreducible(d), 1);
        for (int i = 0; i < 20; ++i) parts.emplace_back(Vec{rand_word(), 1}, 1);
        parts.emplace_back(random_irreducible(4), 2);
        parts.emplace_back(Vec{rand_word(), 1}, 3);
        const Vec f = product_of(parts, rand_nonzero());
        check(run_plain(fz::factor_as<S>(f)) == expected(parts), name + " planted product of degree " + std::to_string(f.size() - 1));
    }
    {  // 320 linear factors (roots on an NTT modulus) and 150 quadratics (m = 300: powers by NTT).
        Factors parts;
        for (int i = 0; i < 320; ++i) parts.emplace_back(Vec{rand_word(), 1}, 1);
        check(run_plain(fz::factor_as<S>(product_of(parts, 1))) == expected(parts), name + " 320 linear factors");
        parts.clear();
        for (int i = 0; i < 150; ++i) {
            u32 z = rand_nonzero();
            while (pw(z, (P - 1) / 2) == 1) z = rand_nonzero();
            parts.emplace_back(Vec{sub(0, z), rand_word(), 1}, 1);
            Vec& q = parts.back().first;  // x^2 + b x + c irreducible: b^2 - 4c a non-square
            q[0] = mul(sub(mul(q[1], q[1]), z), inv(4));
        }
        check(run_plain(fz::factor_as<S>(product_of(parts, 1))) == expected(parts), name + " 150 quadratics");
    }
}

}  // namespace

int main() {
    test_products();
    std::printf("products: %d checks, %d failures so far\n", checks, failures);
    std::fflush(stdout);
    fps::simd::for_each(fps::simd::All{}, []<class S> {
        test_ring<S>();
        test_elimination<S>();
        test_factor<S>();
        std::printf("%s: %d checks, %d failures so far\n", S::name, checks, failures);
        std::fflush(stdout);
    });
    std::printf("%s: %d checks, %d failures\n", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
