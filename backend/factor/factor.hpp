// Factorization of polynomials modulo P = 998244353 into monic irreducibles (backend kernel;
// namespace factorization, as fps::factor is the frontend's function). From lc-opt's
// problems/polynomial/factorization_of_polynomials (its issue #83: Yun, Berlekamp), rewritten on
// backend/simd, so every target runs it. Design and measurements: notes.md.
//
//   std::vector<fps::factorization::Factor> fs = fps::factorization::factor(f);
//   // fs[i].coefficients: b_0 .. b_d of a monic irreducible (b_d = 1), fs[i].multiplicity >= 1
//   fps::factorization::factor_as<S>(f)  // the same on the simd struct S (tests, benches)
//
// f: coefficients a_0 .. a_n in [0, P), trailing zeros allowed; deg f <= kMaxDegree. The leading
// coefficient is dropped: f = lc(f) prod fs[i]^multiplicity. Factors come sorted by degree, then by
// their coefficients from b_0 up (lexicographic), so the result is the same on every target; the
// random choices inside use a fixed seed, so the time is too. A constant f gives no factors.
// f = 0 throws std::invalid_argument, deg f > kMaxDegree std::length_error.
//
// Cost: O(m^3) per square-free part of degree m (the Krylov sequence and the elimination below),
// two m x m matrices of words: about 0.2-0.3 ms at degree 100 on Zen 3, 2 ms at 256 (notes.md).
// Larger degrees want a distinct-degree split by baby steps and giant steps (Kaltofen-Shoup) on
// NTT products instead; not here yet.
//
// Algorithm (lc-opt's):
// - square-free decomposition by Yun's algorithm (deg f < P, so f' = 0 only for a constant);
// - per square-free part g of degree m: s = x^((P-1)/2) mod g by squarings (a product, then the
//   reduction by the table x^(m + j) mod g), x^P = x s^2; the linear factors L = gcd(g, x^P - x) by
//   Cantor-Zassenhaus root finding (s mod L gives the first split);
// - the rest by Berlekamp: rows q_j = x^(P j) mod g by mat-vecs with the matrix of the
//   multiplication by x^P, 8 rows at a time transposed into A = (Q - I)^T; the kernel of A
//   (dimension k = the number of factors) by fraction-free blocked elimination (linear.hpp);
// - k > 1: two random kernel elements v, w. The factors are grouped by the value of v
//   (v = c_i mod g_i): the c_i are the roots of the minimal polynomial mu of v, found from the
//   coordinates of v^j at the free columns (a k x (k + 1) kernel). Groups by divide and conquer
//   over the roots: gcd(h, prod (v - c)), the product from the powers of v near the top, by
//   Horner on v mod h below. All groups are single factors but for probability about k^2 / 2P; a
//   group where w is not constant is factored again, and if the count still falls short of k
//   every group left is.
// - moduli of degree <= 8 on x86 with AVX2: polynomials in one ymm register (poly.hpp), as root
//   finding runs chains of 30 dependent squarings at tiny degrees.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "backend/factor/kernels.hpp"
#include "backend/factor/linear.hpp"
#include "backend/factor/poly.hpp"
#include "backend/simd/select.hpp"

namespace fps::factorization {

inline constexpr int kMaxDegree = 1 << 12;

struct Factor {
    std::vector<u32> coefficients;  // b_0 .. b_d in [0, P), b_d = 1
    u32 multiplicity = 0;
};

namespace detail {

// Collide (tests only): the first sample becomes v^((P-1)/2), so at most 3 groups (values 0, 1,
// -1), and every other step the second sample is the constant 1, so groups of several factors
// pass as single ones: both repair paths of split() run.
template <class S, bool Collide = false>
class Factorizer {
public:
    // Polynomials of degree <= n.
    explicit Factorizer(int n) : R_(n + 1) {}

    // f: n + 1 words, f[n] != 0, n >= 1, in [0, P).
    std::vector<Factor> run(std::span<const u32> f) {
        const int n = int(f.size()) - 1;
        Poly g = R_.zero();
        const u32 scale = mul(inverse(to_montgomery(f[n])), kR2);  // f[n]^-1 2^64: to Montgomery and monic
        for (int i = 0; i <= n; ++i) g.c[i] = mul(f[i], scale);
        g.n = n + 1;
        square_free(g);
        std::ranges::sort(factors_, [](const Factor& a, const Factor& b) {
            if (a.coefficients.size() != b.coefficients.size()) return a.coefficients.size() < b.coefficients.size();
            return std::ranges::lexicographical_compare(a.coefficients, b.coefficients);
        });
        return std::move(factors_);
    }

private:
    static constexpr int kSamples = 2;  // random kernel elements per Berlekamp step

    Ring<S> R_;
    u64 state_ = 0x9E3779B97F4A7C15ull;
    u32 multiplicity_ = 1;
    std::vector<Factor> factors_;
    // Storage for the m x m arrays, reused so their pages fault in once: the reduction table, then
    // the multiplications by x^P and by v (columns_); (Q - I)^T, then the powers of v (matrix_).
    std::vector<u32> columns_, matrix_;

    u32 random_residue() {  // splitmix64
        u64 z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return u32((z ^ (z >> 31)) % kP);
    }

    void emit(const Poly& g) {
        Factor f{std::vector<u32>(g.n), multiplicity_};
        for (int i = 0; i < g.n; ++i) f.coefficients[i] = from_montgomery(g.c[i]);
        factors_.push_back(std::move(f));
    }

    // Yun: f = prod s_i^i with w = prod_{j >= i} s_j and c = prod_{j > i} s_j^(j - i) at step i.
    void square_free(const Poly& f) {
        const Poly df = R_.derivative(f);
        Poly c = R_.gcd(f, df);
        if (c.n == 1) {
            multiplicity_ = 1;
            return factor_square_free(f);
        }
        Poly w = R_.quotient(f, c);
        for (u32 i = 1; w.n > 1; ++i) {
            Poly y = R_.gcd(w, c);
            const Poly z = R_.quotient(w, y);
            if (z.n > 1) {
                multiplicity_ = i;
                factor_square_free(z);
            }
            w = std::move(y);
            c = R_.quotient(c, w);
        }
    }

    void factor_square_free(const Poly& g) {
        if (g.n == 2) return emit(g);
        Modulus M = R_.modulus(g, std::move(columns_));
        split_square_free(M);
        columns_ = std::move(M.table);
    }

    // M.table is scratch afterwards.
    void split_square_free(Modulus& M) {
        Poly g = M.g;
        const Poly s = R_.power_of_x((kP - 1) / 2, M);
        Poly r = R_.times_x(R_.multiply_mod(s, s, M), M);  // x^P mod g
        Poly u = r;
        u.c[1] = sub(u.c[1], kOne);
        u.set_size(std::max(u.n, 2));
        const Poly linear = R_.gcd(g, u);
        if (linear.n > 1) {
            const Poly hint = R_.remainder(s, linear);
            std::vector<u32> roots;
            find_roots(linear, &hint, roots);
            for (const u32 z : roots) emit(R_.linear(z));
            if (linear.n == g.n) return;
            g = R_.quotient(g, linear);
            r = R_.remainder(r, g);
            M = R_.modulus(g, std::move(M.table));
        }
        if (g.n <= 4) return emit(g);  // degree <= 3 without roots
        berlekamp(M, r);
    }

    // Roots of a monic Q that splits into distinct linear factors; hint: x^((P-1)/2) mod Q or null.
    void find_roots(const Poly& Q, const Poly* hint, std::vector<u32>& roots) {
        const int d = Q.n - 1;
        if (d == 1) return roots.push_back(neg(Q.c[0]));
        if (d == 2) {
            const u32 b = Q.c[1], c = Q.c[0];
            const u32 disc = sub(mul(b, b), mul(to_montgomery(4), c));
            const u32 root = Ring<S>::sqrt(disc), half = inverse(to_montgomery(2));
            roots.push_back(mul(sub(root, b), half));
            roots.push_back(mul(sub(neg(b), root), half));
            return;
        }
        const Modulus M = R_.modulus(Q);
        Poly w = hint ? *hint : R_.power_of_linear(random_residue(), (kP - 1) / 2, M);
        for (;;) {
            w.c[0] = sub(w.c[0], kOne);
            w.set_size(std::max(w.n, 1));
            const Poly G = R_.gcd(Q, w);
            if (G.n > 1 && G.n < Q.n) {
                find_roots(G, nullptr, roots);
                find_roots(R_.quotient(Q, G), nullptr, roots);
                return;
            }
            w = R_.power_of_linear(random_residue(), (kP - 1) / 2, M);
        }
    }

    // Columns x^s u mod g, s < m, of the multiplication by u, into M.table.
    void multiplication_matrix(Modulus& M, const Poly& u) {
        const int m = M.m, stride = M.stride;
        std::vector<u32>& columns = M.table;
        columns.assign(std::size_t(m) * stride, 0);
        std::copy(u.c, u.c + u.n, columns.begin());
        for (int s = 1; s < m; ++s)
            R_.times_x(&columns[std::size_t(s - 1) * stride], &columns[std::size_t(s) * stride], M.g);
    }

    void berlekamp(Modulus& M, const Poly& r) {
        const int m = M.m, stride = M.stride;
        multiplication_matrix(M, r);
        // A = (Q - I)^T for the rows q_j = x^(P j) mod g, 8 rows at a time: the kernel of A is
        // {v : v^P = v mod g}.
        Matrix A(m, m, std::move(matrix_));
        u32* rows = R_.rows();
        for (int j0 = 0; j0 < m; j0 += 8) {
            for (int q = 0; q < 8; ++q) {
                u32* row = rows + q * stride;
                const int j = j0 + q;
                if (j == 0) {
                    std::fill(row, row + stride, 0);
                    row[0] = kOne;
                } else if (j < m) {
                    const u32* previous = q ? row - stride : rows + 7 * stride;
                    combine<S>(row, m, nullptr, previous, m, M.table.data(), stride);
                } else {
                    std::fill(row, row + stride, 0);
                }
            }
            const int width = std::min(8, m - j0);
            if (width == 8) {
                transpose_rows<S>(A.a.data(), A.ld, m, j0, rows, stride);
            } else {
                for (int i = 0; i < m; ++i)
                    for (int q = 0; q < width; ++q) A.row(i)[j0 + q] = rows[q * stride + i];
            }
        }
        for (int i = 0; i < m; ++i) A.row(i)[i] = sub(A.row(i)[i], kOne);
        const Echelon E = eliminate<S>(A);
        const int k = E.dimension();
        if (k == 1) {
            matrix_ = A.release();
            return emit(M.g);
        }
        // Two random elements of the kernel algebra: the first sorts the factors into groups, the
        // second tests the groups for more than one factor.
        std::vector<u32> samples(std::size_t(kSamples) * stride + 16);
        std::vector<u32> values(std::size_t(kSamples) * k);
        for (u32& x : values) x = random_residue();
        E.solve<S>(A, values.data(), kSamples, samples.data(), stride);
        matrix_ = A.release();
        if constexpr (Collide) collide(M.g, samples.data(), stride);
        split(M, E.free, samples.data(), r);
    }

    // The factors of g = M.g (k = free.size() of them) grouped by the value of v = samples[0]
    // (v = c_i mod g_i): the values are the roots of the minimal polynomial mu of v, whose degree
    // is the number of groups; mu comes from the coordinates of the powers of v in the kernel
    // algebra (their entries at the free columns). For large P every group is one factor but for
    // probability k^2 / 2P; otherwise the groups with more factors are factored again.
    void split(Modulus& M, const std::vector<int>& free, const u32* samples, const Poly& r) {
        const int m = M.m, stride = M.stride, k = int(free.size());
        const Poly v = R_.from_words(samples, m);
        multiplication_matrix(M, v);
        std::vector<u32>& powers = matrix_;  // v^j mod g, j <= k
        powers.assign(std::size_t(k + 1) * stride, 0);
        powers[0] = kOne;
        for (int j = 0; j < k; ++j)
            combine<S>(&powers[std::size_t(j + 1) * stride], m, nullptr, &powers[std::size_t(j) * stride], m,
                       M.table.data(), stride);
        Matrix C(k, k + 1, std::move(M.table));
        for (int t = 0; t < k; ++t)
            for (int j = 0; j <= k; ++j) C.row(t)[j] = powers[std::size_t(j) * stride + free[t]];
        const Echelon EC = eliminate<S>(C);
        const int groups = EC.free[0];  // the first power dependent on the lower ones
        std::vector<u32> unit(EC.dimension(), 0);
        unit[0] = kOne;
        Poly minimal = R_.zero();
        EC.solve<S>(C, unit.data(), 1, minimal.c, k + 1);
        minimal.n = groups + 1;
        M.table = C.release();
        std::vector<u32> roots;
        find_roots(minimal, nullptr, roots);
        if (groups == k) return extract(M.g, v, roots, 0, k, powers, stride, m, nullptr);
        std::vector<Poly> pieces;
        pieces.reserve(groups);
        extract(M.g, v, roots, 0, groups, powers, stride, m, &pieces);
        // A group where the second sample is not constant has two factors or more. If the groups
        // left do not make up the k factors, some passed as constant (probability 1/P per pair):
        // then every group left is factored again.
        const std::size_t before = factors_.size();
        std::vector<char> settled(groups, 0);
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < groups; ++i) {
                if (settled[i]) continue;
                const Poly& h = pieces[i];
                if (h.n > 4 && (pass == 1 || !is_constant_on(h, samples + stride, m))) {
                    Modulus N = R_.modulus(h, std::move(M.table));
                    berlekamp(N, R_.remainder(r, h));
                    M.table = std::move(N.table);
                    settled[i] = 1;
                }
            }
            std::size_t count = factors_.size() - before;
            for (int i = 0; i < groups; ++i) count += !settled[i];
            if (count == std::size_t(k)) break;
        }
        for (int i = 0; i < groups; ++i)
            if (!settled[i]) emit(pieces[i]);
    }

    void collide(const Poly& g, u32* samples, int stride) {
        const int m = g.n - 1;
        Poly v = R_.from_words(samples, m), r = R_.constant(kOne);
        for (u64 e = (kP - 1) / 2; e; e >>= 1) {
            if (e & 1) r = R_.multiply_remainder(r, v, g);
            v = R_.multiply_remainder(v, v, g);
        }
        std::fill(samples, samples + stride, 0);
        std::copy(r.c, r.c + r.n, samples);
        if (collisions_++ & 1) {
            std::fill(samples + stride, samples + 2 * stride, 0);
            samples[stride] = kOne;
        }
    }
    int collisions_ = 0;

    // Whether u (m coefficients) is a constant modulo h.
    bool is_constant_on(const Poly& h, const u32* u, int m) { return R_.remainder(R_.from_words(u, m), h).n <= 1; }

    // h is the product of the g_i whose value c_i of v is in roots[lo, hi); vh = v mod h. The left
    // part is gcd(h, Q(v)) for Q = prod over roots[lo, mid) of (t - c): from the powers of v mod g
    // near the top, by Horner on v mod h once that is cheaper (small h, few roots). The parts go
    // to pieces, or are emitted if pieces is null.
    void extract(const Poly& h, const Poly& vh, const std::vector<u32>& roots, int lo, int hi,
                 const std::vector<u32>& powers, int stride, int m, std::vector<Poly>* pieces) {
        if (hi - lo == 1) return pieces ? pieces->push_back(h) : emit(h);
        const int mid = (lo + hi) / 2;
        Poly u = R_.zero();
        if ((mid - lo) * (h.n - 1) <= m) {
            u = shifted(vh, roots[lo]);
            for (int i = lo + 1; i < mid; ++i) u = R_.multiply_remainder(u, shifted(vh, roots[i]), h);
        } else {
            Poly product = R_.constant(kOne);
            product.n = mid - lo + 1;  // the product of mid - lo linear factors
            for (int i = lo; i < mid; ++i) {
                const u32 c = neg(roots[i]);
                for (int j = i - lo + 1; j > 0; --j) product.c[j] = add(product.c[j - 1], mul(product.c[j], c));
                product.c[0] = mul(product.c[0], c);
            }
            combine<S>(u.c, m, nullptr, product.c, mid - lo + 1, powers.data(), stride);
            u.set_size(m);
            u = R_.remainder(u, h);
        }
        const Poly left = R_.gcd(h, u), right = R_.quotient(h, left);
        extract(left, R_.remainder(vh, left), roots, lo, mid, powers, stride, m, pieces);
        extract(right, R_.remainder(vh, right), roots, mid, hi, powers, stride, m, pieces);
    }

    // u - c.
    Poly shifted(Poly u, u32 c) const {
        u.c[0] = sub(u.c[0], c);
        u.set_size(std::max(u.n, 1));
        return u;
    }
};

}  // namespace detail

template <class S>
std::vector<Factor> factor_as(std::span<const u32> f) {
    std::size_t n = f.size();
    while (n > 0 && f[n - 1] == 0) --n;
    if (n == 0) throw std::invalid_argument("fps::factorization::factor: f = 0");
    if (n - 1 > std::size_t(kMaxDegree)) throw std::length_error("fps::factorization::factor: deg f > kMaxDegree");
    if (n == 1) return {};
    return detail::Factorizer<S>(int(n) - 1).run(f.first(n));
}

inline std::vector<Factor> factor(std::span<const u32> f) { return factor_as<simd::Native>(f); }

}  // namespace fps::factorization
