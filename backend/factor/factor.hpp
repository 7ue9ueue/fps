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
// Cost: O(m^3) per square-free part of degree m (the elimination below; the Krylov sequence too
// up to m about 100-210, then O(m^2 log m) by NTT products), one m x m matrix of words (two below
// the NTT degree). Random f on Zen 3 (notes.md): degree 100 about 0.2 ms, 256 about 1 ms, 1024
// about 33 ms, 4096 (the limit: 64 MiB of matrix) about 2 s. Degree 2^20 is out of reach here (4
// TiB of matrix); it wants a distinct-degree split by baby steps and giant steps (Kaltofen-Shoup)
// on NTT products. Up to kKeepDegree, each thread keeps its factorizer (buffers, matrices) between
// calls.
//
// Algorithm (lc-opt's, but for the root finding):
// - square-free decomposition by Yun's algorithm (deg f < P, so f' = 0 only for a constant);
// - per square-free part g of degree m: x^((P-1)/2^j) mod g for j = K .. 1 by squarings (a
//   product, then the reduction by the table x^(m + j) mod g, or by NTT products past the NTT
//   degree, poly.hpp's ntt_products()), x^P = x (x^((P-1)/2))^2; the linear factors L = gcd(g,
//   x^P - x);
// - roots (of L, and of mu below) by 2-power characters: P - 1 = 119 2^23, so w = (x + a)^((P-1)
//   / 2^k) takes 2^k-th roots of unity at the roots; gcds with w^(2^(k-j-1)) - (a root of unity)
//   split them bit by bit of those values, k - j - 1 squarings at depth j (for L, a = 0 and w
//   comes from the squarings above); quadratic nodes in scalars, one inverse per pair;
// - the rest by Berlekamp: rows q_j = x^(P j) mod g, the first 8 by mat-vecs with the matrix of
//   the multiplication by x^P, then each block of 8 the block before times the matrix of the
//   multiplication by q_8 (combine_rows), or by NTT products (large.hpp: each block's rows from
//   its first times x^P .. x^(8P)), transposed into A = (Q - I)^T; the kernel of A (dimension k =
//   the number of factors) by fraction-free blocked elimination (linear.hpp);
// - k > 1: two random kernel elements v, w. The factors are grouped by the value of v
//   (v = c_i mod g_i): the c_i are the roots of the minimal polynomial mu of v, found from the
//   coordinates of v^j at the free columns (a k x (k + 1) kernel). Groups by divide and conquer
//   over the roots: gcd(h, prod (v - c)), the product from the powers of v near the top, by
//   Horner on v mod h below. All groups are single factors but for probability about k^2 / 2P; a
//   group where w is not constant is factored again, and if the count still falls short of k
//   every group left is.
// - moduli of degree <= 8 on x86 with AVX2: polynomials in one ymm register (poly.hpp), as root
//   finding runs chains of dependent squarings at tiny degrees.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
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
// Up to this degree, each thread keeps its factorizer's memory between calls (two matrices of
// 256 x 256 words and polynomial buffers: about 0.6 MiB at most).
inline constexpr int kKeepDegree = 256;

struct Factor {
    std::vector<u32> coefficients;  // b_0 .. b_d in [0, P), b_d = 1
    u32 multiplicity = 0;
};

namespace detail {

// Collide (tests only): the first sample becomes v^((P-1)/2), so at most 3 groups (values 0, 1,
// -1), and every other step the second sample is the constant 1, so groups of several factors
// pass as single ones: both repair paths of split() run. NttDegree (tests): moduli of this degree
// on run their products by NTT; 0: where poly.hpp's ntt_products() says.
template <class S, bool Collide = false, int NttDegree = 0>
class Factorizer {
public:
    // Polynomials of degree <= n.
    explicit Factorizer(int n) : n_(n), R_(n + 1, NttDegree) {}

    int limit() const { return n_; }

    // f: deg + 1 words, f[deg] != 0, 1 <= deg <= limit(), in [0, P). Runs again from the same
    // state (the same random choices) on every call.
    std::vector<Factor> run(std::span<const u32> f) {
        const int n = int(f.size()) - 1;
        state_ = kSeed, collisions_ = 0;
        factors_.clear();
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

    static constexpr u64 kSeed = 0x9E3779B97F4A7C15ull;

    int n_;
    Ring<S> R_;
    u64 state_ = kSeed;
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
        // powers[j] = x^((P-1)/2^j) mod g, j = 1 .. K: the squarings to s = x^((P-1)/2) pass them.
        const int K = levels(M.m);
        std::vector<Poly> powers(K + 1, R_.zero());
        powers[K] = R_.power_of_x((kP - 1) >> K, M);
        for (int j = K - 1; j >= 1; --j) powers[j] = R_.multiply_mod(powers[j + 1], powers[j + 1], M);
        Poly r = R_.times_x(R_.multiply_mod(powers[1], powers[1], M), M);  // x^P mod g
        Poly u = r;
        u.c[1] = sub(u.c[1], kOne);
        u.set_size(std::max(u.n, 2));
        const Poly linear = R_.gcd(g, u);
        if (linear.n > 1) {
            std::vector<u32> roots;
            const int k = std::min(K, levels(linear.n - 1));
            if (linear.n <= 3) small_roots(linear, roots);
            else split_roots(linear, R_.remainder(powers[k], linear), 0, 0, k, roots);
            for (const u32 z : roots) emit(R_.linear(z));
            if (linear.n == g.n) return;
            g = R_.quotient(g, linear);
            r = R_.remainder(r, g);
            M = R_.modulus(g, std::move(M.table));
        }
        if (g.n <= 4) return emit(g);  // degree <= 3 without roots
        berlekamp(M, r);
    }

    // The 2^k-th roots of unity split the roots: k with 2^k >= 4 d^2 (at most 23: 2^23 | P - 1), so
    // that two of d roots get the same value with probability below 1/8.
    static int levels(int d) {
        int k = 2;
        while (k < 23 && (u64(1) << k) < u64(4) * u64(d) * u64(d)) ++k;
        return k;
    }

    // Roots of a monic Q of degree 1 or 2.
    void small_roots(const Poly& Q, std::vector<u32>& roots) {
        if (Q.n == 2) return roots.push_back(neg(Q.c[0]));
        const u32 b = Q.c[1], c = Q.c[0];
        const u32 disc = sub(mul(b, b), mul(to_montgomery(4), c));
        const u32 root = Ring<S>::sqrt(disc), half = inverse(to_montgomery(2));
        roots.push_back(mul(sub(root, b), half));
        roots.push_back(mul(sub(neg(b), root), half));
    }

    // Roots of a monic Q that splits into distinct linear factors.
    void find_roots(const Poly& Q, std::vector<u32>& roots) {
        if (Q.n <= 3) return small_roots(Q, roots);
        const int k = levels(Q.n - 1);
        const Modulus M = R_.modulus(Q);
        split_roots(Q, R_.power_of_linear(random_residue(), (kP - 1) >> k, M), 0, 0, k, roots);
    }

    // Roots of Q (monic, distinct linear factors) by the values of w = (x + a)^((P-1)/2^k) mod Q
    // at them: 2^k-th roots of unity w_k^e (w_k = 3^((P-1)/2^k), 3 a generator mod P), or 0 at
    // x = -a. Q's roots share the low j bits of e, `low`; bit j splits them: the roots with bit j
    // 0 are those of gcd(Q, w^(2^(k-j-1)) - w_(j+1)^low), k - j - 1 squarings modulo Q, fewer at
    // each level down (Cantor and Zassenhaus take about 29 at every node, and a new shift when a
    // node does not split). Roots whose values agree to all k bits (or -a) start again with a new
    // shift.
    void split_roots(const Poly& Q, const Poly& w, int j, u32 low, int k, std::vector<u32>& roots) {
        if (Q.n <= 2) return small_roots(Q, roots);
        if (j == k) return find_roots(Q, roots);
        if (Q.n == 3) return quadratic_roots(Q, w, j, low, k, roots);
        Poly u = w;  // w^(2^(k-j-1)) mod Q
        if (j + 1 < k) {
            const Modulus M = R_.modulus(Q);
            for (int i = j + 1; i < k; ++i) u = R_.multiply_mod(u, u, M);
        }
        u.c[0] = sub(u.c[0], unity_power(j + 1, low));
        u.set_size(std::max(u.n, 1));
        const Poly G = R_.gcd(Q, u);
        if (G.n == 1) return split_roots(Q, w, j + 1, low | 1u << j, k, roots);
        if (G.n == Q.n) return split_roots(Q, w, j + 1, low, k, roots);
        const Poly H = R_.quotient(Q, G);
        split_roots(G, R_.remainder(w, G), j + 1, low, k, roots);
        split_roots(H, R_.remainder(w, H), j + 1, low | 1u << j, k, roots);
    }

    // split_roots for Q = x^2 + b x + c0 in scalars: w^(2^t) mod Q = alpha_t x + beta_t by
    // (alpha x + beta)^2 = alpha (2 beta - alpha b) x + beta^2 - alpha^2 c0. At level j, u =
    // alpha x + beta (t = k - j - 1) takes +-c (c = w_(j+1)^low) at the roots, or 0 at -a. alpha =
    // 0: the same bit j for both (beta = c: 0, -c: 1), the next level; else they part here: the
    // root with value c (or -c) is (c - beta) / alpha, the other -b minus it. One inverse and a few
    // products per level, against Cipolla's square root (about 200 products in F_P^2's chain).
    void quadratic_roots(const Poly& Q, const Poly& w, int j, u32 low, int k, std::vector<u32>& roots) {
        const u32 b = Q.c[1], c0 = Q.c[0];
        std::array<u32, 24> alpha{}, beta{};
        alpha[0] = w.n > 1 ? w.c[1] : 0, beta[0] = w.n > 0 ? w.c[0] : 0;
        for (int t = 1; t < k - j; ++t) {
            const u32 a = alpha[t - 1], e = beta[t - 1];
            alpha[t] = mul(a, sub(add(e, e), mul(a, b)));
            beta[t] = sub(mul(e, e), mul(mul(a, a), c0));
        }
        const auto is_root = [b, c0](u32 r) { return add(mul(add(r, b), r), c0) == 0; };
        for (; j < k; ++j) {
            const u32 c = unity_power(j + 1, low), a = alpha[std::size_t(k - j - 1)], e = beta[std::size_t(k - j - 1)];
            if (a == 0) {
                if (e == c) continue;
                if (e != neg(c)) break;
                low |= 1u << j;
                continue;
            }
            const u32 inverse_alpha = inverse(a);
            u32 r = mul(sub(c, e), inverse_alpha);
            if (!is_root(r)) r = mul(sub(neg(c), e), inverse_alpha);
            if (!is_root(r)) break;
            roots.push_back(r);
            roots.push_back(sub(neg(b), r));
            return;
        }
        small_roots(Q, roots);  // values equal to all k bits (or -a among them)
    }

    // w_t^low for low < 2^t: w_t^(2^b) = w_(t-b), a product over the set bits of low.
    static u32 unity_power(int t, u32 low) {
        u32 r = kOne;
        for (; low; low &= low - 1) r = mul(r, kRootsOfUnity[t - std::countr_zero(low)]);
        return r;
    }

    // w_t = 3^((P-1)/2^t), a primitive 2^t-th root of unity (Montgomery form), t <= 23.
    static constexpr std::array<u32, 24> kRootsOfUnity = [] {
        std::array<u32, 24> w{};
        for (int t = 0; t < 24; ++t) w[t] = power(to_montgomery(3), (kP - 1) >> t);
        return w;
    }();

    // Columns x^s u mod g, s < m, of the multiplication by u, into M.table.
    void multiplication_matrix(Modulus& M, const Poly& u) {
        const int m = M.m, stride = M.stride;
        std::vector<u32>& columns = M.table;
        columns.assign(std::size_t(m) * stride, 0);
        std::copy(u.c, u.c + u.n, columns.begin());
        R_.times_x_chain(columns.data(), m, stride, M.g);
    }

    // Rows u^j mod g, j < count, at stride M.stride (zero from m on), 8 at a time: sink(j0, rows)
    // for j0 = 0, 8, 16, ... gets rows j0 .. min(j0 + 8, count) - 1. With a table, by the
    // multiplication matrices (Sequence); by NTT (M.ntt, large.hpp), each row the one before times u
    // mod g (Multiplier: about 3.5 transforms of 2 bit_ceil(m) words, against m^2 products), or
    // from kPowersFrom rows on each block's rows from its first times u .. u^8 (Powers: about 2.1
    // transforms a row, and a setup of about 30).
    static constexpr int kPowersFrom = 64;
    template <class Sink>
    void power_rows(Modulus& M, const Poly& u, int count, Sink&& sink) {
        const int m = M.m, stride = M.stride;
        u32* buffer = R_.rows();
        if (!M.ntt) {
            multiplication_matrix(M, u);
            Sequence sequence{*this, M, buffer, count};
            for (int j0 = 0; j0 < count; j0 += 8) sink(j0, static_cast<const u32*>(sequence.next(j0)));
            return;
        }
        std::vector<u32> canonical(std::size_t(m), 0);  // u canonical: the rows keep their Montgomery form
        for (int i = 0; i < u.n; ++i) canonical[std::size_t(i)] = from_montgomery(u.c[i]);
        std::fill(buffer, buffer + 16 * stride, 0);
        buffer[0] = kOne;
        if (count >= kPowersFrom) {  // a block's rows from its first: its transform once
            Powers Y(*M.ntt, canonical.data());
            for (int j0 = 0; j0 < count; j0 += 8) {
                u32* rows = buffer + (j0 / 8 % 2) * 8 * stride;
                Y.steps(rows, rows + stride, stride, buffer + ((j0 / 8 + 1) % 2) * 8 * stride);
                sink(j0, static_cast<const u32*>(rows));
            }
            return;
        }
        Multiplier X(*M.ntt, canonical.data());
        for (int j0 = 0; j0 < count; j0 += 8) {
            u32* rows = buffer + (j0 / 8 % 2) * 8 * stride;
            if (j0 > 0) X.step(buffer + ((j0 / 8 + 1) % 2) * 8 * stride + 7 * stride, rows);
            for (int q = 1; q < std::min(8, count - j0); ++q) X.step(rows + (q - 1) * stride, rows + q * stride);
            sink(j0, static_cast<const u32*>(rows));
        }
    }

    // Rows j0 .. j0 + 7 (< count) of the sequence u_j = u_(j-1) X, u_0 = 1, at rows + (j - j0) stride
    // (the rest zero), for X the multiplication matrix of u_1 in M.table: mat-vecs for the first
    // block, then M.table becomes the multiplication matrix of u_8 (from u_7, so u_8 = u_7 X), and
    // each later block is the one before times it (combine_rows: each column word serves several
    // rows). next(j0, rows) for j0 = 0, 8, 16, ...; rows has 16 rows (two blocks, alternating).
    struct Sequence {
        Factorizer& F;
        Modulus& M;
        u32* buffer;
        int count;
        u32* next(int j0) {
            const int m = M.m, stride = M.stride, width = std::min(8, count - j0);
            u32* rows = buffer + (j0 / 8 % 2) * 8 * stride;
            if (j0 == 0) {
                std::fill(rows, rows + 8 * stride, 0);
                rows[0] = kOne;
                if (width > 1) std::copy(M.table.data(), M.table.data() + stride, rows + stride);  // u_1: column 0
                for (int q = 2; q < width; ++q)
                    combine<S>(rows + q * stride, m, nullptr, rows + (q - 1) * stride, m, M.table.data(), stride);
                if (count > 8) {  // X becomes the multiplication matrix of u_8
                    Poly u = F.R_.zero();
                    combine<S>(u.c, m, nullptr, rows + 7 * stride, m, M.table.data(), stride);
                    u.set_size(m);
                    F.multiplication_matrix(M, u);
                }
                return rows;
            }
            const u32* previous = buffer + ((j0 / 8 + 1) % 2) * 8 * stride;
            combine_rows<S>(rows, stride, width, m, nullptr, 0, kOne, previous, stride, 1, m, M.table.data(), stride);
            return rows;
        }
    };

    void berlekamp(Modulus& M, const Poly& r) {
        const int m = M.m;
        // A = (Q - I)^T for the rows q_j = x^(P j) mod g, 8 rows at a time: the kernel of A is
        // {v : v^P = v mod g}.
        Matrix A(m, m, std::move(matrix_));
        power_rows(M, r, m, [&](int j0, const u32* rows) {
            const int width = std::min(8, m - j0);
            if (width == 8) {
                transpose_rows<S>(A.a.data(), A.ld, m, j0, rows, M.stride);
            } else {
                for (int i = 0; i < m; ++i)
                    for (int q = 0; q < width; ++q) A.row(i)[j0 + q] = rows[q * M.stride + i];
            }
        });
        const int stride = M.stride;
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
        std::vector<u32>& powers = matrix_;  // v^j mod g, j <= k
        powers.assign(std::size_t(k + 1) * stride, 0);
        power_rows(M, v, k + 1, [&](int j0, const u32* rows) {
            std::copy(rows, rows + std::min(8, k + 1 - j0) * stride, &powers[std::size_t(j0) * stride]);
        });
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
        find_roots(minimal, roots);
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
    const int degree = int(n) - 1;
    if (degree > kKeepDegree) return detail::Factorizer<S>(degree).run(f.first(n));
    // This thread's factorizer: its buffers and matrices stay allocated between calls.
    thread_local std::unique_ptr<detail::Factorizer<S>> kept;
    if (!kept || kept->limit() < degree) kept = std::make_unique<detail::Factorizer<S>>(std::max(degree, 64));
    return kept->run(f.first(n));
}

inline std::vector<Factor> factor(std::span<const u32> f) { return factor_as<simd::Native>(f); }

}  // namespace fps::factorization
