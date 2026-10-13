// Right kernel of a matrix over F_P (Montgomery entries) by fraction-free Gaussian elimination in
// blocks of kBlockColumns columns, without inverses until the back substitution (lc-opt's).
//
// In a block, pivot t (row R at column c_t) gives the row P_t = R eliminated by the earlier pivots,
// pi_t = P_t[c_t], and every other row R_i becomes, after t pivots,
//   R_i^(t) = L_t R_i - sum_{s<t} M_s[i] S_(s,t) P_s,  L_t = prod_{s<t} pi_s,
//   S_(s,t) = prod_{s<s'<t} pi_s',  M_s[i] = R_i^(s)[c_s]
// (from R^(t+1) = pi_t R^(t) - R^(t)[c_t] P_t). The panel columns R^(t)[c] of all rows, the pivot
// rows and finally the whole rows are each one lazy linear combination (combine).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "backend/factor/kernels.hpp"

namespace fps::factorization::detail {

// rows x cols, row-major with stride ld = round_up(cols, 16), zero from cols on. The storage may
// be handed in and taken back (release), so its pages are touched once for several matrices.
struct Matrix {
    int rows = 0, cols = 0, ld = 0;
    std::vector<u32> a;

    Matrix(int r, int c, std::vector<u32>&& storage = {})
        : rows(r), cols(c), ld(round_up(c, 16)), a(std::move(storage)) {
        a.assign(std::size_t(r) * ld, 0);
    }
    u32* row(int i) { return a.data() + std::size_t(i) * ld; }
    const u32* row(int i) const { return a.data() + std::size_t(i) * ld; }
    std::vector<u32> release() { return std::move(a); }
};

// a[i] <- a[i]^-1 for nonzero a[0, n): one inverse and 3 (n - 1) products.
inline void invert_all(u32* a, int n) {
    std::vector<u32> prefix(n);
    u32 acc = kOne;
    for (int i = 0; i < n; ++i) prefix[i] = acc, acc = mul(acc, a[i]);
    acc = inverse(acc);
    for (int i = n - 1; i >= 0; --i) {
        const u32 inv = mul(acc, prefix[i]);
        acc = mul(acc, a[i]);
        a[i] = inv;
    }
}

// Echelon form left in A by eliminate(): pivot t is the row pivot_rows[t] of A, P_t = pi_t at
// column pivots[t], 0 at the earlier pivots and before. A kernel vector is fixed by its entries at
// the free columns.
struct Echelon {
    int cols = 0;
    std::vector<int> free, pivots, pivot_rows;
    std::vector<u32> scales;  // -1 / pi_t

    int dimension() const { return int(free.size()); }

    // count kernel vectors, vector j with values[j dimension() + i] at free[i], at out + j stride
    // (stride >= cols words, zero from cols on). Back substitution for all at once: row c of X
    // holds entry c of each vector, x[c_t] = -sum_{c > c_t} P_t[c] x[c] / pi_t.
    template <class S>
    void solve(const Matrix& A, const u32* values, int count, u32* out, int stride) const {
        const int k = dimension(), width = round_up(count, 16), rank = int(pivots.size());
        std::vector<u32> X(std::size_t(cols + 1) * width, 0);
        for (int j = 0; j < count; ++j)
            for (int i = 0; i < k; ++i) X[std::size_t(free[i]) * width + j] = values[j * k + i];
        for (int t = rank - 1; t >= 0; --t) {
            const int c = pivots[t];
            u32* x = X.data() + std::size_t(c) * width;
            if (c + 1 < cols) combine<S>(x, count, nullptr, A.row(pivot_rows[t]) + c + 1, cols - c - 1, x + width, width);
            for (int j = 0; j < count; ++j) x[j] = mul(x[j], scales[t]);
        }
        for (int j = 0; j < count; ++j) {
            u32* o = out + std::size_t(j) * stride;
            for (int c = 0; c < cols; ++c) o[c] = X[std::size_t(c) * width + j];
            std::fill(o + cols, o + stride, 0);
        }
    }
};

// Pivot t ends at row t of A: a chosen row is swapped to the front of the trailing rows, so they
// stay contiguous (rows r0 .. rows - 1) and a block's update of them is one combine_rows.
template <class S, int kBlockColumns = 16>
Echelon eliminate(Matrix& A) {
    const int rows = A.rows, cols = A.cols, ld = A.ld;
    Echelon E;
    E.cols = cols;
    std::vector<int>&pivots = E.pivots, &pivot_rows = E.pivot_rows;
    std::vector<u32>& pivot_values = E.scales;
    const int pstride = round_up(rows, 16);
    std::vector<u32> panel(static_cast<std::size_t>(kBlockColumns + 1) * pstride);  // M_0 .. M_(nb-1), then the work column
    std::vector<u32> block(static_cast<std::size_t>(kBlockColumns) * ld);             // P_0 .. P_(nb-1)
    int r0 = 0;  // rows r0 on: trailing, zero at every column before c0
    for (int c0 = 0; c0 < cols; c0 += kBlockColumns) {
        const int c1 = std::min(cols, c0 + kBlockColumns), n = rows - r0;
        const int from = c0 & ~15, len = cols - from;  // rows rounded to 16 words: whole vectors from here
        u32* P = block.data();
        u32* R0 = A.row(r0);
        int nb = 0;  // positions 0 .. nb - 1 of the trailing rows: this block's pivots
        u32 lambda = kOne, sigma[kBlockColumns], coef[kBlockColumns + 1];  // L_nb and S_(s,nb)
        for (int c = c0; c < c1; ++c) {
            if (nb == n) {
                E.free.push_back(c);
                continue;
            }
            // Work column: R^(nb)[c] = L_nb R[c] - sum_s M_s S_(s,nb) P_s[c].
            u32* work = &panel[std::size_t(nb) * pstride];
            for (int i = 0; i < n; ++i) work[i] = R0[std::size_t(i) * ld + c];
            std::fill(work + n, work + pstride, 0);
            for (int s = 0; s < nb; ++s) coef[s] = neg(mul(sigma[s], P[s * ld + c]));
            coef[nb] = lambda;
            combine<S>(work, n, nullptr, coef, nb + 1, panel.data(), pstride);
            int i = nb;
            while (i < n && work[i] == 0) ++i;
            if (i == n) {
                E.free.push_back(c);
                continue;
            }
            if (i != nb) {  // the pivot's row to position nb
                std::swap_ranges(R0 + std::size_t(i) * ld, R0 + std::size_t(i + 1) * ld, R0 + std::size_t(nb) * ld);
                for (int s = 0; s <= nb; ++s) std::swap(panel[std::size_t(s) * pstride + i], panel[std::size_t(s) * pstride + nb]);
            }
            // Pivot row: P_nb = L_nb R - sum_s M_s[nb] S_(s,nb) P_s, from a copy of R after P_(nb-1).
            u32* pivot = P + nb * ld;
            u32* row = R0 + std::size_t(nb) * ld;
            std::copy(row, row + ld, pivot);
            for (int s = 0; s < nb; ++s) coef[s] = neg(mul(panel[std::size_t(s) * pstride + nb], sigma[s]));
            coef[nb] = lambda;
            combine<S>(pivot + from, len, nullptr, coef, nb + 1, P + from, ld);
            std::copy(pivot, pivot + ld, row);
            const u32 pi = work[nb];
            pivots.push_back(c), pivot_rows.push_back(r0 + nb), pivot_values.push_back(pi);
            for (int s = 0; s < nb; ++s) sigma[s] = mul(sigma[s], pi);
            sigma[nb] = kOne, lambda = mul(lambda, pi), ++nb;
        }
        if (nb == 0) continue;
        // Other rows: R^(nb) = L_nb R - sum_s M_s S_(s,nb) P_s, the coefficients -M_s S_(s,nb) in
        // place of M_s (panel column s), read at position i of each.
        for (int s = 0; s < nb; ++s) {
            u32* column = &panel[std::size_t(s) * pstride];
            combine_scaled<S>(column, n, column, neg(sigma[s]), nullptr, 0, column, 0);
        }
        u32* first = R0 + std::size_t(nb) * ld + from;
        combine_rows<S>(first, ld, n - nb, len, first, ld, lambda, panel.data() + nb, 1, pstride, nb, P + from, ld);
        r0 += nb;
    }
    invert_all(pivot_values.data(), int(pivot_values.size()));
    for (u32& s : pivot_values) s = neg(s);
    return E;
}

}  // namespace fps::factorization::detail
