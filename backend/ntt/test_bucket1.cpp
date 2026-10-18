// Tests for backend/ntt: ragged lengths just past a power of two (issue #261). Products and squares
// whose factors have half + e words at a radix-2 top (intrinsics' first groups with a tail for e
// <= half / 16, the radix-2 level past it): cyclic products of length L (the top plan's main
// product, f g mod (x^L - 1)) against their definition, and products from spans at 2^lg; the top
// plan's products and middle products at N = L + r for small r (s = 0: the main product reads v
// itself) against schoolbook or the plain plan; stage_turned against its definition. Every
// variant, factors and out ending their allocations (ASan sees a read past them).
#include "backend/ntt/test_util.hpp"

namespace {

namespace dr = detail::ragged;

template <class F>
void each_variant(F f) {
    for_each_variant(f);
#if defined(__ARM_NEON) && defined(__aarch64__)
    f.template operator()<variant::neon>("neon", kVectorMaxLog);
#endif
    f.template operator()<SelectedMultiply>("selected", kVectorMaxLog);
}

// The tail lengths e of a factor of half + e words: short ones, around the first groups' steps
// (2 vectors of 4 to 16 lanes), around half / 16 (the longest that shares) and past a quarter.
std::vector<std::size_t> tails(std::size_t half) {
    std::vector<std::size_t> es;
    for (const std::size_t e : {std::size_t(1), std::size_t(2), std::size_t(3), std::size_t(7), std::size_t(8), std::size_t(9),
                                std::size_t(31), std::size_t(32), std::size_t(33), std::size_t(64), half / 16 - 1, half / 16,
                                half / 16 + 1, half / 8 + 5, half / 4 + 1, half / 2})
        if (e > 0 && e < half && std::ranges::find(es, e) == es.end()) es.push_back(e);
    return es;
}

// The product of f and g: schoolbook for short factors, else the scalar variant's plain product.
std::vector<u32> exact_product(const std::vector<u32>& f, const std::vector<u32>& g) {
    if (f.size() * g.size() <= (std::size_t(1) << 22)) return schoolbook(f, g);
    std::vector<u32> c(f.size() + g.size() - 1);
    detail::product<variant::scalar, false>(f.data(), f.size(), g.data(), g.size(), c.data(), c.size(),
                                            dr::split_plan(c.size(), dr::kPlain));
    return c;
}

// f g mod (x^L - 1) (f^2 if kSquare) by cyclic_into, f and g ending their allocations, work full of
// garbage; against the exact product folded.
template <class V, bool kSquare>
void check_cyclic(const char* name, const std::vector<u32>& f, const std::vector<u32>& g, int lg) {
    const std::size_t L = std::size_t(1) << lg, n = f.size(), m = g.size();
    PageWords fw(n, n % 7, 0), gw(m, 3 + m % 5, 0), out(L, 1, 0);
    std::copy(f.begin(), f.end(), fw.begin());
    std::copy(g.begin(), g.end(), gw.begin());
    std::vector<detail::Line> work(3 * (L + kPadding) / 16 + 1024);
    for (detail::Line& line : work)
        for (u32& w : line.words) w = u32(rng());
    for (u32& w : out.span()) w = u32(rng());
    const auto [roots, inverse_roots] = detail::tables<V>(lg);
    detail::cyclic_into<V, kSquare>(fw.begin(), n, kSquare ? fw.begin() : gw.begin(), kSquare ? n : m, out.begin(),
                                    work.data()->words, lg, roots, inverse_roots);
    const std::vector<u32> c = exact_product(f, kSquare ? f : g);
    std::vector<u32> want(L);
    for (std::size_t i = 0; i < c.size(); ++i) want[i % L] = add(want[i % L], c[i]);
    expect(std::ranges::equal(out.span(), want), name, (n * 1000 + m % 1000) * 10 + kSquare);
    expect(std::equal(f.begin(), f.end(), fw.begin()) && std::equal(g.begin(), g.end(), gw.begin()), "factors kept", n);
}

// Cyclic products and squares of length 2^lg, 2^8 <= 2^lg <= 2^16, with a factor of half + e words
// (both, or one with the other at half, half - 3 or 5 words, or g longer); random words and P - 1.
// Products from spans at 2^lg with f of half + e words and g of half - e or 3, out cut or not.
template <class V>
void test_tails(const char* name, int max_lg) {
    for (int lg = 8; lg <= std::min(max_lg, 16); ++lg) {
        const std::size_t L = std::size_t(1) << lg, half = L / 2;
        for (const std::size_t e : tails(half)) {
            const int kind = e % 3 == 1 ? 1 : 0;  // P - 1 everywhere, or random
            const auto f = random_poly(half + e, kind);
            for (const std::size_t m : {half + e, half, half - 3, std::size_t(5), half + e + 1, half + 2 * e})
                if (m <= L) check_cyclic<V, false>(name, f, random_poly(m, kind), lg);
            check_cyclic<V, false>(name, random_poly(half, kind), f, lg);
            check_cyclic<V, true>(name, f, f, lg);
            if (lg > 13 && e % 2) continue;  // spans: fewer at the longer lengths
            for (const std::size_t m : {half - e, std::size_t(3)}) {
                const auto g = random_poly(m, kind);
                const auto product = exact_product(f, g);
                test_spans<V>(name, f, g, product, {product.size(), half + 1, L / 4 + 3}, lg % 3);
            }
        }
    }
}

// out[i] = sum_{j < m} f[i + j] g[m - 1 - j] for i < count, in sums of 16 products (16 P^2 < 2^64).
std::vector<u32> fast_middle(const std::vector<u32>& f, const std::vector<u32>& g, std::size_t count) {
    const std::size_t m = g.size();
    std::vector<u32> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        u64 sum = 0;
        for (std::size_t j = 0; j < m;) {
            u64 part = 0;
            for (const std::size_t end = std::min(m, j + 16); j < end; ++j) part += u64(f[i + j]) * g[m - 1 - j];
            sum = (sum + part % P) % P;
        }
        out[i] = u32(sum);
    }
    return out;
}

// The middle product of f (m - 1 + count words read) and g by plan p: detail::middle with f, g and
// out each ending its allocation, out full of garbage first; and middle_into with work of its own,
// which must agree. f and g unchanged.
template <class V>
std::vector<u32> run_middle(const std::vector<u32>& f, const std::vector<u32>& g, std::size_t count, const dr::Plan& p,
                            std::size_t offset) {
    const std::size_t m = g.size(), n = m - 1 + count;
    PageWords fw(n, offset, 0), gw(m, offset + 5, 0), out(count, offset + 2, 0);
    std::copy_n(f.begin(), n, fw.begin());
    std::copy(g.begin(), g.end(), gw.begin());
    for (u32& w : out.span()) w = u32(rng());
    detail::middle<V>(fw.begin(), gw.begin(), m, out.begin(), count, p);
    std::vector<detail::Line> work(detail::middle_words(m, count, p) / 16 + 1);
    for (detail::Line& line : work)
        for (u32& w : line.words) w = u32(rng());
    std::vector<u32> c(count + 1, 7);
    const auto [roots, inverse_roots] = detail::tables<V>(detail::table_log(p));
    detail::middle_into<V>(fw.begin(), n, gw.begin(), m, c.data(), count, work.data()->words, p, roots, inverse_roots);
    expect(std::equal(out.begin(), out.begin() + count, c.begin()) && c[count] == 7, "middle_into", n);
    expect(std::equal(f.begin(), f.begin() + n, fw.begin()) && std::equal(g.begin(), g.end(), gw.begin()), "factors kept", n);
    return {out.begin(), out.begin() + count};
}

// r = N - L in bucket 1 and around the vectors' widths.
std::vector<std::size_t> small_r(std::size_t L) {
    std::vector<std::size_t> rs;
    for (const std::size_t r : {std::size_t(1), std::size_t(2), std::size_t(3), std::size_t(15), std::size_t(16), std::size_t(17),
                                L / 128, L / 32 + 1, L / 16 - 1, L / 16})
        if (r > 0 && std::ranges::find(rs, r) == rs.end()) rs.push_back(r);
    return rs;
}

// The top plan (s = 0) at N = L + r for small r: products and squares from spans against exact ones,
// middle products against schoolbook to 2^10 and against the plain plan past it (one word checked
// against its definition), g of 1, 2, r, r + 1, about half, L - 1, L, L + 1 (folded when staged) and
// N - 1 words.
template <class V>
void test_top(const char* name, int max_lg) {
    for (const int lg : {8, 9, 10, 12, 13, 16}) {
        if (lg + 1 > max_lg) continue;
        const std::size_t L = std::size_t(1) << lg;
        for (const std::size_t r : small_r(L)) {
            const std::size_t N = L + r;
            const dr::Plan p = dr::split_plan(N, dr::kTop);
            expect(p.split && p.s == 0 && p.t == r && p.lg == lg, "top plan", N);
            const auto f = random_poly(N, r % 3 == 1 ? 1 : 0);
            if (lg <= 13) {  // products and squares of N words
                const std::vector<u32> u(f.begin(), f.begin() + N / 2 + 1), w(f.begin(), f.begin() + (N - N / 2));
                const auto uw = exact_product(u, w);
                for (const std::size_t count : {N, N - 1, L + 1}) {
                    PageWords out(count, r % 4, 0);
                    detail::product<V, false>(u.data(), u.size(), w.data(), w.size(), out.begin(), count, p);
                    expect(std::ranges::equal(out.span(), cut(uw, count)), name, N * 10 + 1);
                }
                const std::vector<u32> h(f.begin(), f.begin() + (N + 1) / 2);
                const std::size_t hn = 2 * h.size() - 1;
                PageWords sq(hn, 2, 0);
                detail::product<V, true>(h.data(), h.size(), h.data(), h.size(), sq.begin(), hn, dr::split_plan(hn, dr::kTop));
                expect(std::ranges::equal(sq.span(), exact_product(h, h)), name, N * 10 + 2);
            }
            for (const std::size_t m : {std::size_t(1), std::size_t(2), r, r + 1, N / 2, L - 1, L, L + 1, N - 1}) {
                if (m == 0 || m > N) continue;
                const auto g = random_poly(m, (N + m) % 2);
                const std::size_t count = N - m + 1;
                const std::vector<u32> top = run_middle<V>(f, g, count, p, (N + m) % 9);
                if (lg <= 10) {
                    expect(top == fast_middle(f, g, count), name, N * 1000 + m % 1000);
                    continue;
                }
                const std::vector<u32> plain = run_middle<V>(f, g, count, dr::split_plan(N, dr::kPlain), 3);
                const std::size_t i = random_below(count);
                u64 sum = 0;
                for (std::size_t j = 0; j < m; ++j) sum = (sum + u64(f[i + j]) * g[m - 1 - j]) % P;
                expect(plain[i] == sum, name, N);
                expect(top == plain, name, N * 1000 + m % 1000);
            }
        }
    }
}

// stage_turned: b = g x^-(m - 1) mod (x^L - 1) against its definition, over garbage.
void test_stage() {
    for (const std::size_t L : {std::size_t(64), std::size_t(256)}) {
        for (const std::size_t m : {std::size_t(1), std::size_t(2), L / 2, L - 1, L, L + 1, L + L / 2, 2 * L - 1, 2 * L}) {
            const auto g = random_poly(m, 0);
            std::vector<u32> b(L + 1, 5), want(L);
            for (std::size_t i = 0; i < m; ++i) {
                const std::size_t at = (i + L * 2 - (m - 1)) % L;
                want[at] = add(want[at], g[i]);
            }
            dr::stage_turned(g.data(), m, L, b.data());
            expect(std::equal(want.begin(), want.end(), b.begin()) && b[L] == 5, "stage_turned", L * 1000 + m);
        }
    }
}

}  // namespace

int main() {
    each_variant([]<class V>(const char* name, int max_lg) {
        test_tails<V>(name, max_lg);
        test_top<V>(name, max_lg);
    });
    test_stage();
    return finish();
}
