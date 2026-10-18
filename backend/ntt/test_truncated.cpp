// Tests for backend/ntt: truncated products (issue #155 round 2, truncated.hpp). The shape of every
// length's truncated transform; every variant's group products against the scalar model's, from
// inputs below 4P; multiply and square by the truncated plan against schoolbook and against the
// plain plan, at the edges of the nodes' rounding, balanced, unbalanced (a factor longer than L),
// cut, at P - 1, out overlapping the factors and ending its allocation; the API with the table set
// to truncated plans everywhere (middle products and split plans' top products among them).
#include "backend/ntt/test_util.hpp"

namespace {

namespace dr = detail::ragged;
namespace dt = detail::truncated;

// Every variant this machine runs, the neon one and the selected one too.
template <class F>
void each_variant(F f) {
    for_each_variant(f);
#if defined(__ARM_NEON) && defined(__aarch64__)
    f.template operator()<variant::neon>("neon", kVectorMaxLog);
#endif
    f.template operator()<SelectedMultiply>("selected", kVectorMaxLog);
}

// Schoolbook in sums of 16 products (16 P^2 < 2^64).
std::vector<u32> fast_schoolbook(const std::vector<u32>& a, const std::vector<u32>& b) {
    std::vector<u32> c(a.size() + b.size() - 1);
    for (std::size_t k = 0; k < c.size(); ++k) {
        const std::size_t lo = k >= b.size() ? k - b.size() + 1 : 0, hi = std::min(k, a.size() - 1);
        u64 sum = 0;
        for (std::size_t i = lo; i <= hi;) {
            u64 part = 0;
            for (const std::size_t end = std::min(hi + 1, i + 16); i < end; ++i) part += u64(a[i]) * b[k - i];
            sum = (sum + part % P) % P;
        }
        c[k] = u32(sum);
    }
    return c;
}

// The shape of N's truncated plan: the full nodes tile the leaf form of 2L from 0 to rr, in order,
// each a group (size / 8 a power of 4, at least 16 leaves) at its own words (k size); rr covers N,
// and wastes less than a node of kMinNode (or the deepest level's children); the levels run from
// the root down, each the partial child of the one before.
void check_shape(std::size_t N) {
    if (!dr::truncates(N)) return;
    const dt::Shape s = dt::shape(N);
    const std::size_t L = std::size_t(1) << s.lg, len = 2 * L;
    bool ok = L < N && N < len && s.N == N && s.rr >= N && s.rr < len && s.levels >= 1 && s.levels <= dt::kMaxLevels;
    ok &= s.fulls >= 1 && s.fulls <= dt::kMaxFulls && s.level[0].node.k == 0 && s.level[0].node.size == len;
    std::size_t at = 0;
    for (int i = 0; i < s.fulls; ++i) {
        const dt::Node& n = s.full[i];
        ok &= n.at == at && n.k * n.size == n.at && n.size >= dr::kMinNode && dr::power_of_4(n.size / 8);
        at += n.size;
    }
    ok &= at == s.rr;
    for (int d = 0; d < s.levels; ++d) {
        const dt::Level& l = s.level[d];
        const std::size_t c = l.node.size / std::size_t(l.radix);
        ok &= l.n > 0 && l.n < l.node.size && c >= dr::kMinNode && l.node.at == l.node.k * l.node.size;
        ok &= l.radix == (dr::power_of_4(l.node.size / 8) ? 4 : 2) && (d == 0 || l.radix == 4);
        if (d + 1 < s.levels) {
            const dt::Level& next = s.level[d + 1];
            ok &= next.node.k == std::size_t(l.radix) * l.node.k + l.n / c && next.node.size == c && next.n == l.n % c;
            ok &= next.mu == (l.n / c ? u32(l.radix) : l.mu * u32(l.radix));
        } else {
            ok &= l.n % c == 0;
        }
    }
    // rr: N rounded up to the deepest level's children, or less.
    const dt::Level& last = s.level[s.levels - 1];
    ok &= s.rr - N < std::max(dr::kMinNode, last.node.size / std::size_t(last.radix));
    expect(ok, "shape", N);
}

void test_shapes() {
    for (std::size_t N = 1; N <= (std::size_t(1) << 14); ++N) {
        check_shape(N);
        if (std::has_single_bit(N) || N <= 64) expect(!dr::truncates(N), "truncates: power of two or short", N);
    }
    for (int k = 15; k <= kMaxLog; ++k) {
        const std::size_t len = std::size_t(1) << k;
        for (const std::size_t d : {std::size_t(1), std::size_t(127), std::size_t(128), std::size_t(129), len / 3, len / 4,
                                    len / 8 + 1, len / 2 - 1, len / 16 * 7})
            check_shape(len - d), check_shape(len + d);
        expect(!dr::truncates(len) && !dr::truncates(2 * len - 1), "truncates: past rounding", k);
    }
    expect(!dr::truncates((std::size_t(1) << kMaxLog) + 1), "truncates: past 2^kMaxLog");
}

// V's group product (and square) of k with nv leaves against the scalar model's, inputs below 4P.
template <class V>
void test_groups(const char* name, int max_lg) {
    for (int lg = 7; lg <= std::min(max_lg, 17); lg += 2) {  // groups of 2^7 .. 2^17 words: nv a power of 4
        const std::size_t size = std::size_t(1) << lg, nv = size / 8;
        const int table = std::min(kMaxLog, lg + 4);
        const auto [r, ir] = detail::tables<V>(table);
        for (const std::size_t k : {std::size_t(0), std::size_t(1), std::size_t(3), (std::size_t(1) << (table - lg)) - 1,
                                    std::size_t(5) % (std::size_t(1) << (table - lg))}) {
            Words a(size), b(size), ma(size), mb(size);
            for (std::size_t i = 0; i < size; ++i) {
                a.begin()[i] = u32(rng() % (4 * u64(P))), b.begin()[i] = i % 7 ? u32(rng() % (4 * u64(P))) : 4 * P - 1;
                ma.begin()[i] = a.begin()[i], mb.begin()[i] = b.begin()[i];
            }
            V::group_product(a.begin(), b.begin(), nv, k, r, ir);
            variant::scalar::group_product(ma.begin(), mb.begin(), nv, k, r, ir);
            bool ok = true;
            for (std::size_t i = 0; i < size; ++i) ok &= a.begin()[i] < 2 * P && a.begin()[i] % P == ma.begin()[i];
            expect(ok, name, (lg * 1000 + k) * 10);
            for (std::size_t i = 0; i < size; ++i) ma.begin()[i] = mb.begin()[i] = b.begin()[i] = u32(rng() % (4 * u64(P)));
            V::group_square(b.begin(), nv, k, r, ir);
            variant::scalar::group_product(ma.begin(), mb.begin(), nv, k, r, ir);
            ok = true;
            for (std::size_t i = 0; i < size; ++i) ok &= b.begin()[i] < 2 * P && b.begin()[i] % P == ma.begin()[i];
            expect(ok, name, (lg * 1000 + k) * 10 + 1);
        }
    }
}

// f g (or f^2) of n and m words cut to count by the truncated plan, in caller memory ending its
// allocation (ASan sees a word past it), out full of garbage first: detail::product, and
// ragged_into with work of its own (a top product's path), which must agree.
template <class V, bool kSquare>
std::vector<u32> run_truncated(const std::vector<u32>& f, const std::vector<u32>& g, std::size_t count, std::size_t offset) {
    const std::size_t n = std::min(f.size(), count), m = kSquare ? n : std::min(g.size(), count), N = n + m - 1;
    const dr::Plan p = dr::plan(N, dr::kTruncated);
    expect(p.truncated, "plan truncated", N);
    PageWords fw(n, offset, 0), gw(m, offset + 3, 0), out(count, offset + 7, 0);
    std::copy_n(f.begin(), n, fw.begin());
    std::copy_n(g.begin(), m, gw.begin());
    for (u32& w : out.span()) w = u32(rng());
    const u32* gs = kSquare ? fw.begin() : gw.begin();
    detail::product<V, kSquare>(fw.begin(), n, gs, m, out.begin(), count, p);
    std::vector<detail::Line> work(dr::truncated_words(std::size_t(1) << p.lg) / 16 + 1);
    std::vector<u32> c(N);
    const auto [roots, inverse_roots] = detail::tables<V>(detail::table_log(p));
    detail::ragged_into<V, kSquare>(fw.begin(), n, gs, m, c.data(), work.data()->words, p, roots, inverse_roots);
    expect(std::equal(c.begin(), c.begin() + count, out.begin()), "ragged_into", N);
    expect(std::equal(f.begin(), f.begin() + n, fw.begin()) && std::equal(g.begin(), g.begin() + m, gw.begin()),
           "factors kept", N);
    return {out.begin(), out.begin() + count};
}

// Lengths at the edges of the rounding at L = 2^lg: r = N - L next to the multiples of kMinNode and
// of the levels' chunks, and near L.
std::vector<std::size_t> edge_lengths(int lg) {
    const std::size_t L = std::size_t(1) << lg;
    std::vector<std::size_t> Ns;
    for (const std::size_t r : {std::size_t(1), std::size_t(2), dr::kMinNode - 1, dr::kMinNode, dr::kMinNode + 1, L / 8 - 1,
                                L / 4, L / 4 + 1, L / 2 - 1, L / 2, L / 2 + 1, L / 2 + dr::kMinNode + 3, L / 2 + L / 8,
                                3 * L / 4 - 1, 3 * L / 4, 3 * L / 4 + 5, L - L / 8, L - 2 * dr::kMinNode, L - dr::kMinNode - 1})
        if (r < L && dr::truncates(L + r)) Ns.push_back(L + r);
    return Ns;
}

// Products and squares by the truncated plan at the edge lengths of 2^8 .. 2^12, against
// schoolbook: balanced, unbalanced (f of N - m + 1 words with g of 1, 3 and 200: longer than L),
// cut, P - 1 inputs.
template <class V>
void test_short(const char* name, int) {
    for (int lg = 8; lg <= 12; ++lg) {
        for (const std::size_t N : edge_lengths(lg)) {
            const int kind = N % 3 == 0 ? 1 : 0;
            const std::size_t n = N / 2 + 1, m = N + 1 - n;
            const auto f = random_poly(n, kind), g = random_poly(m, kind), fg = fast_schoolbook(f, g);
            expect(run_truncated<V, false>(f, g, N, N % 5) == fg, name, N * 10);
            const std::size_t cut_count = N - 1 - N % 17;  // cut to fewer words, still truncated
            if (dr::truncates(cut_count / 2 + 1 + std::min(m, cut_count) - 1)) {
                const auto got = run_truncated<V, false>(f, g, cut_count, 1);
                expect(std::equal(got.begin(), got.end(), fg.begin()), name, N * 10 + 1);
            }
            const auto h = random_poly((N + 1) / 2, kind);
            if (dr::truncates(2 * h.size() - 1))
                expect(run_truncated<V, true>(h, h, 2 * h.size() - 1, 2) == fast_schoolbook(h, h), name, N * 10 + 2);
            for (const std::size_t small : {std::size_t(1), std::size_t(3), std::size_t(200)}) {
                const auto u = random_poly(N + 1 - small, kind), v = random_poly(small, kind);
                expect(run_truncated<V, false>(u, v, N, 3) == fast_schoolbook(u, v), name, N * 10 + 3);
                expect(run_truncated<V, false>(v, u, N, 0) == fast_schoolbook(v, u), name, N * 10 + 4);
            }
        }
    }
}

// Long lengths, against the plain plan (checked by evaluation): the edges of 2^14 and 2^16, and
// 1.5, 1.625, 1.75, 1.875 2^k (the bench's) for k = 14 .. 20 within max_lg.
template <class V>
void test_long(const char* name, int max_lg) {
    std::vector<std::size_t> Ns = edge_lengths(14);
    for (const std::size_t N : edge_lengths(16)) Ns.push_back(N);
    for (int k = 14; k <= std::min(max_lg - 1, 20); k += 2) {
        const std::size_t L = std::size_t(1) << k;
        for (const std::size_t N : {L + L / 2, L + 5 * L / 8, 2 * L - L / 4, 2 * L - L / 8, L + 1, L + 3 * L / 8 + 77})
            if (dr::truncates(N)) Ns.push_back(N);
    }
    for (const std::size_t N : Ns) {
        if (int(std::bit_width(N - 1)) > max_lg) continue;
        const std::size_t n = N / 2 + 1, m = N + 1 - n;
        const auto f = random_poly(n, N % 2), g = random_poly(m, N % 2);
        std::vector<u32> plain(N);
        detail::product<V, false>(f.data(), n, g.data(), m, plain.data(), N, dr::plan(N, dr::kPlain));
        bool ok = true;
        for (int i = 0; i < 2; ++i) {
            const u32 x = random_below(P);
            ok &= evaluate(plain, x) == mul(evaluate(f, x), evaluate(g, x));
        }
        expect(ok, name, N);
        expect(run_truncated<V, false>(f, g, N, N % 3) == plain, name, N * 10 + 5);
        const auto h = random_poly((N + 1) / 2, 0);
        std::vector<u32> squared(2 * h.size() - 1);
        detail::product<V, true>(h.data(), h.size(), h.data(), h.size(), squared.data(), squared.size(),
                                 dr::plan(squared.size(), dr::kPlain));
        if (dr::truncates(squared.size())) expect(run_truncated<V, true>(h, h, squared.size(), 1) == squared, name, N * 10 + 6);
    }
}

// The API with every entry of the table truncated (others floor, then plain): multiply and square
// (truncated where it exists), middle products (floor or plain), split plans (whose top products
// are truncated) by ragged_into; out overlapping f and g.
void test_api() {
    for (const int other : {int(dr::kFloor), int(dr::kPlain)}) {
        dr::Kinds all;
        all.fill(dr::encode(dr::kTruncated, other));
        dr::kinds.set(all);
        for (std::size_t N = 300; N <= 4200; N += 97) {
            const std::size_t n = N / 2 + 1, m = N + 1 - n;
            const auto f = random_poly(n, 0), g = random_poly(m, 0), fg = fast_schoolbook(f, g);
            std::vector<u32> out(N + 2, 5);
            multiply(f, g, std::span(out).first(N));
            expect(std::equal(fg.begin(), fg.end(), out.begin()) && out[N] == 5, "api multiply", N);
            std::vector<u32> sq(2 * n - 1);
            square(f, sq);
            expect(sq == fast_schoolbook(f, f), "api square", N);
            std::vector<u32> both(n + m + N);  // out over f and g
            std::copy(f.begin(), f.end(), both.begin());
            std::copy(g.begin(), g.end(), both.begin() + n);
            multiply(std::span(both).first(n), std::span(both).subspan(n, m), std::span(both).first(N));
            expect(std::ranges::equal(std::span(both).first(N), fg), "api overlap", N);
            const auto v = random_poly(N, 0), w = random_poly(N / 2, 0);
            std::vector<u32> mid(N - N / 2 + 1);
            middle_product(v, w, mid);
            const auto vw = fast_schoolbook(v, w);
            expect(std::equal(mid.begin(), mid.end(), vw.begin() + (w.size() - 1)), "api middle", N);
            for (const int kind : {int(dr::kFloor), int(dr::kTop)}) {  // a split's top product, truncated
                const dr::Plan p = dr::plan(N, kind);
                if (!p.split) continue;
                std::vector<detail::Line> work(dr::split_words(std::size_t(1) << p.lg) / 16 + 1);
                std::vector<u32> c(N);
                const auto [roots, inverse_roots] = detail::tables<SelectedMultiply>(p.lg);
                detail::ragged_into<SelectedMultiply, false>(f.data(), n, g.data(), m, c.data(), work.data()->words, p, roots,
                                                             inverse_roots);
                expect(c == fg, "api split", N * 10 + kind);
            }
        }
    }
    dr::kinds.set(dr::kKindDefaults);
    // choose() never truncated; choose_product() as the table says.
    for (std::size_t N = 65; N <= (std::size_t(1) << 20); N = N * 9 / 8 + 1) {
        expect(!dr::choose(N).truncated, "choose", N);
        expect(dr::choose_product(N).truncated == (dr::product_kind(dr::kKindDefaults[dr::entry(N)]) == dr::kTruncated &&
                                                   dr::truncates(N)),
               "choose_product", N);
    }
}

}  // namespace

int main() {
    test_shapes();
    each_variant([]<class V>(const char* name, int max_lg) {
        test_groups<V>(name, max_lg);
        test_short<V>(name, max_lg);
        test_long<V>(name, max_lg);
    });
    test_api();
    return finish();
}
