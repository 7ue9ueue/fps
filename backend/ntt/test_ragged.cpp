// Tests for backend/ntt: products of ragged length (issue #155, ragged.hpp). multiply, square and
// middle_product by every plan of each length (the plain product and the splits) and by the chosen
// one, against schoolbook multiplication to a few thousand words and against the plain product
// (checked by evaluation) past it: balanced, unbalanced and cut shapes, inputs at P - 1, out
// overlapping the factors, every variant.
#include "backend/ntt/test_util.hpp"

namespace {

// Schoolbook with sums of 16 products in 64 bits (16 P^2 < 2^64).
std::vector<u32> fast_schoolbook(const std::vector<u32>& a, const std::vector<u32>& b) {
    if (a.empty() || b.empty()) return {};
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

// out[i] = sum_{j < m} f[i + j] g[m - 1 - j] for i < count, in sums of 16 products.
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

// The middle product of f (m - 1 + count words read) and g by plan p into count words, in caller
// memory as run() below: detail::middle (the API's path); for the plain plan also middle_into with
// work of its own (the path of a top product), which must agree.
template <class V>
std::vector<u32> run_middle(const std::vector<u32>& f, const std::vector<u32>& g, std::size_t count,
                            const detail::ragged::Plan& p, std::size_t offset = 0) {
    const std::size_t m = g.size(), n = m - 1 + count;
    PageWords fw(n, offset, 0), gw(m, offset + 5, 0), out(count, offset + 2, 0);
    std::copy_n(f.begin(), n, fw.begin());
    std::copy(g.begin(), g.end(), gw.begin());
    for (u32& w : out.span()) w = u32(rng());
    detail::middle<V>(fw.begin(), gw.begin(), m, out.begin(), count, p);
    if (!p.split) {
        std::vector<detail::Line> work(detail::middle_words(m, count, p) / 16 + 1);
        std::vector<u32> c(count);
        const auto [roots, inverse_roots] = detail::tables<V>(p.lg);
        detail::middle_into<V>(fw.begin(), n, gw.begin(), m, c.data(), count, work.data()->words, p, roots, inverse_roots);
        expect(std::equal(c.begin(), c.end(), out.begin()), "middle_into", n);
    }
    expect(std::equal(f.begin(), f.begin() + n, fw.begin()) && std::equal(g.begin(), g.end(), gw.begin()), "factors kept", n);
    return {out.begin(), out.begin() + count};
}

// Every plan of the middle product of length m - 1 + count, and the API's, against want.
template <class V>
void check_middle(const char* name, const std::vector<u32>& f, const std::vector<u32>& g, std::size_t count,
                  const std::vector<u32>& want, std::size_t offset = 0) {
    const std::size_t m = g.size(), n = m - 1 + count;
    std::array<detail::ragged::Plan, 4> plans;
    const int options = detail::ragged::options(n, plans);
    for (int i = 0; i < options; ++i)
        expect(run_middle<V>(f, g, count, plans[i], offset) == want, name, ((n * 1000000 + m) * 10 + i) * 10 + 2);
    std::vector<u32> chosen(count);
    middle_product<V>(std::span(f).first(n), g, chosen);
    expect(chosen == want, name, (n * 1000000 + m) * 100 + 3);
}

// f g (or f^2: g = f) of n and m words by plan p into out (n + m - 1 words, apart from f and g).
template <class V, bool kSquare>
void run_plan(const u32* f, std::size_t n, const u32* g, std::size_t m, u32* out, const detail::ragged::Plan& p) {
    const std::size_t len = std::size_t(1) << p.lg;
    const std::size_t words = p.split ? detail::ragged::split_words(len) : 2 * (len + kPadding) + 1024;
    std::vector<detail::Line> work(words / 16 + 1);
    const auto [roots, inverse_roots] = detail::tables<V>(p.lg);
    detail::ragged_into<V, kSquare>(f, n, g, m, out, work.data()->words, p, roots, inverse_roots);
}

// f g (or f^2) by plan p, cut to count <= n + m - 1 words: in caller memory `offset` words into a
// page and ending its allocation (ASan sees any access past it), out full of garbage first;
// detail::product (the API's path); for the plain plan also run_plan (the path of a top product),
// which must agree.
template <class V, bool kSquare>
std::vector<u32> run(const std::vector<u32>& f, const std::vector<u32>& g, std::size_t count, const detail::ragged::Plan& p,
                     std::size_t offset = 0) {
    const std::size_t n = std::min(f.size(), count), m = kSquare ? n : std::min(g.size(), count), N = n + m - 1;
    PageWords fw(n, offset, 0), gw(m, offset + 3, 0), out(count, offset + 7, 0);
    std::copy_n(f.begin(), n, fw.begin());
    std::copy_n(g.begin(), m, gw.begin());
    for (u32& w : out.span()) w = u32(rng());
    const u32* gs = kSquare ? fw.begin() : gw.begin();
    detail::product<V, kSquare>(fw.begin(), n, gs, m, out.begin(), count, p);
    if (!p.split) {
        std::vector<u32> c(N);
        run_plan<V, kSquare>(fw.begin(), n, gs, m, c.data(), p);
        expect(std::equal(c.begin(), c.begin() + count, out.begin()), "ragged_into", N);
    }
    expect(std::equal(f.begin(), f.begin() + n, fw.begin()) && std::equal(g.begin(), g.begin() + m, gw.begin()),
           "factors kept", count);
    return {out.begin(), out.begin() + count};
}

// Every plan of f g cut to count words (the plans of n + m - 1 for n, m cut to count) and the chosen
// one (multiply, square) against want.
template <class V, bool kSquare>
void check_plans(const char* name, const std::vector<u32>& f, const std::vector<u32>& g, std::size_t count,
                 const std::vector<u32>& want, std::size_t offset = 0) {
    const std::size_t n = std::min(f.size(), count), m = std::min(g.size(), count);
    if (!n || !m) return;
    std::array<detail::ragged::Plan, 4> plans;
    const int options = detail::ragged::options(n + m - 1, plans);
    for (int i = 0; i < options; ++i) {
        const auto got = run<V, kSquare>(f, g, count, plans[i], offset);
        expect(got == want, name, ((n * 1000000 + m) * 10 + i) * 10 + kSquare);
    }
    std::vector<u32> chosen(count + 3, 7);  // the API, zeros past the product
    if constexpr (kSquare) square<V>(f, std::span(chosen).first(count));
    else multiply<V>(f, g, std::span(chosen).first(count));
    expect(std::equal(want.begin(), want.end(), chosen.begin()) && chosen[count] == 7, name, n * 1000000 + m);
}

// Every N to kAll: balanced (n = m or m + 1), unbalanced (m = 1, 2, 7: f longer than the main
// product), cut, squares. The references are shared by every variant.
constexpr std::size_t kAll = 2100;

void every_length() {
    for (std::size_t N = 1; N <= kAll; ++N) {
        const std::size_t n = N / 2 + 1, m = N + 1 - n;
        const int kind = N % 7 == 0 ? 1 : 0;  // every seventh at P - 1
        const auto f = random_poly(n, kind), g = random_poly(m, kind);
        const auto product = fast_schoolbook(f, g);
        const auto h = random_poly((N + 1) / 2, kind);
        const auto squared = fast_schoolbook(h, h);
        const std::size_t cut_count = std::min(N, n + N % 13);
        for_each_variant([&]<class V>(const char* name, int) {
            check_plans<V, false>(name, f, g, N, product, N % 4);
            check_plans<V, true>(name, h, h, squared.size(), squared, N % 3);
            if (N % 5 == 0) check_plans<V, false>(name, f, g, cut_count, cut(product, cut_count), 1);
        });
        // middle products reading N words of f: g of about half, of a few words, and of all but count
        const auto v = random_poly(N, kind);
        for (const std::size_t m : {N / 2 + 1, 1 + N % 3, N + 1 - std::min(N, 1 + N % 5)}) {
            if (m > N) continue;
            const auto w = random_poly(m, kind);
            const std::size_t count = N - m + 1;
            const auto want = fast_middle(v, w, count);
            for_each_variant([&]<class V>(const char* name, int) { check_middle<V>(name, v, w, count, want, N % 3); });
        }
        if (N % 3 == 0 && N > 8) {
            for (const std::size_t small : {1, 2, 7}) {
                const auto u = random_poly(N + 1 - small, kind), v = random_poly(small, kind);
                const auto uv = fast_schoolbook(u, v);
                for_each_variant([&]<class V>(const char* name, int) { check_plans<V, false>(name, u, v, N, uv); });
            }
        }
    }
}

// Lengths around powers of two and the bench's (2^k + 1, 1.25, 1.5, 1.75 2^k, 2^(k+1) - 1) to
// max_lg, against the plain product, which is checked by evaluation.
template <class V>
void long_lengths(const char* name, int max_lg) {
    for (int k = 11; k < std::min(max_lg, 21); ++k) {
        const std::size_t len = std::size_t(1) << k;
        for (const std::size_t N : {len - 1, len, len + 1, len + len / 4, len + len / 2, 2 * len - len / 4, 2 * len - 1, len + 77}) {
            const std::size_t n = N / 2 + 1, m = N + 1 - n;
            const auto f = random_poly(n, k % 2), g = random_poly(m, k % 2);
            std::array<detail::ragged::Plan, 4> plans;
            detail::ragged::options(N, plans);
            const std::vector<u32> plain = run<V, false>(f, g, N, plans[0]);
            bool ok = true;
            for (int i = 0; i < 2; ++i) {
                const u32 x = random_below(P);
                ok &= evaluate(plain, x) == mul(evaluate(f, x), evaluate(g, x));
            }
            expect(ok, name, N);
            check_plans<V, false>(name, f, g, N, plain, k % 3);
            check_plans<V, false>(name, f, g, n, cut(plain, n));
            const auto h = random_poly((N + 1) / 2, 1);
            detail::ragged::options(2 * h.size() - 1, plans);
            const std::vector<u32> squared = run<V, true>(h, h, 2 * h.size() - 1, plans[0]);
            check_plans<V, true>(name, h, h, squared.size(), squared);
            // middle products reading N words: g of N / 2 and of N - 9 (out of 10 words)
            const auto v = random_poly(N, 1 - k % 2);
            for (const std::size_t m : {N / 2, N - 9}) {
                const auto w = random_poly(m, 0);
                const std::size_t count = N - m + 1;
                detail::ragged::options(N, plans);
                const std::vector<u32> plain = run_middle<V>(v, w, count, plans[0]);
                const std::size_t i = random_below(count);  // one word against its definition
                u64 sum = 0;
                for (std::size_t j = 0; j < m; ++j) sum = (sum + u64(v[i + j]) * w[m - 1 - j]) % P;
                expect(plain[i] == sum, name, N);
                check_middle<V>(name, v, w, count, plain, k % 3);
            }
        }
    }
}

// out overlapping f, g or both (the split plans then run into the region and copy).
template <class V>
void overlap(const char* name) {
    for (const std::size_t N : {129, 300, 1000, 1500}) {
        const std::size_t n = N / 2 + 1, m = N + 1 - n;
        const auto f = random_poly(n, 0), g = random_poly(m, 0), product = fast_schoolbook(f, g);
        std::vector<u32> both(n + m + N);
        std::copy(f.begin(), f.end(), both.begin());
        std::copy(g.begin(), g.end(), both.begin() + n);
        multiply<V>(std::span(both).first(n), std::span(both).subspan(n, m), std::span(both).first(N));
        expect(std::ranges::equal(std::span(both).first(N), product), name, N);
        std::vector<u32> square_in_place(f);
        square_in_place.resize(2 * n - 1);
        square<V>(std::span(square_in_place).first(n), square_in_place);
        expect(square_in_place == fast_schoolbook(f, f), name, N + 1);
        std::vector<u32> mid(f);  // the middle product into f: f of N words, g of m
        mid.resize(N);
        for (std::size_t i = n; i < N; ++i) mid[i] = random_below(P);
        const std::vector<u32> v(mid);
        const auto want = fast_middle(v, g, N - m + 1);
        middle_product<V>(mid, g, std::span(mid).first(N - m + 1));
        expect(std::equal(want.begin(), want.end(), mid.begin()), name, N + 2);
    }
}

// plan(): every kind for every N to 2^14 and around each power of two is a valid split or the
// plain plan: s a power of two in [max(2^kMinLog, L / kMaxChunks), L / 2], t <= s or s = 0, s + 2t
// <= L, L + s + t >= N (t = 0: s >= r); entry() inside the table; the compiled kinds the least of
// their plans.
void test_plans() {
    std::vector<std::size_t> lengths;
    for (std::size_t N = 1; N <= 16384; ++N) lengths.push_back(N);
    for (int k = 15; k <= kMaxLog; ++k)
        for (const std::size_t d : {std::size_t(1), std::size_t(3), (std::size_t(1) << k) / 3, (std::size_t(1) << k) / 2 + 7})
            lengths.insert(lengths.end(), {(std::size_t(1) << k) - d, (std::size_t(1) << k) + d});
    for (const std::size_t N : lengths) {
        if (N > (std::size_t(1) << kMaxLog)) continue;
        std::array<detail::ragged::Plan, 4> plans;
        const int count = detail::ragged::options(N, plans);
        expect(!plans[0].split && (std::size_t(1) << plans[0].lg) >= N && plans[0].lg >= kMinLog, "plan: plain", N);
        for (int i = 1; i < count; ++i) {
            const auto& p = plans[i];
            const std::size_t L = std::size_t(1) << p.lg, r = N - L;
            bool ok = p.split && L < N && N <= 2 * L && p.s + 2 * p.t <= L && (p.s == 0 || p.t <= p.s);
            ok &= p.s == 0 || (std::has_single_bit(p.s) && p.s >= std::max<std::size_t>(64, L / detail::ragged::kMaxChunks) && p.s <= L / 2);
            ok &= p.t ? p.s + p.t == r : p.s >= r;
            expect(ok, "plan: split", N * 10 + i);
        }
        if (N > 64) {
            const int e = detail::ragged::entry(N);
            expect(e >= 0 && e < detail::ragged::kRows * detail::ragged::kBuckets, "plan: entry", N);
        }
    }
    // The compiled table holds the least kind of each plan at the buckets' middles, as measured.
    for (int i = 0; i < detail::ragged::kRows * detail::ragged::kBuckets; ++i) {
        const int lg = kMinLog + i / detail::ragged::kBuckets, b = i % detail::ragged::kBuckets + 1;
        const std::size_t L = std::size_t(1) << lg, N = L + std::max<std::size_t>(1, (2 * b - 1) * L / 32);
        const int kind = detail::ragged::kKindDefaults[i];
        bool least = true;
        for (int k = 0; k < kind; ++k) least &= !(detail::ragged::plan(N, k) == detail::ragged::plan(N, kind));
        expect(least, "plan: compiled kind not the least of its plan", i);
    }
}

// measure_ragged at a small budget: valid kinds, the rows past max_lg start's or the new kind at
// max_lg; a huge margin keeps start; the times logged. "ntt.ragged" through tune::save and load.
void test_tuning() {
    using detail::ragged::Kinds;
    Kinds mixed{};
    for (std::size_t i = 0; i < mixed.size(); ++i) mixed[i] = std::uint8_t(i * 7 % detail::ragged::kKinds);
    const Budget quick = {9, 1, 2e3, 0};
    std::vector<RaggedTimes> log;
    const Kinds measured = measure_ragged(mixed, quick, &log);
    expect(std::ranges::all_of(measured, [](std::uint8_t k) { return k < detail::ragged::kKinds; }), "ragged tuning: valid");
    for (int b = 0; b < detail::ragged::kBuckets; ++b) {
        const int last = (quick.max_lg - kMinLog) * detail::ragged::kBuckets + b;
        for (int lg = quick.max_lg + 1; lg < kMaxLog; ++lg) {
            const int i = (lg - kMinLog) * detail::ragged::kBuckets + b;
            expect(measured[i] == (measured[last] != mixed[last] ? measured[last] : mixed[i]), "ragged tuning: past max_lg", i);
        }
    }
    for (const RaggedTimes& t : log) expect(t.kinds.size() == t.ns.size() && t.ns[0] > 0, "ragged tuning: log");
    expect(!log.empty(), "ragged tuning: log", log.size());
    const Kinds kept = measure_ragged(mixed, {8, 1, 2e3, 1e9});
    for (int i = 0; i < 3 * detail::ragged::kBuckets; ++i) {  // rows 2^6 to 2^8: start's plan
        const int lg = kMinLog + i / detail::ragged::kBuckets, b = i % detail::ragged::kBuckets + 1;
        const std::size_t L = std::size_t(1) << lg, N = L + std::max<std::size_t>(1, (2 * b - 1) * L / 32);
        int least = 0;  // the least kind with mixed[i]'s plan at N
        while (!(detail::ragged::plan(N, least) == detail::ragged::plan(N, mixed[i]))) ++least;
        expect(kept[i] == least, "ragged tuning: margin keeps start", i);
    }
    const std::string path = "fps-ntt-ragged-wisdom-" + std::to_string(std::random_device{}());
    detail::ragged::kinds.set(mixed);
    expect(fps::tune::save(path), "ragged wisdom: save");
    detail::ragged::kinds.set(detail::ragged::kKindDefaults);
    expect(fps::tune::load(path) && detail::ragged::kinds.get() == mixed, "ragged wisdom: load");
    std::remove(path.c_str());
    detail::ragged::kinds.set(detail::ragged::kKindDefaults);
}

// Long ragged lengths, the selected variant at -O2 only (the main product 2^23 and 2^24): every plan
// against the plain one, products and middle products.
void huge() {
#if !__has_feature(address_sanitizer)
    for (const std::size_t N : {(std::size_t(1) << 23) + (std::size_t(1) << 21) + 3, (std::size_t(1) << 24) + 7}) {
        const std::size_t n = N / 2 + 1, m = N + 1 - n;
        const auto f = random_poly(n, 0), g = random_poly(m, 1);
        std::array<detail::ragged::Plan, 4> plans;
        detail::ragged::options(N, plans);
        const auto plain = run<SelectedMultiply, false>(f, g, N, plans[0]);
        check_plans<SelectedMultiply, false>("selected, long", f, g, N, plain);
        const auto v = random_poly(N, 0), w = random_poly(N / 2, 0);
        const auto middle = run_middle<SelectedMultiply>(v, w, N - N / 2 + 1, plans[0]);
        check_middle<SelectedMultiply>("selected, long middle", v, w, N - N / 2 + 1, middle);
    }
#endif
}

}  // namespace

int main() {
    test_plans();
    test_tuning();
    every_length();
    for_each_variant([]<class V>(const char* name, int max_lg) {
        long_lengths<V>(name, max_lg);
        overlap<V>(name);
    });
    huge();
    return finish();
}
