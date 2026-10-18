// Tests for backend/ntt: middle products by the truncated plan (issue #155 round 3, truncated.hpp's
// Middle products). Against schoolbook at the edges of the nodes' rounding at 2^8 .. 2^12, g of 1,
// 2, 3 words, about half, all but a few and all of f, inputs at P - 1, through detail::middle (out
// ending its allocation) and middle_into with work of its own (a split's top product); against the
// plain plan to 2^21 (1.5, 1.625, 1.75, 1.875 2^k and edges); out over f, g or both; the API with the
// middle table truncated everywhere, a split's top middle product truncated; choose_middle and the
// compiled table; the middle table's tuning and wisdom. Every variant (test_truncated.cpp's list).
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
// out each ending its allocation (ASan sees a word past them), out full of garbage first; and
// middle_into with work of its own, which must agree. f and g unchanged.
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
    std::vector<u32> c(count + 1, 7);
    const auto [roots, inverse_roots] = detail::tables<V>(detail::table_log(p));
    detail::middle_into<V>(fw.begin(), n, gw.begin(), m, c.data(), count, work.data()->words, p, roots, inverse_roots);
    expect(std::equal(out.begin(), out.begin() + count, c.begin()) && c[count] == 7, "middle_into", n);
    expect(std::equal(f.begin(), f.begin() + n, fw.begin()) && std::equal(g.begin(), g.end(), gw.begin()), "factors kept", n);
    return {out.begin(), out.begin() + count};
}

// Lengths at the edges of the rounding at L = 2^lg (test_truncated.cpp's): r = N - L next to the
// multiples of kMinNode and of the levels' chunks, and near L.
std::vector<std::size_t> edge_lengths(int lg) {
    const std::size_t L = std::size_t(1) << lg;
    std::vector<std::size_t> Ns;
    for (const std::size_t r : {std::size_t(1), std::size_t(2), dr::kMinNode - 1, dr::kMinNode, dr::kMinNode + 1, L / 8 - 1,
                                L / 4, L / 4 + 1, L / 2 - 1, L / 2, L / 2 + 1, L / 2 + dr::kMinNode + 3, L / 2 + L / 8,
                                3 * L / 4 - 1, 3 * L / 4, 3 * L / 4 + 5, L - L / 8, L - 2 * dr::kMinNode, L - dr::kMinNode - 1})
        if (r < L && dr::truncates(L + r)) Ns.push_back(L + r);
    return Ns;
}

// The sizes of g for f of N words: 1, 2, 3, about half, all but a few, all.
std::vector<std::size_t> g_sizes(std::size_t N) {
    return {1, 2, 3, N / 2, N / 2 + 1, N - 200, N - 17, N - 1, N};
}

// The edge lengths of 2^8 .. 2^12 against schoolbook: random inputs and P - 1.
template <class V>
void test_short(const char* name, int) {
    for (int lg = 8; lg <= 12; ++lg) {
        for (const std::size_t N : edge_lengths(lg)) {
            const dr::Plan p = dr::plan(N, dr::kTruncated);
            expect(p.truncated, "plan truncated", N);
            const int kind = N % 3 == 0 ? 1 : 0;
            const auto f = random_poly(N, kind);
            for (const std::size_t m : g_sizes(N)) {
                const auto g = random_poly(m, kind);
                const std::size_t count = N - m + 1;
                expect(run_middle<V>(f, g, count, p, N % 5) == fast_middle(f, g, count), name, N * 1000 + m % 1000);
            }
        }
    }
}

// 1.5, 1.625, 1.75, 1.875 2^k for k = 13 .. 20 within max_lg, and the edges of 2^14 and 2^16, against
// the plain plan (one word checked against its definition).
template <class V>
void test_long(const char* name, int max_lg) {
    std::vector<std::size_t> Ns = edge_lengths(14);
    for (const std::size_t N : edge_lengths(16)) Ns.push_back(N);
    for (int k = 13; k <= std::min(max_lg - 1, 20); ++k) {
        const std::size_t L = std::size_t(1) << k;
        for (const std::size_t N : {L + L / 2, L + 5 * L / 8, 2 * L - L / 4, 2 * L - L / 8, L + 3 * L / 8 + 77})
            if (dr::truncates(N)) Ns.push_back(N);
    }
    for (const std::size_t N : Ns) {
        if (int(std::bit_width(N - 1)) > max_lg) continue;
        const auto f = random_poly(N, N % 2);
        for (const std::size_t m : {N / 2, N - N / 3 - 5, std::size_t(7)}) {
            const auto g = random_poly(m, (N + m) % 2);
            const std::size_t count = N - m + 1;
            const std::vector<u32> plain = run_middle<V>(f, g, count, dr::plan(N, dr::kPlain), 3);
            const std::size_t i = random_below(count);
            u64 sum = 0;
            for (std::size_t j = 0; j < m; ++j) sum = (sum + u64(f[i + j]) * g[m - 1 - j]) % P;
            expect(plain[i] == sum, name, N);
            expect(run_middle<V>(f, g, count, dr::plan(N, dr::kTruncated), N % 7) == plain, name, N * 10 + 1);
        }
    }
}

// The middle half of the table, every entry set to kind (the products' half as it was).
void set_middle(int kind) {
    dr::Kinds kinds = dr::kinds.get();
    std::fill(kinds.begin() + dr::kTable, kinds.end(), std::uint8_t(kind));
    dr::kinds.set(kinds);
}

// The API with the middle table truncated everywhere: middle_product against schoolbook, out
// apart, over f, over g and over both; a split's top middle product (choose_middle) truncated.
void test_api() {
    set_middle(dr::kTruncated);
    for (std::size_t N = 300; N <= 4300; N += 97) {
        const auto v = random_poly(N, N % 2), g = random_poly(N / 2, 0);
        const std::size_t m = g.size(), count = N - m + 1;
        const auto want = fast_middle(v, g, count);
        expect(dr::choose_middle(N).truncated == dr::truncates(N), "choose_middle truncated", N);
        std::vector<u32> out(count + 1, 9);
        middle_product(v, g, std::span(out).first(count));
        expect(std::equal(want.begin(), want.end(), out.begin()) && out[count] == 9, "api middle", N);
        std::vector<u32> both(N + m);  // f then g; out at f, at g, across both
        for (const std::size_t at : {std::size_t(0), N, N / 2, N - count / 2}) {
            std::copy(v.begin(), v.end(), both.begin());
            std::copy(g.begin(), g.end(), both.begin() + N);
            std::vector<u32> wide(both);
            wide.resize(std::max(both.size(), at + count));
            middle_product(std::span(wide).first(N), std::span(wide).subspan(N, m), std::span(wide).subspan(at, count));
            expect(std::ranges::equal(std::span(wide).subspan(at, count), want), "api middle overlap", N * 10 + at % 10);
        }
        for (const int kind : {int(dr::kFloor), int(dr::kTop)}) {  // a split whose top middle product is truncated
            const dr::Plan p = dr::plan(N, kind);
            if (!p.split) continue;
            std::vector<detail::Line> work(detail::middle_words(m, count, p) / 16 + 1);
            std::vector<u32> c(count);
            const auto [roots, inverse_roots] = detail::tables<SelectedMultiply>(p.lg);
            detail::middle_into<SelectedMultiply>(v.data(), N, g.data(), m, c.data(), count, work.data()->words, p, roots,
                                                  inverse_roots);
            expect(c == want, "api middle split", N * 10 + kind);
        }
    }
    dr::kinds.set(dr::kKindDefaults);
}

// choose_middle as the compiled table says; its kinds the least of their plans at the buckets'
// middles (measure_ragged stores those); the products' half unchanged by a middle entry.
void test_choose() {
    for (std::size_t N = 65; N <= (std::size_t(1) << kMaxLog); N = N * 9 / 8 + 1) {
        const int v = dr::kKindDefaults[dr::kTable + dr::entry(N)];
        const dr::Plan p = dr::choose_middle(N);
        expect(p == dr::plan(N, dr::middle_kind(v)), "choose_middle", N);
        expect(!dr::choose(N).truncated, "choose", N);
    }
    expect(dr::choose_middle(64) == dr::plan(64, dr::kPlain) && dr::choose_middle(1) == dr::plan(1, dr::kPlain), "short");
    for (int i = 0; i < dr::kTable; ++i) {
        const int lg = kMinLog + i / dr::kBuckets, b = i % dr::kBuckets + 1;
        const std::size_t L = std::size_t(1) << lg, N = L + std::max<std::size_t>(1, (2 * b - 1) * L / 32);
        const int kind = dr::middle_kind(dr::kKindDefaults[dr::kTable + i]);
        bool least = true;
        for (int k = 0; k < kind; ++k) least &= !(dr::plan(N, k) == dr::plan(N, kind));
        expect(least && dr::kKindDefaults[dr::kTable + i] < dr::kKinds, "compiled middle kind not the least of its plan", i);
    }
}

// The least kind with kind's plan at N.
int least_kind(std::size_t N, int kind) {
    int least = 0;
    while (!(dr::plan(N, least) == dr::plan(N, kind))) ++least;
    return least;
}

// measure_ragged's middle half at a small budget: valid kinds, the rows past max_lg start's or the
// new kind at max_lg, a huge margin keeps start's plan, the middle times logged; wisdom keeps it.
void test_tuning() {
    dr::Kinds mixed{};
    for (std::size_t i = 0; i < mixed.size(); ++i) mixed[i] = std::uint8_t((i * 5 + i / 16) % dr::kKinds);
    const Budget quick = {9, 1, 2e3, 0};
    std::vector<RaggedTimes> log;
    const dr::Kinds measured = measure_ragged(mixed, quick, &log);
    for (int i = 0; i < dr::kTable; ++i) expect(measured[dr::kTable + i] < dr::kKinds, "middle tuning: valid", i);
    for (int b = 0; b < dr::kBuckets; ++b) {
        const int last = dr::kTable + (quick.max_lg - kMinLog) * dr::kBuckets + b;
        for (int lg = quick.max_lg + 1; lg < kMaxLog; ++lg) {
            const int i = dr::kTable + (lg - kMinLog) * dr::kBuckets + b;
            expect(measured[i] == (measured[last] != mixed[last] ? measured[last] : mixed[i]), "middle tuning: past max_lg", i);
        }
    }
    for (const RaggedTimes& t : log)
        expect(t.middle.size() == t.kinds.size() && std::ranges::all_of(t.middle, [](double x) { return x > 0; }),
               "middle tuning: log");
    const dr::Kinds kept = measure_ragged(mixed, {8, 1, 2e3, 1e9});
    for (int i = 0; i < 3 * dr::kBuckets; ++i) {  // rows 2^6 to 2^8: start's plan, its least kind
        const int lg = kMinLog + i / dr::kBuckets, b = i % dr::kBuckets + 1;
        const std::size_t L = std::size_t(1) << lg, N = L + std::max<std::size_t>(1, (2 * b - 1) * L / 32);
        const int want = std::ranges::count_if(log, [&](const RaggedTimes& t) { return t.lg == lg && t.bucket == b; })
                             ? least_kind(N, dr::middle_kind(mixed[dr::kTable + i]))
                             : int(dr::kPlain);  // one plan only
        expect(kept[dr::kTable + i] == want, "middle tuning: margin keeps start", i);
    }
    const std::string path = "fps-ntt-middle-wisdom-" + std::to_string(std::random_device{}());
    dr::kinds.set(mixed);
    expect(fps::tune::save(path), "middle wisdom: save");
    dr::kinds.set(dr::kKindDefaults);
    expect(fps::tune::load(path) && dr::kinds.get() == mixed, "middle wisdom: load");
    std::remove(path.c_str());
    dr::kinds.set(dr::kKindDefaults);
}

// Long lengths, the selected variant at -O2 only (2L = 2^24 and 2^25): truncated against plain.
void huge() {
#if !__has_feature(address_sanitizer)
    const std::size_t M = std::size_t(1) << 22;
    for (const std::size_t N : {2 * M + M + 3, 4 * M + 3 * M + 1}) {  // 1.5 2^23 + 3, 1.75 2^24 + 1
        const auto f = random_poly(N, 0), g = random_poly(N / 2, 1);
        const std::size_t count = N - g.size() + 1;
        const auto plain = run_middle<SelectedMultiply>(f, g, count, dr::plan(N, dr::kPlain), 0);
        expect(run_middle<SelectedMultiply>(f, g, count, dr::plan(N, dr::kTruncated), 1) == plain, "huge middle", N);
    }
#endif
}

}  // namespace

int main() {
    each_variant([]<class V>(const char* name, int max_lg) {
        test_short<V>(name, max_lg);
        test_long<V>(name, max_lg);
    });
    test_api();
    test_choose();
    test_tuning();
    huge();
    return finish();
}
