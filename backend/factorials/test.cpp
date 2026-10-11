// Tests the factorials kernel: the model against the definition (fact[i] = fact[i - 1] i,
// fact[i] inverse[i] = 1), then every Chains variant this machine can run (every simd struct,
// K = 1, 2, 4, 8, both chain products, and every candidate of select.hpp) and the selected
// `tables` against the model: with the compiled defaults, and with each candidate set for every
// class it runs at. Sizes: every n up to 1100, powers of two and their neighbours up to 2^20, both
// sides of every change of the lane length C up to 2^14 and of every size class up to 2^13, random
// n, and n = P - 1, P for small P; 32 lanes with the scan's prefetch also at 2^k - 1, 2^k, 2^k + 1
// up to 2^22. Placements: tables 64-byte aligned, and at word offsets 0-15
// from it, the same or different for the two. Words before and after the tables must stay
// untouched. Then the size classes, and tune.hpp: measure() at a small budget, wisdom text and
// "factorials.variant" through tune::save / load. The sanitizer build runs a subset (kAll). Exits
// nonzero on a failure.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <set>
#include <span>
#include <string>
#include <vector>

#include <unistd.h>

#include "backend/factorials/factorials.hpp"
#include "backend/factorials/tune.hpp"
#include "backend/simd/select.hpp"

namespace {

using fps::factorials::u32;
using fps::factorials::u64;

constexpr u32 kGuard = 0xdeadbeef;
constexpr std::size_t kGuardWords = 16;
int failures = 0;

void fail(const std::string& what) {
    if (++failures <= 20) std::printf("FAIL %s\n", what.c_str());
}

// A table of n words at `offset` words past a 64-byte boundary, with guard words around it.
class Table {
public:
    Table(std::size_t n, std::size_t offset) : words_(n + 2 * kGuardWords + 32, kGuard), n_(n) {
        const auto at = reinterpret_cast<std::uintptr_t>(words_.data() + kGuardWords);
        data_ = words_.data() + kGuardWords + (64 - at % 64) % 64 / 4 + offset;
    }
    u32* data() { return data_; }
    bool guarded() const {
        for (std::size_t i = 1; i <= kGuardWords; ++i)
            if (data_[n_ - 1 + i] != kGuard || data_[-std::ptrdiff_t(i)] != kGuard) return false;
        return true;
    }

private:
    std::vector<u32> words_;
    u32* data_;
    std::size_t n_;
};

// Runs f(fact, inverse, n) on tables at the given word offsets and compares with the model's tables.
template <u32 P, class F>
void compare(const std::string& name, const std::vector<u32>& want_fact, const std::vector<u32>& want_inverse, F f,
             std::size_t fact_offset, std::size_t inverse_offset) {
    const std::size_t n = want_fact.size();
    Table fact(n, fact_offset), inverse(n, inverse_offset);
    f(fact.data(), inverse.data(), n);
    const std::string where = name + " P=" + std::to_string(P) + " n=" + std::to_string(n) + " offsets " +
                              std::to_string(fact_offset) + ", " + std::to_string(inverse_offset);
    for (std::size_t i = 0; i < n; ++i) {
        if (fact.data()[i] != want_fact[i] || inverse.data()[i] != want_inverse[i]) {
            fail(where + " i=" + std::to_string(i) + ": " + std::to_string(fact.data()[i]) + ", " +
                 std::to_string(inverse.data()[i]) + "; want " + std::to_string(want_fact[i]) + ", " +
                 std::to_string(want_inverse[i]));
            return;
        }
    }
    if (!fact.guarded() || !inverse.guarded()) fail(where + ": wrote outside the tables");
}

template <u32 P, class F>
void check(const std::string& name, std::size_t n, F f, std::size_t fact_offset = 0, std::size_t inverse_offset = 0) {
    std::vector<u32> want_fact(n), want_inverse(n);
    fps::factorials::model::tables<P>(want_fact.data(), want_inverse.data(), n);
    compare<P>(name, want_fact, want_inverse, f, fact_offset, inverse_offset);
}

// The model against the definition.
template <u32 P>
void model(std::size_t n) {
    std::vector<u32> fact(n), inverse(n);
    fps::factorials::model::tables<P>(fact.data(), inverse.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        const u64 want = i == 0 ? 1 : u64(fact[i - 1]) * i % P;
        if (fact[i] != want || u64(fact[i]) * inverse[i] % P != 1 || inverse[i] >= P)
            return fail("model P=" + std::to_string(P) + " n=" + std::to_string(n) + " i=" + std::to_string(i));
    }
}

std::vector<std::size_t> sizes(std::size_t limit, std::size_t min) {
    std::set<std::size_t> n;
    for (std::size_t i = min; i <= std::min<std::size_t>(1100, limit); ++i) n.insert(i);
    for (int k = 6; k <= 20; ++k)
        for (std::size_t m : {(std::size_t(1) << k) - 1, std::size_t(1) << k, (std::size_t(1) << k) + 1})
            if (m >= min && m <= limit) n.insert(m);
    std::mt19937_64 rng(20261011);
    for (int t = 0; t < 20; ++t) n.insert(min + rng() % (std::min<std::size_t>(limit, 1 << 18) - min + 1));
    return {n.begin(), n.end()};
}

using fps::factorials::Step;

// The sanitizer build compiles 5x slower than -O2: it runs every K and both chain products, at
// fewer sizes and primes; the -O2 build runs everything.
#if defined(__SANITIZE_ADDRESS__)
#define FACTORIALS_TEST_ALL 0
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define FACTORIALS_TEST_ALL 0
#endif
#endif
#if !defined(FACTORIALS_TEST_ALL)
#define FACTORIALS_TEST_ALL 1
#endif
constexpr bool kAll = FACTORIALS_TEST_ALL;

template <class S, int K, Step step, bool prefetch = false>
std::string variant() {
    return std::string("Chains<") + S::name + ", " + std::to_string(K) + (step == Step::shoup ? ", shoup" : ", montgomery") +
           (prefetch ? ", prefetch>" : ">");
}

template <class S, int K, u32 P, Step step, bool prefetch = false>
void chains(std::size_t limit) {
    using C = fps::factorials::Chains<S, K, P, step, prefetch>;
    const std::string name = variant<S, K, step, prefetch>();
    const auto run = [](u32* fact, u32* inverse, std::size_t n) { C::run(fact, inverse, n); };
    const std::vector<std::size_t> base = sizes(limit, C::kMinSize);
    std::set<std::size_t> n(base.begin(), base.end());
    for (std::size_t m = C::kMinSize + 1; m <= std::min<std::size_t>(limit, 1 << 14); ++m)
        if (C::chunk(m) != C::chunk(m - 1)) n.insert({m - 1, m});
    if (limit == P) n.insert({P - 1, P});
    for (std::size_t m : n) check<P>(name, m, run);
}

// Placements, on fewer sizes: the lane grid starts at h = (16 - offset) % S::lanes words, and
// positions below h come from constant tables.
template <class S, int K, u32 P, Step step>
void placements(std::size_t limit) {
    using C = fps::factorials::Chains<S, K, P, step>;
    std::set<std::size_t> n;
    for (std::size_t m = C::kMinSize; m <= 300; ++m) n.insert(m);
    for (std::size_t m = 301; m <= std::min<std::size_t>(limit, 1 << 12); ++m)
        if (C::chunk(m) != C::chunk(m - 1)) n.insert({m - 1, m});
    for (int k = 9; k <= 17; ++k) n.insert({(std::size_t(1) << k) - 1, std::size_t(1) << k, (std::size_t(1) << k) + 1});
    std::mt19937_64 rng(20261012);
    for (int t = 0; t < 8; ++t) n.insert(C::kMinSize + rng() % ((1 << 18) - C::kMinSize));
    if (limit == P) n.insert({P - 16, P - 1, P});
    const std::pair<std::size_t, std::size_t> offsets[] = {{1, 1}, {4, 4}, {12, 12}, {15, 15}, {0, 8}, {3, 13}, {9, 2}};
    const std::string name = variant<S, K, step>();
    const auto run = [](u32* fact, u32* inverse, std::size_t m) { C::run(fact, inverse, m); };
    for (std::size_t m : n)
        if (m <= limit)
            for (auto [f, i] : offsets) check<P>(name, m, run, f, i);
}

// 32 lanes with the scan's prefetch (issue #70): sizes just below, at and above 2^18 .. 2^max_lg,
// at the placements' offsets and both tables aligned.
template <class S, Step step>
void wide(int max_lg) {
    constexpr u32 P = 998244353;
    constexpr int K = 32 / S::lanes;
    using C = fps::factorials::Chains<S, K, P, step, true>;
    const std::string name = variant<S, K, step, true>();
    const auto run = [](u32* fact, u32* inverse, std::size_t m) { C::run(fact, inverse, m); };
    const std::pair<std::size_t, std::size_t> offsets[] = {{0, 0}, {1, 1}, {4, 4}, {12, 12},
                                                           {15, 15}, {0, 8}, {3, 13}, {9, 2}};
    for (int k = 18; k <= max_lg; ++k)
        for (std::size_t n : {(std::size_t(1) << k) - 1, std::size_t(1) << k, (std::size_t(1) << k) + 1}) {
            std::vector<u32> fact(n), inverse(n);
            fps::factorials::model::tables<P>(fact.data(), inverse.data(), n);
            for (auto [f, i] : offsets) compare<P>(name, fact, inverse, run, f, i);
        }
}

template <class S>
void test_isa() {
    const int before = failures;
    constexpr Step shoup = Step::shoup, montgomery = Step::montgomery;
    if constexpr (S::lanes > 1) chains<S, 1, 998244353, shoup>(1 << 16);
    chains<S, 2, 998244353, shoup>(1 << 20);
    chains<S, 2, 998244353, montgomery>(kAll ? 1 << 20 : 1 << 16);
    chains<S, 4, 998244353, shoup>(kAll ? 1 << 20 : 1 << 16);
    chains<S, 2, 4099, shoup>(4099);
    chains<S, 2, 1073741789, montgomery>(1 << 12);  // the largest prime below 2^30
    placements<S, 2, 998244353, shoup>(1 << 20);
    placements<S, 2, 4099, shoup>(4099);
    if constexpr (S::lanes == 16) wide<S, shoup>(kAll ? 22 : 19), wide<S, montgomery>(kAll ? 22 : 19);
    if constexpr (S::lanes >= 4 && S::lanes < 16 && kAll) wide<S, shoup>(20);
    if constexpr (kAll) {
        chains<S, 4, 998244353, montgomery>(1 << 16);
        chains<S, 8, 998244353, shoup>(1 << 16);
        chains<S, 2, 4099, montgomery>(4099);
        chains<S, 4, 65537, shoup>(65537);
        chains<S, 4, 1073741789, shoup>(1 << 12);
        placements<S, 2, 998244353, montgomery>(1 << 16);
    }
    std::printf("%-9s %s\n", S::name, failures == before ? "ok" : "FAIL");
}

namespace fx = fps::factorials;

// Every candidate of select.hpp, also those test_isa leaves out (K = 1 Montgomery, K = 8, prefetch).
template <class... V>
void candidates(fx::Candidates<V...>) {
    (chains<typename V::Isa, V::vectors, 998244353, V::scan, V::prefetch>(kAll ? 1 << 18 : 1 << 14), ...);
}

const auto selected = [](u32* fact, u32* inverse, std::size_t n) {
    fx::tables<998244353>(std::span(fact, n), std::span(inverse, n));
};

// `tables` with each variant set for every class it runs at (the loop below), on both sides of
// every class boundary up to 2^13 and at a few larger sizes.
void dispatch() {
    const fx::Choices saved = fx::variants.get();
    for (std::size_t v = 0; v < fx::kMinSizes.size(); ++v) {
        fx::Choices forced{};
        for (int c = 0; c < fx::kClasses; ++c) forced[c] = fx::kMinSizes[v] <= fx::class_start(c) ? std::uint8_t(v) : 0;
        if (!fx::valid(forced)) fail("a forced table is not valid");
        fx::variants.set(forced);
        std::set<std::size_t> n = {0, 1, 2, 16, (std::size_t(1) << 16) + 1, (std::size_t(1) << 18) - 1};
        for (int c = 0; fx::class_end(c) <= 1 << 13; ++c) n.insert({fx::class_start(c), fx::class_end(c)});
        for (std::size_t m : n) check<998244353>("tables, variant " + fx::variant_names()[v], m, selected);
    }
    fx::variants.set(saved);
}

void classes() {
    for (int c = 0; c + 1 < fx::kClasses; ++c) {
        if (fx::class_end(c) + 1 != fx::class_start(c + 1) || fx::class_start(c) > fx::class_end(c))
            fail("class " + std::to_string(c) + " does not end where class " + std::to_string(c + 1) + " starts");
        for (std::size_t n : {fx::class_start(c), fx::class_end(c)})
            if (fx::size_class(n) != c) fail("size_class(" + std::to_string(n) + ") is not " + std::to_string(c));
        for (int count : {1, 2, 4}) {
            const std::vector<std::size_t> p = fx::points(c, count);
            if (p.back() != fx::class_end(c) || p.front() < fx::class_start(c) || !std::ranges::is_sorted(p))
                fail("points of class " + std::to_string(c));
        }
    }
    if (fx::size_class(17) != 0 || fx::size_class(std::size_t(1) << 29) != fx::kClasses - 1 ||
        fx::class_start(fx::kClasses - 1) != (std::size_t(7) << 23) + 1)
        fail("the first or last class");
}

// measure() at a small budget: valid; start's above 2^sparse_lg without a sparse part; with one,
// a class takes the variant of its octave's size (or of 2^sparse_lg above); start's everywhere
// with a huge margin. Then the wisdom text and "factorials.variant" through tune::save / load.
void tuning() {
    const fx::Choices measured = fx::measure(fx::kDefaults, {9, 9, 2, 1, 2e3, 0, 0});
    if (!fx::valid(measured)) fail("measure: not a valid table");
    for (int c = 0; c < fx::kClasses; ++c)
        if (fx::class_end(c) > 512 && measured[c] != fx::kDefaults[c]) fail("measure: changed a class above 2^9");
    std::vector<fx::ClassTimes> log;
    const fx::Choices sparse = fx::measure(fx::kDefaults, {8, 10, 2, 1, 2e3, 0, 0}, &log);
    if (!fx::valid(sparse) || log.size() < 2 || log.back().n != std::vector<std::size_t>{1024})
        fail("measure: sparse sizes");
    for (int c = fx::size_class(257); c < fx::kClasses; ++c) {  // a fastest: the clock may tie two
        const fx::ClassTimes& t = log[log.size() - 2 + (fx::class_end(c) > 512)];
        const auto chosen = std::ranges::find(t.ids, int(sparse[c]));
        if (chosen == t.ids.end() || t.ns[0][chosen - t.ids.begin()] != *std::ranges::min_element(t.ns[0]))
            fail("measure: class " + std::to_string(c) + " is not its octave's fastest");
    }
    if (fx::measure(fx::kDefaults, {9, 11, 2, 1, 2e3, 1e9, 1e9}) != fx::kDefaults)
        fail("measure: changed within the margin");

    if (fx::parse(fx::text(measured)) != measured) fail("wisdom text does not read back");
    std::string text = fx::text(measured);
    const std::string body = text.substr(text.find(' '));
    if (fx::parse(std::to_string(fx::kSignature ^ 1) + body)) fail("parse took another signature");
    if (fx::parse(text + " 0")) fail("parse took a value too many");
    if (fx::parse(text.substr(0, text.rfind(' ')))) fail("parse took a value too few");
    fx::Choices bad = measured;
    bad[0] = std::uint8_t(fx::kMinSizes.size());
    if (fx::parse(fx::text(bad))) fail("parse took a variant that does not exist");
    std::size_t widest = 0;
    for (std::size_t v = 0; v < fx::kMinSizes.size(); ++v)
        if (fx::kMinSizes[v] > fx::kMinSizes[widest]) widest = v;
    bad = measured;
    bad[0] = std::uint8_t(widest);
    if (fx::kMinSizes[widest] > fx::class_start(0) && fx::parse(fx::text(bad)))
        fail("parse took a variant below its least n");

    const std::string path =
        (std::filesystem::temp_directory_path() / ("fps-factorials-" + std::to_string(::getpid()))).string();
    fx::variants.set(measured);
    if (!fps::tune::save(path)) fail("save");
    fx::variants.set(fx::kDefaults);
    if (!fps::tune::load(path) || fx::variants.get() != measured)
        fail("factorials.variant did not load back from wisdom");
    std::filesystem::remove(path);
    fx::variants.set(fx::kDefaults);
}

}  // namespace

int main() {
    for (std::size_t n : {0, 1, 2, 3, 100, 4099}) model<4099>(n);
    for (std::size_t n : {0, 1, 2, 1000, 1 << 16}) model<998244353>(n);
    fps::simd::for_each(fps::simd::All{}, []<class S> { test_isa<S>(); });
    candidates(fx::Selected{});
    for (std::size_t n : sizes(1 << 20, 0)) check<998244353>("tables", n, selected);
    for (int c = 0; c < fx::kClasses && fx::class_end(c) <= 1 << 22; ++c)  // where the variant changes
        if (c == 0 || fx::kDefaults[c] != fx::kDefaults[c - 1])
            for (std::size_t n : {fx::class_start(c) - 1, fx::class_start(c)}) check<998244353>("tables", n, selected);
    dispatch();
    classes();
    tuning();
#if FACTORIALS_TEST_ALL
    const auto small = [](u32* fact, u32* inverse, std::size_t n) {
        fps::factorials::tables<4099>(std::span(fact, n), std::span(inverse, n));
    };
    for (std::size_t n : {std::size_t(0), std::size_t(4098), std::size_t(4099)}) check<4099>("tables", n, small);
#endif
    const std::size_t large[] = {std::size_t(1) << 17, std::size_t(1) << 19, (std::size_t(1) << 20) + 3,
                                 (std::size_t(5) << 20) + 17};
    for (std::size_t n : large)
        for (std::size_t offset : {0, 4, 7}) check<998244353>("tables", n, selected, offset, offset);
    std::printf("%s\n", failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
