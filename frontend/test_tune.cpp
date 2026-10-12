// Tests of fps/tune.hpp: save, load and tune_all() round trips; wisdom from another CPU or target,
// malformed or missing, changes nothing; an unwritable path; every registered choice is in the
// wisdom. f * g and Factorials give the O(n^2) reference's results under the compiled defaults,
// under wisdom that forces each candidate of each choice, and after tune_all(), which runs while
// another thread computes. test_tune_no_tuner.cpp builds with -DFPS_NO_TUNER.
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#include "fps/fps.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = 998244353;
using Poly = fps::Polynomial<P>;
namespace fs = std::filesystem;

int failures = 0;

void check(bool ok, const std::string& what, int line) {
    if (ok) return;
    if (++failures <= 20) std::printf("FAIL line %d: %s\n", line, what.c_str());
}
#define CHECK(cond, ...) check(bool(cond), std::string(#cond) __VA_OPT__(+" (" + __VA_ARGS__ + ")"), __LINE__)

const fs::path kDir = fs::temp_directory_path() / ("fps-test-tune-" + std::to_string(getpid()));

std::string read(const fs::path& path) {
    std::ifstream in(path);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

void write(const fs::path& path, const std::string& text) { std::ofstream(path) << text; }

// The active choices, as save_wisdom() writes them.
std::string active() {
    const fs::path path = kDir / "active";
    CHECK(fps::save_wisdom(path.string()));
    return read(path);
}

std::vector<std::string> lines(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream s(text);
    for (std::string line; std::getline(s, line);) out.push_back(line);
    return out;
}

std::string joined(const std::vector<std::string>& lines) {
    std::string out;
    for (const std::string& line : lines) out += line + '\n';
    return out;
}

// text with the line of choice `name` replaced by "<name> <values>".
std::string with(const std::string& text, const std::string& name, const std::string& values) {
    std::vector<std::string> out = lines(text);
    bool found = false;
    for (std::string& line : out)
        if (line.starts_with(name + ' ')) line = name + ' ' + values, found = true;
    CHECK(found, name);
    return joined(out);
}

// Loads text from a file; true if load_wisdom() took it.
bool load(const std::string& text) {
    const fs::path path = kDir / "load";
    write(path, text);
    return fps::load_wisdom(path.string());
}

std::string repeat(const std::string& s, int times) {
    std::string out;
    for (int i = 0; i < times; ++i) out += s;
    return out;
}

std::mt19937_64 rng(66);

Vec random_vec(std::size_t n) {
    Vec a(n);
    for (u32& c : a) c = u32(rng() % P);
    return a;
}

Vec reference_product(const Vec& a, const Vec& b) {
    if (a.empty() || b.empty()) return {};
    Vec c(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = u32((c[i + j] + u64(a[i]) * b[j]) % P);
    return c;
}

// Shapes on every path of f * g: both sides of small::beats_ntt's cutoffs on every target
// (n = m up to 129, min(n, m) up to 257 next to 4096), and NTT lengths.
std::vector<std::pair<std::size_t, std::size_t>> shapes() {
    const std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33,
                                            63, 64, 65, 103, 104, 105, 127, 128, 129, 255, 256, 257};
    std::vector<std::pair<std::size_t, std::size_t>> out;
    for (const std::size_t n : sizes) out.push_back({n, n});
    for (const std::size_t m : sizes) out.push_back({4096, m});
    for (const std::size_t m : {37, 38, 39, 179, 180, 181}) out.push_back({m, 3000});
    out.push_back({1023, 1025});
    out.push_back({4096, 4096});
    return out;
}

struct Case {
    Vec a, b, product;
};

std::vector<Case> cases() {
    std::vector<Case> out;
    for (const auto& [n, m] : shapes()) {
        Case c{random_vec(n), random_vec(m), {}};
        c.product = reference_product(c.a, c.b);
        out.push_back(std::move(c));
    }
    return out;
}

// i! for i < n, by running products.
Vec reference_factorials(std::size_t n) {
    Vec fact(n);
    u64 f = 1;
    for (std::size_t i = 0; i < n; ++i) fact[i] = u32(f = i ? f * i % P : 1);
    return fact;
}

// Table sizes in every class of factorials' choice up to 2^17: a class's end and one past it.
std::vector<std::size_t> table_sizes() {
    std::vector<std::size_t> out = {0, 1, 2, 16, 17, 18};
    for (int c = 0; fps::factorials::class_end(c) <= (std::size_t(1) << 17); ++c)
        for (const std::size_t n : {fps::factorials::class_end(c), fps::factorials::class_end(c) + 1}) out.push_back(n);
    return out;
}

// f * g and Factorials against the references, under the active choices.
int check_results(const std::vector<Case>& products, const Vec& fact, const std::string& when) {
    int wrong = 0;
    for (const Case& c : products) {
        const Poly h = Poly(std::span<const u32>(c.a)) * Poly(std::span<const u32>(c.b));
        const bool ok = std::ranges::equal(h.span(), c.product);
        wrong += !ok;
        CHECK(ok, when + ", " + std::to_string(c.a.size()) + " x " + std::to_string(c.b.size()));
    }
    for (const std::size_t n : table_sizes()) {
        const fps::Factorials<P> table(n);
        bool ok = true;
        for (std::size_t i = 0; i < n; ++i) ok &= table[i] == fact[i] && u64(table.inverse(i)) * fact[i] % P == 1;
        wrong += !ok;
        CHECK(ok, when + ", Factorials(" + std::to_string(n) + ")");
    }
    return wrong;
}

// Every registered choice is in the wisdom, after the format, CPU and target lines.
void test_format(const std::string& wisdom) {
    const std::vector<std::string> l = lines(wisdom);
    CHECK(l.size() == 6, std::to_string(l.size()) + " lines");
    if (l.size() < 3) return;
    CHECK(l[0] == "fps wisdom 1", l[0]);
    CHECK(l[1] == "cpu " + fps::tune::cpu_name(), l[1]);
    CHECK(l[2] == "target " + std::string(fps::tune::target_name()), l[2]);
    for (const std::string name : {"small.variant ", "small.ntt ", "factorials.variant "})
        CHECK(std::ranges::count_if(l, [&](const std::string& line) { return line.starts_with(name); }) == 1, name);
}

// Wisdom load_wisdom() must refuse, leaving the active choices as they were.
void test_refused(const std::string& wisdom) {
    const std::vector<std::string> l = lines(wisdom);
    std::vector<std::pair<std::string, std::string>> refused = {
        {"empty file", ""},
        {"a cut header", joined({l[0], l[1], l[2]}).substr(0, 20)},
        {"no target line", joined({l[0], l[1]})},
        {"another format", "fps wisdom 2\n" + joined({l.begin() + 1, l.end()})},
        {"another CPU", joined({l[0], "cpu Another CPU 9000"}) + joined({l.begin() + 2, l.end()})},
        {"a choice line without values", wisdom + "small.ntt\n"},
        {"too few values", with(wisdom, "small.ntt", "1 2 3")},
        {"a value that is no number", with(wisdom, "small.ntt", repeat("1 ", 19) + "x")},
        {"a negative NTT time", with(wisdom, "small.ntt", "-1" + repeat(" 1", 19))},
        {"a variant that does not exist",
         with(wisdom, "small.variant", std::to_string(fps::small::kSignature) + " 255" + repeat(" 0", fps::small::kClasses - 1))},
        {"another candidate list", with(wisdom, "factorials.variant", "12345" + repeat(" 0", fps::factorials::kClasses))},
    };
    for (const std::string_view other : {"scalar", "sse", "avx2", "avx512", "zen5", "arm"})
        if (other != fps::tune::target_name())
            refused.push_back({"target " + std::string(other),
                               joined({l[0], l[1], "target " + std::string(other)}) + joined({l.begin() + 3, l.end()})});
    // Good lines, then a malformed one: nothing is set, not even the good ones.
    refused.push_back({"good lines before a bad one", with(wisdom, "small.ntt", repeat("1000 ", 19) + "1000") + "small.variant x\n"});

    for (const auto& [what, text] : refused) {
        CHECK(!load(text), what);
        CHECK(active() == wisdom, what + ": the choices changed");
    }
    CHECK(!fps::load_wisdom((kDir / "missing").string()));
    CHECK(active() == wisdom, "a missing file changed the choices");
}

void test_unwritable() {
    CHECK(!fps::save_wisdom((kDir / "no-such-directory" / "wisdom").string()));
    CHECK(!fs::exists(kDir / "no-such-directory"));
    CHECK(!fps::save_wisdom(kDir.string()), "a directory");
    CHECK(!fps::save_wisdom(""), "an empty path");
}

// Wisdom that sets each candidate of each choice everywhere it can run: f * g and Factorials still
// give the reference results, and saving gives back the file loaded.
void test_forced(const std::string& wisdom, const std::vector<Case>& products, const Vec& fact) {
    std::vector<std::pair<std::string, std::string>> forced;
    for (int v = 0; v < fps::small::kCandidates; ++v) {
        fps::small::Choices choices;
        choices.fill(std::uint8_t(v));
        forced.push_back({"small.variant " + std::to_string(v), with(wisdom, "small.variant", fps::small::text(choices))});
    }
    for (const double scale : {0.125, 8.0}) {  // the NTT 8x faster or slower: other cutoffs
        std::array<double, 20> times = fps::small::kNttDefaults;
        for (double& t : times) t *= scale;
        forced.push_back({"small.ntt x" + std::to_string(scale), with(wisdom, "small.ntt", fps::tune::text(times))});
    }
    for (std::size_t v = 0; v < fps::factorials::kMinSizes.size(); ++v) {
        fps::factorials::Choices choices;
        for (int c = 0; c < fps::factorials::kClasses; ++c)
            choices[c] = std::uint8_t(fps::factorials::kMinSizes[v] <= fps::factorials::class_start(c) ? v : 0);
        forced.push_back(
            {"factorials.variant " + std::to_string(v), with(wisdom, "factorials.variant", fps::factorials::text(choices))});
    }
    for (const auto& [what, text] : forced) {
        CHECK(load(text), what);
        CHECK(active() == text, what + ": saved wisdom differs from the file loaded");
        check_results(products, fact, what);
    }
    CHECK(load(wisdom));
    CHECK(active() == wisdom, "loading the first wisdom back");
}

// tune_all() while another thread computes products: every result is right, before and after.
void test_tune(const std::string& wisdom, const std::vector<Case>& products, const Vec& fact) {
    bool done = false;
    int wrong = 0;
    std::thread worker([&] {
        while (!std::atomic_ref(done).load()) wrong += check_results(products, fact, "during tune_all()");
    });
    fps::tune_all();
    std::atomic_ref(done).store(true);
    worker.join();
    CHECK(wrong == 0);
    check_results(products, fact, "after tune_all()");

    const std::string tuned = active();
    const std::vector<std::string> l = lines(tuned), w = lines(wisdom);
    CHECK(l.size() == w.size() && std::equal(w.begin(), w.begin() + 3, l.begin()), "tuned wisdom header");
    std::printf("choices changed by tune_all(): %d of 3\n",
                int(std::ranges::count(std::views::iota(std::size_t(3), std::min(l.size(), w.size())), false,
                                       [&](std::size_t i) { return l[i] == w[i]; })));
    fps::tune_all();  // once per process: no second measurement
    CHECK(active() == tuned, "a second tune_all() changed the choices");

    // The round trip: back to the defaults, then the tuned wisdom again.
    CHECK(load(wisdom));
    CHECK(active() == wisdom);
    CHECK(load(tuned));
    CHECK(active() == tuned);
}

}  // namespace

int main() {
    fs::create_directories(kDir);
    const std::vector<Case> products = cases();
    const Vec fact = reference_factorials(table_sizes().back());

    check_results(products, fact, "compiled defaults");
    const std::string wisdom = active();
    test_format(wisdom);
    if (lines(wisdom).size() == 6) {
        CHECK(load(wisdom), "its own wisdom");
        CHECK(active() == wisdom);
        test_refused(wisdom);
        test_unwritable();
        test_forced(wisdom, products, fact);
        test_tune(wisdom, products, fact);
    }
    fs::remove_all(kDir);
    std::printf("%s\n", failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
