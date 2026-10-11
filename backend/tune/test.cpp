// Tests backend/tune: tables (defaults, set, readers on other threads), text and parse, the
// registry and run(), wisdom (round trip; files from another CPU or target, malformed or missing,
// change nothing), time() and time_each(). test_no_tuner.cpp builds with -DFPS_NO_TUNER. Exits
// nonzero on a failure.
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "backend/tune/tune.hpp"

namespace {

int failures = 0;

void fail(const std::string& what) {
    if (++failures <= 20) std::printf("FAIL %s\n", what.c_str());
}

using Doubles = std::array<double, 3>;
using Words = std::array<std::uint32_t, 4>;

inline constexpr Doubles kDoubles = {0.1, 2.5e-7, 123456789.125};
inline constexpr Words kWords = {0, 1, 7, 4294967295u};
inline constinit fps::tune::Table<kDoubles> doubles;
inline constinit fps::tune::Table<kWords> words;

int measured = 0;  // calls of the "test.words" measure()

template <auto& table>
fps::tune::Choice choice(std::string_view name, void (*measure)()) {
    using Values = typename std::remove_cvref_t<decltype(table)>::Values;
    return {name, measure, [] { return fps::tune::text(table.get()); },
            [](std::string_view s, bool apply) {
                const auto values = fps::tune::parse<Values>(s);
                if (values && apply) table.set(*values);
                return values.has_value();
            }};
}

const bool registered = fps::tune::add(choice<doubles>("test.doubles", [] { doubles.set({1, 2, 3}); })) &&
                        fps::tune::add(choice<words>("test.words", [] {
                            ++measured;
                            words.set({9, 8, 7, 6});
                        }));

void tables() {
    if (doubles.get() != kDoubles || words.get() != kWords) fail("a table before set() is not its defaults");
    const Words& before = words.get();
    words.set({1, 2, 3, 4});
    if (words.get() != Words{1, 2, 3, 4}) fail("set() did not publish");
    if (before != kWords) fail("set() changed what get() returned before");
    words.set(kWords);
}

void texts() {
    const Doubles odd = {std::numeric_limits<double>::min(), 1.0 / 3, -0.0};
    if (fps::tune::parse<Doubles>(fps::tune::text(odd)) != odd) fail("doubles do not read back exactly");
    if (fps::tune::text(kWords) != "0 1 7 4294967295") fail("text of words: " + fps::tune::text(kWords));
    if (fps::tune::parse<Words>("  0 1  7 4294967295 ") != kWords) fail("parse with extra spaces");
    for (const char* bad : {"", "0 1 7", "0 1 7 4294967295 5", "0 1 7 4294967296", "0 1 7 x", "0 1 7 4294967295x", "-1 1 7 2"})
        if (fps::tune::parse<Words>(bad)) fail(std::string("parse accepted \"") + bad + "\"");
}

void readers() {
    std::atomic<bool> done = false;
    std::atomic<int> torn = 0;
    std::vector<std::thread> threads;
    words.set({0, 1, 2, 3});
    for (int t = 0; t < 3; ++t)
        threads.emplace_back([&] {
            while (!done.load()) {
                const Words& w = words.get();
                if (w[1] != w[0] + 1 || w[2] != w[0] + 2) ++torn;
            }
        });
    for (std::uint32_t k = 0; k < 2000; ++k) words.set({k, k + 1, k + 2, k + 3});
    done = true;
    for (auto& t : threads) t.join();
    if (torn) fail("a reader saw values from two set() calls");
    words.set(kWords);
}

std::string file_text(const std::string& path) {
    std::ifstream in(path);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

void write(const std::string& path, const std::string& text) { std::ofstream(path) << text; }

void wisdom() {
    const std::string path = (std::filesystem::temp_directory_path() / ("fps-tune-test-" + std::to_string(::getpid()))).string();
    doubles.set({4, 5, 6});
    words.set({10, 20, 30, 40});
    if (!fps::tune::save(path)) return fail("save failed");
    const std::string saved = file_text(path);
    const std::string head = "fps wisdom 1\ncpu " + fps::tune::cpu_name() + "\ntarget " + std::string(fps::tune::target_name()) + "\n";
    if (saved != head + "test.doubles 4 5 6\ntest.words 10 20 30 40\n") fail("saved wisdom:\n" + saved);
    doubles.set(kDoubles);
    words.set(kWords);
    if (!fps::tune::load(path) || doubles.get() != Doubles{4, 5, 6} || words.get() != Words{10, 20, 30, 40})
        fail("wisdom did not load back");

    const auto rejected = [&](const std::string& text, const std::string& why) {
        write(path, text);
        doubles.set(kDoubles);
        words.set(kWords);
        if (fps::tune::load(path)) fail("loaded wisdom " + why);
        if (doubles.get() != kDoubles || words.get() != kWords) fail("wisdom " + why + " changed a table");
    };
    rejected("fps wisdom 1\ncpu another CPU\ntarget " + std::string(fps::tune::target_name()) +
             "\ntest.words 1 2 3 4\n", "from another CPU");
    rejected("fps wisdom 1\ncpu " + fps::tune::cpu_name() + "\ntarget other\ntest.words 1 2 3 4\n", "for another target");
    rejected("fps wisdom 2\ncpu " + fps::tune::cpu_name() + "\ntarget " + std::string(fps::tune::target_name()) +
             "\ntest.words 1 2 3 4\n", "of another format");
    rejected(head + "test.words 1 2 3 4\ntest.doubles 1 2\n", "with a malformed line");
    rejected(head + "test.words 1 2 3 4\nnospace\n", "with a line without values");
    std::filesystem::remove(path);
    if (fps::tune::load(path)) fail("loaded a missing file");

    write(path, head + "other.kernel 1 2\ntest.words 1 2 3 4\n");
    if (!fps::tune::load(path) || words.get() != Words{1, 2, 3, 4} || doubles.get() != kDoubles)
        fail("wisdom naming a choice this program lacks: not loaded, or other tables changed");
    std::filesystem::remove(path);
    words.set(kWords);
}

void run_once() {
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back([] { fps::tune::run(); });
    for (auto& t : threads) t.join();
    fps::tune::run();
    if (measured != 1) fail("run() measured test.words " + std::to_string(measured) + " times");
    if (doubles.get() != Doubles{1, 2, 3} || words.get() != Words{9, 8, 7, 6}) fail("run() did not set every table");
}

void timing() {
    volatile std::uint32_t sink = 0;
    const auto [short_ns, long_ns] = fps::tune::time(
        3, 1e5, [&] { for (int i = 0; i < 100; ++i) sink = sink + 1; },
        [&] { for (int i = 0; i < 10000; ++i) sink = sink + 1; });
    if (!(short_ns > 0 && long_ns > 10 * short_ns)) fail("time(): 100 and 10000 steps took " + std::to_string(short_ns) + " and " + std::to_string(long_ns) + " ns");
    const std::vector<double> each = fps::tune::time_each(3, 1e5, {2, 0, 1}, [&](int id) {
        for (int i = 0; i < (id == 0 ? 100 : id == 1 ? 10000 : 1000000); ++i) sink = sink + 1;
    });
    if (each.size() != 3 || !(each[1] > 0 && each[2] > 10 * each[1] && each[0] > 10 * each[2]))
        fail("time_each(): ids 2, 0, 1 (10^6, 100, 10^4 steps) out of order");
}

void names() {
    const std::string cpu = fps::tune::cpu_name();
    if (cpu.empty() || cpu.find('\n') != std::string::npos || cpu.front() == ' ' || cpu.back() == ' ')
        fail("cpu_name: \"" + cpu + "\"");
    std::printf("cpu %s, target %s\n", cpu.c_str(), std::string(fps::tune::target_name()).c_str());
}

}  // namespace

int main() {
    if (!registered) fail("add()");
    tables();
    texts();
    readers();
    wisdom();
    run_once();
    timing();
    names();
    std::printf("%s\n", failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
