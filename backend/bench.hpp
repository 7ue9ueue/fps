// Timing harness for backend/<kernel>/bench.cpp. tools/kernel.py builds and runs it.
//
//   int main(int argc, char** argv) {
//       bench::Suite suite(argc, argv);
//       for (int lg = 10; lg <= 22; ++lg)
//           suite.add("forward/" + std::to_string(lg), [lg] {   // setup, not timed
//               return [buffers = Buffers(lg)]() mutable {        // one timed call
//                   forward(buffers);
//                   bench::keep(buffers);
//               };
//           }, std::size_t(1) << lg);                              // items per call, optional
//       return suite.run();
//   }
//
// Names: "<op>/<size>" times what the library uses on this target; the gate compares these.
// "<op>@<variant>/<size>" times one variant; it is reported, never gated.
// Output: one line per case, "<name>\t<ns per call>\t<items per call>"; the time is the median of
// 5 samples of >= 2 ms each. Items (coefficients, vectors) give a per-element unit; 0 if not set.
// Arguments: --list prints the names; another argument is a regex the names must contain.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <regex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bench {

// Makes the compiler assume `value` is read, so the work that produced it is kept.
template <class T>
inline void keep(const T& value) {
    const void* p = &value;
    asm volatile("" : : "r"(p) : "memory");
}

class Suite {
public:
    Suite(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            if (std::string_view(argv[i]) == "--list") list_ = true;
            else filter_ = argv[i];
        }
    }

    // setup() runs once, untimed, and returns the callable to time.
    template <class Setup>
    void add(std::string name, Setup setup, std::size_t items = 0) {
        cases_.push_back({std::move(name), items, [setup] { return measure(setup()); }});
    }

    int run() const {
        const std::regex filter(filter_);
        for (const auto& [name, items, time] : cases_) {
            if (!std::regex_search(name, filter)) continue;
            if (list_) std::printf("%s\n", name.c_str());
            else std::printf("%s\t%.1f\t%zu\n", name.c_str(), time(), items);
            std::fflush(stdout);
        }
        return 0;
    }

private:
    static constexpr double kMinSample = 2e-3;  // seconds
    static constexpr int kSamples = 5;

    template <class Run>
    static double seconds(Run& run, long reps) {
        const auto start = std::chrono::steady_clock::now();
        for (long i = 0; i < reps; ++i) run();
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    template <class Run>
    static double measure(Run run) {
        run();  // page faults, caches, branch predictors
        long reps = 1;
        while (seconds(run, reps) < kMinSample) reps *= 2;
        std::array<double, kSamples> ns;
        for (double& t : ns) t = seconds(run, reps) / double(reps) * 1e9;
        std::ranges::sort(ns);
        return ns[kSamples / 2];
    }

    struct Case {
        std::string name;
        std::size_t items;
        std::function<double()> time;  // ns per call
    };
    std::vector<Case> cases_;
    std::string filter_;
    bool list_ = false;
};

}  // namespace bench
