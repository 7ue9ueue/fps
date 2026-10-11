// Run-time tuning of backend choices: variants, cutoffs, memory layouts (issue #54). Header-only.
//
// A choice is a table of values a kernel reads once per call, e.g. small's NTT times per transform
// length (backend/small/select.hpp). Its compiled default per target comes from tools/tune.py
// (backend/<kernel>/tuned.hpp, measured on the target's machine); tuning replaces it with
// measurements on this machine.
//
//   inline constexpr std::array<double, 20> kDefaults = ...;
//   inline constinit tune::Table<kDefaults> table;
//   table.get()                    // the active values: kDefaults until set(); one acquire load
//   table.set(values)              // publish new values. Thread-safe; what get() returned stays valid
//
//   tune::add({name, measure, text, parse})  // register a choice, from an inline variable
//   tune::run()                    // every registered choice measures and sets itself, once per process
//   tune::save(path)               // wisdom: this CPU and target, then each choice's active values
//   tune::load(path)               // false, setting nothing, if missing, malformed, or made on another
//                                  // CPU or target; choices it does not name keep their values
//
//   tune::kTarget, target_name()   // this build's target (BACKEND.md), from the compiler's macros
//   tune::cpu_name()               // x86: the brand string; Arm Linux: implementer, variant, part, revision
//   tune::time(samples, min_ns, f...)   // median ns per call of each f, the f timed in turn
//   tune::time_each(samples, min_ns, ids, f)   // the same for f(id), ids chosen at run time
//   tune::text(values), tune::parse<Values>(text)
//
// Tuning runs only when called; until then every lookup reads the compiled defaults, so benches and
// tests are deterministic (notes.md: why not lazily). -DFPS_NO_TUNER compiles the defaults in as
// constants and leaves out set(), the registry, run() and wisdom.
#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

#ifndef FPS_NO_TUNER
#include <atomic>
#include <fstream>
#include <mutex>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#endif

namespace fps::tune {

// The targets of BACKEND.md, in the order of TARGETS in tools/kernel.py (tune.py --check).
enum class Target { scalar, sse, avx2, avx512, zen5, arm };

#if defined(__znver5__)
inline constexpr Target kTarget = Target::zen5;
#elif defined(__AVX512F__) && defined(__AVX512BW__)
inline constexpr Target kTarget = Target::avx512;
#elif defined(__AVX2__)
inline constexpr Target kTarget = Target::avx2;
#elif defined(__ARM_NEON) && defined(__aarch64__)
inline constexpr Target kTarget = Target::arm;
#elif defined(__SSE4_1__)
inline constexpr Target kTarget = Target::sse;
#else
inline constexpr Target kTarget = Target::scalar;
#endif

constexpr std::string_view target_name(Target target = kTarget) {
    constexpr std::string_view names[] = {"scalar", "sse", "avx2", "avx512", "zen5", "arm"};
    return names[int(target)];
}

namespace detail {

template <class F>
double elapsed_ns(F& f, long calls) {
    const auto start = std::chrono::steady_clock::now();
    for (long i = 0; i < calls; ++i) f();
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
}

// Calls per sample: doubled from 1 until a sample takes min_ns. The first call is untimed.
template <class F>
long calls_for(F& f, double min_ns) {
    f();
    long calls = 1;
    while (elapsed_ns(f, calls) < min_ns) calls *= 2;
    return calls;
}

}  // namespace detail

// Median ns per call of each f over `samples` rounds; a round times each f in turn, for at least
// min_ns each, so drifts in clock speed hit every f alike. A sample of several calls starts after
// one untimed call, which takes the cost of switching code (notes.md: up to 16% of 50 us samples).
template <class... F>
std::array<double, sizeof...(F)> time(int samples, double min_ns, F... f) {
    constexpr std::size_t kCount = sizeof...(F);
    const std::array<long, kCount> calls = {detail::calls_for(f, min_ns)...};
    std::array<std::vector<double>, kCount> ns;
    for (int s = 0; s < samples; ++s) {
        std::size_t i = 0;
        ((calls[i] > 1 ? void(f()) : void(), ns[i].push_back(detail::elapsed_ns(f, calls[i]) / double(calls[i])), ++i), ...);
    }
    std::array<double, kCount> median;
    for (std::size_t i = 0; i < kCount; ++i) {
        std::ranges::sort(ns[i]);
        median[i] = ns[i][ns[i].size() / 2];
    }
    return median;
}

// As time(), for candidates chosen at run time: median ns per call of f(id) for each id. Calls per
// sample from one timed call after an untimed one (no doubling: a choice with many candidates and
// sizes spends its budget on samples); when that is one call, the timed call is the first sample.
// E.g. factorials' variants per size class.
template <class F>
std::vector<double> time_each(int samples, double min_ns, const std::vector<int>& ids, F f) {
    const std::size_t count = ids.size();
    std::vector<long> calls(count);
    std::vector<std::vector<double>> ns(count);
    for (std::size_t i = 0; i < count; ++i) {
        auto g = [&] { f(ids[i]); };
        g();
        const double once = detail::elapsed_ns(g, 1);
        calls[i] = std::max(1L, long(min_ns / std::max(once, 1.0)));
        if (calls[i] == 1) ns[i].push_back(once);
    }
    for (int s = 0; s < samples; ++s)
        for (std::size_t i = 0; i < count; ++i) {
            if (int(ns[i].size()) >= samples) continue;
            auto g = [&] { f(ids[i]); };
            if (calls[i] > 1) g();
            ns[i].push_back(detail::elapsed_ns(g, calls[i]) / double(calls[i]));
        }
    std::vector<double> median(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::ranges::sort(ns[i]);
        median[i] = ns[i][ns[i].size() / 2];
    }
    return median;
}

// The values separated by single spaces; doubles in the shortest form that reads back exactly.
template <class Values>
std::string text(const Values& values) {
    std::string out;
    char buffer[32];
    for (const auto x : values) {
        if (!out.empty()) out += ' ';
        out.append(buffer, std::to_chars(buffer, buffer + sizeof buffer, x).ptr);
    }
    return out;
}

// Exactly as many numbers as Values holds, separated by spaces; nothing for anything else.
template <class Values>
std::optional<Values> parse(std::string_view s) {
    Values values{};
    const char* p = s.data();
    const char* const end = p + s.size();
    for (auto& x : values) {
        while (p < end && *p == ' ') ++p;
        const auto [next, error] = std::from_chars(p, end, x);
        if (error != std::errc{}) return std::nullopt;
        p = next;
    }
    while (p < end && *p == ' ') ++p;
    if (p != end) return std::nullopt;
    return values;
}

#ifdef FPS_NO_TUNER

template <const auto& kDefaults>
class Table {
public:
    using Values = std::remove_cvref_t<decltype(kDefaults)>;
    static constexpr const Values& get() noexcept { return kDefaults; }
};

#else

namespace detail {

inline std::mutex& publish_mutex() {
    static std::mutex mutex;
    return mutex;
}

}  // namespace detail

// A choice's values: kDefaults (a std::array with static storage) until set(). Constant-initialized,
// so get() is valid during static initialization too.
template <const auto& kDefaults>
class Table {
public:
    using Values = std::remove_cvref_t<decltype(kDefaults)>;

    const Values& get() const noexcept { return *active_.load(std::memory_order_acquire); }

    // Publishes a copy of values. Earlier values stay allocated (they are few and small), so a
    // reader holding them while another thread sets new ones stays valid.
    void set(const Values& values) {
        const std::lock_guard lock(detail::publish_mutex());
        published_ = new Node{values, published_};
        active_.store(&published_->values, std::memory_order_release);
    }

private:
    struct Node {
        Values values;
        Node* next;
    };
    std::atomic<const Values*> active_{&kDefaults};
    Node* published_ = nullptr;  // every set(), newest first
};

// A registered choice: what run(), save() and load() call.
struct Choice {
    std::string_view name;                         // one word, e.g. "small.ntt"
    void (*measure)();                             // measure on this machine and set the table
    std::string (*text)();                         // the active values
    bool (*parse)(std::string_view, bool apply);   // false if malformed; sets them if apply
};

namespace detail {

inline std::mutex& registry_mutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::vector<Choice>& registry() {
    static std::vector<Choice> choices;
    return choices;
}

inline std::vector<Choice> registered() {
    const std::lock_guard lock(registry_mutex());
    return registry();
}

inline std::string trimmed(std::string_view s) {
    std::string out;
    for (const char c : s) {
        const bool space = c == ' ' || c == '\t' || c == '\n' || c == '\r';
        if (!space) out += c;
        else if (!out.empty() && out.back() != ' ') out += ' ';
    }
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

}  // namespace detail

// Returns true, so that a kernel can register from an inline variable's initializer.
inline bool add(const Choice& choice) {
    const std::lock_guard lock(detail::registry_mutex());
    detail::registry().push_back(choice);
    return true;
}

inline void run() {
    static std::once_flag once;
    std::call_once(once, [] {
        for (const Choice& choice : detail::registered()) choice.measure();
    });
}

inline std::string cpu_name() {
    std::string name;
#if defined(__x86_64__) || defined(__i386__)
    unsigned words[12] = {};
    if (__get_cpuid_max(0x80000000, nullptr) >= 0x80000004)
        for (unsigned i = 0; i < 3; ++i)
            __get_cpuid(0x80000002 + i, &words[4 * i], &words[4 * i + 1], &words[4 * i + 2], &words[4 * i + 3]);
    const char* brand = reinterpret_cast<const char*>(words);
    name.assign(brand, std::find(brand, brand + sizeof words, '\0'));
#elif defined(__APPLE__)
    char brand[256] = {};
    std::size_t size = sizeof brand - 1;
    if (sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0) == 0) name = brand;
#elif defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    for (std::string line; std::getline(cpuinfo, line) && !(line.empty() && !name.empty());) {
        for (const std::string_view key : {"CPU implementer", "CPU variant", "CPU part", "CPU revision"})
            if (line.starts_with(key) && line.find(':') != std::string::npos)
                name += std::string(key.substr(4)) + ' ' + line.substr(line.find(':') + 1) + ' ';
    }
#endif
    name = detail::trimmed(name);
    return name.empty() ? "unknown" : name;
}

inline constexpr std::string_view kWisdomFormat = "fps wisdom 1";

inline bool save(const std::string& path) {
    std::ofstream out(path);
    out << kWisdomFormat << "\ncpu " << cpu_name() << "\ntarget " << target_name() << '\n';
    for (const Choice& choice : detail::registered()) out << choice.name << ' ' << choice.text() << '\n';
    out.close();
    return !out.fail();
}

inline bool load(const std::string& path) {
    std::ifstream in(path);
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) lines.push_back(line);
    if (lines.size() < 3 || lines[0] != kWisdomFormat || lines[1] != "cpu " + cpu_name() ||
        lines[2] != "target " + std::string(target_name()))
        return false;
    const std::vector<Choice> choices = detail::registered();
    for (const bool apply : {false, true})  // all lines checked before any is set
        for (std::size_t i = 3; i < lines.size(); ++i) {
            const std::string_view line = lines[i];
            const std::size_t space = line.find(' ');
            if (space == std::string_view::npos) return false;
            const auto choice = std::ranges::find(choices, line.substr(0, space), &Choice::name);
            if (choice != choices.end() && !choice->parse(line.substr(space + 1), apply)) return false;
        }
    return true;
}

#endif  // FPS_NO_TUNER

}  // namespace fps::tune
