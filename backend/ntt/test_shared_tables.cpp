// Tests for Transform(lg, tables) (issue #91 round 2): a transform on the tables of a longer one
// gives the words of a transform with tables of its own, at every length up to the longer one's,
// in every op that reads the tables (forward, inverse, forward_upper, inverse_upper, the values
// form, leaf products, the inverse into other memory, a cyclic product); after the owner is moved,
// and when moved onto another transform's tables in place; and the errors (tables shorter than
// the length; the values form on tables past 2^kValuesMaxLog).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backend/ntt/ntt.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using namespace fps::ntt;
constexpr u32 P = kModulus;

#if __has_feature(address_sanitizer)
constexpr int kMaxTestLog = 18;
#else
constexpr int kMaxTestLog = 20;
#endif

std::mt19937_64 rng(912);
int failures = 0, checks = 0;

void expect(bool ok, const std::string& what, u64 detail = 0) {
    ++checks;
    if (ok) return;
    if (++failures <= 20) std::printf("FAIL: %s (%llu)\n", what.c_str(), static_cast<unsigned long long>(detail));
}

// count words, 32-byte aligned, then kPadding words.
struct Words {
    struct alignas(32) Lanes {
        u32 lane[8];
    };
    explicit Words(const std::vector<u32>& w) : data((w.size() + kPadding + 7) / 8) { std::copy(w.begin(), w.end(), begin()); }
    u32* begin() { return data.data()->lane; }
    std::vector<Lanes> data;
};

std::vector<u32> random_words(std::size_t n) {
    std::vector<u32> p(n);
    for (u32& x : p) x = u32(rng() % P);
    return p;
}

// Every op that reads the tables, on `shared` and on `own` (same length), the same words.
void compare(const Transform& shared, const Transform& own, const std::string& where) {
    const std::size_t len = own.length(), half = len / 2;
    const int lg = std::countr_zero(len);
    const std::vector<u32> f = random_words(len), g = random_words(len);
    const auto both = [&](const char* op, auto run) {
        Words x(f), y(f);
        run(shared, x.begin()), run(own, y.begin());
        expect(std::equal(x.begin(), x.begin() + len, y.begin()), where + " " + op);
    };
    both("forward", [](const Transform& t, u32* w) { t.forward(w); });
    both("inverse", [](const Transform& t, u32* w) { t.inverse(w); });
    if (lg > kMinLog) {
        both("forward_upper", [](const Transform& t, u32* w) { t.forward_upper(w); });
        both("inverse_upper", [](const Transform& t, u32* w) { t.inverse_upper(w); });
    }
    if (lg <= kValuesMaxLog) {
        both("forward values", [](const Transform& t, u32* w) { t.forward(w, Form::values); });
        both("inverse values", [](const Transform& t, u32* w) { t.inverse(w, Form::values); });
        if (lg > kMinLog) both("forward_upper values", [](const Transform& t, u32* w) { t.forward_upper(w, Form::values); });
    }
    Words b(g);
    both("multiply", [&](const Transform& t, u32* w) { t.multiply(w, b.begin(), w); });
    both("multiply_add", [&](const Transform& t, u32* w) { t.multiply_add(w, w, b.begin()); });
    {
        Words a(f);
        std::vector<u32> x(half + 3), y(half + 3);
        shared.inverse(a.begin(), std::span(x.data(), half + 3), Form::leaves, {.offset = half - 3});
        own.inverse(a.begin(), std::span(y.data(), half + 3), Form::leaves, {.offset = half - 3});
        expect(x == y, where + " inverse into other memory");
    }
    {
        Words work_x{std::vector<u32>(len)}, work_y{std::vector<u32>(len)};
        std::vector<u32> x(half), y(half);
        shared.cyclic_product(std::span(f.data(), half), b.begin(), work_x.begin(), x);
        own.cyclic_product(std::span(f.data(), half), b.begin(), work_y.begin(), y);
        expect(x == y, where + " cyclic_product");
    }
}

}  // namespace

int main() {
    for (const int top : {kMinLog + 2, 12, 16, 17, kMaxTestLog}) {
        Transform tables(top);
        for (int lg = kMinLog; lg <= top; ++lg) {
            const Transform own(lg), shared(lg, tables);
            compare(shared, own, "tables 2^" + std::to_string(top) + ", length 2^" + std::to_string(lg));
        }
        // The owner moved: the tables stay where they were.
        Transform view(kMinLog + 1, tables);
        Transform moved(std::move(tables));
        compare(view, Transform(kMinLog + 1), "after a move of the tables, 2^" + std::to_string(top));
        compare(Transform(top, moved), Transform(top), "the moved tables, 2^" + std::to_string(top));
    }
    {  // Moved onto other tables in place, as backend/product_tree's reserve_transforms does.
        auto small = std::make_unique<Transform>(10);
        Transform t(8, *small);
        const Transform& ref = t;
        auto big = std::make_unique<Transform>(14);
        t = Transform(8, *big);
        small.reset();
        compare(ref, Transform(8), "moved onto other tables");
        Transform own(9);
        own = Transform(9, *big);
        compare(own, Transform(9), "an owner moved onto other tables");
    }
    {  // Errors.
        const Transform tables(12);
        bool threw = false;
        try {
            const Transform t(13, tables);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        expect(threw, "tables shorter than the length: std::invalid_argument");
        const Transform past(kValuesMaxLog + 1);
        const Transform view(10, past);
        Words w{std::vector<u32>(1024)};  // 32-byte aligned, as Transform asks
        threw = false;
        try {
            view.forward(w.begin(), Form::values);
        } catch (const std::length_error&) {
            threw = true;
        }
        expect(threw, "values form on tables past 2^kValuesMaxLog: std::length_error");
        view.forward(w.begin());  // the leaf form works
    }
    std::printf("%s ntt shared tables: %d checks, %d failures\n", failures ? "FAIL" : "pass", checks, failures);
    return failures != 0;
}
