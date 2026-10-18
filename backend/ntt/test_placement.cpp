// Tests for issue #236: the fused ops' top level picked by where out lies from the work array
// (avx2's split_top: the radix-8 top, or the radix-2 level after each half's radix-4 group; the
// intrinsics' radix-2 columns in reverse just past the input), and detail::place(). Each fused op
// (inverse_product, cyclic_product, toeplitz_product, inverse_hooked) at 2^14, 2^16 and 2^18 with
// out at several byte offsets from work modulo 4 KiB, near it and far from it, on avx2 and the
// tuned variant, against the scalar variant's results with the same inputs; words around out
// untouched. place(): a multiple of 64 below its period, at least period / 6 - 64 bytes from each
// point it avoids, for random addresses.
#include "backend/ntt/test_util.hpp"

namespace {

#if __has_feature(address_sanitizer)
constexpr int kMaxTestLog = 16;
#else
constexpr int kMaxTestLog = 18;
#endif
constexpr u32 kSentinel = 0xdeadbeef;
int checks = 0;
// Byte offsets of out past work modulo 4 KiB: less than 640 bytes past it or 448 before (avx2: the
// radix-2 level; 4 and 64: the intrinsics' radix-2 columns in reverse), and farther (avx2: the
// radix-8 top but at 2^18).
constexpr std::size_t kOffsets[] = {0, 4, 64, 576, 640, 2048, 3456, 3584, 3712, 4032};

// count words at byte offset `at` modulo 4 KiB past a 4 KiB boundary, between sentinels.
struct Placed {
    Placed(std::size_t count, std::size_t at)
        : count(count), skip(at / 4 + 1024), storage(count + skip + 1024 + 2048, kSentinel) {
        const auto base = reinterpret_cast<std::uintptr_t>(storage.data());
        const std::size_t to_page = (4096 - base % 4096) % 4096 / 4;
        p = storage.data() + to_page + skip;
    }
    std::span<u32> span() { return {p, count}; }
    bool kept() const {
        for (const u32* q = storage.data(); q < p; ++q)
            if (*q != kSentinel) return false;
        for (const u32* q = p + count; q < storage.data() + storage.size(); ++q)
            if (*q != kSentinel) return false;
        return true;
    }
    std::size_t count, skip;
    std::vector<u32> storage;
    u32* p;
};

// A work array of len + kPadding words starting on a 4 KiB boundary.
struct Work {
    explicit Work(std::size_t len) : storage(len + kPadding + 1024) {
        const auto base = reinterpret_cast<std::uintptr_t>(storage.data());
        p = storage.data() + (4096 - base % 4096) % 4096 / 4;
    }
    std::vector<u32> storage;
    u32* p;
};

std::vector<u32> random_leaves(std::size_t len) {
    std::vector<u32> v(len + kPadding, 0);
    for (std::size_t i = 0; i < len; ++i) v[i] = random_below(P);
    return v;
}

// A leaf form in a 32-byte aligned array (the asm loads leaf forms aligned).
struct Form {
    explicit Form(const std::vector<u32>& v) : w(v.size()) { std::copy(v.begin(), v.end(), w.begin()); }
    u32* p() { return w.begin(); }
    Words w;
};

template <class V>
void test_length(const std::string& name, const Transform& t, int lg) {
    const std::size_t len = t.length(), half = len / 2;
    const std::vector<u32> a = random_leaves(len), b = random_leaves(len), c = random_leaves(len), d = random_leaves(len);
    const std::vector<u32> e = random_leaves(len), source = random_poly(half, 0);
    Form A(a), B(b), C(c), D(d);
    struct Op {
        const char* name;
        std::size_t count;
        Output o;
    };
    const Op outputs[] = {{"all", len, {}}, {"upper, negated", half, {half, true}}, {"lower", half, {}}};
    for (const Op& op : outputs) {
        // The scalar variant's results, out in a vector.
        Work work0(len);
        std::vector<u32> ip(op.count), ip2(op.count), cp(op.count), tp(op.count), hk(op.count);
        const Term one[] = {{A.p(), B.p()}}, two[] = {{A.p(), B.p()}, {C.p(), D.p()}};
        t.inverse_product<variant::scalar>(one, work0.p, std::span<u32>(ip), op.o);
        t.inverse_product<variant::scalar>(two, work0.p, std::span<u32>(ip2), op.o);
        if (op.count <= half) t.cyclic_product<variant::scalar>(std::span<const u32>(source), B.p(), work0.p, std::span<u32>(cp), op.o);
        Form E0(e);
        t.toeplitz_product<variant::scalar>(A.p(), B.p(), E0.p(), C.p(), D.p(), work0.p, std::span<u32>(tp), op.o);
        const auto hook = [&](u32* leaves, std::size_t first, std::size_t n) { std::copy_n(a.data() + 8 * first, 8 * n, leaves); };
        t.inverse_hooked<variant::scalar>(hook, work0.p, std::span<u32>(hk), op.o);
        for (const std::size_t at : kOffsets) {
            const std::string what = name + " lg " + std::to_string(lg) + " out " + op.name + " at " + std::to_string(at);
            Work work(len);
            const auto check = [&](const char* fused, const std::vector<u32>& expected, auto run) {
                Placed out(op.count, at);
                run(out.span());
                ++checks;
                expect(std::equal(expected.begin(), expected.end(), out.p), (what + " " + fused).c_str());
                expect(out.kept(), (what + " " + fused + ": words around out").c_str());
            };
            check("inverse_product of 1", ip, [&](std::span<u32> o) { t.inverse_product<V>(one, work.p, o, op.o); });
            check("inverse_product of 2", ip2, [&](std::span<u32> o) { t.inverse_product<V>(two, work.p, o, op.o); });
            if (op.count <= half)
                check("cyclic_product", cp, [&](std::span<u32> o) {
                    t.cyclic_product<V>(std::span<const u32>(source), B.p(), work.p, o, op.o);
                });
            check("toeplitz_product", tp, [&](std::span<u32> o) {
                Form E(e);
                t.toeplitz_product<V>(A.p(), B.p(), E.p(), C.p(), D.p(), work.p, o, op.o);
            });
            check("inverse_hooked", hk, [&](std::span<u32> o) { t.inverse_hooked<V>(hook, work.p, o, op.o); });
        }
    }
}

template <class V>
void test_variant(const std::string& name) {
    for (int lg = 14; lg <= kMaxTestLog; lg += 2) test_length<V>(name, Transform(lg), lg);
}

void test_place() {
    for (const std::size_t period : {std::size_t(2048), std::size_t(4096)}) {
        for (int i = 0; i < 20000; ++i) {
            const std::uintptr_t f = rng() % (1 << 20) * 4, g = rng() % (1 << 20) * 4, out = rng() % (1 << 20) * 4;
            const std::size_t span = i % 2 ? rng() % (1 << 22) * 4 : 4 * ((std::size_t(1) << (6 + i % 17)) + kPadding);
            const auto* fp = reinterpret_cast<const void*>(f + 4096);
            const auto* gp = reinterpret_cast<const void*>(g + 4096);
            const auto* op = reinterpret_cast<const void*>(out + 4096);
            const std::size_t a = detail::place(fp, gp, op, span, period);
            const auto distance = [period](std::size_t x, std::size_t y) {
                const std::size_t d = (x % period + period - y % period) % period;
                return std::min(d, period - d);
            };
            const std::size_t least = std::min({distance(a, f + 4096), distance(a, out + 4096), distance(a + span, g + 4096)});
            expect(a % 64 == 0 && a < period, "place: a multiple of 64 below the period", a);
            expect(least + 64 >= period / 6, "place: far from f, out and g - span", least);
        }
    }
    for (int lg = kMinLog; lg <= kMaxLog; ++lg) {
        const std::size_t p = detail::place_period(lg);
        expect(p == 2048 || p == 4096, "place_period: 2 or 4 KiB", p);
        expect((p == 2048) == (lg == 11 || lg == 12), "place_period: 2 KiB at 2^11 and 2^12 only", std::size_t(lg));
    }
}

}  // namespace

int main() {
    test_place();
#ifdef __AVX2__
    test_variant<variant::avx2>("avx2");
#endif
    test_variant<variant::tuned>("tuned");
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS: %d checks\n", checks);
    return 0;
}
