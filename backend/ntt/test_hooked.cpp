// Tests for Transform's hooked ops (issue #149): forward_hooked and inverse_hooked, with and
// without a pruned column bit, on every variant this machine runs and the tuned one. References:
// the scalar variant's forward of a zero-padded copy and inverse. The hook sees each leaf exactly
// once, in runs of a power of two of at least 4 leaves starting at a multiple of the run; the
// forward's hook sees the finished leaves and its changes stay; the inverse's leaves carry 2^-32.
// Sources with every column bit zero, both output halves, negated outputs, the argument checks.
#include "backend/ntt/test_util.hpp"

namespace {

#if __has_feature(address_sanitizer)
constexpr int kMaxTestLog = 17;
#else
constexpr int kMaxTestLog = 20;
#endif
constexpr int kScalarTestLog = 13;
constexpr u32 kR = u32((u64(1) << 32) % P);  // the 2^32 the inverse's leaves are divided by

std::vector<u32> scalar_leaves(const Transform& t, const std::vector<u32>& in) {
    Words w(t.length());
    std::fill(std::copy(in.begin(), in.end(), w.begin()), w.begin() + t.length(), 0u);
    t.forward<variant::scalar>(w.begin());
    return {w.begin(), w.begin() + t.length()};
}

std::vector<u32> scalar_coefficients(const Transform& t, const std::vector<u32>& F) {
    Words w(t.length());
    std::copy(F.begin(), F.end(), w.begin());
    t.inverse<variant::scalar>(w.begin());
    return {w.begin(), w.begin() + t.length()};
}

// The runs a hook saw: each leaf once, runs of a power of two >= 4 at a multiple of the run.
struct Runs {
    explicit Runs(std::size_t leaves) : seen(leaves, 0) {}
    void add(std::size_t first, std::size_t count) {
        ok &= count >= 4 && (count & (count - 1)) == 0 && first % count == 0 && first + count <= seen.size();
        for (std::size_t v = first; v < first + count && v < seen.size(); ++v) ++seen[v];
    }
    bool good() const { return ok && std::all_of(seen.begin(), seen.end(), [](int c) { return c == 1; }); }
    std::vector<int> seen;
    bool ok = true;
};

// Column bits to prune at length len: none, and every power of two from 1 to len / 4.
std::vector<std::size_t> column_bits(std::size_t len) {
    std::vector<std::size_t> bits = {0};
    for (std::size_t b = 1; b <= len / 4; b *= 2) bits.push_back(b);
    return bits;
}

template <class V>
void test_forward(const std::string& name, const Transform& t, int lg) {
    const std::size_t len = t.length();
    for (std::size_t zero : column_bits(len)) {
        for (std::size_t n : {len, len / 2 + 3, len / 2, std::size_t(1 + rng() % len)}) {
            std::vector<u32> in = random_poly(n, n % 3 == 0 ? 1 : 0);
            for (std::size_t i = 0; i < n; ++i)
                if (i & zero) in[i] = 0;
            const std::vector<u32> expected = scalar_leaves(t, in);
            Words F(len);
            Runs runs(len / 8);
            bool seen_right = true;
            t.forward_hooked<V>(std::span<const u32>(in), F.begin(), [&](u32* leaves, std::size_t first, std::size_t count) {
                runs.add(first, count);
                seen_right &= leaves == F.begin() + 8 * first;
                for (std::size_t i = 0; i < 8 * count; ++i) {
                    seen_right &= leaves[i] == expected[8 * first + i];
                    leaves[i] = leaves[i] + 1 == P ? 0 : leaves[i] + 1;
                }
            }, zero);
            bool kept = true;
            for (std::size_t i = 0; i < len; ++i) kept &= F.begin()[i] == (expected[i] + 1 == P ? 0 : expected[i] + 1);
            const std::string what = name + " forward_hooked zero " + std::to_string(zero) + " n " + std::to_string(n);
            expect(runs.good(), (what + ": runs").c_str(), lg);
            expect(seen_right, (what + ": leaves seen").c_str(), lg);
            expect(kept, (what + ": leaves kept").c_str(), lg);
            if (lg > 14) break;  // one source size at the long lengths
        }
    }
}

template <class V>
void test_inverse(const std::string& name, const Transform& t, int lg) {
    const std::size_t len = t.length();
    std::vector<u32> A(len);
    for (auto& x : A) x = random_below(P);
    std::vector<u32> c = scalar_coefficients(t, A);
    for (auto& x : c) x = mul(x, kR);
    for (std::size_t unused : column_bits(len)) {
        for (std::size_t offset : {std::size_t(0), len / 2}) {
            for (bool negate : {false, true}) {
                for (std::size_t count : {len - offset, len / 2, std::size_t(5)}) {
                    if (count > len - offset) continue;
                    Words work(len);
                    std::vector<u32> out(count + 2, 7);
                    Runs runs(len / 8);
                    t.inverse_hooked<V>(
                        [&](u32* leaves, std::size_t first, std::size_t n) {
                            runs.add(first, n);
                            std::copy_n(A.begin() + std::ptrdiff_t(8 * first), 8 * n, leaves);
                        },
                        work.begin(), std::span<u32>(out.data() + 1, count), {offset, negate}, unused);
                    bool ok = out[0] == 7 && out[count + 1] == 7;
                    for (std::size_t i = 0; i < count; ++i) {
                        if ((offset + i) & unused) continue;
                        const u32 x = c[offset + i];
                        ok &= out[i + 1] == (negate ? sub(0, x) : x);
                    }
                    const std::string what = name + " inverse_hooked unused " + std::to_string(unused) + " offset " +
                                             std::to_string(offset) + (negate ? " negated" : "") + " count " + std::to_string(count);
                    expect(runs.good(), (what + ": runs").c_str(), lg);
                    expect(ok, what.c_str(), lg);
                    if (lg > 14) break;
                }
                if (lg > 14 && negate) break;
            }
        }
    }
}

template <class V>
void test_variant(const std::string& name, int max_lg) {
    for (int lg = kMinLog; lg <= max_lg; ++lg) {
        const Transform t(lg);
        test_forward<V>(name, t, lg);
        test_inverse<V>(name, t, lg);
    }
}

void test_checks() {
    const Transform t(8);
    Words F(256), G(256);
    std::vector<u32> in(10), out(256);
    const auto none = [](u32*, std::size_t, std::size_t) {};
    bool threw = false;
    try {
        t.forward_hooked(std::span<const u32>(in), F.begin(), none, 3);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "forward_hooked: column bit 3");
    threw = false;
    try {
        t.inverse_hooked(none, G.begin(), std::span<u32>(out), {}, 12);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "inverse_hooked: column bit 12");
    threw = false;
    try {
        t.inverse_hooked(none, G.begin(), std::span<u32>(out), {.offset = 3});
    } catch (const std::out_of_range&) {
        threw = true;
    }
    expect(threw, "inverse_hooked: offset 3");
    std::vector<u32> long_in(257);
    threw = false;
    try {
        t.forward_hooked(std::span<const u32>(long_in), F.begin(), none);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    expect(threw, "forward_hooked: source past the length");
}

}  // namespace

int main() {
    test_checks();
    test_variant<variant::scalar>("scalar", kScalarTestLog);
    fps::simd::for_each(fps::simd::All{}, []<class S> {
        test_variant<variant::intrinsics<S>>(std::string("intrinsics ") + S::name, S::lanes == 1 ? kScalarTestLog : kMaxTestLog);
    });
#ifdef __AVX2__
    test_variant<variant::avx2>("avx2", kMaxTestLog);
#endif
    test_variant<variant::tuned>("tuned", kMaxTestLog);
    return finish();
}
