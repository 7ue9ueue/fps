// Tests for two ops backend/gcd uses (issue #157 round 2), for every variant this machine runs and
// the tuned one, both forms, lengths 2^kMinLog .. 2^16 (forward_upper from 2^(kMinLog + 1)):
// - forward_upper(std::span in, H, form) equals forward_upper(h, form) of h = in zero-padded, for
//   in of 0, 1, a quarter, half - 1 and half words; in is not written.
// - inverse(F, work, out, form) equals inverse(F, out, form) for out of 0, 1, 7, half, len - 1 and
//   len words; F kept when it is not work, and F = work too; past the length it throws.
// Random words, all P - 1 and zeros.
#include "backend/ntt/test_util.hpp"

namespace {

constexpr int kTestMaxLog = 16;
int checks = 0;

void check(bool ok, const std::string& what, u64 detail) {
    ++checks;
    expect(ok, what.c_str(), detail);
}

template <class V>
void test_variant(const char* name) {
    const std::string tag = name;
    for (int lg = kMinLog; lg <= kTestMaxLog; ++lg) {
        const Transform t(lg);
        const std::size_t len = t.length(), half = len / 2;
        for (const Form form : {Form::leaves, Form::values}) {
            const std::string f = tag + (form == Form::values ? " values" : " leaves");
            for (int kind = 0; kind < 3; ++kind) {
                if (kind && lg > 10) break;
                if (lg > kMinLog)
                    for (const std::size_t n : {std::size_t(0), std::size_t(1), half / 2, half - 1, half}) {
                        const std::vector<u32> in = random_poly(n, kind), before = in;
                        Words h(half), H(half);
                        std::fill_n(h.begin(), half, 0u);
                        std::copy(in.begin(), in.end(), h.begin());
                        t.forward_upper<V>(h.begin(), form);
                        t.forward_upper<V>(std::span<const u32>(in), H.begin(), form);
                        check(std::equal(h.begin(), h.begin() + half, H.begin()), f + " forward_upper from a span",
                               u64(lg) * 1000 + n);
                        check(in == before, f + " forward_upper wrote its source", u64(lg));
                    }
                // F: a transform in this form of random words.
                const std::vector<u32> p = random_poly(len, kind);
                Words F(len);
                std::copy(p.begin(), p.end(), F.begin());
                t.forward<V>(F.begin(), form);
                const std::vector<u32> kept(F.begin(), F.begin() + len);
                for (const std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(7), half, len - 1, len}) {
                    std::vector<u32> want(n + 1, 12345u), got(n + 1, 12345u);
                    t.inverse<V>(F.begin(), std::span<u32>(want.data(), n), form);
                    Words work(len);
                    t.inverse<V>(F.begin(), work.begin(), std::span<u32>(got.data(), n), form);
                    check(got == want, f + " inverse through work", u64(lg) * 1000 + n);
                    check(std::equal(kept.begin(), kept.end(), F.begin()), f + " inverse through work kept F", u64(lg));
                    // In place: F is the work array.
                    Words G(len);
                    std::copy(kept.begin(), kept.end(), G.begin());
                    std::vector<u32> in_place(n + 1, 12345u);
                    t.inverse<V>(G.begin(), G.begin(), std::span<u32>(in_place.data(), n), form);
                    check(in_place == want, f + " inverse through F itself", u64(lg) * 1000 + n);
                }
            }
        }
        bool threw = false;
        try {
            Words F(len), work(len);
            std::vector<u32> out(len + 1);
            t.inverse<V>(F.begin(), work.begin(), std::span<u32>(out), Form::leaves);
        } catch (const std::out_of_range&) {
            threw = true;
        }
        check(threw, tag + " inverse through work past the length throws", u64(lg));
    }
}

}  // namespace

int main() {
    for_each_variant([]<class V>(const char* name, int) { test_variant<V>(name); });
#if defined(__ARM_NEON) && defined(__aarch64__)
    test_variant<variant::neon>("neon");
#endif
    test_variant<variant::tuned>("tuned");
    std::printf("%d checks\n", checks);
    return finish();
}
