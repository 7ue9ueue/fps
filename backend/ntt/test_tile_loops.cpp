// Tests for issue #210: the tiles' lowest levels as loops over their groups. avx2 kernels against
// the per-group kernels they replace, word for word (group 0, whose per-group kernels are the
// identity forms, modulo P): inverse_h4 against inverse at h = 4, inverse_leaves_loop against
// inverse_leaves, forward_h4_pair against forward_pair; inputs random and at the edge of each
// range (forward < 4P, inverse < 2P, leaves canonical), outputs in range, words past the groups
// untouched. Then the ops whose tiles run them, on every variant this machine runs and the tuned
// one, at every length: inverse in place and into other memory, products and squares (fused
// forwards of both factors at h = 4), cyclic_product, inverse_hooked with column bits, against
// the scalar model.
#include "backend/ntt/test_util.hpp"

namespace {

#if __has_feature(address_sanitizer)
constexpr int kMaxTestLog = 17;
#else
constexpr int kMaxTestLog = 21;
#endif
constexpr int kScalarTestLog = 14;
constexpr u32 kR = u32((u64(1) << 32) % P);

#ifdef __AVX2__
constexpr u32 kSentinel = 0xdeadbeef;
namespace av = fps::ntt::detail::avx2;
namespace kn = fps::ntt::avx2_kernels;
using fps::ntt::detail::slot;

// Words of vectors (32-byte aligned) with kSentinel past them.
struct Vectors {
    explicit Vectors(std::size_t vectors) : count(8 * vectors), w(count + kPadding) {
        std::fill_n(w.begin() + count, kPadding, kSentinel);
    }
    av::Vec* v() { return reinterpret_cast<av::Vec*>(w.begin()); }
    u32* begin() { return w.begin(); }
    bool padding_kept() { return std::all_of(begin() + count, begin() + count + kPadding, [](u32 x) { return x == kSentinel; }); }
    std::size_t count;
    Words w;
};

// Random words below bound, with runs of bound - 1 and 0 (the edges).
void fill(u32* f, std::size_t n, u32 bound, int kind) {
    for (std::size_t i = 0; i < n; ++i)
        f[i] = kind == 0 ? random_below(bound) : kind == 1 ? bound - 1 : kind == 2 ? (i % 3 ? bound - 1 : 0) : random_below(bound) % 8;
}

bool below(const u32* f, std::size_t n, u32 bound) { return std::all_of(f, f + n, [bound](u32 x) { return x < bound; }); }
bool equal_mod(const u32* f, const u32* g, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        if (f[i] % P != g[i] % P) return false;
    return true;
}

void test_kernels() {
    constexpr int lg = 20;  // tables of 2^lg: group indices below 2^lg / 16
    Words rt(detail::table_words(lg)), irt(detail::table_words(lg));  // 32-byte aligned: aligned stores
    variant::avx2::build_table(rt.begin(), (std::size_t(1) << lg) / 16, false);
    variant::avx2::build_table(irt.begin(), (std::size_t(1) << lg) / 16, true);
    const u32 *r = rt.begin(), *ir = irt.begin();
    const std::size_t h4_groups = (std::size_t(1) << lg) / 128;  // groups at h = 4: 16 vectors each
    for (int kind = 0; kind < 4; ++kind) {
        for (const std::size_t count : {1, 2, 3, 8, 16}) {
            for (const std::size_t k : {std::size_t(0), std::size_t(1), std::size_t(7), std::size_t(8),
                                        std::size_t(h4_groups - count), std::size_t(random_below(h4_groups - count))}) {
                // inverse_h4 against inverse(): in place, inputs < 2P.
                Vectors a(16 * count), b(16 * count);
                fill(a.begin(), a.count, 2 * P, kind);
                std::copy_n(a.begin(), a.count, b.begin());
                kn::inverse_h4(a.v(), count, ir, k);
                for (std::size_t g = 0; g < count; ++g) av::inverse_group(b.v() + 16 * g, 4, k + g, ir);
                bool ok = k == 0 ? equal_mod(a.begin(), b.begin(), a.count) : std::equal(a.begin(), a.begin() + a.count, b.begin());
                ok &= below(a.begin(), a.count, 2 * P) && a.padding_kept();
                expect(ok, "inverse_h4", (k << 8) | (count << 2) | std::size_t(kind));
                // forward_h4_pair against forward_pair (forward_identity for group 0): inputs < 4P.
                Vectors x(16 * count), y(16 * count), x2(16 * count), y2(16 * count);
                fill(x.begin(), x.count, 4 * P, kind);
                fill(y.begin(), y.count, 4 * P, (kind + 1) % 4);
                std::copy_n(x.begin(), x.count, x2.begin());
                std::copy_n(y.begin(), y.count, y2.begin());
                kn::forward_h4_pair(x.v(), y.v(), count, r, k);
                for (std::size_t g = 0; g < count; ++g) av::forward_groups(x2.v() + 16 * g, y2.v() + 16 * g, 4, k + g, r);
                ok = k == 0 ? equal_mod(x.begin(), x2.begin(), x.count) && equal_mod(y.begin(), y2.begin(), y.count)
                            : std::equal(x.begin(), x.begin() + x.count, x2.begin()) && std::equal(y.begin(), y.begin() + y.count, y2.begin());
                ok &= below(x.begin(), x.count, 4 * P) && below(y.begin(), y.count, 4 * P) && x.padding_kept() && y.padding_kept();
                expect(ok, "forward_h4_pair", (k << 8) | (count << 2) | std::size_t(kind));
            }
        }
        // inverse_leaves_loop against inverse_leaves: canonical leaves in, < 2P out.
        const std::size_t h1_groups = (std::size_t(1) << lg) / 32;
        for (const std::size_t count : {8, 16, 64}) {
            for (const std::size_t first : {std::size_t(0), std::size_t(8), std::size_t(h1_groups - count),
                                            std::size_t(random_below(h1_groups - count) / 8 * 8)}) {
                Vectors a(4 * count), b(4 * count);
                fill(a.begin(), a.count, P, kind);
                std::copy_n(a.begin(), a.count, b.begin());
                kn::inverse_leaves_loop(a.v(), count, ir + slot(first), ir + slot(2 * first));
                for (std::size_t g = first; g < first + count; g += 2)
                    kn::inverse_leaves(b.v() + 4 * (g - first), ir + slot(g), ir + slot(2 * g));
                bool ok = std::equal(a.begin(), a.begin() + a.count, b.begin());
                ok &= below(a.begin(), a.count, 2 * P) && a.padding_kept();
                expect(ok, "inverse_leaves_loop", (first << 8) | (count << 2) | std::size_t(kind));
            }
        }
    }
}
#endif

std::vector<u32> scalar_coefficients(const Transform& t, const std::vector<u32>& F) {
    Words w(t.length());
    std::copy(F.begin(), F.end(), w.begin());
    t.inverse<variant::scalar>(w.begin());
    return {w.begin(), w.begin() + t.length()};
}

std::vector<u32> scalar_leaves(const Transform& t, const std::vector<u32>& f) {
    Words w(t.length());
    std::copy(f.begin(), f.end(), w.begin());
    t.forward<variant::scalar>(w.begin());
    return {w.begin(), w.begin() + t.length()};
}

// f g mod (x^L - 1) from the scalar model.
std::vector<u32> scalar_cyclic(const Transform& t, const std::vector<u32>& f, const std::vector<u32>& g) {
    const std::size_t len = t.length();
    Words F(len), G(len);
    std::copy(f.begin(), f.end(), F.begin());
    std::copy(g.begin(), g.end(), G.begin());
    t.forward<variant::scalar>(F.begin());
    t.forward<variant::scalar>(G.begin());
    t.multiply<variant::scalar>(F.begin(), G.begin(), F.begin());
    t.inverse<variant::scalar>(F.begin());
    return {F.begin(), F.begin() + len};
}

template <class V>
void test_ops(const std::string& name, int max_lg) {
    for (int lg = kMinLog; lg <= max_lg; ++lg) {
        const Transform t(lg);
        const std::size_t len = t.length();
        const bool scalar_ref = lg <= 18;
        for (int kind = 0; kind < 2; ++kind) {
            // Inverse in place and into other memory (F kept).
            std::vector<u32> F(len);
            for (auto& x : F) x = kind ? P - 1 : random_below(P);
            const std::vector<u32> c = scalar_ref ? scalar_coefficients(t, F) : std::vector<u32>{};
            Words w(len);
            std::copy(F.begin(), F.end(), w.begin());
            t.inverse<V>(w.begin());
            const std::vector<u32> got(w.begin(), w.begin() + len);
            if (scalar_ref) expect(got == c, (name + " inverse").c_str(), std::size_t(lg));
            else expect(scalar_leaves(t, got) == F, (name + " inverse (forward back)").c_str(), std::size_t(lg));
            std::vector<u32> out(len - 3);
            t.inverse<V>(F.data(), std::span<u32>(out));
            expect(std::equal(out.begin(), out.end(), got.begin()), (name + " inverse(F, out)").c_str(), std::size_t(lg));
            if (!scalar_ref) continue;
            // Products and squares (Tree's tiles), cyclic_product (cycle_trees).
            const auto f = random_poly(len / 2, kind), g = random_poly(len / 2, kind);
            std::vector<u32> fg(len - 1), ff(len - 1);
            multiply<V>(f, g, fg);
            square<V>(f, ff);
            const auto fz = [&](const std::vector<u32>& p) {
                std::vector<u32> z(len);
                std::copy(p.begin(), p.end(), z.begin());
                return z;
            };
            const std::vector<u32> cyc = scalar_cyclic(t, fz(f), fz(g));
            expect(std::equal(fg.begin(), fg.end(), cyc.begin()), (name + " multiply").c_str(), std::size_t(lg));
            const std::vector<u32> sq = scalar_cyclic(t, fz(f), fz(f));
            expect(std::equal(ff.begin(), ff.end(), sq.begin()), (name + " square").c_str(), std::size_t(lg));
            const auto a = random_poly(len, kind), b = random_poly(len, kind);
            Words B(len), work(len);
            std::copy(b.begin(), b.end(), B.begin());
            t.forward<V>(B.begin());
            std::vector<u32> e(len);
            t.cyclic_product<V>(a, B.begin(), work.begin(), e);
            const std::vector<u32> ab = scalar_cyclic(t, a, b);
            expect(e == ab, (name + " cyclic_product").c_str(), std::size_t(lg));
            // inverse_hooked with and without a column bit (inverse_pruned_with's tiles).
            std::vector<u32> cr = c;  // the hook's leaves carry 2^-32
            for (auto& x : cr) x = mul(x, kR);
            for (std::size_t unused : {std::size_t(0), std::size_t(8), std::size_t(64), len / 16}) {
                if (unused > len / 4) continue;
                std::vector<u32> o(len);
                t.inverse_hooked<V>([&](u32* leaves, std::size_t first, std::size_t n) { std::copy_n(F.begin() + std::ptrdiff_t(8 * first), 8 * n, leaves); },
                                    work.begin(), std::span<u32>(o), {}, unused);
                bool ok = true;
                for (std::size_t i = 0; i < len; ++i) ok &= (i & unused) || o[i] == cr[i];
                expect(ok, (name + " inverse_hooked unused " + std::to_string(unused)).c_str(), std::size_t(lg));
            }
        }
    }
}

}  // namespace

int main() {
#ifdef __AVX2__
    test_kernels();
#endif
    test_ops<variant::scalar>("scalar", kScalarTestLog);
    fps::simd::for_each(fps::simd::All{}, []<class S> {
        test_ops<variant::intrinsics<S>>(std::string("intrinsics ") + S::name, S::lanes == 1 ? kScalarTestLog : kMaxTestLog);
    });
#ifdef __AVX2__
    test_ops<variant::avx2>("avx2", kMaxTestLog);
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    test_ops<variant::avx512>("avx512", kMaxTestLog);
#endif
    test_ops<variant::tuned>("tuned", kMaxTestLog);
    return finish();
}
