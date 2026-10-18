// Tests of ntt::dft and ntt::idft (dft.hpp, issue #138), for every variant this machine runs and the
// tuned one, against chirp::evaluate_geometric at w and w^-1 (backend/chirp), every length 2^6 ..
// 2^23 (variants but the tuned one and Native to 2^16; under ASan to 2^20):
// - dft of f of 0, 1, L / 2, L - 1, L, L + 1, 2L + 3 and 3L words (longer ones fold mod x^L - 1;
//   fewer sizes from 2^13), random, all P - 1 and zero, into out at several page offsets, out full
//   of garbage; f unchanged.
// - idft of those values gives f mod (x^L - 1); idft of random values against the chirp at w^-1
//   times 1 / L; in place (out = values) and dft in place (out = f).
// - the permutation alone on every simd struct this target runs, both ways, against the order,
//   with the natural side at 6 offsets from a page and staged or not.
// - the errors: L not a power of two, below 2^6 or above 2^23, idft's sizes apart.
#include <sys/mman.h>

#include "backend/chirp/geometric.hpp"
#include "backend/ntt/dft.hpp"
#include "backend/ntt/test_util.hpp"

namespace {

int checks = 0;

void check(bool ok, const std::string& what, u64 detail) {
    ++checks;
    expect(ok, what.c_str(), detail);
}

#if __has_feature(address_sanitizer)
constexpr int kLongLog = 20;
#else
constexpr int kLongLog = kValuesMaxLog;
#endif
constexpr int kEveryVariantLog = 16;

u32 root_of(int lg, bool inverse) {
    const u32 w = power(3, (P - 1) >> lg);
    return inverse ? power(w, P - 2) : w;
}

// f mod (x^L - 1).
std::vector<u32> folded(const std::vector<u32>& f, std::size_t L) {
    std::vector<u32> r(L);
    for (std::size_t i = 0; i < f.size(); ++i) r[i % L] = add(r[i % L], f[i]);
    return r;
}

// out[i] = f(w^i) by the chirp z-transform.
std::vector<u32> chirp_dft(const std::vector<u32>& f, std::size_t L, u32 w) {
    std::vector<u32> out(L);
    fps::chirp::evaluate_geometric(f, 1, w, out);
    return out;
}

struct Case {
    std::vector<u32> f, values;  // values[i] = f(w^i)
};

std::vector<std::size_t> sizes(int lg) {
    const std::size_t L = std::size_t(1) << lg;
    if (lg <= 12) return {0, 1, L / 2, L - 1, L, L + 1, 2 * L + 3, 3 * L};
    if (lg <= 18) return {L / 2 + 3, L, 2 * L + 1};
    return {L};
}

template <class V>
void test_variant(const std::string& name, int lg, const std::vector<Case>& cases, const std::vector<u32>& y,
                  const std::vector<u32>& y_inverse) {
    const std::size_t L = std::size_t(1) << lg;
    for (std::size_t c = 0; c < cases.size(); ++c) {
        const Case& k = cases[c];
        const std::size_t offset = (c * 5 + std::size_t(lg)) % 23;  // words past a page: any alignment
        PageWords fw(k.f.size(), offset + 3, 0), out(L, offset, 0);
        std::copy(k.f.begin(), k.f.end(), fw.begin());
        for (u32& x : out.span()) x = u32(rng());
        dft<V>(std::span<const u32>(fw.begin(), k.f.size()), out.span());
        check(std::ranges::equal(out.span(), k.values), name + " dft", u64(lg) * 100000000 + k.f.size());
        check(std::ranges::equal(std::span(fw.begin(), k.f.size()), k.f), name + " dft wrote f", u64(lg));
        PageWords back(L, offset + 7, 0);
        for (u32& x : back.span()) x = u32(rng());
        idft<V>(out.span(), back.span());
        check(std::ranges::equal(back.span(), folded(k.f, L)), name + " idft(dft(f))", u64(lg) * 100000000 + k.f.size());
    }
    // idft of random values, in place.
    std::vector<u32> g(y);
    idft<V>(g, g);
    check(g == y_inverse, name + " idft in place", u64(lg));
    // dft in place: out = f, f of L words.
    dft<V>(g, g);
    check(g == y, name + " dft in place", u64(lg));
}

// The permutation alone on S, the natural side `offset` words past a page and ending its
// allocation: word p = 64 g + 8 i + t to e = rev6(8 t + i) 2^(lg - 6) + rev(g), and back. Tiles of
// 2^kRho blocks from 2^12; each side through its buffer from 2^0 (kStaged) or never.
template <class S, int kRho, bool kValues, bool kNatural>
void test_permute(int lg, std::size_t offset) {
    using fps::ntt::detail::dft::reverse;
    namespace dd = fps::ntt::detail::dft;
    constexpr int kValuesLog = kValues ? 0 : dd::kNever, kNaturalLog = kNatural ? 0 : dd::kNever;
    const std::size_t L = std::size_t(1) << lg;
    const int n = lg - 6;
    Words from(L), back(L);
    PageWords to(L, offset, 0);
    for (std::size_t p = 0; p < L; ++p) from.begin()[p] = u32(p);
    dd::permute<S, false, kRho, 12, kValuesLog, kNaturalLog>(from.begin(), to.begin(), lg);
    bool ok = true;
    for (std::size_t p = 0; p < L; ++p) {
        const std::size_t g = p / 64, i = p / 8 % 8, t = p % 8;
        ok &= to.begin()[(reverse(8 * t + i, 6) << n) + reverse(g, n)] == p;
    }
    const std::string name = std::string(S::name) + " tiles " + std::to_string(kRho) + (kValues ? " values" : "") +
                             (kNatural ? " natural" : "");
    check(ok, "permute " + name, u64(lg) * 100 + offset);
    dd::permute<S, true, kRho, 12, kValuesLog, kNaturalLog>(to.begin(), back.begin(), lg);
    check(std::equal(from.begin(), from.begin() + L, back.begin()), "permute back " + name, u64(lg) * 100 + offset);
}

template <class E>
bool throws(auto f) {
    try {
        f();
    } catch (const E&) {
        return true;
    }
    return false;
}

void test_errors() {
    std::vector<u32> f(100, 1), out(1 << 10);
    for (std::size_t L : {0, 1, 2, 32, 48, 96, 100, 1000}) {
        const bool power = std::has_single_bit(L);
        std::span<u32> o(out.data(), L);
        if (power) {
            check(throws<std::length_error>([&] { dft(f, o); }), "dft below 2^6", L);
            check(throws<std::length_error>([&] { idft(o, o); }), "idft below 2^6", L);
        } else {
            check(throws<std::invalid_argument>([&] { dft(f, o); }), "dft not a power of two", L);
            check(throws<std::invalid_argument>([&] { idft(o, o); }), "idft not a power of two", L);
        }
    }
    check(throws<std::invalid_argument>([&] { idft(std::span<const u32>(out.data(), 64), std::span(out.data(), 128)); }),
          "idft sizes apart", 0);
    // 2^24 and 2^25 words: unwritten memory, never touched.
    const std::size_t words = std::size_t(1) << 25, bytes = words * sizeof(u32);
    void* map = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    check(map != MAP_FAILED, "mmap", 0);
    if (map == MAP_FAILED) return;
    const std::span<u32> big(static_cast<u32*>(map), words);
    for (std::size_t L : {std::size_t(1) << 24, std::size_t(1) << 25}) {
        check(throws<std::length_error>([&] { dft(f, big.first(L)); }), "dft above 2^23", L);
        check(throws<std::length_error>([&] { idft(big.first(L), big.first(L)); }), "idft above 2^23", L);
    }
    munmap(map, bytes);
}

}  // namespace

int main() {
    for (int lg = kMinLog; lg <= kLongLog; ++lg) {
        const std::size_t L = std::size_t(1) << lg;
        const u32 w = root_of(lg, false), wi = root_of(lg, true), inverse_L = power(u32(L), P - 2);
        std::vector<Case> cases;
        int kind = 0;
        for (const std::size_t n : sizes(lg)) {
            Case c;
            c.f = random_poly(n, lg <= 12 ? kind++ % 3 : 0);
            c.values = chirp_dft(c.f, L, w);
            cases.push_back(std::move(c));
        }
        const std::vector<u32> y = random_poly(L, 0);
        std::vector<u32> y_inverse = chirp_dft(y, L, wi);
        for (u32& x : y_inverse) x = mul(x, inverse_L);
        const auto run = [&]<class V>(const char* name, int max_lg) {
            if (lg <= std::min(max_lg, kEveryVariantLog)) test_variant<V>(name, lg, cases, y, y_inverse);
        };
        for_each_variant(run);
#if defined(__ARM_NEON) && defined(__aarch64__)
        if (lg <= kEveryVariantLog) test_variant<variant::neon>("neon", lg, cases, y, y_inverse);
#endif
        test_variant<variant::tuned>("tuned", lg, cases, y, y_inverse);
        if (lg > kEveryVariantLog) test_variant<Native>("native", lg, cases, y, y_inverse);
        if (lg <= 16)
            fps::simd::for_each(fps::simd::All{}, [lg]<class S> {
                for (const std::size_t offset : {0, 1, 4, 15, 16, 17})
                    test_permute<S, 6, false, false>(lg, offset), test_permute<S, 6, true, true>(lg, offset),
                        test_permute<S, 5, true, false>(lg, offset), test_permute<S, 4, false, true>(lg, offset);
            });
    }
    test_errors();
    std::printf("%d checks\n", checks);
    return finish();
}
