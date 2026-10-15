// Tests for Transform's doubling ops (issue #149 round 2): truncate_blocks, join_blocks and
// forward_upper_hooked, and forward_hooked's unused source words (32 <= zero < length / 4), on
// every variant this machine runs and the tuned one. References: the definitions (f mod (x^b -
// r(j)^2) from the coefficients, at short lengths) and the scalar variant; truncate_blocks then
// join_blocks gives the rows cut to x < b / 2; the halves of a transform of 2L from these ops
// against forward() of 2L.
#include "backend/ntt/test_util.hpp"

namespace {

#if __has_feature(address_sanitizer)
constexpr int kMaxTestLog = 17;
#else
constexpr int kMaxTestLog = 20;
#endif
constexpr int kScalarTestLog = 14;
constexpr int kDefinitionLog = 11;  // the definitions are quadratic

std::vector<u32> coefficients(const Transform& t, const std::vector<u32>& F) {
    Words w(t.length());
    std::copy(F.begin(), F.end(), w.begin());
    t.inverse<variant::scalar>(w.begin());
    return {w.begin(), w.begin() + t.length()};
}

std::vector<u32> leaves_of(const Transform& t, const std::vector<u32>& f) {
    Words w(t.length());
    std::copy(f.begin(), f.end(), w.begin());
    t.forward<variant::scalar>(w.begin());
    return {w.begin(), w.begin() + t.length()};
}

// f mod (x^b - c), b words.
std::vector<u32> residue(const std::vector<u32>& f, std::size_t b, u32 c) {
    std::vector<u32> g(b, 0);
    u32 ct = 1;
    for (std::size_t t = 0; t < f.size(); t += b, ct = mul(ct, c))
        for (std::size_t i = 0; i < b; ++i) g[i] = add(g[i], mul(f[t + i], ct));
    return g;
}

// The runs a hook saw: each leaf of [lo, hi) once, runs of a power of two >= 4 at a multiple of the
// run, in increasing order.
struct Runs {
    Runs(std::size_t lo_, std::size_t hi_) : lo(lo_), next(lo_), seen(hi_ - lo_, 0) {}
    void add(std::size_t first, std::size_t count) {
        ok &= count >= 4 && (count & (count - 1)) == 0 && first % count == 0 && first >= next && first + count <= lo + seen.size();
        next = first + count;
        for (std::size_t v = first; v < first + count && v >= lo && v - lo < seen.size(); ++v) ++seen[v - lo];
    }
    bool good() const { return ok && std::all_of(seen.begin(), seen.end(), [](int c) { return c == 1; }); }
    std::size_t lo, next;
    std::vector<int> seen;
    bool ok = true;
};

std::vector<std::size_t> blocks(std::size_t len) {
    std::vector<std::size_t> b;
    for (std::size_t x = 32; x <= len / 8; x *= 2) b.push_back(x);
    return b;
}

// truncate_blocks on V from a random leaf form F; returns F's blocks and copy's halves (upper
// halves of copy cleared), and checks the hook.
template <class V>
std::pair<std::vector<u32>, std::vector<u32>> truncated(const std::string& name, const Transform& t, const std::vector<u32>& F,
                                                        std::size_t b, int lg) {
    const std::size_t len = t.length();
    Words G(len), copy(len);
    std::copy(F.begin(), F.end(), G.begin());
    for (std::size_t i = 0; i < len; ++i) copy.begin()[i] = 12345;
    Runs runs(0, len / 8);
    std::vector<u32> seen(len, 0);
    t.truncate_blocks<V>(G.begin(), b, copy.begin(), [&](u32* leaves, std::size_t first, std::size_t count) {
        runs.add(first, count);
        expect(leaves == G.begin() + 8 * first, (name + " truncate_blocks: hook pointer").c_str(), lg);
        for (std::size_t i = 0; i < 8 * count; ++i) {
            seen[8 * first + i] = leaves[i];
            leaves[i] = leaves[i] + 1 == P ? 0 : leaves[i] + 1;  // the hook's changes stay
        }
    });
    expect(runs.good(), (name + " truncate_blocks: runs, block " + std::to_string(b)).c_str(), lg);
    std::vector<u32> out(G.begin(), G.begin() + len), halves(len, 0);
    bool kept = true;
    for (std::size_t i = 0; i < len; ++i) kept &= out[i] == (seen[i] + 1 == P ? 0 : seen[i] + 1);
    expect(kept, (name + " truncate_blocks: hook saw the leaves, changes kept").c_str(), lg);
    for (std::size_t i = 0; i < len; ++i) {
        out[i] = seen[i];
        if (i % b < b / 2) halves[i] = copy.begin()[i];
    }
    return {out, halves};
}

// The scalar variant against the definition: F's block j becomes the leaf form of c_j, copy holds
// c_j, c_j = (f mod (x^b - r(j)^2)) mod x^(b/2) times b / 8.
void test_definition(int lg) {
    const Transform t(lg);
    const std::size_t len = t.length();
    std::vector<u32> F(len);
    for (auto& x : F) x = random_below(P);
    const std::vector<u32> f = coefficients(t, F);
    for (std::size_t b : blocks(len)) {
        const auto [out, halves] = truncated<variant::scalar>("scalar", t, F, b, lg);
        // c: the polynomial with rows f's rows cut to x < b/2, times b / 8; c mod (x^b - r(j)^2) = c_j.
        std::vector<u32> c(len, 0);
        for (std::size_t i = 0; i < len; ++i)
            if (i % b < b / 2) c[i] = mul(f[i], u32(b / 8));
        bool ok = true;
        for (std::size_t j = 0; j < len / b; ++j) {
            const std::vector<u32> cj = residue(c, b, mul(root(j), root(j)));
            for (std::size_t i = 0; i < b / 2; ++i) ok &= halves[j * b + i] == cj[i];
            for (std::size_t i = b / 2; i < b; ++i) ok &= cj[i] == 0;
        }
        expect(ok, "scalar truncate_blocks: copy against the definition", b);
        expect(out == leaves_of(t, c), "scalar truncate_blocks: leaves against the definition", b);
    }
}

template <class V>
void test_truncate(const std::string& name, int lg) {
    const Transform t(lg);
    const std::size_t len = t.length();
    std::vector<u32> F(len);
    for (auto& x : F) x = random_below(P);
    for (std::size_t b : blocks(len)) {
        const auto [out, halves] = truncated<V>(name, t, F, b, lg);
        const auto [want, want_halves] = truncated<variant::scalar>("scalar", t, F, b, lg);
        expect(out == want, (name + " truncate_blocks: leaves, block " + std::to_string(b)).c_str(), lg);
        expect(halves == want_halves, (name + " truncate_blocks: copy, block " + std::to_string(b)).c_str(), lg);
        // join_blocks undoes the levels above the blocks: the rows of f cut to x < b/2.
        Words w(len);
        for (std::size_t i = 0; i < len; ++i) w.begin()[i] = i % b < b / 2 ? halves[i] : random_below(P);
        t.join_blocks<V>(w.begin(), b, power(u32(b / 8), P - 2));
        const std::vector<u32> f = coefficients(t, F);
        bool ok = true;
        for (std::size_t i = 0; i < len; ++i)
            if (i % b < b / 2) ok &= w.begin()[i] == f[i];
        expect(ok, (name + " join_blocks after truncate_blocks, block " + std::to_string(b)).c_str(), lg);
        if (lg > 16 && b > 256) break;
    }
}

// join_blocks with a scale against its definition (short lengths, scalar) and the scalar variant.
template <class V>
void test_join(const std::string& name, int lg) {
    const Transform t(lg);
    const std::size_t len = t.length();
    for (std::size_t b : blocks(len)) {
        std::vector<u32> in(len);
        for (auto& x : in) x = random_below(P);
        const u32 scale = random_below(P);
        Words w(len), s(len);
        std::copy(in.begin(), in.end(), w.begin());
        std::copy(in.begin(), in.end(), s.begin());
        t.join_blocks<V>(w.begin(), b, scale);
        t.join_blocks<variant::scalar>(s.begin(), b, scale);
        bool same = true, defined = true;
        for (std::size_t i = 0; i < len; ++i)
            if (i % b < b / 2) same &= w.begin()[i] == s.begin()[i];
        if (lg <= kDefinitionLog) {
            std::vector<u32> c(len, 0);
            for (std::size_t i = 0; i < len; ++i)
                if (i % b < b / 2) c[i] = s.begin()[i];
            const u32 inverse = power(scale, P - 2);
            for (std::size_t j = 0; j < len / b; ++j) {
                const std::vector<u32> cj = residue(c, b, mul(root(j), root(j)));
                for (std::size_t i = 0; i < b / 2; ++i) defined &= mul(cj[i], inverse) == in[j * b + i];
            }
        }
        expect(same, (name + " join_blocks against scalar, block " + std::to_string(b)).c_str(), lg);
        expect(defined, (name + " join_blocks against the definition, block " + std::to_string(b)).c_str(), lg);
        if (lg > 16 && b > 256) break;
    }
}

// forward_upper_hooked against forward_upper of the source with the column bit's words zeroed; the
// source's words there are random (not used from 16 on), or zero below 16.
template <class V>
void test_upper(const std::string& name, int lg) {
    const Transform t(lg);
    const std::size_t len = t.length(), half = len / 2;
    std::vector<std::size_t> bits = {0};
    for (std::size_t z = 1; z < len / 8; z *= 2) bits.push_back(z);
    for (std::size_t zero : bits) {
        std::vector<u32> h(half), clean(half);
        for (std::size_t i = 0; i < half; ++i) {
            h[i] = random_below(P);
            clean[i] = i & zero ? 0 : h[i];
            if ((i & zero) && zero < 16) h[i] = 0;
        }
        Words want(half), got(half);
        std::copy(clean.begin(), clean.end(), want.begin());
        t.forward_upper<variant::scalar>(want.begin());
        std::copy(h.begin(), h.end(), got.begin());
        Runs runs(len / 16, len / 8);
        bool pointer = true;
        t.forward_upper_hooked<V>(got.begin(), [&](u32* leaves, std::size_t first, std::size_t count) {
            runs.add(first, count);
            pointer &= leaves == got.begin() + 8 * (first - len / 16);
        }, zero);
        bool ok = true;
        for (std::size_t i = 0; i < half; ++i) ok &= got.begin()[i] == want.begin()[i];
        const std::string what = name + " forward_upper_hooked zero " + std::to_string(zero);
        expect(ok, what.c_str(), lg);
        expect(runs.good() && pointer, (what + ": runs").c_str(), lg);
        if (lg > 16 && zero > 1024) break;
    }
}

// forward_hooked with random words where i & zero is not 0, 32 <= zero < length / 4.
template <class V>
void test_unused(const std::string& name, int lg) {
    const Transform t(lg);
    const std::size_t len = t.length();
    for (std::size_t n : {len, len / 2, len / 4}) {  // runs in order without a column bit too
        const std::vector<u32> in = random_poly(n, 0);
        Words F(len);
        Runs runs(0, len / 8);
        t.forward_hooked<V>(std::span<const u32>(in), F.begin(), [&](u32*, std::size_t first, std::size_t count) {
            runs.add(first, count);
        });
        expect(runs.good(), (name + " forward_hooked: runs in order, n " + std::to_string(n)).c_str(), lg);
    }
    for (std::size_t zero = detail::kUnreadWidth; zero < len / 4; zero *= 2) {
        for (std::size_t n : {len, len / 2 + 3, len / 2}) {
            std::vector<u32> in(n), clean(n);
            for (std::size_t i = 0; i < n; ++i) in[i] = random_below(P), clean[i] = i & zero ? 0 : in[i];
            const std::vector<u32> want = [&] {
                std::vector<u32> c(len, 0);
                std::copy(clean.begin(), clean.end(), c.begin());
                return leaves_of(t, c);
            }();
            Words F(len);
            Runs runs(0, len / 8);
            t.forward_hooked<V>(std::span<const u32>(in), F.begin(), [&](u32*, std::size_t first, std::size_t count) {
                runs.add(first, count);
            }, zero);
            bool ok = true;
            for (std::size_t i = 0; i < len; ++i) ok &= F.begin()[i] == want[i];
            const std::string what = name + " forward_hooked, zero " + std::to_string(zero) + " n " + std::to_string(n);
            expect(ok, (what + ": words past the column bit unused").c_str(), lg);
            expect(runs.good(), (what + ": runs in order").c_str(), lg);
        }
        if (lg > 16 && zero > 1024) break;
    }
}

// The transform of 2L of Q (rows of b words, x < b/2, degree below L) from the ops: the lower half
// by truncate_blocks on any leaf form of length L whose rows cut to x < b/2 are Q's (times b / 8),
// the upper half by join_blocks and forward_upper_hooked.
template <class V>
void test_doubling(const std::string& name, int lg) {
    const Transform t(lg), t2(lg + 1);
    const std::size_t len = t.length();
    for (std::size_t b : blocks(len)) {
        std::vector<u32> g(len), q(2 * len, 0);
        for (std::size_t i = 0; i < len; ++i) {
            g[i] = random_below(P);
            if (i % b < b / 2) q[i] = g[i];
        }
        Words F(2 * len);
        const std::vector<u32> G = leaves_of(t, g);
        std::copy(G.begin(), G.end(), F.begin());
        t.truncate_blocks<V>(F.begin(), b, F.begin() + len, [](u32*, std::size_t, std::size_t) {});
        t.join_blocks<V>(F.begin() + len, b, power(u32(b / 8), P - 2));
        t2.forward_upper_hooked<V>(F.begin() + len, [](u32*, std::size_t, std::size_t) {}, b / 2);
        const std::vector<u32> want = leaves_of(t2, q);
        bool ok = true;
        for (std::size_t i = 0; i < len; ++i) ok &= F.begin()[i] == mul(want[i], u32(b / 8));
        for (std::size_t i = len; i < 2 * len; ++i) ok &= F.begin()[i] == want[i];
        expect(ok, (name + " doubling, block " + std::to_string(b)).c_str(), lg);
        if (lg > 16 && b > 256) break;
    }
}

template <class V>
void test_variant(const std::string& name, int max_lg) {
    for (int lg = 8; lg <= max_lg; ++lg) {
        test_truncate<V>(name, lg);
        test_join<V>(name, lg);
        test_upper<V>(name, lg);
        test_unused<V>(name, lg);
        if (lg < max_lg) test_doubling<V>(name, lg);
    }
}

void test_checks() {
    const Transform t(10);
    Words F(1024), copy(1024);
    const auto none = [](u32*, std::size_t, std::size_t) {};
    for (std::size_t b : {std::size_t(16), std::size_t(48), std::size_t(256)}) {
        bool threw = false;
        try {
            t.truncate_blocks(F.begin(), b, copy.begin(), none);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        expect(threw, "truncate_blocks: block out of range", b);
        threw = false;
        try {
            t.join_blocks(F.begin(), b, 1);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        expect(threw, "join_blocks: block out of range", b);
    }
    bool threw = false;
    try {
        Transform(kMinLog).forward_upper_hooked(F.begin(), none);
    } catch (const std::length_error&) {
        threw = true;
    }
    expect(threw, "forward_upper_hooked: lg = kMinLog");
}

}  // namespace

int main() {
    test_checks();
    for (int lg = 8; lg <= kDefinitionLog; ++lg) test_definition(lg);
    test_variant<variant::scalar>("scalar", kScalarTestLog);
    fps::simd::for_each(fps::simd::All{}, []<class S> {
        test_variant<variant::intrinsics<S>>(std::string("intrinsics ") + S::name, S::lanes == 1 ? kScalarTestLog : kMaxTestLog);
    });
#ifdef __AVX2__
    test_variant<variant::avx2>("avx2", kMaxTestLog);
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    test_variant<variant::avx512>("avx512", kMaxTestLog);
#endif
    test_variant<variant::tuned>("tuned", kMaxTestLog);
    return finish();
}
