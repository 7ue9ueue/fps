// Tests of fps/values.hpp in both layouts: conversions, products and sums against cyclic schoolbook
// products mod (x^L - 1), the raw functions at every offset, DESIGN.md's two examples (many
// products with one factor; ABC222 H's online self-convolution), long lengths, threads, errors.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "fps/fps.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = std::vector<u32>;
constexpr u32 P = 998244353;
using Poly = fps::Polynomial<P>;
using Values = fps::Values<P>;
using fps::Layout;

int failures = 0;

void check(bool ok, const char* what, int line, u64 a = 0, u64 b = 0) {
    if (ok) return;
    if (++failures <= 20)
        std::printf("FAIL line %d: %s (%llu, %llu)\n", line, what, (unsigned long long)a, (unsigned long long)b);
}
#define CHECK(cond, ...) check(bool(cond), #cond, __LINE__ __VA_OPT__(, ) __VA_ARGS__)

template <class Exception, class F>
bool throws(F f) {
    try {
        f();
    } catch (const Exception&) {
        return true;
    }
    return false;
}

std::mt19937_64 rng(19);

constexpr Layout kLayouts[] = {Layout::automatic, Layout::pointwise};

// Lengths 2^0 .. 2^11: below, at and above the backend's shortest transform (2^6).
std::vector<std::size_t> lengths() {
    std::vector<std::size_t> L;
    for (int lg = 0; lg <= 11; ++lg) L.push_back(std::size_t(1) << lg);
    return L;
}

// Sizes of an input at length L: 0, 1, powers of two up to L and their neighbours, L, and random.
std::vector<std::size_t> sizes(std::size_t L) {
    std::vector<std::size_t> s = {0, 1, L, std::size_t(rng() % (L + 1))};
    for (std::size_t p = 2; p <= L; p *= 2) s.insert(s.end(), {p - 1, p, p + 1});
    std::erase_if(s, [L](std::size_t n) { return n > L; });
    return s;
}

// Kinds: 0 random, 1 all P - 1.
Vec random_vec(std::size_t n, int kind = 0) {
    Vec a(n);
    for (u32& c : a) c = kind == 1 ? P - 1 : u32(rng() % P);
    return a;
}

Vec vec(const Poly& f) { return Vec(f.span().begin(), f.span().end()); }
Poly poly(const Vec& a) { return Poly(std::span<const u32>(a)); }

u32 add(u32 a, u32 b) { return u32((u64(a) + b) % P); }
u32 sub(u32 a, u32 b) { return u32((u64(a) + P - b) % P); }
u32 mul(u32 a, u32 b) { return u32(u64(a) * b % P); }

// a mod (x^L - 1), L coefficients.
Vec fold(const Vec& a, std::size_t L) {
    Vec c(L);
    for (std::size_t i = 0; i < a.size(); ++i) c[i % L] = add(c[i % L], a[i]);
    return c;
}

// a b mod (x^L - 1).
Vec cyclic(const Vec& a, const Vec& b, std::size_t L) {
    Vec c(L);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[(i + j) % L] = add(c[(i + j) % L], mul(a[i], b[j]));
    return c;
}

template <class Op>
Vec termwise(const Vec& a, const Vec& b, Op op) {
    Vec c(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) c[i] = op(a[i], b[i]);
    return c;
}

Values values(const Vec& a, std::size_t L, Layout layout) { return fps::to_values(poly(a), L, layout); }

// to_coefficients on an lvalue, and on an rvalue copy (transformed in place).
Vec coefficients(const Values& A) {
    const Vec c = vec(fps::to_coefficients(A));
    Values copy = A;
    CHECK(vec(fps::to_coefficients(std::move(copy))) == c, A.size());
    CHECK(copy.size() == 0);
    return c;
}

void test_conversions() {
    for (const Layout layout : kLayouts)
        for (const std::size_t L : lengths())
            for (const std::size_t n : sizes(L))
                for (int kind = 0; kind < 2; ++kind) {
                    const Vec a = random_vec(n, kind);
                    const Values A = values(a, L, layout);
                    CHECK(A.size() == L);
                    CHECK(coefficients(A) == fold(a, L), L, n);
                    CHECK(coefficients(Values(L, layout)) == Vec(L), L);
                }
}

void test_products() {
    for (const Layout layout : kLayouts)
        for (const std::size_t L : lengths())
            for (const std::size_t n : sizes(L)) {
                const int kind = rng() % 4 == 0;
                const Vec a = random_vec(n, kind), b = random_vec(rng() % (L + 1), kind);
                const Vec c = random_vec(rng() % (L + 1)), d = random_vec(L, kind);
                const Values A = values(a, L, layout), B = values(b, L, layout);
                const Values C = values(c, L, layout), D = values(d, L, layout);
                const Vec ab = cyclic(a, b, L), cd = cyclic(c, d, L);
                const Vec sum = termwise(ab, cd, add), difference = termwise(ab, cd, sub);
                const Vec fa = fold(a, L), fb = fold(b, L);

                CHECK(coefficients(A * B) == ab, L, n);
                CHECK(coefficients(Values(A) * B) == ab, L, n);
                CHECK(coefficients(A + B) == termwise(fa, fb, add), L, n);
                CHECK(coefficients(Values(A) + B) == termwise(fa, fb, add), L, n);
                CHECK(coefficients(A - B) == termwise(fa, fb, sub), L, n);
                CHECK(coefficients(Values(A) - B) == termwise(fa, fb, sub), L, n);
                CHECK(coefficients(A * B + C * D) == sum, L, n);
                CHECK(coefficients(A * B - C * D) == difference, L, n);
                CHECK(coefficients(A * B * C) == cyclic(ab, c, L), L, n);

                Values E = A;
                E *= B;
                CHECK(coefficients(E) == ab, L, n);
                E += C * D;
                CHECK(coefficients(E) == sum, L, n);
                E -= C * D;
                E -= C * D;
                CHECK(coefficients(E) == difference, L, n);

                Values F(L, layout);
                F.multiply_add(A, B);
                F.multiply_add(C, D);
                CHECK(coefficients(F) == sum, L, n);

                // Aliasing: A A, A += A, A -= A, A += A B, A += B A, A += A A.
                Values G = A;
                G *= G;
                CHECK(coefficients(G) == cyclic(a, a, L), L, n);
                G = A;
                G += G;
                CHECK(coefficients(G) == termwise(fa, fa, add), L, n);
                G -= G;
                CHECK(coefficients(G) == Vec(L), L, n);
                for (int form = 0; form < 3; ++form) {
                    G = A;
                    if (form == 0) G.multiply_add(G, B);
                    if (form == 1) G.multiply_add(B, G);
                    if (form == 2) G.multiply_add(G, G);
                    const Vec want = termwise(fa, form == 2 ? cyclic(a, a, L) : ab, add);
                    CHECK(coefficients(G) == want, L, form);
                }

                // Scalars: integers of at most 64 bits, taken mod P.
                const u32 r = u32(rng() % P);
                for (const auto& [scalar, residue] : std::vector<std::pair<long long, u32>>{
                         {0, 0}, {1, 1}, {P - 1, P - 1}, {r, r}, {-1, P - 1}, {-3LL * P - 5, P - 5}}) {
                    Vec want(L);
                    for (std::size_t i = 0; i < L; ++i) want[i] = mul(ab[i], residue);
                    CHECK(coefficients(scalar * (A * B)) == want, L, residue);
                    CHECK(coefficients((A * B) * scalar) == want, L, residue);
                    Values H = A * B;
                    H *= scalar;
                    CHECK(coefficients(H) == want, L, residue);
                }
                const u64 big = ~u64(0);
                Vec want(L);
                for (std::size_t i = 0; i < L; ++i) want[i] = mul(fa[i], u32(big % P));
                CHECK(coefficients(big * A) == want, L);
            }
}

// raw::to_values into an A that held something; raw::to_coefficients and raw::add_coefficients
// on every range for L <= 32, on edge and random ranges above.
void test_raw() {
    for (const Layout layout : kLayouts)
        for (const std::size_t L : lengths()) {
            const Vec a = random_vec(std::min(L, L / 2 + 1)), b = random_vec(L);
            Values A = values(random_vec(L), L, layout);
            fps::raw::to_values<P>(a, A);
            const Values B = values(b, L, layout);
            const Vec ab = cyclic(a, b, L);
            const Values AB = A * B;
            std::vector<std::pair<std::size_t, std::size_t>> ranges = {
                {0, L}, {L, 0}, {0, 0}, {L - 1, 1}, {L / 2, L - L / 2}};
            if (L <= 32)
                for (std::size_t offset = 0; offset <= L; ++offset)
                    for (std::size_t count = 0; offset + count <= L; ++count) ranges.emplace_back(offset, count);
            for (int t = 0; t < 20; ++t) {
                const std::size_t offset = rng() % (L + 1);
                ranges.emplace_back(offset, rng() % (L - offset + 1));
            }
            for (const auto& [offset, count] : ranges) {
                Vec out = random_vec(count), want = out;
                for (std::size_t i = 0; i < count; ++i) want[i] = add(want[i], ab[offset + i]);
                fps::raw::add_coefficients<P>(AB, out, offset);
                CHECK(out == want, L, offset);
                fps::raw::to_coefficients<P>(AB, out, offset);
                const auto first = ab.begin() + std::ptrdiff_t(offset);
                CHECK(out == Vec(first, first + std::ptrdiff_t(count)), L, offset);
            }
            fps::raw::to_values<P>(std::span<const u32>(), A);
            CHECK(coefficients(A) == Vec(L), L);
        }
}

void test_errors() {
    const Poly f{1, 2, 3};
    for (const Layout layout : kLayouts) {
        for (const std::size_t L : {0, 3, 6, 96, 1000})
            CHECK(throws<std::domain_error>([&] { (void)fps::to_values(Poly{}, L, layout); }), L);
        CHECK(throws<std::domain_error>([&] { Values(3, layout); }));
        CHECK(throws<std::domain_error>([&] { (void)fps::to_values(f, 2, layout); }));
        CHECK(throws<std::length_error>([&] { (void)fps::to_values(f, std::size_t(1) << 26, layout); }));
        CHECK(throws<std::length_error>([&] { Values(std::size_t(1) << 26, layout); }));
        CHECK(throws<std::domain_error>([&] { (void)fps::to_values(f, std::size_t(3) << 26, layout); }));
    }
    const Values A = fps::to_values(f, 64), B = fps::to_values(f, 128), C = fps::to_values(f, 64, Layout::pointwise);
    const Values S = fps::to_values(f, 4), T = fps::to_values(f, 8);  // both held at length 64
    for (const auto& [X, Y] : {std::pair{&A, &B}, std::pair{&A, &C}, std::pair{&S, &T}, std::pair{&A, &S}}) {
        CHECK(throws<std::domain_error>([&] { (void)(*X + *Y); }));
        CHECK(throws<std::domain_error>([&] { (void)(*X - *Y); }));
        CHECK(throws<std::domain_error>([&] { (void)(*X * *Y); }));
        CHECK(throws<std::domain_error>([&] { (void)(Values(*X) + *Y); }));
        CHECK(throws<std::domain_error>([&] { (void)(Values(*X) - *Y); }));
        CHECK(throws<std::domain_error>([&] { (void)(Values(*X) * *Y); }));
        CHECK(throws<std::domain_error>([&] { Values Z = *X; Z += *Y; }));
        CHECK(throws<std::domain_error>([&] { Values Z = *X; Z -= *Y; }));
        CHECK(throws<std::domain_error>([&] { Values Z = *X; Z *= *Y; }));
        CHECK(throws<std::domain_error>([&] { Values Z = *X; Z.multiply_add(*X, *Y); }));
        CHECK(throws<std::domain_error>([&] { Values Z = *X; Z.multiply_add(*Y, *X); }));
        CHECK(throws<std::domain_error>([&] { Values Z = *X; Z.multiply_add(*Y, *Y); }));
    }
    Values D(8);
    CHECK(throws<std::domain_error>([&] { fps::raw::to_values<P>(Vec(9), D); }));
    Vec out(10);
    CHECK(throws<std::domain_error>([&] { fps::raw::to_coefficients<P>(D, out, 0); }));
    CHECK(throws<std::domain_error>([&] { fps::raw::add_coefficients<P>(D, out, 0); }));
    CHECK(throws<std::domain_error>([&] { fps::raw::to_coefficients<P>(D, std::span(out).first(1), 8); }));
    CHECK(throws<std::domain_error>([&] { fps::raw::add_coefficients<P>(D, std::span(out).first(0), 9); }));
    fps::raw::to_coefficients<P>(D, std::span(out).first(0), 8);  // an empty range at L is fine

    // The empty value form (L = 0): only empty ranges, and a mismatch with any other L.
    const Values E;
    CHECK(E.size() == 0);
    CHECK((E + E).size() == 0 && (E * E).size() == 0 && (3 * E).size() == 0);
    CHECK(fps::to_coefficients(E).size() == 0);
    CHECK(throws<std::domain_error>([&] { (void)(E + A); }));
    Values moved = A;
    Values target = std::move(moved);
    CHECK(moved.size() == 0 && target.size() == 64);
}

// DESIGN.md, Examples: one polynomial in many products, and a sum of products.
void test_many_products() {
    for (const Layout layout : kLayouts) {
        const std::size_t L = 2048;
        const Vec b = random_vec(1000);
        const Values B = fps::to_values(poly(b), L, layout);
        std::vector<Poly> results;
        std::vector<Vec> many;
        for (int i = 0; i < 8; ++i) many.push_back(random_vec(rng() % 1049));
        for (const Vec& a : many) results.push_back(fps::to_coefficients(fps::to_values(poly(a), L, layout) * B));
        for (std::size_t i = 0; i < many.size(); ++i) CHECK(vec(results[i]) == cyclic(many[i], b, L), i);

        const Vec a1 = random_vec(1024), b1 = random_vec(1024), a2 = random_vec(700), b2 = random_vec(1300);
        const Values A1 = values(a1, L, layout), B1 = values(b1, L, layout);
        const Values A2 = values(a2, L, layout), B2 = values(b2, L, layout);
        const Poly s = fps::to_coefficients(A1 * B1 + A2 * B2);
        CHECK(vec(s) == termwise(cyclic(a1, b1, L), cyclic(a2, b2, L), add));
    }
}

// DESIGN.md, Examples: online self-convolution by divide and conquer (ABC222 H's shape). Here
// f = 1 + x f^2, the Catalan numbers: f_0 = 1, f_t = h_(t-1), h = f^2 built online. solve(l, r)
// starts with h[t], t in [l, r), holding every term f_i f_j with max(i, j) < l.
class Online {
public:
    Online(std::size_t n, Layout layout) : f_(n), h_(n), layout_(layout) { solve(0, n); }
    const Vec& f() const { return f_; }

private:
    // Pairs with max(i, j) = t, for t in [l, r), into h[t, r), one t at a time.
    void leaf(std::size_t l, std::size_t r) {
        for (std::size_t t = l; t < r; ++t) {
            f_[t] = t == 0 ? 1 : h_[t - 1];
            for (std::size_t u = t; u < r; ++u) {
                const std::size_t j = u - t;
                if (j < t) h_[u] = add(h_[u], mul(2, mul(f_[t], f_[j])));
                if (j == t) h_[u] = add(h_[u], mul(f_[t], f_[t]));
            }
        }
    }

    // After solve(l, m), the pairs with max(i, j) in [l, m) that land in [m, r). For l = 0 these
    // are f[0, m)^2. For l > 0 the other index is below L = r - l <= l: 2 f[l, m) f[0, L), whose
    // upper half (offset m - l) has no wrapped terms. prefix[L] = the values of 2 f[0, L).
    void solve(std::size_t l, std::size_t r) {
        if (r - l <= 32) return leaf(l, r);
        const std::size_t m = (l + r) / 2, L = r - l;
        solve(l, m);
        auto [slot, fresh] = buffer_.try_emplace(L, L, layout_);
        Values& A = slot->second;
        fps::raw::to_values<P>(std::span(f_).subspan(l, m - l), A);
        if (l == 0) A *= A;
        else A *= prefix(L);
        fps::raw::add_coefficients<P>(A, std::span(h_).subspan(m, r - m), m - l);
        solve(m, r);
    }

    const Values& prefix(std::size_t L) {
        auto it = prefix_.find(L);
        if (it == prefix_.end()) {
            Vec twice(L);
            for (std::size_t i = 0; i < L; ++i) twice[i] = add(f_[i], f_[i]);
            it = prefix_.emplace(L, fps::to_values(poly(twice), L, layout_)).first;
        }
        return it->second;
    }

    Vec f_, h_;
    Layout layout_;
    std::map<std::size_t, Values> buffer_, prefix_;
};

void test_online() {
    const std::size_t n = 4096;
    Vec catalan(n);
    catalan[0] = 1;
    for (std::size_t t = 1; t < n; ++t)
        for (std::size_t i = 0; i < t; ++i) catalan[t] = add(catalan[t], mul(catalan[i], catalan[t - 1 - i]));
    for (const Layout layout : kLayouts) CHECK(Online(n, layout).f() == catalan);
}

u32 evaluate(const Vec& a, u32 z) {
    u32 r = 0;
    for (std::size_t i = a.size(); i-- > 0;) r = add(mul(r, z), a[i]);
    return r;
}

// Long lengths at a point z with z^L = 1, where a b mod (x^L - 1) takes the value a(z) b(z). The
// pointwise layout's longest split (2^23) and the leaf form it keeps above (2^24).
void test_long() {
    const u32 order23 = fps::mod_pow<P>(3, (P - 1) >> 23);  // z^(2^23) = 1
    struct Case {
        int lg;
        std::size_t na, nb;
        Layout layout;
    };
    for (const Case& c : {Case{20, 1 << 19, 1 << 19, Layout::automatic}, Case{20, 1 << 19, 1 << 19, Layout::pointwise},
                          Case{23, 3000, 1 << 23, Layout::pointwise}, Case{24, 3000, 1 << 24, Layout::pointwise}}) {
        const std::size_t L = std::size_t(1) << c.lg;
        const Vec a = random_vec(c.na), b = random_vec(c.nb);
        const Vec ab = vec(fps::to_coefficients(values(a, L, c.layout) * values(b, L, c.layout)));
        CHECK(ab.size() == L);
        for (int t = 0; t < 3; ++t) {
            const u32 z = c.na + c.nb <= L ? u32(rng() % P) : fps::mod_pow<P>(order23, rng());
            CHECK(evaluate(ab, z) == mul(evaluate(a, z), evaluate(b, z)), c.lg, z);
        }
    }
}

// Lengths no other test uses yet, so the threads build their tables at the same time.
void test_threads() {
    std::vector<std::thread> threads;
    std::vector<int> ok(4, 0);
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([t, &ok] {
            std::mt19937_64 local{u64(t)};
            bool good = true;
            for (int lg = 12; lg <= 14; ++lg) {
                const std::size_t L = std::size_t(1) << lg;
                Vec a(L / 2), b(L / 2);
                for (u32& c : a) c = u32(local() % P);
                for (u32& c : b) c = u32(local() % P);
                const Layout layout = kLayouts[(t + lg) % 2];
                const Values A = values(a, L, layout), B = values(b, L, layout);
                const Vec ab = vec(fps::to_coefficients(A * B));
                for (int k = 0; k < 3; ++k) {
                    const u32 z = u32(local() % P);
                    good &= evaluate(ab, z) == mul(evaluate(a, z), evaluate(b, z));
                }
            }
            ok[std::size_t(t)] = good;
        });
    for (std::thread& t : threads) t.join();
    for (const int good : ok) CHECK(good);
}

}  // namespace

int main() {
    test_threads();
    test_conversions();
    test_products();
    test_raw();
    test_errors();
    test_many_products();
    test_online();
    test_long();
    if (failures) std::printf("%d failures\n", failures);
    return failures != 0;
}
