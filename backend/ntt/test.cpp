// Tests for backend/ntt. Every variant this machine runs: tables, transforms against the leaf form
// computed directly and against the scalar model, products against schoolbook multiplication and,
// for long ones, evaluation at random points. AVX2: each asm kernel against a scalar model of its
// butterfly, with inputs at the edges of its lazy ranges. Convolution's memory: kept per thread,
// the caller's, several threads.
#include "backend/ntt/ntt.hpp"

#include <array>
#include <cstdio>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using namespace fps::ntt;

constexpr u32 P = kModulus;
constexpr int kScalarMaxLog = 20;  // the scalar variant is slow; ASan makes it slower

std::mt19937_64 rng(2026);
int failures = 0;

void expect(bool ok, const char* what, u64 detail = 0) {
    if (ok) return;
    if (++failures <= 20) std::printf("FAIL: %s (%llu)\n", what, static_cast<unsigned long long>(detail));
}

u32 random_below(u64 bound) { return u32(rng() % bound); }

u32 mul(u32 x, u32 y) { return u32(u64(x) * y % P); }
u32 add(u32 x, u32 y) { return u32((u64(x) + y) % P); }
u32 sub(u32 x, u32 y) { return u32((u64(x) % P + P - y % P) % P); }
u32 power(u32 x, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, x = mul(x, x))
        if (e & 1) r = mul(r, x);
    return r;
}

// count words, 32-byte aligned, then kPadding words: exactly what the API asks for, so ASan sees
// any read past it.
struct Words {
    struct alignas(32) Lanes {
        u32 lane[8];
    };
    explicit Words(std::size_t count) : data((count + kPadding) / 8) {}
    u32* begin() { return data.data()->lane; }
    std::vector<Lanes> data;
};

std::vector<u32> random_poly(std::size_t n, int kind) {
    std::vector<u32> p(n);
    for (auto& x : p) x = kind == 0 ? random_below(P) : kind == 1 ? P - 1 : kind == 2 ? 0 : random_below(10);
    return p;
}

// r(k) from its definition: the product of 3^((P - 1) / 2^(j + 2)) over the set bits j of k.
u32 root(u64 k) {
    u32 r = 1;
    for (int j = 0; k >> j; ++j)
        if (k >> j & 1) r = mul(r, power(3, (P - 1) >> (j + 2)));
    return r;
}

// Tables against the definition: built within the static prefix and past it; the static tables.
template <class V>
void test_tables(const char* name) {
    for (int lg : {kMinLog, kMinLog + 1, 10, detail::kStaticLog, detail::kStaticLog + 2}) {
        const std::size_t count = (std::size_t(1) << lg) / 16;
        std::vector<Words> t;
        for (int inverse = 0; inverse < 2; ++inverse) {
            t.emplace_back(detail::table_words(lg));
            V::build_table(t.back().begin(), count, inverse);
        }
        for (std::size_t k = 0; k < count; ++k) {
            const u32* e = t[0].begin() + detail::slot(k);
            const u32* ie = t[1].begin() + detail::slot(k);
            expect(e[0] == root(k), name, k);
            expect(mul(e[0], ie[0]) == 1, name, k);
            for (const u32* x : {e, ie}) expect(x[8] == u32((u64(x[0]) << 32) / P), name, k);
            if (k < detail::kStaticCount) {
                for (int inverse = 0; inverse < 2; ++inverse)
                    for (int w : {0, 8})
                        expect(detail::kStaticTables.words[inverse][detail::slot(k) + w] == t[inverse].begin()[detail::slot(k) + w],
                               "static tables", k);
            }
        }
    }
}

// f mod (x^8 - w_v) for every leaf v, computed directly: w_v = (-1)^v r(v / 2).
std::vector<u32> leaf_form(const std::vector<u32>& f) {
    std::vector<u32> out(f.size());
    for (std::size_t v = 0; v < f.size() / 8; ++v) {
        const u32 w = v % 2 ? sub(0, root(v / 2)) : root(v / 2);
        u32 wi = 1;
        for (std::size_t i = 0; i < f.size(); i += 8, wi = mul(wi, w))
            for (int l = 0; l < 8; ++l) out[8 * v + l] = add(out[8 * v + l], mul(f[i + l], wi));
    }
    return out;
}

std::vector<u32> run_forward(const Transform& t, const std::vector<u32>& f, auto variant) {
    Words w(f.size());
    std::copy(f.begin(), f.end(), w.begin());
    t.forward<decltype(variant)>(w.begin());
    return {w.begin(), w.begin() + f.size()};
}

std::vector<u32> run_inverse(const Transform& t, const std::vector<u32>& f, auto variant) {
    Words w(f.size());
    std::copy(f.begin(), f.end(), w.begin());
    t.inverse<decltype(variant)>(w.begin());
    return {w.begin(), w.begin() + f.size()};
}

// Forward against the leaf form (small lengths) and the scalar model; inverse undoes it.
template <class V>
void test_transforms(const char* name, int max_lg) {
    for (int lg = kMinLog; lg <= max_lg; ++lg) {
        const Transform t(lg);
        for (int kind = 0; kind < 4; ++kind) {
            if (kind > 1 && lg > 12) break;
            const auto f = random_poly(t.length(), kind);
            const auto leaves = run_forward(t, f, V{});
            if (lg <= 11) expect(leaves == leaf_form(f), name, lg);
            if (lg <= 18) expect(leaves == run_forward(t, f, variant::scalar{}), name, lg);
            expect(run_inverse(t, leaves, V{}) == f, name, lg);
            if (lg <= 18) {
                const auto g = random_poly(t.length(), kind);
                expect(run_inverse(t, g, V{}) == run_inverse(t, g, variant::scalar{}), name, lg);
            }
        }
    }
}

std::vector<u32> schoolbook(const std::vector<u32>& a, const std::vector<u32>& b) {
    if (a.empty() || b.empty()) return {};
    std::vector<u64> c(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = (c[i + j] + u64(a[i]) * b[j]) % P;
    return {c.begin(), c.end()};
}

u32 evaluate(const std::vector<u32>& p, u32 x) {
    u32 r = 0;
    for (std::size_t i = p.size(); i--;) r = add(mul(r, x), p[i]);
    return r;
}

template <class V>
void test_product(const char* name, std::size_t n, std::size_t m, int kind, bool exact) {
    const auto a = random_poly(n, kind), b = random_poly(m, kind);
    Convolution conv(n, m);
    std::copy(a.begin(), a.end(), conv.a());
    std::copy(b.begin(), b.end(), conv.b());
    const u32* p = conv.template multiply<V>();
    const std::vector<u32> c(p, p + conv.size());
    bool ok = c.size() == (n && m ? n + m - 1 : 0);
    for (const u32 x : c) ok &= x < P;
    if (exact) {
        ok &= c == schoolbook(a, b);
    } else {
        for (int i = 0; i < 3; ++i) {
            const u32 x = random_below(P);
            ok &= evaluate(c, x) == mul(evaluate(a, x), evaluate(b, x));
        }
    }
    expect(ok, name, n * 100000000 + m);
}

template <class V>
void test_products(const char* name, int max_lg) {
    for (std::size_t n = 0; n <= 40; ++n)
        for (std::size_t m = 0; m <= 40; m += 3) test_product<V>(name, n, m, 0, true);
    for (int round = 0; round < 60; ++round)
        test_product<V>(name, 1 + random_below(1500), 1 + random_below(1500), round % 4, true);
    // Every transform length: sparse and dense first levels, factors at and around half the length.
    for (int lg = kMinLog; lg <= max_lg; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        const bool exact = lg <= 11;
        test_product<V>(name, len / 2, len / 2, lg % 2, exact);
        test_product<V>(name, len / 2 + 1, len / 2 - 1, 0, exact);
        test_product<V>(name, len - 1, 2, 0, exact);
        test_product<V>(name, len / 2, 2, 0, exact);
        if (lg <= 22) {
            test_product<V>(name, len / 4 + 1, 1, 3, exact);
            test_product<V>(name, len / 4 + 1, len / 4 + 1, 3, exact);
            test_product<V>(name, len / 2 - 5, len / 4 + 7, 0, exact);
        }
    }
}

// Factors of at most half the length with garbage in their upper halves, which must not matter.
template <class V>
void test_upper_halves_unread(const char* name, std::size_t n, std::size_t m) {
    const auto a = random_poly(n, 0), b = random_poly(m, 0);
    Convolution conv(n, m);
    std::copy(a.begin(), a.end(), conv.a());
    std::copy(b.begin(), b.end(), conv.b());
    const std::size_t half = conv.length() / 2;
    for (std::size_t i = half; i < 2 * half; ++i) conv.a()[i] = u32(rng()), conv.b()[i] = u32(rng());
    const u32* c = conv.template multiply<V>();
    expect(std::vector<u32>(c, c + n + m - 1) == schoolbook(a, b), name, n * 100000000 + m);
}

// count words `offset` words past a 4 KiB boundary, then kPadding words that end the allocation,
// so ASan sees a read past them.
class PageWords {
public:
    PageWords(std::size_t count, std::size_t offset)
        : offset_(offset), words_(static_cast<u32*>(::operator new(bytes(count, offset), std::align_val_t(4096)))) {}
    ~PageWords() { ::operator delete(words_, std::align_val_t(4096)); }
    PageWords(const PageWords&) = delete;
    PageWords& operator=(const PageWords&) = delete;
    u32* begin() { return words_ + offset_; }

private:
    static std::size_t bytes(std::size_t count, std::size_t offset) { return (offset + count + kPadding) * sizeof(u32); }
    std::size_t offset_;
    u32* words_;
};

// a in the caller's memory at offsets of 0 to 1008 words in a page, full of garbage: the product
// comes back in it, its padding is not written.
template <class V>
void test_caller_memory(const char* name, std::size_t n, std::size_t m) {
    const auto f = random_poly(n, 0), g = random_poly(m, 0);
    const std::size_t len = Convolution::length(n, m);
    const auto product = schoolbook(f, g);
    for (const std::size_t offset : {0, 16, 32, 48, 512, 1008}) {
        PageWords a(len, offset);
        for (std::size_t i = 0; i < len + kPadding; ++i) a.begin()[i] = u32(rng());
        const std::vector<u32> padding(a.begin() + len, a.begin() + len + kPadding);
        std::copy(f.begin(), f.end(), a.begin());  // before constructing
        Convolution conv(n, m, a.begin());
        std::copy(g.begin(), g.end(), conv.b());
        const u32* c = conv.template multiply<V>();
        bool ok = c == a.begin() && conv.a() == a.begin() && conv.length() == len;
        ok &= std::vector<u32>(c, c + conv.size()) == product;
        ok &= std::vector<u32>(a.begin() + len, a.begin() + len + kPadding) == padding;
        expect(ok, name, offset * 100000000 + n);
    }
}

template <class V>
void test_variant(const char* name, int max_lg) {
    test_tables<V>(name);
    test_transforms<V>(name, max_lg);
    test_products<V>(name, max_lg);
    for (int lg = kMinLog; lg <= 11; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        test_upper_halves_unread<V>(name, len / 2, len / 2);
        test_upper_halves_unread<V>(name, len / 2 - 3, len / 4 + 9);
        test_caller_memory<V>(name, len / 2, len / 2);
        test_caller_memory<V>(name, len - 9, 10);
        test_caller_memory<V>(name, len / 4 + 1, len / 4 + 1);
    }
    test_caller_memory<V>(name, 0, 5);
}

// Products on one thread, inputs from its own generator: the number of wrong ones. Lengths cross
// 2^kStaticLog, so threads build their own tables at the same time.
int thread_products(unsigned seed) {
    std::mt19937_64 local(seed);
    const auto poly = [&](std::size_t n) {
        std::vector<u32> p(n);
        for (u32& x : p) x = u32(local() % P);
        return p;
    };
    int wrong = 0;
    for (int round = 0; round < 24; ++round) {
        const bool exact = round % 3 != 2;
        const std::size_t n = exact ? 1 + local() % 1200 : (std::size_t(1) << 16) + local() % 70000;
        const std::size_t m = 1 + local() % n;
        const auto f = poly(n), g = poly(m);
        Convolution conv(n, m);
        std::copy(f.begin(), f.end(), conv.a());
        std::copy(g.begin(), g.end(), conv.b());
        const u32* c = conv.multiply();
        const std::vector<u32> h(c, c + conv.size());
        if (exact) {
            wrong += h != schoolbook(f, g);
        } else {
            const u32 x = u32(local() % P);
            wrong += evaluate(h, x) != mul(evaluate(f, x), evaluate(g, x));
        }
    }
    return wrong;
}

// The memory kept per thread: two Convolutions alive at once, shorter tables after longer ones,
// threads, and the errors. (Every other product test reuses regions of other sizes.)
void test_memory() {
    {
        const auto f = random_poly(300, 0), g = random_poly(200, 0), u = random_poly(40, 0), v = random_poly(50, 0);
        Convolution outer(300, 200);
        std::copy(f.begin(), f.end(), outer.a());
        std::copy(g.begin(), g.end(), outer.b());
        {
            Convolution inner(40, 50);
            std::copy(u.begin(), u.end(), inner.a());
            std::copy(v.begin(), v.end(), inner.b());
            const u32* c = inner.multiply();
            expect(std::vector<u32>(c, c + 89) == schoolbook(u, v), "memory: inner product");
        }
        const u32* c = outer.multiply();
        expect(std::vector<u32>(c, c + 499) == schoolbook(f, g), "memory: outer product");
    }
    for (const int lg : {19, 17, 18, 17})
        test_product<SelectedMultiply>("memory: tables", std::size_t(1) << (lg - 1), 77, 0, false);
    std::array<int, 4> wrong{};
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < wrong.size(); ++t) threads.emplace_back([t, &wrong] { wrong[t] = thread_products(t); });
    for (auto& thread : threads) thread.join();
    for (const int w : wrong) expect(w == 0, "memory: threads", u64(w));

    bool threw = false;
    try {
        PageWords a(64, 4);  // 16 bytes past a page boundary
        Convolution conv(10, 10, a.begin());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "memory: a not 64-byte aligned");
    threw = false;
    try {
        Convolution conv(std::size_t(1) << kMaxLog, 2);
    } catch (const std::length_error&) {
        threw = true;
    }
    expect(threw, "memory: product too long");
}

// Random values below bound, the largest ones in the first words.
void fill(u32* p, std::size_t count, u64 bound) {
    for (std::size_t i = 0; i < count; ++i) p[i] = random_below(bound);
    for (std::size_t i = 0; i < count && i < 4; ++i) p[i] = u32(bound - 1 - i);
}

// Forward butterfly of one lane: a + x c + y (b + x d), ..., mod P.
std::array<u32, 4> forward_model(const std::array<u32, 4>& f, u32 x, u32 y, u32 z) {
    const u32 A = add(f[0], mul(x, f[2])), C = sub(f[0], mul(x, f[2]));
    const u32 B = add(f[1], mul(x, f[3])), D = sub(f[1], mul(x, f[3]));
    return {add(A, mul(y, B)), sub(A, mul(y, B)), add(C, mul(z, D)), sub(C, mul(z, D))};
}

std::array<u32, 4> inverse_model(const std::array<u32, 4>& f, u32 x, u32 y, u32 z) {
    const u32 ab = add(f[0], f[1]), cd = add(f[2], f[3]);
    const u32 amb = mul(y, sub(f[0], f[1])), cmd = mul(z, sub(f[2], f[3]));
    return {add(ab, cd), add(amb, cmd), mul(x, sub(ab, cd)), mul(x, sub(amb, cmd))};
}

// Twiddle tables for transforms of length 2^lg.
struct Tables {
    explicit Tables(int lg) : forward_(detail::table_words(lg)), inverse_(detail::table_words(lg)) {
        Native::build_table(forward_.begin(), (std::size_t(1) << lg) / 16, false);
        Native::build_table(inverse_.begin(), (std::size_t(1) << lg) / 16, true);
    }
    const u32* r(std::size_t k) { return forward_.begin() + detail::slot(k); }  // value; quotient at [8]
    const u32* ir(std::size_t k) { return inverse_.begin() + detail::slot(k); }
    const u32* r() { return forward_.begin(); }
    const u32* ir() { return inverse_.begin(); }

private:
    Words forward_, inverse_;
};

constexpr int kTableLog = 12;
constexpr std::size_t kGroups = (std::size_t(1) << kTableLog) / 32;  // groups k with r(2k + 1) in the table

// intrinsics<S>: each kernel against the lane models, inputs at the edges of their ranges.
template <class S>
struct IntrinsicsKernels {
    using K = detail::intrinsics::Ntt<S>;
    using Leaf = typename K::Leaf;

    // Runs kernel on count groups of quarters of q words from group first, inputs below in_bound;
    // checks every lane against model(f, group) and outputs below out_bound.
    template <class Kernel, class Model>
    static void check(const std::string& name, std::size_t q, std::size_t count, u64 in_bound, u64 out_bound,
                      Kernel kernel, Model model) {
        Words v(4 * q * count);
        fill(v.begin(), 4 * q * count, in_bound);
        const std::vector<u32> in(v.begin(), v.begin() + 4 * q * count);
        kernel(v.begin());
        for (std::size_t g = 0; g < count; ++g)
            for (std::size_t j = 0; j < q; ++j) {
                std::array<u32, 4> f;
                for (std::size_t t = 0; t < 4; ++t) f[t] = in[4 * q * g + j + q * t];
                const auto expected = model(f, g);
                for (std::size_t t = 0; t < 4; ++t) {
                    const u32 got = v.begin()[4 * q * g + j + q * t];
                    expect(got < out_bound && got % P == expected[t], name.c_str(), j);
                }
            }
    }

    static void run(const std::string& name) {
        Tables t(kTableLog);
        const auto fm = [&](std::size_t k) {
            return [&t, k](const std::array<u32, 4>& f, std::size_t g) {
                return forward_model(f, *t.r(k + g), t.r(2 * (k + g))[0], t.r(2 * (k + g))[1]);
            };
        };
        const auto im = [&](std::size_t k) {
            return [&t, k](const std::array<u32, 4>& f, std::size_t g) {
                return inverse_model(f, *t.ir(k + g), t.ir(2 * (k + g))[0], t.ir(2 * (k + g))[1]);
            };
        };
        for (std::size_t q : {32, 128}) {
            for (int round = 0; round < 8; ++round) {
                const std::size_t k = round == 0 ? 0 : 1 + random_below(kGroups - 1);  // 0: identity groups
                check(name + " forward_group", q, 1, 4 * u64(P), 4 * u64(P),
                      [&](u32* f) { K::forward_group(f, q, k, t.r()); }, fm(k));
                check(name + " inverse_group", q, 1, 2 * u64(P), 2 * u64(P),
                      [&](u32* f) { K::inverse_group(f, q, k, t.ir()); }, im(k));
            }
        }
        for (int round = 0; round < 30; ++round) {
            const std::size_t count = 1 + random_below(4), k = round == 0 ? 0 : random_below(kGroups - count);
            check(name + " forward bottom", 8, count, 4 * u64(P), P,
                  [&](u32* f) { Leaf::forward(f, count, k, t.r()); }, fm(k));
            check(name + " inverse bottom", 8, count, 2 * u64(P), 2 * u64(P),
                  [&](u32* f) { Leaf::inverse(f, count, k, t.ir()); }, im(k));
        }
        test_multiply(name + " leaf products", t, 1, [&](u32* a, u32* b, std::size_t count, std::size_t k) {
            Leaf::multiply(a, b, count, k, t.r(), t.ir());
        });
        if constexpr (S::lanes == 16) {  // pairs of groups on S
            using W = detail::intrinsics::WideBottom<S>;
            for (int round = 0; round < 30; ++round) {
                const std::size_t count = 2 + 2 * random_below(2), k = 2 * random_below(kGroups / 2 - 2);
                check(name + " wide forward bottom", 8, count, 4 * u64(P), P,
                      [&](u32* f) { W::forward(f, count, k, t.r()); }, fm(k));
                check(name + " wide inverse bottom", 8, count, 2 * u64(P), 2 * u64(P),
                      [&](u32* f) { W::inverse(f, count, k, t.ir()); }, im(k));
            }
            test_multiply(name + " wide leaf products", t, 2, [&](u32* a, u32* b, std::size_t count, std::size_t k) {
                W::multiply(a, b, count, k, t.r(), t.ir());
            });
        }
    }

    // run(a, b, count, first): Leaf::multiply or a variant of it on groups of 32 words, count and
    // first multiples of step: forward butterfly, products of the leaves mod x^8 - w (with the
    // Montgomery factor 2^-32), inverse butterfly.
    template <class Run>
    static void test_multiply(const std::string& name, Tables& t, std::size_t step, Run run) {
        const u32 r_inverse = power(u32((u64(1) << 32) % P), P - 2);
        for (int round = 0; round < 50; ++round) {
            const std::size_t count = step * (1 + random_below(4 / step)), k0 = step * random_below(kGroups / step - 4);
            Words a(32 * count), b(32 * count);
            fill(a.begin(), 32 * count, 4 * u64(P));
            fill(b.begin(), 32 * count, 4 * u64(P));
            const std::vector<u32> a0(a.begin(), a.begin() + 32 * count), b0(b.begin(), b.begin() + 32 * count);
            run(a.begin(), b.begin(), count, k0);
            for (std::size_t j = 0; j < count; ++j) {
                const std::size_t k = k0 + j;
                const u32 x = *t.r(k), y = t.r(2 * k)[0], z = t.r(2 * k)[1];
                const std::array<u32, 4> w{y, P - y, z, P - z};
                u32 fa[4][8], fb[4][8], c[4][8];
                for (int l = 0; l < 8; ++l) {
                    std::array<u32, 4> xa, xb;
                    for (int s = 0; s < 4; ++s) xa[s] = a0[32 * j + 8 * s + l], xb[s] = b0[32 * j + 8 * s + l];
                    const auto ya = forward_model(xa, x, y, z), yb = forward_model(xb, x, y, z);
                    for (int s = 0; s < 4; ++s) fa[s][l] = ya[s], fb[s][l] = yb[s];
                }
                for (int s = 0; s < 4; ++s)
                    for (int l = 0; l < 8; ++l) {
                        u32 sum = 0;
                        for (int i = 0; i < 8; ++i)
                            sum = add(sum, mul(fb[s][i], l >= i ? fa[s][l - i] : mul(w[s], fa[s][l - i + 8])));
                        c[s][l] = mul(sum, r_inverse);
                    }
                for (int l = 0; l < 8; ++l) {
                    const auto expected = inverse_model({c[0][l], c[1][l], c[2][l], c[3][l]}, *t.ir(k),
                                                        t.ir(2 * k)[0], t.ir(2 * k)[1]);
                    for (int s = 0; s < 4; ++s) {
                        const u32 got = a.begin()[32 * j + 8 * s + l];
                        expect(got < 2 * P && got % P == expected[s], name.c_str(), 32 * j + 8 * s + l);
                    }
                }
            }
        }
    }
};

// Every simd struct this machine runs; Scalar is slow, so shorter.
void test_intrinsics() {
    fps::simd::for_each(fps::simd::All{}, []<class S> {
        const std::string name = std::string("intrinsics ") + S::name;
        IntrinsicsKernels<S>::run(name);
        test_variant<variant::intrinsics<S>>(name.c_str(), S::lanes == 1 ? kScalarMaxLog : kMaxLog);
    });
}

#ifdef __AVX2__

namespace impl = detail::avx2;
using Vec = impl::Vec;

// Variant avx2 with each bottom stage at every length.
template <class Bottom>
struct avx2_bottom : variant::avx2 {
    static void multiply(u32* a, u32* b, std::size_t n, std::size_t m, int lg, const u32* r, const u32* ir) {
        const std::size_t len = std::size_t(1) << lg;
        impl::product<Bottom>(a, b, 2 * n <= len, 2 * m <= len, lg, r, ir);
    }
};
using avx2_group = avx2_bottom<impl::GroupBottom>;
using avx2_pair = avx2_bottom<impl::PairBottom>;

void test_bottoms() {
    for (int lg = kMinLog; lg <= kMaxLog; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        const bool exact = lg <= 11;
        for (const auto& [n, m] : {std::pair{len / 2, len / 2}, {len / 2, std::size_t(2)}, {len / 4 + 1, len / 4 + 1},
                                   {len - len / 4, len / 4 + 1}, {len / 4 + 1, len - 1 - len / 4}}) {
            test_product<avx2_group>("avx2 one-group bottom", n, m, 0, exact);
            test_product<avx2_pair>("avx2 two-group bottom", n, m, 0, exact);
        }
    }
}

Vec* vec(Words& w) { return reinterpret_cast<Vec*>(w.begin()); }

// Runs kernel on 4h vectors with inputs below in_bound; checks every lane against model and
// outputs below out_bound.
template <class Kernel, class Model>
void test_loop(const char* name, std::size_t h, u64 in_bound, u64 out_bound, Kernel kernel, Model model) {
    Words v(32 * h);
    fill(v.begin(), 32 * h, in_bound);
    const std::vector<u32> in(v.begin(), v.begin() + 32 * h);
    kernel(vec(v), h);
    for (std::size_t j = 0; j < 8 * h; ++j) {
        std::array<u32, 4> f;
        for (std::size_t t = 0; t < 4; ++t) f[t] = in[j + 8 * h * t];
        const auto expected = model(f);
        for (std::size_t t = 0; t < 4; ++t) {
            const u32 got = v.begin()[j + 8 * h * t];
            expect(got < out_bound && got % P == expected[t], name, j);
        }
    }
}

void test_loops() {
    namespace k = impl::kernels;
    Tables t(kTableLog);
    for (std::size_t h : {2, 4, 16}) {
        for (int round = 0; round < 8; ++round) {
            const std::size_t g = 1 + random_below(kGroups - 1);
            const u32 *x = t.r(g), *y = t.r(2 * g), *ix = t.ir(g), *iy = t.ir(2 * g);
            test_loop("forward", h, 4 * u64(P), 4 * u64(P), [&](Vec* f, std::size_t n) { k::forward(f, n, x, y); },
                      [&](const std::array<u32, 4>& f) { return forward_model(f, *x, y[0], y[1]); });
            test_loop("inverse", h, 2 * u64(P), 2 * u64(P), [&](Vec* f, std::size_t n) { k::inverse(f, n, ix, iy); },
                      [&](const std::array<u32, 4>& f) { return inverse_model(f, *ix, iy[0], iy[1]); });
            test_loop("forward_identity", h, 4 * u64(P), 4 * u64(P),
                      [&](Vec* f, std::size_t n) { k::forward_identity(f, n, t.r()); },
                      [&](const std::array<u32, 4>& f) { return forward_model(f, 1, 1, *t.r(1)); });
            test_loop("forward_top", h, P, 4 * u64(P), [&](Vec* f, std::size_t n) { k::forward_top(f, n, t.r()); },
                      [&](const std::array<u32, 4>& f) { return forward_model(f, 1, 1, *t.r(1)); });
            test_loop("inverse_identity", h, 2 * u64(P), 2 * u64(P),
                      [&](Vec* f, std::size_t n) { k::inverse_identity(f, n, t.ir()); },
                      [&](const std::array<u32, 4>& f) { return inverse_model(f, 1, 1, *t.ir(1)); });
            // forward_pair: forward() on two arrays.
            Words a(32 * h), b(32 * h);
            fill(a.begin(), 32 * h, 4 * u64(P));
            fill(b.begin(), 32 * h, 4 * u64(P));
            Words a1 = a, b1 = b;
            k::forward_pair(vec(a), vec(b), h, x, y);
            k::forward(vec(a1), h, x, y);
            k::forward(vec(b1), h, x, y);
            for (std::size_t i = 0; i < 32 * h; ++i)
                expect(a.begin()[i] == a1.begin()[i] && b.begin()[i] == b1.begin()[i], "forward_pair", i);
        }
        // scale_radix2: s (f0 + f1), s (f0 - f1), canonical.
        Words v(16 * h);
        fill(v.begin(), 16 * h, 2 * u64(P));
        const std::vector<u32> in(v.begin(), v.begin() + 16 * h);
        const u32 s = random_below(P);
        impl::scale_radix2(vec(v), h, s);
        for (std::size_t j = 0; j < 8 * h; ++j) {
            expect(v.begin()[j] == mul(s, add(in[j], in[j + 8 * h])), "scale_radix2 sum", j);
            expect(v.begin()[j + 8 * h] == mul(s, sub(in[j], in[j + 8 * h])), "scale_radix2 difference", j);
        }
        // inverse_radix4: s inverse_identity(f), canonical.
        Words w(32 * h);
        fill(w.begin(), 32 * h, 2 * u64(P));
        const std::vector<u32> top(w.begin(), w.begin() + 32 * h);
        impl::inverse_radix4(vec(w), h, t.ir(), s);
        for (std::size_t j = 0; j < 8 * h; ++j) {
            std::array<u32, 4> f;
            for (std::size_t q = 0; q < 4; ++q) f[q] = top[j + 8 * h * q];
            const auto expected = inverse_model(f, 1, 1, *t.ir(1));
            for (std::size_t q = 0; q < 4; ++q) expect(w.begin()[j + 8 * h * q] == mul(s, expected[q]), "inverse_radix4", j);
        }
    }
}

// forward_bottom and inverse_bottom: groups of 4 vectors (h = 1), each with its own twiddles.
void test_bottom_levels() {
    Tables t(kTableLog);
    for (int round = 0; round < 50; ++round) {
        const std::size_t count = 2 + 2 * random_below(2), first = 2 * random_below(kGroups / 2 - 2);
        for (int inverse = 0; inverse < 2; ++inverse) {
            const u64 bound = inverse ? u64(P) : 4 * u64(P);  // inverse: leaf form in
            Words v(32 * count);
            fill(v.begin(), 32 * count, bound);
            const std::vector<u32> in(v.begin(), v.begin() + 32 * count);
            if (inverse) impl::inverse_bottom(vec(v), count, first, t.ir());
            else impl::forward_bottom(vec(v), count, first, t.r());
            for (std::size_t g = 0; g < count; ++g) {
                const std::size_t k = first + g;
                for (int l = 0; l < 8; ++l) {
                    std::array<u32, 4> f;
                    for (int s = 0; s < 4; ++s) f[s] = in[32 * g + 8 * s + l];
                    const auto expected = inverse ? inverse_model(f, *t.ir(k), t.ir(2 * k)[0], t.ir(2 * k)[1])
                                                  : forward_model(f, *t.r(k), t.r(2 * k)[0], t.r(2 * k)[1]);
                    for (int s = 0; s < 4; ++s) {
                        const u32 got = v.begin()[32 * g + 8 * s + l];
                        expect(got < (inverse ? 2 * P : P) && got % P == expected[s],
                               inverse ? "inverse_bottom" : "forward_bottom", k);
                    }
                }
            }
        }
    }
}

// The product bottom stage (GroupBottom) on batches of four vectors: forward butterfly with
// h = 1, products of the leaves mod x^8 - w (with the Montgomery factor 2^-32), inverse butterfly.
void test_group_bottom() {
    Tables t(kTableLog);
    const u32 r_inverse = power(u32((u64(1) << 32) % P), P - 2);
    for (int round = 0; round < 50; ++round) {
        const std::size_t batches = 1 + random_below(4), k0 = random_below(kGroups - batches);
        Words a(32 * batches), b(32 * batches);
        fill(a.begin(), 32 * batches, 4 * u64(P));
        fill(b.begin(), 32 * batches, 4 * u64(P));
        const std::vector<u32> a0(a.begin(), a.begin() + 32 * batches), b0(b.begin(), b.begin() + 32 * batches);
        impl::GroupBottom::run(vec(a), vec(b), 4 * batches, k0, t.r(), t.ir());
        for (std::size_t j = 0; j < batches; ++j) {
            const std::size_t k = k0 + j;
            const u32 x = *t.r(k), y = t.r(2 * k)[0], z = t.r(2 * k)[1];
            const std::array<u32, 4> w{y, P - y, z, P - z};
            u32 fa[4][8], fb[4][8], c[4][8];
            for (int l = 0; l < 8; ++l) {
                std::array<u32, 4> xa, xb;
                for (int s = 0; s < 4; ++s) xa[s] = a0[32 * j + 8 * s + l], xb[s] = b0[32 * j + 8 * s + l];
                const auto ya = forward_model(xa, x, y, z), yb = forward_model(xb, x, y, z);
                for (int s = 0; s < 4; ++s) fa[s][l] = ya[s], fb[s][l] = yb[s];
            }
            for (int s = 0; s < 4; ++s)
                for (int l = 0; l < 8; ++l) {
                    u32 sum = 0;
                    for (int i = 0; i < 8; ++i)
                        sum = add(sum, mul(fb[s][i], l >= i ? fa[s][l - i] : mul(w[s], fa[s][l - i + 8])));
                    c[s][l] = mul(sum, r_inverse);
                }
            for (int l = 0; l < 8; ++l) {
                const auto expected = inverse_model({c[0][l], c[1][l], c[2][l], c[3][l]}, *t.ir(k), t.ir(2 * k)[0],
                                                    t.ir(2 * k)[1]);
                for (int s = 0; s < 4; ++s) {
                    const u32 got = a.begin()[32 * j + 8 * s + l];
                    expect(got < 2 * P && got % P == expected[s], "group bottom", 32 * j + 8 * s + l);
                }
            }
        }
    }
}

void test_avx2() {
    test_loops();
    test_bottom_levels();
    test_group_bottom();
    test_bottoms();
    test_variant<variant::avx2>("avx2", kMaxLog);
}

#endif

}  // namespace

int main() {
    test_memory();
    test_variant<variant::scalar>("scalar", kScalarMaxLog);
    test_intrinsics();
#ifdef __AVX2__
    test_avx2();
#endif
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
