// Tests for backend/ntt: Transform's leaf-form ops (issue #49): leaf products, the values form,
// inverse(F, out, ...) into other memory, every variant.
#include "backend/ntt/test_util.hpp"

namespace {

// A random leaf form: canonical words, a fraction of them at the edges 0 and P - 1.
std::vector<u32> random_words(std::size_t n) {
    std::vector<u32> w(n);
    for (u32& x : w) {
        const u64 kind = rng() % 8;
        x = kind == 0 ? 0 : kind == 1 ? P - 1 : random_below(P);
    }
    return w;
}

// The values form computed directly from the leaf form: leaf v at rho_v z^bitrev3(i), rho_v =
// r(4v), z = r(2) (common.hpp), to word 64 (v / 8) + 8 i + v % 8.
std::vector<u32> values_form(const std::vector<u32>& leaves) {
    std::vector<u32> out(leaves.size());
    const u32 z = root(2);
    for (std::size_t v = 0; v < leaves.size() / 8; ++v) {
        const std::vector<u32> leaf(leaves.begin() + 8 * v, leaves.begin() + 8 * v + 8);
        for (int i = 0; i < 8; ++i) {
            const int j = (i & 1) << 2 | (i & 2) | (i >> 2 & 1);
            out[64 * (v / 8) + 8 * i + v % 8] = evaluate(leaf, mul(root(4 * v), power(z, j)));
        }
    }
    return out;
}

// Leaf products against the scalar model, with out and acc on a or b; values form against its
// definition (short lengths) and the scalar model, and back; inverse(F, out, ...) into memory of
// any alignment, with offsets, add, and F kept.
template <class V>
void test_leaf_ops(const char* name, int max_lg) {
    for (int lg = kMinLog; lg <= max_lg; ++lg) {
        const Transform t(lg);
        const std::size_t len = t.length();
        const auto a = random_words(len), b = random_words(len), acc = random_words(len);
        Words x(len), y(len), z(len);
        const auto load = [](Words& w, const std::vector<u32>& v) { std::copy(v.begin(), v.end(), w.begin()); };
        const auto words = [len](Words& w) { return std::vector<u32>(w.begin(), w.begin() + len); };
        // Leaf products against the model: out = a b into z, into a, into b; acc += a b into z, a, b.
        Words ref(len);
        {
            load(x, a), load(y, b), load(ref, acc);
            t.template multiply<variant::scalar>(x.begin(), y.begin(), z.begin());
            const auto model = words(z);
            t.template multiply_add<variant::scalar>(ref.begin(), x.begin(), y.begin());
            const auto model_add = words(ref);
            load(z, acc);
            t.template multiply<V>(x.begin(), y.begin(), z.begin());
            expect(words(z) == model && words(x) == a && words(y) == b, name, 1000 + lg);
            t.template multiply<V>(x.begin(), y.begin(), x.begin());
            expect(words(x) == model && words(y) == b, name, 2000 + lg);
            load(x, a);
            t.template multiply<V>(x.begin(), y.begin(), y.begin());
            expect(words(y) == model && words(x) == a, name, 3000 + lg);
            load(y, b), load(z, acc);
            t.template multiply_add<V>(z.begin(), x.begin(), y.begin());
            expect(words(z) == model_add && words(x) == a && words(y) == b, name, 4000 + lg);
            load(x, acc);  // acc = a: x += x y
            t.template multiply_add<V>(x.begin(), x.begin(), y.begin());
            Words m(len);
            load(m, acc), load(ref, acc);
            t.template multiply_add<variant::scalar>(m.begin(), ref.begin(), y.begin());
            expect(words(x) == words(m), name, 5000 + lg);
            load(x, a), load(y, acc);  // acc = b: y += x y
            t.template multiply_add<V>(y.begin(), x.begin(), y.begin());
            load(m, acc), load(ref, acc);
            t.template multiply_add<variant::scalar>(m.begin(), x.begin(), ref.begin());
            expect(words(y) == words(m), name, 6000 + lg);
        }
        // Leaf products are products mod x^L - 1 (cyclic) through the transforms.
        if (lg <= 10) {
            const auto f = random_poly(len, 0), g = random_poly(len, 0);
            load(x, f), load(y, g);
            t.forward(x.begin()), t.forward(y.begin());
            t.template multiply<V>(x.begin(), y.begin(), x.begin());
            t.inverse(x.begin());
            std::vector<u32> cyclic(len);
            const auto full = schoolbook(f, g);
            for (std::size_t i = 0; i < full.size(); ++i) cyclic[i % len] = add(cyclic[i % len], full[i]);
            expect(words(x) == cyclic, name, 7000 + lg);
        }
        // The values form.
        if (lg <= kValuesMaxLog) {
            const auto f = random_words(len);
            load(x, f);
            t.template forward<variant::scalar>(x.begin());
            const auto leaves = words(x);
            load(x, f);
            t.template forward<V>(x.begin(), Form::values);
            const auto values = words(x);
            if (lg <= 10) expect(values == values_form(leaves), name, 8000 + lg);
            if (lg <= 18 || std::is_same_v<V, variant::scalar>) {
                load(y, f);
                t.template forward<variant::scalar>(y.begin(), Form::values);
                expect(values == words(y), name, 9000 + lg);
            }
            t.template inverse<V>(x.begin(), Form::values);
            expect(words(x) == f, name, 10000 + lg);
            // Canonical words in, any of them: the inverse against the model's.
            const auto g = random_words(len);
            load(x, g), load(y, g);
            t.template inverse<V>(x.begin(), Form::values);
            if (lg <= 18) {
                t.template inverse<variant::scalar>(y.begin(), Form::values);
                expect(words(x) == words(y), name, 11000 + lg);
            }
        }
        // Out of place, both forms: counts, offsets, add; out at word offsets in a page, F kept.
        for (const Form form : {Form::leaves, Form::values}) {
            if (form == Form::values && lg > kValuesMaxLog) continue;
            const auto f = random_poly(len, lg % 2);
            load(x, f);
            t.template forward<V>(x.begin(), form);
            const auto F = words(x);
            using Case = std::tuple<std::size_t, std::size_t, bool>;  // count, offset, add
            std::vector<Case> cases = {{len, 0, false}, {len - 1, 0, false}, {len / 2 + 1, 0, false}, {1, 0, false},
                                       {0, 0, false}, {len / 2, len / 2, true}, {len / 4 + 3, len / 2 - 5, false},
                                       {7, len - 7, true}, {len, 0, true}};
            if (lg <= 12)
                for (std::size_t count = 1; count <= 40; count += 3) cases.emplace_back(count, 0, false);
            for (const auto& [count, offset, add_to] : cases) {
                PageWords out(count, (count + offset + lg) % 37, 0);
                std::vector<u32> before(count);
                for (u32& w : before) w = random_below(P);
                std::copy(before.begin(), before.end(), out.begin());
                t.template inverse<V>(x.begin(), out.span(), form, {offset, add_to});
                bool ok = words(x) == F;
                for (std::size_t i = 0; i < count; ++i) ok &= out.begin()[i] == (add_to ? add(before[i], f[offset + i]) : f[offset + i]);
                expect(ok, name, (12 + int(form)) * 1000000 + lg * 1000 + count % 1000);
            }
        }
    }
    const Transform t(kMinLog);
    std::vector<u32> out(70);
    bool threw = false;
    try {
        t.inverse(out.data(), std::span(out).first(10), Form::leaves, {60, false});
    } catch (const std::out_of_range&) {
        threw = true;
    }
    expect(threw, name, 99);
    if (kMaxLog > kValuesMaxLog) {
        threw = false;
        try {
            const Transform u(kValuesMaxLog + 1);
            u.forward(out.data(), Form::values);
        } catch (const std::length_error&) {
            threw = true;
        }
        expect(threw, name, 98);
    }
}

}  // namespace

int main() {
    for_each_variant([]<class V>(const char* name, int max_lg) {
        test_leaf_ops<V>(name, max_lg);
    });
    return finish();
}
