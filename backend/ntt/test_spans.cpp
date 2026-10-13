// Tests for backend/ntt: products, squares and middle products from caller spans (multiply, square,
// middle_product): spans of any alignment, cut and padded outputs, overlap, every variant.
#include "backend/ntt/test_util.hpp"

namespace {

// square(f, out) against `product` (f^2), as test_spans; and out = f.
template <class V>
void test_square(const char* name, const std::vector<u32>& f, const std::vector<u32>& product,
                 std::initializer_list<std::size_t> counts, std::size_t offset) {
    PageWords fw(f.size(), offset, 0);
    std::copy(f.begin(), f.end(), fw.begin());
    for (const std::size_t count : counts) {
        PageWords out(count, offset + 3, 0);
        for (u32& w : out.span()) w = u32(rng());
        square<V>(fw.span(), out.span());
        bool ok = std::ranges::equal(out.span(), cut(product, count)) && std::ranges::equal(fw.span(), f);
        expect(ok, name, f.size() * 1000 + count % 1000);
    }
    std::vector<u32> in_place(f);
    square<V>(in_place, in_place);
    expect(in_place == cut(product, f.size()), name, f.size());
}

// square(f, out): every size to 40, then shapes around each length's levels.
template <class V>
void test_squares(const char* name, int max_lg) {
    for (std::size_t n = 0; n <= 40; ++n) {
        const auto f = random_poly(n, n % 4);
        const std::size_t size = n ? 2 * n - 1 : 0;
        test_square<V>(name, f, schoolbook(f, f), {0, 1, n, size, size + 5}, n % 4);
    }
    for (int lg = kMinLog; lg <= max_lg; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        const std::vector<std::size_t> sizes = lg <= 16 ? std::vector<std::size_t>{len / 2, len / 2 - 1, len / 4 + 1, len / 8 + 3}
                                                        : std::vector<std::size_t>{len / 2, len / 4 + 1};
        for (const std::size_t n : sizes) {
            const auto f = random_poly(n, lg % 2);
            const std::size_t size = 2 * n - 1, offset = lg % 3 == 0 ? 0 : lg % 3 == 1 ? 1 : 16 + lg;
            const auto product = reference_product<V>(name, f, f);
            if (lg <= 16) test_square<V>(name, f, product, {size, n, n / 2 + 1, 9, size + 3}, offset);
            else test_square<V>(name, f, product, {size, n + 7}, offset);
        }
    }
}

// middle_product(f, g, out) for out of each count, against `product` (f g): out[i] = product[m - 1
// + i]. Spans as test_spans; f is cut to the words the call may read (ASan sees any read past
// them). Then out at f and at g.
template <class V>
void test_middle(const char* name, const std::vector<u32>& f, const std::vector<u32>& g,
                 const std::vector<u32>& product, std::initializer_list<std::size_t> counts, std::size_t offset) {
    const std::size_t n = f.size(), m = g.size();
    for (const std::size_t count : counts) {
        const std::size_t read = m - 1 + std::max<std::size_t>(count, 1);
        PageWords fw(read, offset, 0), gw(m, offset + 5, 0), out(count, offset + 3, 0);
        std::copy_n(f.begin(), read, fw.begin());
        std::copy(g.begin(), g.end(), gw.begin());
        for (u32& w : out.span()) w = u32(rng());
        middle_product<V>(fw.span(), gw.span(), out.span());
        bool ok = std::ranges::equal(out.span(), std::span(product).subspan(m - 1, count));
        ok &= std::ranges::equal(gw.span(), g) && std::ranges::equal(fw.span(), std::span(f).first(read));
        expect(ok, name, (n * 100000 + m) * 1000 + count % 1000);
    }
    const std::size_t count = n - m + 1;
    std::vector<u32> both(f);
    both.insert(both.end(), g.begin(), g.end());
    const auto want = std::span(product).subspan(m - 1, count);
    middle_product<V>(std::span(both).first(n), std::span(both).subspan(n, m), std::span(both).first(count));
    expect(std::ranges::equal(std::span(both).first(count), want), name, 1);
    std::copy(f.begin(), f.end(), both.begin());  // out over the end of f and all of g
    middle_product<V>(std::span(both).first(n), std::span(both).subspan(n, m), std::span(both).subspan(n + m - count, count));
    expect(std::ranges::equal(std::span(both).subspan(n + m - count, count), want), name, 2);
}

// middle_product(f, g, out): every n <= 30 and m <= n, then shapes around each length: f of the
// whole length and one word past it, g of 1, half and all of f; the errors.
template <class V>
void test_middles(const char* name, int max_lg) {
    for (std::size_t n = 1; n <= 30; ++n)
        for (std::size_t m = 1; m <= n; ++m) {
            const auto f = random_poly(n, (n + m) % 4), g = random_poly(m, (n + m) % 4);
            const std::size_t all = n - m + 1;
            test_middle<V>(name, f, g, schoolbook(f, g), {0, 1, all, all / 2}, (n + m) % 4);
        }
    for (int lg = kMinLog; lg <= std::min(max_lg, 23); ++lg) {  // the reference product to 2^25
        const std::size_t len = std::size_t(1) << lg;
        using Shapes = std::vector<std::pair<std::size_t, std::size_t>>;
        const Shapes shapes = lg <= 16 ? Shapes{{len, len / 2}, {len, 1}, {len, len}, {len + 1, len / 2}, {len - 1, len / 4 + 3},
                                                {len, len - 1}, {len / 2 + 1, len / 2}}
                                       : Shapes{{len, len / 2}, {len + 1, len / 4}, {len - 3, 5}};
        for (const auto& [n, m] : shapes) {
            const auto f = random_poly(n, lg % 2), g = random_poly(m, lg % 2);
            const auto product = reference_product<V>(name, f, g);
            const std::size_t all = n - m + 1, offset = lg % 3 == 0 ? 0 : lg % 3 == 1 ? 1 : 16 + lg;
            if (lg <= 16) test_middle<V>(name, f, g, product, {all, all / 2 + 1, 1, all - 1 ? all - 1 : 1}, offset);
            else test_middle<V>(name, f, g, product, {all}, offset);
        }
    }
    const std::vector<u32> f(10, 1), g(4, 1);
    std::vector<u32> out(8);
    const auto throws = [&](std::span<const u32> a, std::span<const u32> b, std::span<u32> o) {
        try {
            middle_product<V>(a, b, o);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    expect(throws(f, {}, {}) && throws(g, f, {}) && throws(f, g, out) && !throws(f, g, std::span(out).first(7)),
           name, 0);
}

}  // namespace

int main() {
    for_each_variant([]<class V>(const char* name, int max_lg) {
        test_multiply_spans<V>(name, max_lg);
        test_squares<V>(name, max_lg);
        test_middles<V>(name, max_lg);
    });
    return finish();
}
