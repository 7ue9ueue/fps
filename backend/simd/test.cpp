// Tests every simd struct this machine can run against lane-by-lane scalar arithmetic.
// Random vectors plus edge words (0, 1, 2^31, 2^32 - 1). Exits nonzero on a failure.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#include "backend/simd/select.hpp"

namespace {

using fps::simd::u32;
using fps::simd::u64;

std::mt19937_64 rng(20261010);
int failures = 0;

void fail(const char* isa, const std::string& what) {
    if (++failures <= 20) std::printf("FAIL %s: %s\n", isa, what.c_str());
}

u32 word() {
    static constexpr u32 kEdges[] = {0, 1, 2, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff};
    const u64 r = rng();
    return r % 8 == 0 ? kEdges[(r >> 3) % std::size(kEdges)] : u32(r >> 32);
}

template <class S>
using Lanes = std::array<u32, S::lanes>;

template <class S>
Lanes<S> lanes_of(typename S::V v) {
    Lanes<S> out;
    S::store(out.data(), v);
    return out;
}

template <class S>
Lanes<S> random_lanes() {
    Lanes<S> out;
    for (u32& x : out) x = word();
    return out;
}

template <class S>
typename S::V vec(const Lanes<S>& x) { return S::load(x.data()); }

[[maybe_unused]] u64 pair(const u32* x, int i) { return u64(x[2 * i]) | u64(x[2 * i + 1]) << 32; }

// Checks out lane by lane against expect(i).
template <class S, class Expect>
void check(const char* what, typename S::V got, Expect expect) {
    const Lanes<S> out = lanes_of<S>(got);
    for (int i = 0; i < S::lanes; ++i) {
        if (out[i] != expect(i)) {
            fail(S::name, std::string(what) + " lane " + std::to_string(i) + ": got " + std::to_string(out[i]) +
                              ", want " + std::to_string(expect(i)));
            return;
        }
    }
}

// Checks 64-bit lanes against expect(i).
template <class S, class Expect>
void check64(const char* what, typename S::V got, Expect expect) {
    const Lanes<S> out = lanes_of<S>(got);
    for (int i = 0; i < S::lanes / 2; ++i) {
        if (pair(out.data(), i) != expect(i)) {
            fail(S::name, std::string(what) + " 64-bit lane " + std::to_string(i));
            return;
        }
    }
}

template <class S, int k>
void test_mul_shr(const Lanes<S>& a, Lanes<S> b) {
    for (int i = 0; i < S::lanes; ++i) {  // (a b) >> k must fit in 32 bits
        const u64 limit = k == 32 ? ~u64(0) : (u64(1) << (32 + k % 32)) - 1;
        if (a[i] && u64(a[i]) * b[i] > limit) b[i] = u32(std::min<u64>(limit / a[i], 0xffffffff));
    }
    check<S>(("mul_shr<" + std::to_string(k) + ">").c_str(), S::template mul_shr<k>(vec<S>(a), vec<S>(b)),
             [&](int i) { return u32(u64(a[i]) * b[i] >> k); });
}

template <class S, unsigned mask>
void test_blend(const Lanes<S>& a, const Lanes<S>& b) {
    check<S>("blend", S::template blend<mask>(vec<S>(a), vec<S>(b)),
             [&](int i) { return mask >> i & 1 ? b[i] : a[i]; });
}

template <class S>
void test_common() {
    const Lanes<S> a = random_lanes<S>(), b = random_lanes<S>(), c = random_lanes<S>();
    const auto va = vec<S>(a), vb = vec<S>(b), vc = vec<S>(c);

    u32 buffer[3 * S::lanes];
    for (int offset = 0; offset < S::lanes; ++offset) {  // unaligned loads and stores
        for (u32& x : buffer) x = word();
        const auto v = S::load(buffer + offset);
        u32 copy[3 * S::lanes] = {};
        S::store(copy + offset, v);
        for (int i = 0; i < S::lanes; ++i)
            if (copy[offset + i] != buffer[offset + i]) fail(S::name, "load/store at offset " + std::to_string(offset));
    }
    for (int k = 0; k <= S::lanes; ++k) {  // the first k lanes, nothing past them
        u32 out[3 * S::lanes];
        for (u32& x : out) x = 0xdeadbeef;
        S::store_first(out + 1, va, k);
        for (int i = 0; i < 3 * S::lanes; ++i)
            if (out[i] != (i >= 1 && i <= k ? a[i - 1] : 0xdeadbeef)) fail(S::name, "store_first " + std::to_string(k));
    }
    check<S>("broadcast", S::broadcast(a[0]), [&](int) { return a[0]; });
    check<S>("add", S::add(va, vb), [&](int i) { return a[i] + b[i]; });
    check<S>("sub", S::sub(va, vb), [&](int i) { return a[i] - b[i]; });
    check<S>("min", S::min(va, vb), [&](int i) { return std::min(a[i], b[i]); });
    check<S>("mul_lo", S::mul_lo(va, vb), [&](int i) { return a[i] * b[i]; });
    check<S>("mul_sub", S::mul_sub(vc, va, vb), [&](int i) { return c[i] - a[i] * b[i]; });
    check<S>("shr<1>", S::template shr<1>(va), [&](int i) { return a[i] >> 1; });
    check<S>("shr<13>", S::template shr<13>(va), [&](int i) { return a[i] >> 13; });
    check<S>("shr<31>", S::template shr<31>(va), [&](int i) { return a[i] >> 31; });
    check<S>("shl<1>", S::template shl<1>(va), [&](int i) { return a[i] << 1; });
    check<S>("shl<13>", S::template shl<13>(va), [&](int i) { return a[i] << 13; });
    check<S>("shl<31>", S::template shl<31>(va), [&](int i) { return a[i] << 31; });
    check<S>("mulhi", S::mulhi(va, vb), [&](int i) { return u32(u64(a[i]) * b[i] >> 32); });
    test_mul_shr<S, 1>(a, b);
    test_mul_shr<S, 7>(a, b);
    test_mul_shr<S, 16>(a, b);
    test_mul_shr<S, 29>(a, b);
    test_mul_shr<S, 31>(a, b);
    test_mul_shr<S, 32>(a, b);
    test_blend<S, 0>(a, b);
    test_blend<S, 0xffff>(a, b);
    test_blend<S, 0xaaaa>(a, b);
    test_blend<S, 0x5555>(a, b);
    test_blend<S, 0x1>(a, b);
    test_blend<S, 0x8421>(a, b);

    std::array<int, S::lanes> from;
    for (int& f : from) f = int(rng() % S::lanes);
    check<S>("permute", S::permute(va, S::index(from)), [&](int i) { return a[from[i]]; });
}

template <class S>
void test_x86() {
    const Lanes<S> a = random_lanes<S>(), b = random_lanes<S>();
    const auto va = vec<S>(a), vb = vec<S>(b);
    check64<S>("mul_even", S::mul_even(va, vb), [&](int i) { return u64(a[2 * i]) * b[2 * i]; });
    check64<S>("add64", S::add64(va, vb), [&](int i) { return pair(a.data(), i) + pair(b.data(), i); });
    check64<S>("sub64", S::sub64(va, vb), [&](int i) { return pair(a.data(), i) - pair(b.data(), i); });
    check64<S>("shr64<1>", S::template shr64<1>(va), [&](int i) { return pair(a.data(), i) >> 1; });
    check64<S>("shr64<32>", S::template shr64<32>(va), [&](int i) { return pair(a.data(), i) >> 32; });
    check64<S>("shr64<63>", S::template shr64<63>(va), [&](int i) { return pair(a.data(), i) >> 63; });
    check64<S>("shl64<1>", S::template shl64<1>(va), [&](int i) { return pair(a.data(), i) << 1; });
    check64<S>("shl64<32>", S::template shl64<32>(va), [&](int i) { return pair(a.data(), i) << 32; });
    check64<S>("shl64<63>", S::template shl64<63>(va), [&](int i) { return pair(a.data(), i) << 63; });
    check<S>("join_high", S::join_high(va, vb), [&](int i) { return i % 2 ? b[i] : a[i + 1]; });
}

// load_mask: every lane pattern of a window at every offset of a buffer; masked-out lanes read 0.
template <class S>
void test_load_mask() {
    u32 buffer[3 * S::lanes];
    for (u32& x : buffer) x = word();
    for (int offset = 0; offset <= 2 * S::lanes; ++offset)
        for (unsigned mask : {0u, 1u, (1u << S::lanes) - 1, (1u << S::lanes) - 2, unsigned(word())}) {
            mask &= (1u << S::lanes) - 1;
            check<S>("load_mask", S::load_mask(buffer + offset, mask),
                     [&](int i) { return mask >> i & 1 ? buffer[offset + i] : 0u; });
        }
}

template <class S>
void test_permute2() {
    const Lanes<S> a = random_lanes<S>(), b = random_lanes<S>();
    Lanes<S> idx;
    for (u32& x : idx) x = u32(rng() % (2 * S::lanes));
    check<S>("permute2", S::permute2(vec<S>(a), vec<S>(idx), vec<S>(b)),
             [&](int i) { return idx[i] < S::lanes ? a[idx[i]] : b[idx[i] - S::lanes]; });
}

template <class S>
void test_halves() {
    const Lanes<S> a = random_lanes<S>(), b = random_lanes<S>();
    check<S>("load_halves", S::load_halves(a.data(), b.data() + 8), [&](int i) { return i < 8 ? a[i] : b[i]; });
    Lanes<S> out = b;
    S::template store_half<0>(out.data() + 3, vec<S>(a));
    S::template store_half<1>(out.data(), vec<S>(a));  // overwrites words 3 .. 7 again
    for (int i = 0; i < S::lanes; ++i)
        if (out[i] != (i < 8 ? a[8 + i] : i < 11 ? a[i - 3] : b[i])) fail(S::name, "store_half lane " + std::to_string(i));
    check<S>("broadcast_halves", S::broadcast_halves(&a[3], &b[5]), [&](int i) { return i < 8 ? a[3] : b[5]; });
}

template <class S>
void test_ifma() {
    const Lanes<S> a = random_lanes<S>(), b = random_lanes<S>(), c = random_lanes<S>();
    const auto va = vec<S>(a), vb = vec<S>(b), vc = vec<S>(c);
    constexpr u64 kMask = (u64(1) << 52) - 1;
    const auto product = [&](int i) {
        return (unsigned __int128)(pair(a.data(), i) & kMask) * (pair(b.data(), i) & kMask);
    };
    check64<S>("madd52lo", S::madd52lo(vc, va, vb), [&](int i) { return pair(c.data(), i) + (u64(product(i)) & kMask); });
    check64<S>("madd52hi", S::madd52hi(vc, va, vb), [&](int i) { return pair(c.data(), i) + u64(product(i) >> 52); });
}

template <class S>
void test_neon() {
    const Lanes<S> a = random_lanes<S>(), b = random_lanes<S>(), c = random_lanes<S>(), d = random_lanes<S>();
    const auto va = vec<S>(a), vb = vec<S>(b), vc = vec<S>(c), vd = vec<S>(d);
    const auto wide = [](typename S::W w) {
        std::array<u64, 2> out;
        std::memcpy(out.data(), &w, sizeof w);
        return out;
    };
    const auto cw = S::mul_wide_lo(vc, vd);  // any 64-bit accumulator
    for (int half = 0; half < 2; ++half) {
        const auto p = wide(half ? S::mul_wide_hi(va, vb) : S::mul_wide_lo(va, vb));
        const auto q = wide(half ? S::mla_wide_hi(cw, va, vb) : S::mla_wide_lo(cw, va, vb));
        const auto r = wide(half ? S::mls_wide_hi(cw, va, vb) : S::mls_wide_lo(cw, va, vb));
        for (int i = 0; i < 2; ++i) {
            const u64 product = u64(a[2 * half + i]) * b[2 * half + i];
            if (p[i] != product) fail(S::name, "mul_wide");
            if (q[i] != wide(cw)[i] + product) fail(S::name, "mla_wide");
            if (r[i] != wide(cw)[i] - product) fail(S::name, "mls_wide");
        }
    }
    const auto x = S::mul_wide_lo(va, vb), y = S::mul_wide_hi(vc, vd);
    check<S>("lo_halves", S::lo_halves(x, y), [&](int i) { return u32(i < 2 ? wide(x)[i] : wide(y)[i - 2]); });
    check<S>("hi_halves", S::hi_halves(x, y), [&](int i) { return u32((i < 2 ? wide(x)[i] : wide(y)[i - 2]) >> 32); });
    check<S>("qdmulh", S::qdmulh(va, vb), [&](int i) {
        if (a[i] == 0x80000000 && b[i] == 0x80000000) return 0x7fffffffu;  // 2 a b = 2^63 saturates
        return u32(2 * std::int64_t(std::int32_t(a[i])) * std::int32_t(b[i]) >> 32);
    });
    check<S>("hsub", S::hsub(va, vb), [&](int i) {
        return u32((std::int64_t(std::int32_t(a[i])) - std::int32_t(b[i])) >> 1);
    });
}

template <class S>
void test_isa() {
    const int before = failures;
    for (int trial = 0; trial < 2000; ++trial) {
        test_common<S>();
        if constexpr (requires(typename S::V v) { S::mul_even(v, v); }) test_x86<S>();
        if constexpr (requires(typename S::V v) { S::permute2(v, v, v); }) test_permute2<S>();
        if constexpr (requires(const u32* p) { S::load_mask(p, 0u); }) test_load_mask<S>();
        if constexpr (requires(const u32* p) { S::load_halves(p, p); }) test_halves<S>();
        if constexpr (requires(typename S::V v) { S::madd52lo(v, v, v); }) test_ifma<S>();
        if constexpr (requires(typename S::V v) { S::qdmulh(v, v); }) test_neon<S>();
    }
    std::printf("%-9s %s\n", S::name, failures == before ? "ok" : "FAIL");
}

}  // namespace

int main() {
    fps::simd::for_each(fps::simd::All{}, []<class S> { test_isa<S>(); });
    std::printf("native: %s\n", fps::simd::Native::name);
    return failures ? 1 : 0;
}
