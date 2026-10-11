// Scalar model of modarith.hpp: one function per operation. Each takes lazy values (a value with
// the bound it is known to be below), checks the input ranges the operation's contract needs,
// and returns the residue with the range the vector code guarantees. Tests run every variant
// against it; a contract violation throws, so a test that draws inputs out of range fails too.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "backend/modarith/modarith.hpp"

namespace fps::modarith::model {

struct Lazy {
    u64 value;  // < bound
    u64 bound;
};

template <u32 P>
struct Model {
    using O = Ops<simd::Scalar, P>;
    static constexpr u64 kTwo31 = u64(1) << 31, kTwo32 = u64(1) << 32;

    static Lazy add(Lazy a, Lazy b) {
        require(a.bound <= P && b.bound <= P, "add: a, b < P");
        return {(a.value + b.value) % P, P};
    }
    static Lazy sub(Lazy a, Lazy b) {
        require(a.bound <= P && b.bound <= P, "sub: a, b < P");
        return {(a.value + P - b.value) % P, P};
    }
    template <u32 M>
    static Lazy reduce(Lazy x) {
        require(M <= kTwo31 && x.bound <= 2 * u64(M), "reduce: x < 2M <= 2^32");
        return {x.value < M ? x.value : x.value - M, M};
    }
    template <u32 M>
    static Lazy sub_lazy(Lazy a, Lazy b) {
        require(b.bound <= u64(M) + 1 && a.bound + M <= kTwo32, "sub_lazy: b <= M, a + M < 2^32");
        return {a.value + M - b.value, a.bound + M};
    }
    static Lazy montgomery(Lazy a, Lazy b) {
        require(a.bound <= kTwo31 && b.bound <= kTwo31, "montgomery: a, b < 2^31");
        require(product_below(a, b, kTwo32 * P), "montgomery: a b < 2^32 P");
        return {mul(mul(a.value % P, b.value % P), inverse(kTwo32 % P)), 2 * u64(P)};
    }
    static Lazy shoup(Lazy x, u32 w) {
        require(x.bound <= kTwo32 && w < P, "shoup: x < 2^32, w < P");
        return {mul(x.value % P, w), 2 * u64(P)};
    }
    static Lazy shoup_doubling(Lazy x, u32 w) {
        require(x.bound <= kTwo31 && w < P, "shoup_doubling: x < 2^31, w < P");
        return {mul(x.value % P, w), 2 * u64(P)};
    }
    static Lazy barrett(Lazy a, Lazy b) {
        require(a.bound <= P && b.bound <= P, "barrett: a, b < P");
        return {mul(a.value, b.value), O::kBarrettBound};
    }
    static Lazy montgomery52(Lazy a, Lazy b) {
        require(a.bound <= kTwo32 && b.bound <= kTwo32, "montgomery52: a, b < 2^32");
        return {mul(mul(a.value % P, b.value % P), inverse(power(2, 52))), 2 * u64(P)};
    }
    static Lazy shoup52(Lazy x, u32 w) {
        require(x.bound <= kTwo32 && w < P, "shoup52: x < 2^32, w < P");
        return {mul(x.value % P, w), 2 * u64(P)};
    }

    static u64 mul(u64 a, u64 b) { return a * b % P; }  // a, b < P
    static u64 power(u64 a, u64 e) {
        u64 r = 1;
        for (a %= P; e; e >>= 1, a = mul(a, a))
            if (e & 1) r = mul(r, a);
        return r;
    }
    static u64 inverse(u64 a) { return power(a, P - 2); }

private:
    static void require(bool ok, const char* what) {
        if (!ok) throw std::domain_error(std::string("contract violated: ") + what + ", P = " + std::to_string(P));
    }
    // (a.bound - 1) (b.bound - 1) < limit: the largest inputs stay below limit.
    static bool product_below(Lazy a, Lazy b, u64 limit) {
        return (unsigned __int128)(a.bound - 1) * (b.bound - 1) < limit;
    }
};

}  // namespace fps::modarith::model
