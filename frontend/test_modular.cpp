// Tests of fps/modular.hpp against brute force.
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>

#include "fps/fps.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what, std::uint64_t a, std::uint64_t b) {
    if (ok) return;
    if (++failures <= 10) std::printf("FAIL %s (%llu, %llu)\n", what, (unsigned long long)a, (unsigned long long)b);
}

template <std::uint32_t P>
void test_prime(std::mt19937_64& rng) {
    static_assert(fps::mod_pow<P>(0, 0) == 1 % P);
    static_assert(fps::mod_mul<P>(P - 1, P - 1) == 1);
    const std::uint32_t edges[] = {0, 1, 2, P / 2, P - 2, P - 1};
    for (std::uint32_t a : edges)
        for (std::uint32_t b : edges)
            check(fps::mod_mul<P>(a, b) == std::uint64_t(a) * b % P, "mod_mul edge", a, b);
    for (int i = 0; i < 10000; ++i) {
        const std::uint32_t a = std::uint32_t(rng() % P), b = std::uint32_t(rng() % P);
        check(fps::mod_mul<P>(a, b) == std::uint64_t(a) * b % P, "mod_mul", a, b);
        const std::uint64_t e = rng() % 50;
        std::uint64_t power = 1 % P;
        for (std::uint64_t k = 0; k < e; ++k) power = power * a % P;
        check(fps::mod_pow<P>(a, e) == power, "mod_pow small e", a, e);
        const std::uint64_t big = rng();  // a^big = a^(big mod (P - 1)) for a != 0
        if (a != 0)
            check(fps::mod_pow<P>(a, big) == fps::mod_pow<P>(a, big % (P - 1)), "mod_pow large e", a, big);
        if (a != 0) check(fps::mod_mul<P>(a, fps::mod_inverse<P>(a)) == 1, "mod_inverse", a, 0);
    }
    check(fps::mod_pow<P>(0, ~std::uint64_t(0)) == 0, "0^max", 0, 0);
    bool threw = false;
    try {
        fps::mod_inverse<P>(0);
    } catch (const std::domain_error&) {
        threw = true;
    }
    check(threw, "mod_inverse(0) throws", 0, 0);
}

}  // namespace

int main() {
    std::mt19937_64 rng(1);
    test_prime<2>(rng);
    test_prime<3>(rng);
    test_prime<998244353>(rng);
    test_prime<1000000007>(rng);
    test_prime<1073741789>(rng);  // largest prime below 2^30
    test_prime<4294967291>(rng);  // largest prime below 2^32
    if (failures) std::printf("%d failures\n", failures);
    return failures != 0;
}
