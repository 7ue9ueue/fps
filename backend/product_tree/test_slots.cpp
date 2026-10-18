// Tests the lanes' slot loads (issue #91 round 2): load_full on every simd struct the machine runs
// against the words in Montgomery form, for degrees 1 .. 40, each factor's words ending at a page
// followed by an unreadable page (a read past a factor faults); load_linear alike; Slots::load
// of full and partial slots through product() (test.cpp). Exits nonzero on a failure.
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "backend/product_tree/kernels.hpp"

namespace {

namespace pt = fps::product_tree::detail;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
constexpr u32 P = 998244353;

std::mt19937_64 rng(9123);
int failures = 0, checks = 0;

void expect(bool ok, const std::string& what, u64 detail = 0) {
    ++checks;
    if (!ok && ++failures <= 20) std::printf("FAIL: %s (%llu)\n", what.c_str(), static_cast<unsigned long long>(detail));
}

// 8 factors of n words each, each ending where an unreadable page starts.
struct Guarded {
    std::size_t page = std::size_t(sysconf(_SC_PAGESIZE));
    std::vector<std::pair<void*, std::size_t>> maps;
    const u32* p[8];

    explicit Guarded(std::size_t n, int kind) {
        const std::size_t bytes = (n * 4 + page - 1) / page * page;
        for (int l = 0; l < 8; ++l) {
            void* m = mmap(nullptr, bytes + page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            mprotect(static_cast<char*>(m) + bytes, page, PROT_NONE);
            maps.push_back({m, bytes + page});
            u32* f = reinterpret_cast<u32*>(static_cast<char*>(m) + bytes) - n;
            for (std::size_t j = 0; j < n; ++j) f[j] = kind == 1 ? P - 1 : u32(rng() % P);
            p[l] = f;
        }
    }
    ~Guarded() {
        for (auto [m, b] : maps) munmap(m, b);
    }
};

template <class L>
void test_struct(const char* name) {
    for (u32 d = 1; d <= 40; ++d) {
        for (int kind = 0; kind < 2; ++kind) {
            const Guarded g(d + 1, kind);
            std::vector<u32> c(8 * (d + 1) + 16, 0xdeadbeef);
            pt::load_full<L>(g.p, d, c.data());
            std::size_t bad = 0;
            for (u32 j = 0; j <= d; ++j)
                for (int l = 0; l < 8; ++l) bad += c[8 * j + l] != pt::multiply_mod(g.p[l][j], pt::kR);
            for (std::size_t i = 8 * (d + 1); i < c.size(); ++i) bad += c[i] != 0xdeadbeef;
            expect(bad == 0, std::string("load_full<") + name + "> degree " + std::to_string(d), bad);
            if (d == 1) {
                std::vector<u32> e(32, 0xdeadbeef);
                pt::load_linear<L>(g.p, e.data());
                expect(std::equal(c.begin(), c.begin() + 16, e.begin()), std::string("load_linear<") + name + ">");
            }
        }
    }
}

}  // namespace

int main() {
    test_struct<fps::simd::Scalar>("Scalar");
    test_struct<fps::product_tree::Lanes>("Lanes");
#if defined(__SSE4_1__)
    test_struct<fps::simd::Sse>("Sse");
#endif
#if defined(__AVX2__)
    test_struct<fps::simd::Avx2>("Avx2");
#endif
#if defined(__AVX512VL__)
    test_struct<fps::simd::Avx512Vl>("Avx512Vl");
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
    test_struct<fps::simd::Neon>("Neon");
#endif
    std::printf("%s product_tree slots: %d checks, %d failures\n", failures ? "FAIL" : "pass", checks, failures);
    return failures != 0;
}
