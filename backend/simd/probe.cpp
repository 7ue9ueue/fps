// Latency and reciprocal throughput of the instructions the simd structs use, in core cycles.
// Each probe times an unrolled asm block: latency from one dependent chain, throughput from 12
// independent destinations (accumulating instructions: 12 independent chains). A chain of
// dependent scalar register adds (1 cycle each) in the same run converts time to cycles; it is
// timed before and after the probes, to show frequency drift (e.g. AVX-512 licences).
//
// Not built by tools/kernel.py. On a VM, from the checkout, with the target's flags:
//   docker run --rm -v $PWD:/w -w /w -u $(id -u) <image of tools/kernel.py> \
//     g++ -std=c++23 -O2 <flags> backend/simd/probe.cpp -o build/probe
//   flock /tmp/bench.lock taskset -c $(($(nproc) - 1)) build/probe
// Output: "<instruction>\t<latency>\t<reciprocal throughput>" per line, cycles. Mixed probes
// ("a + b") interleave 6 of each per block; their throughput is cycles per instruction.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace {

using Block = void (*)(long);

struct Probe {
    const char* name;
    Block latency;     // kLatencyOps dependent instructions per call of the inner loop
    Block throughput;  // kThroughputOps independent instructions per iteration
};

constexpr int kLatencyOps = 64, kThroughputOps = 96;

#define REPT8(body) body body body body body body body body  // not .rept: LLVM drops \() in it
#define REPT64(body) ".rept 64\n" body "\n.endr\n"
#define IRP12(body) ".irp r,0,1,2,3,4,5,6,7,8,9,10,11\n" body "\n.endr\n"
#define IRP6A(body) ".irp r,0,1,2,3,4,5\n" body "\n.endr\n"
#define IRP6B(body) ".irp r,6,7,8,9,10,11\n" body "\n.endr\n"

#if defined(__x86_64__)
#define XMM_CLOBBERS "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", \
                     "xmm11", "xmm12", "xmm13", "xmm14", "xmm15", "memory"
#if defined(__AVX512F__)
#define CLOBBERS XMM_CLOBBERS, "k1"
#else
#define CLOBBERS XMM_CLOBBERS
#endif
#elif defined(__aarch64__)
#define CLOBBERS "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", \
                 "v14", "v15", "memory"
#endif

// lat: one instruction with destination register 0 that reads register 0. tp: one instruction
// with destination register \r that reads only registers 14 and 15 (or \r itself if it accumulates).
#define PROBE(name, lat, tp)                                                                          \
    Probe {                                                                                           \
        name, [](long n) { for (long i = 0; i < n; ++i) asm volatile(REPT64(lat) ::: CLOBBERS); },    \
            [](long n) { for (long i = 0; i < n; ++i) asm volatile(REPT8(IRP12(tp)) ::: CLOBBERS); } \
    }
#define MIX(name, tp_a, tp_b)                                                                                   \
    Probe {                                                                                                     \
        name, nullptr,                                                                                          \
            [](long n) { for (long i = 0; i < n; ++i) asm volatile(REPT8(IRP6A(tp_a) IRP6B(tp_b)) ::: CLOBBERS); } \
    }

std::vector<Probe> probes() {
    return {
#if defined(__x86_64__)
#define X3(op, reg) PROBE(op " " reg, op " %%" reg "1, %%" reg "0, %%" reg "0", op " %%" reg "15, %%" reg "14, %%" reg "\\r")
#define XIMM(op, imm, reg) \
    PROBE(op " " reg, op " $" imm ", %%" reg "0, %%" reg "0", op " $" imm ", %%" reg "15, %%" reg "\\r")
        X3("vpaddd", "xmm"), X3("vpmulld", "xmm"), X3("vpmuludq", "xmm"),
        X3("vpaddd", "ymm"), X3("vpsubd", "ymm"), X3("vpaddq", "ymm"), X3("vpminud", "ymm"),
        X3("vpmulld", "ymm"), X3("vpmuludq", "ymm"), XIMM("vpsrlq", "32", "ymm"), XIMM("vpsllq", "32", "ymm"),
        PROBE("vpblendd ymm", "vpblendd $0xaa, %%ymm1, %%ymm0, %%ymm0", "vpblendd $0xaa, %%ymm15, %%ymm14, %%ymm\\r"),
        X3("vpermd", "ymm"), X3("vpshufb", "ymm"),
        MIX("vpmuludq + vpmulld ymm", "vpmuludq %%ymm15, %%ymm14, %%ymm\\r", "vpmulld %%ymm15, %%ymm14, %%ymm\\r"),
        MIX("vpmuludq + vpaddd ymm", "vpmuludq %%ymm15, %%ymm14, %%ymm\\r", "vpaddd %%ymm15, %%ymm14, %%ymm\\r"),
        MIX("vpmuludq + vpsrlq ymm", "vpmuludq %%ymm15, %%ymm14, %%ymm\\r", "vpsrlq $32, %%ymm15, %%ymm\\r"),
#if defined(__AVX512F__)
        X3("vpaddd", "zmm"), X3("vpaddq", "zmm"), X3("vpminud", "zmm"), X3("vpmulld", "zmm"),
        X3("vpmuludq", "zmm"), XIMM("vpsrlq", "32", "zmm"), XIMM("vpsllq", "32", "zmm"),
        PROBE("vpblendmd zmm", "vpblendmd %%zmm1, %%zmm0, %%zmm0%{%%k1%}", "vpblendmd %%zmm15, %%zmm14, %%zmm\\r%{%%k1%}"),
        X3("vpermd", "zmm"),
        MIX("vpmuludq + vpmulld zmm", "vpmuludq %%zmm15, %%zmm14, %%zmm\\r", "vpmulld %%zmm15, %%zmm14, %%zmm\\r"),
        MIX("vpmuludq + vpaddd zmm", "vpmuludq %%zmm15, %%zmm14, %%zmm\\r", "vpaddd %%zmm15, %%zmm14, %%zmm\\r"),
        MIX("vpmuludq + vpsrlq zmm", "vpmuludq %%zmm15, %%zmm14, %%zmm\\r", "vpsrlq $32, %%zmm15, %%zmm\\r"),
#endif
#if defined(__AVX512IFMA__) && defined(__AVX512VL__)
        X3("vpmadd52luq", "ymm"), X3("vpmadd52huq", "ymm"), X3("vpmadd52luq", "zmm"), X3("vpmadd52huq", "zmm"),
        MIX("vpmadd52luq + vpmuludq zmm", "vpmadd52luq %%zmm15, %%zmm14, %%zmm\\r", "vpmuludq %%zmm15, %%zmm14, %%zmm\\r"),
#endif
#undef X3
#undef XIMM
#elif defined(__aarch64__)
#define A3(op, d, s) PROBE(op " " d, op " v0." d ", v0." s ", v1." s, op " v\\r\\()." d ", v14." s ", v15." s)
#define ACC(op, d, s) PROBE(op " " d, op " v0." d ", v1." s ", v2." s, op " v\\r\\()." d ", v14." s ", v15." s)
        A3("add", "4s", "4s"), A3("sub", "4s", "4s"), A3("umin", "4s", "4s"), A3("mul", "4s", "4s"),
        ACC("mla", "4s", "4s"), ACC("mls", "4s", "4s"), A3("umull", "2d", "2s"), A3("umull2", "2d", "4s"),
        ACC("umlal", "2d", "2s"), ACC("umlal2", "2d", "4s"), A3("sqdmulh", "4s", "4s"), A3("shsub", "4s", "4s"),
        A3("uzp1", "4s", "4s"), A3("uzp2", "4s", "4s"),
        PROBE("ushr 2d", "ushr v0.2d, v0.2d, #32", "ushr v\\r\\().2d, v15.2d, #32"),
        PROBE("shrn 2s", "shrn v0.2s, v0.2d, #29", "shrn v\\r\\().2s, v15.2d, #29"),
        PROBE("bsl 16b", "bsl v0.16b, v1.16b, v2.16b", "bsl v\\r\\().16b, v14.16b, v15.16b"),
        PROBE("tbl 16b", "tbl v0.16b, {v0.16b}, v1.16b", "tbl v\\r\\().16b, {v14.16b}, v15.16b"),
        MIX("mul + umull", "mul v\\r\\().4s, v14.4s, v15.4s", "umull v\\r\\().2d, v14.2s, v15.2s"),
        MIX("mul + sqdmulh", "mul v\\r\\().4s, v14.4s, v15.4s", "sqdmulh v\\r\\().4s, v14.4s, v15.4s"),
        MIX("mul + add", "mul v\\r\\().4s, v14.4s, v15.4s", "add v\\r\\().4s, v14.4s, v15.4s"),
        MIX("umull + uzp2", "umull v\\r\\().2d, v14.2s, v15.2s", "uzp2 v\\r\\().4s, v14.4s, v15.4s"),
#undef A3
#undef ACC
#endif
    };
}

// Median over 7 runs of the seconds per call of block(n), n grown until a run takes >= 10 ms.
double seconds_per_call(Block block, long& n) {
    const auto time = [&] {
        const auto start = std::chrono::steady_clock::now();
        block(n);
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    for (n = 1024; time() < 0.01; n *= 2) {}
    double runs[7];
    for (double& r : runs) r = time();
    std::sort(runs, runs + 7);
    return runs[3] / double(n);
}

// One dependent register-register add per cycle. Not add-immediate: Golden Cove and later
// fold chains of those at rename (lc-intel measured 24 "GHz" with them).
double add_cycle_seconds() {
    long n;
    const double s = seconds_per_call([](long calls) {
        long x = 0, one = 1;
        for (long i = 0; i < calls; ++i) {
#if defined(__x86_64__)
            asm volatile(REPT64("add %1, %0") : "+r"(x) : "r"(one));
#else
            asm volatile(REPT64("add %0, %0, %1") : "+r"(x) : "r"(one));
#endif
        }
    }, n);
    return s / kLatencyOps;
}

}  // namespace

int main() {
    const double before = add_cycle_seconds();
    std::printf("# clock from dependent adds: %.3f GHz\n", 1e-9 / before);
    std::printf("# instruction\tlatency\treciprocal throughput (cycles)\n");
    for (const Probe& p : probes()) {
        long n;
        const double tp = seconds_per_call(p.throughput, n) / kThroughputOps / before;
        if (p.latency) std::printf("%s\t%.2f\t%.3f\n", p.name, seconds_per_call(p.latency, n) / kLatencyOps / before, tp);
        else std::printf("%s\t-\t%.3f\n", p.name, tp);
    }
    std::printf("# clock after the probes: %.3f GHz\n", 1e-9 / add_cycle_seconds());
    return 0;
}
