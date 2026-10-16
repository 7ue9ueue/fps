// Pinning a hot loop's address on arm (issues #197, #230).
//
//   simd::align_loop<k>(p, q, n);   // right before the loop: padding to 32 bytes, then k nops
//   do { ... } while (--n != 0);    // a do-while on those registers: no guard in between
//
// Neoverse V2 runs some hot loops 3-12% slower at one start than at another, the same code: small's
// loop with all R vectors at any start but 0 mod 32 (backend/small/notes/197-r1.md), chirp's at any
// but 12 mod 32, the division recurrence's inner loop where it spans 3 blocks of 32 bytes
// (backend/simd/notes/230-r1.md). Which start is fast depends on the loop's instructions, so it is
// measured per loop. -mcpu=neoverse-v2 pads a loop to 32 bytes only when that skips at most 15 bytes
// (-falign-loops=32:16): where such a loop landed, and so its speed, changed with any edit of the
// code before it and with the compiler.
//
// align_loop emits `.p2align 5` and k nops, with x... (the registers the loop starts from: its
// pointers, counters and bound; 1 to 4 of them) as in-out operands, so the compiler computes them
// before the padding. It may still put other setup (constants) in between, a fixed number of
// instructions for given code and compiler: check the objdump with GCC 16.2 and 15.2. A start of
// 20-28 mod 32 cannot be pinned: GCC's own loop alignment then pads to the next 32 bytes. The
// padding (at most 7 + k nops) runs at each entry of the loop. On other targets it does nothing.
#pragma once

namespace fps::simd {

#if defined(__aarch64__)
#define FPS_ALIGN_LOOP_PAD ".p2align 5\n\t.rept %c[k]\n\tnop\n\t.endr"
template <int k = 0, class A>
[[gnu::always_inline]] inline void align_loop(A& a) {
    asm volatile(FPS_ALIGN_LOOP_PAD : "+r"(a) : [k] "i"(k));
}
template <int k = 0, class A, class B>
[[gnu::always_inline]] inline void align_loop(A& a, B& b) {
    asm volatile(FPS_ALIGN_LOOP_PAD : "+r"(a), "+r"(b) : [k] "i"(k));
}
template <int k = 0, class A, class B, class C>
[[gnu::always_inline]] inline void align_loop(A& a, B& b, C& c) {
    asm volatile(FPS_ALIGN_LOOP_PAD : "+r"(a), "+r"(b), "+r"(c) : [k] "i"(k));
}
template <int k = 0, class A, class B, class C, class D>
[[gnu::always_inline]] inline void align_loop(A& a, B& b, C& c, D& d) {
    asm volatile(FPS_ALIGN_LOOP_PAD : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : [k] "i"(k));
}
#undef FPS_ALIGN_LOOP_PAD
#else
template <int k = 0, class... T>
[[gnu::always_inline]] inline void align_loop(T&...) {}
#endif

}  // namespace fps::simd
