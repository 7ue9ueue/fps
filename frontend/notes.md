# frontend

The user API in `include/fps/`, planned in `DESIGN.md`, built issue by issue with `/build`.
Tests: `frontend/test_*.cpp`, one per area. Bench: `frontend/bench.cpp` (API calls as `<op>/<size>`,
the backend call they wrap as `<op>@kernel/<size>`).

Prototype: a local, unpushed branch (`claude/frontend-lc-opt-backend-8360f6`, worktree
`.claude/worktrees/frontend-lc-opt-backend-8360f6`) implements most of DESIGN.md over a private copy
of lc-opt's kernels. Read it and move code from it; do not copy its `include/fps/lcopt/` (that work
belongs in `backend/`).

## Log

- 2026-10-11: scalars (`mod_mul`, `mod_pow`, `mod_inverse`) from the prototype's `modular.hpp`,
  as the first piece through the pipeline. Tests against brute force on 6 primes from 2 to 2^32 - 5.
- 2026-10-11: `Factorials` (#17): tables of i! and 1 / i!, `binomial(n, k)`, from the prototype's
  `factorials.hpp`. The backend has no kernel for it, so `detail::factorial_tables` is the simple
  loop: two chains of n - 1 dependent `mod_mul` and one `mod_inverse`. lc-opt's 32-lane product
  chains are backend issue #33. `static_assert(P == 998244353)` as DESIGN.md (Moduli) says.
  - DESIGN.md: its line on `Factorials` did not say which n are valid, what `inverse(i)` is, what
    C(n, k) is for k > n, or whether indices are checked. Now: n <= P, as P! = 0 has no inverse
    (n > P throws before the tables are allocated); `inverse(i)` = 1/i!; C(n, k) = 0 for k > n;
    indices unchecked, as `operator[]`.
  - Tests: the tables against running products for n = 0..3, 2^k - 1, 2^k, 2^k + 1 (k <= 20) and
    random n; `detail::factorial_tables` on P = 2, 3, 5, 101 for every n <= P (Wilson at n = P)
    and on 2^32 - 5; binomials against Pascal's triangle (300 rows) and against products of k
    factors (random); both errors. Five injected bugs (bound, binomial edge, index, table entry,
    first entry) all fail the test.
  - Bench (`frontend/bench.cpp`): `factorials/<lg>` builds the tables, `binomial/16` is 2^16
    random C(m, k) from a table of 2^16. No `@kernel` pair: no backend call is wrapped. Median ns
    per entry (min-max), 11 rounds, GCC 15.2, the code of PR #34:

    | Machine (target) | factorials/10 | /14 | /18 | /22 | binomial/16 |
    |---|---:|---:|---:|---:|---:|
    | `lc-bench` (avx2) | 7.13 (7.12-7.15) | 7.02 (7.01-7.03) | 10.26 (10.21-10.30) | 10.60 (10.58-10.63) | 3.58 (3.47-3.64) |
    | `lc-intel` (avx512) | 5.91 (5.91-5.93) | 5.86 (5.86-5.89) | 7.64 (7.62-7.68) | 7.83 (7.83-7.85) | 1.56 (1.54-1.59) |
    | `lc-zen5` (zen5) | 5.95 (5.94-5.96) | 5.86 (5.85-5.86) | 6.89 (6.88-6.97) | 7.11 (7.10-7.14) | 1.78 (1.76-1.78) |
    | `lc-arm` (arm) | 8.24 (8.23-8.25) | 8.09 (8.09-8.10) | 9.66 (9.64-9.72) | 10.12 (10.07-10.14) | 2.90 (2.88-3.01) |

    The loop is latency-bound: each product waits on the one before. lc-opt's chains: about
    0.6 ns per entry (estimate from its notes, #33). The rise from lg 14 to lg 18 is most likely
    page faults on the fresh, zero-filled vectors (guess, not profiled).
- 2026-10-11: `Polynomial` (#16): the class, `f + g`, `f - g`, `-f`, `f * g`, scalars, compound
  forms, from the prototype's `polynomial.hpp`. Storage is a `std::vector<std::uint32_t>`.
  `static_assert(P == 998244353)` as DESIGN.md (Moduli) says, so no product for other primes.
  - DESIGN.md: its Examples need `1 - x`, `x + c` and `-1 * f`, which its list of operators did
    not have. Now `f + c`, `c + f`, `f - c`, `c - f` are listed, and a scalar c is any integer
    type, taken mod P. Conventions: a size past a backend limit throws `std::length_error` (an
    NTT product past 2^25 coefficients; `ntt::Convolution` throws it), not `std::domain_error`.
  - Sums, negation and scaling run on the backend's primitives (`detail::lanewise` on
    `simd::Native` with `modarith::Ops`: `add`, `sub`, Shoup's product), since GCC 15.2 does not
    vectorize such loops at -O2.
  - `f * g`, n and m coefficients: a factor with at most one nonzero term (c x^k; constants and
    zero too) is a shift and scale, O(n + m). Else schoolbook if min(n, m) <= 32 or n m < 2^18
    (`detail::schoolbook`, 64-bit sums), else `ntt::Convolution`. Schoolbook is here because a
    short `ntt::Convolution` costs 52 us on `lc-bench` (22 us on `lc-arm`), mostly the first touch
    of a fresh huge page, against 0.43 us for schoolbook at 32 x 32. Cutoffs measured on
    `lc-bench`: schoolbook 0.24 ns per term; it wins up to about 450 x 450 (384 x 384: 36 us
    against 56 us; 512 x 512: 64 against 55) and up to m = 32 at n = 2^14, 2^17, 2^20 (m = 32:
    136 us against 163 at 2^14; m = 48: 192 against 163). Backend issues: #37 (`small`: a tuned
    short product and per-target cutoffs), #38 (`ntt`: memory kept across calls, product into
    caller memory).
  - Tests (`test_polynomial.cpp`): every member and operator against O(n^2) references, on sizes
    0, 1, 2^k - 1, 2^k, 2^k + 1 (k <= 10), every pair of them for `f * g`, random inputs with
    many zeros or all P - 1 (the largest schoolbook sums); shapes at both cutoffs; NTT lengths
    2^10 .. 2^21 at random points; shifts by c x^k with c = 0, 1, random; aliasing (f += f,
    f -= f, f *= f); 4 threads; both errors. 10 injected bugs all fail it: schoolbook's reduction
    cadence and a block bound, the shift's offset and scale, single-term detection, `degree`,
    negation, `f - c` on an empty f, the residue of a negative c, `truncate`.
  - Bench (`frontend/bench.cpp`): `multiply/<lg>` is `f * g` for factors of 2^lg / 2 coefficients,
    `multiply@kernel/<lg>` the same `ntt::Convolution` call by hand (construct, copy in,
    multiply). Below lg = 10 the API runs schoolbook. Per-round ratio API / kernel, median of 11
    rounds (IQR in the PR), GCC 15.2, main at 06feb5c (#35) plus this change; `lc-arm` before #35,
    which did not change its scalar NTT:

    | lg | `lc-bench` (avx2) | `lc-intel` (avx512) | `lc-zen5` (zen5) | `lc-arm` (arm, scalar NTT) |
    |---:|---:|---:|---:|---:|
    | 6 | 0.008 (436 ns, 52 us) | 0.008 (404 ns, 51 us) | 0.010 (303 ns, 30 us) | 0.017 (359 ns, 22 us) |
    | 9 | 0.282 | 0.253 | 0.361 | 0.418 |
    | 10 | 1.006 | 1.015 | 1.006 | 1.006 |
    | 12 | 1.008 | 1.023 | 1.009 | 1.003 |
    | 14 | 1.017 | 1.046 | 1.016 | 1.002 |
    | 16 | 1.026 | 1.044 | 1.023 | 1.002 |
    | 18 | 1.031 | 1.028 | 1.021 | 1.002 |
    | 20 | 1.038 | 1.044 | 1.019 | |
    | 22 | 1.058 | 1.046 | 1.035 | |

    The API is 1-6% slower than the kernel case from 2^10 on x86. `ntt::Convolution` leaves the
    product in its own memory, which it unmaps when it dies, so `f * g` must write it elsewhere.
    That write is the whole gap, and no frontend change can avoid it. Same-run decomposition
    (`lc-intel`, 2^14; us):
    - kernel case: 113.4;
    - plus a copy into one buffer kept across calls: 118.5;
    - plus a new zeroed vector, no copy: 118.7;
    - plus a copy into a new vector: 119.1;
    - the API: 118.9.

    Any write of the product's size after the product costs the same 5 us. The copy alone, timed
    by itself, takes 1.4 us; guess: the fresh mapping per call (#38) makes it dearer. #38 asks
    for a product into caller memory.

    Copy destinations tried (`lc-bench`, out of a page-aligned array, 2^12 to 2^16 words):
    - a 64-byte aligned vector is 4-5x slower (equal page offsets: 4K aliasing);
    - 16 bytes past a page boundary is 15-20x slower;
    - `std::vector`'s own offset is as fast as 64 bytes past.

    On `lc-intel` the offset changes memcpy by at most 10%.
  - Review of #39, round 1 (four findings), and what changed:
    1. The header now states the cost of `f * g` against the kernel (0.6% to 6% on x86) and the
       fast path: `fps::ntt::Convolution` itself today, `fps::raw::mul` after #38.
    2. Element-wise operators. They copied an lvalue f, then ran the stream in place: two
       passes. Now each writes its result in one pass into new, unwritten memory; `+` and `-`
       with an rvalue left operand still work in place. `std::vector` cannot allocate without
       writing, and with a non-initializing allocator GCC 15 copies it word by word (checked in
       the asm), so the storage is now `detail::Words`: a `unique_ptr<uint32_t[]>` and a size,
       copied with memcpy. New bench pairs `add`, `sub`, `neg`, `scale` (`f * c`) at lg 10, 14,
       18, 22; the kernel case is the modarith stream into new memory, as the operator's result.
    3. Cutoff `n m < 2^18` -> `n m < 3 2^16` (443 x 443, the crossover measured on `lc-bench`).
       New bench pair `multiply/480x480`, a shape the old cutoff ran schoolbook.
    4. Scalars are integer types of at most 64 bits (`detail::Integer`); `__int128` no longer
       compiles (static_assert in the test). DESIGN.md says so.
  - Bench order (a note from the backend coordinator: cases in one process affect each other).
    The cases run Factorials first, then the element-wise pairs, then the products, which map
    fresh memory per call. Every case makes its inputs in its setup, so they live only while it
    runs.
  - Tests: copies, moves and self-assignment; the rvalue forms; shapes at the new cutoff
    (443 x 443, 443 x 444, 3071 x 64, 64 x 3072). `dirty_heap(n)` frees n words of P - 1 before
    `truncate` and `reverse`, so a word they forget to write shows up: glibc reuses the block (a
    dropped zero fill in `reverse` fails 31 checks on `lc-amd`); the Mac's allocator zeroes small
    blocks on free, so there it passes anyway. 10 mutations of the new code: 8 fail the test on
    the Mac, the `reverse` one fails it on Linux, and the tenth (a cutoff of 4 2^16) is no bug.
  - API / kernel after these changes, median of 11 rounds, GCC 15.2, main 27905ef plus this PR:

    | case | `lc-bench` (avx2) | `lc-intel` (avx512) | `lc-zen5` (zen5) | `lc-arm` (arm) |
    |---|---:|---:|---:|---:|
    | add/10, /14, /18, /22 | 0.77, 0.97, 1.01, 1.02 | 0.99, 0.99, 1.00, 1.00 | 0.60, 0.83, 0.95, 1.03 | 0.87, 0.94, 0.99, 0.98 |
    | sub | 0.75, 0.97, 1.01, 0.99 | 1.02, 1.03, 1.00, 1.00 | 0.62, 0.84, 0.96, 1.04 | 0.96, 0.97, 1.00, 0.98 |
    | neg | 0.83, 0.86, 0.92, 0.93 | 0.95, 0.98, 1.00, 1.00 | 0.46, 0.44, 0.69, 0.73 | 0.89, 0.73, 0.90, 0.97 |
    | scale | 1.00, 1.00, 1.02, 0.97 | 0.98, 0.98, 0.99, 1.00 | 0.98, 0.84, 0.89, 1.08 | 0.99, 1.00, 1.00, 1.00 |
    | multiply/10, /14, /18, /22 | 1.005, 1.019, 1.027, 1.061 | 1.013, 1.047, 1.030, 1.045 | 1.004, 1.015, 1.017, 1.033 | 0.989, 0.981, 0.985 (/18) |
    | multiply/480x480 | 1.017 | 1.015 | 1.003 | 0.992 |

    The element-wise pairs run the same modarith ops on the same inputs into the same kind of
    memory, yet range from 0.44 to 1.08. Placement, not work: on `lc-zen5`, running each kernel
    case before its API case (sizes 18 and 22 only) moved scale/22 from 1.077 to 0.973 and
    neg/22 from 0.73 to 0.85, and a second run of avx512 add/10 gave 1.06 against 0.99. So this
    harness resolves these pairs to about 10% at 2^22 on `lc-zen5` and 5% at small sizes on
    `lc-intel`. The products keep their gap (it grows with size on x86, and the decomposition
    above explains it); on `lc-arm` the API now comes out 1-2% faster, again placement.
  - Review of #39, round 2 (one finding): `+=` and `-=` with a longer g grew the left operand
    with `Words::resize` (copy and zero fill), then rewrote every word: two passes, which the
    rvalue `+` and `-` (e.g. `1 - x - x * x`) inherited. Now they build the result the way the
    `const&` operator does, in one pass. Tests: shorter left operands (0 to 1000 coefficients)
    for `+=`, `-=` and the rvalue forms, after `dirty_heap`. After #30 no default build selects
    `ntt::variant::scalar`, so the products bench runs lg 6 to 22 on every target. The header
    now says that `Words` has no spare capacity (growth one word at a time is O(n^2)) and that
    copy assignment always allocates; no caller does either on a hot path.
- 2026-10-11: `detail::factorial_tables` calls the backend kernel `factorials::tables` (issue #33,
  `backend/factorials/notes.md`) for 2^12 < P < 2^30, and the kernel's scalar model (the loop
  above) for other P. `frontend/bench.cpp` has `factorials@kernel/<lg>`: the same two
  zero-filled vectors and the kernel call. `test_factorials.cpp` also runs every n <= 4099 at
  P = 4099, where the kernel path runs up to n = P.
  - `factorials/<lg>` new/base against main (27905ef), 11 rounds, lg 10 / 14 / 18 / 22: `lc-zen5`
    0.155 / 0.061 / 0.211 / 0.250, `lc-arm` 0.203 / 0.122 / 0.252 / 0.293. Against
    `factorials@kernel/<lg>`: 0.998-1.008 (Zen 5), 0.985-1.001 (Neoverse V2).
  - The allocation now costs more than the kernel from lg 18 on: at lg 22 on Zen 5, 1.79 ns per
    entry through the API against 0.46 for the kernel into warm buffers, so 1.3 ns is the
    zero-fill and page faults of the two vectors. Tables allocated without zero-fill and in huge
    pages (DESIGN.md plans library-owned huge pages) should cut most of it (guess, not tried).
    64-byte aligned tables would also let a later kernel round keep zmm on Emerald Rapids at
    large n: 0.44 against 0.58 ns per entry (`backend/factorials/notes.md`, Log).
- 2026-10-11: `Factorials` stores its tables in `detail::Words`, allocated unwritten (#51), so the
  kernel writes each entry once; no zero fill first. `Words` moved from `polynomial.hpp` to its own
  header, `fps/words.hpp`, as both headers use it. API and behaviour unchanged. The bench's
  `factorials@kernel/<lg>` now calls the kernel on two unwritten tables
  (`make_unique_for_overwrite`), the memory the API must return. No 64-byte alignment: with
  `unique_ptr<uint32_t[]>` it needs an aligned `new` and a matching deleter, more than one line.
  - Tests: copies and moves of `Factorials` (construction, assignment, self-assignment) on every
    size. A dropped write of the last entry fails them: i! != 0 for i < P, whatever the memory held.
  - Bench, 11 rounds, GCC 15.2, origin/main 03c970b against this change; ns per entry, medians:

    | Machine (target) | lg | new/base | API / kernel (IQR) | API | kernel (fresh) | base API |
    |---|---:|---:|---:|---:|---:|---:|
    | `lc-bench` (avx2) | 10 / 14 / 18 / 22 | 0.964 / 0.893 / 1.029 / 1.013 | 1.004 / 1.000 / 1.003 / 1.000 (0.995-1.010) | 1.34 / 0.67 / 4.08 / 4.50 | 1.34 / 0.67 / 4.06 / 4.49 | 1.39 / 0.75 / 3.96 / 4.43 |
    | `lc-intel` (avx512) | 10 / 14 / 18 / 22 | 0.955 / 0.799 / 0.858 / 0.837 | 1.002 / 1.000 / 1.000 / 0.999 (0.990-1.009) | 1.02 / 0.49 / 2.06 / 2.28 | 1.02 / 0.49 / 2.06 / 2.27 | 1.07 / 0.62 / 2.41 / 2.72 |
    | `lc-zen5` (zen5) | 10 / 14 / 18 / 22 | 0.941 / 0.887 / 0.932 / 0.878 | 0.970 / 1.001 / 1.001 / 1.006 (0.969-1.013) | 0.86 / 0.32 / 1.36 / 1.54 | 0.88 / 0.32 / 1.36 / 1.53 | 0.91 / 0.36 / 1.45 / 1.76 |
    | `lc-arm` (arm) | 10 / 14 / 18 / 22 | 0.974 / 0.948 / 0.972 / 1.021 | 1.001 / 1.000 / 1.000 / 0.990 (0.984-1.013) | 1.63 / 0.94 / 2.37 / 2.87 | 1.63 / 0.94 / 2.37 / 2.88 | 1.68 / 0.99 / 2.44 / 2.87 |

    The API is within noise of the kernel on fresh tables at every lg. Dropping the fill saves 12-16%
    at lg 22 on `lc-intel` and `lc-zen5`, nothing on `lc-arm` and `lc-bench`; on `lc-bench` lg 18
    is 3% slower (IQR 1.028-1.031; cause not known; guess: the page faults now land inside the
    kernel's 32 scattered lanes instead of a sequential fill). From lg 18 the fresh pages dominate:
    at lg 22 the kernel into warm buffers takes 0.73 (Zen 3), 0.58 (Emerald Rapids), 0.46 (Zen 5)
    and 0.92 (Neoverse V2) ns per entry (`backend/factorials/notes.md`), against 4.49, 2.27, 1.53 and
    2.88 into fresh ones. Only memory the library keeps or maps in huge pages (DESIGN.md, Memory)
    can close that; not tried here.
- 2026-10-11: `f * g` short products on the backend kernel `small` (issue #37,
  `backend/small/notes.md`). `detail::schoolbook` (scalar, 0.24 ns per term on Zen 3) and
  `detail::schoolbook_wins` (`min(n, m) <= 32 || n m < 3 2^16`, set when a Convolution cost 51 us)
  are gone: `f * g` calls `small::multiply` where `small::beats_ntt(n, m)` says it is faster than
  the NTT on this target (a cost model in the backend), else `convolution()` as before.
  - Behaviour: a short factor next to one past the NTT's 2^25 now gets its product from
    `small::multiply` where that beats the NTT's extrapolated time, instead of
    `std::length_error`. The error test now multiplies 2^25 by 4096 coefficients (was 33), and
    2^25 x 2 is checked at random points.
  - Tests: shapes around the cutoffs, which differ per target (24 x 24 to 128 x 129, 3000 x 4 to
    3000 x 128, both orders), random and all P - 1. Pass at -O2 and ASan/UBSan on every target's
    test machines and the Mac.
  - Bench: new pairs `multiply/<n>x<m>` (16x16, 4096x4, 4096x16) against `small::multiply` into
    new memory (`multiply@kernel/<n>x<m>`). `lc-bench`, 5 rounds: 85 against 80 ns (the scans for
    a single term and the cutoff), 1737 against 1729, 4538 against 4535.
  - `multiply/<lg>` against main (e754d90), 5 rounds, median of paired new/base ratios:

    | lg (n x n) | `lc-bench` | `lc-intel` | `lc-zen5` | `lc-arm` |
    |---|---:|---:|---:|---:|
    | 6 (32 x 32) | 0.336 (436 -> 147 ns, NTT) | 0.313 (NTT) | 0.245 (76 ns, small) | 0.745 (NTT) |
    | 7 (64 x 64) | 0.202 | 0.190 | 0.183 | 0.427 |
    | 8 (128 x 128) | 0.125 | 0.126 | 0.110 | 0.277 |
    | 9 (256 x 256) | 0.075 | 0.080 | 0.066 | 0.168 |
    | 10 to 22, 480x480 (NTT, code unchanged) | 0.981-1.007 | 0.975-1.013 | 0.999-1.007 | 0.999-1.002 |

    Below lg 10 main ran the scalar loop up to 443 x 443; with the Convolution of issue #38 the NTT
    wins there from about 32 x 32 (Zen 3, Neoverse V2) to 104 x 104 (Zen 5).
