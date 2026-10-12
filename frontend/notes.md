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
- 2026-10-11: `Values` (#19): `fps::Values<P>` with `+`, `-`, `*` (value by value), scalars,
  `multiply_add`, `fps::Layout` (`automatic`, `pointwise`), `to_values`, `to_coefficients`, and
  `raw::to_values`, `raw::to_coefficients`, `raw::add_coefficients`, on `ntt::Transform`. From the
  prototype's `values.hpp`, which had no layouts; the split into values from lc-opt-explore
  `showcase-abc222_h_dc` v2 (`Split`), rewritten as plain scalar code.
  - Storage: max(L, 64) + 16 words, 64-byte aligned (the transform needs 32 and reads 16 words
    of padding). L < 64 is held at 64: x^L - 1 divides x^64 - 1, so sums and products are exact
    mod x^64 - 1 and the conversion back folds mod x^L - 1. One code path for every L.
  - Per length, a table set built once (`std::call_once`) and kept: the `ntt::Transform`, the leaf
    roots r(k) and, up to 2^23, the 8th roots rho_v of each leaf's w_v. A fresh `Transform` per
    call would rebuild L / 16 table entries and allocate.
  - Automatic layout: the backend's leaf form; `*` is an 8 x 8 product per leaf mod x^8 - w_v
    (64-bit sums, two `%` per word). Pointwise: each leaf twisted by rho_v^k and taken through an
    8-point transform, so `*` is one product per word (modarith Barrett on `simd::Native`). Above
    2^23 no such roots exist (P - 1 = 119 2^23): pointwise keeps the leaf form there.
  - `to_coefficients(const Values&)` copies A into a per-thread scratch (A is const and the inverse
    works in place), inverts there and copies out; on an rvalue (`to_coefficients(A * B)`) it
    inverts in A's own memory. `Polynomial` gets `detail::unwritten_polynomial` (no zero fill
    before the copy out); nothing else in `polynomial.hpp` changes. `Values` has its own add,
    subtract and scale streams on `detail::lanewise` (an earlier version moved `Polynomial`'s
    streams to `detail` to share them; undone on the rebase below, to leave `Polynomial` as main
    has it while #18 edits the same lines).
  - Backend issue #49: leaf products, the pointwise split, and an inverse into other memory.
    Searched first: #38 (Convolution memory) and #4 (ntt intrinsics) do not cover them.
  - DESIGN.md: `Values()` (empty, for containers) and `Values(L, layout)` (zero: the start of a
    `multiply_add` sum and the buffer for `raw::to_values`; the ABC222 H example needs one), which
    it did not list; scalars are any integer of at most 64 bits, as for `Polynomial`; pointwise
    above 2^23 keeps the leaf form (no roots); which errors throw what; the raw functions' scratch,
    which "never allocate" did not allow for (`to_coefficients` of a const A needs L words to run
    the in-place inverse); `Layout` declared before `Values`.
  - Tests (`test_values.cpp`): both layouts, L = 2^0 .. 2^11, inputs of 0, 1, 2^k - 1, 2^k,
    2^k + 1 and random sizes, random and all P - 1: conversions, every operator and its rvalue
    form, aliasing (`A *= A`, `A.multiply_add(A, B)`, `(B, A)`, `(A, A)`), scalars 0, 1, P - 1,
    -1, -3P - 5, 2^64 - 1, against cyclic schoolbook products; the raw functions on every range for
    L <= 32 and on edge and random ranges above; DESIGN.md's two examples (one factor in many
    products and a sum of products; ABC222 H's divide and conquer, here f = 1 + x f^2 against the
    Catalan numbers, N = 4096); L = 2^20 exact products, 2^23 and 2^24 pointwise with wrapping at
    2^23-th roots of unity; 4 threads building tables of new lengths at once; every error. 12
    injected bugs (leaf modulus sign, a split twiddle, the fold, `multiply_add`, `add_coefficients`
    on short ranges, the odd-leaf rho, the layout check, the leaf product's wrap, the zero fill,
    the 2^23 limit, 1/8, the raw range check) all fail it; the 2^23 one crashes.
  - Bench (`frontend/bench.cpp`), L = 2^lg, 11 rounds, GCC 15.2, main 7ddddf3 plus this change (before #47).
    `to_values` (f of L / 2 coefficients into new memory) against `Transform::forward` by hand
    (new aligned memory, copy, zero fill, forward); `to_coefficients` against a copy into new
    aligned memory and the inverse in place; `add_coefficients` (ABC222 H's step, the upper half
    added into a kept array) against copy, inverse and a modarith add. API / kernel, median of the
    per-round ratios:

    | case | `lc-bench` (avx2) | `lc-intel` (avx512) | `lc-zen5` (zen5) | `lc-arm` (arm) |
    |---|---:|---:|---:|---:|
    | to_values/10, /14, /18, /22 | 1.018, 0.993, 1.003, 0.992 | 1.011, 0.996, 0.981, 1.000 | 0.999, 0.998, 0.997, 1.001 | 1.007, 1.000, 0.999, 1.001 |
    | to_coefficients | 1.020, 1.090, 1.125, 1.173 | 1.025, 1.090, 1.147, 1.172 | 1.040, 1.094, 1.123, 1.145 | 1.020, 1.024, 1.026, 1.036 |
    | add_coefficients | 0.984, 0.994, 1.013, 0.992 | 1.005, 1.003, 0.959, 1.001 | 0.991, 0.989, 0.998, 0.988 | 1.002, 0.994, 1.002, 1.004 |

    `lc-zen5`: from the gate's run of 311471a (21 rounds, main 03c970b); the rest from separate
    runs of 11 rounds.

    `to_coefficients` pays the copy out of the scratch; the kernel case reads the coefficients in
    place. Only an inverse into other memory closes it (#49): a `Polynomial` is 16-byte aligned
    and has no padding, so the frontend cannot run the inverse in it. Aligning every
    `Polynomial` would change all their allocations and the copy placement measured for #16, to
    save one copy here; #49 saves the copy in as well.

    No kernel to pair with, ns per coefficient, lg 10 / 14 / 18 / 22:

    | case | `lc-bench` | `lc-intel` | `lc-zen5` | `lc-arm` |
    |---|---|---|---|---|
    | values_product (leaf form) | 6.91 / 6.86 / 6.87 / 6.88 | 4.21 / 4.18 / 4.22 / 4.23 | 4.39 / 4.35 / 4.35 / 4.35 | 4.73 / 4.70 / 4.70 / 4.70 |
    | values_product_pointwise | 0.23 / 0.19 / 0.18 / 0.33 | 0.17 / 0.15 / 0.36 / 0.40 | 0.08 / 0.06 / 0.11 / 0.17 | 0.34 / 0.30 / 0.30 / 0.34 |
    | to_values_pointwise | 7.55 / 7.78 / 8.12 / 8.52 | 5.23 / 5.53 / 5.63 / 6.12 | 5.11 / 5.21 / 5.38 / 5.56 | 7.29 / 7.81 / 8.41 / 9.11 |
    | to_coefficients_pointwise | 8.67 / 9.03 / 9.45 / 10.00 | 6.07 / 6.39 / 6.61 / 7.21 | 6.27 / 6.40 / 6.60 / 6.83 | 8.01 / 8.53 / 9.16 / 10.01 |
    | to_values (for scale) | 0.60 / 0.85 / 1.18 / 1.48 | 0.55 / 0.85 / 1.14 / 1.70 | 0.32 / 0.45 / 0.61 / 0.77 | 1.13 / 1.66 / 2.25 / 2.94 |

    The scalar leaf product costs 2 to 14 times `to_values` (one forward transform), so until #49
    a single product through `Values` is slower than `f * g`; a pointwise conversion costs 3 to 20
    times `to_values`.
    Products in the pointwise layout are already a stream (0.06 to 0.40 ns). Not tuned here, as
    AGENTS.md (Performance) asks: these are #49's.
  - Gate runs. The first failed avx2 at 1.0081, from `add`, `sub`, `neg` and `scale` at +1% to
    +6%, with their `@kernel` cases (bench code this change does not touch) at +1% to +10%
    alike. On `lc-bench` against main, those cases alone gave 1.0001 with the `Values` cases
    registered and 0.9966 without; after the factorials, 1.0048 (the factorials) with and 0.9991
    without, the element-wise cases within 0.5% either way except the noisy lg 22. So neither
    the code nor the registration moves them; they move only in some full-suite runs (+2-3% in
    three gate runs, not in a fourth with the same code). Placement, as in #16's log; not fixed.
  - After #48 (Convolution memory per thread) the gate found arm `multiply/<lg>` and
    `multiply@kernel/<lg>` 3-5% slower from 2^10. Not the run order: still so with the `Values`
    cases last. `nm` of the two binaries: with `Transform::forward` and `inverse` in the program,
    GCC kept the NTT's top-level helpers out of line, shared by both paths. Forcing them inline
    (`backend/ntt/intrinsics.hpp`, 5 attributes; `backend/ntt/notes.md`) gave 0.999 against main,
    from 1.034. The `Values` cases now run last anyway: they leave tables and scratch behind.
    (Dropped on the rebase below: #61 and #68 rewrote those levels, and the slowdown is gone.)
  - 2026-10-12: rebased onto main 006fa57 (after #37, #38, #51, #61, #65, #68). No conflicts in
    the frontend; the tests pass unchanged on the Mac (scalar, arm), `lc-amd` (scalar, sse, avx2),
    `lc-intel-test`, `lc-zen5-test` and `lc-arm`, at -O2 and under ASan/UBSan.
    - The NTT inlining commit is dropped: a backend change, and no longer needed. Without it, on
      `lc-arm`, `multiply/<lg>` against main is 0.998 to 1.007 (lg 6 to 22) and the frontend
      geomean 0.9996 and 0.9992 in two runs of 11 rounds. No backend issue needed.
    - Bench, GCC 15.2, 11 rounds, against main 006fa57: geomean of gated new/base 1.0047 (`lc-bench`;
      1.0045 and 1.0085 in two runs with `Polynomial`'s streams still in `detail`), 0.9860
      (`lc-intel`), 0.9922 (`lc-zen5`), 0.9992 (`lc-arm`). On `lc-bench` the excess comes from
      `add`, `sub`, `neg` at lg 10 and 14 (+2% to +4%, IQR within 1.02-1.04), whose code is main's
      byte for byte in the source. Run alone (`--filter "^(add|sub|neg|scale)"`) the same binaries
      give 0.996 to 1.000 for them. Placement again, as in the gate runs above; not fixed.
    - API / kernel, median of the 11 per-round ratios, lg 10, 14, 18, 22 (the runs with the
      streams in `detail`; `Values` runs the same code either way):

      | case | `lc-bench` (avx2) | `lc-intel` (avx512) | `lc-zen5` (zen5) | `lc-arm` (arm) |
      |---|---:|---:|---:|---:|
      | to_values | 1.017, 0.991, 1.002, 1.007 | 1.010, 0.999, 1.003, 1.000 | 1.008, 1.006, 0.994, 1.000 | 1.005, 1.001, 1.000, 1.001 |
      | to_coefficients | 1.048, 1.081, 1.115, 1.189 | 1.035, 1.105, 1.179, 1.186 | 1.043, 1.101, 1.136, 1.169 | 1.026, 1.030, 1.033, 1.036 |
      | add_coefficients | 1.013, 0.990, 1.015, 1.003 | 0.999, 1.001, 0.982, 0.998 | 0.982, 0.987, 0.998, 0.931 | 0.999, 1.004, 0.998, 0.997 |

      `to_coefficients` still pays the copy out of the scratch (#49, part 3). The ns per
      coefficient of the cases with no kernel are within 5% of the table above (`lc-bench`
      values_product 6.86 to 6.92, pointwise conversions 7.55 to 10.02).
  - Gate of 2026-10-12 (PR #50): avx2 1.0096 over 42 rounds, with `multiply/6` (f * g, 32 x 32,
    the NTT) at 1.253, `/7` 1.037, `/8` 1.019, and their `@kernel` cases unmoved. Not placement:
    the same four shapes alone (`--filter "^multiply(@kernel)?/[6789]$"`) gave 1.262, 1.042,
    1.018 and 0.977, and 0.997 to 1.000 with `values(suite)` not called, the rest of the source
    the same. Cause: with `Transform::forward` and `inverse` (avx2) in the program, GCC 15.2
    inlines `product<GroupBottom>` into `Polynomial::convolution` and calls `Tree::visit` out of
    line; without them it calls `Convolution::multiply [clone .part.0]`, which calls
    `Tree<GroupBottom>::tile<4> [clone .isra.0]` directly (`nm`, `objdump` on `lc-amd`).
    `forward_groups`, `forward_trees`, `forward_leaves`, `inverse_leaves` and `inverse_trees`
    become out-of-line functions both paths share. Which of these costs the 35 ns is a guess. A
    program that uses `Values` and `f * g` pays it; the backend can make its codegen not depend
    on that: issue #81.
    - Fix here: the `Values` cases move to `frontend/bench_values.cpp`, a binary of their own,
      and `tools/kernel.py bench` times every `bench*.cpp` of a unit (AGENTS.md says so).
      `frontend/bench.cpp` is main's again. `polynomial.hpp` is unchanged apart from
      `detail::unwritten_polynomial`.
    - Rebased onto main d1f5f4d (after #79, #80). Bench against it, 11 rounds, gated geomean:
      `lc-bench` 1.0024 (`multiply/6` 1.005, all `multiply/<lg>` 0.995 to 1.005), `lc-intel`
      1.0008, `lc-zen5` 1.0012, `lc-arm` 0.9967. API / kernel, lg 10 / 14 / 18 / 22:

      | case | `lc-bench` (avx2) | `lc-intel` (avx512) | `lc-zen5` (zen5) | `lc-arm` (arm) |
      |---|---:|---:|---:|---:|
      | to_values | 0.992, 0.992, 1.002, 1.001 | 0.996, 1.004, 0.999, 1.001 | 0.986, 1.005, 1.000, 1.006 | 0.996, 1.002, 1.000, 1.000 |
      | to_coefficients | 1.010, 1.087, 1.120, 1.182 | 1.015, 1.103, 1.184, 1.178 | 1.024, 1.106, 1.134, 1.159 | 1.025, 1.031, 1.036, 1.043 |
      | add_coefficients | 1.028, 0.996, 1.012, 0.997 | 1.006, 0.998, 0.999, 0.995 | 1.031, 1.010, 1.000, 0.989 | 0.994, 1.004, 0.995, 0.999 |
- 2026-10-11: products (#18): `mul(f, g, n)`, `square(f)`, `product(fs)`, `middle_product(f, g)`
  and `raw::mul<P>(a, b, out)` in `include/fps/products.hpp`, from the prototype's
  `polynomial.hpp`. One engine, `detail::Products<P>` in `polynomial.hpp`, runs them and `f * g`:
  factors cut to n, then a shift (one nonzero term), `small::multiply` where `small::beats_ntt`
  says so (the choice `f * g` makes since #37; no cutoff of the frontend's own), or
  `ntt::Convolution(n, m, a)` with a = the result's own memory (#38). Both kernels write into the
  result, so nothing is copied out.
  - Storage: `detail::Words(n, capacity, Unwritten)` allocates a 64-byte aligned block of
    `length + kPadding` words (`std::aligned_alloc`; the other constructors use `std::malloc`, as
    `new[]` did, so their placement is unchanged). A product's result keeps the block: up to
    2 (n + m - 1) + 16 words for `f * g`, up to 4 n + 16 for `mul` with a short n. A copy is tight.
    The length is checked against 2^25 before the block is allocated. `small::multiply` writes the
    whole product, so `mul` with a short n gets n + m - 1 words for it.
  - `middle_product`: DESIGN.md said "the transposed product" without saying which coefficients.
    Now: coefficients m - 1 .. n - 1 of f g, n = f.size() >= m = g.size() >= 1, else
    `std::domain_error`, as the prototype; with g reversed it is the transposed product. It runs
    all of f g (`small::multiply` or the `Convolution`, by `beats_ntt(n, m)`) in the result's memory
    and moves the n - m + 1 coefficients down in place. A cyclic product of length >= n would halve
    the transform: backend issue #55. (Before #37 merged it had its own direct loop for short sums,
    with the old frontend cutoff; dropped with that cutoff.)
  - `square(f)`: the engine with f twice, so it costs `f * f`. DESIGN.md said "cheaper than
    f * f"; now it says so once the backend squares (#55, one forward transform less).
  - `product(fs)`: split where the sums of sizes come closest to halves (lc-opt's tree splits on
    degrees the same way), each node `f * g`. 1 for no factors, empty if a factor is empty. A
    `std::vector` overload, since P cannot be deduced through the span. lc-opt's tree: #56.
  - `raw::mul`: allocates nothing per call; the NTT path runs `Convolution(n, m)` in the thread's
    region and copies out (a tight output is not in the backend: commented on #38).
    `small::multiply` writes all n + m - 1 words, so a shorter `out` (the usual case: a, b and out
    of n) runs `low_product`, the old frontend schoolbook (blocks of 64 outputs, 64-bit sums),
    computing only the n outputs; a truncated short product is asked of #37. `out` must not overlap
    `a` or `b` (`std::domain_error`, an O(1) check), since both short kernels write as they read.
  - DESIGN.md also: `mul` returns n coefficients; raw functions allocate nothing per call (scratch
    is the library's, per thread); `raw::mul`'s out is apart from a and b.
  - Tests (`test_products.cpp`): `mul` and `raw::mul` on every pair of edge sizes (0, 1,
    2^k +- 1, k <= 10) with outputs 0, 1, 2, the shorter size, the product's size -1, 0, +1, +70,
    3 s + 100 and random; shapes on the NTT path and at the cutoffs; shifts; long factors against
    `f * g`; `square` on edge sizes and all P - 1; `middle_product` on every pair n >= m, at its
    cutoffs and long, against the definition and against f g; `product` of 0 to 100 factors,
    unbalanced, with an empty factor, 2^14 linear factors at random points; results resized and
    copied; 4 threads; every error (`domain_error` for the middle product and overlaps,
    `length_error` past 2^25 for each function). Under ASan the test sets
    `max_malloc_fill_size`, so every allocation starts as 0xbe bytes: with that, dropping the zero
    fill past the transform's length fails on the Mac too (it passed before, on fresh pages).
    Mutations killed: the middle move's offset, the middle loop's reduction cadence, `raw::mul`'s
    zero fill, the shift's tail fill and its truncation, the NTT path's zero fill. Splitting
    `product` always at 1 still passes: balance is speed, not correctness.
  - Bench (`frontend/bench.cpp`): `multiply@kernel` is now `Convolution(n, m, a)` into a new
    `Words` block like the API's result (`convolve`). New pairs at lg 11, 14, 18, 22 (transform
    length 2^lg): `mul` (f, g, n of 2^(lg-1)), `square`, `middle_product` (f of 2^(lg-1), g of
    2^(lg-2)), `raw_mul` (out kept across calls; kernel: the product left in the region);
    `product/<lg>` (2^lg linear factors, lg 10, 14, 18) has no pair. API / kernel, median of 11
    paired rounds (IQR in the PR), GCC 15.2, main 8367228 plus this change:

    | case | `lc-bench` (avx2) | `lc-intel` (avx512) | `lc-zen5` (zen5) | `lc-arm` (arm) |
    |---|---:|---:|---:|---:|
    | multiply/10, /14, /18, /22 | 0.999, 1.000, 1.001, 0.999 | 0.998, 0.988, 1.005, 0.999 | 1.002, 1.001, 1.001, 1.000 | 1.000, 1.000, 1.002, 1.001 |
    | mul/11, /14, /18, /22 | 1.001, 1.000, 0.998, 0.999 | 0.999, 0.999, 1.001, 0.997 | 1.002, 1.001, 1.000, 1.005 | 1.000, 1.000, 0.999, 1.000 |
    | square | 0.998, 1.000, 1.001, 1.002 | 0.999, 1.000, 1.000, 0.999 | 1.000, 1.000, 1.000, 1.002 | 1.000, 1.000, 1.000, 1.000 |
    | middle_product | 0.978, 1.006, 1.007, 1.003 | 1.002, 1.003, 1.005, 1.012 | 1.004, 1.003, 1.006, 1.006 | 1.002, 1.002, 1.003, 1.004 |
    | raw_mul | 1.009, 1.012, 1.022, 1.030 | 1.009, 1.015, 1.018, 1.023 | 1.009, 1.014, 1.015, 1.023 | 1.005, 1.003, 1.004, 1.007 |

    `multiply` against main (same run): 0.95-0.97 from 2^14 on `lc-bench`, 0.92-0.96 on
    `lc-intel`, 0.96-0.98 on `lc-zen5`, 0.98-0.99 on `lc-arm`: the copy out is gone (#16's
    1-6% gap). The two gaps left are copies the backend can remove: the middle product's move
    (#55) and `raw::mul`'s copy out (#38). `product/<lg>`, ns per factor at lg 10 / 14 / 18:
    `lc-bench` 130 / 155 / 189, `lc-intel` 107 / 133 / 169, `lc-zen5` 88 / 103 / 121, `lc-arm`
    118 / 170 / 236; lc-opt's tree did 500000 linear factors in about 20 ms on `lc-amd`
    (40 ns per factor, its notes).
- 2026-10-12: products (#18) rebased on main bfd491b and moved to `ntt::multiply(f, g, out)`
  (backend #38 round 2, PR #79): it reads the factors from the caller and writes f g mod
  x^out.size() into the caller's memory, so the frontend no longer needs `Convolution(n, m, a)`.
  - `f * g`, `mul`, `square`, `product` and `raw::mul` call it into the result (or `out`) of exactly
    n words; the NTT path keeps no spare words and `raw::mul` copies nothing out. `middle_product`
    calls it into n words (f g mod x^n, the same transform length as the whole product) and moves
    the n - m + 1 coefficients down. The choice between `small::multiply` and the NTT is
    `small::beats_ntt`, as for `f * g` since #37; the frontend has no cutoff of its own.
  - `detail::Words(n, capacity, Unwritten)` is now a plain block of max(n, capacity) words
    (`make_unique_for_overwrite`), no longer 64-byte aligned with `kPadding` words: only
    `small::multiply` with an output cut short (`mul`) and the middle product keep spare words.
    `checked_length` is gone: `ntt::multiply` throws `std::length_error` before it writes.
  - DESIGN.md (Conventions): the `std::length_error` line now says that a short factor next to a
    longer one runs `small::multiply` where that is faster, which has no limit (#37's behaviour:
    2^25 x 2 returns; 2^25 x 4096 throws). The test checks both.
  - Bench: every `@kernel` case of a product is now `ntt::multiply` into new memory of the
    result's size (`ntt_product`), or into the same kept `out` for `raw_mul`. Tests pass at -O2
    and ASan/UBSan on the Mac (scalar, arm), `lc-amd` (scalar, sse, avx2), `lc-intel-test`,
    `lc-zen5-test` and `lc-arm`. A mutation (the middle product's NTT output one word short)
    fails 234 checks.
  - API / kernel, median of 11 paired rounds (IQR in brackets where wider than 0.005), GCC 15.2,
    `kernel.py bench frontend --base bfd491b --filter 'mul|square|middle|product'`:

    | case | `lc-bench` (avx2) | `lc-intel` (avx512) | `lc-zen5` (zen5) | `lc-arm` (arm) |
    |---|---:|---:|---:|---:|
    | multiply/10, /14, /18, /22 | 1.005, 1.001, 1.005, 1.003 | 1.007, 1.001, 1.001, 1.000 | 1.006, 1.000, 0.999, 1.000 | 1.001, 1.000, 1.000, 1.000 |
    | mul/11, /14, /18, /22 | 0.998, 1.001, 1.000, 1.001 | 1.003, 1.000, 1.000, 1.000 | 1.004, 0.999, 1.000, 1.002 | 1.000, 1.000, 1.000, 1.001 |
    | square | 0.999, 1.000, 1.000, 0.997 | 1.002, 1.000, 1.000, 1.002 | 1.012, 1.000, 1.000, 1.000 | 1.001, 1.000, 1.000, 1.000 |
    | middle_product | 1.001, 1.007, 1.009, 1.013 | 1.004, 1.005, 1.014, 1.015 | 1.008, 1.004, 1.011, 1.023 | 1.003, 1.003, 1.004, 1.005 |
    | raw_mul | 1.004, 1.000, 1.003, 1.001 | 1.003, 1.000, 1.000, 1.000 | 1.005, 1.001, 1.000, 1.000 | 1.001, 1.000, 1.000, 0.999 |
    | raw_mul/16x16, /24x24 | 2.245, 2.082 | 2.150, 2.224 | 2.458, 3.007 | 1.176, 0.996 |
    | multiply/16x16 | 1.106 | 1.096 | 1.134 | 1.074 |

    - `mul`, `square` and `raw_mul` are within 0.5% of `ntt::multiply` (one case, zen5
      square/11, 1.2%). `raw_mul` was 1.003-1.030 with the copy out; that gap is gone.
    - `middle_product`: 0.3% to 2.3%, growing with size: the move of n - m + 1 words down. Only
      a cyclic product written at out[0] can remove it (#55).
    - `multiply/16` came out 1.027 (`lc-bench`) and 1.058 (`lc-intel`) in this run; alone, 21
      rounds, 0.999 and 1.003 (lg 15 and 17 also 0.994-1.001). Placement of the run, not work.
    - `raw_mul/<n>x<n>`: `raw::mul` into n words runs the frontend's truncated loop
      (`low_product`) against `small::multiply`'s whole product: 2.1-3.0x on x86 (40-124 ns),
      1.0-1.2x on Neoverse V2. Asked of #37 (a short product cut to out.size()).
    - `multiply/16x16`: 1.07-1.13 (40-92 ns), the scans for a single term and `beats_ntt` before
      `small::multiply`; 4096x4 and 4096x16 are 1.001-1.008.
  - `multiply/<lg>` against main bfd491b (which ran `Convolution(n, m, a)`), same run:
    0.92-0.99 from 2^10 on every target (lg 22: 0.924, 0.946, 0.996, 0.980), except lg 16 on
    `lc-intel`, 1.020-1.022 in both runs. At 2^6-2^9, where `beats_ntt` still picks the NTT,
    `ntt::multiply` is slower than the Convolution it replaces: lg 6 / 7 / 8 / 9 = 1.353 / 1.050
    / 1.074 / 0.973 (`lc-bench`), 1.032 / 1.060 / 1.017 / 1.011 (`lc-intel`), 1.017 / 0.995 /
    0.980 / 1.007 (`lc-zen5`), 1.078 / 1.007 / 0.986 / 0.986 (`lc-arm`). PR #79 measured up to 10%
    at 2^6-2^8. `small`'s tuner times the NTT as a `Convolution(n, m, out)`, so `beats_ntt` is
    now fitted to the wrong NTT: noted on #37 and #38.
  - `product/<lg>`, ns per factor at lg 10 / 14 / 18: `lc-bench` 84 / 109 / 141, `lc-intel`
    63 / 86 / 117, `lc-zen5` 48 / 59 / 76, `lc-arm` 97 / 141 / 201 (were 130 / 155 / 189 on
    `lc-bench` before #37's tuned `small` and #79).
  - Review of #59, round 1 (two findings), and what changed:
    1. `test_results` assumed that NTT results keep spare words, which stopped being true with
       `ntt::multiply`. It now copies, resizes and adds to the results that do keep them:
       `mul` of 16 x 16 into 20 (`small::multiply` writes all 31 words) and middle products on
       both paths (3000 by 1000 on the NTT, 3000 by 16 on `small`). The test asserts each path
       with `beats_ntt`. A `resize` that reuses spare words without zeroing them fails it.
    2. `product` copied every factor at the leaves (`return fs[0]` for a side of one). A side of
       one factor is now multiplied by reference. `product/<lg>` against the PR's previous head
       6001cc7, `lc-bench`, 11 rounds: lg 10 / 14 / 18 = 0.803 / 0.847 / 0.885 (IQR within
       0.006). Now 69 / 93 / 127 ns per factor.
- 2026-10-12: tuning (#66): `fps::tune_all()`, `fps::save_wisdom(path)`, `fps::load_wisdom(path)`
  in `include/fps/tune.hpp`, on backend/tune's `run`, `save` and `load` (#54). The header includes
  `backend/small/tune.hpp` and `backend/factorials/tune.hpp`, so "small.variant", "small.ntt" and
  "factorials.variant" register; ntt's goes there when #78 adds it.
  - The name: DESIGN.md had `fps::tune()`, which does not compile: `fps::tune` is the backend
    tuner's namespace ("redefinition of 'tune' as different kind of symbol"). Renaming the
    namespace (17 backend files, three running backend rounds on them: #78, #70, #37) was the
    other way; `tune_all()` keeps this change in the frontend. DESIGN.md says so.
  - Behaviour, as DESIGN.md now states it: both wisdom functions return bool and never throw on a
    bad path (`save_wisdom` false if the file cannot be written; `load_wisdom` false, setting
    nothing, if it is missing, malformed, or from another CPU or target). `tune_all()` runs once
    per process and measures from the compiled defaults. `-DFPS_NO_TUNER`: the three compile;
    `tune_all()` does nothing, the other two touch no file and return false, so the usual
    `if (!load_wisdom(p)) { tune_all(); save_wisdom(p); }` still runs. The kernel tune headers
    are not included then.
  - Tests (`test_tune.cpp`): its own wisdom loads back; 17 refused files (empty, a cut header, no
    target line, another format, CPU or target (each of the other five), a line without values,
    too few values, not a number, a negative NTT time, a variant past the list, another candidate
    list, good lines before a bad one) and a missing file each leave the saved wisdom byte-identical;
    unwritable paths (missing directory, a directory, "") return false and create nothing;
    wisdom forcing each candidate of each choice everywhere it can run (every small variant, the
    NTT times x1/8 and x8, every factorials variant) loads, saves back identical, and f * g (66
    shapes on both sides of every target's cutoffs, up to 4096 x 4096) and Factorials (every
    class end to 2^17, and one past) match O(n^2) references; then `tune_all()` while another
    thread checks the same results, a second `tune_all()` changes nothing, and the defaults and
    the tuned wisdom load back in turn. `test_tune_no_tuner.cpp`: the -DFPS_NO_TUNER behaviour
    above, f * g and Factorials. Both pass at -O2 and ASan/UBSan on every target's test machine
    and on the Mac (scalar, arm). Mutations: `load_wisdom` returning true fails the test; dropping
    factorials' tune header fails to compile the test.
  - Cost of `tune_all()`, one fresh process, ms (3 runs; lines of 3 that changed): `lc-bench`
    (Zen 3, avx2) 119-130 (2-3), `lc-intel` (avx512) 118-121 (3), `lc-zen5` 105-106 (3), `lc-arm`
    159 (1); the Mac (M4, arm build, laptop) 217. DESIGN.md said "seconds"; now "about 0.1 s".
  - A call with loaded wisdom against the compiled defaults, the same choices (save the defaults,
    load them back: the tables are then heap copies). `wisdom_cost.cpp` (explore dir
    `build-66/`): processes alternating on the last core under the bench lock, bench.hpp
    timing, GCC 15.2, target flags, this branch on d1f5f4d. Paired wisdom / defaults, median of 7
    (min-max):

    | case | `lc-bench` | `lc-intel` | `lc-zen5` | `lc-arm` |
    |---|---:|---:|---:|---:|
    | multiply 8x8 / 16x16 / 32x32 | 1.000 / 0.997 / 0.996 | 0.997 / 1.000 / 0.997 | 1.000 / 0.987 / 1.000 | 0.998 / 1.001 / 1.000 |
    | multiply 64x64 / 4096x16 | 1.003 / 1.006 | 1.020 / 1.000 | 1.005 / 1.001 | 1.000 / 1.001 |
    | multiply 512x512 / 8192x8192 | 0.995 / 0.998 | 1.000 / 0.997 | 0.995 / 0.999 | 1.001 / 1.000 |
    | Factorials 64 / 1024 / 16384 | 0.939 / 1.023 / 1.001 | 1.082 / 0.975 / 1.001 | 1.058 / 0.997 / 0.999 | 1.001 / 1.000 / 1.000 |

    The Factorials 64 and 1024 shifts (narrow, both signs) are heap placement, not the tables: a
    control process doing the same file work with a failed load (tables stay the defaults), 9
    rounds of three: wisdom / control for Factorials 64 1.002 (Zen 3), 0.940 (Emerald Rapids; that
    run wisdom / defaults 1.000, control / defaults 1.065), 1.003 (Zen 5); 1024 1.007, 1.012, 1.010;
    64x64 1.000, 1.004, 1.000. So a loaded table costs nothing measurable beyond the choice it
    holds.
  - Bench: no new pair (the three are no kernel wrappers; tune_all() is a measurement). The
    existing pairs, new against d1f5f4d, 11 rounds: geomean of gated cases `lc-bench` 1.0073 and
    1.0048 (two runs), `lc-intel` 0.9856, `lc-zen5` 0.9921, `lc-arm` 0.9933. No kernel or
    frontend operation changed; the moves are layout and placement (the tune headers add code and
    allocate the registry at start-up). Their `@kernel` cases, code in bench.cpp that this change
    does not touch, move with them: on `lc-bench` add/sub/neg at lg 10 and 14 read 1.026-1.037,
    and add@kernel/14 1.076, sub@kernel/14 1.082, multiply/7 1.059 with multiply@kernel/7 1.029;
    on `lc-zen5` multiply/10 1.039 with multiply@kernel/10 1.041.
  - API / kernel (new ns medians, run 2 on `lc-bench`): factorials 1.000-1.012 on all four
    machines (`lc-intel` /10 1.019, `lc-zen5` /10 1.032); multiply from lg 10 1.03-1.07 on x86
    and 1.01-1.02 on `lc-arm`, the known copy out of the Convolution (above; measured on d1f5f4d,
    before #59 moved f * g onto `ntt::multiply(f, g, out)`); multiply 4096x4 and 4096x16
    1.00-1.01; element-wise 0.59-1.03, placement as before.
  - `polynomial.hpp`'s cutoff comment: the issue asked for 38-180 / 24-104; #59 rewrote that
    comment with its own ranges first, so this change keeps #59's.
- 2026-10-12: `product(fs)` runs the backend kernel `product_tree::product` (issue #56) into the
  result's memory, through `product(items, out, proj)` with `proj = f.span()`: no array of spans
  (4 MB at 2^18, fresh pages each call). `product/<lg>` against main a34ce14, 7 rounds,
  new/base at lg 10 / 14 / 18: `lc-bench` 0.309 / 0.309 / 0.322 (21 / 29 / 41 ns per factor),
  `lc-intel` 0.395 / 0.364 / 0.361, `lc-zen5` 0.355 / 0.329 / 0.333, `lc-arm` 0.372 / 0.353 / 0.348
  (5 rounds).
  New pair `product@kernel/<lg>`: the kernel on spans built in the setup, into new memory. API /
  kernel, `lc-bench`, 5 rounds: 1.005 / 1.002 / 1.003.
- 2026-10-12: series (#20): `derivative`, `integral`, `inverse(f, n)`, `log(f, n)`, `exp(f, n)` (n
  defaults to `f.size()`) and `raw::inverse`, `raw::log`, `raw::exp` in `include/fps/series.hpp`.
  The prototype's `series.hpp` wraps its private copy of lc-opt (`include/fps/lcopt/`), which stays
  there; the algorithms here are written plainly, and lc-opt's tuned ones are backend issue #87
  (searched first: no open issue covered series).
  - Internals for #21 (division), #22 (pow, sqrt) and #23: `detail::series::{inverse, log, exp}(f,
    out, scratch)` on spans, writing into the caller's `out`, scratch passed in (n, 2 n and 3 n
    words). The raw functions take it from one per-thread `Words` kept across calls.
  - Newton on `raw::mul` (shift, `small::multiply` or `ntt::multiply`, as `small::beats_ntt` picks;
    no cutoff of the frontend's own), precision doubling from 1. inverse: two products a step,
    e = f g mod x^2k, then g[k, 2k) = -(g e[k, 2k)) mod x^k. log f = integral of f' (1/f) mod
    x^(n-1). exp: g[k, 2k) = g (f - log g)[k, 2k) mod x^k, a whole log of g per step.
  - Few terms: f mod x^n with at most 16 nonzero terms past f[0] (lc-opt's and the prototype's
    `kSparseTaps`) runs a recurrence with one tap per term, O(n t), all three functions;
    `inverse(1 - x, n)` is O(n). 64-bit sums of at most 16 products below P^2 < 2^60.
  - Reciprocals 1 / i (log, exp, integral) from the backend's `factorials::tables`: (i - 1)! / i!.
    A per-call linear recurrence (r[i] = -(P / i) r[P mod i]) would divide by a variable i per
    entry.
  - `raw::exp`: out may be f (step k reads f[k, 2k) before it writes there; words of out past
    f.size() are not read as f), else apart (`std::domain_error`). `raw::log`: any overlap, f is
    read first. `raw::inverse`: apart.
  - DESIGN.md: the result sizes of `derivative` (f.size() - 1) and `integral` (f.size() + 1), and
    `f.size() < P` for `integral` and `n <= P` for `log` and `exp` (they divide by 1 .. n - 1; a
    sparse f has no 2^25 limit to stop it first); the precondition on f[0] applies for n > 0
    only, since only f mod x^n is read; `raw::inverse` and `raw::log` listed with their overlap
    rules, as "same names in fps::raw" implies.
  - Tests (`test_series.cpp`): every function and raw form against O(n^2) recurrences on sizes 0,
    1, 2^k +- 1 (k <= 10), f of n - 1, n, n + 2 coefficients, random, mostly zero, all P - 1 and
    1 to 20 terms; exactly 0, 1, 2, 15, 16, 17, 18 terms, near index 0 and spread (both paths);
    closed forms at n = 10^6 (1/(1 - x), log(1/(1 - x)), log(1 - x), exp(x), exp(5 x^3));
    identities at 2^14 +- 1, 2^17, 300001 (f inverse(f) = 1, log exp, exp log) and on the
    recurrence at 200000; aliasing (exp in place, also with f shorter than out; log shifted both
    ways); outputs pre-filled with P - 1; 4 threads; every error, including n > P and
    f.size() >= P on 4 GB of unwritten memory. 15 mutations: 14 fail the test (one by timeout:
    the dropped n > P check runs exp over 4 GB); a sparse bound of 15 terms is no bug.
    Pass at -O2 and ASan/UBSan on the Mac (scalar, arm), `lc-amd` (scalar, sse, avx2),
    `lc-intel-test`, `lc-zen5-test` and `lc-arm`.
  - Bench (`frontend/bench_series.cpp`, its own binary): `inverse/<lg>`, `log/<lg>`, `exp/<lg>` on
    n = 2^(lg-1) random coefficients, as `mul/<lg>`; `raw_exp/14`, `/18`; `inverse_sparse`,
    `log_sparse`, `exp_sparse` (1 - x - x^2, x + x^2/2 + 3 x^5; n = 2^lg); `derivative` and
    `integral` (n = 2^lg). No `@kernel` pairs: there is no series kernel (#87). Instead, each
    Newton case per round over `mul@kernel/<lg>` (`ntt::multiply` of two factors of n into n
    words, bench.cpp, same run), median of 11 rounds (IQR within 0.01 except zen5 lg 11, 0.1),
    GCC 15.2, base a34ce14 plus this change:

    | lg | `lc-bench` (avx2) inverse / log / exp | `lc-intel` (avx512) | `lc-zen5` (zen5) | `lc-arm` (arm) |
    |---:|---|---|---|---|
    | 11 | 3.03 / 4.98 / 10.58 | 2.96 / 4.93 / 10.19 | 3.90 / 6.33 / 15.34 | 2.73 / 4.29 / 8.10 |
    | 14 | 2.79 / 4.53 / 8.63 | 2.74 / 4.44 / 8.34 | 2.87 / 4.75 / 9.77 | 2.75 / 4.17 / 7.77 |
    | 18 | 2.75 / 4.27 / 7.89 | 2.64 / 4.12 / 7.58 | 2.76 / 4.44 / 8.46 | 2.79 / 4.12 / 7.71 |
    | 22 | 2.85 / 4.31 / 8.21 | 2.80 / 4.17 / 7.90 | 2.70 / 4.24 / 7.89 | 2.81 / 4.09 / 7.67 |

    ns per coefficient at lg 22 (n = 2^21), inverse / log / exp: `lc-bench` 26.0 / 39.2 / 74.7,
    `lc-intel` 24.9 / 37.1 / 70.3, `lc-zen5` 12.3 / 19.4 / 36.0, `lc-arm` 49.1 / 71.4 / 134.1.
    lc-opt's counts (`lib/poly/notes.md`) give about 1.7 / 2 / 2.6 products (guess), so #87 is
    worth 1.6x (inverse) to 3x (exp).
    - `exp/<lg>` over `raw_exp/<lg>` (result allocated against a kept out): 0.999-1.001.
    - Recurrences, ns per coefficient at n = 2^14 / 2^18 / 2^22, inverse / log / exp sparse:
      `lc-bench` 4.35 / 7.0 / 9.3, `lc-intel` 4.8 / 7.0 / 8.8, `lc-zen5` 3.7 / 5.4 / 7.5,
      `lc-arm` 6.4 / 9.6 / 12.3 (flat in n).
    - `derivative` (a scalar loop, `% P` per word) 1.45 / 1.21 / 0.90 / 1.62 ns per coefficient;
      `integral` (factorial tables, then two products per word) 2.3-2.5 / 1.9 / 1.4 / 2.8-3.0
      (`lc-bench`, `lc-intel`, `lc-zen5`, `lc-arm`).
    - The existing cases against main: geomean of the gated cases 0.9994 (`lc-bench`), 1.0001
      (`lc-intel`), 0.9765 (`lc-zen5`), 0.9990 (`lc-arm`). On `lc-zen5` lg 11 products came out
      faster (`square`, `raw_mul`, `mul`, `multiply` /11: 0.82-0.83) with `series.hpp` in
      bench.cpp's program; codegen, as #81 (guess, not inspected). Elsewhere 0.955 (`add/22`,
      `lc-arm`) to 1.035 (`neg/22`, `lc-arm`): placement, as in earlier entries.
- 2026-10-12: issue #81 (round 1): the 25% `multiply/6` slowdown with `Values` in `bench.cpp`
  (the PR #50 entry above) is code layout, not the NTT's inlining. With every function of the
  product compiled identically in both programs the gap stayed (144 -> 175 ns on Zen 3), and
  shifting the whole program's code moves 2^6 products by up to 25% (`backend/ntt/notes.md`,
  #93). `bench_values.cpp` stays a binary of its own; its comment says why.
- 2026-10-12: `square` and `middle_product` on the backend's new kernels (backend issue #55,
  `backend/ntt/notes.md`). `detail::Products::square`: as `multiply(a, a, 2n - 1)` (shift,
  `small::multiply`), with `ntt::square` (one forward transform) on the NTT path.
  `detail::Products::middle`: `ntt::middle_product` into the result on the NTT path, a cyclic
  product of length >= n with the result at word 0: no move, no spare words; the
  `small::multiply` path keeps its move. The NTT path now takes f up to 2^25 words, so the length
  error test uses 2^25 + 1. `bench.cpp`: `square@kernel` and `middle_product@kernel` call the new
  kernels into new memory (`ntt_square`, `ntt_middle`). Against main a536462, 7 rounds, GCC 15.2,
  new/base at lg 11, 14, 18, 22 (f of 2^(lg-1); g of 2^(lg-2)):

  | case | `lc-bench` (avx2) | `lc-intel` (avx512) | `lc-zen5` (zen5) | `lc-arm` (arm) |
  |---|---:|---:|---:|---:|
  | square | 0.786, 0.749, 0.730, 0.709 | 0.799, 0.780, 0.731, 0.724 | 0.821, 0.791, 0.756, 0.700 | 0.772, 0.756, 0.735, 0.725 |
  | middle_product | 0.502, 0.490, 0.481, 0.494 | 0.501, 0.484, 0.456, 0.492 | 0.524, 0.491, 0.495, 0.451 | 0.482, 0.484, 0.484, 0.484 |

  API / kernel (same runs, new medians): square 0.99-1.01 (`lc-intel` /22 0.96: the kernel case
  the slower), middle_product 0.99-1.01, on all four.

- 2026-10-12: `Values` on the backend's leaf-form ops (backend issue #49; `backend/ntt/notes.md`,
  Leaf-form ops). Changed in `values.hpp` only:
  - A * B and `multiply_add` in the automatic layout: `ntt::Transform::multiply`/`multiply_add`
    (the scalar `leaf_product` loop is gone). The pointwise layout's conversions:
    `Transform::forward(w, Form::values)` and the inverse with `Form::values` (`split_leaf`,
    `join_leaf` and the rho tables are gone; the words' order inside a block of 8 leaves is the
    backend's, opaque as before). `to_coefficients` and the raw functions: the backend's inverse
    into other memory (`inverse(A, out, form, {offset, add})`), A kept; L < 64 still goes
    through the scratch and folds. `to_coefficients(Values&&)` is the const form now (it ran the
    inverse in A's memory, then copied out). The pointwise products stay the frontend's stream.
  - `Plan` is gone: a `Transform` per length (`detail::values::transform`), which also holds the
    values form's tables (L / 16 words), about 5 L / 16 words in all, against L / 4 + 5 L / 16.
  - `kernel.py bench frontend --base d16ca08`, 5 rounds, new/base at lg 10 / 14 / 18 / 22:
    values_product 0.14 (Zen 3), 0.19 (Emerald Rapids), 0.14 (Zen 5), 0.29 (Neoverse V2) at
    every lg; to_values_pointwise 0.13-0.22, 0.21-0.35, 0.13-0.19, 0.25-0.38;
    to_coefficients_pointwise 0.11-0.18, 0.17-0.28, 0.10-0.15, 0.20-0.33; to_coefficients 0.88-1.01,
    0.92-1.08 (1.08 at 2^14), 0.94-1.06 (1.06 at 2^22), 0.97-0.98; add_coefficients 0.98-1.04,
    0.92-1.07, 0.98-1.10, 1.00-1.03 (the 2^10 losses ran the region path, which now copies on
    the stack to 2^10). Geomean of the gated cases 0.534, 0.603, 0.517, 0.659.
  - `bench_values.cpp`: @kernel pairs for every op but the pointwise products;
    `to_coefficients@kernel` is the inverse into other memory, `to_coefficients@copy` what the
    API did before.
  - `multiply/6` (f * g, 32 x 32, bench.cpp) 1.13 on Zen 3 (152 -> 172 ns) with its @kernel at
    1.00, 1.00 elsewhere: code placement (#81, #93).

- 2026-10-12: division (#21): `divmod(f, g)`, `f / g`, `f % g`, `/=`, `%=` in
  `include/fps/division.hpp`, from the prototype's `division.hpp`, which wrapped its private copy
  of lc-opt's blocked division. The backend has no division kernel: moving lc-opt's is backend
  issue #102 (opened by the coordinator; the measurements below are on it too).
  - Sizes, as the prototype: q has max(n - d, 0) coefficients and r has d = deg g (f's, then
    zeros, when n <= d), so `f / x^k` drops the lowest k coefficients and `f % x^k` is
    `f.truncate(k)`, as DESIGN.md's table says. g = 0 of any size throws `std::domain_error`.
    DESIGN.md did not state the sizes or the error; now it does (Functions, after
    `middle_product`).
  - Paths. The divisor is scanned once (`detail::division::divisor`: deg g, and whether g[0, d)
    is zero). g = c x^d: q = f[d, n) / c (a copy for c = 1), r = f mod x^d, O(n). Else, with
    G = x^d g(1/x), F = x^(n-1) f(1/x), k = n - d: rev q = F / G mod x^k, by the recurrence
    G[0] Q[i] = F[i] - sum_j G[j] Q[i - j] when G mod x^k has at most 16 terms past G[0]
    (`series::sparse` and `tap_sum`, as for inverse: short divisors, quotients of <= 17
    coefficients, sparse tops such as x^d - x - 1), else `series::inverse` (Newton) and one
    product on the `raw::mul` paths; r = f - (q g mod x^d), one more product. `f / g` computes
    no remainder; `f % g` with a monomial g or n <= d computes no quotient. Scratch: 3 k words of
    `series::scratch`.
  - Tests (`test_division.cpp`): every pair of edge sizes (0, 1, 2^k +- 1, k <= 10) twice, with
    f random, mostly zero, all P - 1 or of few terms, and g random, mostly zero, all P - 1, of few
    terms, or with trailing zeros, against long division, through all five forms, after
    `dirty_heap`, with ASan's `max_malloc_fill_size` as in `test_products.cpp`; monomials c x^k
    for c = 1, 2, P - 1, random and k = 0, 1, 5, n - 1, n, n + 3, and `f / Poly::x(k)`,
    `f % Poly::x(k)`; constants; G mod x^k with exactly 0, 1, 2, 15, 16, 17, 18 terms, near the
    top and spread (both paths); x^d - x - 1 and 1 - x^d / 2; f = q g + r exactly at 7 large
    shapes (2^17 + 1 by 2^16 up to 500000 by 277012; lc-opt's n_max_02 shape, 500000 by 53336;
    2^20 by 4) and two sparse ones (10^6 by x^500000 - x - 1); aliasing (`f /= f`, `f %= f`,
    `divmod(f, f)`); 4 threads; every error, and `std::length_error` from the remainder's product
    at d = 2^25 (the quotient, by the recurrence, still returns). Pass at -O2 and ASan/UBSan on
    the Mac (scalar, arm), `lc-amd` (scalar, sse, avx2), `lc-intel-test`, `lc-zen5-test` and
    `lc-arm`. 10 mutations (the scale's c = 1 test, the recurrence's subtraction, F's offset, the
    remainder's subtraction length, its zero fill, the monomial scan's bound, the final reverse,
    a dropped coefficient of G, G one word short, the unscaled taps) all fail it; the zero fill
    only under ASan's fill on the Mac.
  - Bench (`frontend/bench_division.cpp`, its own binary, as `bench_series.cpp`): `divmod`,
    `divide` (f / g), `modulo` (f % g) for f of 2^lg by g of 2^(lg-1), and `divmod_long` by g of
    2^(lg-6), the shapes of #102; `divmod_short` (by x^3 + 2x + 5), `divide_monomial` and
    `modulo_monomial` (by x^(n/2)) for f of 2^lg. No `@kernel` pairs: no division kernel. Each
    case per round over `mul@kernel/<lg>` (`ntt::multiply` of two factors of 2^(lg-1) into
    2^(lg-1) words, bench.cpp, same run), median of 7 rounds (IQR within 0.02; zen5 lg 11 within
    0.2), GCC 15.2, main ad1e1b4 plus this change, lg 11 / 14 / 18 / 22:

    | case | `lc-bench` (avx2) | `lc-intel` (avx512) | `lc-zen5` (zen5) | `lc-arm` (arm) |
    |---|---|---|---|---|
    | divmod | 7.35 / 7.25 / 6.95 / 7.15 | 7.41 / 7.16 / 6.86 / 7.05 | 8.54 / 7.48 / 7.23 / 6.94 | 6.95 / 6.98 / 7.03 / 7.00 |
    | divide | 6.29 / 6.18 / 5.93 / 6.14 | 6.35 / 6.10 / 5.86 / 5.98 | 7.44 / 6.42 / 6.18 / 5.95 | 5.92 / 5.96 / 6.01 / 5.99 |
    | modulo | 7.33 / 7.25 / 6.95 / 7.15 | 7.39 / 7.08 / 6.90 / 7.06 | 8.55 / 7.48 / 7.25 / 6.96 | 6.94 / 6.98 / 7.03 / 6.99 |
    | divmod_long (lg 14 / 18 / 22) | 6.28 / 6.02 / 6.33 | 6.18 / 6.08 / 6.24 | 6.56 / 6.33 / 6.10 | 6.06 / 6.12 / 6.07 |

    ns per coefficient of f at lg 22, divmod / divide: `lc-bench` 32.9 / 28.2, `lc-intel` 31.4 /
    26.7, `lc-zen5` 16.8 / 14.4, `lc-arm` 60.8 / 52.0. lc-opt's division did k = 446665,
    d = 53335 in about 6.2 ms on `lc-amd` (its notes); here 2^22 by 2^21 takes 138 ms on Zen 3.
    - Where the time goes (a count, not profiled): the quotient has k = 2^(lg-1) + 1
      coefficients, one past a power of two, so the Newton inverse takes one more step whose
      first product runs a transform of 2^lg, and F (1 / G) mod x^k one of 2^(lg+1): about 2.4
      and 2 units, and the remainder 1 more. lc-opt's blocks avoid both. Not tuned here (#102).
    - `divmod_short` (the recurrence, 2 taps), ns per coefficient of f: 4.17 (Zen 3), 2.88-2.91
      (Emerald Rapids), 3.22-3.24 (Zen 5), 4.04-4.05 (Neoverse V2), flat in n: a `% P` and a
      `mod_mul` per coefficient, scalar (#87, #102).
    - `divide_monomial`, `modulo_monomial`: 0.15-0.26 ns per coefficient of f (a scan of n / 2
      zeros and a copy of n / 2 words).
  - The other bench binaries: `fps.hpp` now includes `division.hpp`, which nothing in them
    instantiates. GCC's `bench.cpp`, `bench_series.cpp` and `bench_values.cpp` are not
    byte-identical to main's, but `objdump -d` of `bench.cpp` on `lc-amd` (avx2) differs only in
    rip-relative offsets of constants (32 bytes later in .rodata); the instruction addresses are
    the same. Clang's assembly of `bench.cpp` is identical. `mul@kernel/<lg>` against main in the
    same runs: 0.999-1.005.

- 2026-10-12: points (#24): `evaluate(f, points)`, `interpolate<P>(points, values)`,
  `to_newton_basis(f, points)` on any span of points, in `include/fps/points.hpp`. The prototype's
  `points.hpp` wraps its private lc-opt (`include/fps/lcopt/points.hpp`); here the algorithms are
  written plainly on product trees and #21's division. lc-opt's tuned versions are backend issue
  #104 (opened by the backend coordinator).
  - Tree: node [l, r) holds prod (x - x_i), split at (l + r) / 2, heap-indexed; a node of <= 32
    points is a leaf multiplied out by a loop, the others are `f * g` of their children (small or
    NTT, as `small::beats_ntt` picks). The backend's `product_tree::product` returns only the root,
    and these need every node, so it is not used: keeping the nodes is #104's (its brief says to
    add tree API to `backend/product_tree` if needed).
  - evaluate: down the tree, f mod T per node (`detail::division::modulo`, a new span form of `%`
    that `operator%` now calls; skipped while f is shorter than T, so few coefficients at many
    points divide only near the leaves), then Horner on the leaf's remainder with its points side
    by side (independent chains). interpolate: c_i = y_i / M'(x_i) by the same descent and one
    inversion, then up: leaf sums of c_i T / (x - x_i) by synthetic division, nodes f_L T_R + f_R T_L.
    A zero M'(x_i) means a repeated point: `std::domain_error`, at no extra cost, so DESIGN.md's
    "distinct points" is checked. to_newton_basis: g = (g mod T_L) + T_L (g / T_L) per node (one
    divmod), synthetic division by x - x_i at the leaves; the tree has n points, the last (never
    read) set to 0.
  - DESIGN.md: one paragraph on sizes, preconditions and errors, which it did not state: m
    coefficients from interpolate; to_newton_basis reads the first n - 1 points; points or values
    >= P throw `std::domain_error`; 2^25 points or more throw `std::length_error` before any work
    (the root would pass the NTT's 2^25 coefficients; without the check the tree was built, about
    2.7 GB, before the root's product threw).
  - Tests (`test_points.cpp`): evaluate on every pair of edge sizes (0, 1, 2^k +- 1, k <= 10) for n
    and m, random, all P - 1 and mostly-zero f, random and repeated points, all-zero points (nodes
    x^k: the monomial division) and P - 1, against Horner; long f by 5 and 33 points, 3 and 40
    coefficients by 2^16 points, and 2^15 to 2^17 both ways at 300 random points; interpolate on
    every edge size, random and 0 .. m / P - 1 .. points, against Lagrange (O(m^2)), at 2^14 and
    2^17 + 1 back through evaluate and Horner, and a polynomial of degree < m back exactly;
    to_newton_basis on every edge size, random, repeated and zero points, with a spare point,
    against expanding the basis, and at 2^15 + 1 and 100000 at 20 points; 4 threads; every
    error, repeats next to each other, in one leaf, across leaves, all equal. 14 mutations: 9 fail
    it; 5 are equivalent (a monic T's top word; the zero fill of outputs that start zeroed; extra
    zero words in the leaf loop; a dropped repeat check, which `mod_inverse(0)` then reports as
    the same `domain_error`). A precondition bound one point short only fails under ASan, so the
    test also asks n = 2 with no points. Pass at -O2 and ASan/UBSan on the Mac (scalar, arm),
    `lc-amd` (scalar, sse, avx2), `lc-intel-test`, `lc-zen5-test` and `lc-arm`.
  - Bench (`frontend/bench_points.cpp`, its own binary as for series and division): `evaluate`,
    `interpolate`, `newton` for n = m = 2^lg, lg 10, 14, 17, random points (distinct for
    interpolate). No `@kernel` pairs: no backend call is wrapped (#104). Median of 7 rounds, GCC
    15.2, ns per point, lg 10 / 14 / 17:

    | case | `lc-bench` (avx2) | `lc-intel` (avx512) | `lc-zen5` (zen5) | `lc-arm` (arm) |
    |---|---|---|---|---|
    | evaluate | 214 / 338 / 460 | 179 / 289 / 397 | 175 / 239 / 303 | 266 / 495 / 720 |
    | interpolate | 370 / 522 / 664 | 303 / 444 / 566 | 286 / 365 / 443 | 446 / 722 / 987 |
    | newton | 152 / 226 / 296 | 120 / 186 / 249 | 119 / 157 / 194 | 189 / 326 / 459 |

    - Where the time goes at 2^17 (a phase timer, not committed; best of 7, the same
      build): the tree alone 13.7 ms of evaluate's 60.0 on `lc-bench` (10.9 of 52.5 `lc-intel`,
      6.7 of 39.2 `lc-zen5`, 19.1 of 94.1 `lc-arm`): the descent is 78-83%. Each node's division
      runs its own Newton inverse of the reversed child and two products (fps/division.hpp,
      about 7 units of a product of its size); lc-opt's transposed descent needs no division.
    - Against lc-opt (#104's brief, `lc-bench`, n = 2^17): its Newton basis takes about 10.4 ms
      (trees 4.45, division 2.06, descent 3.9); `newton/17` here 38.8 ms, 3.7x; the tree 13.7 ms
      against 4.45, 3.1x.
  - The other bench binaries against main 52521ca, 7 rounds: geomean of the gated cases 1.0004
    (`lc-bench`), 1.0000 (`lc-intel`), 1.0021 (`lc-zen5`), 1.0004 (`lc-arm`); division, which now
    runs `%` through `modulo`, 0.994-1.007 everywhere. On `lc-zen5` `multiply/8`-`/10` and
    `to_values/10` came out 1.03-1.10 in that run (code this change does not touch); rerun alone
    (11 rounds) they gave 0.994-1.003: placement, as in earlier entries.
