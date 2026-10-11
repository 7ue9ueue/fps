# fps: API design

Status: draft, 2026-10-11. Design only, no code yet.

Univariate polynomials and power series modulo a prime P, with the speed of `cpp_hpc/lc-opt` and an
API that reads like math. The code moves here from `lc-opt/lib` (poly, ntt, multimod, mem) and from
the polynomial problems that still keep code of their own. lc-opt's `AGENTS.md` rules carry over.

## Decisions

| Topic | Decision | Why |
|---|---|---|
| Namespace | `fps` (formal power series) | Short; the usual name in contest code |
| Variables | One (univariate) | Scope |
| Types | `Polynomial` (coefficients), `Fraction` (the series p/q) and `Values` (value form); no general `Series` type | A series at precision n is a polynomial; laziness was too complex for v1. `Fraction`, not `Rational`, which reads as a rational number |
| Evaluation | Eager | Simple; n is known when a function runs |
| Precision | Series functions return `f.size()` coefficients unless n is given | The common case; no hidden global state |
| Huge indices | `Fraction`: `f[k]` for any k (Bostan-Mori); sums, products, prefix sums stay fractions | Covers 1/(1-x), linear recurrences, k up to 2^64 |
| Coefficients | `std::uint32_t` in [0, P), in and out | Same as lc-opt; no wrapper type |
| Modulus | Template parameter, known at compile time | Kernels specialize on it |
| Moduli in v1 | 998244353 only; other primes later | The case lc-opt is tuned for; other moduli are new work |
| Functions | Free functions: `fps::exp(f)` | Reads like math |
| Special cases | Detected, not named: `compose(f, x + c)` is a Taylor shift, `f * x^k` a shift | Fewer names to learn |
| Memory | Owned by the library: huge pages, per-thread scratch | No arenas or scratch sizes in the API |
| Backend | Codelets per CPU, chosen at compile time where possible, else by tuning (FFTW-style wisdom) | Different CPUs want different kernels |
| Packaging | Header-only for now | Simple to use. Cost: the ISA is fixed by the compiler flags; no run-time dispatch across CPUs, which would need a compiled library |
| `Values` layout | The backend picks; `Layout::pointwise` asks for the value-by-value layout | That layout wins when one conversion feeds many products (ABC222 H v2) |

Not in v1: several variables, moduli other than 998244353, moduli known only at run time, composite
moduli (e.g. 2^64), lazy series, series defined by equations (f = 1 + x f^2), D-finite recurrences,
`Relaxed` (see Later).

## Types

```cpp
namespace fps {
template <std::uint32_t P> class Polynomial;  // coefficients c[0] + c[1] x + ...
template <std::uint32_t P> class Fraction;    // the infinite series p/q
template <std::uint32_t P> class Values;      // a polynomial in value form, length L
template <std::uint32_t P> class Factorials;  // n!, 1/n!, binomials
}
```

Users name their modulus once: `using Poly = fps::Polynomial<998244353>;`.

## Conventions

- Values are canonical: every coefficient is in [0, P), in and out.
- `size()` is the stored length, trailing zeros included. The library never trims. `degree()` is
  the index of the last nonzero coefficient, -1 for zero.
- Precision n means "mod x^n": the first n coefficients. A series function reads only f mod x^n.
- Defaults: one-argument series functions return `f.size()` coefficients; `compose(f, g)` returns
  max(`f.size()`, `g.size()`); `*` is the exact product.
  Short closed forms need n: `inverse(1 - x)` has 2 coefficients, `inverse(1 - x, n)` has n.
- Special inputs are detected at run time and take a faster path, with the same result:
  few nonzero terms (`inverse(1 - x, n)` by recurrence, O(n)), a monomial factor or divisor
  (`f * x^k`, `f / x^k`, `f % x^k`, O(n)), a linear inner function (`compose(f, x + c)`,
  `compose(f, c x)`).
- A broken precondition throws `std::domain_error` (e.g. `inverse` with f[0] = 0). A result that
  may not exist is a `std::optional` (`sqrt`, `inverse_mod`).
- Thread safety: functions are safe to call from several threads; each thread has its own scratch.
- Moduli: v1 supports P = 998244353 only; any other P fails to compile (`static_assert`). Other
  primes come later (see Later).

## Polynomial

```cpp
template <std::uint32_t P>
class Polynomial {
public:
    Polynomial();                             // zero
    explicit Polynomial(std::size_t n);       // n zeros
    Polynomial(std::initializer_list<std::uint32_t> c);
    explicit Polynomial(std::span<const std::uint32_t> c);
    static Polynomial x(std::size_t k = 1);   // x^k

    std::size_t size() const;
    int degree() const;
    std::uint32_t& operator[](std::size_t i);               // i < size()
    std::uint32_t operator[](std::size_t i) const;
    std::uint32_t coeff(std::size_t i) const;               // 0 past the end
    void resize(std::size_t n);
    std::span<std::uint32_t> span();
    std::span<const std::uint32_t> span() const;

    Polynomial truncate(std::size_t n) const; // f mod x^n, the same as f % x^n
    Polynomial reverse(std::size_t n) const;  // x^(n-1) f(1/x), first n coefficients
    std::uint32_t operator()(std::uint32_t c) const;        // f(c)
};
```

Arithmetic, all exact: `f + g`, `f - g`, `-f`, `f * g`, `f * c`, `c * f`, `f / g` (quotient),
`f % g` (remainder), and the compound forms (`+=`, `*=`, ...). With `x = Poly::x()`:

| Code | Meaning |
|---|---|
| `f * Poly::x(k)` | f x^k |
| `f / Poly::x(k)` | drop the lowest k coefficients |
| `f % Poly::x(k)`, `f.truncate(k)` | f mod x^k |

### Functions

```cpp
// Products
Polynomial mul(f, g, n);                  // f g mod x^n
Polynomial square(f);                     // f^2, cheaper than f * f
Polynomial product(std::span<const Polynomial> fs);   // balanced product tree
Polynomial middle_product(f, g);          // the transposed product

// Division and calculus
std::pair<Polynomial, Polynomial> divmod(f, g);       // quotient, remainder
Polynomial derivative(f);
Polynomial integral(f);                   // constant term 0

// Series: n defaults to f.size()
Polynomial inverse(f, n);                 // 1/f; f[0] != 0
Polynomial log(f, n);                     // f[0] == 1
Polynomial exp(f, n);                     // f[0] == 0
Polynomial pow(f, std::uint64_t e, n);    // any f, leading zeros included
std::optional<Polynomial> sqrt(f, n);
Polynomial compose(f, g, n);              // f(g) mod x^n; n defaults to max(f.size(), g.size());
                                          // linear g: Taylor shift
Polynomial compositional_inverse(f, n);   // f[0] == 0, f[1] != 0

// Evaluation and interpolation
std::vector<std::uint32_t> evaluate(f, points);          // points: a span, or geometric(...)
Polynomial interpolate(points, values);                  // distinct points
Geometric geometric(std::uint32_t a, std::uint32_t r, std::size_t m);   // the points a r^i, i < m
std::vector<std::uint32_t> extrapolate(values, std::uint32_t c, std::size_t m);
                                          // f(c + i), i < m, from values = f(0 .. n-1)
Polynomial to_newton_basis(f, points);

// Algebra
Polynomial gcd(f, g);                     // monic
std::optional<Polynomial> inverse_mod(f, m);   // f^-1 mod m
std::vector<std::uint32_t> roots(f);      // distinct roots in [0, P)
std::vector<std::pair<Polynomial, int>> factor(f);   // monic irreducible factors, multiplicities
```

`Factorials<P> fact(n)`: `fact[i]`, `fact.inverse(i)`, `fact.binomial(n, k)` for i, n < size.

Scalars: `fps::mod_mul<P>(a, b)`, `fps::mod_pow<P>(a, e)`, `fps::mod_inverse<P>(a)`.

## Fraction

The infinite series p/q, stored as the two polynomials. `f[k]` is coefficient k for any k up to
2^64 (Bostan-Mori). Sums, products, prefix sums and derivatives of fractions are fractions again,
so they are exact polynomial arithmetic on p and q, and `[k]` still answers at the end. Eager:
nothing is cached or computed until asked.

```cpp
template <std::uint32_t P>
class Fraction {
public:
    Fraction(Polynomial<P> p, Polynomial<P> q);   // p/q; throws if q[0] == 0
    Fraction(Polynomial<P> p);                    // p/1

    std::uint32_t operator[](std::uint64_t k) const;   // coefficient k
    Polynomial<P> take(std::size_t n) const;           // the first n coefficients
    const Polynomial<P>& numerator() const;
    const Polynomial<P>& denominator() const;
};
```

Operations, all exact and all returning a `Fraction` (a, b fractions; f a polynomial; c a scalar):

| Code | Meaning |
|---|---|
| `a + b`, `a - b`, `-a`, `c * a` | Termwise sum, difference, scaling |
| `a * b`, `f * a` | Product of series (convolution of the sequences) |
| `a / f`, `a / b` | Division of series; needs f[0] != 0, b's numerator [0] != 0 |
| `a / (1 - x)` | Prefix sums: coefficient k is a_0 + ... + a_k |
| `a * Poly::x(k)` | The sequence moved k places later |
| `derivative(a)` | Coefficient k is (k + 1) a_(k+1); `x * derivative(a)` gives k a_k |

Costs, with d the larger of deg p and deg q:

| Operation | Cost |
|---|---|
| `f[k]` | O(d log d log k). Small d: microseconds. d = 10^5, k = 10^18: about 60 products of length 2d, roughly 100 ms (guess, not measured) |
| `take(n)` | O(n d) by recurrence for small d, else `mul(p, inverse(q, n), n)` |
| `a + b`, `a * b`, `a / b` | One or two polynomial products; the degrees add |

- Degrees grow and are never reduced: `fib + fib` has denominator q^2, not q. Long chains make
  `[k]` slower. See Later (`reduced()`).
- Each `[k]` is a full O(d log d log k) computation. For many small indices, call `take(n)` once.

## Values

A polynomial in value form. A product of two `Values` is value by value, O(L), so a polynomial
used in many products is converted once and reused. A sum of products needs one conversion back.

```cpp
template <std::uint32_t P>
class Values {
public:
    std::size_t size() const;             // L, a power of two
    Values& operator+=(const Values&);
    Values& operator-=(const Values&);
    Values& operator*=(const Values&);    // value by value
    Values& operator*=(std::uint32_t c);
    void multiply_add(const Values& a, const Values& b);   // *this += a b
};
// Also A + B, A - B, A * B, c * A.

enum class Layout { automatic, pointwise };   // automatic: the backend picks

Values<P> to_values(f, std::size_t L, Layout layout = Layout::automatic);   // f.size() <= L; zeros above
Polynomial<P> to_coefficients(const Values<P>& A);   // L coefficients: the product mod (x^L - 1)
```

- `to_coefficients(A * B)` is a b mod (x^L - 1): coefficient L + i adds onto i. For the exact
  product, L > deg a + deg b. With a smaller L the wrapped terms land low; a middle product keeps
  only the upper part.
- `Layout::pointwise` is the hint for a polynomial used in many products per conversion: the
  value-by-value layout wins there (ABC222 H v2).
- Operands must have the same L and layout; a mismatch throws.
- The contents are opaque: no `operator[]`. The layout depends on the CPU (lc-opt's transform
  ends at blocks of 8 coefficients, not single values), so it is not a DFT. `fps::dft` and
  `fps::idft` give the true values at the roots, in a documented order, for users who need them.

## Raw functions

Same names in `fps::raw`, on spans. They write into the caller's memory and never allocate;
n is `out.size()`. For hot loops, and for code that keeps its own buffers.

```cpp
fps::raw::exp<P>(std::span<const std::uint32_t> f, std::span<std::uint32_t> out);  // out may be f
fps::raw::mul<P>(a, b, out);
fps::raw::to_values<P>(std::span<const std::uint32_t> f, Values<P>& A);            // into an existing A
fps::raw::to_coefficients<P>(const Values<P>& A, std::span<std::uint32_t> out, std::size_t offset);
                                          // out[i] = coefficient offset + i
fps::raw::add_coefficients<P>(const Values<P>& A, std::span<std::uint32_t> out, std::size_t offset);
                                          // out[i] += coefficient offset + i
```

Each function documents whether `out` may overlap an input.

## Backend and tuning

Not part of the user API, except for tuning:

```cpp
fps::tune();                     // measure the codelets on this machine (seconds)
fps::save_wisdom(path);
fps::load_wisdom(path);          // without wisdom: compile-time and hardware-fact choices
```

Choices are made in this order: compile time (the ISA from compiler macros), hardware facts at
run time (cache sizes, CPU features), then measurement (wisdom). Every variant stays testable by a
flag that forces it.

## Examples

```cpp
#include <fps/fps.hpp>
constexpr std::uint32_t P = 998244353;
using Poly = fps::Polynomial<P>;
const Poly x = Poly::x();
```

exp of an input series:

```cpp
Poly g = fps::exp(Poly(read_coefficients(n)));
```

1/(1-x), the first 10^6 terms:

```cpp
Poly g = fps::inverse(1 - x, 1'000'000);   // O(n): sparse input
```

Fractions: coefficients at 10^18, and arithmetic that stays exact.

```cpp
using Frac = fps::Fraction<P>;
const std::uint64_t K = 1'000'000'000'000'000'000ULL;

Frac geo(Poly{1}, 1 - x);           // 1 + x + x^2 + ...
geo[K];                             // 1

Frac fib(x, 1 - x - x * x);         // 0, 1, 1, 2, 3, 5, 8, ...
fib[6];                             // 8
fib[K];                             // F(10^18) mod P
fib.take(10);                       // 0 1 1 2 3 5 8 13 21 34

(fib / (1 - x))[K];                 // F_0 + F_1 + ... + F_K
(fib * fib)[K];                     // sum of F_i F_(K-i)
(x * fps::derivative(fib))[K];      // K F_K
(fib + 3 * geo)[K];                 // F_K + 3

Frac strings(1 + x, 1 - x - x * x); // binary strings of length k with no "11"
strings[3];                         // 5

Frac tilings(1 - x * x, 1 - 4 * x * x + Poly::x(4));   // domino tilings of a 3 x k board
tilings[10];                        // 571
```

Catalan numbers, C = (1 - sqrt(1 - 4x)) / (2x). The user tracks precision: one more term, since
dividing by x drops one.

```cpp
Poly s = *fps::sqrt(1 - 4 * x, n + 1);
Poly c = (1 - s) / x * fps::mod_inverse<P>(2);   // n coefficients
```

f(x + c), and evaluation on a geometric sequence:

```cpp
Poly g = fps::compose(f, x + c);                          // Taylor shift, O(n log n)
auto v = fps::evaluate(f, fps::geometric(a, r, m));       // chirp z-transform
```

One polynomial in many products, and a sum of products:

```cpp
auto B = fps::to_values(b, L);
for (const Poly& a : many) results.push_back(fps::to_coefficients(fps::to_values(a, L) * B));
Poly s = fps::to_coefficients(A1 * B1 + A2 * B2);
```

Online self-convolution by divide and conquer (ABC222 H, O(N log^2 N)). The prefix f[0, L) is
converted once per level; nothing is allocated in the recursion.

```cpp
void solve(std::size_t l, std::size_t r) {
    if (r - l <= 32) return leaf(l, r);           // schoolbook, user code
    const std::size_t m = (l + r) / 2, L = r - l;
    solve(l, m);
    fps::Values<P>& A = buffer[L];
    fps::raw::to_values<P>(std::span(f).subspan(l, m - l), A);
    A *= prefix[L];                               // values of f[0, L)
    fps::raw::add_coefficients<P>(A, std::span(h).subspan(m, r - m), m - l);
    solve(m, r);
}
```

The B-ary version (lc-opt-explore `showcase-abc222_h_dc`, v1-v5: 2783 ms to about 1150 ms at
N = 10^7) uses `multiply_add` over the earlier blocks, then one `add_coefficients`.

## Sources in lc-opt

| API | lc-opt source | Work |
|---|---|---|
| `*`, `mul`, `square` | `lib/ntt/product.hpp`, `lib/multimod` | Wrap |
| `inverse`, `exp`, `log`, `pow`, `sqrt` | `lib/poly/{inverse,exp,log,pow,sqrt}.hpp` | Wrap |
| Sparse paths | `lib/poly/{sparse,holonomic,divider}.hpp` | Wrap, detect sparsity |
| `compose`, `compositional_inverse` | `lib/poly/{composition,compositional_inverse}.hpp` | Wrap |
| `compose` with linear g | polynomial_taylor_shift | Move problem code, detect |
| `product` | `lib/poly/product_tree.hpp`, product_of_polynomial_sequence | Move problem code |
| `evaluate`, `interpolate`, `to_newton_basis` | `lib/poly/{evaluation,interpolation,newton}.hpp` | Wrap |
| `evaluate`, `interpolate` on `geometric` | `lib/poly/chirp.hpp`, the two geometric problems | Move problem code |
| `extrapolate` | shift_of_sampling_points_of_polynomial | Move problem code |
| `divmod`, `gcd`, `inverse_mod` | `lib/poly/{division,gcd}.hpp` | Wrap |
| `roots`, `factor` | polynomial_root_finding, factorization_of_polynomials | Move problem code |
| `Factorials` | `lib/poly/factorials.hpp` | Wrap |
| `Values` | `lib/poly/transform.hpp`; value-by-value layout from ABC222 H v2-v5 | Wrap, new layout |
| `Fraction`, `middle_product`, `dft`, `idft` | none (middle products exist inside log and evaluation) | New |
| Free functions with n | `lc-opt/.claude/worktrees/ntt-library-problems-633247/lib/easy/poly.hpp` | Prior art |

lc-opt targets 998244353 on Zen 3 AVX2. Other moduli and CPUs are new work.

## Later

- Other moduli: other NTT primes (P = c 2^k + 1) with one NTT, 1000000007 with its own fast path,
  then any prime P < 2^30 with the generic multimod path.
- A compiled library, if run-time dispatch across CPUs is needed.
- More on `Fraction`:
  - `reduced()`: divide p and q by their gcd, so chains of operations stay small.
  - Termwise product: the series of a_k b_k (e.g. F_k^2, F_k 2^k) is a fraction too; heavier algebra.
  - Every m-th term: a_(mk+r) is a fraction too; Bostan-Mori computes this internally anyway.
  - From a matrix or an automaton: walks of length k in a graph, strings that avoid patterns;
    q = det(I - x A). The most common source of fractions in contest problems.
- Guessing linear recurrences: `find_linear_recurrence(a)` (Berlekamp-Massey: the shortest q with
  a_n + q1 a_(n-1) + ... + qd a_(n-d) = 0; [0, 1, 1, 2, 3, 5, 8, 13] gives 1 - x - x^2), returning
  a `Fraction`; and `guess_linear_recurrence_term(a, k)` (that, then coefficient k; needs about 2d
  terms, and is silently wrong with too few).
- D-finite recurrences: `Recurrence` (coefficients polynomial in n, e.g. Catalan's
  n a_n = (4n - 2) a_(n-1)), with `terms(n)` in O(n · order) and `term(k)` for k < P in about
  O(sqrt(k) log k); `factorial(k)`. From `lib/poly/holonomic.hpp` and shift of sampling points.
- `Relaxed` and `RelaxedSquare`: online convolution, `push(x_i)` returns h_i. ABC222 H becomes a
  short loop.
- Series defined by equations, on top of `Relaxed`.
- A lazy series type on top of `Polynomial`, if a use case needs it.
- Fast I/O (`lc-opt/lib/io`).

## Open questions

None.
