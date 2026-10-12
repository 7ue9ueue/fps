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
- Precision n means "mod x^n": the first n coefficients. A series function reads only f mod x^n,
  so its precondition on f[0] applies for n > 0 only (`inverse(f, 0)` is empty for any f).
- Defaults: one-argument series functions return `f.size()` coefficients; `compose(f, g)` returns
  max(`f.size()`, `g.size()`); `*` is the exact product.
  Short closed forms need n: `inverse(1 - x)` has 2 coefficients, `inverse(1 - x, n)` has n.
- Special inputs are detected at run time and take a faster path, with the same result:
  few nonzero terms (`inverse(1 - x, n)` by recurrence, O(n)), a monomial factor or divisor
  (`f * x^k`, `f / x^k`, `f % x^k`, O(n)), a linear inner function (`compose(f, x + c)`,
  `compose(f, c x)`).
- A broken precondition throws `std::domain_error` (e.g. `inverse` with f[0] = 0); a size past a
  backend limit throws `std::length_error` (e.g. a product past 2^25 coefficients that runs the NTT;
  a short factor next to a longer one runs `small::multiply` where that is faster, which has no
  limit). A result that may not exist is a `std::optional` (`sqrt`, `inverse_mod`).
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

Arithmetic, all exact: `f + g`, `f - g`, `-f`, `f * g`, `f / g` (quotient), `f % g` (remainder);
with a scalar c, an integer of at most 64 bits taken mod P: `f + c`, `c + f`, `f - c`, `c - f`,
`f * c`, `c * f`; and the compound forms (`+=`, `*=`, ...). With `x = Poly::x()`:

| Code | Meaning |
|---|---|
| `f * Poly::x(k)` | f x^k |
| `f / Poly::x(k)` | drop the lowest k coefficients |
| `f % Poly::x(k)`, `f.truncate(k)` | f mod x^k |

### Functions

```cpp
// Products
Polynomial mul(f, g, n);                  // f g mod x^n: n coefficients
Polynomial square(f);                     // f^2; one forward transform less than f * f
Polynomial product(std::span<const Polynomial> fs);   // balanced product tree; 1 if fs is empty
Polynomial middle_product(f, g);          // coefficients m - 1 .. n - 1 of f g (below)

// Division and calculus
std::pair<Polynomial, Polynomial> divmod(f, g);       // quotient, remainder
Polynomial derivative(f);                 // f.size() - 1 coefficients (0 for an empty f)
Polynomial integral(f);                   // constant term 0, f.size() + 1 coefficients; f.size() < P

// Series: n defaults to f.size()
Polynomial inverse(f, n);                 // 1/f; f[0] != 0
Polynomial log(f, n);                     // f[0] == 1; n <= P
Polynomial exp(f, n);                     // f[0] == 0; n <= P
Polynomial pow(f, std::uint64_t e, n);    // any f, leading zeros included; 0^0 = 1; n <= P
std::optional<Polynomial> sqrt(f, n);     // g with g^2 = f mod x^n (below), or nullopt
Polynomial compose(f, g, n);              // f(g) mod x^n; n defaults to max(f.size(), g.size());
                                          // any f, g; constant g: f(g[0]); linear g: Taylor shift
Polynomial compositional_inverse(f, n);   // f[0] == 0 (n >= 1), f[1] != 0 (n >= 2); n <= P

// Evaluation and interpolation
std::vector<std::uint32_t> evaluate(f, points);          // points: a span, or geometric(...)
Polynomial interpolate(points, values);                  // distinct points
Geometric geometric(std::uint32_t a, std::uint32_t r, std::size_t m);   // the points a r^i, i < m
std::vector<std::uint32_t> extrapolate(values, std::uint32_t c, std::size_t m);
                                          // f(c + i), i < m, from values = f(0 .. n-1)
Polynomial to_newton_basis(f, points);

// Algebra
Polynomial gcd(f, g);                     // monic, deg + 1 coefficients; gcd(0, 0) = 0 (none)
std::optional<Polynomial> inverse_mod(f, m);   // f^-1 mod m: deg m coefficients, nullopt if
                                          // gcd(f, m) != 1; m != 0; a constant m gives 0
std::vector<std::uint32_t> roots(f);      // distinct roots in [0, P), increasing; f != 0
std::vector<std::pair<Polynomial, int>> factor(f);   // monic irreducible factors, multiplicities;
                                          // f != 0, deg f < P; by degree, then coefficients
                                          // from the constant term; a constant f gives none
```

`middle_product(f, g)`, for n = `f.size()` >= m = `g.size()` >= 1 (else `std::domain_error`):
the n - m + 1 coefficients m - 1 .. n - 1 of f g, h[i] = sum over j < m of f[i + j] g[m - 1 - j].
They are the coefficients whose sums meet all of g. With g reversed it is the transposed product,
sum over j of f[i + j] g[j]. It runs a cyclic product of length >= n (the wrapped terms land
below m - 1), not n + m - 1. `product(fs)` runs the backend's product tree (small factors 8 at a
time in lanes, one transform of each length per node); it also takes a `std::vector`.

`divmod(f, g)`, `f / g`, `f % g`: f = q g + r with deg r < d = deg g. q has max(`f.size()` - d, 0)
coefficients and r has d (f's, then zeros, if `f.size()` <= d), so `f / x^k` drops the lowest k
coefficients and `f % x^k` is `f.truncate(k)`; neither is trimmed. g = 0, of any size, throws
`std::domain_error`. `f / g` computes no remainder. Divisors of few terms are detected (#147),
each O(n t) for t terms below the leading one: g = c x^d is a shift; t <= 16 terms, the highest at
least 8 (t + 1) below the leading one (`x^L - 1`, `x^d - x - 1` for d >= 25), run long division in
vector passes; else t <= 2 (`x^3 + 2x + 5`) runs a recurrence. Every other g runs the backend's
blocked division, whose transforms stay within 2^25 words (f of 2^25 + 2^10 by g of 2^25 + 1
runs).

`evaluate(f, points)` returns `points.size()` values, for any points (repeats too).
`interpolate<P>(points, values)` returns m = `points.size()` coefficients; it needs as many values
and distinct points (a repeat throws `std::domain_error`: it makes M'(x_i) = 0 for M =
prod (x - x_i), which the algorithm divides by, so the check is free). `to_newton_basis(f, points)`
returns c of `f.size()` = n coefficients with f = sum_k c_k prod_(i < k) (x - x_i); it reads the
first n - 1 points, any of them (fewer throws `std::domain_error`). Points and values >= P throw
`std::domain_error`; 2^25 or more points to `interpolate` (n >= 2^25 to `to_newton_basis`) throw
`std::length_error`. `evaluate` has no size limit: the backend splits f and the points into chunks
of 2^24 (#139; the limit was the frontend's own tree's, whose root passed the NTT's 2^25).

On geometric points (fps/geometric.hpp, the backend's chirp kernels): `evaluate(f,
geometric(a, r, m))` returns m values for any a, r (repeats too); `interpolate<P>(geometric(a, r,
m), values)` returns m coefficients and needs m values and distinct points (a != 0 for m >= 2,
r != 0 for m >= 3, r^k != 1 for 0 < k < m; else `std::domain_error`). `extrapolate<P>(values, c,
m)` reads f of degree < n = `values.size()` <= P from f(0 .. n - 1) and returns f(c + i) for i < m,
c + i taken mod P. a, r, c and values >= P throw `std::domain_error`. `Geometric`'s constructor is
explicit, so a braced list such as `{1, 2, 3}` stays a list of points. The NTT behind them throws
`std::length_error` past 2^25 words (interpolate: m > 2^25, checked first).

`sqrt(f, n)` is the root of the polynomial f mod x^n = x^(2k) u, u[0] != 0: x^k s with s^2 = u
mod x^(n - k) and s[0] the smaller of the two square roots of u[0] in [0, P) (`sqrt(1 - 4x)` starts
with 1). Its top k coefficients are free in g^2 = f mod x^n; this fixes them as if f had no terms
past x^(n - 1). The zero polynomial for f mod x^n = 0; nullopt if the lowest nonzero term has an
odd index or a coefficient that is not a square. `pow` needs n <= P, as `log` and `exp`: it runs
(u / u[0])^e as exp(e log(u / u[0])), where only e mod P counts below x^P.

`Factorials<P> fact(n)`, n <= P: `fact[i]` = i!, `fact.inverse(i)` = 1/i!, `fact.binomial(n, k)`
= C(n, k) (0 for k > n), for i, n < `fact.size()`. n > P throws: P! = 0 has no inverse. Indices
are not checked, as in `operator[]`.

Scalars: `fps::mod_mul<P>(a, b)`, `fps::mod_pow<P>(a, e)`, `fps::mod_inverse<P>(a)`. They work for
any prime P < 2^32, not only 998244353: they need no kernels.

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
| `a + b`, `a - b`, `-a`, `c * a`, `a * c` | Termwise sum, difference, scaling |
| `a * b`, `f * a`, `a * f` | Product of series (convolution of the sequences) |
| `a / f`, `a / b` | Division of series; needs f[0] != 0, b's numerator [0] != 0 (else `std::domain_error`) |
| `a / (1 - x)` | Prefix sums: coefficient k is a_0 + ... + a_k |
| `a * Poly::x(k)` | The sequence moved k places later |
| `derivative(a)` | Coefficient k is (k + 1) a_(k+1); `x * derivative(a)` gives k a_k |

Costs, with d the larger of deg p and deg q:

| Operation | Cost |
|---|---|
| `f[k]` | O(d log d log k). k = 10^18: 3-6 us for Fibonacci, 6-16 us at d = 16; d = 10^5: 120 products of factors of about d, 44 ms (Zen 5), 86-92 ms (Emerald Rapids), 94 ms (Zen 3), 167 ms (Neoverse V2), measured (#23); a Bostan-Mori kernel is #114 |
| `take(n)` | `mul(p, inverse(q, n), n)`; `inverse` runs a recurrence, O(n t), for q with t <= 16 nonzero terms past q[0] |
| `a + b`, `a * b`, `a / b` | Three (`a + b`, `a - b`), two (`a * b`, `a / b`) or one (`a / f`, `f * a`) polynomial products; the degrees add |

- Degrees grow and are never reduced: `fib + fib` has denominator q^2, not q. Long chains make
  `[k]` slower. See Later (`reduced()`).
- Each `[k]` is a full O(d log d log k) computation. For many small indices, call `take(n)` once.

## Values

A polynomial in value form. A product of two `Values` is value by value, O(L), so a polynomial
used in many products is converted once and reused. A sum of products needs one conversion back.

```cpp
enum class Layout { automatic, pointwise };   // automatic: the backend picks

template <std::uint32_t P>
class Values {
public:
    Values();                             // empty (L = 0), for containers; assign before use
    explicit Values(std::size_t L, Layout layout = Layout::automatic);   // zero
    std::size_t size() const;             // L, a power of two
    Values& operator+=(const Values&);
    Values& operator-=(const Values&);
    Values& operator*=(const Values&);    // value by value
    Values& operator*=(c);                // c an integer of at most 64 bits, taken mod P
    void multiply_add(const Values& a, const Values& b);   // *this += a b; a or b may be *this
};
// Also A + B, A - B, A * B, c * A, A * c.

Values<P> to_values(f, std::size_t L, Layout layout = Layout::automatic);   // f.size() <= L; zeros above
Polynomial<P> to_coefficients(const Values<P>& A);   // L coefficients: the product mod (x^L - 1)
```

- `to_coefficients(A * B)` is a b mod (x^L - 1): coefficient L + i adds onto i. For the exact
  product, L > deg a + deg b. With a smaller L the wrapped terms land low; a middle product keeps
  only the upper part. On an rvalue, as there, the inverse transform runs in A's own memory.
- `Layout::pointwise` is the hint for a polynomial used in many products per conversion: the
  value-by-value layout wins there (ABC222 H v2). L above 2^23 has no such values (P - 1 =
  119 2^23 has no larger power-of-two roots of unity) and keeps the automatic layout.
- `Values(L, layout)` is the zero polynomial: the start of a sum of products (`multiply_add`) and
  the buffer that `raw::to_values` fills.
- Operands must have the same L and layout; a mismatch throws `std::domain_error`, as do L not a
  power of two and f.size() > L. L > 2^25 throws `std::length_error` (the backend's longest
  transform).
- The contents are opaque: no `operator[]`. The layout depends on the CPU (lc-opt's transform
  ends at blocks of 8 coefficients, not single values), so it is not a DFT. `fps::dft` and
  `fps::idft` give the true values at the roots, in a documented order, for users who need them.
- `fps::dft(f, L)` returns f(w^i) for i < L, natural order, with w = 3^((P - 1) / L) (3
  generates the units mod P): the DFT of f mod (x^L - 1), any `f.size()`. `fps::idft<P>(values)`
  returns the f of L = `values.size()` coefficients with f(w^i) = values[i]. L is 0 or any divisor
  of P - 1, not only a power of two (else `std::domain_error`): both run a chirp z-transform
  (fps/geometric.hpp), one middle product of transforms of length about 2L, slower than
  `to_values` for powers of two. Among powers of two they stop at 2^23 (P - 1 = 119 2^23 has no
  roots of higher order), while `Values` goes on to 2^25: its leaves x^8 - w need roots of order
  L/8 only. The middle product limits dft to f.size() + L - 1 <= 2^25 and idft to L <= 2^24
  (`std::length_error`, before allocating).

## Raw functions

Same names in `fps::raw`, on spans. They write into the caller's memory and never allocate,
except for a per-thread scratch that grows to the longest length a thread uses and is then
reused (`raw::mul` multiplies there; `raw::to_coefficients` and `raw::add_coefficients` transform
a copy of A there); n is `out.size()`. For hot loops, and for code that keeps its own buffers.

```cpp
fps::raw::exp<P>(std::span<const std::uint32_t> f, std::span<std::uint32_t> out);  // out may be f, else apart
fps::raw::inverse<P>(f, out);                                                      // out apart from f
fps::raw::log<P>(f, out);                                                          // out may overlap f
fps::raw::pow<P>(f, std::uint64_t e, out);                                         // out apart from f
bool fps::raw::sqrt<P>(f, out);           // out apart from f; false, out not written, if no root
fps::raw::mul<P>(a, b, out);                                                       // out apart from a, b
fps::raw::to_values<P>(std::span<const std::uint32_t> f, Values<P>& A);            // into an existing A
fps::raw::to_coefficients<P>(const Values<P>& A, std::span<std::uint32_t> out, std::size_t offset);
                                          // out[i] = coefficient offset + i
fps::raw::add_coefficients<P>(const Values<P>& A, std::span<std::uint32_t> out, std::size_t offset);
                                          // out[i] += coefficient offset + i
                                          // offset + out.size() <= L, else std::domain_error
```

Each function documents whether `out` may overlap an input.

## Backend and tuning

Not part of the user API, except for tuning:

```cpp
fps::tune_all();                 // measure the backend's choices on this machine (about 0.1 s)
bool fps::save_wisdom(const std::string& path);   // false if path cannot be written
bool fps::load_wisdom(const std::string& path);   // false, setting nothing, if path is missing,
                                                  // malformed, or from another CPU or target
```

Choices are made in this order: compile time (the ISA from compiler macros), hardware facts at
run time (cache sizes, CPU features), then measurement (wisdom). Every variant stays testable by a
flag that forces it.

- Explicit only: nothing tunes on first use, so a call costs the same every time. Without wisdom,
  every call uses the compiled defaults, measured per target on our machines.
- `tune_all()` runs once per process; it measures from the compiled defaults and replaces what
  `load_wisdom` set. Choices change speed, never results.
- Wisdom records the CPU and the target; a file from another CPU or target, a malformed file and
  a missing one change nothing. Neither function throws on a bad path.
- `-DFPS_NO_TUNER` (single-file contest bundles) compiles the defaults in as constants: `tune_all`
  does nothing, and `save_wisdom` and `load_wisdom` touch no file and return false.
- The name is `tune_all`, not `tune`: `fps::tune` is the backend tuner's namespace
  (`backend/tune/`), and a function cannot share its name.

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
