# API

Every method computes, for X ~ N(mu, Sigma) and the rectangle
S = [lower, upper] (bounds may be infinite), the selected raw integrals
`zeroth` = P(X in S), `first` = E[X 1{X in S}], and
`second` = E[X X^T 1{X in S}]. The truncated mean is `first / zeroth`.

## C++ (`<mvn_moments/moments.hpp>`)

`MonteCarloSolver`, `HaltonSolver`, and `GenzSolver` share one shape:

```cpp
// Setup, the only call that allocates: validates config, sizes buffers for up
// to max_dimension variables, seeds the random stream.
mvn::Status Init(Eigen::Index max_dimension, const Config& config = {},
                 std::uint64_t seed = 42);

// No allocation, no exceptions. Problems may shrink or grow up to
// max_dimension between calls.
mvn::Status Compute(mean, covariance, lower, upper,
                    mvn::Order order = mvn::Order::kAll);

const mvn::Result& GetResult() const;  // Empty before Init or after failure.
```

- `Config` is `SamplingConfig` (Monte Carlo, Halton) or `GenzConfig`.
- Inputs are `Eigen::Ref`, so fixed-size types such as `Eigen::Vector3d` bind
  without copies.
- The random stream advances across calls; a new solver with the same seed
  reproduces the sequence.
- Solvers are movable, not copyable, and not thread-safe.

`Analytic(mean, covariance, lower, upper, order, result)` is the closed form
for one dimension; it writes a caller-owned `mvn::Result result(1)`.

**Result.** `zeroth` and `zeroth_error` are `std::optional<double>`;
`First()`, `Second()`, `FirstError()`, and `SecondError()` are Eigen views of
preallocated storage, empty when not selected. Only Genz fills the errors.
`samples` and `accepted` count draws.

**Order.** A bit set: `kZeroth | kFirst`, `kAll` (default), `kNone`.

**Status.**

| Status | Meaning |
| --- | --- |
| `kOk` | Success. |
| `kInvalidInput` | Bad shape, values, bounds, or config; asymmetric covariance; dimension above capacity; uninitialized solver. |
| `kNotPositiveDefinite` | Covariance not positive definite. |
| `kNumericallySingular` | Genz: covariance nearly singular. |
| `kNoAcceptedSamples` | Monte Carlo/Halton never entered the rectangle. |
| `kNumericalError` | Overflow, or an underflowed probability. |

`StatusMessage(status)` describes a status.

**Configuration.** `samples` (default 100000; Genz: total over shifts),
`batch_size` (4096; bounds buffer memory), `timing` (needs
`MVN_ENABLE_BENCHMARKS=ON`), and for Genz `shifts` (10, at least 2).

**Helpers.** `<mvn_moments/helper.hpp>` has `IsSymmetric`,
`IsPositiveDefinite`, `CholeskyLower`, and random-problem generators for
tests and benchmarks. They allocate and throw: setup only.

## Python

```python
from mvn_moments import GenzSolver, GenzConfig, Order, genz_quasi_monte_carlo

result = genz_quasi_monte_carlo(mean, covariance, lower, upper,
                                order=Order.ALL, config=GenzConfig(), seed=42)

solver = GenzSolver(max_dimension, config=GenzConfig(), seed=42)
result = solver.compute(mean, covariance, lower, upper, order=Order.ALL)
```

The other methods are `monte_carlo` / `MonteCarloSolver` and
`quasi_monte_carlo` / `HaltonSolver` (with `SamplingConfig`), and
`analytic(mean, covariance, lower, upper, order=Order.ALL)`.

Results have `samples`, `accepted`, `zeroth`, `first`, `second`, and the
`*_error` fields, as floats/None and NumPy arrays. Invalid inputs raise
`ValueError`; numerical failures raise `RuntimeError`.
