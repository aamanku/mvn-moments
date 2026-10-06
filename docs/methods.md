# Methods

| Method | Use when | Limits |
| --- | --- | --- |
| Genz (`GenzSolver`) | Default choice. Every point lies in the rectangle, so rare rectangles work; reports standard errors across shifts. | Nearly singular covariances return `kNumericallySingular`; errors can be understated for very skewed integrands; increase `shifts` when error bars matter. |
| Monte Carlo (`MonteCarloSolver`) | Simple baseline. | Rejection sampling: about `samples * P(X in S)` draws are accepted, so small probabilities need large budgets or return `kNoAcceptedSamples`. Error O(N^-1/2). |
| Halton QMC (`HaltonSolver`) | Low dimensions, moderate probabilities. | Rejection as above; the gain over Monte Carlo shrinks as the dimension grows. |
| Analytic (`Analytic`) | One dimension. | n = 1 only. |

Run `examples/compare_accuracy.py` (see [development](development.md)) to
measure error and time for your problem sizes. Equations and references are at
the top of each source file in `src/`.

## Real-time use

After `Init`, `Compute` (and `Analytic`):

- never allocates: all buffers, the Cholesky factor, and the Genz lattice are
  sized in `Init`, and the batch math uses Eigen column operations that need no
  heap or large stack buffers;
- never throws, and reports failures through `Status`, leaving the solver
  usable;
- runs in time proportional to `samples * n^2`, plus O(n^3) to factor a
  changed covariance (an unchanged one is cached).

`tests/realtime.cpp` enforces this: it fails if `Compute` makes any heap
allocation, across changed covariances, smaller problems, order changes, and
failures. Keep `MVN_ENABLE_BENCHMARKS` off in real-time builds and create one
solver per thread.

## Specialized low-dimensional methods and automatic selection

`Analytic` uses closed-form 1D moments. `Bivariate` integrates the analytic
conditional moments of the second coordinate over the first coordinate using
adaptive Gauss(7)-Kronrod(15) quadrature. It computes all selected moments in
one deterministic integration, preserving tail probabilities without four-corner
CDF subtraction. Infinite outer bounds are truncated at +/-40 standard
deviations; omitted probability mass is below double's smallest subnormal.
Narrow intervals use stable width calculations and Gauss-Legendre conditional
moments. Quadrature estimates are not certified error bounds; nonconvergence,
underflow, and nonfinite results return explicit errors.

Python-only `MomentsSolver()` infers and validates the actual dimension on every call: 1D to `Analytic`,
2D to `Bivariate`, and higher dimensions to Genz. Selection does not switch
silently to a sampling method when specialized quadrature fails. Genz warns
once when explicitly used for dimensions 1 or 2.
