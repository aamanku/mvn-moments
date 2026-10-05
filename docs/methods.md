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
