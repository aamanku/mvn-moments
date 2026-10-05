# mvn-moments

![mvn-moments: N-dimensional Gaussian moments over hyperrectangles, C++17 / Python](https://raw.githubusercontent.com/aamanku/mvn-moments/main/docs/assets/social-preview.png)

Probability, mean, and second moment of a multivariate normal distribution
over a rectangle, with a C++17 library (Eigen) and Python bindings. Built for
real-time loops: solvers allocate once in `Init`, then `Compute` never
allocates or throws and returns a status code.

For a multivariate normal random vector and an axis-aligned rectangle, it computes

![X follows N(mu, Sigma); S is the rectangle from lower to upper. zeroth = P(X in S); first = E[X 1{X in S}]; second = E[X X^T 1{X in S}].](https://raw.githubusercontent.com/aamanku/mvn-moments/main/docs/assets/moments.png)

so the truncated mean is `first / zeroth` when `zeroth > 0`. Each source file starts with the
equations it implements and links to its references.

## Python

```sh
pip install --pre mvn-moments
```

Wheels cover CPython 3.9-3.14 on Linux, Windows, and macOS.

Call `mvn_moments.help()` after `import mvn_moments` for a manual-style overview of the
available methods, options, results, errors, and the example below.

```python
import numpy as np
from mvn_moments import GenzSolver, genz_quasi_monte_carlo

n = 5
mean = np.zeros(n)
covariance = 0.5 * np.eye(n) + 0.5  # Unit variances, correlation 0.5.
lower, upper = np.full(n, -1.0), np.full(n, 2.0)

# One call.
result = genz_quasi_monte_carlo(mean, covariance, lower, upper)
print(result.zeroth, result.first / result.zeroth)

# Repeated calls, e.g. in a loop: allocate once, then compute.
solver = GenzSolver(n, seed=42)
result = solver.compute(mean, covariance, lower, upper)
```

Invalid inputs raise `ValueError`; numerical failures raise `RuntimeError`.

## C++

Requires CMake 3.19+ and a C++17 compiler. An installed Eigen 3.4+ (e.g.
`libeigen3-dev`) is used if found; otherwise Eigen 3.4.0 is downloaded.

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build
cmake --install build --prefix /your/prefix
```

```cmake
find_package(mvn_moments CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE mvn_moments::mvn_moments)
```

```cpp
#include <mvn_moments/moments.hpp>

int main()
{
    constexpr int n = 5;
    using Vector = Eigen::Matrix<double, n, 1>;
    using Matrix = Eigen::Matrix<double, n, n>;

    const Vector mean = Vector::Zero();
    const Vector lower = Vector::Constant(-1), upper = Vector::Constant(2);
    // Unit variances, correlation 0.5.
    const Matrix covariance =
        0.5 * Matrix::Identity() + Matrix::Constant(0.5);

    // Setup: allocates for problems with up to n variables.
    mvn::GenzSolver solver;
    if (solver.Init(n) != mvn::Status::kOk) {
        return 1;
    }

    // Loop body: no allocation, no exceptions.
    const mvn::Status status = solver.Compute(mean, covariance, lower, upper);
    if (status != mvn::Status::kOk) {
        return 1;  // mvn::StatusMessage(status) explains why.
    }

    const mvn::Result& result = solver.GetResult();
    const Eigen::VectorXd truncated_mean = result.First() / *result.zeroth;
    return 0;
}
```

## Methods

| Method | C++ | Python | Notes |
| --- | --- | --- | --- |
| Genz lattice QMC | `GenzSolver` | `genz_quasi_monte_carlo` | No rejection, so rare rectangles work; reports standard errors |
| Monte Carlo | `MonteCarloSolver` | `monte_carlo` | Rejection sampling |
| Halton QMC | `HaltonSolver` | `quasi_monte_carlo` | Rejection sampling with shifted Halton points |
| Closed form | `Analytic` | `analytic` | One dimension only |

Select moments with `Order` (e.g. `Order::kZeroth | Order::kFirst`), and set
the sample budget with `SamplingConfig` or `GenzConfig`.

More in [docs](https://github.com/aamanku/mvn-moments/blob/main/docs/README.md): API, methods and real-time use, development,
and releasing.

## License

BSD-3-Clause. Wheels contain compiled Eigen code (MPL-2.0); see
`licenses/eigen`.
