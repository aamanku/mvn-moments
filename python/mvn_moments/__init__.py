from ._core import (
    Order, Result, SamplingConfig, GenzConfig, MonteCarloSolver, HaltonSolver,
    GenzSolver, analytic, monte_carlo, quasi_monte_carlo,
    genz_quasi_monte_carlo, BivariateConfig, MomentsConfig, Method, MomentsSolver,
    bivariate, moments,
)

__all__ = ["Order", "Result", "SamplingConfig", "GenzConfig",
           "MonteCarloSolver", "HaltonSolver", "GenzSolver", "analytic",
           "monte_carlo", "quasi_monte_carlo", "genz_quasi_monte_carlo",
           "BivariateConfig", "MomentsConfig", "Method", "MomentsSolver",
           "bivariate", "moments", "help"]


def help():
    """Print a manual-style API overview and the README's Python example."""
    print("""MVN_MOMENTS(3)                  Python Library                  MVN_MOMENTS(3)

NAME
    mvn_moments - Gaussian probability and moments over hyperrectangles

SYNOPSIS
    import mvn_moments
    mvn_moments.help()

    result = method(mean, covariance, lower, upper, order=Order.ALL)
    solver = Solver(max_dimension, config=Config(), seed=42)
    solver = MomentsSolver(config=MomentsConfig(), seed=42)
    result = solver.compute(mean, covariance, lower, upper, order=Order.ALL)

DESCRIPTION
    Computes probability and raw first and second moments of an N-dimensional
    Gaussian over an axis-aligned hyperrectangle. Bounds may be infinite.
    Use a one-shot function or reuse a solver for repeated computations.

METHODS
    moments / MomentsSolver
        Selects analytic for 1D, bivariate for 2D, and Genz for higher dimensions.
        Python-only: no dimension or Init call; infers and validates input shapes
        on every compute call. Uses MomentsConfig (genz and bivariate settings).
        solver.method reports
        Method.ANALYTIC, Method.BIVARIATE, or Method.GENZ after success.

    bivariate
        Deterministic 2D conditional quadrature for probability and all moments.
        Uses BivariateConfig: relative_tolerance (default 1e-10), max_intervals
        (default 4096, maximum 65536), timing. No sampling standard errors.
        Convergence estimates are not certified bounds; failure raises RuntimeError.

    genz_quasi_monte_carlo / GenzSolver
        Genz lattice quasi-Monte Carlo; use above two dimensions. Avoids rejection
        sampling, so rare rectangles work. Reports standard errors across
        random shifts. Warns once when used in 1D/2D; prefer analytic/bivariate.
        Uses GenzConfig. Nearly singular covariances can fail;
        increase shifts when error estimates matter.

    monte_carlo / MonteCarloSolver
        Independent random sampling with rejection; a simple baseline.
        Uses SamplingConfig. Small rectangle probabilities require larger
        sample budgets and can yield no accepted samples.

    quasi_monte_carlo / HaltonSolver
        Shifted Halton quasi-Monte Carlo with rejection. Useful in low
        dimensions with moderate rectangle probabilities. Uses SamplingConfig.
        Gains over Monte Carlo shrink as dimension increases; rare rectangles
        can yield no accepted samples.

    analytic
        Closed-form computation for one dimension only. Takes mean,
        covariance, lower, upper, and order; has no solver, config, or seed.

OPTIONS
    order
        Select moments with Order.ZEROTH, Order.FIRST, and Order.SECOND.
        Combine with |; Order.ALL is the default, Order.NONE selects nothing.

    config
        SamplingConfig for Monte Carlo/Halton; GenzConfig for Genz.
        samples: sample budget (default 100000; total across Genz shifts).
        batch_size: buffer batch size (default 4096).
        shifts: Genz random shifts (default 10; at least 2).

    seed
        Random seed for sampling methods (default 42). A solver advances its
        random stream across calls; recreate it to reproduce the sequence.

RETURN VALUE
    Result.zeroth: P(X in S).
    Result.first: E[X 1{X in S}], a NumPy vector.
    Result.second: E[X X^T 1{X in S}], a NumPy matrix.
    The truncated mean is first / zeroth when zeroth > 0.
    Unselected moments are None (zeroth) or empty arrays (first, second).
    Only Genz fills zeroth_error, first_error, and second_error.
    samples and accepted record sample counts.

ERRORS
    ValueError
        Invalid shapes, values, bounds, covariance, configuration, or solver
        capacity; analytic also rejects dimensions other than one.

    RuntimeError
        Numerical failure, including no accepted samples in rejection methods.

EXAMPLES
    import numpy as np
    from mvn_moments import MomentsSolver, moments

    n = 5
    mean = np.zeros(n)
    covariance = 0.5 * np.eye(n) + 0.5  # Unit variances, correlation 0.5.
    lower, upper = np.full(n, -1.0), np.full(n, 2.0)

    # One call.
    result = moments(mean, covariance, lower, upper)
    print(result.zeroth, result.first / result.zeroth)

    # Repeated calls, e.g. in a loop: allocate once, then compute.
    solver = MomentsSolver(seed=42)
    result = solver.compute(mean, covariance, lower, upper)

SEE ALSO
    https://github.com/aamanku/mvn-moments
    https://github.com/aamanku/mvn-moments/blob/main/docs/api.md
    https://github.com/aamanku/mvn-moments/blob/main/docs/methods.md""")
