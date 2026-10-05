"""Run after installation: python examples/benchmark.py > results.csv 2> timings.log."""

import csv
import sys
from time import perf_counter_ns

import numpy as np

from mvn_moments import (
    GenzConfig,
    GenzSolver,
    HaltonSolver,
    MonteCarloSolver,
    Order,
    SamplingConfig,
    analytic,
)

RUNS = 1000
PROBLEMS_PER_DIMENSION = 3
SAMPLES = 10000
BATCH_SIZE = 4096
SEED = 42
DIMENSIONS = (1, 2, 4, 8)
# Method name -> (solver type, config type); analytic is a plain function.
METHODS = {
    "analytic": (None, None),
    "monte_carlo": (MonteCarloSolver, SamplingConfig),
    "quasi_monte_carlo": (HaltonSolver, SamplingConfig),
    "genz_quasi_monte_carlo": (GenzSolver, GenzConfig),
}


def generate_problem(dimension, rng):
    """Use the same parameter distributions and SPD construction as C++."""
    mean = rng.uniform(-1.0, 1.0, dimension)
    matrix = rng.uniform(-1.0, 1.0, (dimension, dimension))
    covariance = matrix @ matrix.T / dimension + np.eye(dimension)
    center = rng.uniform(-1.0, 1.0, dimension)
    half_width = rng.uniform(0.25, 2.0, dimension)
    return mean, covariance, center - half_width, center + half_width


def solve(method, problem, seed):
    """Time RUNS calls with identical inputs, including binding cost.

    One solver serves all runs, as in a caller's loop. Its random stream
    advances, so each run draws new samples; runs that fail (e.g. no samples in
    an unlikely rectangle) are counted, and the last successful result is kept.
    """
    solver_type, config_type = METHODS[method]
    if solver_type is None:
        def compute():
            return analytic(*problem, order=Order.ALL)
    else:
        config = config_type()
        config.samples = SAMPLES
        config.batch_size = BATCH_SIZE
        try:
            solver = solver_type(len(problem[0]), config=config, seed=seed)
        except ValueError as error:
            print(f"[error] {method}: {error}", file=sys.stderr)
            return None

        def compute():
            return solver.compute(*problem, order=Order.ALL)

    result = None
    failures = 0
    error = None
    total_ns = 0
    for _ in range(RUNS):
        start = perf_counter_ns()
        try:
            result = compute()
        except (ValueError, RuntimeError) as caught:
            failures += 1
            error = caught
        total_ns += perf_counter_ns() - start

    if failures == RUNS:
        print(f"[error] {method}: {error}", file=sys.stderr)
        return None
    if failures:
        print(f"[warning] {method}: {failures} of {RUNS} runs failed: {error}",
              file=sys.stderr)

    total_ms = total_ns / 1_000_000
    print(f"{method} | calls={RUNS} | total_ms={total_ms:.12g} "
          f"| mean_ms={total_ms / RUNS:.12g}", file=sys.stderr)
    return result


def write_result(writer, dimension, problem, seed, method, result,
                 reference_method, reference):
    row = [dimension, problem, seed, method, RUNS, result.samples,
           result.accepted, result.zeroth, np.linalg.norm(result.first),
           np.linalg.norm(result.second), reference_method]
    if reference is None:
        row.extend(["", "", ""])
    else:
        row.extend([abs(result.zeroth - reference.zeroth),
                    np.linalg.norm(result.first - reference.first),
                    np.linalg.norm(result.second - reference.second)])
    writer.writerow(row)


def main():
    rng = np.random.default_rng(SEED)
    writer = csv.writer(sys.stdout)
    writer.writerow([
        "dimension", "problem", "solve_seed", "method", "runs", "samples",
        "accepted", "probability", "first_norm", "second_norm", "reference",
        "probability_difference", "first_difference_norm", "second_difference_norm",
    ])
    failed = False
    for dimension in DIMENSIONS:
        for problem_index in range(PROBLEMS_PER_DIMENSION):
            problem = generate_problem(dimension, rng)
            seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
            print(f"Problem dimension={dimension} index={problem_index} seed={seed}",
                  file=sys.stderr)
            reference = None
            reference_method = "analytic" if dimension == 1 else "monte_carlo"
            methods = [m for m in METHODS if dimension == 1 or m != "analytic"]
            for method in methods:
                result = solve(method, problem, seed)
                if result is None:
                    failed = True
                    continue
                if method == reference_method:
                    reference = result
                write_result(writer, dimension, problem_index, seed, method,
                             result, reference_method, reference)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
