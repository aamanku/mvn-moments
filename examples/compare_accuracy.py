"""Compare raw rectangle moments with SciPy references; save Matplotlib plots.

Outputs accuracy vs budget, accuracy vs time (trade-off), and a per-time-budget
method selection chart with its table.
"""

import argparse
import csv
from pathlib import Path
import sys
from time import perf_counter_ns
import warnings

import matplotlib
matplotlib.use("Agg")  # Select the file backend before importing pyplot.
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, stats

from mvn_moments import (
    GenzConfig, Order, SamplingConfig,
    analytic, genz_quasi_monte_carlo, monte_carlo, quasi_monte_carlo,
)

RUNS = 10
# Half-decade budgets give the time axis enough points to compare methods.
SAMPLE_COUNTS = (100, 300, 1000, 3000, 10000, 30000, 100000)
SEED = 42

METRICS = ("probability", "first", "second")
METRIC_TITLES = ("probability", "first moment", "second moment")
# Fixed categorical order (validated light palette); markers add a non-color cue.
STYLE = {
    "MVN MC": ("#2a78d6", "o", "MC"),
    "MVN QMC": ("#eb6834", "s", "QMC"),
    "MVN Genz": ("#1baf7a", "D", "Genz"),
    "SciPy CDF": ("#eda100", "^", "SciPy"),
    "MVN analytic": ("#e87ba4", "*", "Analytic"),
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
EMPTY_CELL = "#f0efec"
# Errors below this are rounding noise; it is also the log-plot floor.
ERROR_FLOOR = 1e-16
# Time budgets (ms per call) for the selection chart.
TIME_BUDGETS_MS = (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)
# Methods within this factor of the best error count as tied; the faster wins.
TIE_FACTOR = 2.0


def correlated_covariance(sigma):
    # Independent correlated pairs allow reliable 2D quadrature references.
    correlation = np.eye(sigma.size)
    for i in range(0, sigma.size - 1, 2):
        correlation[i, i + 1] = correlation[i + 1, i] = 0.5
    return np.outer(sigma, sigma) * correlation


def problems():
    # Nonzero means and asymmetric bounds exercise first and cross moments.
    cases = [
        ("1D asymmetric", np.array([0.3]), np.array([[1.4]]),
         np.array([-0.7]), np.array([1.6])),
        ("2D correlated", np.array([0.2, -0.4]),
         np.array([[1.0, 0.6], [0.6, 1.5]]),
         np.array([-1.0, -1.5]), np.array([1.4, 0.8])),
        ("3D correlated blocks", np.array([0.3, -0.2, 0.5]),
         correlated_covariance(np.sqrt([1.0, 1.4, 0.8])), np.array([-1.0, -1.3, -0.8]),
         np.array([1.5, 1.0, 1.8])),
    ]
    for dimension in range(4, 11):
        mean = np.linspace(-0.4, 0.6, dimension)
        sigma = np.linspace(0.8, 1.3, dimension)
        # Use correlated coordinates with axis-aligned bounds. Widths are
        # chosen for mass 0.5 under independence; actual correlated rectangle
        # probabilities are estimated separately, not assumed to equal 0.5.
        coordinate_mass = 0.5 ** (1.0 / dimension)
        half_width = stats.norm.ppf((1.0 + coordinate_mass) / 2.0)
        cases.append((f"{dimension}D correlated blocks", mean, correlated_covariance(sigma),
                      mean - half_width * sigma, mean + half_width * sigma))
    return cases


def univariate_moments(mean, variance, lower, upper):
    """Closed-form raw integrals of 1, x, x^2 over a finite interval.

    An independent SciPy reference; deliberately not the library's analytic().
    """
    sigma = np.sqrt(variance)
    a, b = (lower - mean) / sigma, (upper - mean) / sigma
    # Survival differences avoid cancellation for positive-tail intervals.
    probability = (stats.norm.sf(a) - stats.norm.sf(b) if a >= 0
                   else stats.norm.cdf(b) - stats.norm.cdf(a))
    difference = stats.norm.pdf(a) - stats.norm.pdf(b)
    boundary = a * stats.norm.pdf(a) - b * stats.norm.pdf(b)
    first = mean * probability + sigma * difference
    second = ((mean ** 2 + variance) * probability
              + 2 * mean * sigma * difference + variance * boundary)
    return probability, np.array([first]), np.array([[second]])


def quadrature_moments(mean, covariance, lower, upper, tolerance):
    """Adaptive quadrature for a correlated block of at most two dimensions."""
    if mean.size != 2:
        raise ValueError("Quadrature reference requires a 2D block")
    normal = stats.multivariate_normal(mean=mean, cov=covariance)
    errors = []

    def moment(indices):
        def integrand(*coordinates):
            return normal.pdf(coordinates) * np.prod([coordinates[i] for i in indices])
        with warnings.catch_warnings():
            warnings.simplefilter("error", integrate.IntegrationWarning)
            value, error = integrate.nquad(
                integrand, list(zip(lower, upper)),
                opts=dict(epsabs=tolerance, epsrel=tolerance, limit=100))
        errors.append(error)
        return value

    probability = moment(())
    first = np.array([moment((i,)) for i in range(2)])
    second = np.zeros((2, 2))
    for i in range(2):
        for j in range(i + 1):
            second[i, j] = second[j, i] = moment((i, j))
    return probability, first, second, max(errors)


def reference_moments(mean, covariance, lower, upper):
    """Combine exact 1D formulas and checked 2D quadrature across independent blocks."""
    dimension = mean.size
    # Find connected covariance blocks; full higher-dimensional correlation
    # must not silently be treated as independent pairs.
    unseen = set(range(dimension))
    blocks = []
    for start in range(dimension):
        if start not in unseen:
            continue
        block = {start}
        for _ in range(dimension):
            connected = {j for i in block for j in range(dimension)
                         if covariance[i, j] != 0}
            if connected <= block:
                break
            block |= connected
        unseen -= block
        blocks.append(sorted(block))

    probability = 1.0
    conditional_mean = np.zeros(dimension)
    conditional_covariance = np.zeros((dimension, dimension))
    max_error = 0.0
    max_refinement = 0.0
    for block in blocks:
        index = np.ix_(block, block)
        if len(block) == 1:
            i = block[0]
            mass, first, second = univariate_moments(
                mean[i], covariance[i, i], lower[i], upper[i])
        elif len(block) == 2:
            arguments = (mean[block], covariance[index], lower[block], upper[block])
            coarse = quadrature_moments(*arguments, tolerance=1e-10)
            fine = quadrature_moments(*arguments, tolerance=1e-12)
            change = max(float(np.max(np.abs(np.asarray(a) - b)))
                         for a, b in zip(coarse[:3], fine[:3]))
            if change > 1e-10 or fine[3] > 1e-10:
                raise RuntimeError("Quadrature reference failed tolerance/refinement checks")
            max_error = max(max_error, fine[3])
            max_refinement = max(max_refinement, change)
            mass, first, second = fine[:3]
        else:
            raise ValueError("Reference supports independent blocks of size at most 2")
        if mass <= 0:
            raise RuntimeError("Reference probability underflow")
        block_mean = first / mass
        conditional_mean[block] = block_mean
        conditional_covariance[index] = second / mass - np.outer(block_mean, block_mean)
        probability *= mass
    first = probability * conditional_mean
    second = probability * (conditional_covariance + np.outer(conditional_mean, conditional_mean))
    metadata = dict(reference_method="closed-form 1D / adaptive 2D quadrature",
                    max_block_quadrature_error=max_error,
                    max_block_refinement_difference=max_refinement)
    return probability, first, second, metadata


# Method -> (function, config type); analytic has no config and only runs in 1D.
METHODS = {
    "MVN MC": (monte_carlo, SamplingConfig),
    "MVN QMC": (quasi_monte_carlo, SamplingConfig),
    "MVN Genz": (genz_quasi_monte_carlo, GenzConfig),
    "MVN analytic": (analytic, None),
}


def estimate(method, mean, covariance, lower, upper, samples, seed):
    """Return (zeroth, first, second) and reported standard errors or None."""
    if method == "SciPy CDF":
        probability = stats.multivariate_normal.cdf(
            upper, mean=mean, cov=covariance, lower_limit=lower,
            maxpts=samples, abseps=1e-9, releps=1e-9,
            rng=np.random.default_rng(seed))
        # CDF computes probability only; no first/second moments are returned.
        return (probability, None, None), None
    function, config_type = METHODS[method]
    if config_type is None:
        result = function(mean, covariance, lower, upper, order=Order.ALL)
    else:
        config = config_type()
        config.samples = samples
        result = function(mean, covariance, lower, upper, order=Order.ALL,
                          config=config, seed=seed)
    errors = None
    if result.zeroth_error is not None:
        errors = (result.zeroth_error, result.first_error, result.second_error)
    return (result.zeroth, result.first, result.second), errors


def squared_errors(estimate_values, reference):
    # Per-entry squared errors keep vector/matrix metrics comparable in size.
    return np.array([np.mean(np.square(np.asarray(value) - expected)) if value is not None else np.nan
                     for value, expected in zip(estimate_values, reference[:3])])


def measure(case_index, case, method, reference):
    """RMSE and mean call time over RUNS seeds, one record per draw budget."""
    name, mean, covariance, lower, upper = case
    # Analytic has no budget; it is measured once and recorded as 0 samples.
    budgets = (0,) if method == "MVN analytic" else SAMPLE_COUNTS
    # One untimed call absorbs first-call costs (imports, caches).
    try:
        estimate(method, mean, covariance, lower, upper, budgets[0], SEED)
    except RuntimeError:
        pass

    records = []
    for samples in budgets:
        sums = np.zeros(3)
        reported_sums = np.zeros(3)
        reported = False
        failures = 0
        elapsed_ns = 0
        for run in range(RUNS):
            # Common replicate seeds, independent across runs; budgets
            # reuse seeds to avoid changing randomness unnecessarily.
            seed = SEED + case_index * RUNS + run
            try:
                start = perf_counter_ns()
                values, errors = estimate(method, mean, covariance, lower, upper,
                                          samples, seed)
                elapsed_ns += perf_counter_ns() - start
                sums += squared_errors(values, reference)
                if errors is not None:
                    # RMS of reported standard errors, per entry.
                    reported = True
                    reported_sums += squared_errors(errors, (0.0, 0.0, 0.0))
            except RuntimeError as error:
                failures += 1
                if failures == 1:
                    print(f"[warning] {name}, {method}, N={samples}: {error}",
                          file=sys.stderr)
        # Do not hide failed trials by averaging only successes.
        rmse = np.sqrt(sums / RUNS) if failures == 0 else np.full(3, np.nan)
        reported_se = (np.sqrt(reported_sums / RUNS) if reported and failures == 0
                       else np.full(3, np.nan))
        records.append(dict(problem=name, dimension=mean.size, method=method, samples=samples,
                            runs=RUNS, failures=failures,
                            mean_ms=elapsed_ns / RUNS / 1e6 if failures == 0 else np.nan,
                            probability_rmse=rmse[0],
                            first_rmse=rmse[1], second_rmse=rmse[2],
                            probability_reported_se=reported_se[0],
                            first_reported_se=reported_se[1],
                            second_reported_se=reported_se[2]))
    return records


def method_rows(records, problem, method):
    """Timed, successful runs of one method on one problem, cheapest first."""
    rows = [r for r in records if r["problem"] == problem and r["method"] == method
            and np.isfinite(r["mean_ms"])]
    return sorted(rows, key=lambda r: r["mean_ms"])


def add_header(fig, title, handles, columns, fontsize=10):
    """Two-line title and one-row legend in a fixed-height band (inches)."""
    height = fig.get_figheight()
    # Lay out axes first: tight_layout would otherwise reserve suptitle space twice.
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / height))
    fig.suptitle(title, y=1 - 0.08 / height, va="top", fontsize=11, color=TEXT_PRIMARY)
    fig.legend(handles=handles, loc="upper center", ncol=columns, frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.55 / height), fontsize=fontsize)


def legend_handles(methods):
    return [Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], linewidth=2,
                   markersize=7, label=m) for m in methods]


def plot_accuracy(records, cases, output_dir):
    """RMSE against draw budget per problem and moment; analytic error is in the legend."""
    fig, axes = plt.subplots(len(cases), 3, figsize=(13, 3.3 * len(cases)), squeeze=False)
    for row, (name, *_) in enumerate(cases):
        for method, (color, marker, _) in STYLE.items():
            rows = [r for r in records if r["problem"] == name and r["method"] == method]
            samples = [r["samples"] for r in rows]
            for metric, ax in enumerate(axes[row]):
                errors = np.array([r[f"{METRICS[metric]}_rmse"] for r in rows])
                reported = np.array([r[f"{METRICS[metric]}_reported_se"] for r in rows])
                if not rows or (method == "SciPy CDF" and metric != 0):
                    continue
                if method == "MVN analytic":
                    # A legend-only entry; a text box would collide with the legend.
                    ax.plot([], [], " ", label=f"Analytic reference error: {errors[0]:.2g}")
                    continue
                ax.loglog(samples, np.maximum(errors, ERROR_FLOOR),
                          color=color, marker=marker, linewidth=2, label=method)
                if np.isfinite(reported).any():
                    ax.loglog(samples, np.maximum(reported, ERROR_FLOOR),
                              linestyle="--", color=color, label=f"{method} reported SE")
        for metric, ax in enumerate(axes[row]):
            ax.set_title(f"{name}: {METRIC_TITLES[metric]}")
            ax.set_xlabel("Draw budget (MVN) / requested maxpts (SciPy CDF)")
            ax.set_ylabel("RMSE vs formula/quadrature (raw integrals)")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle("Rectangle moments vs formulas and checked 2D quadrature\n"
                 f"{RUNS:,} runs per budget; SciPy CDF: probability only; plot floor 1e-16")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"accuracy.{extension}", dpi=180)
    plt.close(fig)


def plot_tradeoff(records, cases, output_dir):
    """Small multiples of RMSE against mean time per call (lower-left is better)."""
    fig, axes = plt.subplots(len(cases), 3, figsize=(13, 3.0 * len(cases)), squeeze=False)
    for row, (name, *_) in enumerate(cases):
        for method, (color, marker, _) in STYLE.items():
            rows = method_rows(records, name, method)
            if not rows:
                continue
            times = np.array([r["mean_ms"] for r in rows])
            for metric, ax in enumerate(axes[row]):
                errors = np.array([r[f"{METRICS[metric]}_rmse"] for r in rows])
                ok = np.isfinite(errors)
                if ok.any():
                    ax.loglog(times[ok], np.maximum(errors[ok], ERROR_FLOOR), color=color,
                              marker=marker, markersize=6, linewidth=2)
        for metric, ax in enumerate(axes[row]):
            ax.set_title(f"{name}: {METRIC_TITLES[metric]}", color=TEXT_PRIMARY, fontsize=10)
            ax.set_xlabel("Mean time per call (ms)", color=TEXT_SECONDARY)
            ax.set_ylabel("RMSE (raw integral)", color=TEXT_SECONDARY)
            ax.grid(True, which="both", alpha=0.25)
    add_header(fig, "Accuracy vs time: lower-left is better (points are draw budgets "
               f"{SAMPLE_COUNTS[0]:,}–{SAMPLE_COUNTS[-1]:,})\n"
               "Times are Python-call means over runs; SciPy CDF computes probability only; "
               f"errors floored at {ERROR_FLOOR:g}",
               legend_handles(STYLE), len(STYLE))
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"tradeoff.{extension}", dpi=150)
    plt.close(fig)


def achievable(rows, metric, budget_ms):
    """Lowest measured RMSE among runs that finished within the time budget.

    Conservative: a method is credited only with what it was measured to do.
    """
    best = None
    for r in rows:
        error = r[f"{metric}_rmse"]
        if not np.isfinite(error) or r["mean_ms"] > budget_ms:
            continue
        error = max(error, ERROR_FLOOR)
        if best is None or error < best["rmse"]:
            best = dict(rmse=error, time_ms=r["mean_ms"], samples=r["samples"])
    return best


def select(records, problem, metric, budget_ms):
    """Most accurate method within the budget; near-ties go to the faster one."""
    options = {}
    for method in STYLE:
        option = achievable(method_rows(records, problem, method), metric, budget_ms)
        if option is not None:
            options[method] = option
    if not options:
        return None, options
    lowest = min(option["rmse"] for option in options.values())
    tied = [m for m, option in options.items() if option["rmse"] <= TIE_FACTOR * lowest]
    return min(tied, key=lambda m: options[m]["time_ms"]), options


def plot_selection(records, cases, output_dir):
    """Per metric, the recommended method for each problem and time budget."""
    selections = []
    methods = list(STYLE)
    colors = ListedColormap([EMPTY_CELL] + [STYLE[m][0] for m in methods])
    fig, axes = plt.subplots(3, 1, figsize=(12, 4.6 * 3), squeeze=False)
    for metric, ax in enumerate(axes[:, 0]):
        grid = np.zeros((len(cases), len(TIME_BUDGETS_MS)))
        for row, (name, mean, *_) in enumerate(cases):
            for column, budget in enumerate(TIME_BUDGETS_MS):
                winner, options = select(records, name, METRICS[metric], budget)
                record = dict(problem=name, dimension=mean.size, metric=METRICS[metric],
                              time_budget_ms=budget, best_method=winner or "",
                              best_rmse=options[winner]["rmse"] if winner else np.nan,
                              best_time_ms=options[winner]["time_ms"] if winner else np.nan,
                              best_samples=options[winner]["samples"] if winner else "")
                for method in methods:
                    record[f"{STYLE[method][2]}_rmse"] = (options[method]["rmse"]
                                                          if method in options else np.nan)
                selections.append(record)
                if winner is None:
                    ax.text(column, row, "–", ha="center", va="center",
                            color=TEXT_SECONDARY, fontsize=8)
                    continue
                grid[row, column] = methods.index(winner) + 1
                # Text stays in ink colors; the cell color carries identity.
                ax.text(column, row, f"{STYLE[winner][2]}\n{options[winner]['rmse']:.0e}",
                        ha="center", va="center", color=TEXT_PRIMARY, fontsize=7,
                        linespacing=1.1)
        ax.pcolormesh(np.arange(len(TIME_BUDGETS_MS) + 1) - 0.5, np.arange(len(cases) + 1) - 0.5,
                      grid, cmap=colors, vmin=-0.5, vmax=len(methods) + 0.5,
                      edgecolors=SURFACE, linewidth=2)
        ax.set_xlim(-0.5, len(TIME_BUDGETS_MS) - 0.5)
        ax.set_ylim(len(cases) - 0.5, -0.5)
        ax.set_xticks(range(len(TIME_BUDGETS_MS)), [f"{b:g}" for b in TIME_BUDGETS_MS])
        ax.set_yticks(range(len(cases)), [case[0] for case in cases])
        ax.tick_params(length=0, colors=TEXT_SECONDARY)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Time budget per call (ms)", color=TEXT_SECONDARY)
        ax.set_title(f"{METRIC_TITLES[metric].capitalize()}: best method and its RMSE",
                     color=TEXT_PRIMARY, loc="left", fontsize=11)
    add_header(fig, "Which method to use: lowest measured RMSE among runs finishing within "
               f"the budget\nmethods within {TIE_FACTOR:g}x of the best error count as tied; "
               "the faster one is shown",
               [Patch(facecolor=STYLE[m][0], label=m) for m in methods]
               + [Patch(facecolor=EMPTY_CELL, label="nothing finished in time (–)")],
               len(methods) + 1, fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"selection.{extension}", dpi=150)
    plt.close(fig)
    return selections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("build/accuracy"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = problems()

    records = []
    references = []
    for case_index, case in enumerate(cases):
        name, mean, covariance, lower, upper = case
        reference = reference_moments(mean, covariance, lower, upper)
        cdf = stats.multivariate_normal.cdf(
            upper, mean=mean, cov=covariance, lower_limit=lower,
            maxpts=1000000, abseps=1e-9, releps=1e-9,
            rng=np.random.default_rng(SEED))
        references.append({"problem": name, "dimension": mean.size, "probability": reference[0],
                           "scipy_cdf": cdf, **reference[3],
                           "first": reference[1].tolist(), "second": reference[2].tolist()})
        print(f"{name}: reference P={reference[0]:.12g}, "
              f"max block quadrature error={reference[3]['max_block_quadrature_error']:.3g}",
              file=sys.stderr)
        for method in STYLE:
            if method != "MVN analytic" or mean.size == 1:
                records += measure(case_index, case, method, reference)

    plot_accuracy(records, cases, args.output_dir)
    plot_tradeoff(records, cases, args.output_dir)
    selections = plot_selection(records, cases, args.output_dir)
    for name, rows in (("errors.csv", records), ("references.csv", references),
                       ("selection.csv", selections)):
        with (args.output_dir / name).open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 1 if any(r["failures"] for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
