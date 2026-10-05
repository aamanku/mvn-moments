# Development

Conventions are in [`AGENTS.md`](../AGENTS.md): Google C++ style with the
repository `.clang-format`, status codes instead of exceptions, and bounded
loops.

## Tests

Run CTest in both configurations and check formatting:

```sh
cmake -S . -B build && cmake --build build && ctest --test-dir build
cmake -S . -B build-benchmark -DMVN_ENABLE_BENCHMARKS=ON
cmake --build build-benchmark && ctest --test-dir build-benchmark
clang-format --dry-run --Werror include/mvn_moments/*.hpp \
    include/mvn_moments/detail/*.hpp src/*.cpp bindings/*.cpp \
    examples/*.cpp tests/*.hpp tests/*.cpp
python -m pip install . && python tests/python_smoke.py
```

| Test | Covers |
| --- | --- |
| `smoke.cpp` | `Order`, timer macros |
| `helpers.cpp` | `helper.hpp` |
| `numerical.cpp` | All methods: analytic references, validation statuses, batch invariance, solver reuse |
| `genz.cpp` | Normal quantile, lattice quality vs SciPy, exact probabilities, permutation invariance |
| `realtime.cpp` | No heap allocation in `Compute` (test builds define `EIGEN_RUNTIME_NO_MALLOC`) |
| `python_smoke.py` | Python bindings |

## Benchmarks

```sh
./build-benchmark/mvn_benchmark > results.csv 2> timings.log
python examples/benchmark.py > python-results.csv 2> python-timings.log
```

Both solve three random problems in dimensions 1, 2, 4, and 8 with every
method, reusing one solver for 1000 calls, and write per-call times to stderr.

## Accuracy comparison

```sh
pip install -r examples/requirements.txt
python examples/compare_accuracy.py --output-dir build/accuracy
```

Compares every method, and SciPy's `multivariate_normal.cdf` for probability,
against quadrature references across sample budgets. Writes accuracy,
error-versus-time, and method-selection plots plus CSV files.
