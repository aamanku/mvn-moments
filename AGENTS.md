# C++ conventions

Use Google C++ style with the repository formatting overrides: function opening
braces go on a new line, including empty functions. Indent with four spaces;
never use literal tabs. Always use braces for
control-flow bodies, including
single-statement bodies. Use the repository `.clang-format`; do not omit braces
for brevity. C++ function names use PascalCase; enum values use a `k` prefix.
Keep Python public names in snake_case.

Keep changes minimal, readable, and explicit about errors. Benchmark
instrumentation uses the existing compile-time macros.

After C++ changes, build and run CTest in normal and benchmark configurations.
Check formatting with `clang-format --dry-run --Werror` on changed C++ files.

# Bounded loops

All loops must have an explicit finite bound. Do not use `while` or `do-while`
loops, or open-ended `for` loops. Use a bounded `for` loop and terminate early
with `break` when the original continuation condition becomes false. Derive
bounds from input sizes or a documented numeric limit; guard against overflow.
Range-based loops over finite containers are allowed. If a search exhausts its
bound without finding the required result, report an explicit error. For loops
whose bound is a termination guard (including iterative replacements for
potentially unbounded recursion), log a warning if the bound is exhausted while
the continuation condition remains true. Do not warn when the work completes
on the final permitted iteration, or for loops intended to traverse all inputs
or execute a fixed number of iterations.

# Visual grouping

Use a single blank line between logical clusters so code is easy to skim:
validation, setup, computation, result assignment, and error checks/return.
Keep closely related statements together. Separate function/type definitions and
successive computation phases; do not add a blank line after every statement or
inside a tightly coupled expression. Preserve these groups when formatting.
