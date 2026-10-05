# Releasing

Releases are built and published by `.github/workflows/wheels.yml`. Every push
and pull request builds the wheels and sdist and runs all tests; only a pushed
`v*` tag publishes to PyPI. Publishing uses PyPI Trusted Publishing, so no API
token is stored anywhere.

## One-time setup

1. **PyPI.** Sign in at https://pypi.org, open
   https://pypi.org/manage/account/publishing/, and add a pending GitHub
   publisher:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `mvn-moments` |
   | Owner | `aamanku` |
   | Repository name | `mvn-moments` |
   | Workflow name | `wheels.yml` |
   | Environment name | `pypi` |

   The first successful upload creates the project, and the pending publisher
   becomes a normal one.

2. **TestPyPI.** Repeat at https://test.pypi.org (a separate account) with
   environment name `testpypi`.

3. **GitHub.** In the repository, open *Settings > Environments* and create
   `pypi` and `testpypi`. Optionally add yourself as a required reviewer on
   `pypi`, so every release waits for a click.

## Release checklist

1. **Choose the version.** Python versions follow PEP 440: `0.1.0a1` (alpha),
   `0.1.0b1` (beta), `0.1.0rc1` (release candidate), `0.1.0` (final). A
   version can be uploaded only once; a broken release must be followed by a
   new version, not a re-upload.

2. **Bump it.** Set `version` in `pyproject.toml`. If the numeric part
   changed, also update `project(mvn_moments VERSION X.Y.Z ...)` in
   `CMakeLists.txt`; CMake accepts only numbers, so it stays `0.1.0` for
   `0.1.0a1`.

3. **Check locally.**

   ```sh
   cmake -S . -B build && cmake --build build && ctest --test-dir build
   cmake -S . -B build-benchmark -DMVN_ENABLE_BENCHMARKS=ON
   cmake --build build-benchmark && ctest --test-dir build-benchmark
   clang-format --dry-run --Werror include/mvn_moments/*.hpp \
       include/mvn_moments/detail/*.hpp src/*.cpp bindings/*.cpp \
       examples/*.cpp tests/*.hpp tests/*.cpp
   python -m pip install . && python tests/python_smoke.py
   ```

4. **Push to `main`** and wait for the *Wheels* workflow to pass on all
   platforms. Linux aarch64, Windows, and macOS are only tested there.

5. **Rehearse on TestPyPI.** In *Actions > Wheels > Run workflow*, pick `main`,
   check *Publish the built distributions to TestPyPI*, and run. Then, in a
   fresh virtual environment:

   ```sh
   pip install --pre -i https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ mvn-moments
   python tests/python_smoke.py
   ```

   TestPyPI also accepts each version only once. To rehearse again after a
   fix, use a new pre-release version (e.g. `0.1.0a2`).

6. **Tag and publish.** The tag must be `v` plus the exact `pyproject.toml`
   version; the publish job stops if they differ.

   ```sh
   git tag v0.1.0a1
   git push origin v0.1.0a1
   ```

7. **Verify.** Check https://pypi.org/project/mvn-moments/ and install in a
   fresh environment: `pip install --pre mvn-moments` while releases are
   pre-releases, `pip install mvn-moments` after a final release.

## If something goes wrong

- **The publish job failed before uploading** (e.g. tag mismatch): delete the
  tag (`git push --delete origin vX`, `git tag -d vX`), fix, and tag again.
- **A bad version reached PyPI:** yank it on PyPI (*Manage > Releases >
  Options > Yank*), which hides it from new installs, and release a new
  version.

## Maintenance

- **New Python versions.** Add the version to `build` under
  `[tool.cibuildwheel]` in `pyproject.toml` and to the classifiers, once
  cibuildwheel builds it by default and NumPy publishes wheels for it (the
  wheel tests need NumPy).
- **Action versions.** The workflow pins `actions/*` to major versions and
  `pypa/cibuildwheel` and `pypa/gh-action-pypi-publish` to exact releases;
  update them when GitHub reports deprecations or new Python versions need a
  newer cibuildwheel.
- **Eigen.** Wheels compile against the checksum-pinned Eigen 3.4.0 in
  `CMakeLists.txt`. When changing it, update the URL, the SHA-256, and the
  license texts in `licenses/eigen`.
