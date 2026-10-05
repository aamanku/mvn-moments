#ifndef HELPER_HPP_
#define HELPER_HPP_

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include "moments.hpp"
#include "timer.hpp"

namespace mvn {
struct Hyperrectangle {
    Eigen::VectorXd lower;
    Eigen::VectorXd upper;
};

struct NormalDistribution {
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
};

namespace detail {
// Check products before allocating: Eigen::Index is signed.
inline Status ValidateSize(Eigen::Index rows, Eigen::Index cols)
{
    const auto max_entries = std::numeric_limits<Eigen::Index>::max() /
                             static_cast<Eigen::Index>(sizeof(double));
    if (rows < 0 || cols <= 0 || rows > max_entries / cols) {
        return Status::kInvalidInput;
    }

    return Status::kOk;
}

// Elementwise |a-b| <= atol + rtol*max(|a|, |b|) for a nonempty square
// matrix; the caller validates shape and finiteness.
inline bool Symmetric(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                      double rtol, double atol)
{
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
        for (Eigen::Index j = 0; j < i; ++j) {
            const long double a = matrix(i, j), b = matrix(j, i);
            if (std::abs(a - b) > static_cast<long double>(atol) +
                                      static_cast<long double>(rtol) *
                                          std::max(std::abs(a), std::abs(b))) {
                return false;
            }
        }
    }

    return true;
}

// Overwrites the lower triangle of matrix with its Cholesky factor, in place
// and without allocating. The lower triangle is authoritative; the caller
// validates shape, finiteness, and symmetry. The only Cholesky entry point.
inline Status FactorCholesky(Eigen::Ref<Eigen::MatrixXd> matrix)
{
    const Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> factor(matrix);
    if (factor.info() != Eigen::Success) {
        return Status::kNotPositiveDefinite;
    }
    if (!matrix.allFinite()) {
        return Status::kNumericalError;
    }

    return Status::kOk;
}

inline void ValidateMatrix(const Eigen::Ref<const Eigen::MatrixXd>& matrix)
{
    if (matrix.rows() == 0 || matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("Expected a nonempty square matrix");
    }
    if (!matrix.allFinite()) {
        throw std::invalid_argument("Matrix entries must be finite");
    }
}
}  // namespace detail

// The functions below are setup, test, and benchmark utilities: they allocate
// and throw, so keep them out of real-time paths.

// Invalid shape/nonfinite values/tolerances throw; asymmetry returns false.
// Elementwise comparison: |a-b| <= atol + rtol*max(|a|, |b|).
inline bool IsSymmetric(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                        double rtol = 1e-12, double atol = 0.0)
{
    detail::ValidateMatrix(matrix);
    if (!std::isfinite(rtol) || !std::isfinite(atol) || rtol < 0 || atol < 0) {
        throw std::invalid_argument(
            "Symmetry tolerances must be finite and nonnegative");
    }

    return detail::Symmetric(matrix, rtol, atol);
}

// Invalid inputs and arithmetic overflow throw; non-SPD matrices return false.
// Numerical Cholesky test, without jitter. Lower triangle is authoritative
// within symmetry tolerance. Near-singular inputs may fail at double precision.
inline bool IsPositiveDefinite(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                               double rtol = 1e-12, double atol = 0.0)
{
    if (!IsSymmetric(matrix, rtol, atol)) {
        return false;
    }

    Eigen::MatrixXd factor = matrix;
    const Status status = detail::FactorCholesky(factor);
    if (status == Status::kNumericalError) {
        throw std::runtime_error(
            "Cholesky arithmetic overflow; rescale the matrix");
    }

    return status == Status::kOk;
}

// Computes the lower-triangular Cholesky factor of a symmetric positive
// definite matrix.
inline Eigen::MatrixXd CholeskyLower(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix, Timer* timer = nullptr)
{
    (void)timer;
    MVN_SCOPE(timer, "cholesky_lower");
    if (!IsSymmetric(matrix)) {
        throw std::invalid_argument("Covariance must be symmetric");
    }

    Eigen::MatrixXd factor = matrix;
    const Status status = detail::FactorCholesky(factor);
    if (status != Status::kOk) {
        throw std::invalid_argument(StatusMessage(status));
    }

    return factor.triangularView<Eigen::Lower>();
}

// Caller owns/seeds RNG. Centers in [-1, 1], half-widths in [0.25, 2].
inline Hyperrectangle GenerateHyperrectangle(Eigen::Index n,
                                             std::mt19937_64& rng)
{
    if (n <= 0) {
        throw std::invalid_argument("Rectangle dimension must be positive");
    }

    if (detail::ValidateSize(n, 1) != Status::kOk) {
        throw std::invalid_argument("Rectangle dimension is too large");
    }
    Hyperrectangle rectangle{Eigen::VectorXd(n), Eigen::VectorXd(n)};
    std::uniform_real_distribution<double> center(-1.0, 1.0),
        half_width(0.25, 2.0);

    for (Eigen::Index i = 0; i < n; ++i) {
        const double c = center(rng), w = half_width(rng);
        rectangle.lower(i) = c - w;
        rectangle.upper(i) = c + w;
    }

    return rectangle;
}

// Generates parameters, not samples. Covariance = A*A^T/n + I.
// A convenient fixture, not a uniform draw over SPD matrices.
inline NormalDistribution GenerateMvn(Eigen::Index n, std::mt19937_64& rng,
                                      Timer* timer = nullptr)
{
    (void)timer;
    MVN_SCOPE(timer, "generate_mvn");
    if (n <= 0) {
        throw std::invalid_argument("Normal dimension must be positive");
    }

    if (detail::ValidateSize(n, n) != Status::kOk) {
        throw std::invalid_argument("Normal dimension is too large");
    }
    NormalDistribution distribution{Eigen::VectorXd(n), Eigen::MatrixXd(n, n)};
    Eigen::MatrixXd a(n, n);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);

    for (Eigen::Index i = 0; i < n; ++i) {
        distribution.mean(i) = uniform(rng);
    }

    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
            a(i, j) = uniform(rng);
        }
    }

    distribution.covariance.noalias() =
        (a * a.transpose()) / static_cast<double>(n);
    distribution.covariance.diagonal().array() += 1.0;
    return distribution;
}

// Returns count x n samples, one sample per row. Factors once.
inline Eigen::MatrixXd SampleMvn(const Eigen::VectorXd& mean,
                                 const Eigen::MatrixXd& covariance,
                                 Eigen::Index count, std::mt19937_64& rng,
                                 Timer* timer = nullptr)
{
    (void)timer;
    MVN_SCOPE(timer, "sample_mvn");
    const auto n = mean.size();
    if (n == 0 || covariance.rows() != n || covariance.cols() != n) {
        throw std::invalid_argument(
            "Mean and covariance dimensions must agree and be nonempty");
    }
    if (!mean.allFinite()) {
        throw std::invalid_argument("Mean must be finite");
    }

    if (detail::ValidateSize(count, n) != Status::kOk) {
        throw std::invalid_argument("Invalid or oversized sample count");
    }

    const auto lower = CholeskyLower(covariance, timer);
    Eigen::MatrixXd samples(count, n);
    Eigen::VectorXd z(n);
    std::normal_distribution<double> normal;

    for (Eigen::Index row = 0; row < count; ++row) {
        for (Eigen::Index i = 0; i < n; ++i) {
            z(i) = normal(rng);
        }
        samples.row(row) =
            (mean + lower.triangularView<Eigen::Lower>() * z).transpose();
    }
    if (!samples.allFinite()) {
        throw std::runtime_error("MVN sample overflow");
    }

    return samples;
}
}  // namespace mvn

#endif  // HELPER_HPP_
