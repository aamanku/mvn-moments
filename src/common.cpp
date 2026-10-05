// Shared pieces of every method: input validation, result bookkeeping, the
// Cholesky factor of the covariance, and moment accumulation.
//
// Cholesky. A symmetric positive definite Sigma factors uniquely as
// Sigma = L L^T with L lower triangular and L_ii > 0; column by column,
//
//     L_jj = sqrt(Sigma_jj - sum_{k<j} L_jk^2)
//     L_ij = (Sigma_ij - sum_{k<j} L_ik L_jk) / L_jj,   i > j.
//
// A non-positive radicand means Sigma is not positive definite. Eigen's LLT
// computes this in place on preallocated storage (FactorCholesky in
// helper.hpp); https://eigen.tuxfamily.org/dox/classEigen_1_1LLT.html
//
// Accumulation. For a batch X with one sample x_r per row and weights w_r
// (1/0 for rejection sampling, Genz's f(u) otherwise),
//
//     first_j    += sum_r w_r X_rj        = X.col(j) . w
//     second_ij  += sum_r w_r X_ri X_rj   = X.col(i) . (w .* X.col(j)),
//
// for i >= j only; MirrorLower copies the lower triangle up so M2 is exactly
// symmetric.

#include "mvn_moments/detail/common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "mvn_moments/helper.hpp"

namespace mvn {
Result::Result(Eigen::Index max_dimension)
    : first_storage(Eigen::VectorXd::Zero(max_dimension)),
      second_storage(Eigen::MatrixXd::Zero(max_dimension, max_dimension)),
      first_error_storage(Eigen::VectorXd::Zero(max_dimension)),
      second_error_storage(Eigen::MatrixXd::Zero(max_dimension, max_dimension))
{
}
}  // namespace mvn

namespace mvn::detail {
Status ValidateInputs(const Eigen::Ref<const Eigen::VectorXd>& mean,
                      const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                      const Eigen::Ref<const Eigen::VectorXd>& lower,
                      const Eigen::Ref<const Eigen::VectorXd>& upper,
                      Order order, Eigen::Index capacity, Timer* timer)
{
    (void)timer;
    MVN_SCOPE(timer, "validate_inputs");
    const auto n = mean.size();

    if (n == 0 || n > capacity || covariance.rows() != n ||
        covariance.cols() != n || lower.size() != n || upper.size() != n) {
        return Status::kInvalidInput;
    }

    if ((static_cast<unsigned>(order) & ~static_cast<unsigned>(Order::kAll)) !=
        0) {
        return Status::kInvalidInput;
    }

    // Bounds may be infinite to express half-open or unbounded rectangles.
    if (!mean.allFinite() || !covariance.allFinite()) {
        return Status::kInvalidInput;
    }

    for (Eigen::Index i = 0; i < n; ++i) {
        if (std::isnan(lower(i)) || std::isnan(upper(i)) ||
            lower(i) > upper(i)) {
            return Status::kInvalidInput;
        }
    }

    if (!Symmetric(covariance, 1e-12, 0.0)) {
        return Status::kInvalidInput;
    }

    return Status::kOk;
}

Status ValidateBudget(std::size_t samples, std::size_t batch_size)
{
    if (samples == 0 || batch_size == 0) {
        return Status::kInvalidInput;
    }

    return Status::kOk;
}

void StartResult(Order order, Eigen::Index n, bool errors, Result& result)
{
    result.samples = 0;
    result.accepted = 0;
    result.dimension = n;
    result.order = order;
    result.errors = errors;

    const bool zeroth = HasOrder(order, Order::kZeroth);
    result.zeroth.reset();
    result.zeroth_error.reset();
    if (zeroth) {
        result.zeroth = 0.0;
    }
    if (zeroth && errors) {
        result.zeroth_error = 0.0;
    }

    result.first_storage.head(n).setZero();
    result.second_storage.topLeftCorner(n, n).setZero();
    result.first_error_storage.head(n).setZero();
    result.second_error_storage.topLeftCorner(n, n).setZero();
}

void ClearResult(Result& result)
{
    result.samples = 0;
    result.accepted = 0;
    result.zeroth.reset();
    result.zeroth_error.reset();
    result.dimension = 0;
    result.order = Order::kNone;
    result.errors = false;
}

bool NothingToCompute(Order order,
                      const Eigen::Ref<const Eigen::VectorXd>& lower,
                      const Eigen::Ref<const Eigen::VectorXd>& upper)
{
    return order == Order::kNone || (lower.array() == upper.array()).any();
}

Status CheckFinite(const Result& result)
{
    if ((result.zeroth && !std::isfinite(*result.zeroth)) ||
        !result.First().allFinite() || !result.Second().allFinite()) {
        return Status::kNumericalError;
    }

    return Status::kOk;
}

Status CachedCholesky(const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                      CholeskyCache& cache, Timer* timer)
{
    (void)timer;
    const auto n = covariance.rows();
    if (cache.dimension == n &&
        cache.covariance.topLeftCorner(n, n) == covariance) {
        return Status::kOk;
    }

    MVN_SCOPE(timer, "cholesky");
    // Invalidate first so a failed factorization never leaves a stale hit.
    cache.dimension = 0;
    auto factor = cache.factor.topLeftCorner(n, n);
    factor = covariance;
    const Status status = FactorCholesky(factor);
    if (status != Status::kOk) {
        return status;
    }

    cache.covariance.topLeftCorner(n, n) = covariance;
    cache.dimension = n;
    return Status::kOk;
}

void Accumulate(const Eigen::Ref<const Eigen::MatrixXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& weight,
                Eigen::Ref<Eigen::VectorXd> first,
                Eigen::Ref<Eigen::MatrixXd> second)
{
    for (Eigen::Index j = 0; j < first.size(); ++j) {
        first(j) += x.col(j).dot(weight);
    }
    for (Eigen::Index j = 0; j < second.cols(); ++j) {
        for (Eigen::Index i = j; i < second.rows(); ++i) {
            second(i, j) += x.col(i).dot(weight.cwiseProduct(x.col(j)));
        }
    }
}

void MirrorLower(Eigen::Ref<Eigen::MatrixXd> matrix)
{
    for (Eigen::Index col = 1; col < matrix.cols(); ++col) {
        for (Eigen::Index row = 0; row < col; ++row) {
            matrix(row, col) = matrix(col, row);
        }
    }
}

Status InitSampling(Eigen::Index max_dimension, const SamplingConfig& config,
                    std::uint64_t seed, Eigen::Index uniform_columns,
                    SamplingState& state)
{
    if (max_dimension <= 0 || uniform_columns < 0) {
        return Status::kInvalidInput;
    }
    const Status budget = ValidateBudget(config.samples, config.batch_size);
    if (budget != Status::kOk) {
        return budget;
    }

    const auto capacity_size = std::min(config.samples, config.batch_size);
    if (capacity_size >
        static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max())) {
        return Status::kInvalidInput;
    }
    const auto capacity = static_cast<Eigen::Index>(capacity_size);
    if (ValidateSize(max_dimension, max_dimension) != Status::kOk ||
        ValidateSize(max_dimension, capacity) != Status::kOk ||
        ValidateSize(uniform_columns, capacity) != Status::kOk) {
        return Status::kInvalidInput;
    }

    state.config = config;
    state.capacity = capacity;
    state.rng.seed(seed);
    state.normal.reset();

    state.cholesky.covariance.resize(max_dimension, max_dimension);
    state.cholesky.factor.resize(max_dimension, max_dimension);
    state.cholesky.dimension = 0;

    state.normals.resize(capacity, max_dimension);
    state.samples.resize(capacity, max_dimension);
    state.inside.resize(capacity);

    const auto halton_rows = uniform_columns == 0 ? 0 : capacity;
    state.uniforms.resize(halton_rows, uniform_columns);
    state.shift.resize(uniform_columns);
    state.radius.resize(halton_rows);
    state.angle.resize(halton_rows);
    return Status::kOk;
}
}  // namespace mvn::detail
