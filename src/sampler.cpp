// Rejection estimator shared by Monte Carlo and Halton quasi-Monte Carlo.
//
// Factor Sigma = L L^T (Cholesky, lower triangular L). If z ~ N(0, I), then
// x = mu + L z ~ N(mu, Sigma), so each moment is an expectation over z:
//
//     Z  = E[1{x in S}],  m1 = E[x 1{x in S}],  M2 = E[x x^T 1{x in S}].
//
// Given N standard-normal draws z_k (pseudo-random or quasi-random), the
// estimator averages over ALL draws, counting rejected ones as zero:
//
//     Z_hat  = (1/N) sum_k 1{x_k in S}
//     m1_hat = (1/N) sum_k 1{x_k in S} x_k
//     M2_hat = (1/N) sum_k 1{x_k in S} x_k x_k^T
//
// With pseudo-random draws these are unbiased, with standard error
// O(N^-1/2); for Z it is sqrt(Z (1 - Z) / N). About N Z draws land in S, so a
// rectangle with Z << 1/N is usually never entered (kNoAcceptedSamples); Genz
// integration (src/genz.cpp) avoids rejection altogether.
//
// Implementation: draws come in batches stored one draw per row, so each
// variable is a contiguous column. Then x_i = mu_i + sum_{k<=i} L_ik z_k, the
// rectangle test, and the sums (Accumulate in common.cpp) are vectorized
// column operations and dot products that need no heap memory.

#include <algorithm>

#include "mvn_moments/detail/common.hpp"

namespace mvn::detail {
namespace {
Status Sample(const Eigen::Ref<const Eigen::VectorXd>& mean,
              const Eigen::Ref<const Eigen::MatrixXd>& covariance,
              const Eigen::Ref<const Eigen::VectorXd>& lower,
              const Eigen::Ref<const Eigen::VectorXd>& upper, Order order,
              Sampler sampler, SamplingState& state, Result& result,
              Timer* timer)
{
    (void)timer;
    Status status = ValidateInputs(mean, covariance, lower, upper, order,
                                   result.Capacity(), timer);
    if (status != Status::kOk) {
        return status;
    }
    status = CachedCholesky(covariance, state.cholesky, timer);
    if (status != Status::kOk) {
        return status;
    }

    const auto n = mean.size();
    StartResult(order, n, false, result);
    if (NothingToCompute(order, lower, upper)) {
        return Status::kOk;
    }

    const auto samples = state.config.samples;
    const auto batch_size = static_cast<std::size_t>(state.capacity);
    const auto factor = state.cholesky.factor.topLeftCorner(n, n);
    auto first = result.first_storage.head(result.First().size());
    auto second = result.second_storage.topLeftCorner(result.Second().rows(),
                                                      result.Second().cols());

    {
        MVN_SCOPE(timer, "sample_and_accumulate");
        std::size_t completed = 0;

        // Overflow-safe ceiling division gives the maximum number of batches.
        const auto batches = samples / batch_size + (samples % batch_size != 0);
        for (std::size_t batch = 0; batch < batches; ++batch) {
            if (completed >= samples) {
                break;
            }

            const auto count = static_cast<Eigen::Index>(
                std::min(samples - completed, batch_size));
            auto z = state.normals.topLeftCorner(count, n);
            auto x = state.samples.topLeftCorner(count, n);
            auto inside = state.inside.head(count);

            {
                MVN_SCOPE(timer, "generate_normals");
                sampler(state, z, completed);
            }

            {
                // x_i = mean_i + sum_k L(i, k) z_k on contiguous columns.
                MVN_SCOPE(timer, "transform");
                for (Eigen::Index i = 0; i < n; ++i) {
                    x.col(i).setConstant(mean(i));
                    for (Eigen::Index k = 0; k <= i; ++k) {
                        x.col(i) += factor(i, k) * z.col(k);
                    }
                }
            }

            if (!x.allFinite()) {
                return Status::kNumericalError;
            }

            {
                MVN_SCOPE(timer, "filter_and_accumulate");
                inside.setOnes();
                for (Eigen::Index i = 0; i < n; ++i) {
                    inside.array() *= ((x.col(i).array() >= lower(i)) &&
                                       (x.col(i).array() <= upper(i)))
                                          .cast<double>();
                }

                result.accepted += static_cast<std::size_t>(inside.sum());
                Accumulate(x, inside, first, second);
            }

            completed += static_cast<std::size_t>(count);
        }
    }

    result.samples = samples;
    if (result.accepted == 0) {
        return Status::kNoAcceptedSamples;
    }

    const double draws = static_cast<double>(samples);
    if (result.zeroth) {
        *result.zeroth = static_cast<double>(result.accepted) / draws;
    }
    first /= draws;
    second /= draws;
    MirrorLower(second);

    return CheckFinite(result);
}
}  // namespace

Status RunSampling(const Eigen::Ref<const Eigen::VectorXd>& mean,
                   const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                   const Eigen::Ref<const Eigen::VectorXd>& lower,
                   const Eigen::Ref<const Eigen::VectorXd>& upper, Order order,
                   Sampler sampler, SamplingState& state, Result& result,
                   Timer* timer)
{
    const Status status = Sample(mean, covariance, lower, upper, order, sampler,
                                 state, result, timer);
    if (status != Status::kOk) {
        ClearResult(result);
    }

    return status;
}
}  // namespace mvn::detail
