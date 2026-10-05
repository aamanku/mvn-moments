// Monte Carlo: the rejection estimator in sampler.cpp with pseudo-random
// standard normals z_k from std::normal_distribution over a 64-bit Mersenne
// Twister (std::mt19937_64) seeded in Init. The stream continues across
// Compute calls, so successive estimates are independent.

#include <memory>
#include <utility>

#include "mvn_moments/detail/common.hpp"

namespace mvn {
namespace {
void DrawNormals(detail::SamplingState& state, Eigen::Ref<Eigen::MatrixXd> z,
                 std::size_t)
{
    // One draw per row, its dimensions in order, so draw order is stable
    // across batches; RNG itself is scalar.
    for (Eigen::Index row = 0; row < z.rows(); ++row) {
        for (Eigen::Index col = 0; col < z.cols(); ++col) {
            z(row, col) = state.normal(state.rng);
        }
    }
}
}  // namespace

MonteCarloSolver::MonteCarloSolver() = default;
MonteCarloSolver::~MonteCarloSolver() = default;
MonteCarloSolver::MonteCarloSolver(MonteCarloSolver&&) noexcept = default;
MonteCarloSolver& MonteCarloSolver::operator=(MonteCarloSolver&&) noexcept =
    default;

Status MonteCarloSolver::Init(Eigen::Index max_dimension,
                              const SamplingConfig& config, std::uint64_t seed)
{
    state_.reset();
    result_ = Result();

    auto state = std::make_unique<detail::SamplingState>();
    const Status status =
        detail::InitSampling(max_dimension, config, seed, 0, *state);
    if (status != Status::kOk) {
        return status;
    }

    state_ = std::move(state);
    result_ = Result(max_dimension);
    return Status::kOk;
}

Status MonteCarloSolver::Compute(
    const Eigen::Ref<const Eigen::VectorXd>& mean,
    const Eigen::Ref<const Eigen::MatrixXd>& covariance,
    const Eigen::Ref<const Eigen::VectorXd>& lower,
    const Eigen::Ref<const Eigen::VectorXd>& upper, Order order)
{
    if (!state_) {
        return Status::kInvalidInput;
    }

    MVN_BENCHMARK_TIMER(timer, state_->config.timing);
    MVN_SCOPE(timer, "monte_carlo");

    return detail::RunSampling(mean, covariance, lower, upper, order,
                               DrawNormals, *state_, result_,
                               MVN_TIMER_ARG(timer));
}

const Result& MonteCarloSolver::GetResult() const
{
    return result_;
}
}  // namespace mvn
