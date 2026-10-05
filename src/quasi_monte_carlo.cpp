// Halton quasi-Monte Carlo: the rejection estimator in sampler.cpp with
// standard normals built from a randomly shifted Halton sequence.
//
// Radical inverse. Write k = sum_j d_j b^j in base b; then
//
//     phi_b(k) = sum_j d_j b^-(j+1),
//
// e.g. phi_2(6) = phi_2(110_2) = 0.011_2 = 3/8. The Halton point k uses the
// first d primes as bases, (phi_2(k), phi_3(k), phi_5(k), ...), and fills
// [0, 1)^d far more evenly than random points (Halton 1960,
// https://doi.org/10.1007/BF01386213). Points start at k = 1, skipping the
// origin.
//
// Random shift. Each call draws Delta ~ U[0, 1)^d and uses
// u_k = frac(phi(k) + Delta), which makes every point uniform while keeping
// the sequence's evenness (Cranley and Patterson 1976,
// https://doi.org/10.1137/0713071).
//
// Box-Muller. Pairs of uniforms (u1, u2) become independent standard normals
//
//     z1 = sqrt(-2 ln u1) cos(2 pi u2),  z2 = sqrt(-2 ln u1) sin(2 pi u2)
//
// (Box and Muller 1958, https://doi.org/10.1214/aoms/1177706645), so d is
// rounded up to an even number; u1 is clamped away from 0.
//
// For smooth integrands such points converge like O((log N)^d / N) instead of
// O(N^-1/2). The rectangle indicator is discontinuous, so the gain here is
// smaller and shrinks as the dimension grows.

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "mvn_moments/detail/common.hpp"
#include "mvn_moments/detail/primes.hpp"

namespace mvn {
namespace {
constexpr double kTwoPi = 2 * EIGEN_PI;

// The first count primes, used as Halton bases.
Status PrimeBases(std::size_t count, std::vector<std::size_t>& primes)
{
    primes.clear();
    primes.reserve(count);
    for (std::uint64_t candidate = 2; candidate < detail::kMaxPrime;
         ++candidate) {
        if (primes.size() >= count) {
            break;
        }
        if (detail::IsPrime(candidate)) {
            primes.push_back(candidate);
        }
    }
    if (primes.size() != count) {
        return Status::kInvalidInput;
    }

    return Status::kOk;
}

double RadicalInverse(std::size_t index, std::size_t base)
{
    double value = 0, place = 1.0 / base;
    // Base >= 2: at most one step per bit of the original index, so the loop
    // always consumes every digit.
    constexpr int kMaxDigits = std::numeric_limits<std::size_t>::digits;
    for (int digit = 0; digit < kMaxDigits; ++digit) {
        if (index == 0) {
            break;
        }
        value += (index % base) * place;
        index /= base;
        place /= base;
    }

    return value;
}

void DrawHalton(detail::SamplingState& state, Eigen::Ref<Eigen::MatrixXd> z,
                std::size_t completed)
{
    // One draw per row. Box-Muller consumes uniforms in pairs.
    const auto dimensions = z.cols() + z.cols() % 2;
    if (completed == 0) {
        // Each call draws a new random shift.
        std::uniform_real_distribution<double> uniform;
        for (Eigen::Index i = 0; i < dimensions; ++i) {
            state.shift(i) = uniform(state.rng);
        }
    }

    auto uniforms = state.uniforms.topLeftCorner(z.rows(), dimensions);
    for (Eigen::Index col = 0; col < dimensions; ++col) {
        for (Eigen::Index row = 0; row < z.rows(); ++row) {
            uniforms(row, col) = std::fmod(
                RadicalInverse(completed + static_cast<std::size_t>(row) + 1,
                               state.bases[col]) +
                    state.shift(col),
                1.0);
        }
    }

    // Vectorized Box-Muller into retained columns.
    auto radius = state.radius.head(z.rows());
    auto angle = state.angle.head(z.rows());
    for (Eigen::Index col = 0; col < z.cols(); col += 2) {
        radius.array() = (-2 * uniforms.col(col)
                                   .array()
                                   .max(std::numeric_limits<double>::epsilon())
                                   .log())
                             .sqrt();
        angle.array() = kTwoPi * uniforms.col(col + 1).array();
        z.col(col) = (radius.array() * angle.array().cos()).matrix();
        if (col + 1 < z.cols()) {
            z.col(col + 1) = (radius.array() * angle.array().sin()).matrix();
        }
    }
}
}  // namespace

HaltonSolver::HaltonSolver() = default;
HaltonSolver::~HaltonSolver() = default;
HaltonSolver::HaltonSolver(HaltonSolver&&) noexcept = default;
HaltonSolver& HaltonSolver::operator=(HaltonSolver&&) noexcept = default;

Status HaltonSolver::Init(Eigen::Index max_dimension,
                          const SamplingConfig& config, std::uint64_t seed)
{
    state_.reset();
    result_ = Result();
    if (max_dimension <= 0 ||
        max_dimension == std::numeric_limits<Eigen::Index>::max()) {
        return Status::kInvalidInput;
    }

    const auto columns = max_dimension + max_dimension % 2;
    auto state = std::make_unique<detail::SamplingState>();
    Status status =
        detail::InitSampling(max_dimension, config, seed, columns, *state);
    if (status != Status::kOk) {
        return status;
    }
    status = PrimeBases(static_cast<std::size_t>(columns), state->bases);
    if (status != Status::kOk) {
        return status;
    }

    state_ = std::move(state);
    result_ = Result(max_dimension);
    return Status::kOk;
}

Status HaltonSolver::Compute(
    const Eigen::Ref<const Eigen::VectorXd>& mean,
    const Eigen::Ref<const Eigen::MatrixXd>& covariance,
    const Eigen::Ref<const Eigen::VectorXd>& lower,
    const Eigen::Ref<const Eigen::VectorXd>& upper, Order order)
{
    if (!state_) {
        return Status::kInvalidInput;
    }

    MVN_BENCHMARK_TIMER(timer, state_->config.timing);
    MVN_SCOPE(timer, "quasi_monte_carlo");

    return detail::RunSampling(mean, covariance, lower, upper, order,
                               DrawHalton, *state_, result_,
                               MVN_TIMER_ARG(timer));
}

const Result& HaltonSolver::GetResult() const
{
    return result_;
}
}  // namespace mvn
