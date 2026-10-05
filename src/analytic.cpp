// Closed-form moments of a univariate normal over an interval.
//
// Let X ~ N(mu, sigma^2) and S = [l, u]. Substitute x = mu + sigma t, so t is
// standard normal with density phi(t) = exp(-t^2 / 2) / sqrt(2 pi) and CDF
// Phi, and the interval becomes [a, b] with a = (l - mu) / sigma and
// b = (u - mu) / sigma. Because phi'(t) = -t phi(t),
//
//     int_a^b phi(t) dt     = Phi(b) - Phi(a)                 =: P
//     int_a^b t phi(t) dt   = phi(a) - phi(b)                 =: D
//     int_a^b t^2 phi(t) dt = P + a phi(a) - b phi(b)         =: P + B
//
// (the last by parts: int t (t phi) dt = [-t phi] + int phi). Expanding
// x = mu + sigma t gives the raw moments
//
//     Z  = P
//     m1 = mu P + sigma D
//     m2 = (mu^2 + sigma^2) P + 2 mu sigma D + sigma^2 B.
//
// Infinite bounds contribute phi(+-inf) = 0 and t phi(t) -> 0. Dividing by
// P recovers the truncated-normal mean and variance in Burkardt, "The
// Truncated Normal Distribution", sections 3.6-3.7:
// https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
//
// For very narrow intervals P, D, and B are differences of nearly equal
// numbers. There the integrals of x^k phi((x - mu) / sigma) / sigma over
// [l, u] are evaluated instead by eight-point Gauss-Legendre quadrature,
//
//     int_l^u f(x) dx ~= h sum_i w_i [f(c - h xi_i) + f(c + h xi_i)],
//     c = (l + u) / 2,  h = (u - l) / 2,
//
// which is exact for polynomials of degree 15 and therefore very accurate
// when the density barely varies across the interval. P is computed with
// survival probabilities in the upper tail; see detail/normal.hpp.

#include <algorithm>
#include <cmath>

#include "mvn_moments/detail/common.hpp"
#include "mvn_moments/detail/normal.hpp"

namespace mvn {
namespace {
Status Integrate(const Eigen::Ref<const Eigen::VectorXd>& mean,
                 const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                 const Eigen::Ref<const Eigen::VectorXd>& lower,
                 const Eigen::Ref<const Eigen::VectorXd>& upper, Order order,
                 Result& result, Timer* timer)
{
    const Status status = detail::ValidateInputs(
        mean, covariance, lower, upper, order, result.Capacity(), timer);
    if (status != Status::kOk) {
        return status;
    }
    if (mean.size() != 1) {
        return Status::kInvalidInput;
    }
    if (!(covariance(0, 0) > 0)) {
        return Status::kNotPositiveDefinite;
    }

    detail::StartResult(order, 1, false, result);
    if (detail::NothingToCompute(order, lower, upper)) {
        return Status::kOk;
    }

    // Long double reduces cancellation in narrow intervals and raw moments.
    const long double mu = mean(0), sigma = std::sqrt(covariance(0, 0));
    const long double a = (static_cast<long double>(lower(0)) - mu) / sigma;
    const long double b = (static_cast<long double>(upper(0)) - mu) / sigma;
    long double probability = 0, first = 0, second = 0;

    // Closed-form differences lose precision for very narrow intervals.
    // An eight-point Gauss-Legendre evaluation avoids subtracting nearly equal
    // CDFs and boundary terms; used only when the density varies very little.
    if (std::isfinite(a) && std::isfinite(b) &&
        b - a < 1e-3L / (1 + std::max(std::abs(a), std::abs(b)))) {
        // Positive nodes xi_i and weights w_i of the symmetric rule.
        constexpr long double nodes[] = {
            0.183434642495649805L, 0.525532409916328986L, 0.796666477413626740L,
            0.960289856497536232L};
        constexpr long double weights[] = {
            0.362683783378361983L, 0.313706645877887287L, 0.222381034453374471L,
            0.101228536290376259L};

        const long double center =
            (static_cast<long double>(lower(0)) + upper(0)) / 2;
        const long double half =
            (static_cast<long double>(upper(0)) - lower(0)) / 2;

        for (int i = 0; i < 4; ++i) {
            for (int sign : {-1, 1}) {
                const long double x = center + sign * half * nodes[i];
                const long double weight = weights[i] * half / sigma *
                                           detail::NormalPdf((x - mu) / sigma);
                probability += weight;
                first += weight * x;
                second += weight * x * x;
            }
        }
    } else {
        const long double pa = detail::NormalPdf(a);
        const long double pb = detail::NormalPdf(b);
        const long double difference = pa - pb;
        const long double boundary =
            (std::isfinite(a) ? a * pa : 0) - (std::isfinite(b) ? b * pb : 0);

        probability = detail::IntervalProbability(a, b);
        first = mu * probability + sigma * difference;
        second = (mu * mu + sigma * sigma) * probability +
                 2 * mu * sigma * difference + sigma * sigma * boundary;
    }

    // Probability underflow or interval cancellation.
    if (probability <= 0 || static_cast<double>(probability) == 0) {
        return Status::kNumericalError;
    }

    if (result.zeroth) {
        *result.zeroth = static_cast<double>(probability);
    }
    if (HasOrder(order, Order::kFirst)) {
        result.first_storage(0) = static_cast<double>(first);
    }
    if (HasOrder(order, Order::kSecond)) {
        result.second_storage(0, 0) = static_cast<double>(second);
    }

    return detail::CheckFinite(result);
}
}  // namespace

Status Analytic(const Eigen::Ref<const Eigen::VectorXd>& mean,
                const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                const Eigen::Ref<const Eigen::VectorXd>& lower,
                const Eigen::Ref<const Eigen::VectorXd>& upper, Order order,
                Result& result, bool timing)
{
    (void)timing;
    MVN_BENCHMARK_TIMER(timer, timing);
    MVN_SCOPE(timer, "analytic");

    const Status status = Integrate(mean, covariance, lower, upper, order,
                                    result, MVN_TIMER_ARG(timer));
    if (status != Status::kOk) {
        detail::ClearResult(result);
    }

    return status;
}
}  // namespace mvn
