// Deterministic bivariate normal moments by conditional integration.
// With T,V independent standard normals, Z1=T, Z2=rho*T+s*V,
// s=sqrt(1-rho^2). At each T=t, the second constraint becomes a<=V<=b.
// Its integrals are P=Phi(b)-Phi(a), D=phi(a)-phi(b),
// H=P+a*phi(a)-b*phi(b). Thus the six integrands, multiplied by phi(t), are
// P, t*P, rho*t*P+s*D, t^2*P, t*(rho*t*P+s*D),
// rho^2*t^2*P+2*rho*t*s*D+s^2*H.
// Integrate these with a bounded adaptive embedded Gauss(7)-Kronrod(15)
// rule, then transform standardized raw moments to the input coordinates.
// Infinite outer bounds are clipped at +/-40 standard deviations: omitted
// mass is below the smallest representable double, not a user tolerance.
// Conditional narrow intervals use Gauss-Legendre to avoid cancellation.
// The Gaussian identities follow integration by parts (see analytic.cpp).
// Quadrature nodes: QUADPACK DQK15, https://netlib.org/quadpack/dqk15.f
#include <algorithm>
#include <array>
#include <cmath>

#include "mvn_moments/detail/common.hpp"
#include "mvn_moments/detail/normal.hpp"
#include "mvn_moments/log.hpp"

namespace mvn {
namespace {
using Moments = std::array<long double, 6>;
constexpr std::size_t kMaxIntervals = 65536;
constexpr double kTailLimit = 40;

struct Integrand {
    long double lower, upper, rho, residual, width;
    long double origin = 0, jacobian = 1;
    bool moments;

    Moments Evaluate(long double t) const
    {
        t = origin + jacobian * t;
        const long double location = rho * t;
        const long double a = (lower - location) / residual;
        const long double b = (upper - location) / residual;
        long double p = 0, d = 0, h = 0;
        if (std::isfinite(a) && std::isfinite(b) &&
            b - a < 1e-3L / (1 + std::max(std::abs(a), std::abs(b)))) {
            constexpr long double nodes[] = {
                0.183434642495649805L, 0.525532409916328986L,
                0.796666477413626740L, 0.960289856497536232L};
            constexpr long double weights[] = {
                0.362683783378361983L, 0.313706645877887287L,
                0.222381034453374471L, 0.101228536290376259L};
            const long double center = (a + b) / 2,
                              half = width / (2 * residual);
            for (int i = 0; i < 4; ++i) {
                for (int sign : {-1, 1}) {
                    const long double v = center + sign * half * nodes[i];
                    const long double w =
                        half * weights[i] * detail::NormalPdf(v);
                    p += w;
                    d += w * v;
                    h += w * v * v;
                }
            }
        } else {
            p = detail::IntervalProbability(a, b);
            if (moments) {
                const long double pa = detail::NormalPdf(a),
                                  pb = detail::NormalPdf(b);
                d = pa - pb;
                h = p + (std::isfinite(a) ? a * pa : 0) -
                    (std::isfinite(b) ? b * pb : 0);
            }
        }

        const long double density = jacobian * detail::NormalPdf(t);
        const long double y = location * p + residual * d;
        return {
            density * p,
            density * t * p,
            density * y,
            density * t * t * p,
            density * t * y,
            density * (location * location * p + 2 * location * residual * d +
                       residual * residual * h)};
    }
};

struct Interval {
    long double lower, upper;
};

bool Rule(const Integrand& integrand, const Interval& interval,
          double tolerance, const Moments& reference, long double fraction,
          Moments& value)
{
    constexpr long double nodes[] = {
        0.991455371120812639L, 0.949107912342758525L,
        0.864864423359769073L, 0.741531185599394440L,
        0.586087235467691130L, 0.405845151377397167L,
        0.207784955007898468L, 0};
    constexpr long double kronrod[] = {
        0.022935322010529225L, 0.063092092629978553L, 0.104790010322250184L,
        0.140653259715525919L, 0.169004726639267903L, 0.190350578064785410L,
        0.204432940075298892L, 0.209482141084727828L};
    constexpr long double gauss[] = {
        0.129484966168869693L, 0.279705391489276668L, 0.381830050505118945L,
        0.417959183673469388L};
    const long double center = (interval.lower + interval.upper) / 2;
    const long double half = (interval.upper - interval.lower) / 2;
    Moments coarse{};
    value = {};
    for (int i = 0; i < 8; ++i) {
        const auto left = integrand.Evaluate(center - half * nodes[i]);
        const auto right =
            i == 7 ? Moments{} : integrand.Evaluate(center + half * nodes[i]);
        for (int j = 0; j < 6; ++j) {
            const long double sum = left[j] + right[j];
            value[j] += kronrod[i] * sum * half;
            if (i % 2 == 1) {
                coarse[j] += gauss[i / 2] * sum * half;
            }
        }
    }

    const int selected = integrand.moments ? 6 : 1;
    for (int j = 0; j < selected; ++j) {
        const long double scale = std::max(reference[0], reference[j]);
        if (!std::isfinite(value[j]) ||
            std::abs(value[j] - coarse[j]) > tolerance * scale * fraction) {
            return false;
        }
    }
    return true;
}

Status Integrate(const Eigen::Ref<const Eigen::VectorXd>& mean,
                 const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                 const Eigen::Ref<const Eigen::VectorXd>& lower,
                 const Eigen::Ref<const Eigen::VectorXd>& upper, Order order,
                 Result& result, const BivariateConfig& config, Timer* timer)
{
    const auto status = detail::ValidateInputs(mean, covariance, lower, upper,
                                               order, result.Capacity(), timer);
    if (status != Status::kOk) {
        return status;
    }
    if (mean.size() != 2 ||
        !(config.relative_tolerance > 0 && config.relative_tolerance < 1) ||
        config.max_intervals == 0 || config.max_intervals > kMaxIntervals) {
        return Status::kInvalidInput;
    }
    if ((covariance.diagonal().array() <= 0).any()) {
        return Status::kNotPositiveDefinite;
    }
    const long double sx = std::sqrt(covariance(0, 0));
    const long double sy = std::sqrt(covariance(1, 1));
    const long double rho = (covariance(0, 1) / sx) / sy;
    if (!(std::abs(rho) < 1)) {
        return Status::kNotPositiveDefinite;
    }

    detail::StartResult(order, 2, false, result);
    if (detail::NothingToCompute(order, lower, upper)) {
        return Status::kOk;
    }
    long double lo =
        std::max(-static_cast<long double>(kTailLimit),
                 (lower(0) - static_cast<long double>(mean(0))) / sx);
    long double hi =
        std::min(static_cast<long double>(kTailLimit),
                 (upper(0) - static_cast<long double>(mean(0))) / sx);
    if (!(hi > lo)) {
        return Status::kNumericalError;
    }
    Integrand integrand{
        (lower(1) - static_cast<long double>(mean(1))) / sy,
        (upper(1) - static_cast<long double>(mean(1))) / sy,
        rho,
        std::sqrt((1 - rho) * (1 + rho)),
        (static_cast<long double>(upper(1)) - lower(1)) / sy,
        0,
        1,
        HasOrder(order, Order::kFirst) || HasOrder(order, Order::kSecond)};

    // A narrow outer interval is integrated in [-1,1], retaining its width
    // from the original bounds rather than subtracting standardized endpoints.
    const long double outer_width =
        (static_cast<long double>(upper(0)) - lower(0)) / sx;
    if (std::isfinite(lower(0)) && std::isfinite(upper(0)) &&
        outer_width < 1e-3L / (1 + std::max(std::abs(lo), std::abs(hi)))) {
        integrand.origin =
            ((static_cast<long double>(lower(0)) + upper(0)) / 2 - mean(0)) /
            sx;
        integrand.jacobian = outer_width / 2;
        lo = -1;
        hi = 1;
    }

    // Split at marginal and conditional transitions so sharp near-singular
    // intervals and tail peaks cannot hide between quadrature nodes.
    std::array<long double, 11> cuts{lo, hi};
    std::size_t count = 2;
    const long double transition_width =
        rho == 0 ? 0 : 12 * integrand.residual / std::abs(rho);
    const long double transition_lower = rho == 0 ? lo : integrand.lower / rho;
    const long double transition_upper = rho == 0 ? hi : integrand.upper / rho;
    const std::array<long double, 9> candidates{
        0,
        rho == 0 ? lo : integrand.lower / rho,
        rho == 0 ? hi : integrand.upper / rho,
        rho * integrand.lower,
        rho * integrand.upper,
        transition_lower - transition_width,
        transition_lower + transition_width,
        transition_upper - transition_width,
        transition_upper + transition_width};
    for (long double cut : candidates) {
        cut = (cut - integrand.origin) / integrand.jacobian;
        if (std::isfinite(cut) && cut > lo && cut < hi) {
            cuts[count++] = cut;
        }
    }
    std::sort(cuts.begin(), cuts.begin() + count);

    MVN_SCOPE(timer, "conditional_quadrature");
    // Scale local absolute tolerances by an initial integral estimate, so
    // negligible far-tail intervals do not demand subnormal relative accuracy.
    Moments reference{};
    for (std::size_t segment = 1; segment < count; ++segment) {
        Moments estimate{};
        Rule(integrand, {cuts[segment - 1], cuts[segment]}, 1, reference, 1,
             estimate);
        for (int j = 0; j < 6; ++j) {
            reference[j] += std::abs(estimate[j]);
        }
    }

    Moments total{};
    std::size_t evaluated = 0;
    for (std::size_t segment = 1; segment < count; ++segment) {
        if (cuts[segment] == cuts[segment - 1]) {
            continue;
        }
        std::array<Interval, 64> stack{};
        std::size_t pending = 1;
        stack[0] = {cuts[segment - 1], cuts[segment]};
        for (std::size_t work = 0; work < config.max_intervals; ++work) {
            if (pending == 0 || evaluated >= config.max_intervals) {
                break;
            }
            const auto interval = stack[--pending];
            ++evaluated;
            Moments value{};
            if (Rule(integrand, interval, config.relative_tolerance, reference,
                     (interval.upper - interval.lower) / (hi - lo), value)) {
                for (int j = 0; j < 6; ++j) {
                    total[j] += value[j];
                }
            } else {
                const long double center =
                    (interval.lower + interval.upper) / 2;
                if (pending + 2 > stack.size() || center == interval.lower ||
                    center == interval.upper) {
                    Log(Level::kWarning,
                        "Bivariate quadrature exhausted its subdivision bound "
                        "before convergence.");
                    return Status::kNumericalError;
                }
                stack[pending++] = {center, interval.upper};
                stack[pending++] = {interval.lower, center};
            }
        }
        if (pending != 0) {
            Log(Level::kWarning,
                "Bivariate quadrature exhausted max_intervals before "
                "convergence.");
            return Status::kNumericalError;
        }
    }

    if (!(total[0] > 0) || static_cast<double>(total[0]) == 0) {
        return Status::kNumericalError;
    }
    const long double mx = mean(0), my = mean(1);
    if (result.zeroth) {
        *result.zeroth = static_cast<double>(total[0]);
    }
    if (HasOrder(order, Order::kFirst)) {
        result.first_storage(0) =
            static_cast<double>(mx * total[0] + sx * total[1]);
        result.first_storage(1) =
            static_cast<double>(my * total[0] + sy * total[2]);
    }
    if (HasOrder(order, Order::kSecond)) {
        result.second_storage(0, 0) = static_cast<double>(
            mx * mx * total[0] + 2 * mx * sx * total[1] + sx * sx * total[3]);
        result.second_storage(1, 1) = static_cast<double>(
            my * my * total[0] + 2 * my * sy * total[2] + sy * sy * total[5]);
        result.second_storage(0, 1) =
            static_cast<double>(mx * my * total[0] + my * sx * total[1] +
                                mx * sy * total[2] + sx * sy * total[4]);
        result.second_storage(1, 0) = result.second_storage(0, 1);
    }
    return detail::CheckFinite(result);
}
}  // namespace

Status Bivariate(const Eigen::Ref<const Eigen::VectorXd>& mean,
                 const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                 const Eigen::Ref<const Eigen::VectorXd>& lower,
                 const Eigen::Ref<const Eigen::VectorXd>& upper, Order order,
                 Result& result, const BivariateConfig& config)
{
    MVN_BENCHMARK_TIMER(timer, config.timing);
    MVN_SCOPE(timer, "bivariate");
    const auto status = Integrate(mean, covariance, lower, upper, order, result,
                                  config, MVN_TIMER_ARG(timer));
    if (status != Status::kOk) {
        detail::ClearResult(result);
    }
    return status;
}
}  // namespace mvn
