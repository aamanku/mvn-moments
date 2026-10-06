// Genz's separation-of-variables method on a randomly shifted lattice.
//
// 1. Standardize. With v_i = sqrt(Sigma_ii) and V = diag(v), the variable
//    y = V^-1 (x - mu) has correlation matrix C = V^-1 Sigma V^-1, and the
//    rectangle becomes a' <= y <= b' with a' = V^-1 (a - mu), b' likewise.
//
// 2. Factor. Write C = L L^T and y = L w with w ~ N(0, I). Splitting
//    L = D U into its diagonal D (the pivots) and a unit lower-triangular U,
//    the constraint on y_i = D_ii (w_i + sum_{j<i} U_ij w_j) is an interval
//    for w_i given the earlier w_j:
//
//        alpha_i <= w_i <= beta_i,
//        alpha_i = a'_i / D_ii - sum_{j<i} U_ij w_j,  beta_i likewise.
//
// 3. Separate variables (Genz 1992). Substituting
//
//        w_i = Phi^-1(Phi(alpha_i) + u_i e_i),
//        e_i = Phi(beta_i) - Phi(alpha_i),
//
//    turns each phi(w_i) dw_i over [alpha_i, beta_i] into e_i du_i over
//    [0, 1], so the probability becomes an integral over the unit cube with
//    no indicator at all:
//
//        Z = int_[0,1]^n f(u) du,  f(u) = prod_{i=1..n} e_i(u_1, ..., u_{i-1}).
//
//    e_1 is constant and nothing depends on u_n, so Z needs only n - 1
//    coordinates. This library extends the method to moments: with
//    x(u) = mu + V P^T L w(u), where P undoes step 4's reordering,
//
//        m1 = int f(u) x(u) du,  M2 = int f(u) x(u) x(u)^T du,
//
//    which use all n coordinates. Every point lies inside the rectangle and
//    carries weight f(u), so rare rectangles work where rejection fails.
//
// 4. Reorder (Gibson, Glasser, and Ochi; see Genz and Bretz 2009). While
//    factoring, step i places the remaining variable with the smallest
//    interval probability given the expected values of the variables already
//    placed. Putting the tightest constraints first makes f flatter, which
//    reduces the integration error.
//
// 5. Integrate on a lattice. With a prime N and a generator g from
//    lattice.cpp, point k = 1..N is
//
//        u_k = |2 frac(k g / N + Delta) - 1|  (componentwise),
//
//    a rank-1 lattice with random shift Delta ~ U[0, 1)^n and the tent map
//    t -> |2t - 1|, which keeps points uniform and periodizes the integrand
//    (the baker's transform; Hickernell 2002). Upper-tail intervals are drawn
//    by reflection, w_i = -Phi^-1(Phi(-beta_i) + u_i e_i), which keeps CDF
//    values away from 1.
//
// 6. Estimate errors. K independent shifts give estimates Q_1..Q_K of each
//    moment; the result is their mean Q, with standard error
//
//        sqrt(sum_j (Q_j - Q)^2 / (K (K - 1))).
//
// Sources:
// - A. Genz, "Numerical computation of multivariate normal probabilities",
//   J. Comput. Graph. Stat. 1 (1992) 141-149,
//   https://doi.org/10.1080/10618600.1992.10477010
// - A. Genz and F. Bretz, "Computation of Multivariate Normal and t
//   Probabilities", Springer 2009, https://doi.org/10.1007/978-3-642-01689-9
// - F. J. Hickernell, "Obtaining O(N^(-2+e)) convergence for lattice
//   quadrature rules", MCQMC 2000 (2002) 274-289,
//   https://doi.org/10.1007/978-3-642-56046-0_18
// - SciPy's port of Genz's MATLAB code, from which PermutedCholesky and the
//   integrand are ported: _permuted_cholesky and _qmvn in
//   https://github.com/scipy/scipy/blob/main/scipy/stats/_qmvnt.py

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "mvn_moments/detail/common.hpp"
#include "mvn_moments/detail/lattice.hpp"
#include "mvn_moments/detail/normal.hpp"
#include "mvn_moments/helper.hpp"

namespace mvn::detail {
// Genz state, allocated by GenzSolver::Init for up to max_dimension
// variables; each call uses the leading entries.
struct GenzState {
    GenzConfig config;
    // Points per batch: min(N, batch_size).
    Eigen::Index capacity = 0;
    std::mt19937_64 rng;
    // Built for max_dimension coordinates; prefixes serve smaller problems.
    Lattice lattice;

    // Genz's standardized, reordered problem. With standard normals y, the
    // permuted standardized variable i is pivot(i) * (unit.row(i) * y), and
    // its constraint is lower(i) <= unit.row(i) * y <= upper(i).
    Eigen::MatrixXd unit;   // Unit lower-triangular factor.
    Eigen::VectorXd pivot;  // Cholesky diagonal.
    Eigen::VectorXd lower;  // Bounds divided by pivots.
    Eigen::VectorXd upper;
    Eigen::VectorXd scale;            // sqrt(diag(covariance)), unpermuted.
    Eigen::VectorXd expected;         // Truncated means while reordering.
    std::vector<Eigen::Index> order;  // order[i] = original index of i.

    // Batches with one point per row, so each variable is a contiguous
    // column: conditional draws, standardized samples, samples in input
    // coordinates, weights, and conditional offsets.
    Eigen::MatrixXd draws;
    Eigen::MatrixXd standardized;
    Eigen::MatrixXd samples;
    Eigen::VectorXd weight;
    Eigen::VectorXd offset;
    Eigen::VectorXd quantile_sign;

    // Per-shift lattice shift and sums, the flattened estimates (one column
    // per shift), and their means and standard errors.
    Eigen::VectorXd shift;
    Eigen::VectorXd first_sum;
    Eigen::MatrixXd second_sum;
    Eigen::MatrixXd estimates;
    Eigen::VectorXd mean;
    Eigen::VectorXd error;
};
}  // namespace mvn::detail

namespace mvn {
namespace {
using detail::GenzState;

// Pivots at or below (k + 1) * kSingularTolerance are treated as singular,
// matching SciPy's _permuted_cholesky on the correlation scale.
constexpr double kSingularTolerance = 1e-10;

// Port of SciPy's _permuted_cholesky (Genz's MVNDST reordering) into state:
// each step picks the remaining variable with the smallest conditional
// probability, given expected values of earlier variables. This is Genz's
// only Cholesky factorization. Non-positive pivots are not positive
// definite; tiny positive pivots are numerically singular.
Status PermutedCholesky(const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                        const Eigen::Ref<const Eigen::VectorXd>& mean,
                        const Eigen::Ref<const Eigen::VectorXd>& lower,
                        const Eigen::Ref<const Eigen::VectorXd>& upper,
                        GenzState& s)
{
    if ((covariance.diagonal().array() <= 0).any()) {
        return Status::kNotPositiveDefinite;
    }

    const auto n = covariance.rows();
    auto cho = s.unit.topLeftCorner(n, n);
    auto scale = s.scale.head(n);
    auto lo = s.lower.head(n);
    auto hi = s.upper.head(n);
    auto y = s.expected.head(n);
    scale = covariance.diagonal().cwiseSqrt();
    for (Eigen::Index j = 0; j < n; ++j) {
        for (Eigen::Index i = 0; i < n; ++i) {
            cho(i, j) = (1.0 / scale(i)) * covariance(i, j) * (1.0 / scale(j));
        }
    }
    lo = (lower - mean).cwiseQuotient(scale);
    hi = (upper - mean).cwiseQuotient(scale);
    s.pivot.head(n).setZero();
    y.setZero();
    for (Eigen::Index i = 0; i < n; ++i) {
        s.order[i] = i;
    }

    // Lower triangle holds finished columns left of k and the Schur
    // complement from k onward; y holds expected values of placed variables.
    for (Eigen::Index k = 0; k < n; ++k) {
        Eigen::Index chosen = k;
        double pivot = 0, smallest = 1, chosen_lower = 0, chosen_upper = 0;
        double largest_diagonal = -std::numeric_limits<double>::infinity();
        for (Eigen::Index i = k; i < n; ++i) {
            largest_diagonal = std::max(largest_diagonal, cho(i, i));
            if (cho(i, i) <= kSingularTolerance) {
                continue;
            }

            const double root = std::sqrt(cho(i, i));
            const double shift = cho.row(i).head(k).dot(y.head(k));
            const double a = (lo(i) - shift) / root;
            const double b = (hi(i) - shift) / root;
            const double probability = detail::IntervalProbability(a, b);
            if (probability <= smallest) {
                chosen = i;
                pivot = root;
                smallest = probability;
                chosen_lower = a;
                chosen_upper = b;
            }
        }
        if (pivot <= (k + 1) * kSingularTolerance) {
            if (largest_diagonal <= 0) {
                return Status::kNotPositiveDefinite;
            }
            return Status::kNumericallySingular;
        }

        if (chosen > k) {
            const auto m = chosen;
            cho(m, m) = cho(k, k);
            cho.row(m).head(k).swap(cho.row(k).head(k));
            for (Eigen::Index j = m + 1; j < n; ++j) {
                std::swap(cho(j, m), cho(j, k));
            }
            for (Eigen::Index j = k + 1; j < m; ++j) {
                std::swap(cho(j, k), cho(m, j));
            }
            std::swap(lo(k), lo(m));
            std::swap(hi(k), hi(m));
            std::swap(s.order[k], s.order[m]);
        }

        cho(k, k) = pivot;
        cho.row(k).tail(n - k - 1).setZero();
        for (Eigen::Index i = k + 1; i < n; ++i) {
            cho(i, k) /= pivot;
            for (Eigen::Index j = k + 1; j <= i; ++j) {
                cho(i, j) -= cho(i, k) * cho(j, k);
            }
        }

        // Truncated-normal mean of the chosen variable, with SciPy's
        // fallbacks when its probability is negligible.
        if (smallest > kSingularTolerance) {
            y(k) = (detail::NormalPdf(chosen_lower) -
                    detail::NormalPdf(chosen_upper)) /
                   smallest;
        } else if (chosen_lower < -10) {
            y(k) = chosen_upper;
        } else if (chosen_upper > 10) {
            y(k) = chosen_lower;
        } else {
            y(k) = (chosen_lower + chosen_upper) / 2;
        }

        s.pivot(k) = pivot;
        cho.row(k).head(k + 1) /= pivot;
        lo(k) /= pivot;
        hi(k) /= pivot;
    }

    return Status::kOk;
}

// Adds one randomly shifted lattice's weighted raw-moment sums to first_sum
// and the lower triangle of second_sum (empty when not selected), and its
// zeroth sum to zeroth. The first `drawn` lattice coordinates drive the
// leading variables.
Status SampleShift(const Eigen::Ref<const Eigen::VectorXd>& mean,
                   Eigen::Index drawn, GenzState& s, double& zeroth,
                   Eigen::Ref<Eigen::VectorXd> first_sum,
                   Eigen::Ref<Eigen::MatrixXd> second_sum, Timer* timer)
{
    const auto n = mean.size();
    const bool moments = first_sum.size() != 0 || second_sum.size() != 0;
    const auto points = s.lattice.points;
    const auto batch_size = static_cast<std::uint64_t>(s.capacity);

    // The first variable has no conditional offset: its interval is shared
    // by every point in this shift, including across batches.
    const bool first_reflect = s.lower(0) > 0;
    const double first_endpoint =
        detail::NormalCdf(first_reflect ? -s.upper(0) : s.lower(0));
    const double first_end =
        detail::NormalCdf(first_reflect ? -s.lower(0) : s.upper(0));

    zeroth = 0;
    first_sum.setZero();
    second_sum.setZero();

    // Ceiling division bounds the number of batches.
    const auto batches = points / batch_size + (points % batch_size != 0);
    std::uint64_t completed = 0;
    for (std::uint64_t batch = 0; batch < batches; ++batch) {
        if (completed >= points) {
            break;
        }

        const auto count = static_cast<Eigen::Index>(
            std::min<std::uint64_t>(points - completed, batch_size));
        auto y = s.draws.topLeftCorner(count, n);
        auto z = s.standardized.topLeftCorner(count, n);
        auto weight = s.weight.head(count);
        auto offset = s.offset.head(count);
        weight.setOnes();

        // Sequential conditioning: variable i is drawn inside its interval
        // given the earlier draws, and the weight collects each interval's
        // probability. Lattice coordinate i drives variable i.
        {
            MVN_SCOPE(timer, "conditional_sampling");
            for (Eigen::Index i = 0; i < n; ++i) {
                offset.setZero();
                for (Eigen::Index k = 0; k < i; ++k) {
                    offset += s.unit(i, k) * y.col(k);
                }
                const bool draw = i < drawn;
                const auto generator = draw ? s.lattice.generator[i] : 0;
                for (Eigen::Index row = 0; row < count; ++row) {
                    const double a = s.lower(i) - offset(row);
                    const double b = s.upper(i) - offset(row);
                    // Reflect upper-tail intervals so both CDF endpoints remain
                    // small. Reuse the lower endpoint for the conditional draw.
                    const bool reflect = a > 0;
                    const double endpoint =
                        i == 0 ? first_endpoint
                               : detail::NormalCdf(reflect ? -b : a);
                    const double end =
                        i == 0 ? first_end
                               : detail::NormalCdf(reflect ? -a : b);
                    const double probability = std::max(end - endpoint, 0.0);
                    weight(row) *= probability;
                    if (!draw) {
                        continue;
                    }

                    // Shifted lattice coordinate with the tent (baker's) map.
                    const auto k = (completed + row + 1) % points;
                    double u = static_cast<double>(k * generator % points) /
                                   static_cast<double>(points) +
                               s.shift(i);
                    if (u >= 1) {
                        u -= 1;
                    }
                    u = std::abs(2 * u - 1);
                    // Clamp to the open unit interval; collapsed intervals
                    // still yield finite draws with zero weight.
                    y(row, i) = std::clamp(endpoint + u * probability,
                                           std::nextafter(0.0, 1.0),
                                           std::nextafter(1.0, 0.0));
                    s.quantile_sign(row) = reflect ? -1.0 : 1.0;
                }
                if (draw) {
                    detail::NormalQuantiles(y.col(i));
                    y.col(i).array() *= s.quantile_sign.head(count).array();
                }
                if (moments) {
                    z.col(i) = s.pivot(i) * (offset + y.col(i));
                }
            }
        }

        MVN_SCOPE(timer, "moment_accumulation");
        zeroth += weight.sum();
        if (moments) {
            auto x = s.samples.topLeftCorner(count, n);
            for (Eigen::Index i = 0; i < n; ++i) {
                const auto original = s.order[i];
                x.col(original) =
                    (mean(original) + s.scale(original) * z.col(i).array())
                        .matrix();
            }
            if (!x.allFinite()) {
                return Status::kNumericalError;
            }

            detail::Accumulate(x, weight, first_sum, second_sum);
        }

        completed += static_cast<std::uint64_t>(count);
    }

    return Status::kOk;
}

// Writes one shift's selected moments as [zeroth, first, second], where
// second is the full matrix in column-major order.
void Flatten(bool has_zeroth, double zeroth,
             const Eigen::Ref<const Eigen::VectorXd>& first_sum,
             const Eigen::Ref<const Eigen::MatrixXd>& second_sum, double points,
             Eigen::Ref<Eigen::VectorXd> flat)
{
    Eigen::Index next = 0;
    if (has_zeroth) {
        flat(next++) = zeroth / points;
    }
    flat.segment(next, first_sum.size()) = first_sum / points;
    next += first_sum.size();

    Eigen::Map<Eigen::MatrixXd> second(flat.data() + next, second_sum.rows(),
                                       second_sum.cols());
    second = second_sum / points;
    detail::MirrorLower(second);
}

// Writes means and standard errors across shifts (one column per shift).
// Deviations are scaled by each entry's largest magnitude so that tiny
// probabilities, e.g. 1e-198, do not underflow when squared.
void Summarize(Eigen::Index rows, GenzState& s, Result& result)
{
    const auto shifts = static_cast<Eigen::Index>(s.config.shifts);
    const auto count = static_cast<double>(shifts);
    const auto estimates = s.estimates.topLeftCorner(rows, shifts);
    auto mean = s.mean.head(rows);
    auto error = s.error.head(rows);
    for (Eigen::Index r = 0; r < rows; ++r) {
        const double average = estimates.row(r).mean();
        const double largest = estimates.row(r).cwiseAbs().maxCoeff();
        const double scale = largest > 0 ? largest : 1.0;
        const double deviations =
            ((estimates.row(r).array() - average) / scale).square().sum();
        mean(r) = average;
        error(r) = scale * std::sqrt(deviations / (count * (count - 1)));
    }

    const auto n = result.First().size();
    const auto m = result.Second().rows();
    Eigen::Index next = 0;
    if (result.zeroth) {
        result.zeroth = mean(next);
        result.zeroth_error = error(next);
        ++next;
    }
    result.first_storage.head(n) = mean.segment(next, n);
    result.first_error_storage.head(n) = error.segment(next, n);
    next += n;
    result.second_storage.topLeftCorner(m, m) =
        mean.segment(next, m * m).reshaped(m, m);
    result.second_error_storage.topLeftCorner(m, m) =
        error.segment(next, m * m).reshaped(m, m);
}

Status Integrate(const Eigen::Ref<const Eigen::VectorXd>& mean,
                 const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                 const Eigen::Ref<const Eigen::VectorXd>& lower,
                 const Eigen::Ref<const Eigen::VectorXd>& upper, Order order,
                 GenzState& s, Result& result, Timer* timer)
{
    (void)timer;
    Status status = detail::ValidateInputs(mean, covariance, lower, upper,
                                           order, result.Capacity(), timer);
    if (status != Status::kOk) {
        return status;
    }
    {
        MVN_SCOPE(timer, "reorder");
        status = PermutedCholesky(covariance, mean, lower, upper, s);
    }
    if (status != Status::kOk) {
        return status;
    }

    const auto n = mean.size();
    detail::StartResult(order, n, true, result);
    if (detail::NothingToCompute(order, lower, upper)) {
        return Status::kOk;
    }

    // The last variable's draw changes no weight, so probability alone
    // needs n - 1 lattice coordinates; moments need all n.
    const auto first_size = result.First().size();
    const auto second_size = result.Second().rows();
    const bool moments = first_size != 0 || second_size != 0;
    const auto drawn = moments ? n : n - 1;
    const auto rows = static_cast<Eigen::Index>(result.zeroth ? 1 : 0) +
                      first_size + second_size * second_size;
    auto first_sum = s.first_sum.head(first_size);
    auto second_sum = s.second_sum.topLeftCorner(second_size, second_size);
    const auto points = s.lattice.points;

    {
        MVN_SCOPE(timer, "sample_and_accumulate");
        std::uniform_real_distribution<double> uniform;
        for (std::size_t shift = 0; shift < s.config.shifts; ++shift) {
            for (Eigen::Index i = 0; i < drawn; ++i) {
                s.shift(i) = uniform(s.rng);
            }

            double zeroth = 0;
            status = SampleShift(mean, drawn, s, zeroth, first_sum, second_sum,
                                 timer);
            if (status != Status::kOk) {
                return status;
            }
            Flatten(
                result.zeroth.has_value(), zeroth, first_sum, second_sum,
                static_cast<double>(points),
                s.estimates.col(static_cast<Eigen::Index>(shift)).head(rows));
        }
    }

    // Standard error of the mean across independent random shifts.
    Summarize(rows, s, result);
    result.samples = s.config.shifts * points;
    result.accepted = result.samples;

    return detail::CheckFinite(result);
}

// Validates the configuration, builds the lattice for max_dimension
// coordinates, and allocates every buffer.
Status Allocate(Eigen::Index max_dimension, const GenzConfig& config,
                std::uint64_t seed, GenzState& s)
{
    if (max_dimension <= 0 || config.shifts < 2) {
        return Status::kInvalidInput;
    }
    Status status = detail::ValidateBudget(config.samples, config.batch_size);
    if (status != Status::kOk) {
        return status;
    }
    const auto per_shift = config.samples / config.shifts;
    if (per_shift < 3 || per_shift >= detail::kMaxLatticePoints ||
        config.shifts > static_cast<std::size_t>(
                            std::numeric_limits<Eigen::Index>::max())) {
        return Status::kInvalidInput;
    }

    // Flattened estimates hold [zeroth, first, second] per shift.
    if (detail::ValidateSize(max_dimension, max_dimension) != Status::kOk ||
        max_dimension * max_dimension >
            std::numeric_limits<Eigen::Index>::max() - 1 - max_dimension) {
        return Status::kInvalidInput;
    }
    const auto rows = 1 + max_dimension + max_dimension * max_dimension;
    const auto shifts = static_cast<Eigen::Index>(config.shifts);
    if (detail::ValidateSize(rows, shifts) != Status::kOk) {
        return Status::kInvalidInput;
    }

    status = detail::CbcLattice(static_cast<std::size_t>(max_dimension),
                                per_shift, s.lattice);
    if (status != Status::kOk) {
        return status;
    }
    const auto capacity = static_cast<Eigen::Index>(
        std::min<std::uint64_t>(s.lattice.points, config.batch_size));
    if (detail::ValidateSize(max_dimension, capacity) != Status::kOk) {
        return Status::kInvalidInput;
    }

    s.config = config;
    s.capacity = capacity;
    s.rng.seed(seed);

    s.unit.resize(max_dimension, max_dimension);
    s.pivot.resize(max_dimension);
    s.lower.resize(max_dimension);
    s.upper.resize(max_dimension);
    s.scale.resize(max_dimension);
    s.expected.resize(max_dimension);
    s.order.resize(static_cast<std::size_t>(max_dimension));

    s.draws.resize(capacity, max_dimension);
    s.standardized.resize(capacity, max_dimension);
    s.samples.resize(capacity, max_dimension);
    s.weight.resize(capacity);
    s.offset.resize(capacity);
    s.quantile_sign.resize(capacity);

    s.shift.resize(max_dimension);
    s.first_sum.resize(max_dimension);
    s.second_sum.resize(max_dimension, max_dimension);
    s.estimates.resize(rows, shifts);
    s.mean.resize(rows);
    s.error.resize(rows);
    return Status::kOk;
}
}  // namespace

GenzSolver::GenzSolver() = default;
GenzSolver::~GenzSolver() = default;
GenzSolver::GenzSolver(GenzSolver&&) noexcept = default;
GenzSolver& GenzSolver::operator=(GenzSolver&&) noexcept = default;

Status GenzSolver::Init(Eigen::Index max_dimension, const GenzConfig& config,
                        std::uint64_t seed)
{
    state_.reset();
    result_ = Result();

    auto state = std::make_unique<GenzState>();
    const Status status = Allocate(max_dimension, config, seed, *state);
    if (status != Status::kOk) {
        return status;
    }

    state_ = std::move(state);
    result_ = Result(max_dimension);
    return Status::kOk;
}

Status GenzSolver::Compute(const Eigen::Ref<const Eigen::VectorXd>& mean,
                           const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                           const Eigen::Ref<const Eigen::VectorXd>& lower,
                           const Eigen::Ref<const Eigen::VectorXd>& upper,
                           Order order)
{
    if (!state_) {
        return Status::kInvalidInput;
    }

    MVN_BENCHMARK_TIMER(timer, state_->config.timing);
    MVN_SCOPE(timer, "genz");

    const Status status = Integrate(mean, covariance, lower, upper, order,
                                    *state_, result_, MVN_TIMER_ARG(timer));
    if (status != Status::kOk) {
        detail::ClearResult(result_);
    }

    return status;
}

const Result& GenzSolver::GetResult() const
{
    return result_;
}
}  // namespace mvn
