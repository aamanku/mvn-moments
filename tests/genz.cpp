#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "check.hpp"
#include "mvn_moments/detail/lattice.hpp"
#include "mvn_moments/detail/normal.hpp"
#include "mvn_moments/moments.hpp"

namespace {
using Eigen::MatrixXd;
using Eigen::VectorXd;

using check::kInf;
using check::Near;
using check::Relative;
using check::Require;
using check::Returns;

// Status of a fresh Genz solver's first call; result receives its output.
mvn::Status Run(const VectorXd& mean, const MatrixXd& covariance,
                const VectorXd& lower, const VectorXd& upper, mvn::Order order,
                std::size_t samples, std::uint64_t seed, mvn::Result& result)
{
    mvn::GenzSolver solver;
    const auto status = solver.Init(
        mean.size(), mvn::GenzConfig{samples, false, 4096, 10}, seed);
    if (status != mvn::Status::kOk) {
        return status;
    }

    const auto computed = solver.Compute(mean, covariance, lower, upper, order);
    result = solver.GetResult();
    return computed;
}

mvn::Result Genz(const VectorXd& mean, const MatrixXd& covariance,
                 const VectorXd& lower, const VectorXd& upper,
                 mvn::Order order = mvn::Order::kAll,
                 std::size_t samples = 100000, std::uint64_t seed = 42)
{
    mvn::Result result;
    Returns(mvn::Status::kOk,
            Run(mean, covariance, lower, upper, order, samples, seed, result),
            "Genz");
    return result;
}

mvn::Result Exact(const VectorXd& mean, const MatrixXd& variance,
                  const VectorXd& lower, const VectorXd& upper)
{
    mvn::Result result(1);
    Returns(
        mvn::Status::kOk,
        mvn::Analytic(mean, variance, lower, upper, mvn::Order::kAll, result),
        "Analytic reference");
    return result;
}

// Weighted Korobov worst-case error (alpha = 2) minimized by CBC, with
// SciPy's weights 1, 1, 0.8, 0.8^2, ...
double LatticeMerit(const std::vector<std::uint64_t>& generator,
                    std::uint64_t points)
{
    // Sum prod(1 + term) - 1 directly, with compensation. Subtracting one
    // after averaging loses precision in the small merit, especially on
    // arm64 platforms where long double has the same precision as double.
    long double total = 0, correction = 0;
    for (std::uint64_t k = 0; k < points; ++k) {
        long double excess = 0;
        for (std::size_t j = 0; j < generator.size(); ++j) {
            const long double weight = j < 2 ? 1.0L : std::pow(0.8L, j - 1);
            const long double x =
                static_cast<long double>(k * generator[j] % points) / points;
            const long double term = weight * (x * x - x + 1.0L / 6);
            excess = std::fma(term, 1 + excess, excess);
        }
        const long double adjusted = excess - correction;
        const long double next = total + adjusted;
        correction = (next - total) - adjusted;
        total = next;
    }

    return static_cast<double>(total / points);
}

void TestNormal()
{
    // A one-ulp change in x moves the tail mass by about x^2 * eps relative,
    // so round-trip tolerances scale with that conditioning.
    const double eps = std::numeric_limits<double>::epsilon();
    for (int exponent = -300; exponent <= 0; exponent += 3) {
        const double p = 0.5 * std::pow(10.0, exponent);
        const double x = mvn::detail::NormalQuantile(p);
        Relative(mvn::detail::NormalCdf(x), p, 16 * eps * (1 + x * x),
                 "Lower-tail quantile");
    }
    // 1 - 2^-k is exact, so the upper branch sees the intended tail mass.
    for (int k = 1; k <= 52; ++k) {
        const double tail = std::ldexp(1.0, -k);
        const double x = mvn::detail::NormalQuantile(1 - tail);
        Relative(mvn::detail::NormalCdf(-x), tail, 16 * eps * (1 + x * x),
                 "Upper-tail quantile");
    }

    Near(mvn::detail::NormalQuantile(0.975), 1.959963984540054, 1e-15,
         "Quantile at 0.975");
    Near(mvn::detail::NormalQuantile(0.5), 0, 0, "Median quantile");
    Require(std::isnan(mvn::detail::NormalQuantile(0)), "Quantile at 0");
    Require(std::isnan(mvn::detail::NormalQuantile(1)), "Quantile at 1");
    Require(std::isnan(mvn::detail::NormalQuantile(
                std::numeric_limits<double>::quiet_NaN())),
            "Quantile at NaN");
}

void TestQuantileBatches()
{
    // Homogeneous central, ordinary-tail, and deep-tail packets, followed
    // by mixed regions, invalid probabilities, and a partial final packet.
    VectorXd probabilities(29);
    probabilities << 0.1, 0.3, 0.7, 0.9, 0.001, 0.01, 0.99, 0.999, 1e-100,
        1e-150, 1e-200, 1e-300, 0.5, 0.01, 0.9, 1e-100, 0, 1, -1, 2, kInf,
        -kInf, std::numeric_limits<double>::quiet_NaN(), 0.075, 0.925,
        std::nextafter(0.0, 1.0), std::nextafter(1.0, 0.0), 0.5, 0.25;
    VectorXd actual = probabilities;
    mvn::detail::NormalQuantiles(actual);

    for (Eigen::Index i = 0; i < probabilities.size(); ++i) {
        const double expected = mvn::detail::NormalQuantile(probabilities(i));
        if (std::isnan(expected)) {
            Require(std::isnan(actual(i)), "Batched invalid quantile");
        } else {
            Near(actual(i), expected, 2e-15 * std::max(1.0, std::abs(expected)),
                 "Batched quantile agrees with scalar AS241");
        }
    }
}

void TestLattice()
{
    // Generators from SciPy 1.18 _cbc_lattice. Exact CBC ties (such as a
    // generator and its modular inverse) may resolve differently, so compare
    // the minimized criterion; the 10007 case has no ties and matches.
    struct Case {
        std::size_t dimensions;
        std::uint64_t target;
        std::uint64_t points;
        std::vector<std::uint64_t> scipy;
    };
    const std::vector<Case> cases = {
        {3, 1000, 997, {1, 379, 433}},
        {6, 10007, 10007, {1, 3822, 2827, 2235, 4682, 2995}},
        {10,
         50000,
         49999,
         {1, 18358, 11921, 15283, 3606, 1580, 9307, 19786, 18792, 5780}},
        {4, 7, 7, {1, 3, 2, 2}},
        {1, 100, 97, {1}},
    };
    mvn::detail::Lattice lattice;
    for (const auto& c : cases) {
        const auto label = "Lattice " + std::to_string(c.target);
        Returns(mvn::Status::kOk,
                mvn::detail::CbcLattice(c.dimensions, c.target, lattice),
                label);
        Require(lattice.points == c.points, label + " prime size");
        Require(lattice.generator.size() == c.dimensions, label + " size");
        Relative(LatticeMerit(lattice.generator, lattice.points),
                 LatticeMerit(c.scipy, c.points), 1e-12, label + " merit");
    }

    Returns(mvn::Status::kOk, mvn::detail::CbcLattice(6, 10007, lattice),
            "Untied lattice");
    Require(lattice.generator == cases[1].scipy, "Untied lattice differs");

    // Generators do not depend on later dimensions, so GenzSolver serves
    // smaller problems with a prefix of one lattice.
    mvn::detail::Lattice prefix;
    Returns(mvn::Status::kOk, mvn::detail::CbcLattice(3, 10007, prefix),
            "Prefix lattice");
    Require(std::equal(prefix.generator.begin(), prefix.generator.end(),
                       lattice.generator.begin()),
            "Lattice prefix differs");

    Returns(mvn::Status::kOk, mvn::detail::CbcLattice(0, 10, lattice),
            "Zero-dimensional lattice");
    Require(lattice.generator.empty(),
            "Zero-dimensional lattice must be empty");
    Returns(mvn::Status::kInvalidInput, mvn::detail::CbcLattice(2, 2, lattice),
            "Lattice below 3");
    Returns(mvn::Status::kInvalidInput,
            mvn::detail::CbcLattice(2, std::uint64_t{1} << 32, lattice),
            "Lattice at 2^32");
}

void TestExactProbabilities()
{
    // One dimension: the weight is the exact interval probability.
    const VectorXd mean = VectorXd::Constant(1, 0.3);
    const MatrixXd variance = MatrixXd::Constant(1, 1, 2.0);
    const VectorXd lower = VectorXd::Constant(1, -1);
    const VectorXd upper = VectorXd::Constant(1, 0.5);
    const auto exact = Exact(mean, variance, lower, upper);
    for (std::size_t samples : {30, 1000, 100000}) {
        const auto result =
            Genz(mean, variance, lower, upper, mvn::Order::kAll, samples);
        Near(*result.zeroth, *exact.zeroth, 1e-14, "1-D probability");
        Near(*result.zeroth_error, 0, 1e-15, "1-D probability error");
    }
    const auto moments = Genz(mean, variance, lower, upper);
    Near(moments.First()(0), exact.First()(0), 1e-6, "1-D first moment");
    Near(moments.Second()(0, 0), exact.Second()(0, 0), 1e-6,
         "1-D second moment");

    // Independent variables: the probability is a product for any budget,
    // with both the n - 1 (zeroth only) and n lattice dimensions.
    VectorXd mu(4), lo(4), hi(4);
    mu << 0.1, -0.2, 0.3, 0;
    lo << -1, -kInf, 0, -0.5;
    hi << 1, 0.4, kInf, 2;
    const VectorXd variances = (VectorXd(4) << 1, 2, 0.5, 3).finished();
    double product = 1;
    VectorXd first(4);
    for (Eigen::Index i = 0; i < 4; ++i) {
        const auto marginal =
            Exact(mu.segment(i, 1), MatrixXd::Constant(1, 1, variances(i)),
                  lo.segment(i, 1), hi.segment(i, 1));
        product *= *marginal.zeroth;
        first(i) = marginal.First()(0) / *marginal.zeroth;
    }
    for (auto order : {mvn::Order::kZeroth, mvn::Order::kAll}) {
        const auto result =
            Genz(mu, variances.asDiagonal().toDenseMatrix(), lo, hi, order);
        Relative(*result.zeroth, product, 1e-13, "Independent probability");
    }
    const auto independent =
        Genz(mu, variances.asDiagonal().toDenseMatrix(), lo, hi);
    for (Eigen::Index i = 0; i < 4; ++i) {
        Near(independent.First()(i), product * first(i), 1e-5,
             "Independent first moment");
    }
}

void TestAccuracy()
{
    // Correlated positive orthant: probability 1/3 and E[X_i; X > 0] =
    // (1 + rho) / (2 sqrt(2 pi)).
    MatrixXd correlated(2, 2);
    correlated << 1, 0.5, 0.5, 1;
    const auto orthant = Genz(VectorXd::Zero(2), correlated, VectorXd::Zero(2),
                              VectorXd::Constant(2, kInf));
    Near(*orthant.zeroth, 1.0 / 3, 1e-6, "Orthant probability");
    Require(*orthant.zeroth_error > 0 && *orthant.zeroth_error < 1e-5,
            "Orthant standard error out of range");
    const double first = 1.5 / (2 * std::sqrt(2 * EIGEN_PI));
    Near(orthant.First()(0), first, 1e-4, "Orthant first moment");
    Near(orthant.First()(1), first, 1e-4, "Orthant first moment");
    Require(orthant.Second().isApprox(orthant.Second().transpose(), 0),
            "Second moment must be symmetric");

    // A rare rectangle where rejection sampling accepts nothing.
    const VectorXd zero = VectorXd::Zero(1);
    const MatrixXd one = MatrixXd::Identity(1, 1);
    const VectorXd a = VectorXd::Constant(1, 30), b = VectorXd::Constant(1, 31);
    const auto exact = Exact(zero, one, a, b);
    const auto rare = Genz(zero, one, a, b);
    Relative(*rare.zeroth, *exact.zeroth, 1e-12, "Rare probability");
    Relative(rare.First()(0), exact.First()(0), 1e-6, "Rare first moment");
    Relative(rare.Second()(0, 0), exact.Second()(0, 0), 1e-6,
             "Rare second moment");
    Require(*rare.zeroth_error >= 0 && rare.FirstError()(0) > 0,
            "Rare standard errors must not underflow");
}

// Both reflected and unreflected deep-tail intervals must retain relative
// accuracy, including the off-diagonal raw second moment.
void TestMixedTails()
{
    const VectorXd mean = VectorXd::Zero(2);
    const MatrixXd covariance = MatrixXd::Identity(2, 2);
    VectorXd lower(2), upper(2);
    lower << -10, 9;
    upper << -9, kInf;
    const auto left = Exact(mean.head(1), MatrixXd::Identity(1, 1),
                            lower.head(1), upper.head(1));
    const auto right = Exact(mean.tail(1), MatrixXd::Identity(1, 1),
                             lower.tail(1), upper.tail(1));
    const auto result = Genz(mean, covariance, lower, upper);

    Relative(*result.zeroth, *left.zeroth * *right.zeroth, 1e-12,
             "Mixed-tail probability");
    Relative(result.First()(0), left.First()(0) * *right.zeroth, 1e-5,
             "Mixed-tail lower first moment");
    Relative(result.First()(1), right.First()(0) * *left.zeroth, 1e-5,
             "Mixed-tail upper first moment");
    Relative(result.Second()(0, 0), left.Second()(0, 0) * *right.zeroth, 1e-5,
             "Mixed-tail lower second moment");
    Relative(result.Second()(1, 1), right.Second()(0, 0) * *left.zeroth, 1e-5,
             "Mixed-tail upper second moment");
    Relative(result.Second()(0, 1), left.First()(0) * right.First()(0), 1e-5,
             "Mixed-tail cross moment");
}

void TestInvariance()
{
    VectorXd mean(3), lower(3), upper(3);
    mean << 0.2, -0.1, 0.4;
    lower << -1, -0.5, -kInf;
    upper << 1.5, 2, 0.8;
    MatrixXd covariance(3, 3);
    covariance << 1, 0.3, -0.2, 0.3, 2, 0.4, -0.2, 0.4, 1.5;
    const auto reference = Genz(mean, covariance, lower, upper);

    // Permuting inputs permutes outputs: reordering depends only on values.
    Eigen::PermutationMatrix<3> permutation;
    permutation.indices() << 2, 0, 1;
    const auto permuted = Genz(
        permutation * mean, permutation * covariance * permutation.transpose(),
        permutation * lower, permutation * upper);
    Relative(*permuted.zeroth, *reference.zeroth, 1e-12,
             "Permuted probability");
    Require((permuted.First() - permutation * reference.First()).norm() < 1e-12,
            "Permuted first moment");
    Require((permuted.Second() -
             permutation * reference.Second() * permutation.transpose())
                    .norm() < 1e-12,
            "Permuted second moment");

    const auto repeat = Genz(mean, covariance, lower, upper);
    Require(*repeat.zeroth == *reference.zeroth &&
                repeat.Second() == reference.Second(),
            "Same seed must reproduce results");
    const auto other =
        Genz(mean, covariance, lower, upper, mvn::Order::kAll, 100000, 7);
    Require(*other.zeroth != *reference.zeroth,
            "Different seeds must shift the lattice");
    Near(*other.zeroth, *reference.zeroth,
         10 * (*reference.zeroth_error + *other.zeroth_error),
         "Seeds disagree beyond their errors");
    Require(reference.samples == 10 * 9973 &&
                reference.accepted == reference.samples,
            "Samples must report shifts times the prime lattice size");
}

void TestConfigAndErrors()
{
    mvn::GenzSolver solver;
    Returns(mvn::Status::kInvalidInput,
            solver.Init(2, mvn::GenzConfig{1000, false, 4096, 1}), "One shift");
    Returns(mvn::Status::kInvalidInput,
            solver.Init(2, mvn::GenzConfig{20, false, 4096, 10}),
            "Two points per shift");

    const VectorXd mean = VectorXd::Zero(2);
    const MatrixXd covariance = MatrixXd::Identity(2, 2);
    const VectorXd lower = VectorXd::Constant(2, -1);
    VectorXd flat_upper = VectorXd::Ones(2);
    flat_upper(1) = lower(1);
    const auto flat = Genz(mean, covariance, lower, flat_upper);
    Require(*flat.zeroth_error == 0 && flat.FirstError().isZero(0) &&
                flat.SecondError().isZero(0),
            "Zero-volume rectangles report zero standard errors");
}

void TestCovariance()
{
    // The permuted Cholesky is Genz's only factorization: non-positive pivots
    // are not positive definite, tiny positive pivots are numerically
    // singular.
    const VectorXd zero = VectorXd::Zero(2);
    const VectorXd upper = VectorXd::Ones(2);
    MatrixXd near_singular(2, 2);
    near_singular << 1, 1 - 1e-14, 1 - 1e-14, 1;

    mvn::Result result;
    Returns(mvn::Status::kNotPositiveDefinite,
            Run(zero, MatrixXd::Ones(2, 2), -upper, upper, mvn::Order::kAll,
                100000, 42, result),
            "Singular covariance");
    Returns(mvn::Status::kNotPositiveDefinite,
            Run(zero, -MatrixXd::Identity(2, 2), -upper, upper,
                mvn::Order::kAll, 100000, 42, result),
            "Negative diagonal");
    Returns(mvn::Status::kNumericallySingular,
            Run(zero, near_singular, -upper, upper, mvn::Order::kAll, 100000,
                42, result),
            "Numerically singular covariance");
}
}  // namespace

int main()
{
    TestNormal();
    TestQuantileBatches();
    TestLattice();
    TestExactProbabilities();
    TestAccuracy();
    TestMixedTails();
    TestInvariance();
    TestConfigAndErrors();
    TestCovariance();
}
