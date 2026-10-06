#include <cmath>
#include <sstream>

#include "check.hpp"
#include "mvn_moments/log.hpp"

namespace {
using check::kInf;
using check::Near;
using check::Relative;
using check::Require;
using check::Returns;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void TestOrthants()
{
    const VectorXd mean = VectorXd::Zero(2), lower = VectorXd::Zero(2),
                   upper = VectorXd::Constant(2, kInf);
    for (double rho : {-0.999999, -0.5, 0.0, 0.5, 0.999999}) {
        const MatrixXd covariance =
            (MatrixXd(2, 2) << 1, rho, rho, 1).finished();
        mvn::Result result(2);
        Returns(mvn::Status::kOk,
                mvn::Bivariate(mean, covariance, lower, upper, mvn::Order::kAll,
                               result),
                "Orthant");
        const double p = 0.25 + std::asin(rho) / (2 * EIGEN_PI);
        const double first = (1 + rho) / (2 * std::sqrt(2 * EIGEN_PI));
        const double boundary =
            std::sqrt((1 - rho) * (1 + rho)) / (2 * EIGEN_PI);
        Near(*result.zeroth, p, 2e-12, "Orthant probability");
        Near(result.First()(0), first, 2e-12, "Orthant first");
        Near(result.First()(1), first, 2e-12, "Orthant second coordinate");
        Near(result.Second()(0, 0), p + rho * boundary, 2e-12,
             "Orthant square");
        Near(result.Second()(0, 1), rho * p + boundary, 2e-12, "Orthant cross");
        Require(!result.errors && !result.zeroth_error && result.samples == 0,
                "Deterministic result metadata");
    }
}

void TestIndependent()
{
    const VectorXd mean = (VectorXd(2) << 0.3, -0.4).finished();
    const MatrixXd covariance = (MatrixXd(2, 2) << 2, 0, 0, 0.7).finished();
    for (int box = 0; box < 4; ++box) {
        VectorXd lower(2), upper(2);
        if (box == 0) {
            lower << -kInf, -kInf;
            upper << kInf, kInf;
        } else if (box == 1) {
            lower << 9, -10;
            upper << 10, -9;
        } else if (box == 2) {
            lower << 0.1, -0.2;
            upper << 0.1 + 1e-10, -0.2 + 1e-10;
        } else {
            lower << -1, -2;
            upper << 2, 1;
        }
        mvn::Result x(1), y(1), result(2);
        Returns(
            mvn::Status::kOk,
            mvn::Analytic(mean.head(1), covariance.topLeftCorner(1, 1),
                          lower.head(1), upper.head(1), mvn::Order::kAll, x),
            "Marginal X");
        Returns(
            mvn::Status::kOk,
            mvn::Analytic(mean.tail(1), covariance.bottomRightCorner(1, 1),
                          lower.tail(1), upper.tail(1), mvn::Order::kAll, y),
            "Marginal Y");
        Returns(mvn::Status::kOk,
                mvn::Bivariate(mean, covariance, lower, upper, mvn::Order::kAll,
                               result),
                "Independent");
        Relative(*result.zeroth, *x.zeroth * *y.zeroth, 2e-10, "Independent P");
        Relative(result.First()(0), x.First()(0) * *y.zeroth, 2e-10,
                 "Independent first X");
        Relative(result.First()(1), y.First()(0) * *x.zeroth, 2e-10,
                 "Independent first Y");
        Relative(result.Second()(0, 0), x.Second()(0, 0) * *y.zeroth, 2e-10,
                 "Independent second X");
        Relative(result.Second()(1, 1), y.Second()(0, 0) * *x.zeroth, 2e-10,
                 "Independent second Y");
        Relative(result.Second()(0, 1), x.First()(0) * y.First()(0), 2e-10,
                 "Independent cross");
    }
}

void TestTransformationAndNarrow()
{
    const VectorXd mean = (VectorXd(2) << 0.3, -0.4).finished();
    const MatrixXd covariance = (MatrixXd(2, 2) << 2, 0.6, 0.6, 0.7).finished();
    const VectorXd lower = VectorXd::Constant(2, -kInf);
    const VectorXd upper = VectorXd::Constant(2, kInf);
    mvn::Result result(2);
    Returns(mvn::Status::kOk,
            mvn::Bivariate(mean, covariance, lower, upper, mvn::Order::kAll,
                           result),
            "Unbounded");
    Near(*result.zeroth, 1, 1e-12, "Unbounded P");
    Require(result.First().isApprox(mean, 1e-11), "Unbounded mean");
    Require(
        result.Second().isApprox(covariance + mean * mean.transpose(), 1e-11),
        "Unbounded second");

    const VectorXd a = (VectorXd(2) << 0.1, -0.2).finished();
    const VectorXd b = a.array() + 1e-10;
    Returns(mvn::Status::kOk,
            mvn::Bivariate(mean, covariance, a, b, mvn::Order::kAll, result),
            "Narrow correlated");
    const VectorXd displacement = (a + b) / 2 - mean;
    const double determinant = covariance(0, 0) * covariance(1, 1) -
                               covariance(0, 1) * covariance(1, 0);
    const double quadratic =
        (covariance(1, 1) * displacement(0) * displacement(0) -
         2 * covariance(0, 1) * displacement(0) * displacement(1) +
         covariance(0, 0) * displacement(1) * displacement(1)) /
        determinant;
    const double probability = (b(0) - a(0)) * (b(1) - a(1)) *
                               std::exp(-quadratic / 2) /
                               (2 * EIGEN_PI * std::sqrt(determinant));
    Relative(*result.zeroth, probability, 1e-10, "Narrow correlated density");
    Near(result.First()(0) / *result.zeroth, (a(0) + b(0)) / 2, 1e-12,
         "Narrow correlated conditional mean");
}

void TestSelectionAndErrors()
{
    VectorXd mean = VectorXd::Zero(2), lower = VectorXd::Constant(2, -1),
             upper = VectorXd::Ones(2);
    MatrixXd covariance = MatrixXd::Identity(2, 2);
    mvn::Result result(2);
    for (unsigned bits = 0; bits < 8; ++bits) {
        const auto order = static_cast<mvn::Order>(bits);
        Returns(mvn::Status::kOk,
                mvn::Bivariate(mean, covariance, lower, upper, order, result),
                "Orders");
        Require(result.First().size() ==
                    (mvn::HasOrder(order, mvn::Order::kFirst) ? 2 : 0),
                "First selection");
        Require(result.Second().rows() ==
                    (mvn::HasOrder(order, mvn::Order::kSecond) ? 2 : 0),
                "Second selection");
    }
    covariance(0, 1) = covariance(1, 0) = 2;
    Returns(mvn::Status::kNotPositiveDefinite,
            mvn::Bivariate(mean, covariance, lower, upper, mvn::Order::kAll,
                           result),
            "Indefinite bivariate");
    Require(result.dimension == 0, "Failure clears result");
    covariance = MatrixXd::Identity(2, 2);
    lower(0) = upper(0);
    Returns(mvn::Status::kOk,
            mvn::Bivariate(mean, covariance, lower, upper, mvn::Order::kAll,
                           result),
            "Zero width");
    Require(*result.zeroth == 0, "Zero width probability");
    lower = VectorXd::Constant(2, -1);
    mvn::BivariateConfig limited;
    limited.max_intervals = 1;
    Returns(mvn::Status::kNumericalError,
            mvn::Bivariate(mean, covariance, lower, upper, mvn::Order::kAll,
                           result, limited),
            "Bound exhaustion");
    Require(result.dimension == 0, "Bivariate failure clears result");
    limited.relative_tolerance = 0;
    Returns(mvn::Status::kInvalidInput,
            mvn::Bivariate(mean, covariance, lower, upper, mvn::Order::kAll,
                           result, limited),
            "Invalid tolerance");
    Returns(mvn::Status::kOk,
            mvn::Bivariate(mean, covariance, lower, upper, mvn::Order::kAll,
                           result),
            "Recovery");
}

void TestWarning()
{
    mvn::GenzSolver solver;
    Returns(mvn::Status::kOk, solver.Init(2, mvn::GenzConfig{1000}),
            "Warning Init");
    const VectorXd mean = VectorXd::Zero(2), lower = VectorXd::Constant(2, -1),
                   upper = VectorXd::Ones(2);
    const MatrixXd covariance = MatrixXd::Identity(2, 2);
    std::ostringstream output;
    auto* previous = std::cerr.rdbuf(output.rdbuf());
    const auto first = solver.Compute(mean, covariance, lower, upper);
    const auto size = output.str().size();
    const auto second = solver.Compute(mean, covariance, lower, upper);
    std::cerr.rdbuf(previous);
    Returns(mvn::Status::kOk, first, "First warning call");
    Returns(mvn::Status::kOk, second, "Second warning call");
    Require(size != 0 && output.str().size() == size, "Warn once");
    Require(output.str().find("MomentsSolver") != std::string::npos,
            "Warning names selector");
}
}  // namespace

int main()
{
    TestOrthants();
    TestIndependent();
    TestTransformationAndNarrow();
    TestSelectionAndErrors();
    TestWarning();
}
