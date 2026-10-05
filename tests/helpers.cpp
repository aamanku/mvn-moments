#include <limits>
#include <random>

#include "check.hpp"
#include "mvn_moments/helper.hpp"

namespace {
using check::Require;
using check::Throws;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void TestMatrixChecks()
{
    MatrixXd covariance(2, 2);
    covariance << 4, 2, 2, 3;
    MatrixXd asymmetric(2, 2);
    asymmetric << 1, 2, 0, 1;
    MatrixXd indefinite(2, 2);
    indefinite << 1, 2, 2, 1;
    const MatrixXd singular = MatrixXd::Ones(2, 2);

    Require(mvn::IsSymmetric(covariance), "Symmetric matrix rejected");
    Require(mvn::IsPositiveDefinite(covariance), "SPD matrix rejected");
    Require(!mvn::IsSymmetric(asymmetric), "Asymmetry missed");
    Require(!mvn::IsPositiveDefinite(singular), "Singular matrix accepted");
    Require(!mvn::IsPositiveDefinite(indefinite), "Indefinite matrix accepted");

    MatrixXd almost_symmetric = singular;
    almost_symmetric(1, 0) += 1e-13;
    Require(mvn::IsSymmetric(almost_symmetric), "Tolerance ignored");

    Throws([] { mvn::IsSymmetric(MatrixXd::Ones(1, 2)); }, "Nonsquare");
    Throws([&] { mvn::IsSymmetric(covariance, -1); }, "Negative tolerance");
    Throws(
        [] {
            mvn::IsSymmetric(MatrixXd::Constant(
                1, 1, std::numeric_limits<double>::quiet_NaN()));
        },
        "NaN matrix");
}

void TestCholesky()
{
    MatrixXd covariance(2, 2);
    covariance << 4, 2, 2, 3;

    const auto lower = mvn::CholeskyLower(covariance);
    Require((lower * lower.transpose()).isApprox(covariance, 1e-12),
            "Cholesky reconstruction failed");
    Throws([] { mvn::CholeskyLower(MatrixXd::Ones(2, 2)); },
           "Singular Cholesky");
}

void TestGenerators()
{
    std::mt19937_64 rng(42), duplicate(42);
    const auto rectangle = mvn::GenerateHyperrectangle(3, rng);
    const auto same = mvn::GenerateHyperrectangle(3, duplicate);
    Require(rectangle.lower.isApprox(same.lower) &&
                rectangle.upper.isApprox(same.upper),
            "Seed reproducibility failed");
    Require((rectangle.lower.array() < rectangle.upper.array()).all(),
            "Invalid generated bounds");

    const auto distribution = mvn::GenerateMvn(3, rng);
    Require(mvn::IsPositiveDefinite(distribution.covariance),
            "Generated covariance is not SPD");

    Throws([&] { mvn::GenerateMvn(0, rng); }, "Zero-dimensional MVN");
    Throws([&] { mvn::GenerateMvn(-1, rng); }, "Negative-dimensional MVN");
    Throws([&] { mvn::GenerateHyperrectangle(0, rng); },
           "Zero-dimensional rectangle");
}

void TestSampleMvn()
{
    std::mt19937_64 rng(42);
    MatrixXd covariance(2, 2);
    covariance << 4, 2, 2, 3;
    VectorXd mean(2);
    mean << 1, -2;

    Throws([&] { mvn::SampleMvn({}, {}, 1, rng); }, "Empty inputs");
    Throws([&] { mvn::SampleMvn(VectorXd::Zero(3), covariance, 1, rng); },
           "Mismatched dimensions");
    Throws([&] { mvn::SampleMvn(mean, covariance, -1, rng); },
           "Negative count");
    Require(mvn::SampleMvn(mean, covariance, 0, rng).rows() == 0,
            "Zero samples must be empty");

    const auto samples = mvn::SampleMvn(mean, covariance, 50000, rng);
    const VectorXd observed_mean = samples.colwise().mean();
    const MatrixXd centered = samples.rowwise() - observed_mean.transpose();
    const MatrixXd observed_covariance =
        centered.transpose() * centered / samples.rows();
    Require((observed_mean - mean).cwiseAbs().maxCoeff() < 0.08,
            "Sample means incorrect");
    Require((observed_covariance - covariance).cwiseAbs().maxCoeff() < 0.15,
            "Sample covariance incorrect");
}
}  // namespace

int main()
{
    TestMatrixChecks();
    TestCholesky();
    TestGenerators();
    TestSampleMvn();
}
