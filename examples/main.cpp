#include "mvn_moments/log.hpp"
#include "mvn_moments/moments.hpp"

int main()
{
    mvn::Log(mvn::Level::kInfo, "Computing rectangle moments");
    const Eigen::VectorXd mean = Eigen::VectorXd::Zero(1);
    const Eigen::MatrixXd covariance = Eigen::MatrixXd::Identity(1, 1);
    const Eigen::VectorXd lower = Eigen::VectorXd::Constant(1, -1);
    const Eigen::VectorXd upper = Eigen::VectorXd::Constant(1, 1);

    // Init allocates once; Compute can then run in a loop without allocating.
    mvn::MonteCarloSolver solver;
    mvn::Status status = solver.Init(1, mvn::SamplingConfig{100000, true}, 42);
    if (status == mvn::Status::kOk) {
        status = solver.Compute(mean, covariance, lower, upper);
    }

    if (status != mvn::Status::kOk) {
        mvn::Log(mvn::Level::kError, mvn::StatusMessage(status));
        return 1;
    }

    return 0;
}
