#include <cstdlib>
#include <new>
#include <string>
#include <vector>

#include "check.hpp"
#include "mvn_moments/moments.hpp"

// Counts global operator new calls while armed; Eigen's own allocations are
// caught separately by EIGEN_RUNTIME_NO_MALLOC, which aborts.
namespace {
bool counting = false;
std::size_t allocations = 0;
}  // namespace

void* operator new(std::size_t size)
{
    if (counting) {
        ++allocations;
    }
    if (void* memory = std::malloc(size == 0 ? 1 : size)) {
        return memory;
    }
    throw std::bad_alloc();
}

void operator delete(void* memory) noexcept
{
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept
{
    std::free(memory);
}

namespace {
using check::kInf;
using check::Problem;
using check::Require;
using check::Returns;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Forbids heap allocation for its lifetime.
class NoAllocation {
   public:
    NoAllocation()
    {
        allocations = 0;
        counting = true;
        Eigen::internal::set_is_malloc_allowed(false);
    }
    ~NoAllocation()
    {
        Eigen::internal::set_is_malloc_allowed(true);
        counting = false;
    }
    NoAllocation(const NoAllocation&) = delete;
    NoAllocation& operator=(const NoAllocation&) = delete;
};

struct Call {
    std::string label;
    Problem problem;
    mvn::Order order;
    mvn::Status expected;
};

// Every call after Init: changed covariances, smaller dimensions, toggled
// orders, failures, and oversized input.
std::vector<Call> Calls(bool rejection)
{
    Problem three{VectorXd(3), MatrixXd(3, 3), VectorXd(3), VectorXd(3)};
    three.mean << 0.2, -0.4, 0.5;
    three.covariance << 1, 0.2, 0.1, 0.2, 2, -0.3, 0.1, -0.3, 1;
    three.lower << -0.7, -0.7, -kInf;
    three.upper << 1, 1, 1;
    auto changed = three;
    changed.covariance(0, 0) = 1.5;

    const Problem four{VectorXd::Zero(4), MatrixXd::Identity(4, 4),
                       VectorXd::Constant(4, -1), VectorXd::Ones(4)};
    const Problem five{VectorXd::Zero(5), MatrixXd::Identity(5, 5),
                       VectorXd::Constant(5, -1), VectorXd::Ones(5)};
    const Problem one{VectorXd::Constant(1, 0.3), MatrixXd::Constant(1, 1, 2),
                      VectorXd::Constant(1, -1), VectorXd::Constant(1, 0.5)};
    const Problem rare{VectorXd::Zero(1), MatrixXd::Identity(1, 1),
                       VectorXd::Constant(1, 30), VectorXd::Constant(1, 31)};
    const Problem indefinite{VectorXd::Zero(2),
                             (MatrixXd(2, 2) << 1, 2, 2, 1).finished(),
                             VectorXd::Constant(2, -1), VectorXd::Ones(2)};

    const auto ok = mvn::Status::kOk;
    return {
        {"first call", three, mvn::Order::kAll, ok},
        {"cached covariance", three, mvn::Order::kAll, ok},
        {"changed covariance", changed, mvn::Order::kAll, ok},
        {"probability only", changed, mvn::Order::kZeroth, ok},
        {"full capacity", four, mvn::Order::kAll, ok},
        {"one dimension", one, mvn::Order::kFirstSecond, ok},
        {"indefinite", indefinite, mvn::Order::kAll,
         mvn::Status::kNotPositiveDefinite},
        {"over capacity", five, mvn::Order::kAll, mvn::Status::kInvalidInput},
        {"rare rectangle", rare, mvn::Order::kAll,
         rejection ? mvn::Status::kNoAcceptedSamples : ok},
        {"after failures", three, mvn::Order::kAll, ok},
    };
}

template <class Solver, class Config>
void CheckSolver(const std::string& name, const Config& config, bool rejection)
{
    const auto calls = Calls(rejection);
    const auto& first = calls.front().problem;
    Solver solver;
    mvn::Status status = mvn::Status::kOk;
    {
        NoAllocation guard;
        status = solver.Compute(first.mean, first.covariance, first.lower,
                                first.upper);
    }
    Returns(mvn::Status::kInvalidInput, status, name + ": before Init");
    Returns(mvn::Status::kOk, solver.Init(4, config, 42), name + ": Init");

    for (const auto& call : calls) {
        const auto& p = call.problem;
        {
            NoAllocation guard;
            status = solver.Compute(p.mean, p.covariance, p.lower, p.upper,
                                    call.order);
        }
        const auto label = name + ": " + call.label;
        Returns(call.expected, status, label);
        Require(allocations == 0, label + " allocated");
    }

    // Fixed-size Eigen inputs bind without temporaries.
    const Eigen::Vector2d mean(0.1, -0.2);
    const Eigen::Matrix2d covariance =
        (Eigen::Matrix2d() << 1, 0.3, 0.3, 2).finished();
    const Eigen::Vector2d lower(-1, -1), upper(1, 2);
    {
        NoAllocation guard;
        status = solver.Compute(mean, covariance, lower, upper);
    }
    Returns(mvn::Status::kOk, status, name + ": fixed-size inputs");
    Require(allocations == 0, name + ": fixed-size inputs allocated");
}

void TestAnalytic()
{
    mvn::Result result(1);
    const Eigen::Matrix<double, 1, 1> mean(0.3), variance(2), lower(-1),
        upper(0.5);
    mvn::Status status = mvn::Status::kOk;
    {
        NoAllocation guard;
        status = mvn::Analytic(mean, variance, lower, upper, mvn::Order::kAll,
                               result);
    }
    Returns(mvn::Status::kOk, status, "Analytic");
    Require(allocations == 0, "Analytic allocated");
}
}  // namespace

int main()
{
    CheckSolver<mvn::MonteCarloSolver>("Monte Carlo", mvn::SamplingConfig{},
                                       true);
    CheckSolver<mvn::MonteCarloSolver>("Monte Carlo small batches",
                                       mvn::SamplingConfig{5000, false, 17},
                                       true);
    CheckSolver<mvn::HaltonSolver>("Halton QMC", mvn::SamplingConfig{}, true);
    CheckSolver<mvn::GenzSolver>("Genz", mvn::GenzConfig{}, false);
    CheckSolver<mvn::GenzSolver>("Genz small batches",
                                 mvn::GenzConfig{5000, false, 17, 10}, false);
    TestAnalytic();
    mvn::Result result(2);
    const Eigen::Vector2d mean(0.1, -0.2), lower(-1, -1), upper(1, 2);
    const Eigen::Matrix2d covariance =
        (Eigen::Matrix2d() << 1, 0.3, 0.3, 2).finished();
    mvn::Status status;
    {
        NoAllocation guard;
        status = mvn::Bivariate(mean, covariance, lower, upper,
                                mvn::Order::kAll, result);
    }
    Returns(mvn::Status::kOk, status, "Bivariate");
    Require(allocations == 0, "Bivariate allocated");
}
