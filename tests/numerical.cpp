#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "check.hpp"
#include "mvn_moments/helper.hpp"
#include "mvn_moments/moments.hpp"

namespace {
using check::kInf;
using check::Near;
using check::Problem;
using check::Require;
using check::Returns;
using check::Same;
using Eigen::MatrixXd;
using Eigen::VectorXd;

const double kNaN = std::numeric_limits<double>::quiet_NaN();

// Positional: {order, samples, seed, batch_size}.
struct Options {
    mvn::Order order = mvn::Order::kAll;
    std::size_t samples = 100000;
    std::uint64_t seed = 42;
    std::size_t batch_size = 4096;
};

// One fresh solver per call, sized for the problem.
template <class Solver, class Config>
mvn::Status Fresh(const Problem& p, const Options& o, const Config& config,
                  mvn::Result& result)
{
    Solver solver;
    const auto status =
        solver.Init(std::max<Eigen::Index>(p.mean.size(), 1), config, o.seed);
    if (status != mvn::Status::kOk) {
        return status;
    }

    const auto computed =
        solver.Compute(p.mean, p.covariance, p.lower, p.upper, o.order);
    result = solver.GetResult();
    return computed;
}

struct Method {
    std::string name;
    std::function<mvn::Status(const Problem&, const Options&, mvn::Result&)>
        run;
};

// Every sampling method, called through its public API.
const std::vector<Method> kSampling = {
    {"Monte Carlo",
     [](const Problem& p, const Options& o, mvn::Result& r) {
         return Fresh<mvn::MonteCarloSolver>(
             p, o, mvn::SamplingConfig{o.samples, false, o.batch_size}, r);
     }},
    {"Halton QMC",
     [](const Problem& p, const Options& o, mvn::Result& r) {
         return Fresh<mvn::HaltonSolver>(
             p, o, mvn::SamplingConfig{o.samples, false, o.batch_size}, r);
     }},
    {"Genz",
     [](const Problem& p, const Options& o, mvn::Result& r) {
         return Fresh<mvn::GenzSolver>(
             p, o, mvn::GenzConfig{o.samples, false, o.batch_size, 10}, r);
     }},
};

// Sampling methods plus Analytic, which ignores the sampling options.
std::vector<Method> AllMethods()
{
    auto methods = kSampling;
    methods.push_back(
        {"Analytic", [](const Problem& p, const Options& o, mvn::Result& r) {
             r = mvn::Result(std::max<Eigen::Index>(p.mean.size(), 1));
             return mvn::Analytic(p.mean, p.covariance, p.lower, p.upper,
                                  o.order, r);
         }});
    return methods;
}

// Requires success and returns the result.
mvn::Result Solve(const Method& method, const Problem& problem,
                  const Options& options = {})
{
    mvn::Result result;
    Returns(mvn::Status::kOk, method.run(problem, options, result),
            method.name + ": solve");
    return result;
}

mvn::Result Exact(const Problem& problem)
{
    mvn::Result result(1);
    Returns(mvn::Status::kOk,
            mvn::Analytic(problem.mean, problem.covariance, problem.lower,
                          problem.upper, mvn::Order::kAll, result),
            "Analytic reference");
    return result;
}

Problem Interval(double mean, double variance, double lower, double upper)
{
    return {VectorXd::Constant(1, mean), MatrixXd::Constant(1, 1, variance),
            VectorXd::Constant(1, lower), VectorXd::Constant(1, upper)};
}

Problem Correlated()
{
    Problem p{VectorXd(3), MatrixXd(3, 3), VectorXd(3), VectorXd(3)};
    p.mean << 0.2, -0.4, 0.5;
    p.covariance << 1, 0.2, 0.1, 0.2, 2, -0.3, 0.1, -0.3, 1;
    p.lower << -0.7, -0.7, -kInf;
    p.upper << 1, 1, 1;
    return p;
}

void TestAnalytic()
{
    const double density = 1 / std::sqrt(2 * EIGEN_PI);

    const auto standard = Exact(Interval(0, 1, -1, 1));
    Near(*standard.zeroth, 0.6826894921370859, 1e-14, "Interval probability");
    Near(standard.First()(0), 0, 1e-14, "Interval first moment");
    Near(standard.Second()(0, 0), 0.1987480430987992, 1e-14,
         "Interval second moment");

    const auto half = Exact(Interval(0, 1, 0, kInf));
    Near(*half.zeroth, 0.5, 1e-14, "Half-line probability");
    Near(half.First()(0), density, 1e-14, "Half-line first moment");
    Near(half.Second()(0, 0), 0.5, 1e-14, "Half-line second moment");

    const auto tail = Exact(Interval(0, 1, 8, kInf));
    check::Relative(*tail.zeroth, 6.220960574271784e-16, 1e-13,
                    "Tail probability");

    const auto narrow = Exact(Interval(0, 1, 0, 1e-10));
    check::Relative(*narrow.zeroth, density * 1e-10, 1e-13,
                    "Narrow probability");
    check::Relative(narrow.First()(0), density * 0.5e-20, 1e-13,
                    "Narrow first moment");
    check::Relative(narrow.Second()(0, 0), density * 1e-30 / 3, 1e-13,
                    "Narrow second moment");

    mvn::Result two(2);
    Returns(mvn::Status::kInvalidInput,
            mvn::Analytic(VectorXd::Zero(2), MatrixXd::Identity(2, 2),
                          VectorXd::Zero(2), VectorXd::Ones(2),
                          mvn::Order::kAll, two),
            "Two-dimensional analytic");

    mvn::Result empty;
    const auto line = Interval(0, 1, -1, 1);
    Returns(mvn::Status::kInvalidInput,
            mvn::Analytic(line.mean, line.covariance, line.lower, line.upper,
                          mvn::Order::kAll, empty),
            "Analytic without result capacity");
}

void TestScalarReference()
{
    // Per-column accumulation with a partial final batch and masked cross
    // moments against a scalar reference driven by the same Gaussian draws.
    const auto p = Correlated();
    std::mt19937_64 rng(11);
    const auto draws = mvn::SampleMvn(p.mean, p.covariance, 4103, rng);

    VectorXd first = VectorXd::Zero(3);
    MatrixXd second = MatrixXd::Zero(3, 3);
    std::size_t accepted = 0;
    for (Eigen::Index row = 0; row < draws.rows(); ++row) {
        const VectorXd x = draws.row(row).transpose();
        if ((x.array() >= p.lower.array()).all() &&
            (x.array() <= p.upper.array()).all()) {
            ++accepted;
            first += x;
            second += x * x.transpose();
        }
    }

    const auto batched = Solve(kSampling[0], p, {mvn::Order::kAll, 4103, 11});
    Near(*batched.zeroth, static_cast<double>(accepted) / 4103, 1e-14,
         "Scalar reference probability");
    Require(batched.samples == 4103 && batched.accepted == accepted &&
                batched.First().isApprox(first / 4103, 1e-12) &&
                batched.Second().isApprox(second / 4103, 1e-12),
            "Batched moments disagree with scalar reference");
}

void TestAccuracy()
{
    const auto interval = Interval(0, 1, -1, 1);
    const auto exact = Exact(interval);

    Problem whole{VectorXd(2), MatrixXd(2, 2), VectorXd::Constant(2, -kInf),
                  VectorXd::Constant(2, kInf)};
    whole.mean << 1, -2;
    whole.covariance << 2, 0.6, 0.6, 1;

    // Correlated standard-normal positive orthant: probability 1/3.
    Problem orthant{VectorXd::Zero(2), MatrixXd(2, 2), VectorXd::Zero(2),
                    VectorXd::Constant(2, kInf)};
    orthant.covariance << 1, 0.5, 0.5, 1;

    for (const auto& method : kSampling) {
        const auto& name = method.name;

        const auto estimate =
            Solve(method, interval, {mvn::Order::kAll, 150000, 123});
        Near(*estimate.zeroth, *exact.zeroth, 0.008, name + " probability");
        Near(estimate.First()(0), 0, 0.008, name + " first moment");
        Near(estimate.Second()(0, 0), exact.Second()(0, 0), 0.008,
             name + " second moment");

        const auto full = Solve(method, whole, {mvn::Order::kAll, 150000, 17});
        Near(*full.zeroth, 1, 0, name + " whole-space probability");
        for (Eigen::Index i = 0; i < 2; ++i) {
            Near(full.First()(i), whole.mean(i), 0.03,
                 name + " whole-space mean");
            for (Eigen::Index j = 0; j < 2; ++j) {
                Near(full.Second()(i, j),
                     whole.covariance(i, j) + whole.mean(i) * whole.mean(j),
                     0.06, name + " whole-space second moment");
            }
        }

        const auto corner =
            Solve(method, orthant, {mvn::Order::kZeroth, 150000, 7});
        Near(*corner.zeroth, 1.0 / 3, 0.008, name + " orthant probability");
    }

    for (const auto& method : AllMethods()) {
        const auto line = Solve(method, Interval(3, 4, -kInf, kInf),
                                {mvn::Order::kAll, 150000, 3});
        Near(*line.zeroth, 1, 0, method.name + " real-line probability");
        Near(line.First()(0), 3, 0.03, method.name + " real-line mean");
        Near(line.Second()(0, 0), 13, 0.15, method.name + " real-line second");
    }
}

void TestSelection()
{
    const auto interval = Interval(0, 1, -1, 1);
    const auto flat = Interval(0, 1, -1, -1);

    for (const auto& method : AllMethods()) {
        const auto& name = method.name;
        const bool genz = name == "Genz";

        const auto none = Solve(method, interval, {mvn::Order::kNone});
        Require(!none.zeroth && none.First().size() == 0 &&
                    none.Second().size() == 0 && !none.zeroth_error,
                name + ": Order::kNone must return empty outputs");

        const auto second = Solve(method, interval, {mvn::Order::kSecond});
        Require(!second.zeroth && second.First().size() == 0 &&
                    second.Second().rows() == 1 && !second.zeroth_error &&
                    second.FirstError().size() == 0 &&
                    second.SecondError().size() == (genz ? 1 : 0),
                name + ": only selected moments may be returned");

        const auto all = Solve(method, interval);
        Require(all.zeroth_error.has_value() == genz &&
                    all.FirstError().size() == (genz ? 1 : 0),
                name + ": only Genz reports standard errors");

        const auto zero = Solve(method, flat);
        Require(*zero.zeroth == 0 && zero.First().isZero(0) &&
                    zero.Second().isZero(0) && zero.samples == 0,
                name + ": zero-volume rectangles return zero moments");

        mvn::Result result;
        Returns(mvn::Status::kInvalidInput,
                method.run(interval, {static_cast<mvn::Order>(8)}, result),
                name + ": unknown order bits");
    }
}

void TestValidation()
{
    const auto valid = Interval(0, 1, -1, 1);
    struct Case {
        std::string label;
        Problem problem;
        mvn::Status expected;
    };
    const std::vector<Case> invalid = {
        {"empty",
         {VectorXd(), MatrixXd(), VectorXd(), VectorXd()},
         mvn::Status::kInvalidInput},
        {"negative variance", Interval(0, -1, -1, 1),
         mvn::Status::kNotPositiveDefinite},
        {"NaN variance", Interval(0, kNaN, -1, 1), mvn::Status::kInvalidInput},
        {"NaN bound", Interval(0, 1, kNaN, 1), mvn::Status::kInvalidInput},
        {"reversed bounds", Interval(0, 1, 2, 1), mvn::Status::kInvalidInput},
        {"mismatched bounds",
         {VectorXd::Zero(1), MatrixXd::Identity(1, 1), VectorXd::Zero(2),
          VectorXd::Ones(1)},
         mvn::Status::kInvalidInput},
        {"asymmetric",
         {VectorXd::Zero(2), (MatrixXd(2, 2) << 1, 0.5, 0, 1).finished(),
          VectorXd::Constant(2, -1), VectorXd::Ones(2)},
         mvn::Status::kInvalidInput},
        {"indefinite",
         {VectorXd::Zero(2), (MatrixXd(2, 2) << 1, 2, 2, 1).finished(),
          VectorXd::Constant(2, -1), VectorXd::Ones(2)},
         mvn::Status::kNotPositiveDefinite},
    };

    for (const auto& method : AllMethods()) {
        for (const auto& c : invalid) {
            // Analytic rejects every multivariate problem as invalid input.
            const auto expected =
                method.name == "Analytic" && c.problem.mean.size() > 1
                    ? mvn::Status::kInvalidInput
                    : c.expected;
            mvn::Result result;
            Returns(expected, method.run(c.problem, {}, result),
                    method.name + ": " + c.label);
            Require(result.dimension == 0 && !result.zeroth &&
                        result.First().size() == 0,
                    method.name + ": failure must leave an empty result");
        }
    }

    for (const auto& method : kSampling) {
        mvn::Result result;
        Returns(mvn::Status::kInvalidInput,
                method.run(valid, {mvn::Order::kAll, 0}, result),
                method.name + ": zero samples");
        Returns(mvn::Status::kInvalidInput,
                method.run(valid, {mvn::Order::kAll, 100000, 42, 0}, result),
                method.name + ": zero batch size");
    }

    // Rejection sampling cannot estimate a rectangle it never enters.
    const auto rare = Interval(0, 1, 30, 31);
    for (const auto& method : kSampling) {
        if (method.name != "Genz") {
            mvn::Result result;
            Returns(mvn::Status::kNoAcceptedSamples,
                    method.run(rare, {mvn::Order::kAll, 1000}, result),
                    method.name + ": rare rectangle");
        }
    }
}

void TestBatchInvariance()
{
    const auto p = Correlated();

    for (const auto& method : kSampling) {
        const auto reference = Solve(method, p, {mvn::Order::kAll, 4103});
        for (std::size_t batch_size :
             {std::size_t{1}, std::size_t{17}, std::size_t{4103},
              std::numeric_limits<std::size_t>::max()}) {
            const auto result =
                Solve(method, p, {mvn::Order::kAll, 4103, 42, batch_size});
            Require(result.samples == reference.samples &&
                        result.accepted == reference.accepted &&
                        std::abs(*result.zeroth - *reference.zeroth) < 1e-14 &&
                        result.First().isApprox(reference.First(), 1e-12) &&
                        result.Second().isApprox(reference.Second(), 1e-12),
                    method.name + ": batch size changed results");
        }
    }
}

// Covariance + 0.5 I and shifted bounds: a different problem with the same
// dimension and order, which consumes the same random numbers. Problems that
// fail factorization consume none and are kept as they are.
Problem Perturb(const Problem& p)
{
    if (!mvn::IsPositiveDefinite(p.covariance)) {
        return p;
    }

    Problem q = p;
    q.covariance.diagonal().array() += 0.5;
    q.lower.array() += 0.1;
    q.upper.array() += 0.1;
    return q;
}

// A solver's k-th result must depend only on its seed, the number of random
// draws consumed so far, and the k-th inputs; not on cached factors, buffer
// contents from larger or smaller problems, or earlier failures. A fresh twin
// replays perturbed versions of the earlier calls (same random consumption,
// different covariances, so its caches differ) and must then agree bitwise.
template <class Solver, class Config>
void CheckReuse(const std::string& name, const Config& config)
{
    const auto three = Correlated();
    auto changed = three;
    changed.covariance(0, 0) = 1.5;
    const auto one = Interval(0.3, 2, -1, 0.5);
    const Problem indefinite{VectorXd::Zero(2),
                             (MatrixXd(2, 2) << 1, 2, 2, 1).finished(),
                             VectorXd::Constant(2, -1), VectorXd::Ones(2)};
    const std::vector<std::pair<Problem, mvn::Order>> calls = {
        {three, mvn::Order::kAll},   {three, mvn::Order::kAll},
        {changed, mvn::Order::kAll}, {changed, mvn::Order::kZeroth},
        {one, mvn::Order::kAll},     {indefinite, mvn::Order::kAll},
        {three, mvn::Order::kAll},   {three, mvn::Order::kFirstSecond},
    };

    Solver reused;
    Returns(mvn::Status::kOk, reused.Init(3, config, 42), name + ": init");
    for (std::size_t k = 0; k < calls.size(); ++k) {
        const auto& [problem, order] = calls[k];
        const auto status = reused.Compute(problem.mean, problem.covariance,
                                           problem.lower, problem.upper, order);

        Solver twin;
        Returns(mvn::Status::kOk, twin.Init(3, config, 42), name + ": init");
        for (std::size_t j = 0; j < k; ++j) {
            const auto burn = Perturb(calls[j].first);
            twin.Compute(burn.mean, burn.covariance, burn.lower, burn.upper,
                         calls[j].second);
        }
        const auto expected = twin.Compute(problem.mean, problem.covariance,
                                           problem.lower, problem.upper, order);

        const auto label = name + ": reuse call " + std::to_string(k);
        Returns(expected, status, label);
        Require(Same(reused.GetResult(), twin.GetResult()),
                label + " changed results");
    }
}

void TestSolverReuse()
{
    for (const mvn::SamplingConfig& config :
         {mvn::SamplingConfig{}, mvn::SamplingConfig{5000, false, 17}}) {
        CheckReuse<mvn::MonteCarloSolver>("Monte Carlo", config);
        CheckReuse<mvn::HaltonSolver>("Halton QMC", config);
    }
    for (const mvn::GenzConfig& config :
         {mvn::GenzConfig{}, mvn::GenzConfig{5000, false, 17, 10}}) {
        CheckReuse<mvn::GenzSolver>("Genz", config);
    }

    // The random stream advances across calls; a new solver with the same
    // seed restarts it.
    const auto p = Correlated();
    mvn::MonteCarloSolver solver, restarted;
    Returns(mvn::Status::kOk, solver.Init(3, {}, 7), "Stream init");
    Returns(mvn::Status::kOk, restarted.Init(3, {}, 7), "Stream init");
    solver.Compute(p.mean, p.covariance, p.lower, p.upper);
    const auto first = solver.GetResult();
    solver.Compute(p.mean, p.covariance, p.lower, p.upper);
    restarted.Compute(p.mean, p.covariance, p.lower, p.upper);
    Require(*solver.GetResult().zeroth != *first.zeroth,
            "Consecutive calls must draw new samples");
    Require(Same(restarted.GetResult(), first),
            "Same seed must restart the stream");

    // Compute before Init, and a failed Init, leave the solver unusable.
    mvn::GenzSolver uninitialized;
    Returns(mvn::Status::kInvalidInput,
            uninitialized.Compute(p.mean, p.covariance, p.lower, p.upper),
            "Compute before Init");
    Returns(mvn::Status::kInvalidInput, uninitialized.Init(0), "Zero capacity");
    Returns(mvn::Status::kInvalidInput,
            uninitialized.Compute(p.mean, p.covariance, p.lower, p.upper),
            "Compute after failed Init");
    Require(uninitialized.GetResult().Capacity() == 0,
            "Failed Init must leave an empty result");
}
}  // namespace

int main()
{
    TestAnalytic();
    TestScalarReference();
    TestAccuracy();
    TestSelection();
    TestValidation();
    TestBatchInvariance();
    TestSolverReuse();
}
