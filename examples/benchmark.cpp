#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <string>

#include "mvn_moments/helper.hpp"
#include "mvn_moments/log.hpp"
#include "mvn_moments/moments.hpp"

namespace {
constexpr std::size_t kRuns = 1000;
constexpr std::size_t kProblemsPerDimension = 3;
constexpr std::size_t kSamples = 10000;
constexpr std::size_t kBatchSize = 4096;
constexpr std::size_t kShifts = 10;
constexpr std::uint64_t kSeed = 42;

// Writes the last of kRuns results; timing uses the library's compile-time
// instrumentation and covers only the timed calls.
using Measure = std::function<mvn::Status(
    const char*, const mvn::NormalDistribution&, const mvn::Hyperrectangle&,
    std::uint64_t, mvn::Result&)>;

struct Method {
    const char* name;
    Measure measure;
};

// One solver is initialized untimed and reused across kRuns calls with
// identical inputs, as in a caller's loop. Its random stream advances, so
// each call draws new samples and an unlikely rectangle can fail some runs;
// those are counted and the last successful result is kept. Fails only if
// every run fails.
template <class Solver, class Config>
mvn::Status MeasureSolver(const char* name, const Config& config,
                          const mvn::NormalDistribution& d,
                          const mvn::Hyperrectangle& r, std::uint64_t seed,
                          mvn::Result& result)
{
    Solver solver;
    mvn::Status status = solver.Init(d.mean.size(), config, seed);
    if (status != mvn::Status::kOk) {
        return status;
    }

    std::size_t failures = 0;
    mvn::Status failure = mvn::Status::kOk;
    {
        MVN_BENCHMARK_TIMER(timer, true);
        for (std::size_t run = 0; run < kRuns; ++run) {
            {
                MVN_SCOPE(timer, name);
                status = solver.Compute(d.mean, d.covariance, r.lower, r.upper);
            }
            if (status == mvn::Status::kOk) {
                result = solver.GetResult();
            } else {
                ++failures;
                failure = status;
            }
        }
    }

    if (failures == kRuns) {
        return failure;
    }
    if (failures != 0) {
        mvn::Log(mvn::Level::kWarning,
                 std::string(name) + ": " + std::to_string(failures) + " of " +
                     std::to_string(kRuns) +
                     " runs failed: " + mvn::StatusMessage(failure));
    }

    return mvn::Status::kOk;
}

const Method kAnalytic = {
    "analytic", [](const char* name, const auto& d, const auto& r,
                   std::uint64_t, mvn::Result& result) {
        (void)name;
        result = mvn::Result(1);
        mvn::Status status = mvn::Status::kOk;
        MVN_BENCHMARK_TIMER(timer, true);
        for (std::size_t run = 0; run < kRuns; ++run) {
            MVN_SCOPE(timer, name);
            status = mvn::Analytic(d.mean, d.covariance, r.lower, r.upper,
                                   mvn::Order::kAll, result);
            if (status != mvn::Status::kOk) {
                break;
            }
        }
        return status;
    }};

const Method kSampling[] = {
    {"monte_carlo",
     [](const char* name, const auto& d, const auto& r, std::uint64_t seed,
        mvn::Result& result) {
         return MeasureSolver<mvn::MonteCarloSolver>(
             name, mvn::SamplingConfig{kSamples, false, kBatchSize}, d, r, seed,
             result);
     }},
    {"quasi_monte_carlo",
     [](const char* name, const auto& d, const auto& r, std::uint64_t seed,
        mvn::Result& result) {
         return MeasureSolver<mvn::HaltonSolver>(
             name, mvn::SamplingConfig{kSamples, false, kBatchSize}, d, r, seed,
             result);
     }},
    {"genz_quasi_monte_carlo",
     [](const char* name, const auto& d, const auto& r, std::uint64_t seed,
        mvn::Result& result) {
         return MeasureSolver<mvn::GenzSolver>(
             name, mvn::GenzConfig{kSamples, false, kBatchSize, kShifts}, d, r,
             seed, result);
     }},
};

// Results go to stdout; timing summaries and errors go to stderr. Generation
// is outside solve scopes.
std::optional<mvn::Result> Solve(const Method& method,
                                 const mvn::NormalDistribution& distribution,
                                 const mvn::Hyperrectangle& rectangle,
                                 std::uint64_t seed)
{
    mvn::Result result;
    const auto status =
        method.measure(method.name, distribution, rectangle, seed, result);
    if (status != mvn::Status::kOk) {
        mvn::Log(mvn::Level::kError,
                 std::string(method.name) + ": " + mvn::StatusMessage(status));
        return std::nullopt;
    }

    return result;
}

void PrintResult(Eigen::Index dimension, std::size_t problem,
                 std::uint64_t seed, const char* method,
                 const mvn::Result& result, const char* reference_method,
                 const std::optional<mvn::Result>& reference)
{
    std::cout << dimension << ',' << problem << ',' << seed << ',' << method
              << ',' << kRuns << ',' << result.samples << ',' << result.accepted
              << ',' << *result.zeroth << ',' << result.First().norm() << ','
              << result.Second().norm() << ',' << reference_method;
    if (reference) {
        std::cout << ',' << std::abs(*result.zeroth - *reference->zeroth) << ','
                  << (result.First() - reference->First()).norm() << ','
                  << (result.Second() - reference->Second()).norm();
    } else {
        std::cout << ",,,";
    }
    std::cout << '\n';
}
}  // namespace

int main()
{
#if !MVN_ENABLE_BENCHMARKS
    mvn::Log(mvn::Level::kWarning,
             "Timing is disabled; rebuild with -DMVN_ENABLE_BENCHMARKS=ON");
#endif
    std::mt19937_64 rng(kSeed);
    bool failed = false;
    std::cout << std::setprecision(12)
              << "dimension,problem,solve_seed,method,runs,samples,accepted,"
                 "probability,"
                 "first_norm,second_norm,reference,probability_difference,"
                 "first_difference_norm,second_difference_norm\n";

    // Problem generators are setup helpers and may throw.
    try {
        for (Eigen::Index dimension : {1, 2, 4, 8}) {
            for (std::size_t problem = 0; problem < kProblemsPerDimension;
                 ++problem) {
                const auto distribution = mvn::GenerateMvn(dimension, rng);
                const auto rectangle =
                    mvn::GenerateHyperrectangle(dimension, rng);
                const auto seed = rng();
                std::optional<mvn::Result> reference;
                const char* reference_method =
                    dimension == 1 ? kAnalytic.name : kSampling[0].name;

                std::cerr << "Problem dimension=" << dimension
                          << " index=" << problem << " seed=" << seed << '\n';
                if (dimension == 1) {
                    reference = Solve(kAnalytic, distribution, rectangle, seed);
                    failed = failed || !reference;
                    if (reference) {
                        PrintResult(dimension, problem, seed, kAnalytic.name,
                                    *reference, reference_method, reference);
                    }
                }

                for (const auto& method : kSampling) {
                    const auto result =
                        Solve(method, distribution, rectangle, seed);
                    failed = failed || !result;
                    if (!result) {
                        continue;
                    }
                    if (std::string(method.name) == reference_method) {
                        reference = result;
                    }
                    PrintResult(dimension, problem, seed, method.name, *result,
                                reference_method, reference);
                }
            }
        }
    } catch (const std::exception& error) {
        mvn::Log(mvn::Level::kError, error.what());
        return 1;
    }

    return failed ? 1 : 0;
}
