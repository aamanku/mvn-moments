#ifndef MOMENTS_HPP_
#define MOMENTS_HPP_

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "timer.hpp"

// Moments of a multivariate normal distribution over a rectangle.
//
// Let X ~ N(mu, Sigma) in R^n, with density
//
//     phi(x) = exp(-(x - mu)^T Sigma^-1 (x - mu) / 2)
//              / sqrt((2 pi)^n det(Sigma)),
//
// and let S = [a, b] = {x : a_i <= x_i <= b_i}, where bounds may be infinite.
// The library computes the raw, unnormalized integrals
//
//     Z  = int_S phi(x) dx         = P(X in S)            (Order::kZeroth)
//     m1 = int_S x phi(x) dx       = E[X 1{X in S}]       (Order::kFirst)
//     M2 = int_S x x^T phi(x) dx   = E[X X^T 1{X in S}]   (Order::kSecond)
//
// The truncated (conditional) moments follow by normalizing:
//
//     E[X | X in S]   = m1 / Z
//     Cov[X | X in S] = M2 / Z - (m1 / Z) (m1 / Z)^T
//
// Each method's equations and sources are at the top of its source file:
//
//     Analytic          closed form for n = 1     src/analytic.cpp
//     MonteCarloSolver  rejection sampling        src/sampler.cpp,
//                                                 src/monte_carlo.cpp
//     HaltonSolver      rejection sampling with   src/sampler.cpp,
//                       random-shift Halton       src/quasi_monte_carlo.cpp
//     GenzSolver        separation of variables   src/genz.cpp,
//                       on a shifted lattice      src/lattice.cpp

namespace mvn {
enum class Order : unsigned {
    kNone = 0b000,
    kZeroth = 0b001,
    kFirst = 0b010,
    kSecond = 0b100,
    kZerothFirst = 0b011,
    kZerothSecond = 0b101,
    kFirstSecond = 0b110,
    kAll = 0b111,
};

constexpr Order operator|(Order left, Order right)
{
    return static_cast<Order>(static_cast<unsigned>(left) |
                              static_cast<unsigned>(right));
}

constexpr Order operator&(Order left, Order right)
{
    return static_cast<Order>(static_cast<unsigned>(left) &
                              static_cast<unsigned>(right));
}

// True if every bit in requested is present (including the empty selection).
constexpr bool HasOrder(Order selection, Order requested)
{
    return (selection & requested) == requested;
}

// Outcome of every computation; the library reports failures only through
// these codes.
enum class Status {
    kOk,
    // Shapes, values, configuration, capacity, or an uninitialized solver.
    kInvalidInput,
    kNotPositiveDefinite,
    // Genz reordering found a pivot below its tolerance.
    kNumericallySingular,
    // Rejection sampling never entered a nondegenerate rectangle.
    kNoAcceptedSamples,
    // Nonfinite samples or moments, or an underflowed probability.
    kNumericalError,
};

constexpr const char* StatusMessage(Status status)
{
    switch (status) {
        case Status::kOk:
            return "OK";
        case Status::kInvalidInput:
            return "Invalid input, configuration, or capacity, or the solver "
                   "is not initialized";
        case Status::kNotPositiveDefinite:
            return "Covariance must be positive definite";
        case Status::kNumericallySingular:
            return "Covariance is numerically singular for Genz reordering";
        case Status::kNoAcceptedSamples:
            return "No samples entered the rectangle; increase samples or use "
                   "Genz";
        case Status::kNumericalError:
            return "Arithmetic overflow, underflow, or cancellation";
    }

    return "Unknown status";
}

// Raw integrals of 1, x, and xx^T over the rectangle from the last
// computation. Storage is allocated once, for up to Capacity() variables;
// the accessors return views of the leading dimension entries and are empty
// when that moment was not selected (or, for errors, not estimated). A failed
// computation leaves the result empty.
struct Result {
    Result() = default;
    explicit Result(Eigen::Index max_dimension);

    Eigen::Index Capacity() const
    {
        return first_storage.size();
    }

    Eigen::VectorBlock<const Eigen::VectorXd> First() const
    {
        return first_storage.head(HasOrder(order, Order::kFirst) ? dimension
                                                                 : 0);
    }

    Eigen::Block<const Eigen::MatrixXd> Second() const
    {
        const auto n = HasOrder(order, Order::kSecond) ? dimension : 0;
        return second_storage.topLeftCorner(n, n);
    }

    Eigen::VectorBlock<const Eigen::VectorXd> FirstError() const
    {
        return first_error_storage.head(
            errors && HasOrder(order, Order::kFirst) ? dimension : 0);
    }

    Eigen::Block<const Eigen::MatrixXd> SecondError() const
    {
        const auto n =
            errors && HasOrder(order, Order::kSecond) ? dimension : 0;
        return second_error_storage.topLeftCorner(n, n);
    }

    // Zero for analytic results.
    std::size_t samples = 0;
    // Sampling points inside the rectangle.
    std::size_t accepted = 0;
    // Missing if not selected; distinct from zero.
    std::optional<double> zeroth;
    // Standard error across randomized replicates; only Genz estimates it.
    std::optional<double> zeroth_error;

    // Problem dimension, selected moments, and whether standard errors were
    // estimated; set by each computation.
    Eigen::Index dimension = 0;
    Order order = Order::kNone;
    bool errors = false;

    // Capacity storage; computations write the leading entries.
    Eigen::VectorXd first_storage;
    Eigen::MatrixXd second_storage;
    Eigen::VectorXd first_error_storage;
    Eigen::MatrixXd second_error_storage;
};

// Budget for each Compute call. batch_size is the maximum number of draws per
// batch; both must be positive.
struct SamplingConfig {
    std::size_t samples = 100000;
    bool timing = false;
    std::size_t batch_size = 4096;
};

struct GenzConfig {
    // Total budget across all shifts.
    std::size_t samples = 100000;
    bool timing = false;
    std::size_t batch_size = 4096;
    // Random lattice shifts; at least 2.
    std::size_t shifts = 10;
};

namespace detail {
struct SamplingState;
struct GenzState;
}  // namespace detail

// Solvers for repeated calls, e.g. in a control loop. Init validates the
// configuration, allocates all storage for up to max_dimension variables, and
// seeds a random stream that advances across Compute calls; a failed Init
// leaves the solver uninitialized. Compute validates
// its inputs, never allocates or throws, and reports failures through Status;
// after a failure the solver remains usable. Not thread-safe; use one solver
// per thread.
//
// All methods compute selected raw integrals of 1, x, and xx^T.
// Order::kNone validates inputs and returns empty outputs.

// Fixed-budget rejection sampling with pseudo-random normals, without
// accuracy certification.
class MonteCarloSolver {
   public:
    MonteCarloSolver();
    ~MonteCarloSolver();
    MonteCarloSolver(MonteCarloSolver&&) noexcept;
    MonteCarloSolver& operator=(MonteCarloSolver&&) noexcept;

    Status Init(Eigen::Index max_dimension, const SamplingConfig& config = {},
                std::uint64_t seed = 42);

    Status Compute(const Eigen::Ref<const Eigen::VectorXd>& mean,
                   const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                   const Eigen::Ref<const Eigen::VectorXd>& lower,
                   const Eigen::Ref<const Eigen::VectorXd>& upper,
                   Order order = Order::kAll);

    // Empty before a successful Init.
    const Result& GetResult() const;

   private:
    std::unique_ptr<detail::SamplingState> state_;
    Result result_;
};

// Rejection sampling with a randomly shifted Halton sequence; otherwise as
// MonteCarloSolver.
class HaltonSolver {
   public:
    HaltonSolver();
    ~HaltonSolver();
    HaltonSolver(HaltonSolver&&) noexcept;
    HaltonSolver& operator=(HaltonSolver&&) noexcept;

    Status Init(Eigen::Index max_dimension, const SamplingConfig& config = {},
                std::uint64_t seed = 42);

    Status Compute(const Eigen::Ref<const Eigen::VectorXd>& mean,
                   const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                   const Eigen::Ref<const Eigen::VectorXd>& lower,
                   const Eigen::Ref<const Eigen::VectorXd>& upper,
                   Order order = Order::kAll);

    const Result& GetResult() const;

   private:
    std::unique_ptr<detail::SamplingState> state_;
    Result result_;
};

// Genz's separation-of-variables integration with variable reordering and a
// randomly shifted CBC rank-1 lattice. Every point lies in the rectangle, so
// rare rectangles do not fail. Uses config.shifts lattices of a prime size
// N <= samples / shifts, built once in Init; Result::samples reports
// shifts * N. Fills the standard errors across shifts.
class GenzSolver {
   public:
    GenzSolver();
    ~GenzSolver();
    GenzSolver(GenzSolver&&) noexcept;
    GenzSolver& operator=(GenzSolver&&) noexcept;

    Status Init(Eigen::Index max_dimension, const GenzConfig& config = {},
                std::uint64_t seed = 42);

    Status Compute(const Eigen::Ref<const Eigen::VectorXd>& mean,
                   const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                   const Eigen::Ref<const Eigen::VectorXd>& lower,
                   const Eigen::Ref<const Eigen::VectorXd>& upper,
                   Order order = Order::kAll);

    const Result& GetResult() const;

   private:
    std::unique_ptr<detail::GenzState> state_;
    Result result_;
};

// Closed form for one dimension; result needs capacity for one variable.
// Does not allocate.
Status Analytic(const Eigen::Ref<const Eigen::VectorXd>& mean,
                const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                const Eigen::Ref<const Eigen::VectorXd>& lower,
                const Eigen::Ref<const Eigen::VectorXd>& upper, Order order,
                Result& result, bool timing = false);
}  // namespace mvn

#endif  // MOMENTS_HPP_
