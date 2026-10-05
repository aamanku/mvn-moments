#ifndef DETAIL_COMMON_HPP_
#define DETAIL_COMMON_HPP_

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "mvn_moments/moments.hpp"

// Internal validation, result setup, retained solver state, moment
// accumulation, and the shared rejection-sampling estimator (equations in
// src/sampler.cpp). Not part of the public API.
namespace mvn::detail {
// Cholesky factor of the last covariance; recomputed only when it changes.
// Storage has capacity for the solver's maximum dimension.
struct CholeskyCache {
    // Leading dimension x dimension blocks hold the cached input and the lower
    // factor. Zero dimension means nothing is cached.
    Eigen::MatrixXd covariance;
    Eigen::MatrixXd factor;
    Eigen::Index dimension = 0;
};

// Monte Carlo and Halton QMC state, allocated by InitSampling.
struct SamplingState {
    SamplingConfig config;
    // Draws per batch: min(samples, batch_size).
    Eigen::Index capacity = 0;
    std::mt19937_64 rng;
    std::normal_distribution<double> normal;
    CholeskyCache cholesky;

    // Batches with one draw per row, so each variable is a contiguous column:
    // standard normals, samples in input coordinates, and rectangle
    // membership as 0/1 weights.
    Eigen::MatrixXd normals;
    Eigen::MatrixXd samples;
    Eigen::VectorXd inside;

    // Halton only: prime bases, shifted uniforms (an even number of columns),
    // the per-call random shift, and Box-Muller radius and angle.
    std::vector<std::size_t> bases;
    Eigen::MatrixXd uniforms;
    Eigen::VectorXd shift;
    Eigen::VectorXd radius;
    Eigen::VectorXd angle;
};

// Shape, capacity, order bits, finiteness, bound order, and covariance
// symmetry. Positive definiteness is checked by each method's factorization.
Status ValidateInputs(const Eigen::Ref<const Eigen::VectorXd>& mean,
                      const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                      const Eigen::Ref<const Eigen::VectorXd>& lower,
                      const Eigen::Ref<const Eigen::VectorXd>& upper,
                      Order order, Eigen::Index capacity, Timer* timer);

Status ValidateBudget(std::size_t samples, std::size_t batch_size);

// Selects moments for n dimensions and zeroes them, including standard
// errors when the method estimates them.
void StartResult(Order order, Eigen::Index n, bool errors, Result& result);

// Empty result after a failure.
void ClearResult(Result& result);

// No moments requested, or a zero-volume rectangle.
bool NothingToCompute(Order order,
                      const Eigen::Ref<const Eigen::VectorXd>& lower,
                      const Eigen::Ref<const Eigen::VectorXd>& upper);

// kNumericalError if any selected moment is not finite.
Status CheckFinite(const Result& result);

// Lower Cholesky factor of covariance, refreshed in cache when it changes.
Status CachedCholesky(const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                      CholeskyCache& cache, Timer* timer);

// For a batch x with one sample per row, adds sum_r weight(r) x_r to first
// and sum_r weight(r) x_r x_r^T to the lower triangle of second; empty
// outputs are skipped. Canonical moment accumulation: contiguous dot
// products, so no product blocking buffers are needed.
void Accumulate(const Eigen::Ref<const Eigen::MatrixXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& weight,
                Eigen::Ref<Eigen::VectorXd> first,
                Eigen::Ref<Eigen::MatrixXd> second);

// Copies the lower triangle into the strict upper triangle.
void MirrorLower(Eigen::Ref<Eigen::MatrixXd> matrix);

// Validates the budget and allocates state for up to max_dimension variables
// with uniform_columns Halton columns (zero for Monte Carlo).
Status InitSampling(Eigen::Index max_dimension, const SamplingConfig& config,
                    std::uint64_t seed, Eigen::Index uniform_columns,
                    SamplingState& state);

// Fills a standard-normal batch, one draw per row. The offset counts draws
// completed before the batch in this call.
using Sampler = void (*)(SamplingState& state, Eigen::Ref<Eigen::MatrixXd> z,
                         std::size_t completed);

// Rejection estimator shared by Monte Carlo and Halton QMC; clears result on
// failure.
Status RunSampling(const Eigen::Ref<const Eigen::VectorXd>& mean,
                   const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                   const Eigen::Ref<const Eigen::VectorXd>& lower,
                   const Eigen::Ref<const Eigen::VectorXd>& upper, Order order,
                   Sampler sampler, SamplingState& state, Result& result,
                   Timer* timer);
}  // namespace mvn::detail

#endif  // DETAIL_COMMON_HPP_
