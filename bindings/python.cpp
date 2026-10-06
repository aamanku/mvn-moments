#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>

#include "mvn_moments/detail/common.hpp"
#include "mvn_moments/helper.hpp"
#include "mvn_moments/moments.hpp"
namespace py = pybind11;

namespace {
using Vector = Eigen::Ref<const Eigen::VectorXd>;
using Matrix = Eigen::Ref<const Eigen::MatrixXd>;

// Python reports failures as exceptions: invalid inputs raise ValueError and
// numerical failures raise RuntimeError.
void Raise(mvn::Status status)
{
    switch (status) {
        case mvn::Status::kOk:
            return;
        case mvn::Status::kInvalidInput:
        case mvn::Status::kNotPositiveDefinite:
            throw py::value_error(mvn::StatusMessage(status));
        default:
            throw std::runtime_error(mvn::StatusMessage(status));
    }
}

template <class Solver, class Config>
void BindSolver(py::module_& module, const char* name)
{
    auto binding = py::class_<Solver>(module, name);
    binding
        .def(py::init([](Eigen::Index max_dimension, const Config& config,
                         std::uint64_t seed) {
                 Solver solver;
                 Raise(solver.Init(max_dimension, config, seed));
                 return solver;
             }),
             py::arg("max_dimension"), py::arg("config") = Config{},
             py::arg("seed") = 42)
        .def(
            "compute",
            [](Solver& solver, const Vector& mean, const Matrix& covariance,
               const Vector& lower, const Vector& upper, mvn::Order order) {
                Raise(solver.Compute(mean, covariance, lower, upper, order));
                return solver.GetResult();
            },
            py::arg("mean"), py::arg("covariance"), py::arg("lower"),
            py::arg("upper"), py::arg("order") = mvn::Order::kAll,
            py::call_guard<py::gil_scoped_release>())
        .def_property_readonly(
            "result", [](const Solver& solver) { return solver.GetResult(); });
}

// Python owns allocation and infers the dimension on every call. C++ users
// retain explicit allocation-free methods and solver capacities.
enum class Method { kNone, kAnalytic, kBivariate, kGenz };
struct MomentsConfig {
    mvn::GenzConfig genz;
    mvn::BivariateConfig bivariate;
};

class MomentsSolver {
   public:
    MomentsSolver(const MomentsConfig& config, std::uint64_t seed)
        : config_(config), seed_(seed)
    {
    }

    mvn::Status Compute(const Vector& mean, const Matrix& covariance,
                        const Vector& lower, const Vector& upper,
                        mvn::Order order)
    {
        method_ = Method::kNone;
        mvn::detail::ClearResult(result_);
        const auto status = mvn::detail::ValidateInputs(
            mean, covariance, lower, upper, order, mean.size(), nullptr);
        if (status != mvn::Status::kOk) {
            return status;
        }
        const auto dimension = mean.size();
        if (mvn::detail::ValidateSize(dimension, dimension) !=
            mvn::Status::kOk) {
            return mvn::Status::kInvalidInput;
        }
        if (result_.Capacity() != dimension) {
            result_ = mvn::Result(dimension);
        }

        mvn::Status computed;
        Method selected;
        if (dimension == 1) {
            selected = Method::kAnalytic;
            computed = mvn::Analytic(mean, covariance, lower, upper, order,
                                     result_, config_.genz.timing);
        } else if (dimension == 2) {
            selected = Method::kBivariate;
            computed = mvn::Bivariate(mean, covariance, lower, upper, order,
                                      result_, config_.bivariate);
        } else {
            selected = Method::kGenz;
            if (genz_dimension_ != dimension) {
                genz_dimension_ = 0;
                const auto initialized =
                    genz_.Init(dimension, config_.genz, seed_);
                if (initialized != mvn::Status::kOk) {
                    return initialized;
                }
                genz_dimension_ = dimension;
            }
            computed = genz_.Compute(mean, covariance, lower, upper, order);
            if (computed == mvn::Status::kOk) {
                result_ = genz_.GetResult();
            }
        }
        if (computed == mvn::Status::kOk) {
            method_ = selected;
        }
        return computed;
    }

    const mvn::Result& GetResult() const
    {
        return result_;
    }
    Method GetMethod() const
    {
        return method_;
    }

   private:
    MomentsConfig config_;
    std::uint64_t seed_;
    Eigen::Index genz_dimension_ = 0;
    Method method_ = Method::kNone;
    mvn::GenzSolver genz_;
    mvn::Result result_;
};

// One-shot call through a solver sized for the problem.
template <class Solver, class Config>
mvn::Result Once(const Vector& mean, const Matrix& covariance,
                 const Vector& lower, const Vector& upper, mvn::Order order,
                 const Config& config, std::uint64_t seed)
{
    Solver solver;
    Raise(solver.Init(std::max<Eigen::Index>(mean.size(), 1), config, seed));
    Raise(solver.Compute(mean, covariance, lower, upper, order));
    return solver.GetResult();
}
}  // namespace

PYBIND11_MODULE(_core, module)
{
    module.doc() = "Rectangular normal moments; core raw moment integrals";
    py::enum_<mvn::Order>(module, "Order")
        .value("NONE", mvn::Order::kNone)
        .value("ZEROTH", mvn::Order::kZeroth)
        .value("FIRST", mvn::Order::kFirst)
        .value("SECOND", mvn::Order::kSecond)
        .value("ZEROTH_FIRST", mvn::Order::kZerothFirst)
        .value("ZEROTH_SECOND", mvn::Order::kZerothSecond)
        .value("FIRST_SECOND", mvn::Order::kFirstSecond)
        .value("ALL", mvn::Order::kAll)
        .def(
            "__or__", [](mvn::Order a, mvn::Order b) { return a | b; },
            py::is_operator())
        .def(
            "__and__", [](mvn::Order a, mvn::Order b) { return a & b; },
            py::is_operator());
    // Moment arrays are copies of the selected leading entries.
    py::class_<mvn::Result>(module, "Result")
        .def_readonly("samples", &mvn::Result::samples)
        .def_readonly("accepted", &mvn::Result::accepted)
        .def_readonly("zeroth", &mvn::Result::zeroth)
        .def_property_readonly(
            "first",
            [](const mvn::Result& r) { return Eigen::VectorXd(r.First()); })
        .def_property_readonly(
            "second",
            [](const mvn::Result& r) { return Eigen::MatrixXd(r.Second()); })
        .def_readonly("zeroth_error", &mvn::Result::zeroth_error)
        .def_property_readonly("first_error",
                               [](const mvn::Result& r) {
                                   return Eigen::VectorXd(r.FirstError());
                               })
        .def_property_readonly("second_error", [](const mvn::Result& r) {
            return Eigen::MatrixXd(r.SecondError());
        });
    py::class_<mvn::SamplingConfig>(module, "SamplingConfig")
        .def(py::init<>())
        .def_readwrite("samples", &mvn::SamplingConfig::samples)
        .def_readwrite("timing", &mvn::SamplingConfig::timing)
        .def_readwrite("batch_size", &mvn::SamplingConfig::batch_size);
    py::class_<mvn::GenzConfig>(module, "GenzConfig")
        .def(py::init<>())
        .def_readwrite("samples", &mvn::GenzConfig::samples)
        .def_readwrite("timing", &mvn::GenzConfig::timing)
        .def_readwrite("batch_size", &mvn::GenzConfig::batch_size)
        .def_readwrite("shifts", &mvn::GenzConfig::shifts);

    py::class_<mvn::BivariateConfig>(module, "BivariateConfig")
        .def(py::init<>())
        .def_readwrite("relative_tolerance",
                       &mvn::BivariateConfig::relative_tolerance)
        .def_readwrite("max_intervals", &mvn::BivariateConfig::max_intervals)
        .def_readwrite("timing", &mvn::BivariateConfig::timing);
    py::class_<MomentsConfig>(module, "MomentsConfig")
        .def(py::init<>())
        .def_readwrite("genz", &MomentsConfig::genz)
        .def_readwrite("bivariate", &MomentsConfig::bivariate);
    py::enum_<Method>(module, "Method")
        .value("NONE", Method::kNone)
        .value("ANALYTIC", Method::kAnalytic)
        .value("BIVARIATE", Method::kBivariate)
        .value("GENZ", Method::kGenz);
    py::class_<MomentsSolver>(module, "MomentsSolver")
        .def(py::init<const MomentsConfig&, std::uint64_t>(),
             py::arg("config") = MomentsConfig{}, py::arg("seed") = 42)
        .def(
            "compute",
            [](MomentsSolver& solver, const Vector& mean,
               const Matrix& covariance, const Vector& lower,
               const Vector& upper, mvn::Order order) {
                Raise(solver.Compute(mean, covariance, lower, upper, order));
                return solver.GetResult();
            },
            py::arg("mean"), py::arg("covariance"), py::arg("lower"),
            py::arg("upper"), py::arg("order") = mvn::Order::kAll,
            py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("method", &MomentsSolver::GetMethod)
        .def_property_readonly("result", [](const MomentsSolver& solver) {
            return solver.GetResult();
        });

    module.def(
        "bivariate",
        [](const Vector& mean, const Matrix& covariance, const Vector& lower,
           const Vector& upper, mvn::Order order,
           const mvn::BivariateConfig& config) {
            mvn::Result result(std::max<Eigen::Index>(mean.size(), 1));
            Raise(mvn::Bivariate(mean, covariance, lower, upper, order, result,
                                 config));
            return result;
        },
        py::arg("mean"), py::arg("covariance"), py::arg("lower"),
        py::arg("upper"), py::arg("order") = mvn::Order::kAll,
        py::arg("config") = mvn::BivariateConfig{},
        py::call_guard<py::gil_scoped_release>());
    module.def(
        "moments",
        [](const Vector& mean, const Matrix& covariance, const Vector& lower,
           const Vector& upper, mvn::Order order, const MomentsConfig& config,
           std::uint64_t seed) {
            MomentsSolver solver(config, seed);
            Raise(solver.Compute(mean, covariance, lower, upper, order));
            return solver.GetResult();
        },
        py::arg("mean"), py::arg("covariance"), py::arg("lower"),
        py::arg("upper"), py::arg("order") = mvn::Order::kAll,
        py::arg("config") = MomentsConfig{}, py::arg("seed") = 42,
        py::call_guard<py::gil_scoped_release>());

    BindSolver<mvn::MonteCarloSolver, mvn::SamplingConfig>(module,
                                                           "MonteCarloSolver");
    BindSolver<mvn::HaltonSolver, mvn::SamplingConfig>(module, "HaltonSolver");
    BindSolver<mvn::GenzSolver, mvn::GenzConfig>(module, "GenzSolver");

    module.def(
        "analytic",
        [](const Vector& mean, const Matrix& covariance, const Vector& lower,
           const Vector& upper, mvn::Order order, bool timing) {
            mvn::Result result(std::max<Eigen::Index>(mean.size(), 1));
            Raise(mvn::Analytic(mean, covariance, lower, upper, order, result,
                                timing));
            return result;
        },
        py::arg("mean"), py::arg("covariance"), py::arg("lower"),
        py::arg("upper"), py::arg("order") = mvn::Order::kAll,
        py::arg("timing") = false, py::call_guard<py::gil_scoped_release>());
    module.def("monte_carlo", &Once<mvn::MonteCarloSolver, mvn::SamplingConfig>,
               py::arg("mean"), py::arg("covariance"), py::arg("lower"),
               py::arg("upper"), py::arg("order") = mvn::Order::kAll,
               py::arg("config") = mvn::SamplingConfig{}, py::arg("seed") = 42,
               py::call_guard<py::gil_scoped_release>());
    module.def("quasi_monte_carlo",
               &Once<mvn::HaltonSolver, mvn::SamplingConfig>, py::arg("mean"),
               py::arg("covariance"), py::arg("lower"), py::arg("upper"),
               py::arg("order") = mvn::Order::kAll,
               py::arg("config") = mvn::SamplingConfig{}, py::arg("seed") = 42,
               py::call_guard<py::gil_scoped_release>());
    module.def("genz_quasi_monte_carlo",
               &Once<mvn::GenzSolver, mvn::GenzConfig>, py::arg("mean"),
               py::arg("covariance"), py::arg("lower"), py::arg("upper"),
               py::arg("order") = mvn::Order::kAll,
               py::arg("config") = mvn::GenzConfig{}, py::arg("seed") = 42,
               py::call_guard<py::gil_scoped_release>());
}
