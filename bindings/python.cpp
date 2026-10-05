#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>

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
    py::class_<Solver>(module, name)
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
