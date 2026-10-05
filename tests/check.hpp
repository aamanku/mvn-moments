#ifndef TESTS_CHECK_HPP_
#define TESTS_CHECK_HPP_

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "mvn_moments/moments.hpp"

// Test checks and fixtures shared by every C++ test: each check throws with
// its label on failure.
namespace check {
const double kInf = std::numeric_limits<double>::infinity();

struct Problem {
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
    Eigen::VectorXd lower;
    Eigen::VectorXd upper;
};

inline void Require(bool condition, const std::string& label)
{
    if (!condition) {
        throw std::runtime_error(label);
    }
}

inline void Near(double actual, double expected, double tolerance,
                 const std::string& label)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::ostringstream message;
        message.precision(17);
        message << label << ": actual " << actual << ", expected " << expected
                << ", tolerance " << tolerance;
        throw std::runtime_error(message.str());
    }
}

inline void Relative(double actual, double expected, double tolerance,
                     const std::string& label)
{
    Near(actual / expected, 1, tolerance, label);
}

// Requires a library call to return the expected status.
inline void Returns(mvn::Status expected, mvn::Status actual,
                    const std::string& label)
{
    if (actual != expected) {
        throw std::runtime_error(label + ": expected status \"" +
                                 mvn::StatusMessage(expected) + "\", got \"" +
                                 mvn::StatusMessage(actual) + "\"");
    }
}

// Requires function() to throw Error; used for the throwing helpers only.
template <class Error = std::invalid_argument, class Function>
void Throws(Function function, const std::string& label)
{
    try {
        function();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error(label + ": expected exception not thrown");
}

// Bitwise equality of every reported field, including standard errors.
inline bool Same(const mvn::Result& a, const mvn::Result& b)
{
    auto equal = [](const auto& x, const auto& y) {
        return x.rows() == y.rows() && x.cols() == y.cols() && x == y;
    };
    return a.samples == b.samples && a.accepted == b.accepted &&
           a.zeroth == b.zeroth && a.zeroth_error == b.zeroth_error &&
           equal(a.First(), b.First()) && equal(a.Second(), b.Second()) &&
           equal(a.FirstError(), b.FirstError()) &&
           equal(a.SecondError(), b.SecondError());
}
}  // namespace check

#endif  // TESTS_CHECK_HPP_
