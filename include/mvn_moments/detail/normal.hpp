#ifndef DETAIL_NORMAL_HPP_
#define DETAIL_NORMAL_HPP_

#include <Eigen/Core>
#include <cmath>
#include <limits>

// Standard-normal functions used by every method; internal.
//
//     phi(x) = exp(-x^2 / 2) / sqrt(2 pi)      density
//     Phi(x) = erfc(-x / sqrt(2)) / 2          CDF
//
// erfc keeps relative accuracy deep in the lower tail, where 1 + erf(x)
// would cancel. The CDF, density, and interval probability accept double or
// long double.
namespace mvn::detail {
template <class T>
T NormalCdf(T x)
{
    return std::erfc(-x / std::sqrt(T{2})) / 2;
}

template <class T>
T NormalPdf(T x)
{
    return std::exp(-x * x / 2) / std::sqrt(2 * static_cast<T>(EIGEN_PI));
}

// Probability of [a, b]: Phi(b) - Phi(a), or by the symmetry
// Phi(x) = 1 - Phi(-x), Phi(-a) - Phi(-b) when a > 0. Upper-tail intervals
// then subtract small survival probabilities, so e.g. [30, 31] does not round
// to 1 - 1 = 0.
template <class T>
T IntervalProbability(T a, T b)
{
    if (a > 0) {
        return NormalCdf(-a) - NormalCdf(-b);
    }

    return NormalCdf(b) - NormalCdf(a);
}

// Evaluates c[0] + c[1]*x + ... + c[7]*x^7 by Horner's rule.
inline double Polynomial(const double (&c)[8], double x)
{
    double value = 0;
    for (int i = 7; i >= 0; --i) {
        value = value * x + c[i];
    }

    return value;
}

// Inverse CDF Phi^-1(p) by Wichura's Algorithm AS 241 (PPND16), rational
// approximations in three regions of q = p - 1/2:
//
//     |q| <= 0.425:  x = q A(r) / B(r),  r = 0.180625 - q^2
//     otherwise:     r = sqrt(-ln(min(p, 1 - p))), and
//                    x = +-C(r - 1.6) / D(r - 1.6)  if r <= 5,
//                    x = +-E(r - 5) / F(r - 5)      if r > 5,
//
// with the sign of q, and A..F degree-7 polynomials from the paper. Relative
// accuracy is about 1e-16 for p in (0, 1); NaN outside it.
// M. J. Wichura, Applied Statistics 37 (1988) 477-484,
// https://doi.org/10.2307/2347330
inline double NormalQuantile(double p)
{
    if (!(p > 0 && p < 1)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    constexpr double a[8] = {
        3.3871328727963666080e0,  1.3314166789178437745e+2,
        1.9715909503065514427e+3, 1.3731693765509461125e+4,
        4.5921953931549871457e+4, 6.7265770927008700853e+4,
        3.3430575583588128105e+4, 2.5090809287301226727e+3};
    constexpr double b[8] = {1.0,
                             4.2313330701600911252e+1,
                             6.8718700749205790830e+2,
                             5.3941960214247511077e+3,
                             2.1213794301586595867e+4,
                             3.9307895800092710610e+4,
                             2.8729085735721942674e+4,
                             5.2264952788528545610e+3};
    constexpr double c[8] = {
        1.42343711074968357734e0,  4.63033784615654529590e0,
        5.76949722146069140550e0,  3.64784832476320460504e0,
        1.27045825245236838258e0,  2.41780725177450611770e-1,
        2.27238449892691845833e-2, 7.74545014278341407640e-4};
    constexpr double d[8] = {1.0,
                             2.05319162663775882187e0,
                             1.67638483018380384940e0,
                             6.89767334985100004550e-1,
                             1.48103976427480074590e-1,
                             1.51986665636164571966e-2,
                             5.47593808499534494600e-4,
                             1.05075007164441684324e-9};
    constexpr double e[8] = {
        6.65790464350110377720e0,  5.46378491116411436990e0,
        1.78482653991729133580e0,  2.96560571828504891230e-1,
        2.65321895265761230930e-2, 1.24266094738807843860e-3,
        2.71155556874348757815e-5, 2.01033439929228813265e-7};
    constexpr double f[8] = {1.0,
                             5.99832206555887937690e-1,
                             1.36929880922735805310e-1,
                             1.48753612908506148525e-2,
                             7.86869131145613259100e-4,
                             1.84631831751005468180e-5,
                             1.42151175831644588870e-7,
                             2.04426310338993978564e-15};

    const double q = p - 0.5;
    if (std::abs(q) <= 0.425) {
        const double r = 0.180625 - q * q;
        return q * Polynomial(a, r) / Polynomial(b, r);
    }

    double r = std::sqrt(-std::log(q < 0 ? p : 1 - p));
    double value = 0;
    if (r <= 5) {
        r -= 1.6;
        value = Polynomial(c, r) / Polynomial(d, r);
    } else {
        r -= 5;
        value = Polynomial(e, r) / Polynomial(f, r);
    }

    return q < 0 ? -value : value;
}
}  // namespace mvn::detail

#endif  // DETAIL_NORMAL_HPP_
