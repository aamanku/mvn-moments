#ifndef DETAIL_PRIMES_HPP_
#define DETAIL_PRIMES_HPP_

#include <cstdint>

// Primality test shared by Halton bases and lattice sizes; internal. A
// composite n has a divisor d with d^2 <= n, so trial division by
// d = 2..floor(sqrt(n)) decides primality exactly.
namespace mvn::detail {
// IsPrime is exact below this limit.
constexpr std::uint64_t kMaxPrime = std::uint64_t{1} << 32;
// Every divisor search stops once divisor^2 exceeds n < kMaxPrime.
constexpr std::uint64_t kMaxDivisor = std::uint64_t{1} << 16;

// Trial division; requires n < kMaxPrime.
inline bool IsPrime(std::uint64_t n)
{
    if (n < 2) {
        return false;
    }

    for (std::uint64_t divisor = 2; divisor <= kMaxDivisor; ++divisor) {
        if (divisor * divisor > n) {
            break;
        }
        if (n % divisor == 0) {
            return false;
        }
    }

    return true;
}
}  // namespace mvn::detail

#endif  // DETAIL_PRIMES_HPP_
