#ifndef DETAIL_LATTICE_HPP_
#define DETAIL_LATTICE_HPP_

#include <cstdint>
#include <vector>

#include "mvn_moments/moments.hpp"

// Rank-1 lattice construction for Genz integration; internal. The method and
// sources are described in src/lattice.cpp.
namespace mvn::detail {
// Lattice sizes stay below 2^32 so products of two residues fit in 64 bits.
constexpr std::uint64_t kMaxLatticePoints = std::uint64_t{1} << 32;

struct Lattice {
    std::vector<std::uint64_t> generator;  // One integer per dimension.
    std::uint64_t points = 0;              // Prime lattice size N.
};

// Largest prime N <= target_points with a fast component-by-component
// generator (Nuyens and Cools, 2006), ported from SciPy's _cbc_lattice.
// Point k has coordinates (k * generator[i] mod N) / N. Requires
// 3 <= target_points < kMaxLatticePoints; zero dimensions give an empty
// generator. Generator i does not depend on later dimensions, so a prefix
// of a larger lattice is the lattice for fewer dimensions. Setup only: it
// allocates.
Status CbcLattice(std::size_t dimensions, std::uint64_t target_points,
                  Lattice& lattice);
}  // namespace mvn::detail

#endif  // DETAIL_LATTICE_HPP_
