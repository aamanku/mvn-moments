cmake_minimum_required(VERSION 3.19)
project(mvn_moments VERSION 1.1.1 LANGUAGES CXX)

# Release uses -O3 with GCC/Clang. Honor explicitly selected build types.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MVN_BUILD_PYTHON "Build Python bindings" OFF)
option(MVN_BUILD_EXAMPLE "Build the C++ example" ON)
option(MVN_BUILD_TESTS "Build infrastructure tests" ON)
option(MVN_ENABLE_BENCHMARKS "Compile scope timing instrumentation" OFF)

option(MVN_DOWNLOAD_EIGEN
    "Download the pinned Eigen 3.4.0 even if Eigen is installed" OFF)

# Use an installed Eigen 3.4+ or 5.x, as on robots. Eigen's version files
# accept only requests within one major version, so each major is requested
# separately; rejected installations are never loaded. Without one, and always
# for wheels, download the checksum-pinned Eigen 3.4.0 headers.
set(MVN_EIGEN_DOWNLOADED OFF)
if(NOT MVN_DOWNLOAD_EIGEN)
    find_package(Eigen3 3.4 QUIET NO_MODULE)
    if(NOT Eigen3_FOUND)
        find_package(Eigen3 5 QUIET NO_MODULE)
    endif()
endif()
if(Eigen3_FOUND)
    message(STATUS "Using installed Eigen ${Eigen3_VERSION}")
else()
    message(STATUS "No installed Eigen 3.4+; downloading Eigen 3.4.0")
    set(MVN_EIGEN_DOWNLOADED ON)
    include(FetchContent)
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()
    # SOURCE_SUBDIR names a missing directory so only headers are used.
    FetchContent_Declare(eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
        SOURCE_SUBDIR mvn_headers_only)
    FetchContent_MakeAvailable(eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}")
endif()

# The Genz lattice construction uses Eigen's header-only FFT module.
get_target_property(MVN_EIGEN_INCLUDE_DIRS Eigen3::Eigen
    INTERFACE_INCLUDE_DIRECTORIES)
find_path(MVN_EIGEN_FFT_DIR unsupported/Eigen/FFT
    PATHS ${MVN_EIGEN_INCLUDE_DIRS} NO_DEFAULT_PATH)
if(NOT MVN_EIGEN_FFT_DIR)
    message(FATAL_ERROR
        "Eigen's unsupported/Eigen/FFT header was not found in "
        "${MVN_EIGEN_INCLUDE_DIRS}; install Eigen with its unsupported modules")
endif()

add_library(mvn_moments
    src/common.cpp
    src/monte_carlo.cpp
    src/sampler.cpp
    src/analytic.cpp
    src/bivariate.cpp
    src/quasi_monte_carlo.cpp
    src/lattice.cpp
    src/genz.cpp)
add_library(mvn_moments::mvn_moments ALIAS mvn_moments)
target_compile_features(mvn_moments PUBLIC cxx_std_17)
target_link_libraries(mvn_moments PUBLIC Eigen3::Eigen)
target_include_directories(mvn_moments PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_definitions(mvn_moments PUBLIC MVN_ENABLE_BENCHMARKS=$<BOOL:${MVN_ENABLE_BENCHMARKS}>)
set_target_properties(mvn_moments PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Fail the build if any LGPL-licensed Eigen header is included.
target_compile_definitions(mvn_moments PRIVATE EIGEN_MPL2_ONLY)
if(MVN_BUILD_TESTS)
    # Lets tests forbid Eigen heap allocation; every translation unit must
    # agree, and installed consumers do not inherit it.
    target_compile_definitions(mvn_moments PUBLIC
        $<BUILD_INTERFACE:EIGEN_RUNTIME_NO_MALLOC>)
endif()

if(MVN_BUILD_EXAMPLE)
    add_executable(mvn_example examples/main.cpp)
    target_link_libraries(mvn_example PRIVATE mvn_moments)
    add_executable(mvn_benchmark examples/benchmark.cpp)
    target_link_libraries(mvn_benchmark PRIVATE mvn_moments)
endif()
if(MVN_BUILD_TESTS)
    enable_testing()
    foreach(name smoke helpers numerical genz realtime bivariate)
        add_executable(mvn_${name}_tests tests/${name}.cpp)
        target_link_libraries(mvn_${name}_tests PRIVATE mvn_moments)
        add_test(NAME ${name} COMMAND mvn_${name}_tests)
    endforeach()
endif()
if(MVN_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_core bindings/python.cpp)
    target_link_libraries(_core PRIVATE mvn_moments)
    target_compile_definitions(_core PRIVATE EIGEN_MPL2_ONLY)
    install(TARGETS _core LIBRARY DESTINATION mvn_moments RUNTIME DESTINATION mvn_moments)
else()
    include(GNUInstallDirs)
    install(TARGETS mvn_moments EXPORT mvn_momentsTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(EXPORT mvn_momentsTargets NAMESPACE mvn_moments::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mvn_moments)
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/mvn_momentsConfigVersion.cmake
        COMPATIBILITY SameMajorVersion)
    # Consumers get the Eigen the library was built with: the same installed
    # Eigen, or the downloaded headers installed alongside the library.
    if(MVN_EIGEN_DOWNLOADED)
        set(MVN_EIGEN_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/mvn_moments/eigen3)
        install(DIRECTORY ${eigen_SOURCE_DIR}/Eigen
            DESTINATION ${MVN_EIGEN_INSTALL_DIR}
            PATTERN CMakeLists.txt EXCLUDE)
        install(FILES ${eigen_SOURCE_DIR}/unsupported/Eigen/FFT
            DESTINATION ${MVN_EIGEN_INSTALL_DIR}/unsupported/Eigen)
        install(DIRECTORY ${eigen_SOURCE_DIR}/unsupported/Eigen/src/FFT
            DESTINATION ${MVN_EIGEN_INSTALL_DIR}/unsupported/Eigen/src)
        install(FILES ${eigen_SOURCE_DIR}/COPYING.MPL2
            ${eigen_SOURCE_DIR}/COPYING.BSD ${eigen_SOURCE_DIR}/COPYING.README
            DESTINATION ${MVN_EIGEN_INSTALL_DIR})
        set(MVN_EIGEN_DEPENDENCY [=[
if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
            "${PACKAGE_PREFIX_DIR}/@MVN_EIGEN_INSTALL_DIR@")
endif()
]=])
    else()
        set(MVN_EIGEN_DEPENDENCY [=[
include(CMakeFindDependencyMacro)
find_dependency(Eigen3 @Eigen3_VERSION@ NO_MODULE)
]=])
    endif()
    string(CONFIGURE "${MVN_EIGEN_DEPENDENCY}" MVN_EIGEN_DEPENDENCY @ONLY)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/mvn_momentsConfig.cmake.in
        "@PACKAGE_INIT@\n${MVN_EIGEN_DEPENDENCY}"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/mvn_momentsTargets.cmake\")\n")
    configure_package_config_file(
        ${CMAKE_CURRENT_BINARY_DIR}/mvn_momentsConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/mvn_momentsConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mvn_moments)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mvn_momentsConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/mvn_momentsConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mvn_moments)
endif()
