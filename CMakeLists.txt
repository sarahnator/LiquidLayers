cmake_minimum_required(VERSION 3.18)
project(MPMFluid CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Dependencies via FetchContent ─────────────────────────────────────────────
include(FetchContent)

# Polyscope — 3D/2D visualization library
FetchContent_Declare(
  polyscope
  GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
  GIT_TAG        v2.3.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(polyscope)

# Eigen — linear algebra (header-only, used for Vec2/Mat2 math)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(mpm_fluid
  src/main.cpp
  src/simulation.cpp
)

target_include_directories(mpm_fluid PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(mpm_fluid PRIVATE
  polyscope
  Eigen3::Eigen
)

# Compile flags
target_compile_options(mpm_fluid PRIVATE
  $<$<CONFIG:Release>:-O3 -march=native>
  $<$<CONFIG:Debug>:  -O0 -g>
)
