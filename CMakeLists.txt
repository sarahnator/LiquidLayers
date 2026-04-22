cmake_minimum_required(VERSION 3.18)
project(MPMFluid CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
include(FetchContent)
FetchContent_Declare(polyscope
  GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
  GIT_TAG v2.3.0 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(polyscope)
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0 GIT_SHALLOW TRUE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

# add_executable(mpm_fluid src/main.cpp src/simulation.cpp)
# target_include_directories(mpm_fluid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# target_link_libraries(mpm_fluid PRIVATE polyscope Eigen3::Eigen)
# target_compile_options(mpm_fluid PRIVATE
#   $<$<CONFIG:Release>:-O3 -march=native>
#   $<$<CONFIG:Debug>:-O0 -g>)

#   add_executable(mpm_elastic src/main_elastic_drop.cpp src/simulation_elastic_drop.cpp)
# target_include_directories(mpm_elastic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# target_link_libraries(mpm_elastic PRIVATE polyscope Eigen3::Eigen)
# target_compile_options(mpm_elastic PRIVATE
#   $<$<CONFIG:Release>:-O3 -march=native>
#   $<$<CONFIG:Debug>:-O0 -g>)


#     add_executable(mpm_layered src/main_layered_drop.cpp src/simulation_layered_drop.cpp)
# target_include_directories(mpm_layered PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# target_link_libraries(mpm_layered PRIVATE polyscope Eigen3::Eigen)
# target_compile_options(mpm_layered PRIVATE
#   $<$<CONFIG:Release>:-O3 -march=native>
#   $<$<CONFIG:Debug>:-O0 -g>)


#     add_executable(2layer src/main_two_layer_drop.cpp src/simulation_two_layer_drop.cpp)
# target_include_directories(2layer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# target_link_libraries(2layer PRIVATE polyscope Eigen3::Eigen)
# target_compile_options(2layer PRIVATE
#   $<$<CONFIG:Release>:-O3 -march=native>
#   $<$<CONFIG:Debug>:-O0 -g>)


#       add_executable(blobs src/main_many_blob_demo.cpp src/simulation_many_blob_demo.cpp)
# target_include_directories(blobs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# target_link_libraries(blobs PRIVATE polyscope Eigen3::Eigen)
# target_compile_options(blobs PRIVATE
#   $<$<CONFIG:Release>:-O3 -march=native>
#   $<$<CONFIG:Debug>:-O0 -g>)


#       add_executable(incompressible src/main_incompressible_blob_tank.cpp src/simulation_incompressible_blob_tank.cpp)
# target_include_directories(incompressible PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# target_link_libraries(incompressible PRIVATE polyscope Eigen3::Eigen)
# target_compile_options(incompressible PRIVATE
#   $<$<CONFIG:Release>:-O3 -march=native>
#   $<$<CONFIG:Debug>:-O0 -g>)


#     add_executable(2layerfluid src/main_two_layer_drop_fluid_only.cpp src/simulation_two_layer_drop_fluid_only.cpp)
# target_include_directories(2layerfluid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# target_link_libraries(2layerfluid PRIVATE polyscope Eigen3::Eigen)
# target_compile_options(2layerfluid PRIVATE
#   $<$<CONFIG:Release>:-O3 -march=native>
#   $<$<CONFIG:Debug>:-O0 -g>)


    add_executable(mpm_fluid src/main_mpm_fluid.cpp src/mpm_fluid.cpp)
target_include_directories(mpm_fluid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(mpm_fluid PRIVATE polyscope Eigen3::Eigen)
target_compile_options(mpm_fluid PRIVATE
  $<$<CONFIG:Release>:-O3 -march=native>
  $<$<CONFIG:Debug>:-O0 -g>)