project(PBBFMM3D LANGUAGES CXX)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_FLAGS "-O3 -fopenmp -std=c++14 -fpermissive -Wall")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/;${CMAKE_MODULE_PATH}")

message("${PROJECT_SOURCE_DIR}")

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(FFTW3 REQUIRED)
find_package(LAPACK REQUIRED)

add_library(pbbfmm3d SHARED 
  ${PROJECT_SOURCE_DIR}/src/kernel_Types.cpp
  ${PROJECT_SOURCE_DIR}/src/H2_3D_Tree.cpp
  ${PROJECT_SOURCE_DIR}/src/read_metadata.cpp
  ${PROJECT_SOURCE_DIR}/src/read_sources.cpp
  ${PROJECT_SOURCE_DIR}/src/write_Into_Binary_File.cpp
  ${PROJECT_SOURCE_DIR}/src/compute.cpp)
target_link_libraries(pbbfmm3d
  ${FFTW3_LIBRARIES}
  ${LAPACK_LIBRARIES})

# add_executable(test_mykernel examples/get_input_through_routine_mykernel.cpp)
# target_link_libraries(test_mykernel
#   pbbfmm3d
#   )
