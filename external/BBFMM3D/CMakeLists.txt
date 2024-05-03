project(BBFMM3D LANGUAGES CXX)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_FLAGS "-O3 -fopenmp -std=c++14 -fpermissive -Wall")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/;${CMAKE_MODULE_PATH}")

message("${PROJECT_SOURCE_DIR}")

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(FFTW REQUIRED)
if ( FFTW_FOUND )
  message("FFTW libs ${FFTW_LIBRARIES}")
endif (FFTW_FOUND)
find_package(LAPACK REQUIRED)

add_library(bbfmm3d SHARED 
  ${PROJECT_SOURCE_DIR}/src/kernel_Types.cpp
  ${PROJECT_SOURCE_DIR}/src/H2_3D_Tree.cpp
  ${PROJECT_SOURCE_DIR}/src/read_metadata.cpp
  ${PROJECT_SOURCE_DIR}/src/read_sources.cpp
  ${PROJECT_SOURCE_DIR}/src/write_Into_Binary_File.cpp)
target_link_libraries(bbfmm3d
  ${FFTW_LIBRARIES}
  ${LAPACK_LIBRARIES})
