project(mschol LANGUAGES CXX C CUDA Fortran)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/;${CMAKE_MODULE_PATH}")

# BOOST
find_package(Boost REQUIRED COMPONENTS filesystem system iostreams)
if(Boost_FOUND)
  message("-- Boost @ ${Boost_INCLUDE_DIRS}")
  message("-- Boost libs ${Boost_LIBRARIES}")
  include_directories(${Boost_INCLUDE_DIRS})
endif(Boost_FOUND)

# SuiteSparse
find_package(SuiteSparse REQUIRED)
if(SuiteSparse_FOUND)
  message("-- SuiteSparse @ ${SuiteSparse_INCLUDE_DIRS}")
  message("-- SuiteSparse libs @ ${SuiteSparse_LIBRARIES}")
  include_directories(${SuiteSparse_INCLUDE_DIRS})
endif(SuiteSparse_FOUND)

# CUDA runtime libs
find_package(CUDAToolkit REQUIRED)

# libigl
option(LIBIGL_GLFW ON)
option(LIBIGL_IMGUI ON)
option(LIBIGL_OPENGL ON)
option(LIBIGL_WITH_OPENGL "Use OpenGL" ON)
option(LIBIGL_WITH_OPENGL_GLFW "Use GLFW" ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui" ON)
add_subdirectory(external/libigl)

# HEADER-ONLY LIBS
include_directories(external/eigen/)
include_directories(external/spdlog/include)
include_directories(external/cuda-samples/Common)

# black-box FMM
add_subdirectory(external/PBBFMM3D)
add_subdirectory(external/BBFMM3D)
include_directories(external/PBBFMM3D/include)
include_directories(external/BBFMM3D/include)

include_directories(${PROJECT_SOURCE_DIR})

# C++17 is for compiling Hlibpro
set(CMAKE_CXX_FLAGS "-O3 -Wall -fopenmp -std=c++17 -fpermissive")
add_subdirectory(src)
add_subdirectory(examples)
