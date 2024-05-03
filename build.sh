#!/bin/bash

echo "================ install dependencies ================="
sudo apt install build-essential gfortran libsuitesparse-dev pkg-config libfftw3-dev libboost-all-dev fftw-dev libopengl-dev libglfw3-dev libxinerama-dev libxcursor-dev libxi-dev libopencv-dev bc

echo "=================== build ====================="
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DLIBIGL_GLFW=ON -DLIBIGL_IMGUI=ON -DLIBIGL_OPENGL=ON ..
make -j8
cd ..
