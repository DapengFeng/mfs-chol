# OpenCV
find_package(OpenCV)
if(OpenCV_FOUND)
  message(${OpenCV_LIBS})
  add_executable(test_diffusion_curves test_diffusion_curves.cc)
  target_link_libraries(test_diffusion_curves
    cuda_kl_chol    
    ${OpenCV_LIBS}
    mschol
    ${Boost_LIBRARIES}
    )

  add_executable(img2points img2points.cc)
  target_link_libraries(img2points
    cuda_kl_chol    
    ${OpenCV_LIBS}
    mschol
    ${Boost_LIBRARIES}
    )

  add_executable(test_uncertainty test_uncertainty.cc)
  target_link_libraries(test_uncertainty
    cuda_kl_chol    
    ${OpenCV_LIBS}
    mschol
    ${Boost_LIBRARIES}
    igl::core
    )      
endif(OpenCV_FOUND)

add_executable(test_poisson test_poisson.cc)
target_link_libraries(test_poisson
  mschol
  cuda_kl_chol
  igl::core
  igl::imgui
  ${SuiteSparse_LIBRARIES})

add_executable(test_diffusion_meshes test_diffusion_meshes.cc)
target_link_libraries(test_diffusion_meshes
  mschol
  cuda_kl_chol
  igl::core
  igl::imgui
  ${SuiteSparse_LIBRARIES})

add_executable(test_helmholtz_waves test_helmholtz_waves.cc)
target_link_libraries(test_helmholtz_waves
  mschol
  cuda_kl_chol
  igl::core
  igl::imgui
  ${SuiteSparse_LIBRARIES})

add_executable(test_direct_svd test_direct_svd.cc)
target_link_libraries(test_direct_svd
  mschol
  cuda_kl_chol
  igl::core
  igl::imgui
  ${SuiteSparse_LIBRARIES})

add_executable(test_helmholtz_waves_neumann test_helmholtz_waves_neumann.cc)
target_link_libraries(test_helmholtz_waves_neumann
  mschol
  cuda_kl_chol
  igl::core
  igl::imgui
  ${SuiteSparse_LIBRARIES})

add_executable(test_elasticity test_elasticity.cc)
target_link_libraries(test_elasticity
  mschol
  cuda_kl_chol
  igl::core
  igl::imgui
  ${SuiteSparse_LIBRARIES})

add_executable(klchol_profiling profiling.cc)
target_link_libraries(klchol_profiling
  mschol
  cuda_kl_chol
  igl::core)

add_executable(test_scalability test_scalability.cc)
target_link_libraries(test_scalability
  mschol
  cuda_kl_chol
  igl::core
  igl::imgui)

# needed by hlibpro
find_package(TBB)
if (TBB_FOUND)
  message(WARNING "tbb version above 2020 may lead to link error")
  message("tbb version: ${TBB_VERSION}")
  message("tbb libs: ${TBB_LIBRARIES}")
endif (TBB_FOUND)
find_package(HDF5 COMPONENTS CXX HL)
find_package(GSL)
find_package(ZLIB)
find_package(FFTW)
find_package(METIS)
find_package(BZip2)

if (TBB_FOUND AND HDF5_FOUND AND GSL_FOUND AND ZLIB_FOUND AND FFTW_FOUND AND METIS_FOUND AND BZIP2_FOUND)
  add_executable(test_hpro3d test_hpro3d.cc)
  target_link_libraries(test_hpro3d
    /usr/lib/libhpro.so
    ${Boost_LIBRARIES}
    ${BZIP2_LIBRARIES}
    ${ZLIB_LIBRARIES}
    ${METIS_LIBRARIES}
    lapack
    blas  
    ${TBB_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${GSL_LIBRARIES}
    ${FFTW_LIBRARIES}
    cuda_kl_chol 
    mschol
    igl::core)  
endif()
