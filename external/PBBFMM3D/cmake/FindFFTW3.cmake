# - Find the FFTW3 library
#
# Usage:
#   find_package(FFTW3 [REQUIRED] [QUIET] )
#     
# It sets the following variables:
#   FFTW3_FOUND               ... true if fftw is found on the system
#   FFTW3_LIBRARIES           ... full path to fftw library
#   FFTW3_INCLUDES            ... fftw include directory
#
# The following variables will be checked by the function
#   FFTW3_USE_STATIC_LIBS    ... if true, only static libraries are found
#   FFTW3_ROOT               ... if set, the libraries are exclusively searched
#                               under this path
#   FFTW3_LIBRARY            ... fftw library to use
#   FFTW3_INCLUDE_DIR        ... fftw include directory
#
#If environment variable FFTW3DIR is specified, it has same effect as FFTW3_ROOT
if( NOT FFTW3_ROOT AND ENV{FFTW3DIR} )
  set( FFTW3_ROOT $ENV{FFTW3DIR} )
endif()
# Check if we can use PkgConfig
include(CMakeFindDependencyMacro)
find_dependency(PkgConfig)
#Determine from PKG
if( PKG_CONFIG_FOUND AND NOT FFTW3_ROOT )
  pkg_check_modules( PKG_FFTW3 QUIET "fftw3" )
endif()
#Check whether to search static or dynamic libs
set( CMAKE_FIND_LIBRARY_SUFFIXES_SAV ${CMAKE_FIND_LIBRARY_SUFFIXES} )
if( ${FFTW3_USE_STATIC_LIBS} )
  set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX} )
else()
  set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX} )
endif()
if( FFTW3_ROOT )
  #find libs
  find_library(
    FFTW3_LIB
    NAMES "fftw3"
    PATHS ${FFTW3_ROOT}
    PATH_SUFFIXES "lib" "lib64"
    NO_DEFAULT_PATH
  )
  find_library(
    FFTW3F_LIB
    NAMES "fftw3f"
    PATHS ${FFTW3_ROOT}
    PATH_SUFFIXES "lib" "lib64"
    NO_DEFAULT_PATH
  )
  find_library(
    FFTW3L_LIB
    NAMES "fftw3l"
    PATHS ${FFTW3_ROOT}
    PATH_SUFFIXES "lib" "lib64"
    NO_DEFAULT_PATH
  )
  #find includes
  find_path(
    FFTW3_INCLUDES
    NAMES "fftw3.h"
    PATHS ${FFTW3_ROOT}
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
  )
else()
  find_library(
    FFTW3_LIB
    NAMES "fftw3"
    PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
  )
  find_library(
    FFTW3F_LIB
    NAMES "fftw3f"
    PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
  )
  find_library(
    FFTW3L_LIB
    NAMES "fftw3l"
    PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
  )
  find_path(
    FFTW3_INCLUDES
    NAMES "fftw3.h"
    PATHS ${PKG_FFTW3_INCLUDE_DIRS} ${INCLUDE_INSTALL_DIR}
  )
endif()
set(FFTW3_LIBRARIES ${FFTW3_LIB} ${FFTW3F_LIB})
if(FFTW3L_LIB)
  set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3L_LIB})
endif()
set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV} )
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3 DEFAULT_MSG
                                  FFTW3_INCLUDES FFTW3_LIBRARIES)
mark_as_advanced(FFTW3_INCLUDES FFTW3_LIBRARIES FFTW3_LIB FFTW3F_LIB FFTW3L_LIB)
