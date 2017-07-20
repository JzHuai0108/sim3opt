cmake_minimum_required(VERSION 2.8)
project(KITTILoopDetector)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/cmake_modules/")

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

SET(CMAKE_VERBOSE_MAKEFILE OFF)
SET(HAS_CVD false) # Set False if you want to build this package without it
SET(HAS_BUNDLER_V4 false)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

IF(HAS_CVD)
  ADD_DEFINITIONS(-DUSE_KLEIN)
  LIST(APPEND SOURCEFILES  HomographyInit.cc  ATANCamera.cc)
# for Klein's implementation of Nister's 5 point algorithm.
# To use Klein's implementation, simply copy the relevant files into this project,
# and link blas and lapack shared libaries. Change the copied files a bit for interfacing
#LIBS += -L/usr/lib
#LIBS += -lblas #for 5 point by Klein
#LIBS += -llapack #for 5 point by Klein

  MESSAGE(FATAL_ERROR "cannot find library 'TooN' or 'libCVD' - this will not work ...")
ENDIF()

IF(HAS_BUNDLER_V4)
  ADD_DEFINITIONS(-DUSE_SNAVELY)
# for Snavely's implementation of Nister's 5 point algorithm
#INCLUDEPATH += /home/Bundler-master/include
#INCLUDEPATH += /home/Bundler-master/lib/5point
#INCLUDEPATH += /home/Bundler-master/lib/imagelib
#INCLUDEPATH += /home/Bundler-master/lib/matrix
#for 5 point algorithm by Snavely, to use Snavely's implementation, first download Pierre Moulon's version of Bundler,
# build the shared libraries of 5point, imagelib, matrix, cminpack and cblas, note edit the CMakeLists.txt in each folder
#LIBS += -L/home/Bundler-master/lib/buildlib/bin
#LIBS += -l5point
#LIBS += -limagelib
#LIBS += -lmatrix
#LIBS += -lcminpack

  MESSAGE(FATAL_ERROR "cannot find library 'lib5point' or 'libmatrix' or 'imagelib' - this will not work ...")
ENDIF()

FIND_PACKAGE(OpenCV REQUIRED) # do not put findOpencv.cmake in cmake_module_path which causes an error
find_package(Eigen3 REQUIRED)
find_package(Cholmod REQUIRED)
FIND_PACKAGE(SuiteSparse REQUIRED)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45.0 COMPONENTS system thread REQUIRED)

SET(g2o_INCLUDE_DIR $ENV{HOME}/ScaViSLAM/svslocal/include) # g2o, sophus
SET(g2o_LIBRARY_DIR $ENV{HOME}/ScaViSLAM/svslocal/lib)
link_directories(${g2o_LIBRARY_DIR})
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} 
$ENV{HOME}/DLoopDetector_demo/DBoW2
$ENV{HOME}/DLoopDetector_demo/DUtils
$ENV{HOME}/DLoopDetector_demo/DUtilsCV
$ENV{HOME}/DLoopDetector_demo/DVision
$ENV{HOME}/DLoopDetector_demo/DLoopDetector
#for g2o
${EIGEN3_INCLUDE_DIR}
${CHOLMOD_INCLUDE_DIR}
${CSPARSE_INCLUDE_DIR}
${g2o_INCLUDE_DIR}

$ENV{HOME}/ScaViSLAM/svslocal/include/TooN
${PROJECT_SOURCE_DIR}/g2o_types
)

LIST(APPEND SOURCEFILES
  kitti_surf.cpp 
  drawPTAMPoints.cpp
  bal_example.cpp

  g2o_types/rand_sampler.cpp
  g2o_types/scale_solver.cpp
  g2o_types/anchored_points.cpp
  g2o_types/maths_utils.cpp
)

LIST(APPEND LINK_LIBS
${OpenCV_LIBS}
cholmod
csparse
cxsparse
${Boost_LIBRARIES}
${g2o_LIBRARY_DIR}/libg2o_core.so
${g2o_LIBRARY_DIR}/libg2o_solver_cholmod.so
${g2o_LIBRARY_DIR}/libg2o_solver_dense.so
${g2o_LIBRARY_DIR}/libg2o_stuff.so
${g2o_LIBRARY_DIR}/libg2o_solver_csparse.so
${g2o_LIBRARY_DIR}/libg2o_csparse_extension.so
${g2o_LIBRARY_DIR}/libg2o_types_sba.so
${g2o_LIBRARY_DIR}/libg2o_types_sim3.so
${g2o_LIBRARY_DIR}/libg2o_types_slam3d.so
$ENV{HOME}/DLoopDetector_demo/lib/libDBoW2.so
$ENV{HOME}/DLoopDetector_demo/lib/libDUtils.so
$ENV{HOME}/DLoopDetector_demo/lib/libDUtilsCV.so
$ENV{HOME}/DLoopDetector_demo/lib/libDVision.so
)
add_executable(kitti_surf ${SOURCEFILES})
TARGET_LINK_LIBRARIES(kitti_surf ${LINK_LIBS})
