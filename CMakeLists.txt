cmake_minimum_required(VERSION 2.8)
project(KITTILoopDetector)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

LIST(APPEND CMAKE_MODULE_PATH 
  "${PROJECT_SOURCE_DIR}/cmake_modules/"
  "${g2o_SOURCE_DIR}/cmake_modules/")
message("cmake_module_path " ${CMAKE_MODULE_PATH})
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

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Cholmod REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45.0 COMPONENTS system thread REQUIRED)

message("g2o_INCLUDE_DIR " ${G2O_INCLUDE_DIR})
message("g2o_LIBRARY_DIR " ${G2O_LIBRARY_DIR})
message("g2o_LIBRARIES " ${G2O_LIBRARIES})
message("Sophus_INCLUDE_DIR " ${Sophus_INCLUDE_DIR})
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS}
${DLoopDetector_PATH}/DBoW2
${DLoopDetector_PATH}/DUtils
${DLoopDetector_PATH}/DUtilsCV
${DLoopDetector_PATH}/DVision
${DLoopDetector_PATH}/DLoopDetector
#for g2o
${Sophus_INCLUDE_DIR}
${EIGEN3_INCLUDE_DIR}
${CHOLMOD_INCLUDE_DIR}
${G2O_INCLUDE_DIR}
${TooN_INCLUDE_DIR}
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
g2o_core
g2o_solver_cholmod
g2o_solver_dense
g2o_stuff
g2o_solver_csparse
g2o_csparse_extension
g2o_types_sba
g2o_types_sim3
g2o_types_slam3d
${DLoopDetector_PATH}/lib/libDBoW2.so
${DLoopDetector_PATH}/lib/libDUtils.so
${DLoopDetector_PATH}/lib/libDUtilsCV.so
${DLoopDetector_PATH}/lib/libDVision.so
)
add_executable(kitti_surf ${SOURCEFILES})
TARGET_LINK_LIBRARIES(kitti_surf ${LINK_LIBS})
