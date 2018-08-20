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
find_package(vio_common REQUIRED)
find_package(DLoopDetector REQUIRED)
find_package(DLib REQUIRED)
find_package(DBoW2 REQUIRED)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45.0 COMPONENTS system thread REQUIRED)
set(vio_g2o_INCLUDE_DIR ${vio_g2o_SOURCE_DIR}/include)
set(vio_g2o_LIBRARY ${vio_g2o_SOURCE_DIR}/lib/libvio_g2o.so)

message("G2O_INCLUDE_DIR " ${G2O_INCLUDE_DIR})
message("G2O_CORE_LIBRARY " ${G2O_CORE_LIBRARY})
message("OpenCV_LIBS " ${OpenCV_LIBS})
message("Sophus_INCLUDE_DIR " ${Sophus_INCLUDE_DIR})
message("DLoopDetector_INCLUDE_DIRS " ${DLoopDetector_INCLUDE_DIRS})

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS}

${DLoopDetector_INCLUDE_DIRS}
# for g2o
${Sophus_INCLUDE_DIR}
${EIGEN3_INCLUDE_DIR}
${CHOLMOD_INCLUDE_DIR}
${G2O_INCLUDE_DIR}
${TooN_INCLUDE_DIR}
${DLib_INCLUDE_DIRS}
${DBoW2_INCLUDE_DIRS}
# for vio_g2o
${vio_g2o_INCLUDE_DIR}
${vio_common_INCLUDE_DIR}
)

LIST(APPEND SOURCEFILES
  FSurf64.cpp
  kitti_surf.cpp 
  drawPTAMPoints.cpp
  bal_example.cpp
)

LIST(APPEND LINK_LIBS
${OpenCV_LIBS}
${Boost_LIBRARIES}
${G2O_CORE_LIBRARY}
${G2O_SOLVER_DENSE}
${G2O_SOLVER_EIGEN}
${G2O_STUFF_LIBRARY}
${G2O_TYPES_SBA}
${G2O_TYPES_SIM3}
${G2O_TYPES_SLAM3D}
${DLib_LIBS} 
${DBoW2_LIBS}
${vio_common_LIBRARIES}
${vio_g2o_LIBRARY}
)

add_executable(kitti_surf ${SOURCEFILES})
TARGET_LINK_LIBRARIES(kitti_surf ${LINK_LIBS})
