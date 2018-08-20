
EIGEN_INCLUDE_DIR=/usr/include/eigen3
OpenCV_DIR=/opt/ros/kinetic/share/OpenCV-3.3.1
CUR_FOLDER=$(pwd)
echo "$CUR_FOLDER has to be /path/to/sim3opt"
mkdir -p $CUR_FOLDER/slam_devel

rm -rf DLoopDetector
echo "Download DLoopDetector_demo-1.0.2.tar.gz from http://webdiis.unizar.es/~dorian/index.php?p=33 ..."
wget http://webdiis.unizar.es/~dorian/dl.php?dl=DLoopDetector-1.0.2.tar.gz
mv dl.php?dl=DLoopDetector-1.0.2.tar.gz DLoopDetector-1.0.2.tar.gz
tar xvzf DLoopDetector-1.0.2.tar.gz
rm DLoopDetector-1.0.2.tar.gz

cp TemplatedLoopDetectorPatch.txt DLoopDetector/TemplatedLoopDetectorPatch.txt
cd DLoopDetector
sed -i '24i#include <unistd.h>' DUtils/FileFunctions.cpp
sed -i '77s/ = 1e3;/;/1' DUtils/Profiler.h
sed -i '78s/ = 1;/;/1' DUtils/Profiler.h

sed -i '37iconst float Profiler::MS = 1e3f;' DUtils/Profiler.cpp
sed -i '38iconst float Profiler::SECONDS = 1.f;' DUtils/Profiler.cpp

sed -i '96r TemplatedLoopDetectorPatch.txt' DLoopDetector/TemplatedLoopDetector.h

make

cd ..
git clone https://github.com/RainerKuemmerle/g2o.git

cd g2o
git checkout 8564e1e
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CUR_FOLDER/slam_devel/g2o
make
make install

cd ../..
wget http://www.edwardrosten.com/cvd/TooN-3.0.tar.gz
tar xvzf TooN-3.0.tar.gz
mv TooN-3.0 TooN
rm TooN-3.0.tar.gz


git clone https://github.com/stevenlovegrove/Sophus.git
cd Sophus
git checkout b474f0
mkdir build
cd build
cmake .. -DEIGEN3_INCLUDE_DIR:PATH=$EIGEN_INCLUDE_DIR -DCMAKE_INSTALL_PREFIX=$CUR_FOLDER/slam_devel/sophus
make
make install

cd ../..
git clone https://github.com/JzHuai0108/vio_common
cd vio_common
mkdir -p build
cd build
cmake .. -DOpenCV_DIR=$OpenCV_DIR -DEIGEN_INCLUDE_DIR=$EIGEN_INCLUDE_DIR -DCMAKE_INSTALL_PREFIX=$CUR_FOLDER/slam_devel/vio_common
make
make install

cd ../..
git clone https://github.com/JzHuai0108/vio_g2o
cd vio_g2o
mkdir -p build
cd build
cmake .. -DEIGEN_INCLUDE_DIR=$EIGEN_INCLUDE_DIR -DINSTALL_PREFIX=$CUR_FOLDER/slam_devel -DCMAKE_INSTALL_PREFIX=$CUR_FOLDER/slam_devel/vio_g2o
make

cd ../..
mkdir -p build
cd build
cmake .. -DCAMKE_BUILD_TYPE=Debug -DOpenCV_DIR=$OpenCV_DIR -DTooN_INCLUDE_DIR=$CUR_FOLDER -DSophus_DIR=$CUR_FOLDER/slam_devel/sophus/lib/cmake/Sophus -Dvio_common_DIR=$CUR_FOLDER/slam_devel/vio_common/lib/cmake/vio_common -Dg2o_SOURCE_DIR=$CUR_FOLDER/g2o -DDLoopDetector_PATH=$CUR_FOLDER/DLoopDetector -Dvio_g2o_SOURCE_DIR=$CUR_FOLDER/vio_g2o
make


