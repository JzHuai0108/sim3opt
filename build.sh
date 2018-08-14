
EIGEN_INCLUDE_DIR=/usr/include/eigen3
CUR_FOLDER=$(pwd)
echo "$CUR_FOLDER has to be /path/to/sim3opt"
mkdir -p $CUR_FOLDER/slam_devel

echo "Download DLoopDetector_demo-1.0.2.tar.gz from http://webdiis.unizar.es/~dorian/index.php?p=33 ..."
wget http://webdiis.unizar.es/~dorian/dl.php?dl=DLoopDetector-1.0.2.tar.gz
mv dl.php?dl=DLoopDetector-1.0.2.tar.gz DLoopDetector-1.0.2.tar.gz
tar xvzf DLoopDetector-1.0.2.tar.gz
rm DLoopDetector-1.0.2.tar.gz

cd DLoopDetector
sed -i '24i#include <unistd.h>' DUtils/FileFunctions.cpp
sed -i '77s/ = 1e3;/;/1' DUtils/Profiler.h
sed -i '78s/ = 1;/;/1' DUtils/Profiler.h

sed -i '37iconst float Profiler::MS = 1e3f;' DUtils/Profiler.cpp
sed -i '38iconst float Profiler::SECONDS = 1.f;' DUtils/Profiler.cpp

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
mkdir -p build
cd build
cmake .. -DCAMKE_BUILD_TYPE=Debug -DTooN_INCLUDE_DIR=$CUR_FOLDER -DSophus_DIR=$CUR_FOLDER/slam_devel/sophus/lib/cmake/Sophus -Dg2o_SOURCE_DIR=$CUR_FOLDER/g2o -DDLoopDetector_PATH=$CUR_FOLDER/DLoopDetector
make


