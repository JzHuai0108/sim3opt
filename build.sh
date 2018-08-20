
EIGEN_INCLUDE_DIR=/usr/include/eigen3
OpenCV_DIR=/opt/ros/kinetic/share/OpenCV-3.3.1
CUR_FOLDER=$(pwd)
echo "$CUR_FOLDER has to be /path/to/sim3opt"
mkdir -p $CUR_FOLDER/slam_devel


rm -rf DLoopDetector
git clone https://github.com/dorian3d/DLoopDetector.git --recursive
sed -i '88r TemplatedLoopDetectorPatch.txt' DLoopDetector/include/DLoopDetector/TemplatedLoopDetector.h
cd DLoopDetector
mkdir -p build
cd build
cmake .. -DOpenCV_DIR=$OpenCV_DIR -DCMAKE_INSTALL_PREFIX=$CUR_FOLDER/slam_devel/DLoopDetector
make
make install


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
cmake .. -DCAMKE_BUILD_TYPE=Debug -DOpenCV_DIR=$OpenCV_DIR -DTooN_INCLUDE_DIR=$CUR_FOLDER -DSophus_DIR=$CUR_FOLDER/slam_devel/sophus/lib/cmake/Sophus -Dvio_common_DIR=$CUR_FOLDER/slam_devel/vio_common/lib/cmake/vio_common -Dg2o_SOURCE_DIR=$CUR_FOLDER/g2o -DDLoopDetector_DIR=$CUR_FOLDER/slam_devel/DLoopDetector/lib/cmake/DLoopDetector -DDBoW2_DIR=$CUR_FOLDER/slam_devel/DLoopDetector/lib/cmake/DBoW2 -DDLib_DIR=$CUR_FOLDER/slam_devel/DLoopDetector/lib/cmake/DLib -Dvio_g2o_SOURCE_DIR=$CUR_FOLDER/vio_g2o
make


