# sim3opt

optimize Sim3 pose graph

This program compares the performance of stepwise and direct pose graph optimization on monocular VO output for KITTI sequence 00.

## Dependencies

1. [DBoW2](http://webdiis.unizar.es/~dorian/index.php?p=33)
2. [g2o](https://github.com/RainerKuemmerle/g2o)
3. [TooN 2.2](http://www.edwardrosten.com/cvd/toon.html)
4. Sophus
5. Eigen3
6. OpenCV
7. Boost

## Installation

1. Install OpenCV and Boost system wide as required by [DBoW2](http://webdiis.unizar.es/~dorian/index.php?p=33). If you have installed a ros distribution with opencv, no need to install OpenCV anymore.
Boost can be installed by

```
sudo apt-get install libboost-dev
```

OpenCV3 can be installed by
```
sudo apt-get install ros-kinetic-opencv3
```

2. Build
 
```
chmod +x build.sh
./build.sh
```

## Run

```
cd ../bin
./kitti_surf 3
```

## Visualization

To visualize the results comparing stepwise pose graph optimization vs direct pose graph optimization, you can do it in matlab or gnuplot. In case of gnuplot, navigate to the folder data/intermediate, in a terminal,

```
gnuplot
plot "stepwise_2solvers.txt" using 3:5 lc rgb "#FF0000"  title 'stepwise' w lines, "direct_embedded.txt" using 3:5 lc rgb "#0000FF" title 'direct' w lines, "StepwiseMapAfterBA6.txt" using 2:4 lc rgb "#00FF00" title 'stepwise traj after BA on corrected map' w lines, "framePoses.txt" using 6:8 lc rgb "#000000" title 'monocular VO output' w lines
```

