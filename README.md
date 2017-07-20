# sim3opt
optimize Sim3 pose graph

This program compares the performance of stepwise and direct pose graph optimization on monocular VO output for KITTI sequence 00.

Installation

1. Download [DLoopDetector_demo-1.0.2.tar.gz](http://webdiis.unizar.es/~dorian/index.php?p=33).
2. Extract DLoopDetector_demo-1.0.2.tar.gz into a folder e.g. DLoopDetector_demo such that the folder has subfolders like DBoW2, DLoopDetector, etc.
3. Build DBoW2 libraries as instructed in that webpage for DBoW2, thus, boost and opencv should be installed.
4. Download and install [g2o](https://github.com/RainerKuemmerle/g2o) as instructed on github, assume g2o's dependence Suitesparse is installed which includes CHOLMOD, CSparse and CXSparse.
5. Download the header only library [TooN 2.2](http://www.edwardrosten.com/cvd/toon.html)
6. Download and unzip kittiloopdetector.zip and put it into the DLoopDetector_demo folder.
7. navigate to DLoopDetector_demo/kittiloopdetector, in a terminal;
    mkdir build; cd build; cmake -DCAMKE_BUILD_TYPE= Debug ..; cd ../bin; ./kitti_surf 3
8. To visualize the results comparing stepwise pose graph optimization vs direct pose graph optimization, you can do it in matlab or gnuplot. In case of gnuplot, navigate to the folder data/intermediate, in a terminal,
    gnuplot; plot "stepwise_2solvers.txt" using 3:5 lc rgb "#FF0000"  title 'stepwise' w lines, "direct_embedded.txt" using 3:5 lc rgb "#0000FF" title 'direct' w lines, "StepwiseMapAfterBA6.txt" using 2:4 lc rgb "#00FF00" title 'stepwise traj after BA on corrected map' w lines, "framePoses.txt" using 6:8 lc rgb "#000000" title 'monocular VO output' w lines

A renovation of the program is pending.
