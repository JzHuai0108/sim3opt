// g2o - General Graph Optimization
// Copyright (C) 2012 R. Kümmerle
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "bal_example.h"

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/command_args.h"

#include <iostream>

using namespace std;
int ba_demo(int argc, char* argv[]){
    int maxIterations;
    bool verbose;
    bool usePCG;
    string outputFilename;
    string inputFilename;
    string statsFilename;
    g2o::CommandArgs arg;
    arg.param("i", maxIterations, 5, "perform n iterations");
    arg.param("o", outputFilename, "", "write points into a vrml file");
    arg.param("pcg", usePCG, false, "use PCG instead of the Cholesky");
    arg.param("v", verbose, false, "verbose output of the optimization process");
    arg.param("stats", statsFilename, "", "specify a file for the statistics");
    arg.paramLeftOver("graph-input", inputFilename, "", "file which will be processed");

    arg.parseArgs(argc, argv);

    double PIXEL_NOISE = 1.0;
    bool ROBUST_KERNEL = true;
    bool STRUCTURE_ONLY = false;
    bool DENSE = false;

    cout << "PIXEL_NOISE: " <<  PIXEL_NOISE << endl;
    cout << "ROBUST_KERNEL: " << ROBUST_KERNEL << endl;
    cout << "STRUCTURE_ONLY: " << STRUCTURE_ONLY<< endl;
    cout << "DENSE: "<<  DENSE << endl;

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(verbose);
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    if (DENSE) {
        linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o
                ::BlockSolver_6_3::PoseMatrixType> >();
    } else {
        linearSolver = g2o::make_unique<g2o::LinearSolverEigen<
                g2o::BlockSolver_6_3::PoseMatrixType> >();
    }

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
       g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));

    optimizer.setAlgorithm(solver);

    double focal_length= 718.856;
    Eigen::Vector2d principal_point(607.1928, 185.2157);

    g2o::CameraParameters * cam_params
            = new g2o::CameraParameters (focal_length, principal_point, 0.);
    cam_params->setId(0);

    if (!optimizer.addParameter(cam_params)) {
        assert(false);
    }

    vector<g2o::VertexSE3Expmap*> cameras;
    vector<g2o::VertexSBAPointXYZ *> points;

    // parse BAL dataset
    cout << "Loading BAL dataset " << inputFilename << endl;

    ifstream ifs(inputFilename.c_str());
    int numCameras, numPoints, numObservations;
    ifs >> numCameras >> numPoints >> numObservations;

    cerr << PVAR(numCameras) << " " << PVAR(numPoints) << " " << PVAR(numObservations) << endl;

    int id = 0;
    cameras.reserve(numCameras);
    for (int i = 0; i < numCameras; ++i, ++id) {
        g2o::VertexSE3Expmap * cam = new g2o::VertexSE3Expmap();
        cam->setId(id);
        optimizer.addVertex(cam);
        cameras.push_back(cam);
    }

    points.reserve(numPoints);
    for (int i = 0; i < numPoints; ++i, ++id) {
        g2o::VertexSBAPointXYZ * p = new g2o::VertexSBAPointXYZ();
        p->setId(id);
        p->setMarginalized(true);
        bool addedVertex = optimizer.addVertex(p);
        if (! addedVertex) {
            cerr << "failing adding vertex" << endl;
        }
        points.push_back(p);
    }

    // read in the observation
    double maxError=0;
    for (int i = 0; i < numObservations; ++i) {
        int camIndex, pointIndex;
        double obsX, obsY;
        ifs >> camIndex >> pointIndex >> obsX >> obsY;

        assert(camIndex >= 0 && (size_t)camIndex < cameras.size() && "Index out of bounds");
        g2o::VertexSE3Expmap* cam = cameras[camIndex];
        assert(pointIndex >= 0 && (size_t)pointIndex < points.size() && "Index out of bounds");
        g2o::VertexSBAPointXYZ * point = points[pointIndex];

        g2o::EdgeProjectXYZ2UV * e = new g2o::EdgeProjectXYZ2UV();
        e->setVertex(0, point);
        e->setVertex(1, cam);
        e->setInformation(Eigen::Matrix2d::Identity()/(PIXEL_NOISE*PIXEL_NOISE));
        e->setMeasurement(Eigen::Vector2d(obsX, obsY));

        if (ROBUST_KERNEL) {
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            rk->setDelta(2.5);
            e->setRobustKernel(rk);
        }
        e->setParameterId(0, 0);

        bool addedEdge = optimizer.addEdge(e);
        if (! addedEdge) {
            cerr << "error adding edge" << endl;
        }

        e->computeError();
        Eigen::Vector2d error = e->error();
        double rootChi2 = error.norm();
        maxError = maxError < rootChi2 ? rootChi2: maxError;

    }
    std::cout<< "max edge error norm "<< maxError << std::endl;
    // read in the camera params
    Eigen::VectorXd cameraParameter(9);
    for (int i = 0; i < numCameras; ++i) {
        for (int j = 0; j < 9; ++j)
            ifs >> cameraParameter(j);
        g2o::VertexSE3Expmap* cam = cameras[i];

        double angle_axis[3]= {cameraParameter[0], cameraParameter[1], cameraParameter[2]};

        double qw2cData[4];
        AngleAxisToQuaternion(angle_axis, qw2cData);
        Eigen:: Quaterniond qw2c(qw2cData[0],qw2cData[1],qw2cData[2],qw2cData[3]);

        Eigen::Vector3d trans(cameraParameter[3], cameraParameter[4], cameraParameter[5]);
        g2o::SE3Quat pose(qw2c,trans);

        cam->setEstimate(pose);
    }

    // read in the points
    Eigen::Vector3d p;
    for (int i = 0; i < numPoints; ++i) {
        ifs >> p(0) >> p(1) >> p(2);

        g2o::VertexSBAPointXYZ * point = points[i];
        point->setEstimate(p);
    }

    cout << "done." << endl;

    optimizer.initializeOptimization();
    if (STRUCTURE_ONLY){
        g2o::StructureOnlySolver<3> structure_only_ba;
        cout << "Performing structure-only BA:"   << endl;
        g2o::OptimizableGraph::VertexContainer points;
        for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it) {
            g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
            if (v->dimension() == 3)
                points.push_back(v);
        }
        structure_only_ba.calc(points, 10);
    }

    cout << endl;
    cout << "Performing full BA:" << endl;
    optimizer.optimize(maxIterations);
    cout << endl;

    if (outputFilename.size() > 0) {
        ofstream fout(outputFilename.c_str());

        // output map points
//        fout<<"%SE3 optimization result: map point in global frame "<<endl;
//        for (vector<g2o::VertexSBAPointXYZ *>::const_iterator it = points.begin(); it != points.end(); ++it) {
//            fout << (*it)->estimate().transpose() << endl;
//        }
//        fout << "    ]\n" << "  }\n" << "}\n" << "  }\n";
        fout<<"% SE3 optimization result: kf id, tcinw, rc2w(qxyzw):"<<endl;
        int jack =0;

        for (vector<g2o::VertexSE3Expmap*>::const_iterator it = cameras.begin(); it != cameras.end(); ++it) {
            g2o::SE3Quat est = (*it)->estimate();

            Eigen::Quaterniond qw2c = est.rotation();

            Eigen::Vector3d twinc(est.translation());
            Eigen::Vector3d tcinw= - qw2c.conjugate()._transformVector(twinc);
            fout<< jack<<" "<< tcinw.transpose()<< " "<<
                   qw2c.conjugate().coeffs().transpose()<<endl;
            ++jack;
        }

        fout.close();
    }
    return 0;
}
