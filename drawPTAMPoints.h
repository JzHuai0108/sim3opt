/**
 * @author  Hauke Strasdat
 *
 * Copyright (C) 2010  Hauke Strasdat
 *                     Imperial College London
 *
 * rss2010_demo.cpp is part of RobotVision.
 *
 * RobotVision is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or any later version.
 */

// JIanzhu Huai used TooN data structures to read the binary output files 
// from sequential KLT based PTAM, and the estimated poses of S_w^c, 
// reproject map points onto the corrected map
#ifndef DRAW_PTAM_POINTS_H_
#define DRAW_PTAM_POINTS_H_
#include <vector>
#include <set>
#include <list>

#include <TooN/TooN.h>
#include <TooN/SVD.h>
#include <TooN/se3.h>
#include "sim3_rv.h"

//#include "maths_utils_rv.h"
#include "DUtils.h" //for file extraction from a folder

/** point transformation function using 3D ridig transformation SE3 */
template <class A> inline TooN::Vector<3>
    transform(const TooN::SE3<A>& T, const TooN::Vector<3,A>& x)
{
  return T.get_rotation()*x + T.get_translation();
}

/** point transformation function using 3D similarity transformation
 * Sim3 */
template <class A> inline TooN::Vector<3>
    transform(const RobotVision::Sim3<A>& T, const TooN::Vector<3,A>& x)
{
  return T.get_scale()*(T.get_rotation()*x) + T.get_translation();
}


template <class A> inline TooN::Vector<4>
    transform(const TooN::SE3<A>& T, const TooN::Vector<4,A>& x)
{
  TooN::Matrix<4,4>  M= TooN::Identity;
  M.slice(0,0,3,3) = T.get_rotation().get_matrix();
  M.T()[3].slice<0,3>() = T.get_translation();
  return  M*x;
}

// load a keyframe file in binary format which is the output of my sequential PTAMM by using KLT tracker
// filename: keyFrameFile; Rw2c, twinc: rotation and translation of the world to camera(this keyframe) transform,
// obsinframe, the observations involved in this frame, each member IdObs, has frame id, point id, and a obs 2-vector
// the point id is continuous from 0, 1,2 and so on as output by PTAMM seq KLT, but frame id is discrete
// pointsinframe: all the points in world observed in this keyframe, a map stores points id and their 3d coordinates

/** observation class */
template <int ObsDim>
class IdObs{
    public:
    int frame_id;
    int point_id;
    TooN::Vector< ObsDim> obs;

  IdObs(){}
  IdObs(int point_id, int frame_id, const TooN::Vector<ObsDim> & obs)
    : frame_id(frame_id), point_id(point_id), obs(obs)
  {
  }
};

void LoadComboKeyFrame(const char *keyFrameFile, TooN::Matrix<3> &Rw2c, TooN::Vector<3>& twinc, std::vector<IdObs<2> > &obsinframe,
                       std::map<unsigned int, TooN::Vector< 3 > > &pointsinframe);


void SavePoses(std::vector<TooN::SE3< > > &posesw2c,std::vector<TooN::Vector<3> > &pointsinw, const char *file);

// load all keyframes and their map points, save them into a file which can be 
// read by matlab and draw a trajectory map superimposed by map points
// transFile each line format 0: frameId, sw2c, Rw2c(3x3 row major order), twinc of S_w^c = [sR, t]
// format 1: sim3 optimization result: kf frameid, sw2i, scaled tiinw, ri2w(qxyzw)
// S221 the similarity transform from w2 frame of these poses and points to a w1 frame
//example: figureKITTIBA("/media/jianzhuhuai0108/viola/relaylatest/kitticolorseq00img2keyframeinfo400_03",
//"transCorrPoses.txt", "LC7map.txt", RobotVision::Sim3(), 1)
void figureKITTIBA(const std::string keyFrameInfoDir, const std::string transFile,
                   const std::string outputFile,
                   const RobotVision::Sim3<> S221, int lineFormat, bool bBALFormat = true);

// draw the original map points and trajectories output by sequential PTAM based on KLT
//example: figureKITTI("/media/jianzhuhuai0108/viola/relaylatest/kitticolorseq00img2keyframeinfo400_03",
// "OriginalMap.txt", RobotVision::Sim3());
void figureKITTI(const std::string keyFrameInfoDir, const std::string outputFile, const RobotVision::Sim3<> S221, bool bBALFormat= true);


#endif
