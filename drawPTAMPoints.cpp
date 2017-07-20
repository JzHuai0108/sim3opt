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

#include "drawPTAMPoints.h"
#include <Eigen/Dense>
#include "sim3_rv.h" //from robot vision

using namespace std;
using namespace TooN;
const int NUMTRACKERCAMPARAMETERS=5;
// load a keyframe file in binary format which is the output of my sequential PTAMM by using KLT tracker
// filename: keyFrameFile; Rw2c, twinc: rotation and translation of the world to camera(this keyframe) transform,
// obsinframe, the observations involved in this frame, each member IdObs, has frame id, point id, and a obs 2-vector
// the point id is continuous from 0, 1,2 and so on as output by PTAMM seq KLT, but frame id is discrete
// pointsinframe: all the points in world observed in this keyframe, a map stores points id and their 3d coordinates


void LoadComboKeyFrame(const char *keyFrameFile, Matrix<3> &Rw2c, Vector<3>& twinc, std::vector<IdObs<2> > &obsinframe,
                       map<unsigned int, Vector< 3 > > &pointsinframe)
{
    try{
        //load keyframe camera
        unsigned int uKFID=0;
        ifstream reader(keyFrameFile, ios::in | ios::binary);
        reader.read((char*)&uKFID, sizeof (int));

        int strLength=0;
        reader.read ((char*)&strLength, sizeof (int));
        char cBuffer[200]={'\0'};
        reader.read(cBuffer, sizeof(char)*strLength);

        double v2CamSize[2];
        reader.read((char*)v2CamSize, sizeof (double)*2);

        double vCamParams2[NUMTRACKERCAMPARAMETERS];
        reader.read((char*)vCamParams2, sizeof (double)*NUMTRACKERCAMPARAMETERS);

        bool bFixed=false;
        reader.read ((char*)Rw2c.get_data_ptr(), sizeof (double)*9);
        reader.read ((char*)twinc.get_data_ptr(), sizeof (double)*3);
        reader.read ((char*)&bFixed, sizeof (bool));

        //input the size of map
        int mapSize=-1;
        reader.read ((char*)&mapSize, sizeof (int));
        obsinframe.resize(mapSize);
        pointsinframe.clear();

        unsigned int uid=0;
        Vector<3> vecpinw;
        double dCosInitAngle=-1;
        Vector<2> v2RootPos;
        for(int counter=0; counter<mapSize; ++counter){
            reader.read ((char*)&uid, sizeof (unsigned int));
            reader.read ((char*)vecpinw.get_data_ptr(), sizeof (double)*3);
            pointsinframe.insert(make_pair(uid, vecpinw));

            reader.read ((char*)(&dCosInitAngle), sizeof (double));
            reader.read ((char*)v2RootPos.get_data_ptr(), sizeof (double)*2);
            obsinframe[counter].frame_id=uKFID;
            obsinframe[counter].point_id=uid;
            obsinframe[counter].obs=v2RootPos;
        }
        reader.close();
    }
    catch (std::ifstream::failure e) {
        std::cerr << "Exception in reading keyframe file:"<<e.what()<<endl;
    }
}
void SavePoses(vector<SE3< > > &posesw2c,vector<Vector<3> > &pointsinw, const char *file)
{
    try{
        ofstream posefile(file, ios::out);
        Matrix<3> rotw2c;
        Vector<3> transcinw;
        posefile<< "% format #poses, then each line is 3x4 Pc2w row major, #points, then each line 1x3 pinw" <<endl;
        posefile<< posesw2c.size()<<endl;
        for(uint i=0; i<posesw2c.size();++i){
            rotw2c=posesw2c[i].get_rotation().get_matrix();
            transcinw=posesw2c[i].inverse().get_translation();
            posefile<< rotw2c(0,0) <<" "<<rotw2c(1,0) <<" "<<rotw2c(2,0)<<" "<<transcinw[0]<<" ";
            posefile<< rotw2c(0,1) <<" "<<rotw2c(1,1) <<" "<<rotw2c(2,1)<<" "<<transcinw[1]<<" ";
            posefile<< rotw2c(0,2) <<" "<<rotw2c(1,2) <<" "<<rotw2c(2,2)<<" "<<transcinw[2]<<endl;
        }
        posefile<< pointsinw.size()<<endl;
        for(uint i=0; i<pointsinw.size();++i)
            posefile<< pointsinw[i][0] <<" "<< pointsinw[i][1] <<" "<< pointsinw[i][2] <<endl;

        posefile.close();
    }
    catch (std::ifstream::failure e) {
        std::cerr << "Exception in writing pose results file:"<<e.what()<<endl;
    }
}
//from ceres solver rotation.h
template<typename T>
inline void QuaternionToAngleAxis(const T* quaternion, T* angle_axis) {
  const T& q1 = quaternion[1];
  const T& q2 = quaternion[2];
  const T& q3 = quaternion[3];
  const T sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;
  // For quaternions representing non-zero rotation, the conversion
  // is numerically stable.
  if (sin_squared_theta > T(0.0)) {
    const T sin_theta = sqrt(sin_squared_theta);
    const T& cos_theta = quaternion[0];
    // If cos_theta is negative, theta is greater than pi/2, which
    // means that angle for the angle_axis vector which is 2 * theta
    // would be greater than pi.
    //
    // While this will result in the correct rotation, it does not
    // result in a normalized angle-axis vector.
    //
    // In that case we observe that 2 * theta ~ 2 * theta - 2 * pi,
    // which is equivalent saying
    //
    //   theta - pi = atan(sin(theta - pi), cos(theta - pi))
    //              = atan(-sin(theta), -cos(theta))
    //
    const T two_theta =
        T(2.0) * ((cos_theta < 0.0)
                  ? atan2(-sin_theta, -cos_theta)
                  : atan2(sin_theta, cos_theta));
    const T k = two_theta / sin_theta;
    angle_axis[0] = q1 * k;
    angle_axis[1] = q2 * k;
    angle_axis[2] = q3 * k;
  } else {
    // For zero rotation, sqrt() will produce NaN in the derivative since
    // the argument is zero.  By approximating with a Taylor series,
    // and truncating at one term, the value and first derivatives will be
    // computed correctly when Jets are used.
    const T k(2.0);
    angle_axis[0] = q1 * k;
    angle_axis[1] = q2 * k;
    angle_axis[2] = q3 * k;
  }
}

//dcm (Cba) to quaternion (qba)
//Cba:B to A transformation matrix
//qba:B to A quaternion, in wxyz format

//Implementation of savage(3-47)
static TooN::Vector<4> rotro2qr(TooN::Matrix<3> dcm)
{
    double tr = dcm(0,0) + dcm(1,1) + dcm(2,2);
    double Pa = 1 + tr;
    double Pb = 1 + 2 * dcm(0,0) - tr;
    double Pc = 1 + 2 * dcm(1,1) - tr;
    double Pd = 1 + 2 * dcm(2,2) - tr;

    TooN::Vector<4> quat=TooN::Zeros;
    if (Pa >= Pb && Pa >= Pc && Pa >= Pd)
    {
        quat[0]= 0.5 * sqrt(Pa);
        quat[1]= (dcm(2,1) - dcm(1,2)) / 4 / quat[0];
        quat[2]= (dcm(0,2) - dcm(2,0)) / 4 / quat[0];
        quat[3]= (dcm(1,0) - dcm(0,1)) / 4 / quat[0];
    }
    else
    {
        if (Pb >= Pc && Pb >= Pd)
        {
            quat[1] = 0.5 * sqrt(Pb);
            quat[2] = (dcm(1,0) + dcm(0,1)) / 4 / quat[1];
            quat[3] = (dcm(0,2) + dcm(2,0)) / 4 / quat[1];
            quat[0] = (dcm(2,1) - dcm(1,2)) / 4 / quat[1];
        }
        else
        {
            if (Pc >= Pd)
            {
                quat[2]= 0.5 * sqrt(Pc);
                quat[3]= (dcm(2,1) + dcm(1,2)) / 4 / quat[2];
                quat[0]= (dcm(0,2) - dcm(2,0)) / 4 / quat[2];
                quat[1]= (dcm(1,0) + dcm(0,1)) / 4 / quat[2];
            }
            else
            {
                quat[3] = 0.5 * sqrt(Pd);
                quat[0] = (dcm(1,0) - dcm(0,1)) / 4 / quat[3];
                quat[1] = (dcm(0,2) + dcm(2,0)) / 4 / quat[3];
                quat[2] = (dcm(2,1) + dcm(1,2)) / 4 / quat[3];
            }
        }
    }

    if (quat[0] <= 0)
    {
        quat[0]=-quat[0];
        quat[1]=-quat[1];
        quat[2]=-quat[2];
        quat[3]=-quat[3];
    }
    return quat;
}
// This function writes the BAL problem to a file in the same format that
// is read by the constructor. copied from ceres solver example
// BAL assumes that both camera pose and point indices start from 0 and are continuous
// the exact BAL format assumes camera frame is right, up and backward, and the point observation has no principal component,
// Here we assume the camera frame is right, down and forward, and the point observation has principal component, i.e., +cx, +cy
void SaveBALFile(std::vector<TooN::SE3< > > &posesw2c, std::vector<TooN::Vector<3> > &pointsinw,
                 std::vector<IdObs<2>  > obs_vec, TooN::Vector<3> fk1k2, const std::string& filename){

    FILE* fptr = fopen(filename.c_str(), "w");
    if (fptr == NULL) {
        cerr << "Error: unable to open file " << filename;
        return;
    };
    int num_cameras_= posesw2c.size();
    int num_points_= pointsinw.size();
    int num_observations_= obs_vec.size();
    int maxPointId= -1;
    int minPointId= 1000;
    fprintf(fptr, "%d %d %d\n", num_cameras_, num_points_, num_observations_);
    for (int i = 0; i < num_observations_; ++i) {
        IdObs<2>& obs= obs_vec[i];
        if(maxPointId< obs.point_id)
            maxPointId= obs.point_id;
        if(minPointId > obs.point_id)
            minPointId= obs.point_id;
        fprintf(fptr, "%d %d", obs.frame_id, obs.point_id);
        for (int j = 0; j < 2; ++j) {
            fprintf(fptr, " %g", obs.obs[j]);
        }
        fprintf(fptr, "\n");
    }
    if(minPointId!=0 || maxPointId!= (int)pointsinw.size() -1 )
    {
        cerr<<"points ids should be 0 to N-1"<<endl;
        exit(1);
    }

//    Matrix<3> Rcv2gl= Data( 1, 0,0, 0, -1, 0, 0, 0, -1); // ptamm used opencv frame notation, bundler used opengl notation
    Matrix<3> Rcv2gl= Data( 1, 0,0, 0, 1, 0, 0, 0, 1);
    for (int i = 0; i < num_cameras_; ++i) {
        double angleaxis[9];
        TooN::Vector<4> qw2c= rotro2qr(Rcv2gl*posesw2c[i].get_rotation().get_matrix());
        double qw2c_data [4] ={ qw2c[0], qw2c[1], qw2c[2], qw2c[3]};
        // Output in angle-axis format.
        QuaternionToAngleAxis(qw2c_data, angleaxis);
        angleaxis [3]= posesw2c[i].get_translation()[0];

//        angleaxis [4]= - posesw2c[i].get_translation()[1];
//        angleaxis [5]= - posesw2c[i].get_translation()[2]; //bundler format used opengl frame notation

        angleaxis [4]= posesw2c[i].get_translation()[1];
        angleaxis [5]= posesw2c[i].get_translation()[2];

        angleaxis [6]= fk1k2[0];
        angleaxis [7]= fk1k2[1];
        angleaxis [8]= fk1k2[2];

        for (int j = 0; j < 9; ++j) {
            fprintf(fptr, "%.16g\n", angleaxis[j]);
        }
    }

    for (int i = 0; i < num_points_; ++i) {
        TooN::Vector<3>& point = pointsinw[i];
        const int point_block_size_=3;
        for (int j = 0; j < point_block_size_; ++j) {
            fprintf(fptr, "%.16g\n", point[j]);
        }
    }
    fclose(fptr);
}

void figureKITTIBA(const  string keyFrameInfoDir, const string transFile,
                   const string outputFile,
                   const RobotVision::Sim3<> S221, int lineFormat, bool bBALFormat)
{

    vector<SE3<> > pose_vec;
    vector<IdObs<2>  > obs_vec;
    vector<Vector<3> > point_vec;

    char buffer[300]={'\0'};

    //load all frames
    vector<string> keyFrameFiles =
            DUtils::FileFunctions::Dir(keyFrameInfoDir.c_str(), ".bin", true);
    map<unsigned int, unsigned int> frameid2poseid; //given a frame id, obtain its position in pose_vec
    vector<int> imgIds; // all IDs of keyframes
    imgIds.resize(keyFrameFiles.size());
    unsigned int count=0;
    for(; count<keyFrameFiles.size();++count)
    {
        std::size_t pos = keyFrameFiles[count].find("KeyFrame");
        int imgId=atoi(keyFrameFiles[count].substr(pos+8, 6).c_str());
        imgIds[count]=imgId;
        frameid2poseid.insert(make_pair(imgId, count));
    }
    assert(imgIds.size()==count);
    assert(frameid2poseid.size()==count);

    map<unsigned int, Vector<3> > pointsallframe; //store the final estimated point vectors from all keyframes, with their IDs
    map<unsigned int, Vector<3> > pointsinframe; //store estimated point vectors from a keyframe, with their IDs, to be merged into pointsallframe
    map<unsigned int, unsigned int> pointid2compactid; // map point id used in keyframe files to continuous compact id, from 0 up
    Matrix<3> Rw2c;
    Vector<3> twinc;
    vector<IdObs < 2 > > obsinframe; // the image observations in one frame

    for(unsigned int frameCounter=0; frameCounter<imgIds.size(); ++frameCounter)
    {

        sprintf(buffer, "%s/KeyFrame%06d.bin", keyFrameInfoDir.c_str(), imgIds[frameCounter]);
        LoadComboKeyFrame(buffer, Rw2c, twinc, obsinframe, pointsinframe);
        assert(obsinframe[0].frame_id==imgIds[frameCounter]);

        pose_vec.push_back(SE3<>(Rw2c,twinc));

        //merge pointsinframe to pointsallframe
        for(map<unsigned int, Vector<3> >::const_iterator mIter=pointsinframe.begin(); mIter!=pointsinframe.end(); ++mIter)
        {
            map<unsigned int, Vector<3> >::iterator cand=pointsallframe.find(mIter->first);
            if(cand!=pointsallframe.end()){
                Vector<3> diff=cand->second-mIter->second;
                if(norm_2(diff)>1)
                    cout<<"difference between last and present optimized point position: "<<diff[0]<<" ,"<<diff[1]<<" ,"<<diff[2]<<endl;
                cand->second=mIter->second;
            }
            else
                pointsallframe.insert(*mIter);
        }
        obs_vec.insert(obs_vec.end(), obsinframe.begin(), obsinframe.end());
    }

    //create compact point id mapping and point_vec
    point_vec.resize(pointsallframe.size());
    unsigned int pointCounter=0, lastPointId=0;
    for(map<unsigned int, Vector<3> >::const_iterator mIter=pointsallframe.begin(); mIter!=pointsallframe.end(); ++mIter, ++pointCounter)
    {
        point_vec[pointCounter]=mIter->second;
        assert(lastPointId<mIter->first||mIter->first==0);
        lastPointId=mIter->first;
        pointid2compactid.insert(make_pair(lastPointId, pointCounter));
    }
    assert(pointCounter==pointsallframe.size());
    //update frame_id and point_id in obs_vec to compact pose and point id
    for(unsigned int obsCounter=0; obsCounter<obs_vec.size(); ++obsCounter)
    {
        map<unsigned int, unsigned int>::const_iterator cIter=frameid2poseid.find(obs_vec[obsCounter].frame_id);
        assert(cIter!=frameid2poseid.end());
        obs_vec[obsCounter].frame_id=cIter->second;
        cIter=pointid2compactid.find(obs_vec[obsCounter].point_id);
        assert(cIter!=pointid2compactid.end());
        obs_vec[obsCounter].point_id=cIter->second;
    }

    //load the corrected states
    ifstream estimatedPoses(transFile.c_str(),ios::in);
    if(estimatedPoses.is_open())//remove the header lines
        for(int header=0; header<1; ++header)
            estimatedPoses.getline(buffer, 300);
    else{
        cout<<"transPoses.txt file not opened!"<<endl;
        exit(-1);
    }

    unsigned int frameId;
    vector<RobotVision::Sim3<> > updated_trans7_list;// the corrected sim3W2C
    vector<SE3<> > cor7_pose_vec;// corrected se3W2C
    while(!estimatedPoses.eof())
    {
        double sw2c;
        double resort[12];
        estimatedPoses.getline(buffer, 300);
        if(estimatedPoses.eof())
            break;
        if(lineFormat==0){
            sscanf(buffer, "%u %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &frameId, &sw2c, &resort[0], &resort[1], &resort[2],
                    &resort[3], &resort[4], &resort[5], &resort[6], &resort[7], &resort[8],
                    &resort[9], &resort[10], &resort[11]);

            Rw2c=Data(resort[0], resort[1], resort[2], resort[3], resort[4], resort[5], resort[6], resort[7], resort[8]);
            twinc=makeVector(resort[9], resort[10], resort[11]);
        }
        else if(lineFormat==1)
        {
            sscanf(buffer, "%u %lf %lf %lf %lf %lf %lf %lf %lf",
                   &frameId, &sw2c, &resort[0], &resort[1], &resort[2],
                    &resort[3], &resort[4], &resort[5], &resort[6]);
            Eigen::Quaterniond qc2w(resort[6], resort[3], resort[4], resort[5]);
            Eigen::Matrix3d Rc2w= qc2w.toRotationMatrix();
            Rw2c=Data(Rc2w(0,0), Rc2w(1,0) , Rc2w(2,0),
                      Rc2w(0,1), Rc2w(1,1) , Rc2w(2,1),
                      Rc2w(0,2), Rc2w(1,2) , Rc2w(2,2));
            twinc= - sw2c*Rw2c*makeVector(resort[0], resort[1], resort[2]);
        }
        else
            assert(false);
        updated_trans7_list.push_back(RobotVision::Sim3<>(Rw2c,twinc, sw2c));
        cor7_pose_vec.push_back(SE3<>(Rw2c,twinc/sw2c));
    }
    estimatedPoses.close();
    assert(updated_trans7_list.size()==frameid2poseid.size());
    //apply corrections to all points, not observations
    vector<Vector<3> > cor7_point_vec;

    // map points into updated frames
    cor7_point_vec = point_vec;
    for (uint i=0; i<obs_vec.size(); ++i)
    {
        int frame_id = obs_vec[i].frame_id;
        int point_id = obs_vec[i].point_id;

        Vector<3> rel_point = transform(pose_vec[frame_id], point_vec[point_id]);
        Vector<3> cor_point
                = transform(updated_trans7_list[frame_id].inverse(),rel_point);
        cor7_point_vec[point_id] = cor_point;//many previous projection may be crushed
    }
    // apply the similarity transform to align with the ground truth
    RobotVision::Sim3<> S122 = S221.inverse();
    for(size_t jack=0; jack< cor7_pose_vec.size(); ++jack)
    {
        RobotVision::Sim3<> Sw12c= RobotVision::Sim3<>(cor7_pose_vec[jack].get_rotation(), cor7_pose_vec[jack].get_translation(), 1.0) * S122;
        cor7_pose_vec[jack]= SE3<>(Sw12c.get_rotation(), Sw12c.get_translation()/ Sw12c.get_scale());
    }
    for(size_t jack=0; jack< cor7_point_vec.size(); ++jack)
    {
        cor7_point_vec[jack] = transform(S221, cor7_point_vec[jack]);
    }
    if(bBALFormat)
    {
        TooN::Vector<3> fk1k2= makeVector(718.856,0,0);
//        TooN::Vector<2> cxy= makeVector(607.1928, 185.2157);
//        for (uint i=0; i<obs_vec.size(); ++i)// remove principal points in bundler format
//        {
//            obs_vec[i].obs -= cxy;
//            obs_vec[i].obs[1] = - obs_vec[i].obs[1];
//        }
        SaveBALFile(cor7_pose_vec, cor7_point_vec, obs_vec, fk1k2,  outputFile.c_str());
    }
    else
        SavePoses(cor7_pose_vec, cor7_point_vec, outputFile.c_str());

    std::cout <<"saved output file "<< outputFile <<std::endl;
}

void figureKITTI(const string keyFrameInfoDir, const string outputFile, const RobotVision::Sim3<> S221, bool bBALFormat)
{
    vector<SE3<> > pose_vec;
    vector<RobotVision::Sim3<> > sim_vec;
    vector<IdObs<2>  > obs_vec;
    vector<Vector<3> > point_vec;

    char buffer[300]={'\0'};

    //load all frames
    vector<string> keyFrameFiles =
            DUtils::FileFunctions::Dir(keyFrameInfoDir.c_str(), ".bin", true);
    map<unsigned int, unsigned int> frameid2poseid; //given a frame id, obtain its position in pose_vec, or sim_vec
    vector<int> imgIds; // all IDs of keyframes
    imgIds.resize(keyFrameFiles.size());
    unsigned int count=0;
    for(; count<keyFrameFiles.size();++count)
    {
        std::size_t pos = keyFrameFiles[count].find("KeyFrame");
        int imgId=atoi(keyFrameFiles[count].substr(pos+8, 6).c_str());

        imgIds[count]=imgId;
        frameid2poseid.insert(make_pair(imgId, count));
    }
    assert(imgIds.size()==count);
    assert(frameid2poseid.size()==count);

    map<unsigned int, Vector<3> > pointsallframe; //store the final estimated point vectors from all keyframes, with their IDs
    map<unsigned int, Vector<3> > pointsinframe; //store estimated point vectors from a keyframe, with their IDs, to be merged into pointsallframe
    map<unsigned int, unsigned int> pointid2compactid; // map point id used in keyframe files to continuous compact id, from 0 up
    Matrix<3> Rw2c;
    Vector<3> twinc;

    vector<IdObs < 2 > > obsinframe; // the image observations in one frame

    for(unsigned int frameCounter=0; frameCounter<imgIds.size(); ++frameCounter)
    {
        sprintf(buffer, "%s/KeyFrame%06d.bin", keyFrameInfoDir.c_str(), imgIds[frameCounter]);
        LoadComboKeyFrame(buffer, Rw2c, twinc, obsinframe, pointsinframe);
        assert(obsinframe[0].frame_id==imgIds[frameCounter]);
        pose_vec.push_back(SE3<>(Rw2c,twinc));
        //double sca=GetMedianDepth(*pose_vec.rbegin(), pointsinframe);
        sim_vec.push_back(RobotVision::Sim3<>(Rw2c,twinc,1.0));

        for(map<unsigned int, Vector<3> >::const_iterator mIter=pointsinframe.begin(); mIter!=pointsinframe.end(); ++mIter)
        {
            map<unsigned int, Vector<3> >::iterator cand=pointsallframe.find(mIter->first);
            if(cand!=pointsallframe.end()){
                Vector<3> diff=cand->second-mIter->second;
                if(norm_2(diff)>1e-1)
                    cout<<"difference between last and present optimized point position: "<<diff[0]<<" ,"<<diff[1]<<" ,"<<diff[2]<<endl;
                cand->second=mIter->second;
            }
            else
                pointsallframe.insert(*mIter);
        }
        obs_vec.insert(obs_vec.end(), obsinframe.begin(), obsinframe.end());
    }

    //create compact point id mapping and point_vec
    point_vec.resize(pointsallframe.size());
    unsigned int pointCounter=0, lastPointId=0;
    for(map<unsigned int, Vector<3> >::const_iterator mIter=pointsallframe.begin(); mIter!=pointsallframe.end(); ++mIter, ++pointCounter)
    {
        point_vec[pointCounter]=mIter->second;
        assert(lastPointId<mIter->first||mIter->first==0);
        lastPointId=mIter->first;
        pointid2compactid.insert(make_pair(lastPointId, pointCounter));
    }
    assert(pointCounter==pointsallframe.size());
    //update frame_id and point_id in obs_vec to compact pose and point id
    for(unsigned int obsCounter=0; obsCounter<obs_vec.size(); ++obsCounter)
    {
        map<unsigned int, unsigned int>::const_iterator cIter=frameid2poseid.find(obs_vec[obsCounter].frame_id);
        assert(cIter!=frameid2poseid.end());
        obs_vec[obsCounter].frame_id=cIter->second;
        cIter=pointid2compactid.find(obs_vec[obsCounter].point_id);
        assert(cIter!=pointid2compactid.end());
        obs_vec[obsCounter].point_id=cIter->second;
    }
    // apply the similarity transform to align with the ground truth
    RobotVision::Sim3<> S122 = S221.inverse();
    for(size_t jack=0; jack< pose_vec.size(); ++jack)
    {
        RobotVision::Sim3<> Sw12c= RobotVision::Sim3<>(pose_vec[jack].get_rotation(), pose_vec[jack].get_translation(), 1.0) * S122;
        pose_vec[jack]= SE3<>(Sw12c.get_rotation(), Sw12c.get_translation()/ Sw12c.get_scale());
    }
    for(size_t jack=0; jack< point_vec.size(); ++jack)
    {
        point_vec[jack] = transform(S221, point_vec[jack]);
    }
    if(bBALFormat)
    {
        TooN::Vector<3> fk1k2= makeVector(718.856,0,0);
//        TooN::Vector<2> cxy= makeVector(607.1928, 185.2157);
//        for (uint i=0; i<obs_vec.size(); ++i)// remove principal points in bundler format
//        {
//            obs_vec[i].obs -= cxy;
//            obs_vec[i].obs[1] = - obs_vec[i].obs[1];
//        }
        SaveBALFile(pose_vec, point_vec, obs_vec, fk1k2, outputFile.c_str());
    }
    else
        SavePoses(pose_vec, point_vec, outputFile.c_str());

    std::cout <<"saved output file "<< outputFile <<std::endl;
}
