/**
 * File: demoDetector.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DLoopDetector
 */

#ifndef __DEMO_DETECTOR__
#define __DEMO_DETECTOR__

#include <iostream>
#include <vector>
#include <set>
#include <string>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/opencv.hpp>
// #include <opencv2/core/eigen.hpp> //for cv2eigen()
// DLoopDetector and DBoW2
#include "DBoW2/DBoW2.h"
#include "DBoW2/FSurf64.h"
#include "DLoopDetector.h"
#include "DUtils/DUtils.h"
#include "DUtilsCV/DUtilsCV.h"
#include "DVision/DVision.h"

#ifdef USE_KLEIN
//for Klein's 5 point algorithm implementation
#include "SmallMatrixOpts.h"
#include "HomographyInit.h"
#include "ATANCamera.h"
using namespace PTAMM;
#else
#include "TooN/TooN.h"
const int NUMTRACKERCAMPARAMETERS=5;
#endif
#ifdef USE_SNAVELY
//for snavely's 5 point implementation
#include "5point.h" //for snavely's 5 point
#include "vector.h" //for snavely's algorithm
#endif

//for g2o 2 view bundle adjustment
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/structure_only/structure_only_solver.h>


using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;
using namespace cv;

using namespace Eigen;

#define USE_KNN_MATCH 0
#define SCHUR_TRICK 1
typedef g2o::EdgeProjectXYZ2UV g2oEdgeSE3;
typedef g2o::VertexSE3Expmap g2oFrameSE3;
typedef g2o::VertexSBAPointXYZ g2oPoint;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

/// Generic class to create functors to extract features
template<class TDescriptor>
class FeatureExtractor
{
public:
    /**
   * Extracts features
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
    virtual void operator()(const cv::Mat &im,
                            vector<cv::KeyPoint> &keys, vector<TDescriptor> &descriptors) const = 0;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/// This functor extracts SURF64 descriptors in the required format
class SurfExtractor: public FeatureExtractor<DBoW2::FSurf64::TDescriptor>
{
public:
    /**
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
    virtual void operator()(const cv::Mat &im,
                            vector<cv::KeyPoint> &keys, vector<vector<float> > &descriptors) const;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

/// @param TVocabulary vocabulary class (e.g: Surf64Vocabulary)
/// @param TDetector detector class (e.g: Surf64LoopDetector)
/// @param TDescriptor descriptor class (e.g: vector<float> for SURF)
template<class TVocabulary, class TDetector, class TDescriptor>
/// Class to run the demo 
class demoDetector
{
public:

    /**
   * @param vocfile vocabulary file to load
   * @param imagedir directory to read images from
   * @param posefile pose file
   * @param width image width
   * @param height image height
   */
    demoDetector(const std::string &vocfile, const std::string &imagedir,
                 const std::string &posefile, int width, int height);
    
    ~demoDetector(){}

    /**
   * Runs the demo
   * @param name demo name
   * @param extractor functor to extract features
   */
    void run(const std::string &name,
             const FeatureExtractor<TDescriptor> &extractor);
    void saveLoops(const string filename) const;
    void loadLoops(const string filename);
    void cleanLoops();
    void computeConstraints( std::vector< DetectionResult >& listPairs, const string constraintFile,
                             const cv::Mat cameraMat,
                             const string coordsFile="", const string tracksFile="") const;
protected:

    /**
   * Reads the robot poses from a file
   * @param filename file
   * @param xs
   * @param ys
   */
    void readPoseFile(const char *filename, std::vector<double> &xs,
                      std::vector<double> &ys, std::vector<int>& lineNo) const;

protected:

    std::string m_vocfile;
    std::string m_imagedir;
    std::string m_posefile;
    int m_width;
    int m_height;
public:
    std::vector<DetectionResult> vDetectionResults; //list of detected matches
    string keyFrameInfoDir; //the directory of all keyframe info files
};
/** observation class */
class IdPair{

public:

    int frame_id;
    int point_id;
    IdPair(){}
    IdPair(int point_id, int frame_id)
        : frame_id(frame_id), point_id(point_id)
    {
    }
    IdPair(const IdPair& rhs):frame_id(rhs.frame_id), point_id(rhs.point_id){}
    bool operator==(const IdPair& rhs) const
    {
        return (frame_id==rhs.frame_id)&&(point_id==rhs.point_id);
    }
    bool operator<(const IdPair& rhs) const
    {
        if(frame_id==rhs.frame_id)
            return point_id<rhs.point_id;
        else
            return frame_id<rhs.frame_id;
    }
    IdPair& operator=(const IdPair&rhs){
        frame_id=rhs.frame_id;
        point_id=rhs.point_id;
        return *this;
    }
};
typedef std::vector<IdPair >
_Track;
typedef std::map<int,_Track >
_TrackMap;

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
demoDetector<TVocabulary, TDetector, TDescriptor>::demoDetector
(const std::string &vocfile, const std::string &imagedir,
 const std::string &posefile, int width, int height)
    : m_vocfile(vocfile), m_imagedir(imagedir), m_posefile(posefile),
      m_width(width), m_height(height)
{
}
// convert DCM matrix to euler angles
// note this function may have ambiguity over special DCMs, see Blanco, SE(3) tutorial for a robust version
void DCM2Euler(cv::Mat& dcm, double eul[]){
    double pitch=asin(-dcm.at<double>(2,0));  //pitch is assumed to be [-pi pi]. singular at pi. use ad-hoc methods to remedy this deficiency
    double roll=atan2(dcm.at<double>(2,1), dcm.at<double>(2,2));
    double heading = atan2(dcm.at<double>(1,0), dcm.at<double>(0,0));
    eul[0]=roll;
    eul[1]=pitch;
    eul[2]=heading;
}
void GetIntrinsics(const cv::Mat &cameraMatrix, double * K){
    K[0] = cameraMatrix.at<double>(0,0); K[1] = cameraMatrix.at<double>(0,1);     K[2] = cameraMatrix.at<double>(0,2);
    K[3] = cameraMatrix.at<double>(1,0);     K[4] = cameraMatrix.at<double>(1,1); K[5] = cameraMatrix.at<double>(1,2);
    K[6] = cameraMatrix.at<double>(2,0);     K[7] = cameraMatrix.at<double>(2,1); K[8] = cameraMatrix.at<double>(2,2);
}

//eul defined in "n": rotate "n" to obtain "b"
//result: Cbn (from b to n)
Eigen::Matrix3d roteu2ro(Eigen::Vector3d eul)
{
    double cr = cos(eul[0]); double sr = sin(eul[0]);	//roll
    double cp = cos(eul[1]); double sp = sin(eul[1]);	//pitch
    double ch = cos(eul[2]); double sh = sin(eul[2]);	//heading
    Eigen::Matrix3d dcm;
    dcm(0,0) = cp * ch;
    dcm(0,1) = (sp * sr * ch) - (cr * sh);
    dcm(0,2) = (cr * sp * ch) + (sh * sr);

    dcm(1,0) = cp * sh;
    dcm(1,1) = (sr * sp * sh) + (cr * ch);
    dcm(1,2) = (cr * sp * sh) - (sr * ch);

    dcm(2,0) = -sp;
    dcm(2,1) = sr * cp;
    dcm(2,2) = cr * cp;
    return dcm;
}

static TooN::Matrix<3> roteu2ro(TooN::Vector<3> eul)
{
    double cr = cos(eul[0]); double sr = sin(eul[0]);	//roll
    double cp = cos(eul[1]); double sp = sin(eul[1]);	//pitch
    double ch = cos(eul[2]); double sh = sin(eul[2]);	//heading
    TooN::Matrix<3> dcm=TooN::Zeros;
    dcm(0,0) = cp * ch;
    dcm(0,1) = (sp * sr * ch) - (cr * sh);
    dcm(0,2) = (cr * sp * ch) + (sh * sr);

    dcm(1,0) = cp * sh;
    dcm(1,1) = (sr * sp * sh) + (cr * ch);
    dcm(1,2) = (cr * sp * sh) - (sr * ch);

    dcm(2,0) = -sp;
    dcm(2,1) = sr * cp;
    dcm(2,2) = cr * cp;
    return dcm;
}
static TooN::Vector<3> rotro2eu(TooN::Matrix<3> R)
{
    TooN::Vector<3> euler = TooN::Zeros;
    euler[0] = atan2(R(2,1), R(2,2));
    euler[1] = -(atan2(R(2,0),  sqrt(1 - R(2,0) * R(2,0))));
    euler[2] = atan2(R(1,0), R(0,0));
    return euler;
}

// extract camera extrinsics from essential matrix which is estimated with matched points, points1 and points2
// points1 observed by camera 1, points2 by camera 2,
// output, Rc12c2 is the rotation from camera 1 frame to camera 2 frame,
// output, Tc1inc2 is the normalized coordinates of the origin of camera 1 frame in camera 2 frame,
// A is camera matrix K=[fx, 0, cx; 0, fy, cy, 0, 0, 1] in pixel units
// one drawback of this function is that it doesnot consider the camera distortion
void ExtractCameras(vector<Point2f>& points1, vector<Point2f>& points2, Mat &Rc12c2, Mat& tc1inc2, const Mat & A){

    Mat mask;//(points1.size(),1, CV_8U,0);
    Mat E;
    double focal=A.at<double>(0,0);
    Point2d pp(A.at<double>(0,2), A.at<double>(1,2));

    if (0) {
        cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3, 0.99, mask);
        E = A.t() * F * A;
    } else {
        E = cv::findEssentialMat(points1, points2, focal, pp, RANSAC, 0.999, 1.0, mask);
    }
    cv::recoverPose(E, points1, points2, Rc12c2, tc1inc2, focal, pp, mask);
}

// load a keyframe file in binary format which is the output of my sequential PTAMM by using KLT tracker
// filename: keyFrameFile; Pw2c consists of Rw2c, twinc: rotation and translation of the world to camera(this keyframe) transform,
// obsinframe, the observations involved in this frame, pointsinframe: all the points in world observed in this keyframe
// so each image observation corresponds to a 3d point in this frame
void LoadComboKeyFrame(const char *keyFrameFile, Mat & Pw2c, vector<cv::Point2f> &obsinc,
                       vector<cv::Point3f> &ptsinc)
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
        double vecRw2c[9];
        double vecTwinc[3];
        bool bFixed=false;
        reader.read ((char*)vecRw2c, sizeof (double)*9);
        reader.read ((char*)vecTwinc, sizeof (double)*3);
        reader.read ((char*)&bFixed, sizeof (bool));
        Mat Rw2c(3,3,CV_64F, vecRw2c, Mat::AUTO_STEP);
        Mat Twinc(3,1,CV_64F, vecTwinc, Mat::AUTO_STEP);
        Pw2c=Mat::eye(4,4, CV_64F);
        cv::Mat tmp = Pw2c(cv::Rect(0,0,3,3));
        Rw2c.copyTo(tmp);
        tmp = Pw2c(cv::Rect(3,0,1,3));
        Twinc.copyTo(tmp);

        //input the size of map
        int mapSize=-1;
        reader.read ((char*)&mapSize, sizeof (int));
        obsinc.resize(mapSize);
        ptsinc.resize(mapSize);
        unsigned int uid=0;
        double vecpinw[3];
        Mat pinw(3,1, CV_64F, vecpinw);
        Mat pinc(3,1, CV_64F, 0);
        double dCosInitAngle=-1;
        double v2RootPos[2];
        for(int counter=0; counter<mapSize; ++counter){
            reader.read ((char*)&uid, sizeof (unsigned int));
            reader.read ((char*)vecpinw, sizeof (double)*3);
            pinc=Rw2c*pinw+Twinc;

            ptsinc[counter].x=(float)pinc.at<double>(0);
            ptsinc[counter].y=(float)pinc.at<double>(1);
            ptsinc[counter].z=(float)pinc.at<double>(2);

            reader.read ((char*)(&dCosInitAngle), sizeof (double));
            reader.read ((char*)v2RootPos, sizeof (double)*2);
            obsinc[counter].x= (float)v2RootPos[0];
            obsinc[counter].y= (float)v2RootPos[1];
        }
        reader.close();
    }
    catch (std::ifstream::failure e) {
        std::cerr << "Exception in reading keyframe file:"<<e.what()<<endl;
    }
}

struct SfMKey{
    unsigned int index; //ID of the key within an image, starting from 0
    float x, y; //image measurement in pixels
    unsigned int color[3]; // RGB in order 0,1,2
    SfMKey(unsigned int id=0, float u=0.f, float v=0.f, unsigned int R=0,
           unsigned int G=0, unsigned int B=0):index(id), x(u), y(v)
    {
        color[0]=R;
        color[1]=G;
        color[2]=B;
    }
    SfMKey(const SfMKey& rhs):index(rhs.index), x(rhs.x), y(rhs.y)
    {
        color[0]=rhs.color[0];
        color[1]=rhs.color[1];
        color[2]=rhs.color[2];
    }
    bool operator==(const SfMKey&rhs) const
    {
        return (index==rhs.index&&x==rhs.x&&y==rhs.y);
    }
    bool operator!=(const SfMKey&rhs) const
    {
        return (index!=rhs.index || x!=rhs.x || y!=rhs.y);
    }

    SfMKey& operator=(const SfMKey&rhs){
        if(this==&rhs)
            return *this;
        index=rhs.index;
        x=rhs.x;
        y=rhs.y;

        color[0]=rhs.color[0];
        color[1]=rhs.color[1];
        color[2]=rhs.color[2];
        return *this;
    }
};
/** Helper class for pose-graph optimisation  (see below)*/
template <typename Trans, int TransDoF> class Constraint
{
public:
    Constraint(int trans_id1,
               int trans_id2,
               const Trans & mean,
               const TooN::Matrix<TransDoF,TransDoF> & fisher_information)
        : trans_id1(trans_id1),
          trans_id2(trans_id2),
          mean(mean),
          fisher_information(fisher_information)
    {}

    int trans_id1;// first
    int trans_id2;// second
    Trans mean;// sim3 or se3 first 2sec, scale sec/first
    TooN::Matrix<TransDoF,TransDoF> fisher_information;
};

//a struct to feed ORB SLAM functions
struct KeyFrame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int mnId; // index of kf in image sequence, 0 based
    int mnFrameId; // index of kf in all kfs, 0 based
    Sophus::SE3d Tw2c;
    KeyFrame(int nid=-1, int kfid=-1): mnId(nid), mnFrameId(kfid){}
    bool isBad(){return false;}
    Eigen::Matrix3d GetRotation(){return Tw2c.rotationMatrix();}
    Eigen::Vector3d GetTranslation(){return Tw2c.translation();}
    void SetPose(Sophus::SE3d _Tw2c){Tw2c= _Tw2c;}
};

//
class MyKeyFrame{
public:
    unsigned int imgID;
    unsigned int increment; // used for a KeyFrame involved in loop closure, its feature index
    // need to increment by step when integrating tracks from loop closure and the original tracks
    vector<SfMKey> keys;// note it holds that keys[i].index==i
    static string m_imagedir;
    static double m_px, m_py, m_f;
    MyKeyFrame(unsigned int id=0):imgID(id), increment(0){}
    MyKeyFrame(const MyKeyFrame& rhs):imgID(rhs.imgID), increment(rhs.increment), keys(rhs.keys)
    {
    }
    void Combine(const MyKeyFrame&rhs)
    {
        const int step=keys.size();
        keys.insert(keys.end(), rhs.keys.begin(), rhs.keys.end());
        for(unsigned int count=step; count<keys.size();++count)
        {
            keys[count].index+=step;
        }
        assert(keys.size()==keys.rbegin()->index+1);
    }
    bool operator==(const MyKeyFrame&rhs) const
    {
        if(imgID==rhs.imgID&&keys.size()==rhs.keys.size()){
            for(unsigned int count=0; count<keys.size(); ++count)
                if(keys[count]!=rhs.keys[count])
                    return false;
            return true;
        }
        return false;
    }
    friend ofstream& operator<<(ofstream &coords, const MyKeyFrame& myFrame){
        char buffer[256]={'\0'};
        sprintf(buffer, "%s/%06d.png", m_imagedir.c_str(),myFrame.imgID);
        coords<<"#index = "<<myFrame.imgID<<", name = "<<buffer<<", keys = "<<myFrame.keys.size()<<
                ", px = "<<std::setprecision(4) <<m_px<<", py = "<<
                std::setprecision(4) <<m_py<<", focal = "<<
                std::setprecision(4) <<m_f<<endl;
        int dummy=0;
        for(unsigned int charade=0; charade<myFrame.keys.size(); ++charade){
            assert(charade==myFrame.keys[charade].index);
            coords<<charade; //key number
            coords<<std::fixed;
            coords<<setprecision(6);
            coords<<" "<<myFrame.keys[charade].x<<" "<<myFrame.keys[charade].y;
            coords<<" "<<dummy<<" "<<dummy;
            coords<<" "<<myFrame.keys[charade].color[0]<<" "<<myFrame.keys[charade].color[1]<<" "<<myFrame.keys[charade].color[2]<<endl;
        }
        return coords;
    }
    //note this function does not match operator<< well, because it would be too complicated
    friend ifstream& operator>>(ifstream &coords, MyKeyFrame& myFrame){
        unsigned int keyNo=0;

        if(!(coords>>myFrame.imgID)){
            cerr<<"End of loopCoords.txt reached!"<<endl;
            return coords;
        }
        myFrame.keys.clear();
        coords>>keyNo;
        int dummy[2]={0};
        SfMKey pier(0);
        for(unsigned int charade=0; charade<keyNo; ++charade){
            coords>>pier.index; //key number
            assert(pier.index==charade);
            coords>>pier.x>>pier.y>>dummy[0]>>dummy[1];
            coords>>pier.color[0]>>pier.color[0]>>pier.color[0];
            myFrame.keys.push_back(pier);
        }
        return coords;
    }

    MyKeyFrame& operator=(const MyKeyFrame&rhs)
    {
        if(this!=&rhs){
            imgID=rhs.imgID; increment=rhs.increment; keys=rhs.keys;
            return *this;
        }
        else
            return *this;
    }

};


// load a keyframe file in binary format which is the output of my sequential PTAMM by using KLT tracker
// filename: keyFrameFile, Pw2c, the world to camera(this keyframe) transform,
// trackMap: a mapping from track id to list of observations, each observation has an image id and feature id w.r.t the image
// myFrame stores image observations
void LoadComboKeyFrame(const char * keyFrameFile, TooN::Matrix<3>& Rw2c, TooN::Vector<3>& twinc,
                       _TrackMap & trackMap, MyKeyFrame& myFrame)
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

        myFrame.imgID=uKFID;
        myFrame.keys.resize(mapSize);

        _TrackMap::iterator probe;
        unsigned int uid=0;
        double vecpinw[3];
        double dCosInitAngle=-1;
        double v2RootPos[2];
        for(int counter=0; counter<mapSize; ++counter){
            reader.read ((char*)&uid, sizeof (unsigned int));
            reader.read ((char*)vecpinw, sizeof (double)*3);

            reader.read ((char*)(&dCosInitAngle), sizeof (double));
            reader.read ((char*)v2RootPos, sizeof (double)*2);
            probe=trackMap.find(uid);
            if(probe==trackMap.end())
            {
                _Track scion;
                scion.push_back(IdPair(counter, uKFID));
                trackMap.insert(make_pair(uid, scion));
            }
            else{
                probe->second.push_back(IdPair(counter, uKFID));
            }
            myFrame.keys[counter].index=counter;
            myFrame.keys[counter].x=(float)v2RootPos[0];
            myFrame.keys[counter].y=(float)v2RootPos[1];
        }
        reader.close();
    }
    catch (std::ifstream::failure e) {
        std::cerr << "Exception in reading keyframe file:"<<e.what()<<endl;
    }
}
Mat Transform(Mat Pw2f, Mat ptsinw)
{
    Mat R=Pw2f(cv::Rect(0,0,3,3));
    Mat t=Pw2f(cv::Rect(3,0,1,3));
    return R*ptsinw+t;
}
//read in the ground truth camera pose, Pw2c from poseFile, lineNo, 0 based
// note in KITTI, the ground truth is Pc2w, we need to invert it
void ReadCameraPose(const char* poseFile, int lineNo, cv::Mat& Pw2c)
{
    Pw2c=Mat::eye(4,4, CV_64F);
    fstream f(poseFile, ios::in);
    string s;
    double resort[12]={0};
    int lineCounter=0;
    while(!f.eof())
    {
        while(lineCounter<=lineNo){
            getline(f, s);
            ++lineCounter;
        }
        if(!f.eof() && !s.empty())
        {
            sscanf(s.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &resort[0], &resort[1], &resort[2], &resort[3],
                    &resort[4], &resort[5], &resort[6], &resort[7],
                    &resort[8], &resort[9], &resort[10], &resort[11]);
            Pw2c.at<double>(0,0)=resort[0]; Pw2c.at<double>(0,1)=resort[1]; Pw2c.at<double>(0,2)=resort[2]; Pw2c.at<double>(0,3)=resort[3];
            Pw2c.at<double>(1,0)=resort[4]; Pw2c.at<double>(1,1)=resort[5]; Pw2c.at<double>(1,2)=resort[6]; Pw2c.at<double>(1,3)=resort[7];
            Pw2c.at<double>(2,0)=resort[8]; Pw2c.at<double>(2,1)=resort[9]; Pw2c.at<double>(2,2)=resort[10]; Pw2c.at<double>(2,3)=resort[11];
            Pw2c.at<double>(3,0)=0; Pw2c.at<double>(3,1)=0; Pw2c.at<double>(3,2)=0; Pw2c.at<double>(3,3)=1;
            Pw2c=Pw2c.inv();
            return;
        }
    }
    f.close();
}
// ---------------------------------------------------------------------------
template<class TVocabulary, class TDetector, class TDescriptor>
void demoDetector<TVocabulary, TDetector, TDescriptor>::
saveLoops(const string filename) const
{
    ofstream loopPairFile((keyFrameInfoDir + '/' + filename).c_str(), ios::out);
    for(unsigned int count=0; count<vDetectionResults.size();++count)
    {
        int firstFrame=min(vDetectionResults[count].query, vDetectionResults[count].match);// depth info from early frames is more reliable
        int secondFrame=max(vDetectionResults[count].query, vDetectionResults[count].match);
        loopPairFile<< firstFrame<<"\t"<< secondFrame<<endl;
    }
    loopPairFile.close();
}
// ---------------------------------------------------------------------------
template<class TVocabulary, class TDetector, class TDescriptor>
void demoDetector<TVocabulary, TDetector, TDescriptor>::
loadLoops(const string filename)
{
    ifstream loopPairFile((keyFrameInfoDir + '/' + filename).c_str(), ios::in);
    vDetectionResults.clear();
    int firstFrame, secondFrame;
    while(loopPairFile>>firstFrame )
    {
        loopPairFile>>secondFrame;
        vDetectionResults.push_back(DetectionResult(firstFrame, secondFrame));
    }
    loopPairFile.close();
}

// save trackMap(track_id --> list of observations, each observation consists of image id (often 0 based) and feature id within this image, 0 based)
ofstream& operator<<(ofstream&tracks, const _TrackMap& trackMap)
{    
    //unsigned int sz=trackMap.size();
    //tracks<<sz<<endl; //saved outside this function
    for(_TrackMap::const_iterator itMap=trackMap.begin(); itMap!=trackMap.end();++itMap){
        unsigned int N=itMap->second.size();
        tracks<<N;
        for(_Track::const_iterator itTrack=itMap->second.begin(); itTrack!=itMap->second.end(); ++itTrack)
        {
            tracks<<" "<<itTrack->frame_id<<" "<<itTrack->point_id;
        }
        tracks<<endl;
    }
    return tracks;
}
//read in the just saved trackMap from ifstream
ifstream& operator>>(ifstream&tracks, _TrackMap& trackMap)
{
    unsigned int sz=trackMap.size();
    assert(sz==0);
    unsigned int N=0;
    IdPair truck;
    while(tracks>>N){
        _Track tempTrack;
        for(unsigned int obs=0; obs<N;++obs)
        {
            tracks>>truck.frame_id>>truck.point_id;
            tempTrack.push_back(truck);
        }
        trackMap.insert(make_pair(sz++, tempTrack));
    }
    assert(tracks.eof());
    return tracks;
}

struct OptParams
{
    OptParams(int num_iters,
              bool use_robust_kernel=false,
              double huber_kernel_width=1)
        : num_iters(num_iters),
          use_robust_kernel(use_robust_kernel),
          huber_kernel_width(huber_kernel_width)
    {
    }

    int num_iters;
    bool use_robust_kernel;
    double huber_kernel_width;
};


// with reference to SVO: semi-direct visual odometry
void setupG2o(g2o::CameraParameters *cam_params, g2o::SparseOptimizer * optimizer)
{
    optimizer->setVerbose(true);

#if SCHUR_TRICK
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver =
            g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType> >();

    g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
#else
    std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver =
            g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType> >();

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver)));
#endif

    solver->setMaxTrialsAfterFailure(5);
    optimizer->setAlgorithm(solver);
    // setup camera
    if (!optimizer->addParameter(cam_params)) {
        assert(false);
    }
}

g2oPoint*
createG2oPoint(Eigen::Vector3d pos,
               size_t id,
               bool fixed)
{
    g2oPoint* v = new g2oPoint();
    v->setId(id);
#if SCHUR_TRICK
    v->setMarginalized(true);
#endif
    v->setFixed(fixed);
    v->setEstimate(pos);
    return v;
}

g2oEdgeSE3*
createG2oEdgeSE3( g2oFrameSE3* v_frame,
                  g2oPoint* v_point,
                  const Vector2d& f_up,
                  double huber_width,
                  double weight=1)
{
    g2oEdgeSE3* e = new g2oEdgeSE3();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_point));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_frame));
    e->setMeasurement(f_up);
    e->information() = weight * Eigen::Matrix2d::Identity();
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();      // TODO: memory leak
    rk->setDelta(huber_width);
    e->setRobustKernel(rk);
    e->setParameterId(0, 0); //old: e->setId(v_point->id());
    return e;
}

void
runSparseBAOptimizer(g2o::SparseOptimizer* optimizer,
                     unsigned int num_iter,
                     double& init_error, double& final_error)
{
    optimizer->initializeOptimization();

    double static lambda = 50.;
    g2o::OptimizationAlgorithmLevenberg * lm
            = static_cast<g2o::OptimizationAlgorithmLevenberg *>(optimizer->solver());
    lm->setUserLambdaInit(lambda);

    optimizer->computeActiveErrors();
    init_error = optimizer->activeChi2();
    optimizer->optimize(num_iter);
    final_error = optimizer->activeChi2();
}
template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
void eigen2cv( const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src, Mat& dst )
{
    if( !(src.Flags & Eigen::RowMajorBit) )
    {
        Mat _src(src.cols(), src.rows(), DataType<_Tp>::type,
                 (void*)src.data(), src.stride()*sizeof(_Tp));
        transpose(_src, dst);
    }
    else
    {
        Mat _src(src.rows(), src.cols(), DataType<_Tp>::type,
                 (void*)src.data(), src.stride()*sizeof(_Tp));
        _src.copyTo(dst);
    }
}

template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& dst )
{
    CV_DbgAssert(src.rows == _rows && src.cols == _cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        Mat _dst(src.cols, src.rows, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else if( src.cols == src.rows )
        {
            src.convertTo(_dst, _dst.type());
            transpose(_dst, _dst);
        }
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    }
    else
    {
        Mat _dst(src.rows, src.cols, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    }
}


// bundle adjustment refinement for two views, we call them first frame and second frame
// pointsXYZ are the points coordinates in the first frame,
// points1 are the point observations in the first frame, size(pointsXYZ)==size(pointsInImage1)
// points2 are the point observations in the second frame, size(pointsXYZ)==size(pointsInImage2)
// K is the camera intrinsic matrix [fx, 0, cx; 0, fy, cy; 0,0 ,1] in pixel units
// opt_params are the optimization parameters
// Rf2s is the estimated Rf2s,
// tvec is the estimated tfins, meaning the position of the first camera in the second frame
// rvec and tvec are to store the output
void BAOptimize(vector<Point3f> & pointsXYZ,vector<Point2f>& points1, vector<Point2f>& points2, const cv::Mat &K,
                const OptParams & opt_params, Mat &Rf2sMat, Mat& tfins){
    double reproj_thresh= sqrt(5.995); //from ORB_SLAM
    g2o::SparseOptimizer optimizer;

    double focal_length= K.at<double>(0,0);
    Vector2d principal_point(K.at<double>(0,2), K.at<double>(1,2));

    g2o::CameraParameters * cam_params
            = new g2o::CameraParameters (focal_length, principal_point, 0.);
    cam_params->setId(0);

    setupG2o(cam_params, &optimizer);

    size_t v_id = 0;//vertex id count the frame and point ids

    // first Keyframe Vertex 1: This Keyframe is set to fixed!
    g2oFrameSE3* v_frame1 = new g2oFrameSE3();
    v_frame1->setId(v_id++);
    v_frame1->setFixed(true);
    Eigen::Vector3d trans(0,0,0);
    Eigen:: Quaterniond qd;
    qd.setIdentity();
    g2o::SE3Quat pose(qd,trans);
    v_frame1->setEstimate(pose);
    optimizer.addVertex(v_frame1);

    // New Keyframe Vertex 2
    g2oFrameSE3* v_frame2 = new g2oFrameSE3();
    v_frame2->setId(v_id++);
    v_frame2->setFixed(false);
    trans[0]=tfins.at<double>(0);
    trans[1]=tfins.at<double>(1);
    trans[2]=tfins.at<double>(2);

    //directly use the buffer allocated by OpenCV
    Eigen::Matrix3d eigenRf2s;
    cv2eigen(Rf2sMat,eigenRf2s);
    qd = eigenRf2s;

    pose.setRotation(qd);
    pose.setTranslation(trans);
    v_frame2->setEstimate(pose);
    optimizer.addVertex(v_frame2);

    // Create Point Vertices
    vector<g2oEdgeSE3*> edges;
    for (size_t i=0; i<points1.size(); ++i){
        g2oPoint* v_pt = createG2oPoint(Vector3d(pointsXYZ[i].x, pointsXYZ[i].y, pointsXYZ[i].z), v_id++, false);
        optimizer.addVertex(v_pt);
        g2oEdgeSE3* e = createG2oEdgeSE3(v_frame1, v_pt, Vector2d(points1[i].x, points1[i].y),
                                         opt_params.huber_kernel_width);
        optimizer.addEdge(e);
        edges.push_back(e);
        e = createG2oEdgeSE3(v_frame2, v_pt, Vector2d(points2[i].x, points2[i].y), opt_params.huber_kernel_width);
        optimizer.addEdge(e);
        edges.push_back(e);
    }
    assert(edges.size()==2*pointsXYZ.size());
    // Optimization
    double init_error, final_error;
    runSparseBAOptimizer(&optimizer, opt_params.num_iters, init_error, final_error);
    cout<<"2-View BA: Error before/after = "<< init_error<<", "<<final_error<<endl;

    // Update Keyframe Positions
    Eigen::Quaterniond frame1Quat= v_frame1->estimate().rotation();
    Eigen::Vector3d frame1Trans = v_frame1->estimate().translation();
    assert(frame1Quat.w()==1.0&&frame1Quat.x()==0.0&&frame1Quat.y()==0.0&&frame1Quat.z()==0.0);
    assert(frame1Trans[0]==0.0&&frame1Trans[1]==0.0&&frame1Trans[2]==0.0);

    eigenRf2s = v_frame2->estimate().rotation().toRotationMatrix();
    eigen2cv(eigenRf2s, Rf2sMat);
    trans = v_frame2->estimate().translation();

    tfins.at<double>(0)=trans[0];
    tfins.at<double>(1)=trans[1];
    tfins.at<double>(2)=trans[2];

    // Update Mappoint Positions
    double sum_diff2 = 0;
    //starting from because we have two vertices for two frames
    for(size_t pointId=2; pointId<pointsXYZ.size()+2; ++pointId)
    {
        g2o::HyperGraph::VertexIDMap::iterator v_it
                = optimizer.vertices().find(pointId);
        if (v_it==optimizer.vertices().end()){
            cerr << "Error: Point " << pointId-2 << " not in graph!" << endl;
            return;
        }
        g2o::VertexSBAPointXYZ * v_p
                = dynamic_cast< g2o::VertexSBAPointXYZ * > (v_it->second);

        if (v_p==0){
            cerr << "Error: Point " << pointId-2 << " is not a PointXYZ!" << endl;
            return;
        }
        Vector3d diff(pointsXYZ[pointId-2].x, pointsXYZ[pointId-2].y, pointsXYZ[pointId-2].z);
        diff-=v_p->estimate();
        sum_diff2 += diff.dot(diff);
    }
    cout<<"Sum of squared change in points positions "<<sum_diff2<<endl;
    // Find Mappoints with too large reprojection error
    const double reproj_thresh_squared = reproj_thresh*reproj_thresh;
    size_t n_incorrect_edges = 0;
    for(vector<g2oEdgeSE3*>::iterator it_e = edges.begin(); it_e != edges.end(); ++it_e)
        if((*it_e)->chi2() > reproj_thresh_squared)
            ++n_incorrect_edges;

    cout<<"2-View BA: Wrong edges of total edges: "<< n_incorrect_edges<<" "<< edges.size()<<endl;
}

// extract keyframe indices 0 based from the folder of keyframe binary files,
// taking advantage of that each keyframe binary file is named after the keyframe's 0 based id
std::vector<int> GetKFIndices(std::string keyFrameInfoDir);
//save cc.txt of keyframe 0 based indices
void SaveKFIndices(const std::vector<int>imgIds, std::string keyFrameInfoDir);
// load cc.txt, assume keyframe ids are in ascending order
void LoadKFIndices(const string ccFile, std::vector<KeyFrame*> &vpKFs);
// load framePoses.txt
void LoadKFPoses(string poseFile, std::vector<Sophus::SE3d> &vpKFs);
// load keyframe id and poses from cc.txt and framePoses.txt
void GetAllKeyFrames(std::string keyFrameInfoDir, std::vector<KeyFrame*> & vpKFs);
// ---------------------------------------------------------------------------
// given pairs of image frames, find their P4P relative transformation
// take care the mapping from image frame id to image filename, and the directories for image files
// 5 methods are used, the ground truth, opencv 8 point algorithm, P4P algorithm, Klein's five point
// algorithm implementation, and Snavely's five point implementation. It was found that Klein's method
// is erratic and sometimes erroneous, Snavely's method is rather slow compared to others. After all, I
// recommend opencv five point algorithm to do relative transform estimation with no 3D information.

// ---------------------------------------------------------------------------
template<class TVocabulary, class TDetector, class TDescriptor>
void demoDetector<TVocabulary, TDetector, TDescriptor>::
computeConstraints( std::vector< DetectionResult >& listPairs, const string constraintFile,
                    const cv::Mat cameraMatrix,
                    const string coordsFile, const string tracksFile) const
{
    Mat invK=cameraMatrix.inv();
    cv::Mat distCoeffs(4,1,cv::DataType<double>::type);
    distCoeffs.at<double>(0) = 0;
    distCoeffs.at<double>(1) = 0;
    distCoeffs.at<double>(2) = 0;
    distCoeffs.at<double>(3) = 0;

    vector<cv::KeyPoint> keys[2];//used for SURF features
    Mat descriptors[2];//feature descriptors
    int minHessian = 400;

    Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(minHessian);

    FlannBasedMatcher matcher;
#if USE_KNN_MATCH
    std::vector< std::vector< DMatch > > matches;
#else
    std::vector< DMatch >  matches;
#endif
    std::vector< DMatch > good_matches;

    const double boundaryRatio=1.0/10;//if the matches falls into the 1/10 boundary area, discard
    const double skewThreshY=1.0/4;// if the points have two vastly different y's then discard the match
    const double skewThreshX=1.0/3;
    const int K=6; //how many neighbors are referenced in KNN
    int rows=m_height, cols=m_width;
    cv::Mat rvec(3,1,cv::DataType<double>::type); // the rotation vector and translation from an early frame to late frame computed by P4P method
    cv::Mat tvec(3,1,cv::DataType<double>::type);


    Mat Pw2c[2]; //the world to first camera frame, to second camera frame transform,
    Mat Pf2s, Rf2s, tfins; //the first to second camera transform, and rotation from first to second camera
    Pw2c[0]=Mat::eye(4,4, CV_64F);
    Pw2c[1]=Mat::eye(4,4, CV_64F);
    Pf2s=Mat::eye(4,4, CV_64F);
    Rf2s=Mat::eye(3,3, CV_64F);
    tfins=Mat::zeros(3,1, CV_64F);
    double euler[3]={0.0};  //a temporary buffer to store the euler angles
    int point_count; //how many matches are found
    std::vector<cv::Point2f> imagePoints; //the projected image points

    _TrackMap trackMap;
    ofstream coords;
    if(!coordsFile.empty()){
        coords.open( (keyFrameInfoDir + '/' + coordsFile).c_str(), std::ofstream::out | std::ofstream::trunc);
    }

    char buffer[300]={'\0'};
    Mat img_matches;    // an combined image on which the matched pairs are drawn
    // prepare profiler to measure times
    DUtils::Profiler profiler;

    FILE * myout= fopen ( (keyFrameInfoDir + '/' + constraintFile).c_str(),"w");
    fprintf(myout, "%% 1st line ground truth, frame1 id, frame2 id, Rf2s in euler angles in radians, and Tc1 in c2 in meters\n");
    fprintf(myout,"%% 2nd line 8 point in OpenCV, camera distortion not considered!\n");
    fprintf(myout,"%% frame 1 id, matches, euler angles RPY Rf2s=R3(-y)R2(-p)R1(-r), and normalized translation T 1st in 2nd frame\n");

    fprintf(myout,"%% 3rd line P4PRansac, matches number, scale 2nd/scale 1st, euler angles RPY for Rc12c2, and Tfins frame\n");
    fprintf(myout,"%% 4th line P4PRansac+ 2ViewBA, matches number, scale 2nd/scale 1st, euler angles RPY for Rc12c2, and Tfins frame\n");
#if USE_KLEIN
    fprintf(myout,"%% 5th line 5points Klein, 1st frameid, matches number, euler angles RPY for Rc12c2, and Tfins frame direction\n");
#endif
#if USE_SNAVELY
    fprintf(myout,"%% 6th line 5points Snavely, 1st frameid, matches number, euler angles RPY for Rc12c2, and Tfins frame direction\n");
#endif
    for(unsigned int count=0; count<listPairs.size();++count)
    {
        int firstFrame=min(listPairs[count].query, listPairs[count].match);// depth info from early frames is more reliable
        int secondFrame=max(listPairs[count].query, listPairs[count].match);
        //(1) compute the ground truth
        ReadCameraPose(m_posefile.c_str(), firstFrame, Pw2c[0]);
        ReadCameraPose(m_posefile.c_str(), secondFrame, Pw2c[1]);
        Pf2s=Pw2c[1]*Pw2c[0].inv();
        Rf2s=Pf2s(cv::Rect(0,0,3,3));
        DCM2Euler(Rf2s, euler);
        tfins=Pf2s(cv::Rect(3,0,1,3));
        fprintf(myout, "%d\t%d\t", firstFrame, secondFrame);
        fprintf(myout, "%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n", euler[0],euler[1],euler[2],
                tfins.at<double>(0,0),tfins.at<double>(1,0),tfins.at<double>(2,0));

        //(2) recover Rfirst2sec and Tfirst in sec from the essential matrix

        keys[0].clear();
        keys[1].clear();
        Rf2s=Mat(3,3, CV_64F, cvScalar(0.));
        tfins=Mat(3,1, CV_64F, cvScalar(0.));
        // get image 1
        sprintf(buffer, "%s/%06d.png", m_imagedir.c_str(),firstFrame);
        cv::Mat im0= cv::imread(buffer, 0); // grey scale
        // get features
        profiler.profile("features");
        surf->detect( im0, keys[0] );
        surf->compute( im0, keys[0], descriptors[0]);
        profiler.stop();
        // get image 2
        sprintf(buffer, "%s/%06d.png", m_imagedir.c_str(),secondFrame);
        cv::Mat im1 = cv::imread(buffer, 0); // grey scale

        // get features
        profiler.profile("features");
        surf->detect( im1, keys[1] );
        surf->compute( im1, keys[1], descriptors[1]);
        profiler.stop();
#if USE_KNN_MATCH
        matcher.knnMatch(descriptors[0], descriptors[1], matches, 2);
#else
        matcher.match( descriptors[0], descriptors[1], matches );
#endif
        good_matches.clear();

        for( int i = 0; i < matches.size(); i++ )
        {
#if USE_KNN_MATCH
            if(matches[i].size()==2)
            {
                if((matches[i][0].distance==0&&matches[i][1].distance>0)||matches[i][1].distance/matches[i][0].distance>thresh)
                {
                    cv::Point2f pt1=keys[0][matches[i][0].queryIdx].pt;
                    cv::Point2f pt2=keys[1][matches[i][0].trainIdx].pt;
                    if(pt1.x>=boundaryRatio*cols&&pt1.y>=boundaryRatio*rows&&
                            pt1.x<=(1-boundaryRatio)*cols&&pt1.y<=(1-boundaryRatio)*rows&&
                            pt2.x>=boundaryRatio*cols&&pt2.y>=boundaryRatio*rows&&
                            pt2.x<=(1-boundaryRatio)*cols&&pt2.y<=(1-boundaryRatio)*rows)
                    {
                        if(std::abs(pt2.y-pt1.y)<skewThreshY*rows&&std::abs(pt2.x-pt1.x)<skewThreshX*cols)
                            good_matches.push_back(matches[i][0]);
                    }
                }
            }

#else

            cv::Point2f pt1=keys[0][matches[i].queryIdx].pt;
            cv::Point2f pt2=keys[1][matches[i].trainIdx].pt;
            if(pt1.x>=boundaryRatio*cols&&pt1.y>=boundaryRatio*rows&&
                    pt1.x<=(1-boundaryRatio)*cols&&pt1.y<=(1-boundaryRatio)*rows&&
                    pt2.x>=boundaryRatio*cols&&pt2.y>=boundaryRatio*rows&&
                    pt2.x<=(1-boundaryRatio)*cols&&pt2.y<=(1-boundaryRatio)*rows)
            {
                if(std::abs(pt2.y-pt1.y)<skewThreshY*rows&&std::abs(pt2.x-pt1.x)<skewThreshX*cols)
                    good_matches.push_back(matches[i]);
            }

#endif
        }
        //since there are some point matches two points in the other image, make the match unique
        map<int, int> uniqueMatch;// a mapping from trainIdx to good_matches index
        map<int, int>::iterator sIt;
        point_count=good_matches.size();
        for (int kettle=0;kettle<point_count;++kettle)
        {
            int trainId=good_matches[kettle].trainIdx;
            sIt=uniqueMatch.find(trainId);

            if(sIt==uniqueMatch.end()){
                uniqueMatch.insert(make_pair(trainId, kettle));
            }
            else{

                if(good_matches[sIt->second].distance<=good_matches[kettle].distance){
                    good_matches[kettle].trainIdx=-1;
                }
                else{
                    good_matches[sIt->second].trainIdx=-1;
                    sIt->second=kettle; //record the minimum distance candidate
                }
            }
        }
        uniqueMatch.clear();
        int mercury=0;
        for (int kettle=0; kettle<good_matches.size(); ++kettle)
        {
            if(good_matches[kettle].trainIdx!=-1)
            {
                good_matches[mercury]=good_matches[kettle];
                ++mercury;
            }
        }
        good_matches.resize(mercury);
        //(2) compute the rotation and translation with Hartley's 8 point algorithm in OpenCV
        point_count=good_matches.size();
        if(point_count>8){
            vector<Point2f> points1(point_count);
            vector<Point2f> points2(point_count);
            for (int kettle=0;kettle<point_count;++kettle)
            {
                points1[kettle].x=keys[0][good_matches[kettle].queryIdx].pt.x;
                points1[kettle].y=keys[0][good_matches[kettle].queryIdx].pt.y;
                points2[kettle].x=keys[1][good_matches[kettle].trainIdx].pt.x;
                points2[kettle].y=keys[1][good_matches[kettle].trainIdx].pt.y;
            }
            cout<<"Running essential matrix decomposition method!"<<endl;
            ExtractCameras( points1,  points2, Rf2s, tfins, cameraMatrix);
            DCM2Euler(Rf2s, euler);
            fprintf(myout, "%d\t%d\t", firstFrame, point_count);
            fprintf(myout, "%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n", euler[0],euler[1],euler[2],
                    tfins.at<double>(0,0),tfins.at<double>(1,0),tfins.at<double>(2,0));

        }
        else
        {
            fprintf(myout, "%d\t%d\t", firstFrame, point_count);
            fprintf(myout, "0\t0\t0\t0\t0\t0\n");
        }

        //-- Draw only "good" matches
        cout<<"Good matches: "<<good_matches.size()<<" between "<<keys[0].size()<<" and "<<keys[1].size()<<" points!"<<endl;
        drawMatches( im0, keys[0], im1, keys[1],
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        //-- Show detected matches
        imshow( "Good Matches for essential matrix", img_matches );
        waitKey(5);
        matches.clear();

        // (3) PnP to compute the constraints and (4) Bundle adjustment refinement by G2O

        vector<cv::Point2f> obsinc[2];
        vector<cv::Point3f> ptsinc[2];

        sprintf(buffer, "%s/KeyFrame%06d.bin", keyFrameInfoDir.c_str(), firstFrame);
        LoadComboKeyFrame(buffer, Pw2c[0], obsinc[0], ptsinc[0]);
        sprintf(buffer, "%s/KeyFrame%06d.bin", keyFrameInfoDir.c_str(), secondFrame);
        LoadComboKeyFrame(buffer, Pw2c[1], obsinc[1], ptsinc[1]);
        rvec=Scalar(0);
        tvec=Scalar(0);
        imagePoints.clear();
        cv::projectPoints(ptsinc[0], rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

        float deltaSum=0;
        for (unsigned int hermit=0; hermit<ptsinc[0].size(); ++hermit){
            Point2f delta=imagePoints[hermit]-obsinc[0][hermit];
            deltaSum+=delta.dot(delta);
        }
        cout<<"firstFrame squared residual: "<<deltaSum<<" on "<<obsinc[0].size()<<" points!"<<endl;

        cv::projectPoints(ptsinc[1], rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

        deltaSum=0;
        for (unsigned int hermit=0; hermit<ptsinc[1].size(); ++hermit){
            Point2f delta=(imagePoints[hermit]-obsinc[1][hermit]);
            deltaSum+=delta.dot(delta);
        }
        cout<<"secondFrame squared residual: "<<deltaSum<<" on "<<obsinc[1].size()<<" points!"<<endl;


        float resp=0;
        // form the training samples
        Mat trainData[2], responses[2];

        Ptr<ml::KNearest> knn[] = {ml::KNearest::create(), ml::KNearest::create()};

        vector<Point3f> surfPoints[2];
        surfPoints[0].resize(good_matches.size());
        surfPoints[1].resize(good_matches.size());
        cv::Mat nearests( 1, K, CV_32FC1);

        cv::Mat sample(1, 2, CV_32FC1);
        for(int cap=0; cap<2 ;++cap){
            trainData[cap]=Mat( obsinc[cap].size(), 2, CV_32F);
            responses[cap]=Mat( obsinc[cap].size(), 1, CV_32F);

            for(unsigned int hermit=0; hermit<obsinc[cap].size(); ++hermit){
                trainData[cap].at<float>(hermit,0)=obsinc[cap][hermit].x;
                trainData[cap].at<float>(hermit,1)=obsinc[cap][hermit].y;
                responses[cap].at<float>(hermit,0)=ptsinc[cap][hermit].z;
            }

            // learn classifier, reference: http://answers.opencv.org/question/90667/how-to-implement-k-nearest-neighbour/
            knn[cap]->setIsClassifier(false);
            knn[cap]->setAlgorithmType(cv::ml::KNearest::Types::KDTREE);
            knn[cap]->setDefaultK(K);
            knn[cap]->train(trainData[cap], cv::ml::ROW_SAMPLE, responses[cap]);
            for( unsigned int hermit=0; hermit<good_matches.size(); ++hermit )
            {
                if(cap==0){
                    sample.at<float>(0) = (float)keys[cap][good_matches[hermit].queryIdx].pt.x;
                    sample.at<float>(1) = (float)keys[cap][good_matches[hermit].queryIdx].pt.y;
                }
                else{
                    sample.at<float>(0) = (float)keys[cap][good_matches[hermit].trainIdx].pt.x;
                    sample.at<float>(1) = (float)keys[cap][good_matches[hermit].trainIdx].pt.y;
                }
                // estimate the response and get the neighbors' labels
                cv::Mat res;
                resp = knn[cap]->findNearest(sample, K, res, nearests);

                Mat measurement(3,1,CV_64F);
                measurement.at<double>(0)=sample.at<float>(0);
                measurement.at<double>(1)=sample.at<float>(1);
                measurement.at<double>(2)=1;
                measurement=resp*invK*measurement;
                surfPoints[cap][hermit].x=(float)measurement.at<double>(0);
                surfPoints[cap][hermit].y=(float)measurement.at<double>(1);
                surfPoints[cap][hermit].z=(float)measurement.at<double>(2);
            }
        }
        //compute the rotation and translation
        point_count=good_matches.size();
        if(point_count>8){
            vector<Point2f> points1(point_count);
            vector<Point2f> points2(point_count);

            std::vector<float> depths[2];
            depths[0].resize(point_count);
            depths[1].resize(point_count);
            for (int kettle=0;kettle<point_count;++kettle)
            {
                points1[kettle]=keys[0][good_matches[kettle].queryIdx].pt;
                points2[kettle]=keys[1][good_matches[kettle].trainIdx].pt;
                depths[0][kettle]=surfPoints[0][kettle].z;
                depths[1][kettle]=surfPoints[1][kettle].z;
            }

            std::cout << "Running PnP with " << point_count << " matched imagePoints!"<< std::endl;
            //cv::solvePnP(objectPoints, points2, cameraMatrix, distCoeffs, rvec, tvec, false, CV_ITERATIVE);

            cv::solvePnPRansac(surfPoints[0], points2, cameraMatrix, distCoeffs,
                    rvec, tvec, false, 100, 3, 10, cv::noArray(), CV_ITERATIVE);//CV_ITERATIVE seems to give better result than CV_P3P
            Rodrigues(rvec, Rf2s);
            DCM2Euler(Rf2s, euler);

            //compute scale change
            std::vector<float>::iterator it = depths[0].begin()+0.5*depths[0].size();
            std::nth_element(depths[0].begin(), it, depths[0].end());
            double sloop=*it;
            it = depths[1].begin()+0.5*depths[1].size();
            std::nth_element(depths[1].begin(), it, depths[1].end());
            sloop=*it/sloop;
            vector<float> depthRatios;
            depthRatios.resize(depths[0].size());
            for(unsigned int plow=0; plow<depths[0].size();++plow)
                depthRatios[plow]=depths[1][plow]/depths[0][plow];
            it = depthRatios.begin()+0.5*depthRatios.size();
            std::nth_element(depthRatios.begin(), it, depthRatios.end());
            cout<<"ratio of median depth "<<sloop<<" and median of depth ratio "<<*it<<endl;

            fprintf(myout, "%d\t%.8f\t", point_count, sloop);
            fprintf(myout, "%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n", euler[0],euler[1],euler[2],
                    tvec.at<double>(0,0),tvec.at<double>(1,0),tvec.at<double>(2,0));

            // refine relative transformation with BA
            BAOptimize(surfPoints[0], points1, points2, cameraMatrix, OptParams(10, true, 3), Rf2s, tvec);
            DCM2Euler(Rf2s, euler);
            fprintf(myout, "%d\t%.8f\t", point_count, sloop);
            fprintf(myout, "%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n", euler[0],euler[1],euler[2],
                    tvec.at<double>(0,0),tvec.at<double>(1,0),tvec.at<double>(2,0));

        }
        else
        {
            //for dummy PnP
            fprintf(myout, "%d\t0.0\t", point_count);
            fprintf(myout, "0\t0\t0\t0\t0\t0\n");
            //for dummy BA
            fprintf(myout, "%d\t0.0\t", point_count);
            fprintf(myout, "0\t0\t0\t0\t0\t0\n");
        }
        // KLT tracking does not work well for key points in keyframe info files
        /*drawKeypoints( im0, firstFrame==firstFrame? obsKeys[0]:obsKeys[1], img_keypoints_1, Scalar(0,255,0), DrawMatchesFlags::DRAW_OVER_OUTIMG );
        drawKeypoints( im1, firstFrame==firstFrame? obsKeys[1]:obsKeys[0], img_keypoints_2, Scalar(0,255,0), DrawMatchesFlags::DRAW_OVER_OUTIMG);
        //-- Show detected (drawn) keypoints
        Mat img_keypoints_1; Mat img_keypoints_2;
        imshow("Keypoints 1", img_keypoints_1 );
        imshow("Keypoints 2", img_keypoints_2 );
        waitKey();

        vector<uchar> status[2]; //status[0] forward searching indicator, status[1] backward searching indicator
        vector<float> err;
        int nWinWidth=21;
        Size winSize(nWinWidth,nWinWidth);
        TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);

        vector<Point2f> vfPoints[3];
        vfPoints[0] = obsinc[1];// points in firstFrame tracked from second frame
        vfPoints[1] = obsinc[1];// points in secondFrame tracked reversely from first frame points, vfPoints[0]

        calcOpticalFlowPyrLK(firstFrame==firstFrame?im1:im0, firstFrame==firstFrame?im0:im1, obsinc[1], vfPoints[0], status[0], err, winSize,
                3, termcrit, OPTFLOW_USE_INITIAL_FLOW, 0.001);
        //what is returned for points failed to track? the initial position
        // if a point is out of boundary, it will ended up failing to track

        calcOpticalFlowPyrLK(firstFrame==firstFrame?im0:im1, firstFrame==firstFrame?im1:im0, vfPoints[0], vfPoints[1], status[1], err, winSize,
                3, termcrit, OPTFLOW_USE_INITIAL_FLOW, 0.001);
        for(unsigned int which=0; which<status[0].size(); ++which){// note status[0].size() is constant, but not mlTrailers.size()
                if(status[0][which]){
                    Point2f delta=vfPoints[1][which] - obsinc[1][which];
                    bool bInImage=(vfPoints[0][which].x>=0.f)&&(vfPoints[0][which].y>=0.f)&&
                        (vfPoints[0][which].x<=(m_width-1))&&(vfPoints[0][which].y<=(m_height-1));

                    if(!status[1][which]||!bInImage||(delta.dot(delta)) > 2)
                        status[0][which] = 0;
                }
                if(status[0][which])
                    vfPoints[2].push_back(vfPoints[0][which]);
        }
        vector<KeyPoint> trackedKeys;
        for(unsigned int hermit=0; hermit<vfPoints[2].size(); ++hermit)
            trackedKeys.push_back(KeyPoint(vfPoints[2][hermit].x,vfPoints[2][hermit].y, 21));

        drawKeypoints( firstFrame==firstFrame?im0:im1, trackedKeys , img_keypoints_1, Scalar(0,0,255), DrawMatchesFlags::DRAW_OVER_OUTIMG );
        //-- Show detected (drawn) keypoints
        imshow("Keypoints 1", img_keypoints_1 );
        waitKey();*/
#if USE_KLEIN
        //(5) 5 point algorithm nister implemented by Klein
        // generates the initial match from two keyframes and a vector of image correspondences.

        PTAMM::ATANCamera Camera("Camera");
        if(point_count>8){
            vector<HomographyMatch> vMatches;
            for (int kettle=0;kettle<point_count;++kettle)
            {
                HomographyMatch m;
                Point2f point1, point2;
                point1.x=keys[0][good_matches[kettle].queryIdx].pt.x;
                point1.y=keys[0][good_matches[kettle].queryIdx].pt.y;
                point2.x=keys[1][good_matches[kettle].trainIdx].pt.x;
                point2.y=keys[1][good_matches[kettle].trainIdx].pt.y;

                TooN::Vector<2> vfirst=makeVector(point1.x, point1.y);
                TooN::Vector<2> vsecond=makeVector(point2.x, point2.y);
                m.v2CamPlaneFirst = Camera.UnProject(vfirst);
                m.v2CamPlaneSecond = Camera.UnProject(vsecond);
                m.m2PixelProjectionJac = Camera.GetProjectionDerivs();
                vMatches.push_back(m);
            }
            SE3<> se3;
            bool bGood;
            HomographyInit HomographyInit;
            cout<<"Running 5 point algorithm by Klein!"<<endl;
            bGood = HomographyInit.Compute(vMatches, 5.0, se3);
            if(!bGood)
            {
                cout << "  Could not init from stereo pair." << endl;
                fprintf(myout, "%d\t%d\t", firstFrame, point_count);
                fprintf(myout, "0\t0\t0\t0\t0\t0\n");
            }
            else{
                // Check that the initialiser estimated a non-zero baseline
                double dTransMagn = sqrt(se3.get_translation() * se3.get_translation());
                if(dTransMagn == 0)
                {
                    cout << "  Estimated zero baseline from stereo pair, try again." << endl;
                    fprintf(myout, "%d\t%d\t", firstFrame, point_count);
                    fprintf(myout, "0\t0\t0\t0\t0\t0\n");
                }

                se3.get_translation() *= 1/dTransMagn;
                // TooN::Vector<3> vecTester=se3.get_rotation().ln();// this is generally not equal to euler angles
                TooN::Vector<3> vecEuler=rotro2eu(se3.get_rotation().get_matrix());
                fprintf(myout, "%d\t%d\t", firstFrame, point_count);
                fprintf(myout, "%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n", vecEuler[0], vecEuler[1], vecEuler[2],
                        se3.get_translation()[0],se3.get_translation()[1],se3.get_translation()[2]);
            }
        }
        else
        {
            fprintf(myout, "%d\t%d\t", firstFrame, point_count);
            fprintf(myout, "0\t0\t0\t0\t0\t0\n");
        }
#endif
#if USE_SNAVELY
        //(6): snavely impelemented nister's 5 point algorithm
        /* Estimate relative pose from a given set of point matches */
        if(point_count>8){
            int num_trials=512;
            double threshold=9.0*0.25;
            double K1[9];
            double K2[9];
            double Rl2r[9];
            double tlinr[3];
            GetIntrinsics(cameraMatrix, K1);
            GetIntrinsics(cameraMatrix, K2);

            v2_t *k1_pts = new v2_t[point_count];
            v2_t *k2_pts = new v2_t[point_count];

            for (int kettle= 0; kettle < point_count; kettle++) {

                k1_pts[kettle] = v2_new(keys[0][good_matches[kettle].queryIdx].pt.x, keys[0][good_matches[kettle].queryIdx].pt.y);
                k2_pts[kettle] = v2_new(keys[1][good_matches[kettle].trainIdx].pt.x, keys[1][good_matches[kettle].trainIdx].pt.y);
            }
            cout<<"Running Snavely's 5 point algorithm!"<<endl;
            int num_inliers = compute_pose_ransac(point_count, k1_pts, k2_pts,
                                                  K1, K2, threshold, num_trials, Rl2r, tlinr);

            delete [] k1_pts;
            delete [] k2_pts;

            cout<<num_inliers<<" inliers found for 5 point algorithm by Snavely!"<<endl;
            TooN::Matrix<3> toonRl2r=Data(Rl2r[0],Rl2r[1],Rl2r[2],Rl2r[3],Rl2r[4],Rl2r[5],Rl2r[6],Rl2r[7],Rl2r[8]);
            TooN::Vector<3> toonTlinr=TooN::makeVector(tlinr[0],tlinr[1],tlinr[2]);
            SE3<> Plinr(SO3<>(toonRl2r), toonTlinr);
            Plinr=Plinr.inverse();
            toonRl2r=Plinr.get_rotation().get_matrix();
            toonTlinr=Plinr.get_translation();
            TooN::Vector<3> vecEuler=rotro2eu(toonRl2r);
            fprintf(myout, "%d\t%d\t", firstFrame, point_count);
            fprintf(myout, "%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n", vecEuler[0], vecEuler[1], vecEuler[2],
                    toonTlinr[0],toonTlinr[1], toonTlinr[2]);
        }
        else
        {
            fprintf(myout, "%d\t%d\t", firstFrame, point_count);
            fprintf(myout, "0\t0\t0\t0\t0\t0\n");
        }
#endif


        //update trackMap
        unsigned int lastTrackId=trackMap.size();
        for (int kettle=0;kettle<point_count;++kettle)
        {
            _Track oneMatch;
            oneMatch.push_back(IdPair(good_matches[kettle].queryIdx, firstFrame));
            oneMatch.push_back(IdPair(good_matches[kettle].trainIdx, secondFrame));
            trackMap.insert(make_pair(lastTrackId++,oneMatch));
        }
        // save all the observations into cc.txt and trackmap which is to be saved afterwards
        if(coords.is_open()){
            //first frame
            coords<<firstFrame<<" "<<keys[0].size()<<endl;
            int dummy=0;
            unsigned int red=0, green=0, blue=0;
            for(unsigned int charade=0; charade<keys[0].size(); ++charade){
                coords<<charade; //key number
                coords<<std::fixed;
                coords<<setprecision(6);
                coords<<" "<<keys[0][charade].pt.x<<" "<<keys[0][charade].pt.y;
                coords<<" "<<dummy<<" "<<dummy;
                coords<<" "<<red<<" "<<green<<" "<<blue<<endl;
            }
            //second frame
            coords<<secondFrame<<" "<<keys[1].size()<<endl;
            for(unsigned int charade=0; charade<keys[1].size(); ++charade){
                coords<<charade; //key number
                coords<<std::fixed;
                coords<<setprecision(6);
                coords<<" "<<keys[1][charade].pt.x<<" "<<keys[1][charade].pt.y;
                coords<<" "<<dummy<<" "<<dummy;
                coords<<" "<<red<<" "<<green<<" "<<blue<<endl;
            }
        }
    }
    fclose(myout);
    if(coords.is_open())
        coords.close();
    if(!tracksFile.empty())
    {
        ofstream tracks((keyFrameInfoDir + '/' + tracksFile).c_str(), ofstream::out | ofstream::trunc);
        tracks<<trackMap.size()<<endl;
        tracks<<trackMap;
        tracks.close();
    }
}

// ---------------------------------------------------------------------------
template<class TVocabulary, class TDetector, class TDescriptor>
void demoDetector<TVocabulary, TDetector, TDescriptor>::run
(const std::string &name, const FeatureExtractor<TDescriptor> &extractor)
{
    cout << "DLoopDetector Demo" << endl
         << "Dorian Galvez-Lopez" << endl
         << "http://webdiis.unizar.es/~dorian" << endl << endl;

    // Set loop detector parameters
    typename TDetector::Parameters params(m_height, m_width);

    // Parameters given by default are:
    // use nss = true
    // alpha = 0.3
    // k = 3
    // geom checking = GEOM_DI
    // di levels = 0

    // We are going to change these values individually:
    params.use_nss = true; // use normalized similarity score instead of raw score
    params.alpha = 0.3; // nss threshold
    params.k = 3; // a loop must be consistent with 1 previous matches
    params.geom_check = GEOM_FLANN; // use FLANN for KITTI dataset removes false positives
    params.di_levels = 2; // use two direct index levels

    // To verify loops you can select one of the next geometrical checkings:
    // GEOM_EXHAUSTIVE: correspondence points are computed by comparing all
    //    the features between the two images.
    // GEOM_FLANN: as above, but the comparisons are done with a Flann structure,
    //    which makes them faster. However, creating the flann structure may
    //    be slow.
    // GEOM_DI: the direct index is used to select correspondence points between
    //    those features whose vocabulary node at a certain level is the same.
    //    The level at which the comparison is done is set by the parameter
    //    di_levels:
    //      di_levels = 0 -> features must belong to the same leaf (word).
    //         This is the fastest configuration and the most restrictive one.
    //      di_levels = l (l < L) -> node at level l starting from the leaves.
    //         The higher l, the slower the geometrical checking, but higher
    //         recall as well.
    //         Here, L stands for the depth levels of the vocabulary tree.
    //      di_levels = L -> the same as the exhaustive technique.
    // GEOM_NONE: no geometrical checking is done.
    //
    // In general, with a 10^6 vocabulary, GEOM_DI with 2 <= di_levels <= 4
    // yields the best results in recall/time.
    // Check the T-RO paper for more information.
    //

    // Load the vocabulary to use
    cout << "Loading " << name << " vocabulary..." << endl;
    TVocabulary voc(m_vocfile);
    cout << "Vocabulary information: " << endl
         << voc << endl << endl;

    // Initiate loop detector with the vocabulary
    cout << "Processing sequence..." << endl;
    TDetector detector(voc, params);

    // Process images
    vector<cv::KeyPoint> keys;
    vector<TDescriptor> descriptors;

    //get the list of filenames

    vector<int> imgIds=GetKFIndices(keyFrameInfoDir);

    //Is keyframefiles have full directory?
    char buffer[300]={'\0'};
    vector<string> filenames;
    filenames.resize(imgIds.size());
    for(unsigned int count=0; count<imgIds.size();++count)
    {
        sprintf(buffer, "%s/%06d.png", m_imagedir.c_str(),imgIds[count]);
        filenames[count]=buffer;
    }

    // load robot poses
    vector<double> xs, ys;
    readPoseFile(m_posefile.c_str(), xs, ys, imgIds);

    // we can allocate memory for the expected number of images
    detector.allocate(filenames.size());

    // prepare visualization windows
    DUtilsCV::GUI::tWinHandler win = "Current image";
    DUtilsCV::GUI::tWinHandler winplot = "Trajectory";

    DUtilsCV::Drawing::Plot::Style normal_style(2); // thickness
    DUtilsCV::Drawing::Plot::Style loop_style('r', 2); // color, thickness

    DUtilsCV::Drawing::Plot implot(480, 640,
                                   - *std::max_element(xs.begin(), xs.end()),
                                   - *std::min_element(xs.begin(), xs.end()),
                                   *std::min_element(ys.begin(), ys.end()),
                                   *std::max_element(ys.begin(), ys.end()), 20);

    // prepare profiler to measure times
    DUtils::Profiler profiler;

    int count = 0;

    // go
    for(unsigned int i = 0; i < filenames.size(); ++i)
    {
        cout << "Adding image " << i << ": " << filenames[i] << "... " << endl;

        // get image
        cv::Mat im = cv::imread(filenames[i].c_str(), 0); // grey scale

        // show image
        DUtilsCV::GUI::showImage(im, true, &win, 10);

        // get features
        profiler.profile("features");
        extractor(im, keys, descriptors);
        profiler.stop();
        
        // add image to the collection and check if there is some loop
        DetectionResult result;

        profiler.profile("detection");
        detector.detectLoop(keys, descriptors, result);
        profiler.stop();

        if(result.detection())
        {
            DetectionResult larry(result);
            larry.match= imgIds[result.match];
            larry.query= imgIds[result.query];
            vDetectionResults.push_back(larry);
            //query is the current value of i, match is the matched image of i
            cout << "- Loop found between "<<result.query<< " and image " << result.match << "!"
                 << endl;
            ++count;
        }
        else
        {
            cout << "- No loop: ";
            switch(result.status)
            {
            case CLOSE_MATCHES_ONLY:
                cout << "All the images in the database are very recent" << endl;
                break;

            case NO_DB_RESULTS:
                cout << "There are no matches against the database (few features in"
                        " the image?)" << endl;
                break;

            case LOW_NSS_FACTOR:
                cout << "Little overlap between this image and the previous one"
                     << endl;
                break;

            case LOW_SCORES:
                cout << "No match reaches the score threshold (alpha: " <<
                        params.alpha << ")" << endl;
                break;

            case NO_GROUPS:
                cout << "Not enough close matches to create groups. "
                     << "Best candidate: " << result.match << endl;
                break;

            case NO_TEMPORAL_CONSISTENCY:
                cout << "No temporal consistency (k: " << params.k << "). "
                     << "Best candidate: " << result.match << endl;
                break;

            case NO_GEOMETRICAL_CONSISTENCY:
                cout << "No geometrical consistency. Best candidate: "
                     << result.match << endl;
                break;

            default:
                break;
            }
        }

        cout << endl;

        // show trajectory
        if(i > 0)
        {
            if(result.detection())
                implot.line(-xs[i-1], ys[i-1], -xs[i], ys[i], loop_style);
            else
                implot.line(-xs[i-1], ys[i-1], -xs[i], ys[i], normal_style);

            DUtilsCV::GUI::showImage(implot.getImage(), true, &winplot, 10);
        }
    }

    if(count == 0)
    {
        cout << "No loops found in this image sequence" << endl;
    }
    else
    {
        cout << count << " loops found in this image sequence!" << endl;
    }
    assert(count==vDetectionResults.size());
    for(int metal=0; metal<count ;++metal)
    {
        cout << "- Loop found between "<< vDetectionResults[metal].query<< " and image " <<  vDetectionResults[metal].match << "!"
             << endl;
    }

    cout << endl << "Execution time:" << endl
         << " - Feature computation: " << profiler.getMeanTime("features") * 1e3
         << " ms/image" << endl
         << " - Loop detection: " << profiler.getMeanTime("detection") * 1e3
         << " ms/image" << endl;

    cout << endl << "Press a key to finish..." << endl;
    DUtilsCV::GUI::showImage(implot.getImage(), true, &winplot, 0);
}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
void demoDetector<TVocabulary, TDetector, TDescriptor>::readPoseFile
(const char *filename, std::vector<double> &xs, std::vector<double> &ys, vector<int> &lineNo)
const
{
    xs.clear();
    ys.clear();

    fstream f(filename, ios::in);

    string s;
    double resort[12]={0};
    int lineCounter=0;
    int pos=0;
    while(!f.eof())
    {
        while(lineCounter<=lineNo[pos]){
            getline(f, s);
            ++lineCounter;
        }
        if(!f.eof() && !s.empty())
        {
            sscanf(s.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &resort[0], &resort[1], &resort[2], &resort[3],
                    &resort[4], &resort[5], &resort[6], &resort[7],
                    &resort[8], &resort[9], &resort[10], &resort[11]);

            //we are only interested in x and z
            xs.push_back(resort[3]);
            ys.push_back(resort[11]);
        }
        ++pos;
    }
    f.close();
}
// ---------------------------------------------------------------------------
#endif

