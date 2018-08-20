/**
 * File: kitti_surf.cpp
 * Date: November 2015
 * Author: Jianzhu Huai
 * Description: demo application of DLoopDetector
 */

#include <iostream>
#include <vector>
#include <set>
#include <string>

// DLoopDetector and DBoW2
#include "DBoW2.h" // defines Surf64Vocabulary
#include "DLoopDetector.h" // defines Surf64LoopDetector
#include "DUtilsCV.h" // defines macros CVXX 

// OpenCV
#include <opencv/cv.h>
#include <opencv/highgui.h>
#if CV24
#include <opencv2/nonfree/features2d.hpp>
#endif
//g2o and Sophus

#include "vio_g2o/scale_solver.h"
#include "vio_g2o/anchored_points.h"

#include "sophus/sim3.hpp"
#include "sophus/se3.hpp"

#include "g2o/types/sim3/sim3.h"

// Demo
//#include "TooN/se3.h"
#include "kittiDetector.h"
#include "drawPTAMPoints.h"
#include "bal_example.h"

using namespace TooN; //TooN SE3
using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;

// All parameters----------------------------------------------------------------------------

static const char *VOC_FILE = "../../resources/surf64_k10L6.voc.gz";
static const char *IMAGE_DIR = "kitti/dataset/sequences/00/image_0"; //for loop closure detection and visualization
static const char *POSE_FILE = "../data/map000000/00.txt"; //ground truth for visualization

static const int IMAGE_W = 1241; // image size
static const int IMAGE_H = 376;
string MyKeyFrame::m_imagedir(IMAGE_DIR);
double MyKeyFrame::m_f=718.856;
double MyKeyFrame::m_px=607.1928;
double MyKeyFrame::m_py=185.2157;
char * KeyFrameInfoDir="../data/map000000";


void LoadLoopCoords(const string loopCoordsFile, map<unsigned int, MyKeyFrame> &loopKeyFrames){
    ifstream loopCoords(loopCoordsFile.c_str(), ifstream::in);
    MyKeyFrame curFrame(0);
    while(!loopCoords.eof()){
        loopCoords>>curFrame;
        if(!loopCoords.eof()){
            assert(curFrame.keys.size()!=0);
            map<unsigned int, MyKeyFrame>::iterator cIt=loopKeyFrames.find(curFrame.imgID);
            if(cIt->second.keys.size()==0)
                cIt->second=curFrame;
            else
                assert(cIt->second==curFrame);
        }
    }
    loopCoords.close();
}

// ---------------------------------------------------------------------------

//load the computed loop constraints by computeConstraints into loopConstraints vector
//each constraints has f, s, Rf2s, Tfins
void LoadLoopConstraints(const string loopConstraintFile, vector<Constraint<SE3<>, 6 > > &loopConstraints)
{
    //load the constraints
    ifstream loopData(loopConstraintFile.c_str(),ios::in);
    char buffer[256]={'\0'};
    int headerLines=5;
#if USE_KLEIN
    ++headerLines;
#endif
#if USE_SNAVELY
    ++headerLines;
#endif
    if(loopData.is_open())//remove the header lines
        for(int header=0; header<headerLines; ++header)
            loopData.getline(buffer, 300);
    else{
        cout<<"loopConstraints.txt file not opened!"<<endl;
        exit(-1);
    }

    unsigned int frameId[2]; //the frame_ids of two frames that are involved in a loop detection
    TooN::Matrix<6> inf6 = Identity(6);

    while(!loopData.eof())
    {
        double resort[6];
        int tempInt;
        double matchCount;

        loopData.getline(buffer, 300); //ground truth

        if(loopData.eof())
            break;
        sscanf(buffer, "%u %u %lf %lf %lf %lf %lf %lf",
               &frameId[0], &frameId[1], &resort[0], &resort[1],
                &resort[2], &resort[3], &resort[4], &resort[5]);
        loopData.getline(buffer, 300);// opencv 5 point

        loopData.getline(buffer, 300);// knn regression
        loopData.getline(buffer, 300);// knn regression + 2 View BA

        sscanf(buffer, "%d %lf %lf %lf %lf %lf %lf %lf",
               &tempInt, &matchCount, &resort[0], &resort[1],
                &resort[2], &resort[3], &resort[4], &resort[5]);
        assert(resort[0]!=0&&resort[1]!=0&&resort[2]!=0);

#if USE_KLEIN
        loopData.getline(buffer, 300);//use the 5 point algorithm results by Klein
#endif
#if USE_SNAVELY
        loopData.getline(buffer, 300);// snavely 5 point
#endif

        TooN::Matrix<3> Rf2s=roteu2ro(makeVector(resort[0], resort[1], resort[2]));

        SE3<> loopCond(Rf2s,makeVector(resort[3], resort[4], resort[5]));

        loopConstraints.push_back(Constraint<SE3<>, 6>(frameId[0], frameId[1], loopCond, inf6));
    }
    loopData.close();
}

void LoadLoopConstraints(const string loopConstraintFile, vector<Constraint<g2o::Sim3, 7 > > &loopConstraints)
{
    //load the constraints
    ifstream loopData(loopConstraintFile.c_str(),ios::in);
    char buffer[256]={'\0'};
    int headerLines=5;
#if USE_KLEIN
    ++headerLines;
#endif
#if USE_SNAVELY
    ++headerLines;
#endif
    if(loopData.is_open())//remove the header lines
        for(int header=0; header<headerLines; ++header)
            loopData.getline(buffer, 300);
    else{
        cout<<"loopConstraints.txt file not opened!"<<endl;
        exit(-1);
    }

    unsigned int frameId[2]; //the frame_ids of two frames that are involved in a loop detection
    TooN::Matrix<7> inf7 = Identity(7);

    double resort[6];
    double sf2s;
    int matchCount;

    while(!loopData.eof())
    {
        loopData.getline(buffer, 300); //ground truth

        if(loopData.eof())
            break;
        sscanf(buffer, "%u %u %lf %lf %lf %lf %lf %lf",
               &frameId[0], &frameId[1], &resort[0], &resort[1],
                &resort[2], &resort[3], &resort[4], &resort[5]);
        loopData.getline(buffer, 300);// opencv 5 point

        loopData.getline(buffer, 300);// knn regression
        loopData.getline(buffer, 300);// knn regression + 2 View BA

        sscanf(buffer, "%d %lf %lf %lf %lf %lf %lf %lf",
               &matchCount, &sf2s, &resort[0], &resort[1],
                &resort[2], &resort[3], &resort[4], &resort[5]);
        assert(resort[0]!=0&&resort[1]!=0&&resort[2]!=0);

#if USE_KLEIN
        loopData.getline(buffer, 300);//use the 5 point algorithm results by Klein
#endif
#if USE_SNAVELY
        loopData.getline(buffer, 300);// snavely 5 point
#endif

        Eigen::Matrix3d Rf2s=roteu2ro(Eigen::Vector3d(resort[0], resort[1], resort[2]));
        g2o::Sim3 loopCond(Rf2s,Eigen::Vector3d(resort[3], resort[4], resort[5]), sf2s);

        loopConstraints.push_back(Constraint<g2o::Sim3, 7>(frameId[0], frameId[1], loopCond, inf7));
    }
    loopData.close();
}

std::vector<int> GetKFIndices(std::string keyFrameInfoDir)
{
    vector<string> keyFrameFiles =
            DUtils::FileFunctions::Dir(keyFrameInfoDir.c_str(), ".bin", true);
    vector<int> imgIds; // all IDs of keyframes
    imgIds.resize(keyFrameFiles.size());
    unsigned int count=0;
    for(; count<keyFrameFiles.size();++count)
    {
        std::size_t pos = keyFrameFiles[count].find("KeyFrame");
        int imgId=atoi(keyFrameFiles[count].substr(pos+8, 6).c_str());
        imgIds[count]=imgId;
    }
    assert(imgIds.size()==count);
    return imgIds;
}

void SaveKFIndices(const std::vector<int>imgIds, std::string outputPath)
{
    ofstream ccs((outputPath +"/cc.txt").c_str(), ios::out);
    for(unsigned int count=0; count<imgIds.size();++count)
        ccs<< imgIds[count] <<endl;
    ccs.close();
}

void LoadKFIndices(const string ccFile, std::vector<KeyFrame*> &vpKFs){
    vpKFs.clear();
    ifstream input(ccFile.c_str(), ifstream::in);
    if(!input.is_open())
    {
        cout<<"cannot open cc.txt"<<endl;
        exit(-1);
    }

    int nid= -1;
    int kfid =0;
    while(!input.eof()){
        input>>nid;
        if(!input.eof()){
            KeyFrame* curKF = new KeyFrame(nid, kfid);
            vpKFs.push_back(curKF);
            ++kfid;
        }
    }
    cout<<"head id and kfid "<<vpKFs.front()->mnId<<" "<< vpKFs.front()->mnFrameId<<
          " tail "<<vpKFs.back()->mnId<<" "<<vpKFs.back()->mnFrameId<<endl;//debug
    input.close();
}
void LoadKFPoses(string poseFile, std::vector<KeyFrame*>& vpKFs)
{
    ifstream poseData(poseFile.c_str(),ios::in);
    char buffer[256]={'\0'};
    int headerLines=2;
    if(poseData.is_open())//remove the header lines
        for(int header=0; header<headerLines; ++header)
            poseData.getline(buffer, 300);
    else{
        cout<<"framePoses.txt file not opened!"<<endl;
        exit(-1);
    }

    int kfid(-1);
    double resort[6];
    double tempTime;
    auto it= vpKFs.begin();
    while(!poseData.eof())
    {
        poseData.getline(buffer, 300); //ground truth

        if(poseData.eof())
            break;
        sscanf(buffer, "%d, %lf, %lf, %lf, %lf, %lf, %lf, %lf",
               &kfid, &tempTime, &resort[0], &resort[1],
                &resort[2], &resort[3], &resort[4], &resort[5]);

        Eigen::Matrix3d Rc2w = roteu2ro(Eigen::Vector3d(resort[0], resort[1], resort[2]));
        Sophus::SE3d Tc2w(Rc2w, Eigen::Vector3d(resort[3], resort[4], resort[5]));
        if(kfid == (*it)->mnId){
            (*it)->Tw2c= Tc2w.inverse();

            ++it;
        }
    }
    assert(it == vpKFs.end());
    poseData.close();
}


// ---------------------------------------------------------------------------
// given all key frame information files and the loop closure detection results by DBoW
// save EGs, cc, coords and tracks as described in 1DSfM of Kyle Wilson
//combine keyframe files produced by PTAMM KLT and DBoW2 loop closurer, save as required by 1DSfM
//note still one requirement not fulfilled: features in each image should be observed in a track
void SaveAsSfMInit(string keyFrameInfoDir, string loopPairsFile,
                   string loopConstraintFile, string loopCoordsFile, string loopTracksFile,
                   string coordsFile, string tracksFile)
{
    //load all frames
    std::vector<int> imgIds= GetKFIndices(keyFrameInfoDir);
    string outputPath = keyFrameInfoDir;
    SaveKFIndices(imgIds, outputPath);

    loopPairsFile=outputPath+"/"+loopPairsFile;
    loopConstraintFile=outputPath+"/"+loopConstraintFile;
    loopCoordsFile=outputPath+"/"+loopCoordsFile; // here may be repetition of some images, verify that the repetitions are the same
    loopTracksFile=outputPath+"/"+loopTracksFile;
    coordsFile=outputPath+"/"+coordsFile;
    tracksFile=outputPath+"/"+tracksFile;
    string EGsFile=outputPath+"/EGs.txt";
    string rotsFile=outputPath+"/rots.txt";

    map<unsigned int, MyKeyFrame> loopKeyFrames;
    ifstream loopPairs(loopPairsFile.c_str(), ios::in);
    int firstFrame, secondFrame;
    map<unsigned int, MyKeyFrame>::iterator mIt;
    while(loopPairs>>firstFrame )
    {
        loopPairs>>secondFrame;
        mIt=loopKeyFrames.find(firstFrame);
        if(mIt==loopKeyFrames.end())
            loopKeyFrames.insert(make_pair(firstFrame, MyKeyFrame(firstFrame)));
        else
            cout<< firstFrame <<" used again in loop closure!"<<endl;
        mIt=loopKeyFrames.find(secondFrame);
        if(mIt==loopKeyFrames.end())
            loopKeyFrames.insert(make_pair(secondFrame, MyKeyFrame(secondFrame)));
        else
            cout<< secondFrame <<" used again in loop closure!"<<endl;
    }
    loopPairs.close();

    //load loopCoordsFile into loopKeyFrames which is to be integrated into coords.txt
    LoadLoopCoords(loopCoordsFile, loopKeyFrames);
    cout<<"loopCoords.txt loaded and loading loopTracks.txt!"<<endl;
    _TrackMap loopTrackMap; // the point appearance tracks across images used in loop closure
    ifstream loopTracks(loopTracksFile.c_str(), ifstream::in);
    unsigned int exptMapSize;
    loopTracks>>exptMapSize;
    loopTracks>>loopTrackMap;
    loopTracks.close();
    assert(exptMapSize==loopTrackMap.size());
    cout<<"Connecting tracks in loopTracks.txt!"<<endl;
    //there may be some points observed in several images, connect them
    map<IdPair, unsigned int> obsMap;
    for(_TrackMap::iterator trackIt=loopTrackMap.begin(); trackIt!=loopTrackMap.end(); )
    {
        _TrackMap::iterator sniper;
        map<IdPair, unsigned int>::iterator obsIt[2];
        assert(trackIt->second.size()==2);
        IdPair cIt[2];
        cIt[0]=trackIt->second[0];
        cIt[1]=trackIt->second[1];

        obsIt[0]=obsMap.find(cIt[0]);
        obsIt[1]=obsMap.find(cIt[1]);
        if(obsIt[0]==obsMap.end()&&obsIt[1]==obsMap.end()){
            obsMap.insert(make_pair(cIt[0], trackIt->first));
            obsMap.insert(make_pair(cIt[1], trackIt->first));
            ++trackIt;
        }
        else
        {
            if(obsIt[0]==obsMap.end()){
                assert(obsIt[1]!=obsMap.end());
                sniper=loopTrackMap.find(obsIt[1]->second);
                assert(sniper!=loopTrackMap.end());
                obsMap.insert(make_pair(cIt[0], obsIt[1]->second));
                sniper->second.push_back(cIt[0]);
            }
            else if(obsIt[1]==obsMap.end()){
                assert(obsIt[0]!=obsMap.end());
                sniper=loopTrackMap.find(obsIt[0]->second);
                assert(sniper!=loopTrackMap.end());
                obsMap.insert(make_pair(cIt[1], obsIt[0]->second));
                sniper->second.push_back(cIt[1]);
            }
            else
            {
                cout<<"The drawback of KNNMatch() shown by a point used twice between two image matching!"<<endl;
                cout<<"The pair is:"<<cIt[0].frame_id<<" "<<cIt[0].point_id<<" "
                   <<cIt[1].frame_id<<" "<<cIt[1].point_id<<endl;
            }
            loopTrackMap.erase(trackIt++);
        }
    }
#if 1
    //assert in each track only one image id can appear
    set<unsigned int> appearedIDs;
    set<unsigned int>::iterator sIt;
    for(_TrackMap::const_iterator mIt=loopTrackMap.begin(); mIt!=loopTrackMap.end(); ++mIt)
    {
        appearedIDs.clear();
        for(unsigned int count=0; count<mIt->second.size(); ++count)
        {
            sIt=appearedIDs.find(mIt->second[count].frame_id);
            if(sIt!=appearedIDs.end())
                assert(0);
            else
                appearedIDs.insert(mIt->second[count].frame_id);
        }
    }
#endif
    obsMap.clear();
    cout<<"loading keyframes and writing the coords.txt file!"<<endl;
    vector<SE3<> > pose_vec;
    TooN::Vector<3> twinc;
    TooN::Matrix<3> Rw2c;
    _TrackMap trackMap; // the point appearance tracks across images
    ofstream coords(coordsFile.c_str(), ofstream::out | ofstream::trunc);
    MyKeyFrame curKeyFrame;
    char buffer[256]={'\0'};
    for(unsigned int frameCounter=0; frameCounter<imgIds.size(); ++frameCounter)
    {
        mIt=loopKeyFrames.find(imgIds[frameCounter]);
        sprintf(buffer, "%s/KeyFrame%06d.bin", keyFrameInfoDir.c_str(), imgIds[frameCounter]);
        LoadComboKeyFrame(buffer, Rw2c,twinc, trackMap, curKeyFrame);
        if(mIt!=loopKeyFrames.end()){// this frame is used for loop closure
            assert(curKeyFrame.imgID==mIt->first);
            mIt->second.increment=curKeyFrame.keys.size();
            curKeyFrame.Combine(mIt->second);
        }
        coords<<curKeyFrame;
        pose_vec.push_back(SE3<>(Rw2c,twinc));
    }
    coords.close();
    assert(pose_vec.size()==imgIds.size());
    cout<<"Maintaining the tracks and saving them in tracks.txt!"<<endl;
    //for each feature in the loop tracks, increment its feature id
    for(_TrackMap::iterator trackIt=loopTrackMap.begin(); trackIt!=loopTrackMap.end(); ++trackIt)
    {
        _Track::iterator tIt=trackIt->second.begin();

        for(; tIt!=trackIt->second.end();++tIt)
        {
            mIt=loopKeyFrames.find(tIt->frame_id);
            assert(mIt!=loopKeyFrames.end());
            assert(mIt->second.increment);
            tIt->point_id+=mIt->second.increment;
        }
    }
    //save tracks
    ofstream tracks(tracksFile.c_str(), std::ofstream::out | std::ofstream::trunc);
    tracks<<trackMap.size()+loopTrackMap.size()<<endl;
    tracks<<trackMap;
    tracks<<loopTrackMap;
    tracks.close();


    cout<<"Saving EGs.txt!"<<endl;
    vector<Constraint<SE3<>, 6> > constraints;
    LoadLoopConstraints(loopConstraintFile, constraints);
    SE3<> Ps2f;
    TooN::Matrix<3> Rs2f;
    TooN::Vector<3> Tsinf;
    ofstream EGs(EGsFile.c_str(), ofstream::out | ofstream::trunc);
    for(unsigned int count=1; count<pose_vec.size(); ++count)
    {
        Ps2f=pose_vec[count-1]*pose_vec[count].inverse();
        Rs2f=Ps2f.get_rotation().get_matrix();
        Tsinf=Ps2f.get_translation();
        Tsinf/=norm_2(Tsinf);
        EGs<<imgIds[count-1]<<" "<<imgIds[count]<<" "<<
                              Rs2f(0,0)<<" "<<Rs2f(0,1)<<" "<<Rs2f(0,2)<<" "<<
                              Rs2f(1,0)<<" "<<Rs2f(1,1)<<" "<<Rs2f(1,2)<<" "<<
                              Rs2f(2,0)<<" "<<Rs2f(2,1)<<" "<<Rs2f(2,2)<<" "<<
                              Tsinf[0]<<" "<<Tsinf[1]<<" "<<Tsinf[2]<<endl;
    }
    for(unsigned int count=0; count<constraints.size(); ++count)
    {
        Rs2f=constraints[count].mean.get_rotation().get_matrix();
        Tsinf=constraints[count].mean.get_translation();
        Tsinf/=norm_2(Tsinf);
        EGs<<constraints[count].trans_id2<<" "<<constraints[count].trans_id1<<" "<<
             Rs2f(0,0)<<" "<<Rs2f(0,1)<<" "<<Rs2f(0,2)<<" "<<
             Rs2f(1,0)<<" "<<Rs2f(1,1)<<" "<<Rs2f(1,2)<<" "<<
             Rs2f(2,0)<<" "<<Rs2f(2,1)<<" "<<Rs2f(2,2)<<" "<<
             Tsinf[0]<<" "<<Tsinf[1]<<" "<<Tsinf[2]<<endl;
    }
    EGs.close();
    //save rots.txt each line is i and Ri=Rw2i
    ofstream rots(rotsFile.c_str(), ofstream::out | ofstream::trunc);
    for(unsigned int count=0; count<pose_vec.size(); ++count)
    {
        Rs2f=pose_vec[count].get_rotation().get_matrix();
        rots<<imgIds[count]<<" "<<
              Rs2f(0,0)<<" "<<Rs2f(0,1)<<" "<<Rs2f(0,2)<<" "<<
              Rs2f(1,0)<<" "<<Rs2f(1,1)<<" "<<Rs2f(1,2)<<" "<<
              Rs2f(2,0)<<" "<<Rs2f(2,1)<<" "<<Rs2f(2,2)<<" "<<endl;
    }
    rots.close();
}

// ----------------------------------------------------------------------------

void SurfExtractor::operator() (const cv::Mat &im, 
                                vector<cv::KeyPoint> &keys, vector<vector<float> > &descriptors) const
{
    // extract surfs with opencv
    static cv::SURF surf_detector(400);

    surf_detector.extended = 0;

    keys.clear(); // opencv 2.4 does not clear the vector
    vector<float> plain;
    surf_detector(im, cv::Mat(), keys, plain);

    // change descriptor format
    const int L = surf_detector.descriptorSize();
    descriptors.resize(plain.size() / L);

    unsigned int j = 0;
    for(unsigned int i = 0; i < plain.size(); i += L, ++j)
    {
        descriptors[j].resize(L);
        std::copy(plain.begin() + i, plain.begin() + i + L, descriptors[j].begin());
    }
}

void GetAllKeyFrames(std::string keyFrameInfoDir, std::vector<KeyFrame*> & vpKFs)
{
    //get keyframe ids
    std::string ccFile = keyFrameInfoDir + "/cc.txt";
    LoadKFIndices(ccFile, vpKFs);
    LoadKFPoses(keyFrameInfoDir +"/framePoses.txt", vpKFs);
}

Eigen::Vector4d toScaleTrans(const g2o::Sim3 & se3q)
{
    Eigen::Vector4d v4;
    v4[0]= se3q.scale();
    v4.tail<3>()= se3q.translation();
    return v4;
}

#define DEBUG 1
void testDirectSim3Optimization(string directFile, bool bUseOneContraint=false)
{
    // optimize sim3 pose graph given loop constraints,
    // optimize scale +trans +rot
    DUtils::Profiler profiler;
    profiler.profile("sim3_direct");

    ofstream logStream(directFile);

    // Setup three optimizers
    g2o::SparseOptimizer optimizer;
    std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver =
            g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType> >();

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
                g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver)));
    optimizer.setAlgorithm(solver);

    vector<KeyFrame*> vpKFs;
    string keyFrameInfoDir=KeyFrameInfoDir;
    GetAllKeyFrames(keyFrameInfoDir, vpKFs);
    assert(vpKFs.front()->mnFrameId==0);
    unsigned int nMaxKFid = vpKFs.back()->mnFrameId;

    vector<Constraint<g2o::Sim3, 7 > > loopConnections;
    LoadLoopConstraints( keyFrameInfoDir+"/loopConstraints.txt",loopConnections);
    if(bUseOneContraint){
        // use only 1 loop constraint
        Constraint<g2o::Sim3, 7 > tempConstraint= loopConnections.front();
        loopConnections.clear();
        loopConnections.push_back(tempConstraint);
    }
    //map frame id to keyframe id
    map<int, int> frameid2kfid;
    for (auto it= vpKFs.begin();  it!= vpKFs.end(); ++it)
    {
        std::pair<map<int, int>::iterator, bool> res=frameid2kfid.insert(
                    std::pair<int, int>( (*it)->mnId, (*it)->mnFrameId));
        assert(res.second);
    }
    for(auto mit = loopConnections.begin(), mend=loopConnections.end(); mit!=mend; mit++)
    {
        map<int, int>::const_iterator iter=frameid2kfid.find(mit->trans_id1);
        assert(iter!= frameid2kfid.end());
        mit->trans_id1= iter->second;
        iter=frameid2kfid.find(mit->trans_id2);
        assert(iter!= frameid2kfid.end());
        mit->trans_id2= iter->second;
    }

    Eigen::Matrix<double,7,7> matLambdasim = Eigen::Matrix<double,7,7>::Identity();
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);

    // SET KEYFRAME VERTICES
    for(size_t i=0, iend=vpKFs.size(); i<iend;++i)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        vio::VertexSim3Expmap* vSim3 = new vio::VertexSim3Expmap();

        int nIDi = pKF->mnFrameId;

        Eigen::Matrix<double,3,3> Rcw = pKF->GetRotation();
        Eigen::Matrix<double,3,1> tcw = pKF->GetTranslation();
        g2o::Sim3 Siw(Rcw,tcw,1.0);
        vScw[nIDi] = Siw;

        vSim3->setEstimate(Siw);

        if(nIDi==0){

            vSim3->setFixed(true);
        }

        vSim3->setId(nIDi);
        vSim3->setMarginalized(false);
        optimizer.addVertex(vSim3);

    }
    // SET LOOP EDGES
    for(auto mit = loopConnections.begin(), mend=loopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = vpKFs[mit->trans_id1];
        const long unsigned int nIDi = pKF->mnFrameId;
        KeyFrame* pConnection = vpKFs[mit->trans_id2];
        assert(pConnection->mnFrameId == mit->trans_id2);
        assert(pKF->mnFrameId == mit->trans_id1);
        const long unsigned int nIDj = pConnection->mnFrameId;

        vio::EdgeSim3* esim = new vio::EdgeSim3();
        esim->setVertex(1, optimizer.vertex(nIDj));
        esim->setVertex(0, optimizer.vertex(nIDi));
        esim->setMeasurement(mit->mean);
        esim->information() = matLambdasim;
        optimizer.addEdge(esim);

    }
    //show loop edges, for debugging
    //            cout<<"loop edges id1, id2, scale, qxyzw, txyz"<<endl;
    //            for(int jack=0; jack<loopConnections.size(); ++jack){
    //                g2o::Sim3  sim=loopConnections[jack].mean;
    //                cout<<loopConnections[jack].trans_id1<<" "<<loopConnections[jack].trans_id2<<" "<<
    //                      sim.scale()<<" "<<sim.rotation().coeffs().transpose()<<" "<< sim.translation().transpose()<<endl;
    //            }
    // SET NORMAL EDGES
    for(size_t i=1, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];
        int nIDi = pKF->mnFrameId;
        g2o::Sim3 Swi = vScw[nIDi].inverse();
        KeyFrame* pParentKF = vpKFs[i-1];

        // Spanning tree edge
        int nIDj = pParentKF->mnFrameId;

        g2o::Sim3 Sjw = vScw[nIDj];
        g2o::Sim3 Sji = Sjw * Swi;


        vio::EdgeSim3* e = new vio::EdgeSim3();
        e->setVertex(1, optimizer.vertex(nIDj));
        e->setVertex(0, optimizer.vertex(nIDi));
        e->setMeasurement(Sji);
        e->information() = matLambdasim;
        optimizer.addEdge(e);

    }

    /// optimize all states, scales, trans and rotations
    // using only this optimization often did not get good results
    optimizer.initializeOptimization();
    optimizer.optimize(100);


    logStream<<"% sim3 optimization result: kf id, sw2i, scaled tiinw, ri2w(qxyzw):"<<endl;

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnFrameId;
        g2o::Sim3 CorrectedSiw;

        vio::VertexSim3Expmap* vSim3 = static_cast<vio::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        CorrectedSiw=vSim3->estimate();

        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        //[R t/s;0 1]
        Sophus::SE3d Tiw(CorrectedSiw.rotation(), CorrectedSiw.translation()/CorrectedSiw.scale());

        logStream<<pKFi->mnId<<" "<< CorrectedSiw.scale()<< " "<<
               vCorrectedSwc[nIDi].translation().transpose()<< " "<<
               //- (Tiw.rotationMatrix().transpose()* Tiw.translation()).transpose()<<" "<< // the same as the above expression
               vCorrectedSwc[nIDi].rotation().coeffs().transpose()<<endl;

        pKFi->SetPose(Tiw);
    }

    logStream.close();
    cout << "saved output file "<< directFile << std::endl;
    profiler.stop();
    cout << endl << "Execution time:" << endl
         << " sim3 direct optimization: " << profiler.getMeanTime("sim3_direct") * 1e3
         << " ms" << endl;
}


// if bStepwise is false, then this function is identical to testDirectSim3Optimization()
void testStepwiseSim3Optimization(string outputFile, bool bUseOneContraint= false, int num_optimizer =2, bool bStepwise = true){
    // optimize sim3 pose graph given loop constraints,
    // first solve for scale with SVD, then optimize scale, second optimize scale + trans, optionally third optimize scale +trans +rot
    // optionally use SVD to solve for trans

    // prepare profiler to measure times
    DUtils::Profiler profiler;
    profiler.profile("tot_optim");

    ofstream logStream(outputFile);

    // Setup optimizers
    num_optimizer = bStepwise? num_optimizer: 3;
    g2o::SparseOptimizer* optimizer = new g2o::SparseOptimizer[num_optimizer];
    for (int jack =0; jack< num_optimizer; ++jack){
        std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver =
                g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType> >();

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
                    g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver)));

        optimizer[jack].setAlgorithm(solver);
    }
    vector<KeyFrame*> vpKFs;
    string keyFrameInfoDir= KeyFrameInfoDir;
    GetAllKeyFrames(keyFrameInfoDir, vpKFs);
    assert(vpKFs.front()->mnFrameId==0);
    unsigned int nMaxKFid = vpKFs.back()->mnFrameId;

    vector<Constraint<g2o::Sim3, 7 > > loopConnections;
    LoadLoopConstraints( keyFrameInfoDir+"/loopConstraints.txt",loopConnections);
    if(bUseOneContraint){// use only 1 loop constraint
        Constraint<g2o::Sim3, 7 > tempConstraint= loopConnections.front();
        loopConnections.clear();
        loopConnections.push_back(tempConstraint);
    }
    //map frame id to keyframe id
    map<int, int> frameid2kfid;
    for (auto it= vpKFs.begin();  it!= vpKFs.end(); ++it)
    {
        std::pair<map<int, int>::iterator, bool> res=frameid2kfid.insert(
                    std::pair<int, int>( (*it)->mnId, (*it)->mnFrameId));
        assert(res.second);
    }
    for(auto mit = loopConnections.begin(), mend=loopConnections.end(); mit!=mend; mit++)
    {
        map<int, int>::const_iterator iter=frameid2kfid.find(mit->trans_id1);
        assert(iter!= frameid2kfid.end());
        mit->trans_id1= iter->second;
        iter=frameid2kfid.find(mit->trans_id2);
        assert(iter!= frameid2kfid.end());
        mit->trans_id2= iter->second;
    }

    Eigen::Matrix<double,1,1> matLambdas = Eigen::Matrix<double,1,1>::Identity();
    Eigen::Matrix<double,4,4> matLambdast = Eigen::Matrix<double,4,4>::Identity();
    Eigen::Matrix<double,7,7> matLambdasim = Eigen::Matrix<double,7,7>::Identity();
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);

    // SET KEYFRAME VERTICES
    for(size_t i=0, iend=vpKFs.size(); i<iend;++i)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        vio::G2oVertexScale* vS = new vio::G2oVertexScale();
        vio::G2oVertexScaleTrans* vST = new vio::G2oVertexScaleTrans();
        vio::VertexSim3Expmap* vSim3 = NULL;
        if(num_optimizer ==3)
            vSim3 = new vio::VertexSim3Expmap();

        int nIDi = pKF->mnFrameId;

        Eigen::Matrix<double,3,3> Rcw = pKF->GetRotation();
        Eigen::Matrix<double,3,1> tcw = pKF->GetTranslation();
        g2o::Sim3 Siw(Rcw,tcw,1.0);
        vScw[nIDi] = Siw;
        vS->setEstimate(Siw.scale());
        vST->setEstimate(toScaleTrans(Siw));
        vST->Rw2i= Sophus::SO3d(Rcw);
        if(num_optimizer ==3)
            vSim3->setEstimate(Siw);

        if(nIDi==0){
            vS->setFixed(true);
            vST->setFixed(true);
            if(num_optimizer==3)
                vSim3->setFixed(true);
        }

        vS->setId(nIDi);
        vS->setMarginalized(false);
        vST->setId(nIDi);
        vST->setMarginalized(false);

        optimizer[0].addVertex(vS);
        optimizer[1].addVertex(vST);
        if(num_optimizer ==3){
            vSim3->setId(nIDi);
            vSim3->setMarginalized(false);
            optimizer[2].addVertex(vSim3);
        }
    }
    // SET LOOP EDGES
    for(auto mit = loopConnections.begin(), mend=loopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = vpKFs[mit->trans_id1];
        const long unsigned int nIDi = pKF->mnFrameId;
        KeyFrame* pConnection = vpKFs[mit->trans_id2];
        assert(pConnection->mnFrameId == mit->trans_id2);
        assert(pKF->mnFrameId == mit->trans_id1);
        const long unsigned int nIDj = pConnection->mnFrameId;

        vio::G2oEdgeScale* es = new vio::G2oEdgeScale();
        es->setVertex(1, optimizer[0].vertex(nIDj));
        es->setVertex(0, optimizer[0].vertex(nIDi));
        es->setMeasurement(mit->mean.scale());
        es->information() = matLambdas;
        optimizer[0].addEdge(es);

        vio::G2oEdgeScaleTrans* est = new vio::G2oEdgeScaleTrans();
        est->setVertex(1, optimizer[1].vertex(nIDj));
        est->setVertex(0, optimizer[1].vertex(nIDi));
        est->setMeasurement(toScaleTrans(mit->mean));
        est->information() = matLambdast;
        optimizer[1].addEdge(est);

        if(num_optimizer==3){
            vio::EdgeSim3* esim = new vio::EdgeSim3();
            esim->setVertex(1, optimizer[2].vertex(nIDj));
            esim->setVertex(0, optimizer[2].vertex(nIDi));
            esim->setMeasurement(mit->mean);
            esim->information() = matLambdasim;
            optimizer[2].addEdge(esim);
        }
    }

    // SET NORMAL EDGES
    for(size_t i=1, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];
        int nIDi = pKF->mnFrameId;
        g2o::Sim3 Swi = vScw[nIDi].inverse();
        KeyFrame* pParentKF = vpKFs[i-1];

        // Spanning tree edge
        int nIDj = pParentKF->mnFrameId;

        g2o::Sim3 Sjw = vScw[nIDj];
        g2o::Sim3 Sji = Sjw * Swi;

        vio::G2oEdgeScale* es = new vio::G2oEdgeScale();
        es->setVertex(1, optimizer[0].vertex(nIDj));
        es->setVertex(0, optimizer[0].vertex(nIDi));
        es->setMeasurement(Sji.scale());
        es->information() = matLambdas;
        optimizer[0].addEdge(es);

        vio::G2oEdgeScaleTrans* est = new vio::G2oEdgeScaleTrans();
        est->setVertex(1, optimizer[1].vertex(nIDj));
        est->setVertex(0, optimizer[1].vertex(nIDi));
        est->setMeasurement(toScaleTrans(Sji));
        est->information() = matLambdast;
        optimizer[1].addEdge(est);
        if(num_optimizer==3){
            vio::EdgeSim3* e = new vio::EdgeSim3();
            e->setVertex(1, optimizer[2].vertex(nIDj));
            e->setVertex(0, optimizer[2].vertex(nIDi));
            e->setMeasurement(Sji);
            e->information() = matLambdasim;
            optimizer[2].addEdge(e);
        }
    }
    if(bStepwise){
        /// optimize scales
        // (1) SVD method to solve for scales, Ax=0; x= [s_0^0, s_0^1; s_0^2; ..., s_0^{N-1}], where s_0^j is the scale of frame j,
        // the solution is the last column of V of SVD of A
        profiler.profile("scale_dlt");
        int num_constraints = loopConnections.size() + vpKFs.size()-1;
        int num_epoch = vpKFs.size();
        Eigen::MatrixXd sm1( num_constraints, num_epoch);
        sm1.setZero();
        for(size_t jack=1; jack< vpKFs.size(); ++jack)
        {
            sm1(jack-1, jack-1)= 1;
            sm1(jack-1, jack)= -1;
        }
        for (size_t jack= 0; jack< loopConnections.size(); ++jack){
            sm1(jack-1 + vpKFs.size(), loopConnections[jack].trans_id1)= loopConnections[jack].mean.scale();
            sm1(jack-1 + vpKFs.size(), loopConnections[jack].trans_id2)= -1;
        }

        JacobiSVD<MatrixXd> svd(sm1, ComputeThinV);
        assert(svd.matrixV().cols()== num_epoch && svd.matrixV().rows()== num_epoch);
        Eigen::Matrix<double, Eigen::Dynamic, 1> S=svd.singularValues();
        //            cout<<"singular values "<<S.transpose() <<endl;
        if(S.coeff(0)*5e-4 > S.coeff(num_epoch-1))
            cout<<"Warning possible unsable result by SVD"<<endl;

        Eigen::Matrix<double, Eigen::Dynamic, 1> allScales(num_epoch, 1);
        allScales= svd.matrixV().block(0, num_epoch-1, num_epoch, 1);
        allScales= allScales/allScales[0];

        //            cout<<endl<<"SVD estimates:"<<endl;
        //            cout<<allScales<<endl;

        // update scale estimates due to SVD
        for(size_t i=0;i<vpKFs.size();++i)
        {
            KeyFrame* pKFi = vpKFs[i];
            const int nIDi = pKFi->mnFrameId;

            vio::G2oVertexScale* vS = static_cast<vio::G2oVertexScale*>(optimizer[0].vertex(nIDi));
            vS->setEstimate( allScales[i]);

            vio::G2oVertexScaleTrans* vST = static_cast<vio::G2oVertexScaleTrans*>(optimizer[1].vertex(nIDi));
            Eigen::Vector4d stw2i =  vST->estimate();
            stw2i[0]= allScales[i];
            vST->setEstimate(stw2i);
        }
        profiler.stop("scale_dlt");
        // (2) unconstrained optimize of scales, often not work when used as
        // the foremost scale solver because g2o deos not support constrained optimization
        // this is not necessary when SVD is used to solve for scales, but it can refine scale to some extent
        /*  optimizer[0].initializeOptimization();
optimizer[0].optimize(200);
#if DEBUG
debug<<"scale optimization results: kf id, sw2i:"<<endl;
#endif
// update scale estimates due to g2o
for(size_t i=0;i<vpKFs.size();++i)
{
    KeyFrame* pKFi = vpKFs[i];
    const int nIDi = pKFi->mnFrameId;
    vio::G2oVertexScale* vS = static_cast<vio::G2oVertexScale*>(optimizer[0].vertex(nIDi));
    double sw2i =  vS->estimate();

    vio::G2oVertexScaleTrans* vST = static_cast<vio::G2oVertexScaleTrans*>(optimizer[1].vertex(nIDi));
    Eigen::Vector4d v4 = vST->estimate();
    v4[0]= sw2i;
#if DEBUG
    debug<< nIDi<<" "<< sw2i<<endl;
#endif
    vST->setEstimate(v4);

    allScales[i] = sw2i; // used in solving for translation by SVD
}*/

        ///trans + scale optimization
        profiler.profile("scale_trans");
        // (1) SVD solve for trans while fixing scales Ax=b. This method is not well founded.
        // It often diverges given poor initial values for scales. Empirically,
        // it is unnecessary for obtaining the final solution compared to optimizing scale+trans with g2o
        /*Eigen::MatrixXd sm2( num_constraints*3, num_epoch*3);
sm2.setZero();
Eigen::Matrix<double, Eigen::Dynamic, 1> bn(num_constraints*3,1);
bn.setZero();

for(size_t jack=1; jack< vpKFs.size(); ++jack)
{
    Eigen::Matrix3d Ri2j= vpKFs[jack]->GetRotation()*(vpKFs[jack-1]->GetRotation().transpose());
    sm2.block((jack-1)*3, (jack-1)*3,3,3)= - allScales[jack]/ allScales[jack-1]*Ri2j;
    sm2.block((jack-1)*3, jack*3,3,3)= Eigen::Matrix3d::Identity();
    bn.block((jack-1)*3, 0,3,1) = vpKFs[jack]->GetTranslation() -
            allScales[jack]/ allScales[jack-1]*Ri2j*vpKFs[jack-1]->GetTranslation();
}

for (size_t jack= 0; jack< loopConnections.size(); ++jack){
    int iota =loopConnections[jack].trans_id1;
    int jude =loopConnections[jack].trans_id2;
    Eigen::Matrix3d Ri2j= vpKFs[jude]->GetRotation()*(vpKFs[iota]->GetRotation().transpose());
    sm2.block((jack-1 + vpKFs.size())*3, loopConnections[jack].trans_id1*3,3,3)=
            - allScales[jude]/ allScales[iota]*Ri2j;
    sm2.block((jack-1 + vpKFs.size())*3, loopConnections[jack].trans_id2*3,3,3)= Eigen::Matrix3d::Identity();

    bn.block((jack-1 + vpKFs.size())*3, 0,3,1) = loopConnections[jack].mean.translation();
}

Eigen::Matrix<double, Eigen::Dynamic, 1> delta(3*num_epoch,1);
//qr is more stable than svd, but still suffer instablity
delta = sm2.colPivHouseholderQr().solve(bn);
bool is_sol_exist = (sm2*delta).isApprox(bn, 1e-8);
cerr<<"is solution exist for trans:"<< is_sol_exist<<endl;

//            debug<<"x"<<endl<<delta<<endl;
Eigen::Matrix<double, Eigen::Dynamic, 3> all_tiinw(num_epoch,3);
for(size_t jack=0; jack< vpKFs.size(); ++jack)
{
    Eigen::Vector3d tiinw= - vpKFs[jack]->GetRotation().transpose()*
    delta.block(jack*3, 0,3,1)/allScales[jack];
    all_tiinw.block(jack, 0, 1, 3) = tiinw.transpose();
}
debug<<"svd scaled translation tiinw:"<<endl<<all_tiinw<<endl;
// update translation estimates due to SVD on trans
for(size_t i=0;i<vpKFs.size();++i)
{
    KeyFrame* pKFi = vpKFs[i];
    const int nIDi = pKFi->mnFrameId;

    vio::G2oVertexScaleTrans* vST = static_cast<vio::G2oVertexScaleTrans*>(optimizer[1].vertex(nIDi));
    Eigen::Vector4d stw2i =  vST->estimate();
    stw2i[0]= allScales[i];
    stw2i.tail<3>()= delta.block(i*3, 0,3,1);
    vST->setEstimate(stw2i);
}*/

        // (2) optimize scale + trans with g2o
        optimizer[1].initializeOptimization();
        optimizer[1].optimize(100);

        profiler.stop("scale_trans");
    }
    if(num_optimizer ==3){
        // update scale and translation estimates due to g2o
        for(size_t i=0;i<vpKFs.size();++i)
        {
            KeyFrame* pKFi = vpKFs[i];
            const int nIDi = pKFi->mnFrameId;

            vio::G2oVertexScaleTrans* vST = static_cast<vio::G2oVertexScaleTrans*>(optimizer[1].vertex(nIDi));
            Eigen::Vector4d stw2i =  vST->estimate();
            g2o::Sim3 CorrectedSiw(vST->Rw2i.unit_quaternion(), stw2i.tail<3>(), stw2i[0]);

            vio::VertexSim3Expmap* vSim3 = static_cast<vio::VertexSim3Expmap*>(optimizer[2].vertex(nIDi));
            vSim3->setEstimate(CorrectedSiw);
        }

        profiler.profile("sim3_optim");
        /// optimize all states, scales, trans and rotations
        // using only this optimization often did not get good results
        optimizer[2].initializeOptimization();
        optimizer[2].optimize(100);
        profiler.stop("sim3_optim");
    }

    logStream<<"% sim3 optimization result: kf frameid, sw2i, scaled tiinw, ri2w(qxyzw):"<<endl;
    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnFrameId;
        g2o::Sim3 CorrectedSiw;
        if(num_optimizer ==3){
            vio::VertexSim3Expmap* vSim3 = static_cast<vio::VertexSim3Expmap*>(optimizer[2].vertex(nIDi));
            CorrectedSiw=vSim3->estimate();
        }else{
            vio::G2oVertexScaleTrans* vST = static_cast<vio::G2oVertexScaleTrans*>(optimizer[1].vertex(nIDi));
            Eigen::Vector4d stw2i =  vST->estimate();
            CorrectedSiw= g2o::Sim3(vST->Rw2i.unit_quaternion(), stw2i.tail<3>(), stw2i[0]);
        }
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        //[R t/s;0 1]
        Sophus::SE3d Tiw(CorrectedSiw.rotation(), CorrectedSiw.translation()/CorrectedSiw.scale());

        logStream<<pKFi->mnId<<" "<< CorrectedSiw.scale()<< " "<<
               vCorrectedSwc[nIDi].translation().transpose()<< " "<<
               //- (Tiw.rotationMatrix().transpose()* Tiw.translation()).transpose()<<" "<< // the same as the above expression
               vCorrectedSwc[nIDi].rotation().coeffs().transpose()<<endl;

        pKFi->SetPose(Tiw);
    }

    logStream.close();
    cout << "saved output file "<< outputFile << std::endl;
    profiler.stop("tot_optim");
    cout << endl << "Execution time:" << endl
         << " stepwise optimization: " << profiler.getMeanTime("tot_optim") * 1e3<< endl
         << " scale svd: " << profiler.getMeanTime("scale_dlt") * 1e3<<endl
         << " scale_trans optimization: " << profiler.getMeanTime("scale_trans") * 1e3<<endl
         << " sim3 optimization: " << profiler.getMeanTime("sim3_optim") * 1e3
         << " ms" << endl;
}

// estimate similarity transform S_{w_2}^{w_1} between the trainPoses {T_i^{w_1}} and the queryPoses {T_i^{w_2}},
// both are of same length
// output the transformed query poses with S_{w_2}^{w_1}
Matrix4d estimateSimilarityTransform(const std::vector<Sophus::SE3d> &queryPoses, const std::vector<Sophus::SE3d>& trainPoses,
                                     std::vector<Sophus::SE3d> & alignedPoses, bool bOnlyScale = false)
{
    // solve for scales and translation and rotation only using positions
    MatrixXd mTrainPoints(3, trainPoses.size()), mQueryPoints(3, queryPoses.size());
    for(size_t jack=0; jack< trainPoses.size(); ++jack)
    {
        mTrainPoints.col(jack)= trainPoses[jack].translation();
        mQueryPoints.col(jack)= queryPoses[jack].translation();
    }
    Matrix4d S221= Matrix4d::Identity();

    if(bOnlyScale){
        // method 1 works for KITTI dataset because the initial gt position and orientation are 0
        vector<std::pair<double, double > > trainExtremes(3,std::make_pair(-1e8, 1e8)),
                queryExtremes(3, std::make_pair(-1e8, 1e8));
        //max and mins
        int dominantAxis[3]= {1,0,1}; //of x,y,z, x and z are dominant
        for(size_t jack=0; jack< trainPoses.size(); ++jack)
        {
            for(size_t zinc=0; zinc< 3; ++zinc){
                if(trainExtremes[zinc].first < mTrainPoints.col(jack)[zinc])
                    trainExtremes[zinc].first = mTrainPoints.col(jack)[zinc];
                if(trainExtremes[zinc].second > mTrainPoints.col(jack)[zinc])
                    trainExtremes[zinc].second = mTrainPoints.col(jack)[zinc];

                if(queryExtremes[zinc].first < mQueryPoints.col(jack)[zinc])
                    queryExtremes[zinc].first = mQueryPoints.col(jack)[zinc];
                if(queryExtremes[zinc].second > mQueryPoints.col(jack)[zinc])
                    queryExtremes[zinc].second = mQueryPoints.col(jack)[zinc];
            }
        }
        double scaleFactors[3];
        double totalScales =0;
        int totalAxes =0;
        for(size_t zinc=0; zinc< 3; ++zinc){
            scaleFactors[zinc] = (trainExtremes[zinc].first - trainExtremes[zinc].second)/
                    (queryExtremes[zinc].first - queryExtremes[zinc].second);;
            if(dominantAxis[zinc]){
                totalScales += scaleFactors[zinc];
                totalAxes ++;
            }
        }

        double scaleFactor = totalScales/totalAxes;
        cout<<"scales:"<< scaleFactors[0] << " "<< scaleFactors[1]
           << " " <<scaleFactors[2] << " average scale "<< scaleFactor<<endl;
        S221.topLeftCorner<3,3>() *= scaleFactor;
    }
    else{
        // method 2 umeyama works for general cases
        S221 = Eigen::umeyama(mQueryPoints, mTrainPoints,true);
    }
    // TODO: optimize scale and translation and rotation with a nonlinear optimizer

    // transform query poses
    alignedPoses = queryPoses;
    Matrix3d tempMat = S221.topLeftCorner<3,3>();
    Quaterniond q221(tempMat);
    q221.normalize();
    for(size_t jack= 0; jack< trainPoses.size(); ++ jack){

        alignedPoses[jack].setQuaternion( q221*queryPoses[jack].unit_quaternion());
        Vector4d v4;
        v4.head<3>()= queryPoses[jack].translation();
        v4[3]=1;
        v4 = S221*v4;
        alignedPoses[jack].translation()=v4.head<3>();
    }
    return S221;
}

// ---------------------------------------------------------------------------
// read in the ground truth of the kITTI odometry, each line is 12 elements of 3x4 Pc2w in row major order
// vTw2c output SE3d poses from the ground truth
void readKITTIPoseFile
(const string poseFile, std::vector<Sophus::SE3d> & vTc2w)
{
    vTc2w.clear();
    ifstream dataptr(poseFile.c_str());
    assert(!dataptr.fail());

    Eigen::Matrix<double,4,4> transMat;
    double precursor=0;
    int lineNum=0;
    while(!dataptr.eof()){
        dataptr>>precursor;
        if(dataptr.fail())
            break;
        transMat.setZero();
        transMat.data()[0]=precursor;
        for (int j=1; j<12; ++j)
            dataptr>>transMat.data()[j];
        ++lineNum;
        vTc2w.push_back(Sophus::SE3d(transMat.transpose()));
        //getline(dataptr, tempStr);
    }
    dataptr.close();
    cout<<"Num of lines:"<<lineNum<<endl;
}

// ---------------------------------------------------------------------------
// read in the computed solutions of Tc2w,
// assume its header is a comment line starting with % character
// vTw2c output SE3d poses, and the id of keyframes in the image sequence (0 based)
// lineFormat: each line is kf frameid, sw2i, scaled tiinw, ri2w(qxyzw)
void readOptimizedSim3PoseFile
(const string poseFile, std::vector<Sophus::SE3d> & vTc2w, std::vector<int> & vFrameIds)
{
    vTc2w.clear();
    ifstream dataptr(poseFile.c_str());
    assert(!dataptr.fail());
    string tempStr;

    getline(dataptr, tempStr); // remove the first comment line

    Eigen::Quaterniond transQuat;
    Eigen::Vector3d transTrans;
    int precursor=0;
    double tempScale;
    int lineNum=0;
    while(!dataptr.eof()){
        dataptr>>precursor;
        if(dataptr.fail())
            break;
        vFrameIds.push_back(precursor);
        dataptr>>tempScale;// discard scale too
        for (int j=0; j<3; ++j)
            dataptr>>transTrans.data()[j];
        dataptr>>transQuat.x() >>transQuat.y() >>transQuat.z() >>transQuat.w();
        ++lineNum;
        vTc2w.push_back(Sophus::SE3d(transQuat, transTrans));
        //getline(dataptr, tempStr);
    }
    dataptr.close();

    cout<<"Num of lines:"<<lineNum<<endl;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

int main(int argc, char  * argv[])
{
    if(argc < 2)
    {
        cout <<"Usage: kitti_surf <case num>" << endl;
        cout <<"possible case numbers are RUN_DBOW2=0, SIMPLE_TEST_CONSTRAINT, COMPUTE_CONSTRAINTS, "<<endl<<
             "OPTIMIZE_SIM3, ALIGN_TRAJECTORIES_ORIGINAL, ALIGN_TRAJECTORIES_OPTIMIZED, TEST_BAL_OPTIMIZER" << endl;
        cout <<"To comnpare the performance of stepwise vs direct pose graph optimization, simply run ./kitti_surf 3"<<endl;
        cout <<"after it is done, you may use gnuplot to visualize the results in data/intermediate folder"<< endl;
        cout << "the corresponding command is "<<endl <<"plot \"stepwise_2solvers.txt\" using 3:5 lc rgb \"#FF0000\" "
             <<" title 'stepwise' w lines, \"direct_embedded.txt\" using 3:5 lc rgb \"#0000FF\" title 'direct' w lines, "
            << "\"StepwiseMapAfterBA6.txt\" using 2:4 lc rgb \"#00FF00\" title 'stepwise traj after BA on corrected map' w lines, "
             << "\"framePoses.txt\" using 6:8 lc rgb \"#000000\" title 'monocular VO output' w lines" <<endl;
        cout <<endl<<" Note framePoses.txt is within data folder, you may need to copy it to the intermediate folder"<< endl;
        return 0;
    }
    demoDetector<Surf64Vocabulary, Surf64LoopDetector, FSurf64::TDescriptor>
            demo(VOC_FILE, IMAGE_DIR, POSE_FILE, IMAGE_W, IMAGE_H);
    cv::Mat cameraMatrix(3,3,cv::DataType<double>::type);

    cameraMatrix = (Mat_<double>(3,3) <<MyKeyFrame::m_f ,0, MyKeyFrame::m_px, 0, MyKeyFrame::m_f, MyKeyFrame::m_py,0,0,1);

    int caseNum = atoi(argv[1]);

    enum TEST_CASE {RUN_DBOW2=0, SIMPLE_TEST_CONSTRAINT, COMPUTE_CONSTRAINTS,
         OPTIMIZE_SIM3, ALIGN_TRAJECTORIES_ORIGINAL, ALIGN_TRAJECTORIES_OPTIMIZED, TEST_BAL_OPTIMIZER} experim;
    experim= static_cast<TEST_CASE>(caseNum);//OPTIMIZE_SIM3;// COMPUTE_CONSTRAINTS; //ALIGN_TRAJECTORIES_OPTIMIZED;

    try
    {
        if(experim==RUN_DBOW2){
            // given the output of sequential PTAMM stored in format of a sequence of keyframes,
            // detect the loop closures with DBow2, save to loopPairs.txt
            // then compute the loop constraints based on several methods
            SurfExtractor extractor;
            demo.keyFrameInfoDir=KeyFrameInfoDir;
            demo.run("SURF64", extractor);
            demo.saveLoops("loopPairs.txt");
            vector<int> kfids=GetKFIndices(demo.keyFrameInfoDir);
            SaveKFIndices(kfids, demo.keyFrameInfoDir);
            demo.computeConstraints(demo.vDetectionResults, "loopConstraints.txt", cameraMatrix);
        }
        else if( experim==SIMPLE_TEST_CONSTRAINT){
            // a simple demo test of computeConstraint
            demo.keyFrameInfoDir=KeyFrameInfoDir;

            vector<DetectionResult> framePairs;
            framePairs.resize(2);
            framePairs[0].query=12;
            framePairs[0].match=4458;
            framePairs[0].status = LOOP_DETECTED;

            framePairs[1].query=0;
            framePairs[1].match=4450;
            framePairs[1].status = LOOP_DETECTED;
            demo.computeConstraints(framePairs, "testPairs.txt", cameraMatrix);
        }
        else if(experim==COMPUTE_CONSTRAINTS){
            // given the output of sequential PTAMM stored in format of a sequence of keyframes,
            // and saved loop closures in loopPairs.txt
            // compute the loop constraints based on several methods
            // save the results in SfMInit accepted files for corroboration
            string keyFrameInfoPath=KeyFrameInfoDir;
            demo.keyFrameInfoDir=keyFrameInfoPath;
            demo.loadLoops("loopPairs.txt");
            demo.computeConstraints(demo.vDetectionResults, "loopConstraints.txt", cameraMatrix, "loopCoords.txt", "loopTracks.txt");

            //optionally, combine all keyframes information and loop closure information into EGs, cc, tracks and coords of 1DSfM Init by Kyle Wilson
            SaveAsSfMInit(keyFrameInfoPath, "loopPairs.txt", "loopConstraints.txt", "loopCoords.txt", "loopTracks.txt",   //assume all frames sharing identical intrinsic parameters
                          "coords.txt", "tracks.txt");
        }
        else if(experim == OPTIMIZE_SIM3)
        {// given detected loops, and computed loop constraints, and the initial keyframe poses estimated by a monocular VO,
            // optimize the whole graph of poses over similarity transforms
            bool bUseOneContraint= false;
            string outputPath = "../data/intermediate/";
            string outputFile= outputPath + "stepwise_2solvers.txt";
            testStepwiseSim3Optimization(outputFile, bUseOneContraint);

            outputFile= outputPath + "stepwise_3solvers.txt";
            testStepwiseSim3Optimization(outputFile, bUseOneContraint, 3, true);

            //generate BAL format file and optimize with g2o, but the performance is very bad for now
            string stepwiseBAL = outputPath + "MapAfterStepwisePGO.txt";
            figureKITTIBA(KeyFrameInfoDir, outputFile, stepwiseBAL,RobotVision::Sim3<>(), 1);

            // call a modified version of ba_demo of g2o so that the optimized values can be output into BAL format
            string sMapAfterBA = outputPath + "StepwiseMapAfterBA6.txt";
            string sStatistic = outputPath + "balstats.txt";

            char *cstr1 = new char[sMapAfterBA.length() + 1];
            strcpy(cstr1, sMapAfterBA.c_str());

            char *cstr2 = new char[sStatistic.length() + 1];
            strcpy(cstr2, sStatistic.c_str());

            char *cstr3 = new char[stepwiseBAL.length() + 1];
            strcpy(cstr3, stepwiseBAL.c_str());

            char* argument[]={"./ba_demo", "-i", "100", "-pcg", "-v",
                              "-o", cstr1,
                              "-stats", cstr2,
                              cstr3};
            int argc = sizeof(argument);
            //TODO: add loop constraints
            ba_demo(argc,argument);

            delete [] cstr1;
            delete [] cstr2;
            delete [] cstr3;

            outputFile= outputPath + "direct_embedded.txt";
            testStepwiseSim3Optimization(outputFile, bUseOneContraint,2, false);

            outputFile= outputPath + "direct_firstedge_embedded.txt";
            testStepwiseSim3Optimization(outputFile, true, 2, false);

            // this result is the same as direct_embedded.txt
            string directFile =outputPath + "direct_pure.txt";
            testDirectSim3Optimization(directFile, bUseOneContraint);

            string directBAL = outputPath + "MapAfterDirectPGO.txt";
            figureKITTIBA(KeyFrameInfoDir, directFile, directBAL,RobotVision::Sim3<>(), 1);

            // this result is the same as direct_firstedge_embedded.txt
            directFile =outputPath + "direct_firstedge_pure.txt";
            testDirectSim3Optimization(directFile, true);

        }else if(experim == ALIGN_TRAJECTORIES_ORIGINAL || experim == ALIGN_TRAJECTORIES_OPTIMIZED)
        {
            bool saveOriginalMap = false; // save the output of sequential PTAM or that after loop closure
            if(experim == ALIGN_TRAJECTORIES_ORIGINAL)
                saveOriginalMap = true;
            // given optimized poses which compose a trajectory,
            // estimate the similarity transform between the ground truth and optimized trajectory
            string outputPath = "../data/intermediate/";
#if 1
            string sim3OptimizedFile = outputPath + "direct_pure.txt";
            string outputFile= outputPath + "LC7map_direct.txt";
#else
            string sim3OptimizedFile = outputPath + "stepwise_2solvers.txt";
            string outputFile= outputPath + "LC7map_stepwise2.txt";
#endif
            if(saveOriginalMap){
                string keyFrameInfoDir=KeyFrameInfoDir;
                sim3OptimizedFile = keyFrameInfoDir +"/framePoses.txt";
                outputFile = outputPath + "OriginalMap.txt";
            }
            string gtFile = POSE_FILE;
            std::vector<Sophus::SE3d> vgtTc2w, voptTc2w;
            std::vector<int> voptFrameID;
            readKITTIPoseFile(gtFile, vgtTc2w);
            if(saveOriginalMap){
                std::vector<KeyFrame*> vpKFs;
                GetAllKeyFrames(KeyFrameInfoDir, vpKFs);
                for(size_t jack=0; jack< vpKFs.size(); ++jack)
                {
                    KeyFrame* iter= vpKFs[jack];
                    voptTc2w.push_back(iter->Tw2c.inverse());
                    voptFrameID.push_back(iter->mnId);
                }
            }
            else
                readOptimizedSim3PoseFile(sim3OptimizedFile, voptTc2w, voptFrameID);
            // purge frames not in the optimized poses
            vector<int>::iterator kale= voptFrameID.begin();
            std::vector<Sophus::SE3d> vgtKFTc2w;
            double totalDistance(0);
            for(size_t jack = 0; jack<vgtTc2w.size(); ++jack)
            {
                if((int)jack == *kale)
                {
                    vgtKFTc2w.push_back(vgtTc2w[jack]);
                    ++kale;
                }
                if(jack)
                    totalDistance += (vgtTc2w[jack].translation() - vgtTc2w[jack-1].translation()).norm();
            }
            assert(kale== voptFrameID.end() && vgtKFTc2w.size() == voptTc2w.size());
            std::vector<Sophus::SE3d> voptTc2wAligned;
            Matrix4d S221=estimateSimilarityTransform(voptTc2w, vgtKFTc2w, voptTc2wAligned);
            cout<< "estimated similarity transform by umeyama "<<endl<<S221<<endl;
            ofstream debug( (outputPath + "transformed.txt").c_str());
            debug<<"% sim3 transformed result: kf id, sw2i, scaled tiinw, ri2w(qxyzw):"<<endl;
            double totalError(0);
            double maxError(0);

            for(size_t i=0;i<voptTc2wAligned.size();++i)
            {
                Eigen::Vector3d dev = vgtKFTc2w[i].translation() - voptTc2wAligned[i].translation();
                totalError += dev.squaredNorm();
                if(dev.norm()> maxError)
                    maxError = dev.norm();

                debug<< voptFrameID[i] <<" 1.0 "<<
                        voptTc2wAligned[i].translation().transpose()<< " "<<
                        voptTc2wAligned[i].unit_quaternion().coeffs().transpose()<<endl;
            }
            double rmse = std::sqrt(totalError/voptTc2wAligned.size()) ;
            std::cout<< "RMSE and Max deviation "<< rmse<< " "<< maxError <<std::endl;
            std::cout << "total distance "<< totalDistance << " ratio of rmse and max error "<<
                         rmse/totalDistance << " "<< maxError/totalDistance<< std::endl;

            debug.close();

            //correct points with optimized poses and then apply similarity transform
            TooN::Matrix<3> r221= Data(S221(0,0),S221(0,1),S221(0,2),
                                       S221(1,0),S221(1,1),S221(1,2),
                                       S221(2,0),S221(2,1),S221(2,2));
            TooN::Matrix<3> squared_sR = r221*r221.T();
            double squared_scale
                    = static_cast<double>(1./3.)
                    *(squared_sR(0,0)+squared_sR(1,1)+squared_sR(2,2));
            double s221 = std::sqrt(squared_scale);
            r221 = r221/s221;

            TooN::Vector<3> t2in1= makeVector(S221(0,3), S221(1,3), S221(2,3));
            RobotVision::Sim3<> S(r221, t2in1, s221);
            if(saveOriginalMap)// save original map points and poses
                figureKITTI(KeyFrameInfoDir, outputFile, S, false);
            else
                figureKITTIBA(KeyFrameInfoDir, sim3OptimizedFile, outputFile, S, 1, false);

        }
        else if(TEST_BAL_OPTIMIZER == experim)
        {
            // call a modified version of bal_example of g2o so that the optimized values can be output into BAL format
            string outputPath = "../data/intermediate/";
            string problemBAL = outputPath + "OriginalMap.txt";
            string sStatistic = outputPath + "balstats.txt";
            string sMapAfterBA = outputPath + "tempMapAfterBA6.txt";

            char *cstr1 = new char[sMapAfterBA.length() + 1];
            strcpy(cstr1, sMapAfterBA.c_str());

            char *cstr2 = new char[sStatistic.length() + 1];
            strcpy(cstr2, sStatistic.c_str());

            char *cstr3 = new char[problemBAL.length() + 1];
            strcpy(cstr3, problemBAL.c_str());

            char* argument[]={"./ba_demo", "-i", "100", "-pcg",
                              "-o", cstr1,
                              "-stats", cstr2,
                              cstr3};
            int argc = sizeof(argument);
            ba_demo(argc,argument);
            std::cout <<"saved optimized map to "<< sMapAfterBA <<std::endl;
            delete [] cstr1;
            delete [] cstr2;
            delete [] cstr3;

        }
        else{
            cout<<"This test option not implemented yet!"<<endl;
        }
    }
    catch(const std::string &ex)
    {
        cout << "Error: " << ex << endl;
    }

    return 0;
}
