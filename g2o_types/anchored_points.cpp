#include "anchored_points.h"
//#include "eigen_utils.h"

using namespace Eigen;
using namespace g2o;
using namespace std; // for debugging output
namespace ScaViSLAM{


/// returns the 3D cross product skew symmetric matrix of a given 3D vector
template<class Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3> skew3d(const Eigen::MatrixBase<Derived> & vec)
{
    typedef typename Derived::Scalar Scalar;
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);
    return (Eigen::Matrix<typename Derived::Scalar, 3, 3>() << Scalar(0.0), -vec[2], vec[1], vec[2],
            Scalar(0.0), -vec[0], -vec[1], vec[0], Scalar(0.0)).finished();
}

g2o::OptimizableGraph::Vertex*  GET_MAP_ELEM(const int & key,
                                             const g2o::OptimizableGraph::VertexIDMap & m)
{
    g2o::OptimizableGraph::VertexIDMap::const_iterator it = m.find(key);
    assert(it!=m.end());
    return dynamic_cast<g2o::OptimizableGraph::Vertex*>(it->second);
}

G2oCameraParameters
::G2oCameraParameters()
    : principle_point_(Eigen::Vector2d(0., 0.)),
      focal_length_(Eigen::Vector2d(1., 1.))
{
}

Eigen::Vector2d  G2oCameraParameters
::cam_map(const Eigen::Vector3d & trans_xyz) const
{
    Eigen::Vector2d proj = project2d(trans_xyz);
    Eigen::Vector2d res;
    res[0] = proj[0]*focal_length_[0] + principle_point_[0];
    res[1] = proj[1]*focal_length_[1] + principle_point_[1];
    return res;
}
Eigen::Vector2d G2oCameraParameters::
normalize                    (const Eigen::Vector2d & uv) const
{
    return (uv- principle_point_).cwiseQuotient(focal_length_);
}

Eigen::Matrix<double,2,6>
G2oCameraParameters::frameJac(const Sophus::SE3d & se3,
                              const Eigen::Vector3d & xyz)const
{
    const Eigen::Vector3d & xyz_trans = se3*xyz;
    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];
    double fx = focal_length_[0];
    double fy = focal_length_[1];

    double one_b_z = 1./z;
    double one_b_z_sq = 1./(z*z);
    double A = -fx*one_b_z;
    double B = -fy*one_b_z;
    double C = fx*x*one_b_z_sq;
    double D = fy*y*one_b_z_sq;

    Eigen::Matrix<double, 2, 6> jac;
    jac <<  A, 0, C, y*C,     -fx-x*C, -y*A,
            0, B, D, fy+y*D, -x*D,     x*B;
    return jac;
}
Eigen::Matrix<double,2,3>
G2oCameraParameters::pointJac(const Sophus::SE3d & T_cw,
                              const Eigen::Vector3d & xyz_w) const
{
    Eigen::Vector3d y = T_cw*xyz_w;
    Eigen::Matrix<double,2,3> J1
            = d_proj_d_y(focal_length_[0],y);
    return -J1*T_cw.rotationMatrix();
}

void G2oVertexSE3
::oplusImpl(const double * update_p)
{
    Eigen::Map<const Vector6d> update(update_p);
    setEstimate(Sophus::SE3d::exp(update)*estimate());
}

//TODO: implement, but first remove camera parameters from vertex state
bool G2oVertexSE3
::write(std::ostream& os) const
{
    os<<"estimate qxyzw, txyz:"<< _estimate.unit_quaternion().coeffs().transpose()<<
        " "<< _estimate.translation().transpose()<<std::endl;
    return os.good();
}

//TODO: implement, but first remove camera parameters from vertex state
bool G2oVertexSE3
::read(std::istream& is)
{
    // input format qxyzw, txyz
    for(int jack=0; jack<7; ++jack)
        is >> _estimate.data()[jack];
    return is.good();
}



void G2oVertexPointXYZ
::oplusImpl(const double * update_p)
{
    Eigen::Map<const Eigen::Vector3d> update(update_p);
    _estimate += update;
}

bool G2oVertexPointXYZ
::write (std::ostream & os) const
{
    const Eigen::Vector3d & lv = estimate();
    for (int i=0; i<3; i++)
    {
        os << lv[i] << " ";
    }
    return true;
}

bool G2oVertexPointXYZ
::read(std::istream& is)
{
    Eigen::Vector3d lv;
    for (int i=0; i<3; i++)
    {
        is >> lv[i];
    }
    setEstimate(lv);
    return true;
}

bool G2oEdgeSE3
::read(std::istream& is)
{
    //input qxyzw, txyz
    for(int jack=0; jack<7;++jack)
        is>> _measurement.data()[jack];
    return is.good();
}

bool G2oEdgeSE3
::write(std::ostream& os) const
{
    os<<"measurement qxyzw txyz:"<< _measurement.unit_quaternion().coeffs().transpose()<<
        " "<< _measurement.translation().transpose()<<std::endl;
    return os.good();
}
typedef Eigen::Matrix<double, 6, 6, Eigen::ColMajor> Matrix6d;

Matrix6d third(const Sophus::SE3d & A, const Vector6d & d)
{
    const Matrix6d & AdjA = A.Adj();

    Matrix6d d_lie = Sophus::SE3d::d_lieBracketab_by_d_a(d);
    //cerr << d_lie << endl;
    return AdjA + 0.5*d_lie*AdjA + (1./12.)*d_lie*d_lie*AdjA;
}


void G2oEdgeSE3
::computeError()
{
    const G2oVertexSE3 * v1 = static_cast<const G2oVertexSE3 *>(_vertices[0]);
    const G2oVertexSE3 * v2 = static_cast<const G2oVertexSE3 *>(_vertices[1]);
    Sophus::SE3d T_21(_measurement);
    _error = (T_21*v1->estimate()*v2->estimate().inverse()).log();
}

void G2oEdgeSE3::
linearizeOplus()
{
    const Sophus::SE3d & T_21 = _measurement;
    Sophus::SE3d I;
    const Vector6d & d = _error;
    _jacobianOplusXi = third(T_21, d);
    _jacobianOplusXj = -third(I, -d);

}
G2oEdgeProjectXYZ2UV::G2oEdgeProjectXYZ2UV() : BaseBinaryEdge<2, Eigen::Vector2d, G2oVertexPointXYZ, G2oVertexSE3>() {
    _cam = 0;
    resizeParameters(1);
    installParameter(_cam, 0);
}
bool G2oEdgeProjectXYZ2UV::read(std::istream& is){
    int paramId;
    is >> paramId;
    setParameterId(0, paramId);
    for (int i=0; i<2; i++){
        is >> _measurement[i];
    }
    for (int i=0; i<2; i++)
        for (int j=i; j<2; j++) {
            is >> information()(i,j);
            if (i!=j)
                information()(j,i)=information()(i,j);
        }
    return true;
}
bool G2oEdgeProjectXYZ2UV::write(std::ostream& os) const {
    os << _cam->id() << " ";
    for (int i=0; i<2; i++){
        os << measurement()[i] << " ";
    }
    for (int i=0; i<2; i++)
        for (int j=i; j<2; j++){
            os << " " << information()(i,j);
        }
    return os.good();
}

void G2oEdgeProjectXYZ2UV::linearizeOplus() {
    G2oVertexSE3* vj = static_cast<G2oVertexSE3*>(_vertices[1]);
    Sophus::SE3d T(vj->estimate());
    if(vj->first_estimate!=NULL) T=*(vj->first_estimate);
    G2oVertexPointXYZ* vi = static_cast<G2oVertexPointXYZ*>(_vertices[0]);
    Eigen::Vector3d xyz_trans((vi->first_estimate==NULL)?T*vi->estimate(): T*(*(vi->first_estimate)));

    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];
    const G2oCameraParameters * cam = static_cast<const G2oCameraParameters *>(parameter(0));
    Eigen::Matrix<double,2,3,Eigen::ColMajor> tmp;
    tmp(0,0) = cam->focal_length_[0];
    tmp(0,1) = 0;
    tmp(0,2) = -x/z*cam->focal_length_[0];
    tmp(1,0) = 0;
    tmp(1,1) = cam->focal_length_[1];
    tmp(1,2) = -y/z*cam->focal_length_[1];

    _jacobianOplusXi = -1./z * tmp * T.rotationMatrix();

//    std::cout<<"J1 original: "<< endl<< _jacobianOplusXi<<std::endl;
//    cout<<"J1 pointJac : "<< endl<< cam->pointJac(T, vi->estimate())<<endl;
    Eigen::Matrix<double,3,6,Eigen::ColMajor> dX_to_dT;
    dX_to_dT.block<3,3>(0,0)=Eigen::Matrix3d::Identity();
    dX_to_dT.block<3,3>(0,3)= - skew3d(xyz_trans);
    _jacobianOplusXj = -1./z * tmp * dX_to_dT;

//    cout<<"J2 original: "<< endl<< _jacobianOplusXj<<endl;
//    cout<<"J2 frameJac : "<< endl<< cam->frameJac(T, vi->estimate())<<endl;
}
void G2oExEdgeProjectXYZ2UV::linearizeOplus() {
    G2oVertexSE3* vj = static_cast<G2oVertexSE3*>(_vertices[1]);
    Sophus::SE3d vjestimate=(vj->first_estimate==NULL? vj->estimate(): *(vj->first_estimate));
    Sophus::SE3d T(Ts2c*vjestimate);
    G2oVertexPointXYZ* vi = static_cast<G2oVertexPointXYZ*>(_vertices[0]);
    Eigen::Vector3d viestimate= (vi->first_estimate==NULL? vi->estimate(): *(vi->first_estimate));
    Eigen::Vector3d xyz_trans = T*viestimate;
    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];
    const G2oCameraParameters * cam = static_cast<const G2oCameraParameters *>(parameter(0));
    Eigen::Matrix<double,2,3,Eigen::ColMajor> tmp;
    tmp(0,0) = cam->focal_length_[0];
    tmp(0,1) = 0;
    tmp(0,2) = -x/z*cam->focal_length_[0];
    tmp(1,0) = 0;
    tmp(1,1) = cam->focal_length_[1];
    tmp(1,2) = -y/z*cam->focal_length_[1];
    _jacobianOplusXi = -1./z * tmp * T.rotationMatrix();
    Eigen::Matrix<double,3,6,Eigen::ColMajor> dX_to_dT;
    dX_to_dT.block<3,3>(0,0)=Eigen::Matrix3d::Identity();
    dX_to_dT.block<3,3>(0,3)= - skew3d(vjestimate*viestimate);
    _jacobianOplusXj = -1./z * tmp *Ts2c.rotationMatrix()* dX_to_dT;
}


EdgeSE3ProjectXYZ::EdgeSE3ProjectXYZ() : BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSE3Expmap>() {
}

bool EdgeSE3ProjectXYZ::read(std::istream& is){
    for (int i=0; i<2; i++){
        is >> _measurement[i];
    }
    for (int i=0; i<2; i++)
        for (int j=i; j<2; j++) {
            is >> information()(i,j);
            if (i!=j)
                information()(j,i)=information()(i,j);
        }
    return true;
}

bool EdgeSE3ProjectXYZ::write(std::ostream& os) const {

    for (int i=0; i<2; i++){
        os << measurement()[i] << " ";
    }

    for (int i=0; i<2; i++)
        for (int j=i; j<2; j++){
            os << " " <<  information()(i,j);
        }
    return os.good();
}


void EdgeSE3ProjectXYZ::linearizeOplus() {
    VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
    SE3Quat T(vj->estimate());
    VertexSBAPointXYZ* vi = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
    Vector3d xyz = vi->estimate();
    Vector3d xyz_trans = T.map(xyz);

    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];
    double z_2 = z*z;

    Matrix<double,2,3> tmp;
    tmp(0,0) = fx;
    tmp(0,1) = 0;
    tmp(0,2) = -x/z*fx;

    tmp(1,0) = 0;
    tmp(1,1) = fy;
    tmp(1,2) = -y/z*fy;

    _jacobianOplusXi =  -1./z * tmp * T.rotation().toRotationMatrix();

    _jacobianOplusXj(0,0) =  x*y/z_2 *fx;
    _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *fx;
    _jacobianOplusXj(0,2) = y/z *fx;
    _jacobianOplusXj(0,3) = -1./z *fx;
    _jacobianOplusXj(0,4) = 0;
    _jacobianOplusXj(0,5) = x/z_2 *fx;

    _jacobianOplusXj(1,0) = (1+y*y/z_2) *fy;
    _jacobianOplusXj(1,1) = -x*y/z_2 *fy;
    _jacobianOplusXj(1,2) = -x/z *fy;
    _jacobianOplusXj(1,3) = 0;
    _jacobianOplusXj(1,4) = -1./z *fy;
    _jacobianOplusXj(1,5) = y/z_2 *fy;
}

Vector2d EdgeSE3ProjectXYZ::cam_project(const Vector3d & trans_xyz) const{
    Vector2d proj = project2d(trans_xyz);
    Vector2d res;
    res[0] = proj[0]*fx + cx;
    res[1] = proj[1]*fy + cy;
    return res;
}
// copied from Raul ORB_SLAM /Thirdparty/g2o/g2o/types/sim3/types_seven_dof_expmap.cpp
// This function was commented both in ORB_SLAM and github g2o for no clear reason
// note that g2o::Sim3 tangential space [omega, upsilon, sigma] determining [rotation, translation, scale]
// in contrast Sophus sim3 tangential space is [upsilon, omega, sigma]
// The exact Jacobian expression can be found in Strasdat's dissertation
/* $J_0=\frac{\partial \left\{ \begin{bmatrix} u
\\ v \end{bmatrix}- \textup{proj}\left[\mathbf{K}_1 \textup{proj}(\mathbf{S}_w^c \dot{p}^w) \right ]\right\}}{\partial [p^w]} = -\begin{bmatrix}
\frac{-f_x}{z} & 0 & \frac{f_xx}{z^2}\\
0 & \frac{-f_y}{z} & \frac{f_yy}{z^2}\end{bmatrix}s\mathbf{R}$*/
/*$J_1=\frac{\partial \left\{ \begin{bmatrix} u
\\ v \end{bmatrix}- \textup{proj}\left[\mathbf{K}_1 \textup{proj}(\mathbf{S}_w^c \dot{p}^w) \right ]\right\}}{\partial [\upsilon, \omega, \sigma]]} = -\begin{bmatrix}
\frac{-f_x}{z} & 0 & \frac{f_xx}{z^2}\\
0 & \frac{-f_y}{z} & \frac{f_yy}{z^2}\end{bmatrix}\begin{bmatrix}
\mathbf{I}_3 & -[\mathbf{x}]_\times & \mathbf{x}
\end{bmatrix}$ where $\mathbf{x}= s\mathbf{R}p^w+\mathbf{t}$*/

//  void EdgeSim3ProjectXYZ::linearizeOplus()
//  {
//    VertexSim3Expmap * vj = static_cast<VertexSim3Expmap *>(_vertices[1]);
//    Sim3 T = vj->estimate();

//    VertexPointXYZ* vi = static_cast<VertexPointXYZ*>(_vertices[0]);
//    Vector3d xyz = vi->estimate();
//    Vector3d xyz_trans = T.map(xyz);

//    double x = xyz_trans[0];
//    double y = xyz_trans[1];
//    double z = xyz_trans[2];
//    double z_2 = z*z;

//    Matrix<double,2,3> tmp;
//    tmp(0,0) = _focal_length(0);
//    tmp(0,1) = 0;
//    tmp(0,2) = -x/z*_focal_length(0);

//    tmp(1,0) = 0;
//    tmp(1,1) = _focal_length(1);
//    tmp(1,2) = -y/z*_focal_length(1);

//    _jacobianOplusXi =  -1./z * tmp * T.rotation().toRotationMatrix();// scale is ignored

//    _jacobianOplusXj(0,0) =  x*y/z_2 * _focal_length(0);
//    _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *_focal_length(0);
//    _jacobianOplusXj(0,2) = y/z *_focal_length(0);
//    _jacobianOplusXj(0,3) = -1./z *_focal_length(0);
//    _jacobianOplusXj(0,4) = 0;
//    _jacobianOplusXj(0,5) = x/z_2 *_focal_length(0);
//    _jacobianOplusXj(0,6) = 0; // scale is ignored


//    _jacobianOplusXj(1,0) = (1+y*y/z_2) *_focal_length(1);
//    _jacobianOplusXj(1,1) = -x*y/z_2 *_focal_length(1);
//    _jacobianOplusXj(1,2) = -x/z *_focal_length(1);
//    _jacobianOplusXj(1,3) = 0;
//    _jacobianOplusXj(1,4) = -1./z *_focal_length(1);
//    _jacobianOplusXj(1,5) = y/z_2 *_focal_length(1);
//    _jacobianOplusXj(1,6) = 0; // scale is ignored
//  }
}
