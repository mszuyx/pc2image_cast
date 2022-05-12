// ROS core
#include <ros/ros.h>
//sensor message
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/CameraInfo.h>
// Import message_filters lib
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
// Import PCL lib
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
// Import Eigen lib
#include <Eigen/Dense>
// Import cv lib
#include <opencv2/core/eigen.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
// Import realsense camera msg type
#include <realsense2_camera/Extrinsics.h>
// Import custom ros msg
#include <pc_gps/gpParam.h>
#include <p2i_cast/homoMatrix.h>

using namespace message_filters;
using namespace Eigen;

MatrixXd H_depth(3,4);
MatrixXd homography(3,3);
MatrixXd homography_inv(3,3);
cv::Mat HM(3,3,CV_32FC1);
cv::Mat HM_A(3,2,CV_32FC1);

class PointCloudToImage{
public:
  PointCloudToImage();
private:
  // Declare sub & pub
  ros::NodeHandle node_handle_;

  ros::Subscriber depth_info_sub_;
  ros::Subscriber cam_info_sub_;
  ros::Subscriber extrinsics_sub_;

  ros::Publisher image_pub_;
  ros::Publisher depth_pub_;
  ros::Publisher homo_pub_;

  // Declare ROS params
  bool warpAffine;
  bool publishDepth;
  double th_dist_;
  double angle_x;
  double angle_y;
  double angle_z;

  // Sync settings
  message_filters::Subscriber<sensor_msgs::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
  message_filters::Subscriber<pc_gps::gpParam> param_sub_;

  typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, pc_gps::gpParam> RSSyncPolicy;
  typedef Synchronizer<RSSyncPolicy> Sync;
  boost::shared_ptr<Sync> sync;

  // Declare functions const pc_gps::gpParamConstPtr param_msg,
  void projection_callback (const sensor_msgs::Image::ConstPtr& image_msg, const sensor_msgs::Image::ConstPtr& depth_msg, const pc_gps::gpParam::ConstPtr& param_msg);
  void depth_info_callback(const sensor_msgs::CameraInfo::ConstPtr& info_msg);
  void cam_info_callback(const sensor_msgs::CameraInfo::ConstPtr& info_msg);
  void extrinsics_callback(const realsense2_camera::Extrinsics::ConstPtr& info_msg);
  void maskGround(const sensor_msgs::Image::ConstPtr& depth_msg, double params[], cv::Mat mask);
  void removeRowColumn(MatrixXd& matrix, unsigned int colToRemove, bool row_flag);

  bool get_depth_info = false;
  bool get_cam_info = false;
  bool get_extrinsics = false;
  bool get_homography = false;

  int depth_height, depth_width;
  float center_x, center_y, constant_x, constant_y;
  image_geometry::PinholeCameraModel depth_model_;
  Matrix4d extrinsics_transformation, fake_transformation;
};

PointCloudToImage::PointCloudToImage():node_handle_("~"){
  // Init ROS related
  ROS_INFO("Inititalizing PointCloud To Image Node...");

  node_handle_.param("warpAffine", warpAffine, false);
  ROS_INFO("Use warpAffine to reduce computation?: %d", warpAffine);
  node_handle_.param("publishDepth", publishDepth, false);
  ROS_INFO("Publish masked depth image?: %d", publishDepth);
  node_handle_.param("th_dist", th_dist_, 0.05);
  ROS_INFO("Distance Threshold: %f", th_dist_);

  node_handle_.param("angle_x", angle_x, 0.0);
  ROS_INFO("Fine tune angle x: %f", angle_x);
  node_handle_.param("angle_y", angle_y, 0.0);
  ROS_INFO("Fine tune angle y: %f", angle_y);
  node_handle_.param("angle_z", angle_z, 0.0);
  ROS_INFO("Fine tune angle z: %f", angle_z);
  
  // Subscribe to realsense topic
  image_sub_.subscribe(node_handle_, "/image_in", 5);
  depth_sub_.subscribe(node_handle_, "/depth_in", 5);
  param_sub_.subscribe(node_handle_, "/param_in", 10);
  
  // ApproximateTime takes a queue size as its constructor argument, hence RSSyncPolicy(xx)
  sync.reset(new Sync(RSSyncPolicy(20), image_sub_, depth_sub_, param_sub_));   //20
  sync->registerCallback(boost::bind(&PointCloudToImage::projection_callback, this, _1, _2, _3));
  
  depth_info_sub_ = node_handle_.subscribe("/depth_info_in", 1, &PointCloudToImage::depth_info_callback, this);
  cam_info_sub_ = node_handle_.subscribe("/camera_info_in", 1, &PointCloudToImage::cam_info_callback, this);
  extrinsics_sub_ = node_handle_.subscribe("/depth_to_color", 1, &PointCloudToImage::extrinsics_callback, this);

  // Publish Init
  std::string image_out_topic;
  node_handle_.param<std::string>("image_out_topic", image_out_topic, "/image_out");
  ROS_INFO("masked RGB image is published in: %s", image_out_topic.c_str());
  image_pub_ = node_handle_.advertise<sensor_msgs::Image>(image_out_topic, 1);

  std::string homo_topic;
  node_handle_.param<std::string>("homo_topic", homo_topic, "/homography_matrix");
  ROS_INFO("homography matrix is published in: %s", homo_topic.c_str());
  homo_pub_ = node_handle_.advertise<p2i_cast::homoMatrix>(homo_topic, 1);

  if (publishDepth){
    std::string depth_out_topic;
    node_handle_.param<std::string>("depth_out_topic", depth_out_topic, "/depth_out");
    ROS_INFO("masked depth image is published in: %s", depth_out_topic.c_str());
    depth_pub_ = node_handle_.advertise<sensor_msgs::Image>(depth_out_topic, 1);
  }
}

void PointCloudToImage::removeRowColumn(MatrixXd& matrix, unsigned int ToRemove, bool row_flag){
  if (row_flag){
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();
    if( ToRemove < numRows )
        matrix.block(ToRemove,0,numRows-ToRemove,numCols) = matrix.bottomRows(numRows-ToRemove);
    matrix.conservativeResize(numRows,numCols);
  }else{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;
    if( ToRemove < numCols )
        matrix.block(0,ToRemove,numRows,numCols-ToRemove) = matrix.rightCols(numCols-ToRemove);
    matrix.conservativeResize(numRows,numCols);
  }
}

void PointCloudToImage::depth_info_callback(const sensor_msgs::CameraInfo::ConstPtr& info_msg){
  if (get_depth_info && get_extrinsics){
    double P[12];
    for(int i = 0; i < 12; ++i) P[i] = info_msg->P[i];
    MatrixXd depth_P(3,4);
    depth_P = Map<Matrix<double,3,4,RowMajor>>(P);
    H_depth = depth_P*fake_transformation;
    removeRowColumn(H_depth,2,false);
    depth_info_sub_.shutdown();
  }else if(get_depth_info==false){
    depth_height = info_msg->height;
    depth_width = info_msg->width;
    depth_model_.fromCameraInfo(info_msg);
    center_x = depth_model_.cx();
    center_y = depth_model_.cy();
    constant_x = 0.001 / depth_model_.fx();
    constant_y = 0.001 / depth_model_.fy();
    get_depth_info = true;
    ROS_INFO("Got depth info!");
  }
}

void PointCloudToImage::cam_info_callback(const sensor_msgs::CameraInfo::ConstPtr& info_msg){
  if (get_cam_info && get_extrinsics && get_depth_info){
    double P[12];
    for(int i = 0; i < 12; ++i) P[i] = info_msg->P[i];
    MatrixXd rgb_P(3,4);
    rgb_P = Map<Matrix<double,3,4,RowMajor>>(P);
    MatrixXd H_rgb(3,4);
    Matrix4d rgb_to_depth;
    rgb_to_depth = fake_transformation*(extrinsics_transformation.inverse());
    H_rgb = rgb_P*rgb_to_depth;
    removeRowColumn(H_rgb,2,false);
    homography = H_depth*(H_rgb.completeOrthogonalDecomposition().pseudoInverse());
    homography_inv = homography.inverse();
    if(warpAffine){
      removeRowColumn(homography_inv,2,true);
      cv::eigen2cv(homography_inv,HM_A);
    }else{
      cv::eigen2cv(homography_inv,HM);
    }
    get_homography = true;
    // std::cout<< "The homography transformation is :" <<std::endl;
    // std::cout<< homography <<std::endl;
    ROS_INFO("Built homography transformation!");
    cam_info_sub_.shutdown();
  }else if(get_cam_info==false){
    get_cam_info = true;
    ROS_INFO("Got camera info!");}
  }

void PointCloudToImage::extrinsics_callback(const realsense2_camera::Extrinsics::ConstPtr& info_msg){
  if (get_extrinsics){
    extrinsics_sub_.shutdown();
  }else{
  double rot[9];
  Matrix3d rotation_m;
  for(int i = 0; i < 9; ++i) rot[i] = info_msg->rotation[i];
  rotation_m = Map<Matrix<double,3,3,RowMajor>>(rot);
  Matrix3d rotation_x;
  rotation_x = AngleAxisd(angle_z, Vector3d::UnitZ()) * AngleAxisd(angle_y, Vector3d::UnitY()) * AngleAxisd(angle_x, Vector3d::UnitX()); 
  rotation_m = rotation_m*rotation_x;
  MatrixXd t(3,1);
  for(int i = 0; i < 3; ++i) t(i) = info_msg->translation[i];
  MatrixXd trans_m(3,4);
  trans_m << rotation_m, t;
  MatrixXd stack(1,4);
  stack << 0.0,0.0,0.0,1.0;
  extrinsics_transformation << trans_m,stack;
  fake_transformation <<  1.0,       0.0,       0.0,  0.0,
                          0.0, 0.8660254,       0.5, -1.0,
                          0.0,      -0.5, 0.8660254,  0.0,
                          0.0,       0.0,       0.0,  1.0;
  // std::cout<< "The extrinsics transformation is:" <<std::endl;
  // std::cout<< extrinsics_transformation <<std::endl;
  // std::cout<< "The fake transformation is:" <<std::endl;
  // std::cout<< fake_transformation <<std::endl;
  get_extrinsics = true;
  ROS_INFO("Got extrinsics info!");}
}

// template<typename T>
void PointCloudToImage::maskGround(const sensor_msgs::Image::ConstPtr& depth_msg, double params[], cv::Mat mask){
  cv::Mat depth_mask = cv::Mat::zeros(cv::Size(depth_width,depth_height), CV_8UC1);

  const uint16_t* depth_row = reinterpret_cast<const uint16_t*>(&depth_msg->data[0]);
  int row_step = depth_msg->step / sizeof(uint16_t);
  for (int v = 0; v < depth_height; ++v, depth_row += row_step){
    for (int u = 0; u < depth_width; ++u){
      uint16_t depth = depth_row[u];
      double x,y,z,dist;
      if (depth != 0){
        x = (u - center_x) * depth * constant_x;
        y = (v - center_y) * depth * constant_y;
        z = depth * 0.001;
        dist = x*params[0]+y*params[1]+z*params[2];
        double adj_th_ = 0.02*z;
        double th_dist_d_ = -(th_dist_ + params[3]);
        if(dist>(th_dist_d_-adj_th_) && dist<(th_dist_d_+adj_th_+(3*th_dist_))){
          depth_mask.at<uchar>(v,u) = 1;
        }
      }
    }
  }

  if(warpAffine){
    cv::warpAffine(depth_mask,mask,HM_A,mask.size());
  }else{
    cv::warpPerspective(depth_mask,mask,HM,mask.size());
  }

  if (publishDepth){
    cv_bridge::CvImagePtr depth_bridge_;
    try{depth_bridge_ = cv_bridge::toCvCopy(depth_msg, "16UC1");}
    catch (cv_bridge::Exception& ex){
      ROS_ERROR("Failed to convert depth image");
      return;
    }
    cv::Mat OutImage = cv::Mat::zeros(depth_bridge_->image.size(), depth_bridge_->image.type());
    depth_bridge_->image.copyTo(OutImage, depth_mask);
    depth_bridge_->image = OutImage;
    // publish output image
    depth_pub_.publish(depth_bridge_->toImageMsg());
  }
}


void PointCloudToImage::projection_callback (const sensor_msgs::Image::ConstPtr& image_msg, const sensor_msgs::Image::ConstPtr& depth_msg, const pc_gps::gpParam::ConstPtr& param_msg){
  // ROS_INFO("callback"); 
  if (get_homography){

  double params[4];
  for(int i = 0; i < 4; ++i) params[i] = param_msg->data[i];

  cv_bridge::CvImagePtr image_bridge_;
  try{image_bridge_ = cv_bridge::toCvCopy(image_msg, "bgr8");}
  catch (cv_bridge::Exception& ex){
    ROS_ERROR("Failed to convert rgb image");
    return;
  }
  cv::Mat mask = cv::Mat::zeros(image_bridge_->image.size(), CV_8UC1);
  maskGround(depth_msg,params,mask);
  cv::Mat OutImage = cv::Mat::zeros(image_bridge_->image.size(), image_bridge_->image.type());
  image_bridge_->image.copyTo(OutImage, mask);
  image_bridge_->image = OutImage;
  // publish output image
  image_pub_.publish(image_bridge_->toImageMsg());

  // publish homography matrix
  p2i_cast::homoMatrix HM_msg;
  for(int i=0; i<9; i++){HM_msg.rgb_to_depth[i] = homography(i);}
  for(int i=0; i<9; i++){HM_msg.depth_to_rgb[i] = homography_inv(i);}
  homo_pub_.publish(HM_msg);
  }
}

int main (int argc, char** argv) {
    ros::init(argc, argv, "PointCloudToImage");
    PointCloudToImage node;
    ros::spin();
    return 0;
 }
