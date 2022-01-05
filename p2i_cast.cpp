// ROS core
#include <ros/ros.h>
//sensor message
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
// Import message_filters lib
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/radius_outlier_removal.h>
// Import eigen lib
#include <Eigen/Dense>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

using namespace message_filters;
using namespace Eigen;

class PointCloudToImage{
public:
    PointCloudToImage();
private:
    // Declare sub & pub
    ros::NodeHandle node_handle_;
    ros::Publisher image_pub_;
    ros::Subscriber info_sub_;
    
    // Declare ROS params
    double unit_size_;
    int radius_;
    double radius_search_;
    int in_radius_;

    // Sync settings
    message_filters::Subscriber<sensor_msgs::PointCloud2> points_sub_;
    message_filters::Subscriber<sensor_msgs::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_node_sub_;
    typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image, sensor_msgs::Imu> RSSyncPolicy;
    typedef Synchronizer<RSSyncPolicy> Sync;
    boost::shared_ptr<Sync> sync;

    // Declare functions
    void projection_callback_ (const sensor_msgs::PointCloud2ConstPtr& input_cloud, const sensor_msgs::ImageConstPtr& input_image, const sensor_msgs::Imu::ConstPtr& imu_msg);
    void infoCallback(const sensor_msgs::CameraInfoConstPtr& info_msg);
    void maskGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, cv_bridge::CvImagePtr& image, cv::Mat Outimage);
    void quaternionToMatrixInv(double q0, double q1, double q2, double q3, Affine3d& transform);
    
    bool get_cam_info = false;
    cv_bridge::CvImagePtr image_bridge_;
    image_geometry::PinholeCameraModel cam_model_;
};

PointCloudToImage::PointCloudToImage():node_handle_("~"){
    // Init ROS related
    ROS_INFO("Inititalizing PointCloud To Image Node...");

    node_handle_.param("unit_size_", unit_size_, 0.1);
    ROS_INFO("Num of Iteration: %f", unit_size_);
    node_handle_.param("radius_", radius_, 10);
    ROS_INFO("Marker radius size: %d", radius_);
    node_handle_.param("radius_search_", radius_search_, 0.8);
    ROS_INFO("radius_search_: %f", radius_search_);
    node_handle_.param("in_radius_", in_radius_, 3);
    ROS_INFO("in_radius_: %d", in_radius_);

    // Subscribe to realsense topic
    points_sub_.subscribe(node_handle_, "/points_in", 10); //points_ground
    image_sub_.subscribe(node_handle_, "/image_in", 5);
    imu_node_sub_.subscribe(node_handle_, "/imu/data", 100);
    // ApproximateTime takes a queue size as its constructor argument, hence RSSyncPolicy(xx)
    sync.reset(new Sync(RSSyncPolicy(20), points_sub_, image_sub_, imu_node_sub_));   
    sync->registerCallback(boost::bind(&PointCloudToImage::projection_callback_, this, _1, _2, _3));
    
    info_sub_ = node_handle_.subscribe("/d455/color/camera_info", 1, &PointCloudToImage::infoCallback, this);

    // Publish Init
    std::string image_topic;
    node_handle_.param<std::string>("image_topic", image_topic, "/masked_image");
    ROS_INFO("masked image: %s", image_topic.c_str());
    image_pub_ = node_handle_.advertise<sensor_msgs::Image>(image_topic, 1);
}

void PointCloudToImage::infoCallback(const sensor_msgs::CameraInfoConstPtr& info_msg){
    if (get_cam_info)
      return;
    cam_model_.fromCameraInfo(info_msg);
    get_cam_info = true;
    ROS_INFO("Got image info!");
  }

void PointCloudToImage::maskGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, cv_bridge::CvImagePtr& image, cv::Mat Outimage){  
    // Overlay calibration points on the image
    //pcl::PointCloud<pcl::PointXYZ> transformed_detector_points;
    //pcl_ros::transformPointCloud(detector_points, transformed_detector_points, transform);
    //cv::Point2d uv;
    //uv.x = 100;
    //uv.y = 100;
    //int radius = 10;
    //cv::circle(image->image, uv, radius, CV_RGB(0,153,255), -1);
    cv::Mat mask = cv::Mat::zeros(image->image.size(), image->image.type());
    
    for (unsigned int i=0; i < cloud->points.size(); i++){
      pcl::PointXYZ pt = (*cloud)[i];
      cv::Point3d pt_cv(pt.x, pt.y, pt.z);
      cv::Point2d uv;
      int radius_f = int(radius_/(pt.z));
      if (radius_f<= 2){
        radius_f = 2;
      } 
      uv = cam_model_.project3dToPixel(pt_cv);
      cv::circle(mask, uv, radius_f, cv::Scalar(255, 255, 255), -1); //CV_RGB(0,153,255)
    }

    image->image.copyTo(Outimage, mask);
  }
  
void PointCloudToImage::quaternionToMatrixInv(double q0, double q1, double q2, double q3, Affine3d& transform){
    double t0 = 2 * (q0 * q1 + q2 * q3);
    double t1 = 1 - 2 * (q1 * q1 + q2 * q2);
    double roll = std::atan2(t0, t1);

    double t2 = 2 * (q0 * q2 - q3 * q1);
    if (t2 >= 1){t2 = 1.0;}
    else if (t2<= -1){t2 = -1.0;}
    double pitch = std::asin(t2);
    
    // axis defined in camera_depth_optical_frame
    transform = AngleAxisd(-(roll+1.5708), Vector3d::UnitX()) * AngleAxisd(-pitch, Vector3d::UnitZ()); 
}

void PointCloudToImage::projection_callback_ (const sensor_msgs::PointCloud2ConstPtr& input_cloud, const sensor_msgs::ImageConstPtr& input_image, const sensor_msgs::Imu::ConstPtr& imu_msg){
    //ROS_INFO("callback"); 
    // Convert pc to pcl::PointXYZ
    pcl::PCLPointCloud2::Ptr input_cloud_pcl (new pcl::PCLPointCloud2 ());
    pcl_conversions::toPCL(*input_cloud, *input_cloud_pcl);
     
    // Apply voxel filter
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw (new pcl::PointCloud<pcl::PointXYZ> ());
    if (unit_size_>0){
      pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2 ());
      pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
      sor.setInputCloud (input_cloud_pcl);
      sor.setLeafSize (float (unit_size_),float (unit_size_),float (unit_size_));
      sor.filter (*cloud_filtered);
      pcl::fromPCLPointCloud2(*cloud_filtered, *cloud_raw);
    }else{
      pcl::fromPCLPointCloud2(*input_cloud_pcl, *cloud_raw);
    }

    // Invert pointcloud back to original attitude
    double q0_in, q1_in, q2_in, q3_in; 
    q0_in=imu_msg->orientation.w;
    q1_in=imu_msg->orientation.x;
    q2_in=imu_msg->orientation.y;
    q3_in=imu_msg->orientation.z;
    Affine3d transform;
    quaternionToMatrixInv(q0_in, q1_in, q2_in, q3_in, transform);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::transformPointCloud (*cloud_raw, *cloud, transform);

    try{
      image_bridge_ = cv_bridge::toCvCopy(input_image, "bgr8");
    }
    catch (cv_bridge::Exception& ex)
    {
      ROS_ERROR("Failed to convert image");
      return;
    }
    cv::Mat OutImage = cv::Mat::zeros(image_bridge_->image.size(), image_bridge_->image.type());

    if (in_radius_>0){
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cleaned (new pcl::PointCloud<pcl::PointXYZ> ());
      pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
      outrem.setInputCloud(cloud);
      outrem.setRadiusSearch(radius_search_);
      outrem.setMinNeighborsInRadius (in_radius_);
      outrem.setKeepOrganized(true);
      outrem.filter (*cloud_cleaned);

      maskGround(cloud_cleaned, image_bridge_, OutImage);
    }else{
      maskGround(cloud, image_bridge_, OutImage);
    }
    
    image_bridge_->image = OutImage;
    
    // publish output image
    image_pub_.publish(image_bridge_->toImageMsg());
}

int main (int argc, char** argv) {
    ros::init(argc, argv, "PointCloudToImage");
    
    PointCloudToImage node;
    
    ros::spin();
    return 0;
 }
