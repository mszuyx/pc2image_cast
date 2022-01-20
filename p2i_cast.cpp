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
// Import PCL lib
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
// Import Eigen lib
#include <Eigen/Dense>
// Import cv lib
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>

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
    int radius_;
    bool do_subtract_;

    // Sync settings
    message_filters::Subscriber<sensor_msgs::PointCloud2> points_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> ng_points_sub_;
    message_filters::Subscriber<sensor_msgs::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_node_sub_;

    typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::Image, sensor_msgs::Imu> RSSyncPolicy_sub;
    typedef Synchronizer<RSSyncPolicy_sub> Sync_sub;
    boost::shared_ptr<Sync_sub> sync_sub;

    typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image, sensor_msgs::Imu> RSSyncPolicy;
    typedef Synchronizer<RSSyncPolicy> Sync;
    boost::shared_ptr<Sync> sync;

    // Declare functions
    void projection_callback_sub_ (const sensor_msgs::PointCloud2ConstPtr& input_cloud, const sensor_msgs::PointCloud2ConstPtr& ng_cloud, const sensor_msgs::ImageConstPtr& input_image, const sensor_msgs::Imu::ConstPtr& imu_msg);
    void projection_callback_ (const sensor_msgs::PointCloud2ConstPtr& input_cloud, const sensor_msgs::ImageConstPtr& input_image, const sensor_msgs::Imu::ConstPtr& imu_msg);
    void infoCallback(const sensor_msgs::CameraInfoConstPtr& info_msg);
    void maskGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr& ng_cloud, cv_bridge::CvImagePtr& image, cv::Mat Outimage);
    void quaternionToMatrixInv(double q0, double q1, double q2, double q3, Affine3d& transform);
    
    bool get_cam_info = false;
    cv_bridge::CvImagePtr image_bridge_;
    image_geometry::PinholeCameraModel cam_model_;
};

PointCloudToImage::PointCloudToImage():node_handle_("~"){
    // Init ROS related
    ROS_INFO("Inititalizing PointCloud To Image Node...");

    node_handle_.param("radius_", radius_, 10);
    ROS_INFO("Marker radius size: %d", radius_);
    node_handle_.param("do_subtract_", do_subtract_, false);
    ROS_INFO("Refine the mask using obstacle point cloud?: %d", do_subtract_);
   
    // Subscribe to realsense topic
    points_sub_.subscribe(node_handle_, "/points_in", 1); //points_ground 10
    image_sub_.subscribe(node_handle_, "/image_in", 1);    // 5
    imu_node_sub_.subscribe(node_handle_, "/imu/data", 1); // 100
    
    if (do_subtract_){
      ng_points_sub_.subscribe(node_handle_, "/ng_points_in", 1); //points_ground 10
      // ApproximateTime takes a queue size as its constructor argument, hence RSSyncPolicy(xx)
      sync_sub.reset(new Sync_sub(RSSyncPolicy_sub(20), points_sub_, ng_points_sub_, image_sub_, imu_node_sub_));   //20
      sync_sub->registerCallback(boost::bind(&PointCloudToImage::projection_callback_sub_, this, _1, _2, _3, _4));
    }else{
      // ApproximateTime takes a queue size as its constructor argument, hence RSSyncPolicy(xx)
      sync.reset(new Sync(RSSyncPolicy(20), points_sub_, image_sub_, imu_node_sub_));   //20
      sync->registerCallback(boost::bind(&PointCloudToImage::projection_callback_, this, _1, _2, _3));
    }
    
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

void PointCloudToImage::maskGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr& ng_cloud, cv_bridge::CvImagePtr& image, cv::Mat Outimage){  
    //cv::circle(image->image, uv, radius, CV_RGB(0,153,255), -1);
    cv::Mat mask = cv::Mat::zeros(image->image.size(), image->image.type());
    
    for (unsigned int i=0; i < cloud->points.size(); i++){
      if (i % 2 == 0){
        pcl::PointXYZ pt = (*cloud)[i];
        cv::Point3d pt_cv(pt.x, pt.y, pt.z);
        cv::Point2d uv;
        
        int radius_f = int(radius_/(0.2*(pt.z+0.0001)));
        radius_f = std::max(2,radius_f);
        uv = cam_model_.project3dToPixel(pt_cv);
        cv::Point2d uv_offset(radius_f,radius_f);
        cv::rectangle(mask, uv-uv_offset, uv+uv_offset, cv::Scalar(255, 255, 255), -1); //CV_RGB(0,153,255)
      }
    }

    if(cloud->size() > 0){
      for (unsigned int i=0; i < ng_cloud->points.size(); i++){
        pcl::PointXYZ pt = (*ng_cloud)[i];
        cv::Point3d pt_cv(pt.x, pt.y, pt.z);
        cv::Point2d uv;

        uv = cam_model_.project3dToPixel(pt_cv);
        cv::Scalar color = mask.at<uchar>(uv);
        if (color[0] == 255){
          // std::cout << color << std::endl;
          double dist = sqrt((pt.z*pt.z) + (pt.y*pt.y));
          int radius_f = int((radius_)/(0.2*(dist+0.0001)));
          radius_f = std::clamp(radius_f,5,20);
          cv::Point2d uv_offset(radius_f,radius_f);
          cv::rectangle(image->image, uv-uv_offset, uv+uv_offset, cv::Scalar(0, 0, 0), -1); //CV_RGB(0,153,255)
        }
      }
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

void PointCloudToImage::projection_callback_sub_ (const sensor_msgs::PointCloud2ConstPtr& input_cloud, const sensor_msgs::PointCloud2ConstPtr& ng_cloud, const sensor_msgs::ImageConstPtr& input_image, const sensor_msgs::Imu::ConstPtr& imu_msg){
    //ROS_INFO("callback"); 
    // Convert pc to pcl::PointXYZ
    pcl::PCLPointCloud2::Ptr input_cloud_pcl (new pcl::PCLPointCloud2 ());
    pcl_conversions::toPCL(*input_cloud, *input_cloud_pcl);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::fromPCLPointCloud2(*input_cloud_pcl, *cloud_raw);

    pcl::PCLPointCloud2::Ptr ng_cloud_pcl (new pcl::PCLPointCloud2 ());
    pcl_conversions::toPCL(*ng_cloud, *ng_cloud_pcl);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ng_cloud_raw (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::fromPCLPointCloud2(*ng_cloud_pcl, *ng_cloud_raw);
    
    // Invert pointcloud back to original attitude
    double q0_in, q1_in, q2_in, q3_in; 
    q0_in=imu_msg->orientation.w;
    q1_in=imu_msg->orientation.x;
    q2_in=imu_msg->orientation.y;
    q3_in=imu_msg->orientation.z;
    Affine3d transform;
    quaternionToMatrixInv(q0_in, q1_in, q2_in, q3_in, transform);
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ()); 
    pcl::transformPointCloud (*cloud_raw, *cloud_raw, transform);
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ng (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::transformPointCloud (*ng_cloud_raw, *ng_cloud_raw, transform);

    try{
      image_bridge_ = cv_bridge::toCvCopy(input_image, "bgr8");
    }
    catch (cv_bridge::Exception& ex)
    {
      ROS_ERROR("Failed to convert image");
      return;
    }
    cv::Mat OutImage = cv::Mat::zeros(image_bridge_->image.size(), image_bridge_->image.type());
    maskGround(cloud_raw, ng_cloud_raw, image_bridge_, OutImage);
    image_bridge_->image = OutImage;
    
    // publish output image
    image_pub_.publish(image_bridge_->toImageMsg());
}

void PointCloudToImage::projection_callback_ (const sensor_msgs::PointCloud2ConstPtr& input_cloud, const sensor_msgs::ImageConstPtr& input_image, const sensor_msgs::Imu::ConstPtr& imu_msg){
    //ROS_INFO("callback"); 
    // Convert pc to pcl::PointXYZ
    pcl::PCLPointCloud2::Ptr input_cloud_pcl (new pcl::PCLPointCloud2 ());
    pcl_conversions::toPCL(*input_cloud, *input_cloud_pcl);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::fromPCLPointCloud2(*input_cloud_pcl, *cloud_raw);

    pcl::PointCloud<pcl::PointXYZ>::Ptr ng_cloud_raw (new pcl::PointCloud<pcl::PointXYZ> ());

    // Invert pointcloud back to original attitude
    double q0_in, q1_in, q2_in, q3_in; 
    q0_in=imu_msg->orientation.w;
    q1_in=imu_msg->orientation.x;
    q2_in=imu_msg->orientation.y;
    q3_in=imu_msg->orientation.z;
    Affine3d transform;
    quaternionToMatrixInv(q0_in, q1_in, q2_in, q3_in, transform);
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ()); 
    pcl::transformPointCloud (*cloud_raw, *cloud_raw, transform);

    try{
      image_bridge_ = cv_bridge::toCvCopy(input_image, "bgr8");
    }
    catch (cv_bridge::Exception& ex)
    {
      ROS_ERROR("Failed to convert image");
      return;
    }
    cv::Mat OutImage = cv::Mat::zeros(image_bridge_->image.size(), image_bridge_->image.type());
    maskGround(cloud_raw, ng_cloud_raw, image_bridge_, OutImage);
    image_bridge_->image = OutImage;
    
    // publish output image
    image_pub_.publish(image_bridge_->toImageMsg());
}

int main (int argc, char** argv) {
    ros::init(argc, argv, "PointCloudToImage");
    PointCloudToImage node;
    ros::MultiThreadedSpinner spinner(4); // Use 4 threads
    spinner.spin();
    // ros::spin();
    return 0;
 }
