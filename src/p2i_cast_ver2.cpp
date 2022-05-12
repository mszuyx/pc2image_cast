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
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
// Import realsense camera msg type
#include <realsense2_camera/Extrinsics.h>

using namespace message_filters;
using namespace Eigen;

class PointCloudToImage{
public:
    PointCloudToImage();
private:
    // Declare sub & pub
    ros::NodeHandle node_handle_;
    ros::Publisher image_pub_;
    ros::Publisher depth_pub_;
    ros::Publisher depth_info_pub_;
    ros::Subscriber depth_info_sub_;
    ros::Subscriber cam_info_sub_;
    ros::Subscriber extrinsics_sub_;

    // Sync settings
    message_filters::Subscriber<sensor_msgs::PointCloud2> points_sub_;
    message_filters::Subscriber<sensor_msgs::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_node_sub_;

    typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image, sensor_msgs::Imu> RSSyncPolicy;
    typedef Synchronizer<RSSyncPolicy> Sync;
    boost::shared_ptr<Sync> sync;

    // Declare functions
    void projection_callback (const sensor_msgs::PointCloud2ConstPtr& input_cloud, const sensor_msgs::ImageConstPtr& input_image, const sensor_msgs::Imu::ConstPtr& imu_msg);
    void depth_info_callback(const sensor_msgs::CameraInfoConstPtr& info_msg);
    void cam_info_callback(const sensor_msgs::CameraInfoConstPtr& info_msg);
    void extrinsics_callback(const realsense2_camera::ExtrinsicsConstPtr info_msg);
    void maskGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, cv_bridge::CvImagePtr& image, cv::Mat Outimage);
    void quaternionToMatrixInv(double q0, double q1, double q2, double q3, Affine3d& transform);
    
    bool get_depth_info = false;
    bool get_cam_info = false;
    bool get_extrinsics = false;
    cv_bridge::CvImagePtr image_bridge_;
    image_geometry::PinholeCameraModel depth_model_;
    float centre_x, centre_y, focal_x, focal_y;
    int depth_height, depth_width;
};

PointCloudToImage::PointCloudToImage():node_handle_("~"){
    // Init ROS related
    ROS_INFO("Inititalizing PointCloud To Image Node...");
   
    // Subscribe to realsense topic
    points_sub_.subscribe(node_handle_, "/points_in", 1); // 10
    image_sub_.subscribe(node_handle_, "/image_in", 1);    // 5
    imu_node_sub_.subscribe(node_handle_, "/imu/data", 1); // 100
    
    // ApproximateTime takes a queue size as its constructor argument, hence RSSyncPolicy(xx)
    sync.reset(new Sync(RSSyncPolicy(20), points_sub_, image_sub_, imu_node_sub_));   //20
    sync->registerCallback(boost::bind(&PointCloudToImage::projection_callback, this, _1, _2, _3));
    
    depth_info_sub_ = node_handle_.subscribe("/depth_info_in", 1, &PointCloudToImage::depth_info_callback, this);
    cam_info_sub_ = node_handle_.subscribe("/camera_info_in", 1, &PointCloudToImage::cam_info_callback, this);
    extrinsics_sub_ = node_handle_.subscribe("/depth_to_color", 1, &PointCloudToImage::extrinsics_callback, this);

    // Publish Init
    std::string image_out_topic;
    node_handle_.param<std::string>("image_out_topic", image_out_topic, "/image_out");
    ROS_INFO("masked RGB image is published in: %s", image_out_topic.c_str());
    image_pub_ = node_handle_.advertise<sensor_msgs::Image>(image_out_topic, 1);

    std::string depth_out_topic;
    node_handle_.param<std::string>("depth_out_topic", depth_out_topic, "/depth_out");
    ROS_INFO("masked depth image is published in: %s", depth_out_topic.c_str());
    depth_pub_ = node_handle_.advertise<sensor_msgs::Image>(depth_out_topic, 1);

    std::string depth_info_topic;
    node_handle_.param<std::string>("image_topic", depth_info_topic, "/depth_info_out");
    ROS_INFO("depth camera info is published in: %s", depth_info_topic.c_str());
    depth_info_pub_ = node_handle_.advertise<sensor_msgs::CameraInfo>(depth_info_topic, 1);
}

void PointCloudToImage::depth_info_callback(const sensor_msgs::CameraInfoConstPtr& info_msg){
    if (get_depth_info){
      depth_info_sub_.shutdown();
    }else{
    depth_height = info_msg->height;
    depth_width = info_msg->width;
    depth_model_.fromCameraInfo(info_msg);
    centre_x = float(depth_model_.cx());
    centre_y = float(depth_model_.cy());
    focal_x = float(depth_model_.fx());
    focal_y = float(depth_model_.fy());
    get_depth_info = true;
    ROS_INFO("Got depth info!");}
  }

void PointCloudToImage::cam_info_callback(const sensor_msgs::CameraInfoConstPtr& info_msg){
    if (get_cam_info){
      cam_info_sub_.shutdown();
    }else{

    get_cam_info = true;
    ROS_INFO("Got camera info!");}
  }

void PointCloudToImage::extrinsics_callback(const realsense2_camera::ExtrinsicsConstPtr info_msg){
    if (get_extrinsics){
      extrinsics_sub_.shutdown();
    }else{

    get_extrinsics = true;
    ROS_INFO("Got extrinsics info!");}
  }

void PointCloudToImage::maskGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, cv_bridge::CvImagePtr& image, cv::Mat Outimage){  
    if (get_depth_info == true){
      cv::Mat mask = cv::Mat(depth_height, depth_width, CV_32FC1,cv::Scalar(std::numeric_limits<float>::max()));
      // cv::Mat mask = cv::Mat::zeros(depth_height, depth_width, CV_16FC1);

      for (unsigned int i=0; i < cloud->points.size();i++){
        if (cloud->points[i].z == cloud->points[i].z){
            float z = cloud->points[i].z*1000.0;
            float u = (cloud->points[i].x*1000.0*focal_x) / z;
            float v = (cloud->points[i].y*1000.0*focal_y) / z;
            int pixel_pos_x = (int)(u + centre_x);
            int pixel_pos_y = (int)(v + centre_y);
        if (pixel_pos_x > (depth_width-1)){
          pixel_pos_x = depth_width -1;
        }
        if (pixel_pos_y > (depth_height-1)){
          pixel_pos_y = depth_height-1;
        }
        mask.at<float>(pixel_pos_y,pixel_pos_x) = z;
        }       
      }

      cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)),  cv::Point(-1, -1), 1);
      // cv::dilate(mask, mask, cv::Mat(2,2), cv::Point(-1, -1), 2, 1, 1);

      mask.convertTo(mask,CV_16UC1);
      sensor_msgs::ImagePtr output_image = cv_bridge::CvImage(std_msgs::Header(), "16UC1", mask).toImageMsg();
      depth_pub_.publish(output_image);
    }
  
    // image->image.copyTo(Outimage, mask);
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
    transform = AngleAxisd(-pitch, Vector3d::UnitZ()) * AngleAxisd(-(roll+1.5708), Vector3d::UnitX());  // Choose wisely to prevent gimbal local
}



void PointCloudToImage::projection_callback(const sensor_msgs::PointCloud2ConstPtr& input_cloud, const sensor_msgs::ImageConstPtr& input_image, const sensor_msgs::Imu::ConstPtr& imu_msg){
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
    maskGround(cloud_raw, image_bridge_, OutImage);
    // image_bridge_->image = OutImage;
    
    // publish output image
    // image_pub_.publish(image_bridge_->toImageMsg());
}

int main (int argc, char** argv) {
    ros::init(argc, argv, "PointCloudToImage");
    PointCloudToImage node;
    ros::spin();
    return 0;
 }
