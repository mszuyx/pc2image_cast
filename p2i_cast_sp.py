#!/usr/bin/env python
import rospy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array
import cv2, cv_bridge
from image_geometry import PinholeCameraModel
from fast_slic.avx2 import SlicAvx2
# from fast_slic import Slic
# from fast_slic.neon import SlicNeon
from scipy.spatial.transform import Rotation as R

class MaskGroundSP:

    def __init__(self, pc_topic_, image_topic_, info_topic_, imu_topic_, out_topic_):
        self.get_cam_info = False
        self.imgOutput = rospy.Publisher(out_topic_, Image, queue_size=1)
        self.image_topic_ = image_topic_
        self.pc_topic_ = pc_topic_
        self.imu_topic_ = imu_topic_
        rospy.Subscriber(info_topic_, CameraInfo, self.info_callback, queue_size=1)
        self.num_sp_ = 128
        self.slic = SlicAvx2(num_components=self.num_sp_, compactness=10, min_size_factor=0)

    def run(self):
        image_sub = Subscriber(self.image_topic_, Image)
        point_sub = Subscriber(self.pc_topic_, PointCloud2)
        imu_sub = Subscriber(self.imu_topic_, Imu)
        ts = ApproximateTimeSynchronizer([image_sub, point_sub, imu_sub], queue_size=20, slop=0.1)
        ts.registerCallback(self.img_callback)

    def info_callback(self,info_msg):
        if self.get_cam_info:
            return
        self.cam_model_ = PinholeCameraModel()
        self.cam_model_.fromCameraInfo(info_msg)
        self.get_cam_info  = True
        rospy.loginfo("Got image info!")

    def img_callback(self, img_msg, pc_msg, imu_msg):
        img = None
        bridge = cv_bridge.CvBridge()

        try:
            img = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except cv_bridge.CvBridgeError as e:
            rospy.logerr( 'image message to cv conversion failed :' )
            rospy.logerr( e )
            print( e )
            return

        pts = pointcloud2_to_xyz_array(pc_msg,True)
        num_pts = len(pts)
        if num_pts>0:
            rot = R.from_quat([imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w]).as_euler('zyx')
            pitch = (rot[2]+1.5708) # check here
            # print(pitch*180/3.1415926)
            rot_m = R.from_euler('zyx', [rot[0], 0, pitch]).as_matrix() 
            pts = np.dot(pts,rot_m)
            radius_ = 10
            mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
            # projected_pts = np.array([self.cam_model_.project3dToPixel(list(pt)) for pt in pts]).astype(int)
            for i in range(num_pts):
                if (i % 2) == 0:
                    projected_pts = self.cam_model_.project3dToPixel(pts[i])
                    radius_f = int(radius_/(0.3*(pts[i,2]+0.0001)))
                    if radius_f <= 2:
                        radius_f = 2
                    cv2.circle(mask, (int(projected_pts[0]),int(projected_pts[1])), radius_f, 1,-1)

            # sp_map = self.slic.iterate(img)
            # for i in range(self.num_sp_-1):
            #     sp_temp = sp_map == i
            #     likelihood = np.sum(mask[sp_temp])
            #     if likelihood > 200:
            #         mask[sp_temp] = 1
            #     else:
            #         mask[sp_temp] = 0

            img = cv2.bitwise_and(img, img, mask=mask)
        self.imgOutput.publish(bridge.cv2_to_imgmsg(img, 'bgr8'))


if __name__ == "__main__" :
    rospy.init_node('maskGroundSP')
    try :
        MSP = MaskGroundSP("/points_in", "/image_in", "/d455/color/camera_info", "/imu/data", "/masked_imageSP")
        MSP.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass