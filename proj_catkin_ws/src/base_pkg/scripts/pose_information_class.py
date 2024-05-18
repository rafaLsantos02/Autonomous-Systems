#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np

class PoseExtractor:
    def __init__(self):
        rospy.init_node('pose_extractor', anonymous=True)
        self.pose_list = []

        # Load camera matrix and distortion coefficients
        self.cam_mat = np.load("/home/artur-ubunto/Desktop/SAut/arucos_tut/calib_data.npz")["camMatrix"]
        self.dist_coef = np.load("/home/artur-ubunto/Desktop/SAut/arucos_tut/calib_data.npz")["distCoef"]

        self.pose_sub = rospy.Subscriber('/pose', Odometry, self.pose_callback)
        self.vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.vel_callback)

    def pose_callback(self, msg):
        pose_data = {
            'position': (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
            'orientation': (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w),
            'frame_id': msg.header.frame_id,
            'timestamp': msg.header.stamp.to_sec()
        }
        self.pose_list.append(pose_data)

    def vel_callback(self, msg):
        if self.pose_list:
            vel_data = {
                'linear_velocity': (msg.linear.x, msg.linear.y, msg.linear.z),
                'angular_velocity': (msg.angular.x, msg.angular.y, msg.angular.z)
            }
            for key, value in vel_data.items():
                self.pose_list[-1].setdefault(key, value)

    def process_pose(self):
        while not rospy.is_shutdown():
            if self.pose_list:
                # Print pose data
                for pose_data in self.pose_list:
                    print("Position:", pose_data['position'])
                    print("Orientation:", pose_data['orientation'])
                    print("Linear Velocity:", pose_data.get('linear_velocity', None))
                    print("Angular Velocity:", pose_data.get('angular_velocity', None))
                    print("Timestamp:", pose_data['timestamp'])
                    print("Frame ID:", pose_data['frame_id'])
                    print("---")
                
                # Clear the pose list after printing
                self.pose_list = []

            rospy.sleep(1)  # Print every second

if __name__ == "__main__":
    print("ready")
    pose_extractor = PoseExtractor()
    pose_extractor.process_pose()
