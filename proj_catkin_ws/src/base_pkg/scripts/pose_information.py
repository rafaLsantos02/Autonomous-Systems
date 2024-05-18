#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np

pose_list = []

def pose_callback(msg):
    pose_data = {
        'position': (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
        'orientation': (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w),
        'frame_id': msg.header.frame_id,
        'timestamp': msg.header.stamp.to_sec()
    }
    pose_list.append(pose_data)

def vel_callback(msg):
    global pose_list
    if pose_list:
        #print("Received Twist message:", msg)
        vel_data = {
            'linear_velocity': (msg.linear.x, msg.linear.y, msg.linear.z),
            'angular_velocity': (msg.angular.x, msg.angular.y, msg.angular.z)
        }
        for key, value in vel_data.items():
            pose_list[-1].setdefault(key, value)



def process_pose():
    global pose_list
    rospy.init_node('pose_extractor', anonymous=True)

    # Load camera matrix and distortion coefficients
    cam_mat = np.load("/home/artur-ubunto/Desktop/SAut/arucos_tut/calib_data.npz")["camMatrix"]
    dist_coef = np.load("/home/artur-ubunto/Desktop/SAut/arucos_tut/calib_data.npz")["distCoef"]

    rospy.Subscriber('/pose', Odometry, pose_callback)
    rospy.Subscriber('/cmd_vel', Twist, vel_callback)

    while not rospy.is_shutdown():
        if pose_list:
            # Print pose data
            for pose_data in pose_list:
                print("Position:", pose_data['position'])
                print("Orientation:", pose_data['orientation'])
                print("Linear Velocity:", pose_data.get('linear_velocity', None))
                print("Angular Velocity:", pose_data.get('angular_velocity', None))
                print("Timestamp:", pose_data['timestamp'])
                print("Frame ID:", pose_data['frame_id'])
                print("---")
            
            # Clear the pose list after printing
            pose_list = []

        rospy.sleep(1)  # Print every second

if __name__ == "__main__":
    print("ready")
    process_pose()
