#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
from cv2 import aruco
import numpy as np
import scipy.spatial.transform as sst
import time

last_print_time = time.time()

calib_data_path1 = r"/home/artur-ubunto/Desktop/Autonomous-Systems/proj_catkin_ws/src/base_pkg/scripts/calib_data.npz"

calib_data1 = np.load(calib_data_path1)

print(calib_data1.files)

cam_mat = calib_data1["camMatrix"]
dist_coef = calib_data1["distCoef"]
r_vectors = calib_data1["rVector"]
t_vectors = calib_data1["tVector"]


MARKER_SIZE = 17

marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
param_markers = aruco.DetectorParameters_create()

def process_image(cv_image):
    global last_print_time
    gray_cv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_cv_image, marker_dict, parameters=param_markers
    )
    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef
        )
        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv.polylines(
                cv_image, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )

            rMat, _ = cv.Rodrigues(rVec[i])
            _, _, _, angleZ, angleY, angleX = cv.RQDecomp3x3(rMat)

            r = sst.Rotation.from_matrix(rMat)
            euler_angles = r.as_euler('xyz', degrees=True)

            marker_vec = tVec[i][0]
            forward_vec = np.array([0,0,1])
            cos_angle = np.dot(marker_vec, forward_vec) / (np.linalg.norm(marker_vec) * np.linalg.norm(forward_vec))
            angle = np.arccos(cos_angle)

            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            # calculate the distance
            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
            )

            #while not rospy.is_shutdown():
            #    print(f"id: {ids[0]}      Dist: {round(distance, 2)}      Angle: {np.degrees(angle):.2f}")
            #    print(f"Vec: {np.round(marker_vec[0])} {np.round(marker_vec[1])} {np.round(marker_vec[2])}")
            #    print("---\n")

            #    rospy.sleep(1)

            # Only print the messages if its information is different than the previous
            if time.time() - last_print_time >= 1:
                last_print_time = time.time()
                print(f"id: {ids[0]}      Dist: {round(distance, 2)}      Angle: {np.degrees(angle):.2f}")
                print(f"Vec: {np.round(marker_vec[0])} {np.round(marker_vec[1])} {np.round(marker_vec[2])}")
                print("---\n")

            # for pose of the marker
            cv.putText(
                cv_image,
                f"id: {ids[0]} Dist: {round(distance, 2)}",
                tuple(corners[0]),
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                cv_image,
                f"Angle: {np.degrees(angle):.2f}",
                (corners[0][0], corners[0][1] + 20),
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (255, 0, 0),
                2,
                cv.LINE_AA,
            )

    cv.imshow("cv_image", cv_image)
    cv.waitKey(1)

def image_callback(data):
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        process_image(cv_image)
    except CvBridgeError as e:
        print(e)

def main():
    rospy.init_node('image_processor', anonymous=True)
    rospy.Subscriber("/camera/rgb/image_color", Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
