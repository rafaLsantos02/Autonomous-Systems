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

calib_data_path1 = r"/home/artur-ubunto/Desktop/SAut/arucos_tut/calib_data.npz"

calib_data1 = np.load(calib_data_path1)

print(calib_data1.files)

cam_mat = calib_data1["camMatrix"]
dist_coef = calib_data1["distCoef"]
r_vectors = calib_data1["rVector"]
t_vectors = calib_data1["tVector"]


MARKER_SIZE = 17

marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
param_markers = aruco.DetectorParameters_create()

class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.callback)
        self.prev_coords = None
        self.prev_distance = None
        self.prev_angles = None

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.process_image(cv_image)
        except CvBridgeError as e:
            print(e)

    def process_image(self, cv_image):
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


                #print(f"id: {ids[0]} Dist: {round(distance, 2)}")
                #print(f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ")
                #print(f"Roll: {np.round(euler_angles[0], 2)} Pitch: {np.round(euler_angles[1], 2)} Yaw: {np.round(euler_angles[2], 2)}",)
                #print(f"Angle: {np.degrees(angle):.2f}")
                #print(f"Vec: {np.round(tVec[i][0][0])} {np.round(tVec[i][0][1])} {np.round(tVec[i][0][2])}")


                # only prints the messages if its information is different than the previous
                marker_vec = tVec[i][0]
                if self.prev_coords is not None:
                    diff = np.abs(marker_vec - self.prev_coords)
                    if np.any(diff >= 1):
                        print(f"id: {ids[0]}      Dist: {round(distance, 2)}      Angle: {np.degrees(angle):.2f}")
                        #print(f"x:{round(marker_vec[0], 1)} y: {round(marker_vec[1], 1)} ")
                        #print(f"Roll: {np.round(euler_angles[0], 2)} Pitch: {np.round(euler_angles[1], 2)} Yaw: {np.round(euler_angles[2], 2)}")
                        print(f"Vec: {np.round(marker_vec[0])} {np.round(marker_vec[1])} {np.round(marker_vec[2])}")
                        print("---\n")
                self.prev_coords = marker_vec.copy()


                # for pose of the marker
                #point = cv.aruco.drawAxis(cv_image, cam_mat, dist_coef, rVec[i], tVec[i], 4)

                cv.putText(
                    cv_image,
                    f"id: {ids[0]} Dist: {round(distance, 2)}",
                    top_right,
                    cv.FONT_HERSHEY_PLAIN,
                    1.3,
                    (0, 0, 255),
                    2,
                    cv.LINE_AA,
                )
                # cv.putText(
                #     cv_image,
                #     f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                #     bottom_right,
                #     cv.FONT_HERSHEY_PLAIN,
                #     1.0,
                #     (0, 0, 255),
                #     2,
                #     cv.LINE_AA,
                #)
                cv.putText(
                    cv_image,
                    f"Angle: {np.degrees(angle):.2f}",
                    (bottom_left),
                    cv.FONT_HERSHEY_PLAIN,
                    1.3,
                    (255, 0, 0),
                    2,
                    cv.LINE_AA,
                )
        cv.imshow("cv_image", cv_image)
        cv.waitKey(1)

def main():
    rospy.init_node('image_processor', anonymous=True)
    ip = ImageProcessor()
    rospy.spin()

if __name__ == '__main__':
    main()
