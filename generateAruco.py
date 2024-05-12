import numpy as np
import cv2

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000

}

aruco_type = "DICT_5X5_1000"
id = 1

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

print("ArUco type '{}' with ID '{}'".format(aruco_type, id))
tag_size = 1000
tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
cv2.aruco.drawDetectedMarkers(arucoDict, id, tag_size, tag, 1)

# Save the tag g
# generated
tag_name = "arucoMarkers/" + aruco_type + "_" + str(id) + ".png"
cv2.imwrite(tag_name, tag)
cv2.imshow("ArUco Tag", tag)

cv2.waitKey(0)

cv2.destroyAllWindows