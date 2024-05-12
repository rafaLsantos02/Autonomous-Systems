import numpy as np
import time
import cv2

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000

}


def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:

        ids = ids.flatten()

        for (markerCorner, markerId) in zip(corners, ids):

            corners = markerCorner.reshape((4,2))
            (topLeft, topRigth, bottomRigth, bottomLeft) = corners

            topRigth = (int(topRigth[0]), int(topRigth[1]))
            bottomRigth = (int(bottomRigth[0]), int(bottomRigth[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRigth, (0, 255, 0))
            cv2.line(image, topRigth, bottomRigth, (0, 255, 0))
            cv2.line(image, bottomRigth, bottomLeft, (0, 255, 0))
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0))

            cX = int((topLeft[0] + bottomRigth[0]) / 2.0)
            cY = int((topLeft[1] + bottomRigth[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(image, str(markerId), (topLeft[0], topLeft[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(markerId))

        return image
    



aruco_type = "DICT_5X5_1000"

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters_create()

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():

    ret, img = cap.read()

    h, w, _ = img.shape

    width = 1000
    height = int(width*(h/w))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)

    detected_markers = aruco_display(corners, ids, rejected, img)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()


