#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import cv2 as cv
from cv2 import aruco
import numpy as np
import scipy.spatial.transform as sst
import time

import threading
import pygame
import pygame.gfxdraw
from python_ugv_sim.utils import environment, vehicles
import pdb
import random

from plot import plot

last_print_time = time.time()

MARKER_SIZE = 17

# marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
param_markers = aruco.DetectorParameters_create()

# ---> Robot position by pionner at real time 
pos_x = 0
pos_y = 0
pos_z = 0
orientation = 0

# ---> Odometry variable
odom = [0, 0]

# ---> Auxiliar variable to guarante that odom is rigth calculated
count = 0

# ---> List with all observations at real time 
observations = []

# ---> Calib Camera
calib_data_path1 = r"/mnt/c/Users/PC/Documents/Universidade/Ano 3 - 2º semestre/SA/Projeto/ekf_slam_demo-main/calib_data.npz"
calib_data1 = np.load(calib_data_path1)

cam_mat = calib_data1["camMatrix"]
dist_coef = calib_data1["distCoef"]
r_vectors = calib_data1["rVector"]
t_vectors = calib_data1["tVector"]


def acquire_observations(cv_image): #ex: process_image
    
    global observations 
    observations = []

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

            # calculate the distance in cm
            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
            )

            aruco_id = ids[0]
            aruco_dist = distance / 100 # Distance in m 
            aruco_angle = np.arccos(cos_angle) # Angle in rad
            

            observations.append(( aruco_dist, aruco_angle, aruco_id))

            
            # marks in reproduced video
            cv.putText(
                cv_image,
                f"id: {aruco_id} Dist: {round(aruco_dist, 2)}",
                tuple(corners[0]),
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                cv_image,
                f"Angle: {np.degrees(aruco_angle):.2f}",
                (corners[0][0], corners[0][1] + 20),
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (255, 0, 0),
                2,
                cv.LINE_AA,
            )            
            

    cv.imshow("cv_image1", cv_image)
    cv.waitKey(1)

    #print from observations that the robot does in each scan
    '''
    global count
    # Iterar sobre zs e desempacotar os valores
    for z in observations:
        (dist, phi, lidx) = z
        print(f"nUMERO: {count}")
        print(f"dist: {dist}, angulo: {phi}, index: {lidx}")
        
    count+=1
    '''
    return 

def image_callback(data):

    bridge = CvBridge()

    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        acquire_observations(cv_image)

    except CvBridgeError as e:
        print(e)

def pose_callback(msg):

  
    global pos_x, pos_y, pos_z, orientation, odom, count
    
    pose_data = {
        'position': (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
        'orientation': (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w),
        'frame_id': msg.header.frame_id,
        'timestamp': msg.header.stamp.to_sec()
    }

    new_pos_x = pose_data['position'][0]
    new_pos_y = pose_data['position'][1]
    new_pos_z = pose_data['position'][2]

    orient_x = pose_data['orientation'][0]
    orient_y = pose_data['orientation'][1]
    orient_z = pose_data['orientation'][2]
    orient_w = pose_data['orientation'][3]
    
    new_orientation = ( np.arctan2(2*(orient_w*orient_z + orient_x*orient_y), 1 - 2*(orient_y**2 + orient_z**2)) )


    #odom[0] = np.sqrt( (new_pos_x - pos_x) **2  +  (new_pos_y - pos_y) **2)        
    #odom[1] = new_orientation - orientation

    pos_x = new_pos_x
    pos_y = new_pos_y
    pos_z = new_pos_z
    orientation = new_orientation

    count = count + 1

    return


'''
EKF SLAM demo
Logic:
    - Prediction update
        - From control inputs, how do we change our state estimate?
        - Moving only changes the state estimate of the robot state, NOT landmark location
        - Moving affects uncertainty of the state
    - Observation update
        - From what we observe, how do we change our state estimation?
        - We reconcile prediction uncertainty and observation uncertainty into a single estimate
          that is more certain than before
'''

# <------------------------- EKF SLAM STUFF --------------------------------->
# ---> Robot Parameters
n_state = 3 # Number of state variables

# ---> Landmarks Variables
n_landmarks = 0
list_landmarks = []

# ---> Noise parameters
R = np.diag([0.002,0.002,0.002]) # sigma_x, sigma_y, sigma_theta
Q = np.diag([0.003,0.005]) # sigma_r, sigma_phi

# ---> EKF Estimation Variables
mu = np.zeros((n_state,1)) # State estimate (robot pose and landmark positions)
sigma = np.zeros((n_state, n_state)) # State uncertainty, covariance matrix

''' didnt really understood mu and sigma, maybe needs some change'''

# ---> Helpful matrix
Fx = np.block([[np.eye(3),np.zeros((n_state,2*n_landmarks))]]) # Used in both prediction and measurement updates

# ---> Search the position of the id of the landmark and if the landmark already exists return 1 else 0
def id_search(list_landmarks, id):

    global n_landmarks

    for i in range(0, len(list_landmarks)):
        if list_landmarks[i] == id:
            return i, 1
        
    list_landmarks.append(id)
    n_landmarks = len(list_landmarks)
    return n_landmarks-1, 0


# <------------------------- EKF SLAM STUFF --------------------------------->

# ---> EKF SLAM steps
def prediction_update(mu,sigma,odom):
    '''
    This function performs the prediction step of the EKF. Using the linearized motion model, it
    updates both the state estimate mu and the state uncertainty sigma based on the model and known
    control inputs to the robot.
    Inputs:
     - mu: state estimate (robot pose and landmark positions)
     - sigma: state uncertainty (covariance matrix)
     - odom_data: deltaD and deltaO
     - dt: discretization time of continuous model
    Outpus:
     - mu: updated state estimate
     - sigma: updated state uncertainty
    '''
    rx,py,theta = mu[0],mu[1],mu[2]
    deltaD,deltaO = odom[0], odom[1]
    # Update state estimate mu with model
    state_model_mat = np.zeros((n_state,1)) # Initialize state update matrix from model
    state_model_mat[0] = deltaD*np.cos(theta) # Update in the robot x position
    state_model_mat[1] = deltaD*np.sin(theta) # Update in the robot y position
    state_model_mat[2] = deltaO # Update for robot heading theta
    mu = mu + np.matmul(np.transpose(Fx),state_model_mat) # Update state estimate, simple use model with current state estimate
    
    mu[2] =np.arctan2(np.sin(mu[2]),np.cos(mu[2])) # Keep the angle between -pi and +pi

    # Update state uncertainty sigma
    state_jacobian = np.zeros((3,3)) # Initialize model jacobian
    state_jacobian[0,2] = -deltaD*np.sin(theta) # Jacobian element, how small changes in robot theta affect robot x
    state_jacobian[1,2] = deltaD*np.cos(theta) # Jacobian element, how small changes in robot theta affect robot y
    G = np.eye(sigma.shape[0]) + np.transpose(Fx).dot(state_jacobian).dot(Fx) # How the model transforms uncertainty
    sigma = G.dot(sigma).dot(np.transpose(G)) + np.transpose(Fx).dot(R).dot(Fx) # Combine model effects and stochastic noise #DUVIDA, aqui devo adicionar o ruído ou como já adicionei na odometria já não preciso colocar aqui
    return mu,sigma

def measurement_update(mu,sigma,zs):
    '''
    This function performs the measurement step of the EKF. Using the linearized observation model, it
    updates both the state estimate mu and the state uncertainty sigma based on range and bearing measurements
    that are made between robot and landmarks.
    Inputs:
     - mu: state estimate (robot pose and landmark positions)
     - sigma: state uncertainty (covariance matrix)
     - zs: list of 3-tuples, (dist,phi,lidx) from measurement function
    Outpus:
     - mu: updated state estimate
     - sigma: updated state uncertainty
    '''
    rx,ry,theta = mu[0, 0],mu[1, 0],mu[2, 0] # robot 
    delta_zs = [np.zeros((2,1)) for i in range(n_landmarks)] # A list of how far an actual measurement is from the estimate measurement
    Ks = [np.zeros((mu.shape[0],2)) for i in range(n_landmarks)] # A list of matrices stored for use outside the measurement for loop
    Hs = [np.zeros((2,mu.shape[0])) for i in range(n_landmarks)] # A list of matrices stored for use outside the measurement for loop
    for z in zs:
        (dist,phi,id) = z

        id_index, id_existence = id_search(list_landmarks, id) # Search the indice of the id and if not exist return to the variable id_existence = 0
        mu_landmark = np.zeros((2,1))

        # Verify if the landmark is already know by the robot
        if id_existence == 1:
            mu_landmark = mu[n_state+id_index*2:n_state+id_index*2+2]
        else:
            mu_landmark[0] = rx + dist*np.cos(phi+theta) # lx, x position of landmark
            mu_landmark[1] = ry+ dist*np.sin(phi+theta) # ly, y position of landmark
            mu = np.block([[mu], [mu_landmark]]) # Attach the new landmark to the matrix mu

            # Create the auxiliar matrix to attach to the sigma 
            aux1 = np.zeros((sigma.shape[0],2))
            aux2 = np.zeros((2,sigma.shape[1]))
            aux3 = 100 * np.eye(2)
            sigma = np.block([[sigma, aux1],
                              [aux2, aux3]])
            
            # Update the auxiliar matrix Fx
            global Fx
            Fx = np.block([[Fx, np.zeros((n_state,2))]])
            
            # Update the size of the auxiliar matrixes (delta_zs, Ks, Hs)
            delta_zs = [np.zeros((2,1)) for i in range(n_landmarks)] # A list of how far an actual measurement is from the estimate measurement
            Ks = [np.zeros((mu.shape[0],2)) for i in range(n_landmarks)] # A list of matrices stored for use outside the measurement for loop
            Hs = [np.zeros((2,mu.shape[0])) for i in range(n_landmarks)] # A list of matrices stored for use outside the measurement for loop


        delta  = mu_landmark - np.array([[rx],[ry]]) # Helper variable
        q = np.linalg.norm(delta)**2 # Helper variable

        dist_est = np.sqrt(q) # Distance between robot estimate and and landmark estimate, i.e., distance estimate
        phi_est = np.arctan2(delta[1,0],delta[0,0])-theta; phi_est = np.arctan2(np.sin(phi_est),np.cos(phi_est)) # Estimated angled between robot heading and landmark
        z_est_arr = np.array([[dist_est],[phi_est]]) # Estimated observation, in numpy array
        z_act_arr = np.array([[dist],[phi]]) # Actual observation in numpy array
        delta_zs[id_index] = z_act_arr-z_est_arr # Difference between actual and estimated observation

        # Helper matrices in computing the measurement update
        Fxj = np.block([[Fx],[np.zeros((2,Fx.shape[1]))]])
        Fxj[n_state:n_state+2,n_state+2*id_index:n_state+2*id_index+2] = np.eye(2)
        H = np.array([[-delta[0,0]/np.sqrt(q),-delta[1,0]/np.sqrt(q),0,delta[0,0]/np.sqrt(q),delta[1,0]/np.sqrt(q)],\
                      [delta[1,0]/q,-delta[0,0]/q,-1,-delta[1,0]/q,+delta[0,0]/q]])
        H = H.dot(Fxj)
        Hs[id_index] = H # Added to list of matrices
        Ks[id_index] = sigma.dot(np.transpose(H)).dot(np.linalg.inv(H.dot(sigma).dot(np.transpose(H)) + Q)) # Add to list of matrices
        
    # After storing appropriate matrices, perform measurement update of mu and sigma
    mu_offset = np.zeros(mu.shape) # Offset to be added to state estimate
    sigma_factor = np.eye(sigma.shape[0]) # Factor to multiply state uncertainty
    for id_index in range(n_landmarks):
        mu_offset += Ks[id_index].dot(delta_zs[id_index]) # Compute full mu offset
        sigma_factor -= Ks[id_index].dot(Hs[id_index]) # Compute full sigma factor
    mu = mu + mu_offset # Update state estimate
    sigma = sigma_factor.dot(sigma) # Update state uncertainty
    return mu,sigma
# <------------------------- EKF SLAM STUFF --------------------------------->


if __name__ == '__main__':

    rospy.init_node('image_processor', anonymous=True)
    rospy.Subscriber("/camera/rgb/image_color", Image, image_callback)
    rospy.Subscriber('/pose', Odometry, pose_callback)
    #rospy.spin()

    # Initialize robot 
    x_init = np.array([0.425, 3.8, 2.6179938779914944]) # px, py, theta

    # Initialize robot state estimate and sigma
    mu[0:3] = np.expand_dims(x_init,axis=1)
    sigma[0:3,0:3] = 0.1*np.eye(3)
    sigma[2,2] = 0 

    # Initialize auxiliar variables
    last_pos_x, last_pos_y, last_orientation = 0, 0, 0
    
    # Initialize rospy.rate so that while runs at the same speed always
    rate = rospy.Rate(9.8)  

    # Loop to ensure that the odom is well calculated 
    while count < 2:
        last_pos_x, last_pos_y, last_orientation = pos_x, pos_y, orientation

    running = True

    while running:

        odom[0] = np.sqrt( (pos_x - last_pos_x) **2  +  (pos_y - last_pos_y) **2)
        odom[1] = orientation - last_orientation

        last_pos_x, last_pos_y, last_orientation = pos_x, pos_y, orientation
             
        # Get measurements
        zs = observations
        # zs = zs.append((distances_list[-1],angles_list[-1],ids_list[-1]))

        # EKF Slam Logic
        mu, sigma = prediction_update(mu,sigma,odom) # Perform EKF prediction update
        mu, sigma = measurement_update(mu,sigma,zs) # Perform EKF measurement update
        
        # Plotting
        plot(mu, sigma, n_landmarks)

        print(f"X: {pos_x}, Y: {pos_y}, \n")
        print(f"deltaD: {odom[0]}, deltaO: {odom[1]} \n")
        print("mu = ", mu[:3,0], "\n ---")

        # Sleep to maintain loop rate
        rate.sleep()



