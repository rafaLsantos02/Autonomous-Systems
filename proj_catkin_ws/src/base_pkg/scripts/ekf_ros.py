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

from plot import plot, plot_firme

last_print_time = time.time()

calib_data_path1 = r"/mnt/c/Users/PC/Documents/Universidade/Ano 3 - 2º semestre/SA/Projeto/ekf_slam_demo-main/calib_data.npz"

calib_data1 = np.load(calib_data_path1)

print(calib_data1.files)

cam_mat = calib_data1["camMatrix"]
dist_coef = calib_data1["distCoef"]
r_vectors = calib_data1["rVector"]
t_vectors = calib_data1["tVector"]

MARKER_SIZE = 17

#marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
param_markers = aruco.DetectorParameters_create()

position_x_list = [0,0]
position_y_list = [0,0]
position_z_list = [0,0]
orientation_x_list = [0,0]
orientation_y_list = [0,0]
orientation_z_list = [0,0]
orientation_w_list = [0,0]

ids_list = [0]
distances_list = [0]
angles_list = [0]
marker_vecs_list = [0]


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
        print(total_markers, marker_IDs)
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

                # Append data to lists
            ids_list.append(ids[0])
                # ids_list = ids[0]
            distances_list.append(round(distance, 2))
            angles_list.append(np.degrees(angle))
            marker_vecs_list.append(np.round(marker_vec, decimals=2))

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

    return ids_list, distances_list, angles_list, marker_vecs_list

def image_callback(data):
    global ids_list, distances_list, angles_list, marker_vecs_list
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        ids_list, distances_list, angles_list, marker_vecs_list = process_image(cv_image)
    except CvBridgeError as e:
        print(e)

def pose_callback(msg):
    global position_x_list, position_y_list, position_z_list, orientation_x_list, orientation_y_list, orientation_z_list, orientation_w_list
    pose_data = {
        'position': (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
        'orientation': (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w),
        'frame_id': msg.header.frame_id,
        'timestamp': msg.header.stamp.to_sec()
    }
    position_x_list.append(pose_data['position'][0])
    #print(position_x_list)
    position_y_list.append(pose_data['position'][1])
    position_z_list.append(pose_data['position'][2])
    orientation_x_list.append(pose_data['orientation'][0])
    orientation_y_list.append(pose_data['orientation'][1])
    orientation_z_list.append(pose_data['orientation'][2])
    orientation_w_list.append(pose_data['orientation'][3])

    return pose_data

def process_pose():
    global position_x_list, position_y_list, position_z_list, orientation_x_list, orientation_y_list, orientation_z_list, orientation_w_list
    extracted_data = []

    # Load camera matrix and distortion coefficients
    cam_mat = np.load(calib_data_path1)["camMatrix"]
    dist_coef = np.load(calib_data_path1)["distCoef"]


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
robot_fov = 3 # robot field of view radius (m)

# ---> Landmark parameters
# Single landmark
# landmarks = [(7,7)] # position from lower-left corner of map (m,m)

# Scattered Landmarks
# landmarks = [(12,12),
#              (4,4),
#              (4,12),
#              (15,10),
#              (9,1)]

# Rectangular pattern
#landmarks = [(4,4),
#             (4,8),
#             (8,8),
#             (12,8),
#             (16,8),
#             (16,4),
#             (12,4),
#             (8,4)]

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
    for i in range(0, len(list_landmarks)):
        if list_landmarks[i] == id:
            return i, 1
        
    list_landmarks.append(id)
    global n_landmarks
    n_landmarks = len(list_landmarks)
    return n_landmarks-1, 0

# ---> Measurement function
def sim_measurement(x,landmarks):
    '''
    This function simulates a measurement between robot and landmark
    Inputs:
     - x: robot state (3x1 numpy array)
     - landmarks: list of 2-tuples, each of (lx,ly) actual position of landmark
    Outputs:
     - zs: list of 3-tuples, each (r,phi,lidx) of range (r) and relative bearing (phi) from robot to landmark,
           and lidx is the (known) correspondence landmark index.
    '''

    ''' maybe this needs to be done diffently maybe because the landmarks is updated throw time and not already set '''

    rx, ry, rtheta = x[0], x[1], x[2]
    zs = [] # List of measurements
    for (lidx,landmark) in enumerate(landmarks): # Iterate over landmarks and indices
        lx,ly = landmark
        dist = np.linalg.norm(np.array([lx-rx,ly-ry])) # distance between robot and landmark
        ''' distances_list[] '''
        phi = np.arctan2(ly-ry,lx-rx) - rtheta # angle between robot heading and landmark, relative to robot frame
        ''' angles_list[] '''
        phi = np.arctan2(np.sin(phi),np.cos(phi)) # Keep phi bounded, -pi <= phi <= +pi
        if dist<robot_fov: # Only append if observation is within robot field of view
            zs.append((dist,phi,lidx))
        ''' this wouldnt be needed '''
    return zs

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


# <------------------------- PLOTTING STUFF --------------------------------->
def show_robot_estimate(mu,sigma,env):
    '''
    Visualize estimated position and uncertainty of the robot
    Inputs:
     - mu: state estimate (robot pose and landmark positions)
     - sigma: state uncertainty (covariance matrix)
     - env: Environment class (from python_ugv_sim)
    '''
    rx,ry = mu[0],mu[1]
    p_pixel = env.position2pixel((rx,ry)) # Transform robot position to pygame surface pixel coordinates
    eigenvals,angle = sigma2transform(sigma[0:2,0:2]) # Get eigenvalues and rotation angle
    sigma_pixel = env.dist2pixellen(eigenvals[0]), env.dist2pixellen(eigenvals[1]) # Convert eigenvalue units from meters to pixels
    show_uncertainty_ellipse(env,p_pixel,sigma_pixel,angle) # Show the ellipse
    
def show_landmark_estimate(mu,sigma,env):
    '''
    Visualize estimated position and uncertainty of a landmark
    '''
    for lidx in range(n_landmarks): # For each landmark location
        lx,ly,lsigma = mu[n_state+lidx*2], mu[n_state+lidx*2+1], sigma[n_state+lidx*2:n_state+lidx*2+2,n_state+lidx*2:n_state+lidx*2+2]
        if ~np.isnan(lx): # If the landmark has been observed
            p_pixel = env.position2pixel((lx,ly)) # Transform landmark location to pygame surface pixel coordinates
            eigenvals,angle = sigma2transform(lsigma) # Get eigenvalues and rotation angle of covariance of landmark
            if np.max(eigenvals)<15: # Only visualize when the maximum uncertainty is below some threshold
                sigma_pixel = max(env.dist2pixellen(eigenvals[0]),5), max(env.dist2pixellen(eigenvals[1]),5) # Convert eigenvalue units from meters to pixels
                show_uncertainty_ellipse(env,p_pixel,sigma_pixel,angle) # Show the ellipse

def show_landmark_location(landmarks,env):
    '''
    Visualize actual landmark location
    '''
    for landmark in landmarks:
        lx_pixel, ly_pixel = env.position2pixel(landmark)
        r_pixel = env.dist2pixellen(0.2) # Radius of the circle for the ground truth locations of the landmarks
        pygame.gfxdraw.filled_circle(env.get_pygame_surface(),lx_pixel,ly_pixel,r_pixel,(0,255,255)) # Blit the circle onto the surface

def show_measurements(x,zs,env):
    '''
    Visualize measurements the occur between the robot and landmarks
    '''
    rx,ry = x[0], x[1]
    rx_pix, ry_pix = env.position2pixel((rx,ry)) # Convert robot position units from meters to pixels
    for z in zs: # For each measurement
        dist,theta,lidx = z # Unpack measurement tuple
        lx,ly = x[0]+dist*np.cos(theta+x[2]),x[1]+dist*np.sin(theta+x[2]) # Set the observed landmark location (lx,ly)
        lx_pix,ly_pix = env.position2pixel((lx,ly)) # Convert observed landmark location units from meters to pixels
        pygame.gfxdraw.line(env.get_pygame_surface(),rx_pix,ry_pix,lx_pix,ly_pix,(155,155,155)) # Draw a line between robot and observed landmark

def sigma2transform(sigma):
    '''
    Finds the transform for a covariance matrix, to be used for visualizing the uncertainty ellipse
    '''
    [eigenvals,eigenvecs] = np.linalg.eig(sigma) # Finding eigenvalues and eigenvectors of the covariance matrix
    angle = 180.*np.arctan2(eigenvecs[1][0],eigenvecs[0][0])/np.pi # Find the angle of rotation for the first eigenvalue
    return eigenvals, angle

def show_uncertainty_ellipse(env,center,width,angle):
    '''
    Visualize an uncertainty ellipse
    Adapted from: https://stackoverflow.com/questions/65767785/how-to-draw-a-rotated-ellipse-using-pygame
    '''
    target_rect = pygame.Rect(center[0]-int(width[0]/2),center[1]-int(width[1]/2),width[0],width[1])
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.ellipse(shape_surf, env.red, (0, 0, *target_rect.size), 2)
    rotated_surf = pygame.transform.rotate(shape_surf, angle)
    env.get_pygame_surface().blit(rotated_surf, rotated_surf.get_rect(center = target_rect.center))

# <------------------------- PLOTTING STUFF --------------------------------->

if __name__ == '__main__':

    rospy.init_node('image_processor', anonymous=True)
    rospy.Subscriber("/camera/rgb/image_color", Image, image_callback)
    rospy.Subscriber('/pose', Odometry, pose_callback)
    process_pose()
    #rospy.spin()

    # Initialize robot and discretization time step
    x_init = np.array([0.425, 3.8, 2.6179938779914944]) # px, py, theta
    odom = [0, 0]
    yaw1 = 0
    yaw2 = 0

    # Initialize robot state estimate and sigma
    mu[0:3] = np.expand_dims(x_init,axis=1)
    sigma[0:3,0:3] = 0.1*np.eye(3)
    sigma[2,2] = 0 
    running = True
    
    while running:
        
        odom[0] = np.sqrt( (position_x_list[-1]-position_x_list[-2]) **2 +  (position_y_list[-1]-position_y_list[-2]) **2)

        yaw1 = ( np.arctan2(2*(orientation_w_list[-1]*orientation_z_list[-1] + orientation_x_list[-1]*orientation_y_list[-1]), 1 - 2*(orientation_y_list[-1]**2 + orientation_z_list[-1]**2)) )

        yaw2 = ( np.arctan2(2*(orientation_w_list[-2]*orientation_z_list[-2] + orientation_x_list[-2]*orientation_y_list[-2]), 1 - 2*(orientation_y_list[-2]**2 + orientation_z_list[-2]**2)) )

        odom[1] = yaw1 - yaw2
        
        print(f"Pos_x: {position_x_list[-1]}, Pos_y: {position_y_list[-1]}, \n ---")
        print(f"deltaD: {odom[0]}, deltaO: {odom[1]}, \n ---")
        print(f"Arucos -> latest ids:{ids_list}, distance:{distances_list}, angle:{angles_list}\n ---")
        
        ''' odom[0] = deltaD -> sqrt( (position_x_list(len(position_x_list-1))-(position_x_list(len(position_x_list-2)))² +  msg.pose.pose.position.y(t)-msg.pose.pose.position.y(t-1))² )
            odom[1] = delta0 -> yaw1 - yaw2
            
            yaw -> np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
                        'orientation': (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
                        Orientation: (0.0, 0.0, 0.754709580222772, 0.6560590289905073) 
                    Convert quaternion to Euler angles (yaw, pitch, roll) with yaw angle we can track the robot's heading as it moves

            u wouldnt be needed because it relates to velicity, for that reason it need to be integrated, but our outputs are refered to position
        '''

        # Get measurements
        #zs = zs.append((distances_list[-1],angles_list[-1],ids_list[-1]))
        #zs = sim_measurement(robot.get_pose(),landmarks) # Simulate measurements between robot and landmarks
        # EKF Slam Logic
        mu, sigma = prediction_update(mu,sigma,odom) # Perform EKF prediction update
        #mu, sigma = measurement_update(mu,sigma,zs) # Perform EKF measurement update
        # Plotting
        print("mu = ", mu[:3,0])
        plot(mu, sigma, n_landmarks)



