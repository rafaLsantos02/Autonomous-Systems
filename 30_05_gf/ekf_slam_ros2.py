#!/usr/bin/env python3
import math
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

import pygame
import pygame.gfxdraw
from map_environment.utils import environment, vehicles

from New import plot



count = 0

last_print_time = time.time()

calib_data_path1 = r"/home/gfirme/modificado/30_05_gf/calibration/calib_data.npz"

calib_data1 = np.load(calib_data_path1)

cam_mat = calib_data1["camMatrix"]
dist_coef = calib_data1["distCoef"]
r_vectors = calib_data1["rVector"]
t_vectors = calib_data1["tVector"]

MARKER_SIZE = 17

marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
param_markers = aruco.DetectorParameters_create()



#Robot position by pionner at real time
pos_x = 0
pos_y = 0
pos_z = 0
orientation = 0

#Info aruco seen at real time
aruco_id = 0
aruco_dist = 0
aruco_angle = 0
marker_vec = 0


# Observation from arucos 
aruco_id = 0
aruco_dist = 0
aruco_angle = 0 
marker_vec = 0

#List with all observations at real time
observations = []

# Calib Camera
calib_data_path = r"/home/gfirme/modificado/proj_catkin_ws/src/base_pkg/scripts/calib_data.npz"
calib_data = np.load(calib_data_path)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]

#Define Aruco Dictionary
#marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
param_markers = aruco.DetectorParameters_create()
MARKER_SIZE = 17




def acquire_observations(cv_image): #ex: process_image
    
    global mu, sigma
    global observations

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
            aruco_dist = distance / 100 #distance in m 
            aruco_angle = np.arccos(cos_angle) ##angle in rad
            #aruco_angle = aruco_angle[0]
            

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

    global aruco_id, aruco_dist, aruco_angle, marker_vec

    bridge = CvBridge()
    
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        acquire_observations(cv_image)

    except CvBridgeError as e:
        print(e)




def pose_callback(msg):

    orient_x, orient_y, orient_z, orient_w = 0,0,0,0
    
    global pos_x, pos_y, pos_z, orientation
    
    pose_data = {
        'position': (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
        'orientation': (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w),
        'frame_id': msg.header.frame_id,
        'timestamp': msg.header.stamp.to_sec()
    }
    pos_x = pose_data['position'][0]
    pos_y = pose_data['position'][1]
    pos_z = pose_data['position'][2]

    orient_x = pose_data['orientation'][0]
    orient_y = pose_data['orientation'][1]
    orient_z = pose_data['orientation'][2]
    orient_w = pose_data['orientation'][3]
    
    orientation = ( np.arctan2(2*(orient_w*orient_z + orient_x*orient_y), 1 - 2*(orient_y**2 + orient_z**2)) )

    return


# ---> Search the position of the id of the landmark and return the index
def id_search( landmark_id ):

    global n_landmarks
    global mu, sigma
    
    #Case: Already seen the landmark
    for i in range(0, len(landmarks)):
        if landmarks[i] == landmark_id:
            return i, 1

    #Case: First time that see these landmark
    landmarks.append(id)
    n_landmarks = len(landmarks)

    return n_landmarks-1, 0
    



# <------------------------- EKF SLAM STUFF --------------------------------->
# ---> Robot Parameters
n_state = 3 # Number of state variables

# ---> Landmark parameters
landmarks = []
n_landmarks = len(landmarks)


# ---> Noise parameters
R = np.diag([0.002,0.002,0.002]) # sigma_x, sigma_y, sigma_theta
Q = np.diag([0.003,0.005]) # sigma_r, sigma_phi
O = np.array([0.001, 0.001, 0.001]) # sigma_odom_x, sigma_odom_y, sigma_odom_theta #DUVIDA

# ---> EKF Estimation Variables
mu = np.zeros((n_state+2*n_landmarks,1)) # State estimate (robot pose and landmark positions)
sigma = np.zeros((n_state+2*n_landmarks,n_state+2*n_landmarks)) # State uncertainty, covariance matrix

mu[:] = np.nan # Initialize state estimate with nan values
np.fill_diagonal(sigma,100) # Initialize state uncertainty with large variances, no correlations


# ---> Helpful matrix
Fx = np.block([[np.eye(3),np.zeros((n_state,2*n_landmarks))]]) # Used in both prediction and measurement updates



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

    #rx,py,= mu[0],mu[1]
    theta = mu[2]

    deltaD = odom[0]
    deltaO = odom[1]

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

def measurement_update(mu,sigma, observations):
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
    rx,ry,theta = mu[0],mu[1],mu[2] # robot 
    delta_zs = [np.zeros((2,1)) for lidx in range(n_landmarks)] # A list of how far an actual measurement is from the estimate measurement
    Ks = [np.zeros((mu.shape[0],2)) for lidx in range(n_landmarks)] # A list of matrices stored for use outside the measurement for loop
    Hs = [np.zeros((2,mu.shape[0])) for lidx in range(n_landmarks)] # A list of matrices stored for use outside the measurement for loop
    
    for z in observations:

        (dist,phi,landmark_id) = z
        lidx, id_existence = id_search( landmark_id )

         # Verify if the landmark is already know by the robot
        if id_existence == 1:
            mu_landmark = mu[n_state+lidx*2:n_state+lidx*2+2]
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
        delta_zs[lidx] = z_act_arr-z_est_arr # Difference between actual and estimated observation

        # Helper matrices in computing the measurement update
        Fxj = np.block([[Fx],[np.zeros((2,Fx.shape[1]))]])
        Fxj[n_state:n_state+2,n_state+2*lidx:n_state+2*lidx+2] = np.eye(2)
        H = np.array([[-delta[0,0]/np.sqrt(q),-delta[1,0]/np.sqrt(q),0,delta[0,0]/np.sqrt(q),delta[1,0]/np.sqrt(q)],\
                      [delta[1,0]/q,-delta[0,0]/q,-1,-delta[1,0]/q,+delta[0,0]/q]])
        H = H.dot(Fxj)
        Hs[lidx] = H # Added to list of matrices
        Ks[lidx] = sigma.dot(np.transpose(H)).dot(np.linalg.inv(H.dot(sigma).dot(np.transpose(H)) + Q)) # Add to list of matrices
    # After storing appropriate matrices, perform measurement update of mu and sigma
    mu_offset = np.zeros(mu.shape) # Offset to be added to state estimate
    sigma_factor = np.eye(sigma.shape[0]) # Factor to multiply state uncertainty

    for lidx in range(n_landmarks):
        mu_offset += Ks[lidx].dot(delta_zs[lidx]) # Compute full mu offset
        sigma_factor -= Ks[lidx].dot(Hs[lidx]) # Compute full sigma factor

    mu = mu + mu_offset # Update state estimate
    sigma = sigma_factor.dot(sigma) # Update state uncertainty
    return mu,sigma
# <------------------------- EKF SLAM STUFF --------------------------------->





# <------------------------- PLOTTING STUFF --------------------------------->

def show_robot_position(landmarks,env):
    '''
    Visualize actual landmark location
    '''
    for landmark in landmarks:
        lx_pixel, ly_pixel = env.position2pixel(landmark)
        r_pixel = env.dist2pixellen(0.2) # Radius of the circle for the ground truth locations of the landmarks
        pygame.gfxdraw.filled_circle(env.get_pygame_surface(),lx_pixel,ly_pixel,r_pixel,(0,255,255)) # Blit the circle onto the surface

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

    # Descartar os primeiros 3 elementos de mu
    aux = mu[3:]

    # Extrair os valores de lx e ly automaticamente
    landmarks = [(aux[i], aux[i+1]) for i in range(0, len(aux), 2)]

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
    
    # Initialize pygame
    pygame.init()
    
    # Initialize robot and discretization time step
    x_init = np.array([1,1,np.pi/2]) # px, py, theta
    robot = vehicles.DifferentialDrive(x_init)
    dt = 0.01 
    ''' timestmaps ultimo - penultimo - comprimento do vetor-1'''
    i = 0
    # Initialize and display environment
    env = environment.Environment(map_image_path="./map_environment/maps/map_blank.png")
    
    # Initialize robot state estimate and sigma
    mu[0:3] = np.expand_dims(x_init,axis=1)
    mu[0] = 0 
    mu[1] = 0
    mu[2] = 0

    last_pos_x = 0
    last_pos_y = 0
    

    sigma[0:3,0:3] = 0.1*np.eye(3)
    sigma[2,2] = 0 
    running = True


    while running:

        odom = [0, 0]

        odom[0] = np.sqrt( (pos_x - last_pos_x) **2  +  (pos_y - last_pos_y) **2)
        odom[1] = orientation - mu[2]


        print(f"X: {pos_x}, Y: {pos_y}, \n ---")

        #print(f"deltaD: {odom[0]}, deltaO: {odom[1]}, \n ---")
        #print(f"Arucos -> latest ids:{aruco_id}, distance:{aruco_dist}, angle:{aruco_angle}, vector:{marker_vec}, \n ---")

        # Get measurementsw
        zs = observations # Simulate measurements between robot and landmarks
        
        # EKF Slam Logic
        mu, sigma = prediction_update(mu,sigma,odom) # Perform EKF prediction update
        #mu, sigma = measurement_update(mu,sigma,zs) # Perform EKF measurement update

        #robot.move_step( mu[0], mu[1])


        #mu[0]= pos_x
        #mu[1] = pos_y

        # Plotting
        print("mu = ", mu[:3,0])
        #print("mu_all = ", mu)
        #print("landmarks= ", n_landmarks)
        plot.plot(mu, sigma, n_landmarks)
        
        last_pos_x = pos_x
        last_pos_y = pos_y



        '''
        # Plotting
        env.show_map() # Re-blit map
        # Show measurements
        show_measurements(robot.get_pose(),zs,env)
        # Show actual locations of robot and landmarks
        env.show_robot(robot) # Re-blit robot
        show_landmark_location(landmarks,env)
        # Show estimates of robot and landmarks (estimate and uncertainty)
        show_robot_estimate(mu,sigma,env)
        show_landmark_estimate(mu,sigma,env)

        pygame.display.update() # Update display

        last_pos_x = pos_x
        last_pos_y = pos_y

        '''