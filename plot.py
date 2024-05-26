import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from matplotlib.patches import Ellipse

# ---> Function auxiliar to plot a arrow as the robot
def aux_arrow(robot_pos):
    x, y, theta = robot_pos[0], robot_pos[1], robot_pos[2]
    dx = np.cos(theta)
    dy = np.cos(theta)
    return x, y, dx, dy

# ---> Function auxiliar to plot the ellipse of the incertain 
def aux_ellipse(sigma):
    [eigenvals,eigenvecs] = np.linalg.eig(sigma) # Finding eigenvalues and eigenvectors of the covariance matrix
    angle = 180.*np.arctan2(eigenvecs[1][0],eigenvecs[0][0])/np.pi # Find the angle of rotation for the first eigenvalue
    width, height = 2 * np.sqrt(eigenvals) # Find the width and the height of the ellipse
    return width, height, angle

def plot(mu, sigma, n_landmarks):
    
    fig, ax = plt.subplot(aspect="equal")
    fig.cla() # Clear the plot

    fig.arrow(*aux_arrow(mu[:3,0]), head_width = 1, color='blue') # Plot the state of the robot where visual is represented by an arrow

    width, height, angle = aux_ellipse(sigma[0:2,0:2]) # Calculate auxiliar values to plot the ellipse of the incertain of the robot state

    robot_sigma = Ellipse(xy=mu[:2,0], width=width, height=height, angle=angle, edgecolor=(0,0,0), fill=False) # Calculate the ellipse of the incertain of the robot state
    fig.add_artist(robot_sigma)

    # plot all the landmarks known until the moment
    for i in range(0, n_landmarks):


    
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])
    plt.title('Estimativa das observações e trajetória')

