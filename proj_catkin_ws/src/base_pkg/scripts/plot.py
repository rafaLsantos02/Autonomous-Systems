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

    n_state = 3
    
    ax = plt.subplot(aspect="equal")
    ax.cla() # Clear the plot

    ax.arrow(*aux_arrow(mu[:3,0]), head_width = 0.5, color='blue') # Plot the state of the robot where visual is represented by an arrow

    width, height, angle = aux_ellipse(sigma[0:2,0:2]) # Calculate auxiliar values to plot the ellipse of the incertain of the robot state

    robot_sigma = Ellipse(xy=mu[:2,0], width=width, height=height, angle=angle, edgecolor=(0,0,0), fill=False) # Calculate the ellipse of the incertain of the robot state
    ax.add_artist(robot_sigma)

    # plot all the landmarks known until the moment
    if n_landmarks != 0:
        for i in range(0, n_landmarks):
            plt.scatter(mu[n_state+2*i,0],mu[n_state+2*i+1,0], s=100, marker=".", color='red')

            width, height, angle = aux_ellipse(sigma[n_state+2*i:n_state+2*i+2, n_state+2*i:n_state+2*i+2]) # Calculate auxiliar values to plot the ellipse of the incertain of the landmark
            landmark_sigma = Ellipse(xy=mu[:2,0], width=width, height=height, angle=angle, edgecolor=(255,0,0), fill=False) # Calculate the ellipse of the incertain of the robot state
            ax.add_artist(landmark_sigma)
    
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    plt.title('Estimativa das observações e trajetória')

    plt.grid(True)
    plt.pause(0.001)

    return

def plot_firme(mu, sigma, n_landmarks):
    n_state = 3  
    fig, ax = plt.subplots()  # Use plt.subplots() para obter fig e ax separadamente
    ax.set_aspect('equal')    # Define o aspecto igual para os eixos
    ax.cla()  # Clear the plot

    # Adiciona uma bolinha para representar a posição do robô
    #ax.scatter(mu[0, 0], mu[1, 0], s=100, marker="o", color='blue')

    ax.arrow(*aux_arrow(mu[:3,0]), head_width=1, color='blue')  # Plot the state of the robot where visual is represented by an arrow

    width, height, angle = aux_ellipse(sigma[0:2,0:2])  # Calculate auxiliary values to plot the ellipse of the uncertainty of the robot state
    print(width, height, angle)
    print(mu[:2,0])
    robot_sigma = Ellipse(xy=mu[:2,0], width=width, height=height, angle=angle, edgecolor=(0,0,0), fill=False)  # Calculate the ellipse of the uncertainty of the robot state
    ax.add_artist(robot_sigma)

    # plot all the landmarks known until the moment
    for i in range(0, n_landmarks):
        ax.scatter(mu[n_state+2*i,0], mu[n_state+2*i+1,0], s=100, marker=".", color='red')

        width, height, angle = aux_ellipse(sigma[n_state+2*i:n_state+2*i+2, n_state+2*i:n_state+2*i+2])  # Calculate auxiliary values to plot the ellipse of the uncertainty of the landmark
        landmark_sigma = Ellipse(xy=mu[n_state+2*i:n_state+2*i+2,0], width=width, height=height, angle=angle, edgecolor=(1,0,0), fill=False)  # Calculate the ellipse of the uncertainty of the robot state
        ax.add_artist(landmark_sigma)
    
    ax.set_xlim([-40, 40])
    ax.set_ylim([-40, 40])
    ax.set_title('Estimativa das observações e trajetória')

    plt.show()
    return
