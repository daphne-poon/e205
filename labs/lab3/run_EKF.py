"""
Author: Andrew Q. Pham
Email: apham@g.hmc.edu
Date of Creation: 2/26/20
Description:
    Extended Kalman Filter implementation to filtering localization estimate
    This code is for teaching purposes for HMC ENGR205 System Simulation Lab 3
    Student code version with parts omitted.
"""

import csv
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import os.path
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

HEIGHT_THRESHOLD = 0.0  # meters
GROUND_HEIGHT_THRESHOLD = -.4  # meters
DT = 0.1
X_LANDMARK = 5.  # meters
Y_LANDMARK = -5.  # meters
EARTH_RADIUS = 6.3781E6  # meters


def load_data(filename):
    """Load data from the csv log

    Parameters:
    filename (str)  -- the name of the csv log

    Returns:
    data (dict)     -- the logged data with data categories as keys
                       and values list of floats
    """
    is_filtered = False
    if os.path.isfile(filename + "_filtered.csv"):
        f = open(filename + "_filtered.csv")
        is_filtered = True
    else:
        f = open(filename + ".csv")

    file_reader = csv.reader(f, delimiter=',')

    # Load data into dictionary with headers as keys
    data = {}
    header = ["X", "Y", "Z", "Time Stamp", "Latitude", "Longitude",
              "Yaw", "Pitch", "Roll", "AccelX", "AccelY", "AccelZ"]
    for h in header:
        data[h] = []

    row_num = 0
    f_log = open("bad_data_log.txt", "w")
    for row in file_reader:
        for h, element in zip(header, row):
            # If got a bad value just use the previous value
            try:
                data[h].append(float(element))
            except ValueError:
                data[h].append(data[h][-1])
                f_log.write(str(row_num) + "\n")

        row_num += 1
    f.close()
    f_log.close()

    return data, is_filtered


def save_data(data, filename):
    """Save data from dictionary to csv

    Parameters:
    filename (str)  -- the name of the csv log
    data (dict)     -- data to log
    """
    header = ["X", "Y", "Z", "Time Stamp", "Latitude", "Longitude",
              "Yaw", "Pitch", "Roll", "AccelX", "AccelY", "AccelZ"]
    f = open(filename, "w")
    num_rows = len(data["X"])
    for i in range(num_rows):
        for h in header:
            f.write(str(data[h][i]) + ",")

        f.write("\n")

    f.close()


def filter_data(data):
    """Filter lidar points based on height and duplicate time stamp

    Parameters:
    data (dict)             -- unfilterd data

    Returns:
    filtered_data (dict)    -- filtered data
    """

    # Remove data that is not above a height threshold to remove
    # ground measurements and remove data below a certain height
    # to remove outliers like random birds in the Linde Field (fuck you birds)
    filter_idx = [idx for idx, ele in enumerate(data["Z"])
                  if ele > GROUND_HEIGHT_THRESHOLD and ele < HEIGHT_THRESHOLD]

    filtered_data = {}
    for key in data.keys():
        filtered_data[key] = [data[key][i] for i in filter_idx]

    # Remove data that at the same time stamp
    ts = filtered_data["Time Stamp"]
    filter_idx = [idx for idx in range(1, len(ts)) if ts[idx] != ts[idx-1]]
    for key in data.keys():
        filtered_data[key] = [filtered_data[key][i] for i in filter_idx]

    return filtered_data


def convert_gps_to_xy(lat_gps, lon_gps, lat_origin, lon_origin):
    """Convert gps coordinates to cartesian with equirectangular projection

    Parameters:
    lat_gps     (float)    -- latitude coordinate
    lon_gps     (float)    -- longitude coordinate
    lat_origin  (float)    -- latitude coordinate of your chosen origin
    lon_origin  (float)    -- longitude coordinate of your chosen origin

    Returns:
    x_gps (float)          -- the converted x coordinate
    y_gps (float)          -- the converted y coordinate
    """
    x_gps = EARTH_RADIUS*(math.pi/180.)*(lon_gps - lon_origin)*math.cos((math.pi/180.)*lat_origin)
    y_gps = EARTH_RADIUS*(math.pi/180.)*(lat_gps - lat_origin)

    return x_gps, y_gps


def wrap_to_pi(angle):
    """Wrap angle data in radians to [-pi, pi]

    Parameters:
    angle (float)   -- unwrapped angle

    Returns:
    angle (float)   -- wrapped angle
    """
    while angle >= math.pi:
        angle -= 2*math.pi

    while angle <= -math.pi:
        angle += 2*math.pi
    return angle


def propogate_state(x_t_prev, u_t, d_t):
    """Propogate/predict the state based on chosen motion model

    Parameters:
    x_t_prev (np.array)  -- the previous state estimate
    u_t (np.array)       -- the current control input

    Returns:
    x_bar_t (np.array)   -- the predicted state
    """
    """STUDENT CODE START"""
    #x_bar_t = np.empty()
    """STUDENT CODE END"""

    return #x_bar_t


def calc_prop_jacobian_x(x_t_prev, u_t, d_t):
    """Calculate the Jacobian of your motion model with respect to state

    Parameters:
    x_t_prev (np.array) -- the previous state estimate
    u_t (np.array)      -- the current control input

    Returns:
    G_x_t (np.array)    -- Jacobian of motion model wrt to x
    """
    """STUDENT CODE START"""
    G_x_t = np.array([
        [1, 0, d_t, 0, 0, 0, 0], 
        [0, 1, 0, d_t, 0, 0, 0], 
        [0, 0, 1, 0, 0, 0, 0], 
        [0, 0, 0, 1, 0, 0, 0], 
        [0, 0, 0, 0, 1, 0, d_t], 
        [0, 0, 0, 0, 1, 0, 0], 
        [0, 0, 0, 0, 1, -1, 0]
    ])
    """STUDENT CODE END"""

    return G_x_t


def calc_prop_jacobian_u(x_t_prev, u_t, d_t):
    """Calculate the Jacobian of motion model with respect to control input

    Parameters:
    x_t_prev (np.array)     -- the previous state estimate
    u_t (np.array)          -- the current control input

    Returns:
    G_u_t (np.array)        -- Jacobian of motion model wrt to u
    """

    """STUDENT CODE START"""
    theta_p = x_t_prev[4][0]

    G_u_t = np.array([
        [0, 0], 
        [0, 0], 
        [d_t*(np.cos(theta_p)), d_t*(np.sin(theta_p))], 
        [-d_t*(np.sin(theta_p)), d_t*(np.cos(theta_p))], 
        [0, 0], 
        [0, 0], 
        [0, 0]
    ])
    """STUDENT CODE END"""

    return G_u_t


def prediction_step(x_t_prev, u_t, sigma_x_t_prev, delta_t):
    """Compute the prediction of EKF

    Parameters:
    x_t_prev (np.array)         -- the previous state estimate
    u_t (np.array)              -- the control input
    sigma_x_t_prev (np.array)   -- the previous variance estimate

    Returns:
    x_bar_t (np.array)          -- the predicted state estimate of time t
    sigma_x_bar_t (np.array)    -- the predicted variance estimate of time t
    """

    """STUDENT CODE START"""
    # Covariance matrix of control input
    sigma_u_t = np.array([
        [0.4170670659498302, 0],
        [0, 0.5488300624965801]
    ]) # add shape of matrix

    gxt = calc_prop_jacobian_x(x_t_prev, u_t, delta_t)
    gut = calc_prop_jacobian_u(x_t_prev, u_t, delta_t)
    # x_bar_t = propogate_state(x_t_prev, u_t, delta_t)
    x_bar_t = gxt@x_t_prev + gut@u_t

    x_bar_t[-1] = (wrap_to_pi(x_bar_t[-1]))/delta_t

    x_bar_t[3] = wrap_to_pi(x_bar_t[3])

    sigma_x_bar_t = gxt@sigma_x_t_prev@(gxt.T) + gut@sigma_u_t@(gut.T)
    """STUDENT CODE END"""

    return [x_bar_t, sigma_x_bar_t]


def calc_meas_jacobian(x_bar_t):
    """Calculate the Jacobian of your measurment model with respect to state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    H_t (np.array)      -- Jacobian of measurment model
    """
    """STUDENT CODE START"""
    H_t = np.array([
        [-1, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0]
    ])
    """STUDENT CODE END"""

    return H_t


def calc_kalman_gain(sigma_x_bar_t, H_t):
    """Calculate the Kalman Gain

    Parameters:
    sigma_x_bar_t (np.array)  -- the predicted state covariance matrix
    H_t (np.array)            -- the measurement Jacobian

    Returns:
    K_t (np.array)            -- Kalman Gain
    """
    """STUDENT CODE START"""
    # Covariance matrix of measurments
    sigma_z_t = np.array([
        [0.06836275526116181, 0, 0],
        [0, 0.12908857925993825, 0],
        [0, 0, 0.014965442010048293]
    ])
    idk = (H_t@sigma_x_bar_t@H_t.T)+sigma_z_t
    idkk = np.linalg.inv(idk)
    K_t = sigma_x_bar_t@(H_t.T)@idkk
    """STUDENT CODE END"""
    return K_t


def calc_meas_prediction(x_bar_t):
    """Calculate predicted measurement based on the predicted state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    z_bar_t (np.array)  -- the predicted measurement
    """

    """STUDENT CODE START"""
    XP = 5
    YP = -5
    z_pxt = XP - x_bar_t[0]
    z_pyt = YP - x_bar_t[1]
    z_theta = x_bar_t[4]
    z_bar_t = np.array([
        z_pxt,
        z_pyt,
        wrap_to_pi(z_theta)
    ])
    """STUDENT CODE END"""

    return z_bar_t


def correction_step(x_bar_t, z_t, sigma_x_bar_t):
    """Compute the correction of EKF

    Parameters:
    x_bar_t       (np.array)    -- the predicted state estimate of time t
    z_t           (np.array)    -- the measured state of time t
    sigma_x_bar_t (np.array)    -- the predicted variance of time t

    Returns:
    x_est_t       (np.array)    -- the filtered state estimate of time t
    sigma_x_est_t (np.array)    -- the filtered variance estimate of time t
    """

    """STUDENT CODE START"""

    H_t = calc_meas_jacobian(x_bar_t)
    z_bar_t = calc_meas_prediction(x_bar_t)
    K_t = calc_kalman_gain(sigma_x_bar_t, H_t)
    
    x_est_t = x_bar_t + K_t@(z_t - z_bar_t)
    sigma_x_est_t = (np.eye(7) - K_t@H_t)@sigma_x_bar_t
    """STUDENT CODE END"""
    return [x_est_t, sigma_x_est_t]

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def rmse(x_vec,y_vec):
    rms_list = []
    error = 0
    count = 0
    for i in range(len(x_vec)):
        x = x_vec[i]
        y = y_vec[i]
        if (x>0 and x< 10) or (y >= 10 and y < 0) :
            print(np.min(abs(y-0), abs(y+10))
            d = np.min(abs(y-0), abs(y+10), abs(x-0), abs(x-10))
        else:
            d = abs(np.min(math.hypot(x, y), math.hypot(x - 10, y), math.hypot(x-10, y + 10), math.hypot(x, y +10)))
        error += d
        count += 1
        rms_list.append((np.sqrt(d/count))**2)

    plt.plot(rms_list)
    plt.xlabel("Time Step")
    plt.ylabel("RMSE")
    return rms_list




def main():
    """Run a EKF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    filepath = "./data/"
    # filename = "2020_2_26__17_21_59"
    filename = "2020_2_26__16_59_7"
    data, is_filtered = load_data(filepath + filename)

    # Save filtered data so don't have to process unfiltered data everytime
    if not is_filtered:
        data = filter_data(data)
        save_data(f_data, filepath+filename+"_filtered.csv")

    # Load data into variables
    x_lidar = data["X"]
    y_lidar = data["Y"]
    z_lidar = data["Z"]
    time_stamps = data["Time Stamp"]
    lat_gps = data["Latitude"]
    lon_gps = data["Longitude"]
    yaw_lidar = data["Yaw"]
    pitch_lidar = data["Pitch"]
    roll_lidar = data["Roll"]
    x_ddot = data["AccelX"]
    y_ddot = data["AccelY"]

    lat_origin = lat_gps[0]
    lon_origin = lon_gps[0]

    #  Initialize filter
    """STUDENT CODE START"""
    N = 7 # number of states
    state_est_t_prev = np.zeros([7,1])
    var_est_t_prev = np.identity(N)

    state_estimates = np.empty((N, len(time_stamps)))
    covariance_estimates = np.empty((N, N, len(time_stamps)))
    gps_estimates = np.empty((2, len(time_stamps)))
    """STUDENT CODE END"""


    #  Run filter over data
    for t, _ in enumerate(time_stamps):

        """STUDENT CODE START"""
        delta_t = time_stamps[t] - time_stamps[t-1]
        delta_t = delta_t/1000000
        
        # Get control input
        u_t = np.array([[x_ddot[t]], [y_ddot[t]]])
        """STUDENT CODE END"""

        # Prediction Step
        state_pred_t, var_pred_t = prediction_step(state_est_t_prev, u_t, var_est_t_prev, delta_t)

        # Get measurement
        """STUDENT CODE START"""
        yaw = wrap_to_pi((math.pi/180)*(yaw_lidar[t]))
        z_t = np.array([[-np.sin(yaw)*x_lidar[t]+np.cos(yaw)*y_lidar[t]],
                         [-np.cos(yaw)*x_lidar[t]-np.sin(yaw)*y_lidar[t]],
                         [yaw]])
        """STUDENT CODE END"""

        # Correction Step
        state_est_t, var_est_t = correction_step(state_pred_t,
                                                 z_t,
                                                 var_pred_t)

        #  For clarity sake/teaching purposes, we explicitly update t->(t-1)
        state_est_t_prev = state_est_t
        var_est_t_prev = var_est_t

        # Log Data

        state_estimates[:, t] = state_est_t[:,0]
        covariance_estimates[:, :, t] = var_est_t

        x_gps, y_gps = convert_gps_to_xy(lat_gps=lat_gps[t],
                                         lon_gps=lon_gps[t],
                                         lat_origin=lat_origin,
                                         lon_origin=lon_origin)
        gps_estimates[:, t] = np.array([x_gps, y_gps])

    """FIND A THING"""
    # shortest_distance(x,y)
    x_vec = state_estimates[0,:]
    y_vec = state_estimates[1,:]
    rmse(x_vec,y_vec)

    # fig, ax = plt.subplots()
    # # Plot or print results here
    # ax.set_ylim(-12,2.5)
    # # ax.set_xlim(4,6)
    # ax.plot(state_estimates[0,:], state_estimates[1, :], 'go', markersize=2)
    # ax.plot(gps_estimates[0], gps_estimates[1], 'bo', markersize=2)
    # ax.plot([0, 10, 10, 0, 0], [0, 0, -10, -10, 0], 'r--')

    # ax.set_xlabel("X Position (m)")
    # ax.set_ylabel("Y Position (m)")
    # ax.legend(["Estimated Position", "GPS Position", "Expected Path"],loc='upper left')

    # # yaw angle as a function of time.
    # # plt.clear()
    # # plt.ylim(-12,2.5)
    # print(state_estimates[4, :])

    # # print(covariance_estimates[:, :, 0])

    # lambda_, v = np.linalg.eig(covariance_estimates[:2, :2, 10])
    # lambda_ = np.sqrt(lambda_)

    # for i in range(0, 800, 100):
    #     for j in range(1,3):
    #         x = state_estimates[0][i]
    #         y = state_estimates[1][i]
    #         # print(x,y)
    #         ell = Ellipse(xy=(x, y),
    #                     width=lambda_[0]*j*2, height=lambda_[1]*j*2,
    #                     angle=np.rad2deg(np.arccos(v[0, 0])), color='black', ls='--')

    #         ell.set_facecolor('none')
    #         ax.add_artist(ell)
    # ax.scatter(x, y)
    plt.show()


    # yaw_lidar_f = [wrap_to_pi((math.pi/180)*x) for x in yaw_lidar]

    # plt.plot(range(len(time_stamps)), state_estimates[4,:])
    # plt.plot(range(len(time_stamps)), yaw_lidar_f)
    # plt.legend(["Estimated Yaw", "Raw Yaw Measurement"])
    # plt.xlabel("Timestep")
    # plt.ylabel("Angle (Radians)")
    # plt.plot(gps_estimates[0], gps_estimates[1], 'bo')

    # plt.show()
    """STUDENT CODE END"""
    return 0


if __name__ == "__main__":
    main()
