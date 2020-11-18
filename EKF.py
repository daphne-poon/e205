import csv
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics 
import os.path
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from mpl_toolkits import mplot3d

NUM_ROBOTS = 5
NUM_STATES = 3
NUM_INPUTS = 2

CENTRALIZED = True
DT = 0.2  # seconds
NUM_STATES = 3  # number of states (x, y, theta)
NUM_ROBOTS = 5  # number of robots
NUM_ODOM = 2  # number of odometry variables (v, w)
NUM_LANDMARKS = 7  # number of landmarks
NUM_MEAS = 2  # number of measurement variables


def wrap_to_pi(angle):
    """Wrap angle data in radians to [-pi, pi]
    """
    while angle >= math.pi:
        angle -= 2*math.pi

    while angle <= -math.pi:
        angle += 2*math.pi
    return angle

def filter_time(data_log, min_time, max_time):
    
    processed_log = data_log[data_log[:,0] >= min_time]
    processed_log = processed_log[processed_log[:,0] <= max_time]
    processed_log[:,0] = processed_log[:,0]-min_time
    
    return processed_log

def update_u(robot_odom, time):
    
    while robot_odom[0,0] <= time:
        robot_odom = np.delete(robot_odom, 0, 0)
        if robot_odom.shape[0] == 1:
            break
            
    u = robot_odom[0, 1:].reshape(-1,1)

    return u, robot_odom

def update_z(meas_list, robot_meas, time, robot_num):

    if robot_meas.shape[0] <= 1:
        return meas_list, robot_meas

    while robot_meas[0,0] <= time:
        
        meas = np.array([robot_meas[0,:]])
        meas[0, 0] = robot_num
        
        if meas_list.shape[0] == 0:
            meas_list = meas
        else:
            meas_list = np.vstack((meas_list, meas))
            
        robot_meas = np.delete(robot_meas, 0, 0)
        
        if robot_meas.shape[0] == 0:
            break

    return meas_list, robot_meas

def calc_prop_jacobian_x(x_t_prev, u_t, d_t):
    """Calculate the Jacobian of your motion model with respect to state

    Parameters:
    x_t_prev (np.array) -- the previous state estimate
    u_t (np.array)      -- the current control input

    Returns:
    G_x_t (np.array)    -- Jacobian of motion model wrt to x
    """
    
    G_x_t = np.eye(x_t_prev.shape[0])

    return G_x_t


def calc_prop_jacobian_u(x_t_prev, u_t, d_t):
    """Calculate the Jacobian of motion model with respect to control input

    Parameters:
    x_t_prev (np.array)     -- the previous state estimate
    u_t (np.array)          -- the current control input

    Returns:
    G_u_t (np.array)        -- Jacobian of motion model wrt to u
    """
    G_u_t = np.zeros((NUM_STATES*NUM_ROBOTS, NUM_INPUTS*NUM_ROBOTS))

    for i in range(NUM_ROBOTS):
        theta = x_t_prev[(i+1)*NUM_STATES-1][0]
        temp_G = np.array([ 
                [d_t*(np.cos(theta)), 0], 
                [d_t*(np.sin(theta)), 0], 
                [0, d_t]])
        G_u_t[i*NUM_STATES:(i+1)*NUM_STATES,i*NUM_INPUTS:(i+1)*NUM_INPUTS] = temp_G
        
    G_u_t

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
    # Covariance matrix of inputs
    # variances where found by running statistics.variance() on the first 95 samples where the robot seems to be static

    sigma_u_t = np.zeros((NUM_INPUTS*NUM_ROBOTS, NUM_INPUTS*NUM_ROBOTS))
    variances = [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1]
    for i in range(len(variances)):
        sigma_u_t[i,i] = variances[i]
        

    for i in range(NUM_ROBOTS):
        x_t_prev[(i+1)*NUM_STATES-1][0] = wrap_to_pi(x_t_prev[(i+1)*NUM_STATES-1][0])

    #gxt = calc_prop_jacobian_x(x_t_prev, u_t, delta_t)
    gxt = np.eye(x_t_prev.shape[0])
    gut = calc_prop_jacobian_u(x_t_prev, u_t, delta_t)
    x_bar_t = gxt@x_t_prev + gut@u_t
    x_bar_t[3] = wrap_to_pi(x_bar_t[3])
    # print(sigma_x_t_prev)
    # print('1', gxt.shape)
    # print('2', sigma_x_t_prev.shape)
    # print('3', gut.shape)
    # print('4', sigma_u_t.shape)
    diag = np.diagonal(gxt@sigma_x_t_prev@(gxt.T) + gut@sigma_u_t@(gut.T))
    np.fill_diagonal(sigma_x_t_prev,diag )
    # print((gxt@sigma_x_t_prev@(gxt.T) + gut@sigma_u_t@(gut.T)).shape, sigma_x_t_prev.shape, diag, sigma_x_bar_t)

    return [x_bar_t, sigma_x_t_prev]

def calc_meas_jacobian(x_bar_t):
    """Calculate the Jacobian of your measurment model with respect to state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    H_t (np.array)      -- Jacobian of measurment model
    """
    """STUDENT CODE START"""
    theta_p = wrap_to_pi(x_bar_t[3][0])

    H_t = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return H_t


def calc_kalman_gain(sigma_x_bar_t, H_t):
    """Calculate the Kalman Gain

    Parameters:
    sigma_x_bar_t (np.array)  -- the predicted state covariance matrix
    H_t (np.array)            -- the measurement Jacobian

    Returns:
    K_t (np.array)            -- Kalman Gain
    """
    # Covariance matrix of measurments
    # variances where found by running statistics.variance() on the first 95 samples where the robot seems to be static
    sigma_z_t = np.array([
        [0.18685869613687928, 0, 0, 0],
        [0, 0.18685869613687928, 0, 0],
        [0, 0, 0.18685869613687928, 0], 
        [0, 0, 0, 0.18685869613687928]
    ])
    temp = np.linalg.inv((H_t@sigma_x_bar_t@H_t.T)+sigma_z_t)
    K_t = sigma_x_bar_t@(H_t.T)@temp
    return K_t


def calc_meas_prediction(x_bar_t):
    """Calculate predicted measurement based on the predicted state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    z_bar_t (np.array)  -- the predicted measurement
    """

    z_bar_t = np.array([
        x_bar_t[0],
        x_bar_t[1],
        x_bar_t[2],
        wrap_to_pi(x_bar_t[3])
    ])

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
    H_t = calc_meas_jacobian(x_bar_t)
    z_bar_t = calc_meas_prediction(x_bar_t)
    K_t = calc_kalman_gain(sigma_x_bar_t, H_t)

    x_est_t = x_bar_t + K_t@(z_t - z_bar_t)
    sigma_x_est_t = (np.eye(4) - K_t@H_t)@sigma_x_bar_t

    return [x_est_t, sigma_x_est_t]

def get_hab(a, b):
    """
    :param a: robot doing the observing
    :param b: robot being observed
    :return: H_ab (Jacobian for measurement)
    """
    hab = np.zeros((NUM_MEAS, NUM_ROBOTS * NUM_STATES))
    hab[0, a * NUM_STATES] = -1
    hab[1, (a * NUM_STATES) + 1] = -1
    hab[0, b * NUM_STATES] = 1
    hab[1, (b * NUM_STATES) + 1] = 1

    return hab


def calc_meas_prediction_2(x_t_pred, a, b):
    """
    :param x_t_pred: predicted states
    :param a: robot doing the observing
    :param b: robot being observed
    :return: z_expected (expected measurement)
    """

    xa = x_t_pred[a * NUM_STATES]
    xb = x_t_pred[b * NUM_STATES]
    ya = x_t_pred[(a * NUM_STATES) + 1]
    yb = x_t_pred[(b * NUM_STATES) + 1]

    dx = xb - xa
    dy = yb - ya

    temp = np.array([dx, dy])

    return temp.T



def correction_step_2(z_t, x_t_pred, sigma_x_t_pred, a, b):
    """Correction step for centralized CL, after a first measurement has been made.
    """

    # Covariance matrix for measurement
    # TODO: fill in!!!
    # sigma_z_t = np.zeros((NUM_ROBOTS * NUM_MEAS, NUM_ROBOTS * NUM_MEAS))
    sigma_z_t = np.zeros((NUM_MEAS, NUM_MEAS))

    # H_ab
    hab = get_hab(a, b)

    # S_ab
    sab = hab @ sigma_x_t_pred @ hab.T + sigma_z_t

    # Kalman Gain
    K_t = sigma_x_t_pred @ hab.T @ np.linalg.pinv(sab)

    # z_exp is the expected measurement given positions of a and b
    z_exp = calc_meas_prediction_2(x_t_pred, a, b)

    # Estimated state

    x_t_est = x_t_pred + (K_t @ (z_t - z_exp.T))

    # Estimated state covariance matrix
    sigma_x_t_est = sigma_x_t_pred - (K_t @ sab @ K_t.T)

    return x_t_est, sigma_x_t_est

def rmse(x_vec,y_vec):
    rms_list = []
    error = 0
    count = 0
    d_list = []
    for i in range(len(x_vec)):
        x = x_vec[i]
        y = y_vec[i]
        if (x>=0 and x<= 10) or (y >= -10 and y <= 0) :
            d = min(abs(y-0), abs(y+10), abs(x-0), abs(x-10))
        else:
            d = abs(min(math.hypot(x, y), math.hypot(x - 10, y), math.hypot(x-10, y + 10), math.hypot(x, y +10)))
        error += d
        count += 1
        rms_list.append((np.sqrt(d/count))**2)
    '''
    plt.plot(rms_list)
    plt.plot(rms_list)
    plt.xlabel("Time Step")
    plt.ylabel("RMSE")
    '''
    return rms_list


def run_EKF():
    #Load data from dat files
    root = "data/MRCLAM_Dataset1/"
    barcodes = np.loadtxt(root + "Barcodes.dat")
    landmark_gt = np.loadtxt(root + "Landmark_Groundtruth.dat")

    robot1_gt = np.loadtxt(root + "Robot1_Groundtruth.dat")
    robot1_meas = np.loadtxt(root + "Robot1_Measurement.dat")
    robot1_odom = np.loadtxt(root + "Robot1_Odometry.dat")

    robot2_gt = np.loadtxt(root + "Robot2_Groundtruth.dat")
    robot2_meas = np.loadtxt(root + "Robot2_Measurement.dat")
    robot2_odom = np.loadtxt(root + "Robot2_Odometry.dat")

    robot3_gt = np.loadtxt(root + "Robot3_Groundtruth.dat")
    robot3_meas = np.loadtxt(root + "Robot3_Measurement.dat")
    robot3_odom = np.loadtxt(root + "Robot3_Odometry.dat")

    robot4_gt = np.loadtxt(root + "Robot4_Groundtruth.dat")
    robot4_meas = np.loadtxt(root + "Robot4_Measurement.dat")
    robot4_odom = np.loadtxt(root + "Robot4_Odometry.dat")

    robot5_gt = np.loadtxt(root + "Robot5_Groundtruth.dat")
    robot5_meas = np.loadtxt(root + "Robot5_Measurement.dat")
    robot5_odom = np.loadtxt(root + "Robot5_Odometry.dat")

    # map barcodes to robot numbers
    map_to = np.array((barcodes), dtype=int)[:,0]
    code_array = np.array((barcodes), dtype=int)[:, 1]
    

    for i in range(len(map_to)):
        robot1_meas[:,1] = np.where(robot1_meas[:,1]==code_array[i], map_to[i], robot1_meas[:,1])
        robot2_meas[:,1] = np.where(robot2_meas[:,1]==code_array[i], map_to[i], robot2_meas[:,1])
        robot3_meas[:,1] = np.where(robot3_meas[:,1]==code_array[i], map_to[i], robot3_meas[:,1])
        robot4_meas[:,1] = np.where(robot4_meas[:,1]==code_array[i], map_to[i], robot4_meas[:,1])
        robot5_meas[:,1] = np.where(robot5_meas[:,1]==code_array[i], map_to[i], robot5_meas[:,1])

    min_time = min(robot1_odom[0,0], robot2_odom[0,0], robot3_odom[0,0], robot4_odom[0,0], robot5_odom[0,0])
    max_time = max(robot1_odom[-1,0], robot2_odom[-1,0], robot3_odom[-1,0], robot4_odom[-1,0], robot5_odom[-1,0])

    robot1_gt = filter_time(robot1_gt, min_time, max_time)
    robot1_meas = filter_time(robot1_meas, min_time, max_time)
    robot1_odom = filter_time(robot1_odom, min_time, max_time)

    robot2_gt = filter_time(robot2_gt, min_time, max_time)
    robot2_meas = filter_time(robot2_meas, min_time, max_time)
    robot2_odom = filter_time(robot2_odom, min_time, max_time)

    robot3_gt = filter_time(robot3_gt, min_time, max_time)
    robot3_meas = filter_time(robot3_meas, min_time, max_time)
    robot3_odom = filter_time(robot3_odom, min_time, max_time)

    robot4_gt = filter_time(robot4_gt, min_time, max_time)
    robot4_meas = filter_time(robot4_meas, min_time, max_time)
    robot4_odom = filter_time(robot4_odom, min_time, max_time)

    robot5_gt = filter_time(robot5_gt, min_time, max_time)
    robot5_meas = filter_time(robot5_meas, min_time, max_time)
    robot5_odom = filter_time(robot5_odom, min_time, max_time)

    gps_estimates = np.vstack((robot1_gt, robot2_gt, robot3_gt, robot4_gt, robot5_gt))

    N = 3 # number of states
    n = 5 # number of robots
    state_pred_prev = 0
    state_est_t_prev = np.concatenate((robot1_gt[0, 1:].reshape(3,-1), robot2_gt[0, 1:].reshape(3,-1), robot3_gt[0, 1:].reshape(3,-1), robot4_gt[0, 1:].reshape(3,-1), robot5_gt[0, 1:].reshape(3,-1)), axis = 0)
    var_est_t_prev = np.identity(NUM_STATES*NUM_ROBOTS)

    delta_t = 0.2
    time_stamps = np.arange(0, int(min(robot1_odom[-1,0], robot2_odom[-1,0], robot3_odom[-1,0], robot4_odom[-1,0], robot5_odom[-1,0])), delta_t)
    
    state_estimates = np.empty((NUM_STATES*NUM_ROBOTS, len(time_stamps)))
    covariance_estimates = np.empty((NUM_STATES*NUM_ROBOTS, NUM_STATES*NUM_ROBOTS, len(time_stamps))) # is this right?
    gps_estimates = np.empty((3, len(time_stamps)))

    total = robot1_odom.shape[0]
    # Run filter over data
    for t in range(1,len(time_stamps)):
        
        time = time_stamps[t]
        
        if t%100 == 0:
            print("Currently Printing Step", t, '/',len(time_stamps), "robot progress", robot1_odom.shape[0]/total )

        #find input
        u1, robot1_odom = update_u(robot1_odom, time)
        u2, robot2_odom = update_u(robot2_odom, time)
        u3, robot3_odom = update_u(robot3_odom, time)
        u4, robot4_odom = update_u(robot4_odom, time)
        u5, robot5_odom = update_u(robot5_odom, time)
        u_t = np.concatenate((u1, u2, u3, u4, u5), axis = 0)

        # Prediction Step
        if(t>1):
            state_pred_prev = state_pred_t
        
        state_pred_t, var_pred_t = prediction_step(state_est_t_prev, u_t, var_est_t_prev, delta_t)
        
        state_est_t = state_pred_t
        var_est_t = var_pred_t
        # print(state_est_t)
        # # Get measurement
        # meas_list = np.empty(0)
        # meas_list, robot1_meas = update_z(meas_list, robot1_meas, time, 1)
        # meas_list, robot2_meas = update_z(meas_list, robot2_meas, time, 2)
        # meas_list, robot3_meas = update_z(meas_list, robot3_meas, time, 3)
        # meas_list, robot4_meas = update_z(meas_list, robot4_meas, time, 4)
        # meas_list, robot5_meas = update_z(meas_list, robot5_meas, time, 5)

        # if meas_list.shape[0] == 0:
        #     state_est_t = state_pred_t
        #     var_est_t = var_pred_t

        # # for i, z_t in enumerate(meas_list):
        # #     # Correction Step
        # #     state_est_t, var_est_t = correction_step(state_pred_t, z_t, var_pred_t)
        # for i, z_t in enumerate(meas_list):
        #     # Correction Step
        #     a = int(z_t[0]-1)
        #     b = int(z_t[1]-1)

        #     if b>4:
        #         continue

        #     theta_a = state_pred_t[(a * NUM_STATES) + 2]

        #     r = z_t[2]
        #     phi = z_t[3]

        #     x_loc = r * math.cos(phi + theta_a)
        #     y_loc = r * math.sin(phi + theta_a)

        #     z_t = np.array([x_loc, y_loc]).reshape(2, 1)
        #     state_est_t, var_est_t = correction_step_2(z_t, state_pred_prev, var_est_t_prev, a, b)
        

        # For clarity sake/teaching purposes, we explicitly update t->(t-1)
        state_est_t_prev = state_est_t
        var_est_t_prev = var_est_t

        # Log Data
        state_estimates[:, t] = state_est_t[:,0]
        covariance_estimates[:, :, t] = var_est_t

    return state_estimates, covariance_estimates, robot1_gt

if __name__ == "__main__":
    
    state_estimates, covariance_estimates, gps_1 = run_EKF()
    state_1 = state_estimates[0:3]

    # Visualize results
    fig, ax1 = plt.subplots()
    ax1.plot(gps_1[10:, 1], gps_1[10:, 2], label = 'GPS Path')
    ax1.plot(state_1[0,10:], state_1[1,10:], label = 'Filtered Path')
    ax1.set_xlim(-10,10)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Plot of the Vehicle’s Path')
    ax1.legend()

    fig, ax2 = plt.subplots()
    ax2.plot(gps_1[10:200, 1], gps_1[10:200, 2], label = 'GPS Path')
    ax2.plot(state_1[0,10:200], state_1[1,10:200], label = 'Filtered Path')
    ax2.set_xlim(-10,10)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Plot of the Vehicle’s Path')
    ax2.legend()

    # fig, ax2 = plt.subplots()
    # ax2.plot(gps_1[3], label = 'Compass Yaw')
    # ax2.plot(state_estimates[3,:], label = 'Filtered Yaw')
    # ax2.set_title('Theta as a Function of Time')
    
    # t = np.arange(0, gps_1[3].shape[0])
    # e = np.sqrt(covariance_estimates[3, 3, :])
    # print(t.shape)
    # ax2.errorbar(t, state_estimates[3,:], e, linestyle='None', errorevery=10, marker='')
    # ax2.legend()

    # print(covariance_estimates[3, 3, :].shape)
    # lambda_ = np.sqrt(covariance_estimates[3, 3, :].shape)

    # for i in range(0, 800, 100):
    #     for j in range(1,3):
    #         y = state_estimates[3][i]
    #         # print(x,y)
    #         ell = Ellipse(xy=(i, y),
    #                     width=lambda_[0]*j*2, height=lambda_[0]*j*2,
    #                     angle=np.rad2deg(np.arccos(v[0, 0])), color='black', ls='--')

    #         ell.set_facecolor('none')
    #         ax2.add_artist(ell)
    # #ax2.scatter(x, y)

    plt.show()

