"""
Author: Sabrina Shen and Daphne Poon
Date of Creation: 10/30/2020
"""

import csv
import random
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import os.path
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

HEIGHT_THRESHOLD = 0.0  # meters
GROUND_HEIGHT_THRESHOLD = -.4  # meters
DT = 0.1
X_LANDMARK = 5  # meters
Y_LANDMARK = -5  # meters
EARTH_RADIUS = 6.3781E6  # meters
SEED = 0
NUM_PARTICLES = 50
# x_min = 10
# x_max = 0
# y_min = -10
# y_max = 0
x_min = 0
x_max = 0
y_min = 0
y_max = 0
V_MIN, V_MAX = -0.001, 0.001


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
    filter_idx = [idx for idx in range(1, len(ts)) if ts[idx] != ts[idx - 1]]
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
    x_gps = EARTH_RADIUS * (math.pi / 180.) * (lon_gps - lon_origin) * math.cos((math.pi / 180.) * lat_origin)
    y_gps = EARTH_RADIUS * (math.pi / 180.) * (lat_gps - lat_origin)

    return x_gps, y_gps


def wrap_to_pi(angle):
    """Wrap angle data in radians to [-pi, pi]

    Parameters:
    angle (float)   -- unwrapped angle

    Returns:
    angle (float)   -- wrapped angle
    """
    while angle >= math.pi:
        angle -= 2 * math.pi

    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle


def get_motion_model(p, u_t, d_t):
    """ p = [x, y, theta, velocity, weight]
        return => [x, y, theta, velocity, 0]
    """

    sigma_x_acc = 0.4170670659498302
    sigma_yaw = 0.014965442010048293

    theta = wrap_to_pi(p[2])

    # add some randomness
    x_acc = random.gauss(u_t[0], sigma_x_acc)
    yaw = wrap_to_pi(random.gauss(theta, sigma_yaw))
    print('xacc & yaw:', x_acc, yaw)
    # new values
    mu_vel = p[3] + (x_acc * d_t)
    # mu_vel = (x_acc)x_acc
    print('motion vel:', mu_vel)
    mu_x = p[0] + (mu_vel * d_t * math.cos(yaw))
    print('motion model x:', mu_x)
    mu_y = p[1] + (mu_vel * d_t * math.sin(yaw))
    print('motion model y:', mu_y)
    mu_theta = yaw

    return create_particle(mu_x, mu_y, mu_theta, mu_vel, 0)


def get_new_weight(z_t, xi):
    """ given a state, calculate the probability of sensor input
    """
    sigma_x = 0.06836275526116181
    sigma_y = 0.12908857925993825
    sigma_theta = 0.014965442010048293
    prob_x = scipy.stats.norm(xi[0], sigma_x).pdf(z_t[0])
    prob_y = scipy.stats.norm(xi[1], sigma_y).pdf(z_t[1])
    prob_theta = scipy.stats.norm(xi[2], sigma_theta).pdf(z_t[2])
    # print(xi[0], z_t[0])
    return prob_x * prob_y * prob_theta


def prediction_step(particles_prev, u_t, z_t, d_t):
    """Compute the prediction of PF"""
    n = len(particles_prev)
    particles = np.zeros((n, 5))
    d_t = 0.1
    # 1. for i = 1 ... n
    for i in range(n):
        # 2. pick p_t-1^i from P_{t-1}
        curr = particles_prev[i]

        # 3. sample x_t^i with probability P(x_t^i | x_{t-1}^i, u_t),
        #    where o_t is the odometry measurement
        xi = get_motion_model(curr, u_t, d_t)

        # 4. calculate w_t^i = P(z_t | x_t^i)
        wi = get_new_weight(z_t, xi)
        xi[4] = wi

        # 5. add p_t^i = [x_t^i w_t^i] to P_t^predict
        particles[i] = xi

    return particles


def correction_step(particles_pred):
    """Compute the correction of PF
    """
    n = len(particles_pred)
    particles = np.zeros((n, 5))

    w = particles_pred[:, -1]

    wtot = np.sum(w)
    # print("wtot: ", wtot)

    for i in range(n):
        # resample
        r = random.uniform(0, 1) * wtot
        # print("r: ", r)
        j = 0
        wsum = w[0]
        while wsum < r:
            # print("wsum: ", wsum)
            # print("j: ", j)
            j = j + 1
            wsum = wsum + w[j]

        # add to p_t
        particles[i] = particles_pred[j]

    return particles


def initialize_filter(x_min, x_max, y_min, y_max, n=NUM_PARTICLES):
    """Return P_0, n states in the work space"""
    random.seed(SEED)
    particles = np.zeros((n, 5))
    for i in range(n):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        theta = random.uniform(-math.pi, math.pi)
        # velocity = random.uniform(V_MIN, V_MAX)
        velocity = 0
        weight = 1 / n
        particles[i] = create_particle(x, y, theta, velocity, weight)
    # print('initial parts', particles)
    return particles


def create_particle(x: float, y: float, theta: float, velocity: float, weight: float):
    """Returns p_t (state + weight)
    """
    return np.array([x, y, theta, velocity, weight])


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
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
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


def rmse(x_vec, y_vec):
    rms_list = []
    error = 0
    count = 0
    for i in range(len(x_vec)):
        x = x_vec[i]
        y = y_vec[i]
        if (x > 0 and x < 10) or (y >= 10 and y < 0):
            d = min(abs(y - 0), abs(y + 10), abs(x - 0), abs(x - 10))
        else:
            d = abs(min(math.hypot(x, y), math.hypot(x - 10, y), math.hypot(x - 10, y + 10), math.hypot(x, y + 10)))
        error += d
        count += 1
        rms_list.append((np.sqrt(d / count)) ** 2)

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
        save_data(f_data, filepath + filename + "_filtered.csv")

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

    # Initialize filter

    N = 4  # number of states
    state_estimates = np.empty((NUM_PARTICLES, N, len(time_stamps)))
    gps_estimates = np.empty((2, len(time_stamps)))
    particles_prev = initialize_filter(x_min, x_max, y_min, y_max, NUM_PARTICLES)

    # Run filter over data
    # for t, _ in enumerate(time_stamps):
    for t in range(200):
        # print("t: ", t)
        delta_t = time_stamps[t] - time_stamps[t - 1]
        delta_t = delta_t / 1000000

        # Get control input
        u_t = np.array([[x_ddot[t]], [y_ddot[t]]])

        # Get sensor measurement
        yaw = wrap_to_pi((math.pi / 180) * (yaw_lidar[t]))
        # define inputs

        z_t = np.array([[X_LANDMARK-(-np.sin(yaw) * x_lidar[t] + np.cos(yaw) * y_lidar[t])],
                        [Y_LANDMARK-(-np.cos(yaw) * x_lidar[t] - np.sin(yaw) * y_lidar[t])],
                        [yaw]])

        # Prediction Step
        particles_pred = prediction_step(particles_prev, u_t, z_t, delta_t)

        # Correction Step
        particles_est = correction_step(particles_pred)

        #  For clarity sake/teaching purposes, we explicitly update t->(t-1)
        particles_prev = particles_est

        # Log Data

        state_estimates[:,:, t] = particles_prev[:, 0:4]

        x_gps, y_gps = convert_gps_to_xy(lat_gps=lat_gps[t],
                                         lon_gps=lon_gps[t],
                                         lat_origin=lat_origin,
                                         lon_origin=lon_origin)
        gps_estimates[:, t] = np.array([x_gps, y_gps])

    """FIND RMSE"""

    fig, ax = plt.subplots()

    def rmse_plot():
        x_vec = state_estimates[0, :]
        y_vec = state_estimates[1, :]
        rmse(x_vec, y_vec)

    def xyplot():
        # ax.set_ylim(-12,2.5)
        # ax.set_xlim(4,6)
        print(state_estimates[:1, 0, :50])
        print(state_estimates[:1, 1, :50])
        ax.plot(np.mean(state_estimates[:, 0, :50], axis = 0), np.mean(state_estimates[:, 1, :50], axis = 0), 'go', markersize=2)
        # ax.plot(gps_estimates[0], gps_estimates[1], 'bo', markersize=2)
        ax.plot([0, 10, 10, 0, 0], [0, 0, -10, -10, 0], 'r--')

        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        # ax.legend(["Estimated Position", "GPS Position", "Expected Path"], loc='upper left')

    def getellipses():
        lambda_, v = np.linalg.eig(covariance_estimates[:2, :2, 10])
        lambda_ = np.sqrt(lambda_)

        for i in range(0, 800, 100):
            for j in range(1, 3):
                x = state_estimates[0][i]
                y = state_estimates[1][i]
                # print(x,y)
                ell = Ellipse(xy=(x, y),
                              width=lambda_[0] * j * 2, height=lambda_[1] * j * 2,
                              angle=np.rad2deg(np.arccos(v[0, 0])), color='black', ls='--')

                ell.set_facecolor('none')
                ax.add_artist(ell)
        ax.scatter(x, y)

    """GET YAW PLOTS"""

    def getyaw():
        yaw_lidar_f = [wrap_to_pi((math.pi / 180) * x) for x in yaw_lidar]

        plt.plot(range(len(time_stamps)), state_estimates[5, :])
        plt.plot(range(len(time_stamps)), yaw_lidar_f)
        plt.legend(["Estimated Yaw", "Raw Yaw Measurement"])
        plt.xlabel("Timestep")
        plt.ylabel("Angle (Radians)")

    # getyaw()
    xyplot()
    # getellipses()
    plt.show()

    return 0


if __name__ == "__main__":
    main()
