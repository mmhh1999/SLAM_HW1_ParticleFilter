'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time


"""
For debugging and testing
"""
def plot_single_particle_trajectory(traj, occupancy_map, out_path):
    """
    traj: list of [x_cm, y_cm, theta_rad]
    occupancy_map: 2D numpy array, shape (H,W)
    out_path: path to save the output image
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.shape[1] != 3:
        raise ValueError("Trajectory should be a list of [x, y, theta] values.")
    
    if traj.shape[0] == 0:
        return

    # world(cm) -> grid index (each cell = 10cm)
    xs = traj[:, 0] / 10.0
    ys = traj[:, 1] / 10.0

    plt.figure()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, occupancy_map.shape[1], 0, occupancy_map.shape[0]])
    plt.plot(xs, ys, linewidth=2)         # trajectory line
    plt.scatter(xs[0], ys[0], c='y', s=20)       # start
    plt.scatter(xs[-1], ys[-1], c='r', s=20)     # end
    plt.savefig(out_path, dpi=200)
    plt.close()


"""
Helper functions
"""

def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])
    return occupancy_map  # Return map for reuse


def visualize_timestep(X_bar, tstep, output_path, occupancy_map=None):
    # Redraw map background each frame to prevent background loss
    if occupancy_map is not None:
        plt.clf()  # Clear figure
        plt.imshow(occupancy_map, cmap='Greys')
        plt.axis([0, 800, 0, 800])
    
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    plt.scatter(x_locs, y_locs, c='r', marker='o', s=5)
    # plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # Initialize particles only in free space (unoccupied cells).
    # Faster convergence than random init over the whole area.
    """
    TODO : Add your code here
    This version converges faster than init_particles_random
    """
    # 1. find free space in the occupancy map
    free_mask = (occupancy_map >= 0.0) & (occupancy_map < 1)  # 1 can be tuned
    free_y, free_x = np.where(free_mask)
    if free_x.size == 0:
        raise ValueError("No free space found in the occupancy map.")

    # 2. sample particle positions from free space
    idx = np.random.choice(free_x.size, num_particles, replace=True)
    cell_x = free_x[idx].astype(np.float64)
    cell_y = free_y[idx].astype(np.float64)

    # 3. convert grid indices to world coordinates (cm)
    resolution = 10.0  # cm per pixel
    # Add random offset within each cell to avoid clustering at cell centers
    x0_vals = (cell_x + np.random.uniform(0.0, 1.0, num_particles)) * resolution
    y0_vals = (cell_y + np.random.uniform(0.0, 1.0, num_particles)) * resolution
    theta0_vals = np.random.uniform(-3.14, 3.14, num_particles)

    # 4. initialize weights for all particles
    w0_vals = np.full((num_particles, 1), 1.0 / num_particles, dtype=np.float64)

    X_bar_init = np.column_stack((x0_vals, y0_vals, theta0_vals, w0_vals))
    return X_bar_init



if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    # For testing and debugging
    # np.random.seed(0)
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    # X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    """
    For testing and debugging
    """
    traj = []  # store single particle trajectory [x_cm, y_cm, theta_rad]

    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)

    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # For debugging and testing
        if meas_type != "L":
            continue

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        for m in range(0, num_particles):
            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)

            ## For debugging: print du vs dx
            dux = u_t1[0] - u_t0[0]
            duy = u_t1[1] - u_t0[1]
            duth = u_t1[2]-u_t0[2]
            dxx = x_t1[0] - x_t0[0]
            dxy = x_t1[1] - x_t0[1]
            dxth = x_t1[2]-x_t0[2]
            if time_idx % 10 == 0:
                print(">>> du=(%.2f, %.2f, %.2f), dx=(%.2f, %.2f, %.2f)" % (dux, duy, duth, dxx, dxy, dxth))

            # For debugging: store the trajectory of the first particle
            if m == 0:  
                traj.append(x_t1.copy())

            """
            SENSOR MODEL
            """
            # if (meas_type == "L"):
            #     z_t = ranges
            #     w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
            #     X_bar_new[m, :] = np.hstack((x_t1, w_t))
            # else:
            #     X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

            # Motion-only debug: do not use sensor model
            X_bar_new[m, :] = np.hstack((x_t1, 1.0 / num_particles))


        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
        # if (meas_type == "L"):
        #     X_bar = resampler.low_variance_sampler(X_bar)

        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output, occupancy_map)

    # For testing and debugging
    # Save trajectory (single particle)
    plot_single_particle_trajectory(traj, occupancy_map, os.path.join(args.output, "single_particle_traj.png"))
    print("Saved single particle trajectory to results/single_particle_traj.png")

