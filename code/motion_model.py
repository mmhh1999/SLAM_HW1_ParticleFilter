'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        
        Odometry Motion Model noise parameters (Ref: Probabilistic Robotics Table 5.6)
        alpha1: rotation noise from rotation
        alpha2: rotation noise from translation  
        alpha3: translation noise from translation
        alpha4: translation noise from rotation
        """
        self._alpha1 = 0.001
        self._alpha2 = 0.001
        self._alpha3 = 0.2
        self._alpha4 = 0.001

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        # Extract odometry readings
        x_bar0, y_bar0, theta_bar0 = u_t0
        x_bar1, y_bar1, theta_bar1 = u_t1
        
        # Extract current particle state
        x0, y0, theta0 = x_t0
        
        # Step 1: Compute relative motion in odometry frame
        delta_trans = np.sqrt((x_bar1 - x_bar0)**2 + (y_bar1 - y_bar0)**2)
        
        if delta_trans < 0.01:
            delta_rot1 = 0.0
        else:
            delta_rot1 = self._normalize_angle(
                np.arctan2(y_bar1 - y_bar0, x_bar1 - x_bar0) - theta_bar0
            )
        
        delta_rot2 = self._normalize_angle(theta_bar1 - theta_bar0 - delta_rot1)
        
        # Step 2: Add Gaussian noise
        delta_rot1_var = self._alpha1 * delta_rot1**2 + self._alpha2 * delta_trans**2
        delta_trans_var = self._alpha3 * delta_trans**2 + self._alpha4 * (delta_rot1**2 + delta_rot2**2)
        delta_rot2_var = self._alpha1 * delta_rot2**2 + self._alpha2 * delta_trans**2
        
        delta_rot1_hat = delta_rot1 - np.random.normal(0, np.sqrt(max(delta_rot1_var, 1e-10)))
        delta_trans_hat = delta_trans - np.random.normal(0, np.sqrt(max(delta_trans_var, 1e-10)))
        delta_rot2_hat = delta_rot2 - np.random.normal(0, np.sqrt(max(delta_rot2_var, 1e-10)))
        
        # Step 3: Apply noisy motion to particle state
        x1 = x0 + delta_trans_hat * np.cos(theta0 + delta_rot1_hat)
        y1 = y0 + delta_trans_hat * np.sin(theta0 + delta_rot1_hat)
        theta1 = self._normalize_angle(theta0 + delta_rot1_hat + delta_rot2_hat)
        
        return np.array([x1, y1, theta1])
