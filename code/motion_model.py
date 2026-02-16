'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
# import math



class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        # self._alpha1 = 0.01*10
        # self._alpha2 = 0.01*10
        # self._alpha3 = 0.01*10
        # self._alpha4 = 0.01*10
        self._alpha1 = 0.0005
        self._alpha2 = 0.0005
        self._alpha3 = 0.0025
        self._alpha4 = 0.0025

    def WrapToPi(self,angle):
        
        angWrap = angle - 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
        return angWrap    
    def sample(self,mu,sigma):
        return np.random.normal(mu,sigma)

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """

        # Check for no motion
        if np.allclose(u_t1, u_t0):
            return x_t0.copy()
        
        # Compute odometry differences
        dx = u_t1[0] - u_t0[0]
        dy = u_t1[1] - u_t0[1]
        dtheta = u_t1[2] - u_t0[2]
        
        # Rotation and translation components
        delta_rot1 = np.arctan2(dy, dx) - u_t0[2]
        delta_rot1 = self.WrapToPi(delta_rot1)
        delta_trans = np.sqrt(dx**2 + dy**2)
        delta_rot2 = dtheta - delta_rot1
        delta_rot2 = self.WrapToPi(delta_rot2)
        
        # Add noise to rotations and translation
        rot1_noise = self.sample(0, self._alpha1 * delta_rot1**2 + self._alpha2 * delta_trans**2)
        trans_noise = self.sample(0, self._alpha3 * delta_trans**2 + self._alpha4 * delta_rot1**2 + self._alpha4 * delta_rot2**2)
        rot2_noise = self.sample(0, self._alpha1 * delta_rot2**2 + self._alpha2 * delta_trans**2)
        
        rot1 = delta_rot1 - rot1_noise
        trans = delta_trans - trans_noise
        rot2 = delta_rot2 - rot2_noise
        
        rot1 = self.WrapToPi(rot1)
        rot2 = self.WrapToPi(rot2)
        
        # Update particle position
        x_new = x_t0[0] + trans * np.cos(x_t0[2] + rot1)
        y_new = x_t0[1] + trans * np.sin(x_t0[2] + rot1)
        theta_new = x_t0[2] + rot1 + rot2
        
        return np.array([x_new, y_new, theta_new])


        # SAMPLE : return 1/2 * sigma(i=1 -> 12) rand(-b,b)

        # return np.random.rand(3)

