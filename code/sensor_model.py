'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        
        Beam Range Finder Model parameters:
        z_hit, z_short, z_max, z_rand: weights for 4 distributions
        sigma_hit: std dev for Gaussian (p_hit)
        lambda_short: decay rate for exponential (p_short)
        """
        self._z_hit = 10
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 10000

        self._sigma_hit = 100
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2
        
        # Store occupancy map
        self._occupancy_map = occupancy_map
        
        # Map resolution: 10cm per pixel
        self._resolution = 10
        
        # Laser offset from robot center: 25cm forward
        self._laser_offset = 25

    def _ray_casting(self, x, y, theta):
        """
        Cast a ray from (x, y) at angle theta and find distance to nearest obstacle.
        
        param[in] x, y : laser position in world frame (cm)
        param[in] theta : ray angle in world frame (rad)
        param[out] distance : distance to obstacle (cm), or max_range if none found
        
        Algorithm: Step along the ray until hitting an obstacle or max_range
        """
        # Step size for ray marching (in cm)
        step_size = 5
        
        # Current position along ray
        dist = 0
        
        while dist < self._max_range:
            # Calculate current point along ray
            ray_x = x + dist * np.cos(theta)
            ray_y = y + dist * np.sin(theta)
            
            # Convert to map coordinates (pixel indices)
            map_x = int(ray_x / self._resolution)
            map_y = int(ray_y / self._resolution)
            
            # Check bounds
            if (map_x < 0 or map_x >= self._occupancy_map.shape[1] or
                map_y < 0 or map_y >= self._occupancy_map.shape[0]):
                return dist  # Out of map, return current distance
            
            # Check if obstacle (occupancy > threshold)
            if self._occupancy_map[map_y, map_x] > self._min_probability:
                return dist
            
            # Check if unknown (-1 in some maps)
            if self._occupancy_map[map_y, map_x] < 0:
                return dist
            
            dist += step_size
        
        return self._max_range

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        
        Algorithm (Probabilistic Robotics Table 6.1):
        For each laser beam, compute P(z | x) as weighted sum of 4 distributions
        """
        # Extract particle pose
        x, y, theta = x_t1
        
        # Calculate laser position (25cm forward from robot center)
        laser_x = x + self._laser_offset * np.cos(theta)
        laser_y = y + self._laser_offset * np.sin(theta)
        
        # Use log probability to avoid numerical underflow
        log_prob = 0.0
        
        # Laser beams: 180 readings spanning 180 degrees
        # Starting from RIGHT (-90°) going LEFT (+90°) in robot frame
        # Subsample for efficiency
        for i in range(0, 180, self._subsampling):
            # Actual measurement
            z_measured = z_t1_arr[i]
            
            # Skip invalid measurements
            if z_measured >= self._max_range or z_measured <= 0:
                continue
            
            # Beam angle in world frame
            # i=0 is -90° (right), i=90 is 0° (front), i=180 is +90° (left)
            beam_angle = theta + np.radians(i - 90)
            
            # Expected measurement via ray casting
            z_expected = self._ray_casting(laser_x, laser_y, beam_angle)
            
            # Compute probability for this beam
            p = self._compute_beam_probability(z_measured, z_expected)
            
            # Accumulate log probability
            if p > 0:
                log_prob += np.log(p)
            else:
                log_prob += -100  # Very small probability
        
        # Convert back from log space
        prob_zt1 = np.exp(log_prob)
        
        return prob_zt1

    def _compute_beam_probability(self, z_measured, z_expected):
        """
        Compute P(z_measured | z_expected) using 4-component mixture model
        
        Components:
        1. p_hit: Gaussian centered at z_expected (correct measurement with noise)
        2. p_short: Exponential decay (unexpected obstacle closer than expected)
        3. p_max: Point mass at max_range (laser returns max when hitting nothing)
        4. p_rand: Uniform distribution (random unexplained noise)
        """
        z = z_measured
        z_star = z_expected
        z_max = self._max_range
        
        # 1. p_hit: Gaussian distribution
        # P(z | z*) = N(z; z*, sigma^2) normalized to [0, z_max]
        if 0 <= z <= z_max:
            p_hit = np.exp(-0.5 * ((z - z_star) / self._sigma_hit) ** 2)
            # Normalization (approximate, using CDF)
            # eta = 1.0 / (norm.cdf(z_max, z_star, self._sigma_hit) - norm.cdf(0, z_star, self._sigma_hit))
            # p_hit *= eta
        else:
            p_hit = 0
        
        # 2. p_short: Exponential distribution (unexpected obstacles)
        # Models objects between robot and expected obstacle
        if 0 <= z <= z_star and z_star > 0:
            eta_short = 1.0 / (1.0 - np.exp(-self._lambda_short * z_star))
            p_short = eta_short * self._lambda_short * np.exp(-self._lambda_short * z)
        else:
            p_short = 0
        
        # 3. p_max: Point mass at max range
        # Laser didn't hit anything, returns max
        if z >= z_max - 1:  # Small tolerance
            p_max = 1.0
        else:
            p_max = 0
        
        # 4. p_rand: Uniform distribution for random noise
        if 0 <= z < z_max:
            p_rand = 1.0 / z_max
        else:
            p_rand = 0
        
        # Weighted sum of all components
        p_total = (self._z_hit * p_hit + 
                   self._z_short * p_short + 
                   self._z_max * p_max + 
                   self._z_rand * p_rand)
        
        # Normalize by sum of weights
        p_total /= (self._z_hit + self._z_short + self._z_max + self._z_rand)
        
        return p_total
