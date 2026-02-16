'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here

        """

        

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        
        # Number of particles
        num_particles = X_bar.shape[0]
        
        # Normalize weights
        total_weight = np.sum(X_bar[:, 3])
        if total_weight > 0:
            X_bar[:, 3] /= total_weight
        else:
            # If all weights are zero, uniform resampling
            X_bar[:, 3] = 1.0 / num_particles
        
        # Low variance resampling
        resampled_particles = []
        r = np.random.uniform(0, 1.0 / num_particles)
        cumulative_weight = X_bar[0, 3]
        index = 0
        
        for particle_idx in range(1, num_particles + 1):
            u = r + (particle_idx - 1) / num_particles
            while u > cumulative_weight:
                index += 1
                cumulative_weight += X_bar[index, 3]
            resampled_particles.append(X_bar[index])
        
        return np.array(resampled_particles)