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
        pass

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        num_particles = X_bar.shape[0]
        weights = X_bar[:, 3]
        
        # Normalize weights
        weights_sum = np.sum(weights)
        if weights_sum == 0:
            weights_normalized = np.ones(num_particles) / num_particles
        else:
            weights_normalized = weights / weights_sum
        
        # Sample indices according to weights
        indices = np.random.choice(num_particles, size=num_particles, 
                                   replace=True, p=weights_normalized)
        
        # Resample particles
        X_bar_resampled = X_bar[indices, :]
        X_bar_resampled[:, 3] = 1.0 / num_particles
        
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        num_particles = X_bar.shape[0]
        X_bar_resampled = np.zeros_like(X_bar)
        
        # Normalize weights
        weights = X_bar[:, 3]
        weights_sum = np.sum(weights)
        if weights_sum == 0:
            weights_normalized = np.ones(num_particles) / num_particles
        else:
            weights_normalized = weights / weights_sum
        
        # Low variance resampling algorithm (Table 4.4)
        r = np.random.uniform(0, 1.0 / num_particles)
        c = weights_normalized[0]
        i = 0
        
        for m in range(num_particles):
            U = r + m * (1.0 / num_particles)
            while U > c:
                i += 1
                if i >= num_particles:
                    i = num_particles - 1
                    break
                c += weights_normalized[i]
            X_bar_resampled[m, :] = X_bar[i, :]
        
        # Reset weights
        X_bar_resampled[:, 3] = 1.0 / num_particles
        
        return X_bar_resampled
