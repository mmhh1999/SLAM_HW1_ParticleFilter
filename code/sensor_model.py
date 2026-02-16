'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

# import cv2
from tqdm import tqdm
import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


# variables :
# LaserReadings = [x, y, theta, xl, yl, thetal, r1......r180]
#
# parameters :
# zHit, zRand, zShort, zMax, sigmaHit, lambdaShort
# L = 25 , n = laser beam numbers


def occupancy(x, resolution, occupancy_map):
    xMap = int(math.floor(x[0] // resolution))
    yMap = int(math.floor(x[1] // resolution))
    return occupancy_map[xMap, yMap]


def inBound(x, resolution, mapSize):
    xMap = math.floor(x[0] // resolution)
    yMap = math.floor(x[1] // resolution)
    if (xMap >= 0) and (xMap < mapSize) and (yMap >= 0) and (yMap < mapSize):
        return True
    else:
        return False


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 120 # 5
        self._z_short = 20 # 0.5
        self._z_max = 20 # 0.5
        self._z_rand = 35 # 200

        self._sigma_hit = 50
        self._lambda_short = 15
        self._dampening = 0.7

        self._min_probability = 0.35
        self._subsampling = 2

        """ Occupancy map specs """
        self.OccMap = occupancy_map
        self.OccMapSize = np.size(occupancy_map)
        self.resolution = 10

        """ Laser specs """
        self.laserMax = 8183  # Laser max range
        self.nLaser = 30
        self.laserX = np.zeros((self.nLaser, 1))
        self.laserY = np.zeros((self.nLaser, 1))
        self.beamsRange = np.zeros((self.nLaser, 1))

        # print("OCCUPANCY MAP size : \n", np.shape(self.OccMap))
        # print("OccMapSize Initialized: ", self.OccMapSize)

    def WrapToPi(self, angle):
        angle_wrapped = angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))
        return angle_wrapped

    def getProbability(self, z_star, z_reading):
        # hit
        if 0 <= z_reading <= self.laserMax:
            pHit = np.exp(-1 / 2 * (z_reading - z_star) ** 2 / (self._sigma_hit ** 2))
            pHit = pHit / (np.sqrt(2 * np.pi * self._sigma_hit ** 2))

        else:
            pHit = 0

        # short
        if 0 <= z_reading <= z_star:
            # eta = 1.0/(1-np.exp(-lambdaShort*z_star))
            eta = 1
            pShort = eta * self._lambda_short * np.exp(-self._lambda_short * z_reading)

        else:
            pShort = 0

        # max
        if z_reading >= self.laserMax:
            pMax = self.laserMax
        else:
            pMax = 0

        # rand
        if 0 <= z_reading < self.laserMax:
            pRand = 1 / self.laserMax
        else:
            pRand = 0

        p = self._z_hit * pHit + self._z_short * pShort + self._z_max * pMax + self._z_rand * pRand
        p /= (self._z_hit + self._z_short + self._z_max + self._z_rand)
        return p, pHit, pShort, pMax, pRand

    def rayCast(self, x_t1):
        """
        Vectorized version of the original ray casting logic.
        Keeps the same behavior: for each beam, find the first range r where:
        - in map bounds (xInt < 800 and yInt < 800)  [same as original]
        - abs(OccMap[yInt, xInt]) > 0.35            [same as original]
        """

        beamsRange = np.zeros(self.nLaser)
        laserX = np.zeros(self.nLaser)
        laserY = np.zeros(self.nLaser)
        angs = np.zeros(self.nLaser)
        L = 25

        xc = x_t1[0]
        yc = x_t1[1]
        myPhi = x_t1[2]

        ang = myPhi - np.pi / 2
        ang = self.WrapToPi(ang)

        offSetX = xc + L * np.cos(ang)
        offSetY = yc + L * np.sin(ang)

        angStep = np.pi / self.nLaser

        # set ray step size (same as original)
        r = np.linspace(0, self.laserMax, 800)

        for i in range(self.nLaser):
            ang += angStep
            ang = self.WrapToPi(ang)

            # Compute all candidate points along the ray at once
            x = offSetX + r * np.cos(ang)
            y = offSetY + r * np.sin(ang)

            xInt = np.floor(x / self.resolution).astype(int)
            yInt = np.floor(y / self.resolution).astype(int)

            # Same bound check as original code (only < 800, no >= 0)
            in_bounds = (0 <= xInt) & (xInt < 800) & (0 <= yInt) & (yInt < 800)

            # Avoid invalid indexing by applying in_bounds first
            hit = np.zeros_like(in_bounds, dtype=bool)
            valid_idx = np.nonzero(in_bounds)[0]
            if valid_idx.size > 0:
                xi = xInt[valid_idx]
                yi = yInt[valid_idx]
                hit_vals = np.abs(self.OccMap[yi, xi]) > 0.35
                hit[valid_idx] = hit_vals

            # Find the first hit along r (same "break at first hit" behavior)
            hit_indices = np.flatnonzero(hit)
            if hit_indices.size > 0:
                idx = hit_indices[0]
                beamsRange[i] = r[idx]

                # Keep the same (unused) phi computation semantics
                _ = np.arctan2((offSetY - yInt[idx]), (offSetX - xInt[idx]))

                angs[i] = ang
                laserX[i] = xInt[idx]
                laserY[i] = yInt[idx]
                # implicit "break" achieved by only taking the first idx

        return beamsRange, laserX, laserY


    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        # Subsample the laser readings to match the number of beams
        step = int(180 / self.nLaser)
        z_reading = [z_t1_arr[n] for n in range(0, 180, step)]

        # Perform ray casting to get expected ranges and hit points
        zt_star, laserX, laserY = self.rayCast(x_t1)

        # Compute probabilities for each beam
        probs = np.zeros(self.nLaser)
        log_likelihood_sum = 0.0
        for i in range(self.nLaser):
            probs[i], _, _, _, _ = self.getProbability(zt_star[i], z_reading[i])
            probs[i] = max(probs[i], 1e-12)  # Avoid zero probabilities
            log_likelihood_sum += np.log(probs[i])

        # Compute the overall likelihood
        # q = self.nLaser / np.abs(log_likelihood_sum)
        # q = float(np.exp(log_likelihood_sum / self.nLaser))  # Geometric mean to avoid underflow
        mean_log_likelihood = log_likelihood_sum / self.nLaser
        q = float(np.exp(self._dampening * mean_log_likelihood)) 
        return q, probs, laserX, laserY
    
    
    
    def beam_range_finder_model_debug(self, z_t1_arr, x_t1):
        """
        Debug helper: returns per-beam component stats for ONE particle.
        Keeps the same ray casting + getProbability logic.
        """
        step = int(180 / self.nLaser)
        z_reading = [z_t1_arr[n] for n in range(0, 180, step)]

        zt_star, laserX, laserY = self.rayCast(x_t1)

        p_mix = np.zeros(self.nLaser, dtype=np.float64)
        p_hit = np.zeros(self.nLaser, dtype=np.float64)
        p_short = np.zeros(self.nLaser, dtype=np.float64)
        p_max = np.zeros(self.nLaser, dtype=np.float64)
        p_rand = np.zeros(self.nLaser, dtype=np.float64)

        for i in range(self.nLaser):
            p, ph, ps, pm, pr = self.getProbability(zt_star[i], z_reading[i])
            p_mix[i] = p
            p_hit[i] = ph
            p_short[i] = ps
            p_max[i] = pm
            p_rand[i] = pr

        # fraction of each component in the mixture numerator, averaged over beams
        denom_w = (self._z_hit + self._z_short + self._z_max + self._z_rand)
        num_hit = self._z_hit * p_hit
        num_short = self._z_short * p_short
        num_max = self._z_max * p_max
        num_rand = self._z_rand * p_rand
        num_sum = num_hit + num_short + num_max + num_rand

        # Avoid divide-by-zero
        frac_hit = np.where(num_sum > 0, num_hit / num_sum, 0.0)
        frac_short = np.where(num_sum > 0, num_short / num_sum, 0.0)
        frac_max = np.where(num_sum > 0, num_max / num_sum, 0.0)
        frac_rand = np.where(num_sum > 0, num_rand / num_sum, 0.0)

        dbg = {
            # Per-beam mean of raw components
            "mean_p_hit": float(np.mean(p_hit)),
            "mean_p_short": float(np.mean(p_short)),
            "mean_p_max": float(np.mean(p_max)),
            "mean_p_rand": float(np.mean(p_rand)),

            # Per-beam mean of mixture fractions
            "mean_frac_hit": float(np.mean(frac_hit)),
            "mean_frac_short": float(np.mean(frac_short)),
            "mean_frac_max": float(np.mean(frac_max)),
            "mean_frac_rand": float(np.mean(frac_rand)),

            # Mixture p range over beams (for sanity)
            "beam_p_mix_min": float(np.min(p_mix)),
            "beam_p_mix_med": float(np.median(p_mix)),
            "beam_p_mix_max": float(np.max(p_mix)),
        }
        return dbg, laserX, laserY
