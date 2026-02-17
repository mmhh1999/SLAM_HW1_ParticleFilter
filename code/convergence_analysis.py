import numpy as np
import matplotlib.pyplot as plt
import os

# Configurable paths
OUTPUT_PATH = 'results'
NUM_PARTICLES = 500
MAP_PATH = 'data/map/wean.dat'
LOG_PATH = 'data/log/robotdata1.log'

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

# Load map
map_obj = MapReader(MAP_PATH)
occupancy_map = map_obj.get_map()

motion_model = MotionModel()
sensor_model = SensorModel(occupancy_map)
resampler = Resampling()


# Initialize particles
from main import init_particles_freespace
X_bar = init_particles_freespace(NUM_PARTICLES, occupancy_map, debug=False)

var_x_list = []
var_y_list = []

# Open log
logfile = open(LOG_PATH, 'r')

u_t0 = None
u_t1 = None

for line in logfile:
    if line.startswith('O'):
        # Odometry update
        tokens = line.strip().split()
        u_t1 = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
        if u_t0 is None:
            u_t0 = u_t1.copy()
    elif line.startswith('L'):
        if u_t0 is None or u_t1 is None:
            continue  # Skip until odometry is initialized
        tokens = line.strip().split()
        ranges = np.array([float(r) for r in tokens[7:]])
        # Motion update
        X_bar_new = np.zeros_like(X_bar)
        for m in range(NUM_PARTICLES):
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)
            X_bar_new[m, 0:3] = x_t1
            # Sensor update
            w_t, _, _, _ = sensor_model.beam_range_finder_model(ranges, x_t1)
            X_bar_new[m, 3] = w_t
        X_bar = X_bar_new
        u_t0 = u_t1.copy()
        # Resampling
        X_bar = resampler.low_variance_sampler(X_bar)
        # Record variance
        xs = X_bar[:, 0]
        ys = X_bar[:, 1]
        var_x = np.var(xs)
        var_y = np.var(ys)
        var_x_list.append(var_x)
        var_y_list.append(var_y)

# Plot convergence analysis
plt.figure()
plt.plot(var_x_list, label='Variance X')
plt.plot(var_y_list, label='Variance Y')
plt.xlabel('Timestep')
plt.ylabel('Variance')
plt.title('Particle Variance vs Timestep (Convergence Analysis)')
plt.legend()
plt.savefig(os.path.join(OUTPUT_PATH, 'convergence_variance.png'))
plt.close()
print(f'Convergence analysis plot saved to: {os.path.join(OUTPUT_PATH, "convergence_variance.png")}')
