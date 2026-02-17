import numpy as np
import matplotlib.pyplot as plt
import os
import time

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling
from main import init_particles_freespace

OUTPUT_PATH = 'results'
MAP_PATH = 'data/map/wean.dat'
LOG_PATH = 'data/log/robotdata1.log'

N_list = [200, 500, 1000]

map_obj = MapReader(MAP_PATH)
occupancy_map = map_obj.get_map()
motion_model = MotionModel()
sensor_model = SensorModel(occupancy_map)
resampler = Resampling()

results = {}

for N in N_list:
    print(f'Running with N={N} particles...')
    X_bar = init_particles_freespace(N, occupancy_map, debug=False)
    var_x_list = []
    var_y_list = []
    t0 = time.time()
    u_t0 = None
    u_t1 = None
    with open(LOG_PATH, 'r') as logfile:
        for line in logfile:
            if line.startswith('O'):
                tokens = line.strip().split()
                u_t1 = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
                if u_t0 is None:
                    u_t0 = u_t1.copy()
            elif line.startswith('L'):
                if u_t0 is None or u_t1 is None:
                    continue
                tokens = line.strip().split()
                ranges = np.array([float(r) for r in tokens[7:]])
                X_bar_new = np.zeros_like(X_bar)
                for m in range(N):
                    x_t0 = X_bar[m, 0:3]
                    x_t1 = motion_model.update(u_t0, u_t1, x_t0)
                    X_bar_new[m, 0:3] = x_t1
                    w_t, _, _, _ = sensor_model.beam_range_finder_model(ranges, x_t1)
                    X_bar_new[m, 3] = w_t
                X_bar = X_bar_new
                u_t0 = u_t1.copy()
                X_bar = resampler.low_variance_sampler(X_bar)
                xs = X_bar[:, 0]
                ys = X_bar[:, 1]
                var_x = np.var(xs)
                var_y = np.var(ys)
                var_x_list.append(var_x)
                var_y_list.append(var_y)
    runtime = time.time() - t0
    results[N] = {
        'var_x': var_x_list,
        'var_y': var_y_list,
        'runtime': runtime
    }
    print(f'N={N} done, runtime: {runtime:.2f}s')

# Plot variance curves
plt.figure(figsize=(10,6))

# Plot total variance (x方差+y方差) for each N
plt.figure(figsize=(10,6))
for N in N_list:
    total_var = np.array(results[N]['var_x']) + np.array(results[N]['var_y'])
    plt.plot(total_var, label=f'N={N}')
plt.xlabel('Timestep')
plt.ylabel('Total Variance (X+Y)')
plt.title('Particle Number Sensitivity: Total Variance vs Timestep')
plt.legend()
plt.savefig(os.path.join(OUTPUT_PATH, 'particle_number_sensitivity.png'))
plt.close()

# Print runtime summary
print('\nParticle Number Sensitivity Runtime:')
for N in N_list:
    print(f'N={N}: {results[N]["runtime"]:.2f}s')

print('Analysis complete. See results/particle_number_sensitivity.png for curves.')
