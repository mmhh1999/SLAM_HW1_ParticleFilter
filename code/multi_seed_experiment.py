import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
import pandas as pd

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling
from main import init_particles_freespace

OUTPUT_PATH = 'results'
MAP_PATH = 'data/map/wean.dat'
LOG_PATH = 'data/log/robotdata1.log'

seeds = [42, 123, 999]
N = 500  # Number of particles
VAR_THRESHOLD = 1e5  # Convergence threshold (adjust as needed)
LAST_N = 50  # Number of steps for final statistics

map_obj = MapReader(MAP_PATH)
occupancy_map = map_obj.get_map()

results = []
all_var_curves = []
all_ll_curves = []

for seed in seeds:
    print(f'Running seed={seed}...')
    np.random.seed(seed)
    random.seed(seed)
    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()
    X_bar = init_particles_freespace(N, occupancy_map, debug=False)
    var_x_list = []
    var_y_list = []
    ll_list = []
    ess_list = []
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
                log_likelihoods = np.zeros(N)
                for m in range(N):
                    x_t0 = X_bar[m, 0:3]
                    x_t1 = motion_model.update(u_t0, u_t1, x_t0)
                    X_bar_new[m, 0:3] = x_t1
                    w_t, _, _, _ = sensor_model.beam_range_finder_model(ranges, x_t1)
                    X_bar_new[m, 3] = w_t
                    log_likelihoods[m] = np.log(w_t + 1e-12)
                X_bar = X_bar_new
                u_t0 = u_t1.copy()
                # Normalize weights for ESS
                w = X_bar[:, 3].astype(np.float64)
                w_sum = np.sum(w)
                w_norm = (w / w_sum) if w_sum > 0 else np.zeros_like(w)
                ess = 1.0 / np.sum(w_norm ** 2) if w_sum > 0 else 0.0
                ess_list.append(ess)
                # Resampling
                X_bar = resampler.low_variance_sampler(X_bar)
                xs = X_bar[:, 0]
                ys = X_bar[:, 1]
                var_x = np.var(xs)
                var_y = np.var(ys)
                var_x_list.append(var_x)
                var_y_list.append(var_y)
                ll_list.append(np.max(log_likelihoods))
    # Convergence criteria
    var_sum = np.array(var_x_list) + np.array(var_y_list)
    converged = np.any(var_sum < VAR_THRESHOLD)
    t_conv = int(np.argmax(var_sum < VAR_THRESHOLD)) if converged else -1
    final_var = float(np.mean(var_sum[-LAST_N:])) if len(var_sum) >= LAST_N else float(np.mean(var_sum))
    final_ll = float(np.mean(ll_list[-LAST_N:])) if len(ll_list) >= LAST_N else float(np.mean(ll_list))
    min_ess = float(np.min(ess_list)) if ess_list else 0.0
    results.append({
        'seed': seed,
        'converged': converged,
        't_conv': t_conv,
        'final_var': final_var,
        'final_ll': final_ll,
        'min_ess': min_ess
    })
    all_var_curves.append(var_sum)
    all_ll_curves.append(ll_list)
    print(f'seed={seed} done, converged={converged}, t_conv={t_conv}, final_var={final_var:.2f}')

# Save results table
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_PATH, 'seed_results.csv'), index=False)
print('\nSeed | Converged | t_conv | FinalVar | FinalLL | MinESS')
print(results_df[['seed', 'converged', 't_conv', 'final_var', 'final_ll', 'min_ess']])

# Plot variance curves for all seeds
plt.figure(figsize=(10,6))
for i, var_curve in enumerate(all_var_curves):
    plt.plot(var_curve, label=f'seed={seeds[i]}')
plt.xlabel('Timestep')
plt.ylabel('Total Variance (X+Y)')
plt.title('Variance vs Timestep (Multiple Seeds)')
plt.legend()
plt.savefig(os.path.join(OUTPUT_PATH, 'multi_seed_variance.png'))
plt.close()

# Plot log-likelihood curves for all seeds
plt.figure(figsize=(10,6))
for i, ll_curve in enumerate(all_ll_curves):
    plt.plot(ll_curve, label=f'seed={seeds[i]}')
plt.xlabel('Timestep')
plt.ylabel('Max Log-Likelihood')
plt.title('Max Log-Likelihood vs Timestep (Multiple Seeds)')
plt.legend()
plt.savefig(os.path.join(OUTPUT_PATH, 'multi_seed_loglikelihood.png'))
plt.close()

print('Analysis complete. See results/seed_results.csv, multi_seed_variance.png, multi_seed_loglikelihood.png')
