import numpy as np
import os
from map_reader import MapReader
from sensor_model import SensorModel, draw_rays_on_map
import matplotlib.pyplot as plt

# Configurable paths
MAP_PATH = r'data/map/wean.dat'
LOG_PATH = r'data/log/robotdata1.log'
OUTPUT_PATH = r'results'

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load map
tmp_map = MapReader(MAP_PATH)
occupancy_map = tmp_map.get_map()

# Initialize sensor model
sensor_model = SensorModel(occupancy_map)

# Pick a fixed particle location (center or user-defined)
# Example: [x, y, theta] in cm, theta in radians
particle = np.array([4055, 4005, np.pi])


# Generate multiple beam images for different log scans
import imageio
beam_imgs = []
max_frames = 20  # Number of frames to save
frame_idx = 0
with open(LOG_PATH, 'r') as f:
    for line in f:
        if line.startswith('L'):
            tokens = line.strip().split()
            ranges = np.array([float(r) for r in tokens[7:]])
            _, laserX, laserY = sensor_model.beam_range_finder_model_debug(ranges, particle)
            save_path = os.path.join(OUTPUT_PATH, f'beam_correctness_{frame_idx:03d}.png')
            draw_rays_on_map(occupancy_map, particle, laserX, laserY, save_path=save_path, show_particle=True)
            beam_imgs.append(save_path)
            frame_idx += 1
            if frame_idx >= max_frames:
                break

# Create gif
gif_path = os.path.join(OUTPUT_PATH, 'beam_correctness.gif')
images = [imageio.imread(img) for img in beam_imgs]
imageio.mimsave(gif_path, images, duration=0.5)
print(f'Beam correctness gif saved to: {gif_path}')
