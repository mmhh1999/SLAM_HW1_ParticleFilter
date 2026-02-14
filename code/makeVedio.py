import os
import re
import cv2
from tqdm import tqdm

# Directory containing png images
image_dir = r"C:\Users\12527\Desktop\SLAM_HW1_ParticleFilter\code\results"

# Output video path
output_path = os.path.join(image_dir, "animation.mp4")

# Video settings
fps = 30  # Change this to control speed (e.g., 30 for faster)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4

# Regular expression to match filenames like 0.png, 12.png, etc.
pattern = re.compile(r"^(\d+)\.png$")

# Collect numeric png files
files = []
for filename in os.listdir(image_dir):
    m = pattern.match(filename)
    if m:
        idx = int(m.group(1))
        files.append((idx, filename))

# Sort by numeric order
files.sort(key=lambda x: x[0])

if not files:
    raise RuntimeError(f"No numeric PNG files found in: {image_dir}")

# Read first frame to get frame size
first_path = os.path.join(image_dir, files[0][1])
first_frame = cv2.imread(first_path, cv2.IMREAD_COLOR)
if first_frame is None:
    raise RuntimeError(f"Failed to read the first image: {first_path}")

height, width = first_frame.shape[:2]

# Create video writer
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if not writer.isOpened():
    raise RuntimeError("Failed to open VideoWriter. Try a different codec or output path.")

# Write frames with progress bar
for _, filename in tqdm(files, desc="Writing video", unit="frame"):
    img_path = os.path.join(image_dir, filename)
    frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"[Warning] Skipped unreadable file: {img_path}")
        continue

    # Ensure consistent size
    if frame.shape[0] != height or frame.shape[1] != width:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    writer.write(frame)

writer.release()
print(f"Video saved to: {output_path}")
