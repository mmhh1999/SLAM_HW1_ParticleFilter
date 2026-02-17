import os
import re
import cv2
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='Directory containing PNG images')
parser.add_argument('--output', required=True, help='Output video file path')
parser.add_argument('--fps', type=int, default=30, help='Frames per second')
args = parser.parse_args()

image_dir = args.input
output_path = args.output
fps = args.fps
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

pattern = re.compile(r"^(\d+)\.png$")
files = []
for filename in os.listdir(image_dir):
    m = pattern.match(filename)
    if m:
        idx = int(m.group(1))
        files.append((idx, filename))
files.sort(key=lambda x: x[0])
if not files:
    raise RuntimeError(f"No numeric PNG files found in: {image_dir}")
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
