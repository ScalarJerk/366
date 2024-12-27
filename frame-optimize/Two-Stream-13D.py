# hyperparameters at line 36

import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import shutil

def save_selected_frames(selected_frames, output_dir_name="Selected_Frames"):
    # Create the output directory inside frames_dir
    output_dir = os.path.join(frames_dir, output_dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory created: {output_dir}")
    else:
        print(f"Directory already exists: {output_dir}")
    
    # Copy each selected frame to the new directory
    for frame in selected_frames:
        shutil.copy(frame, output_dir)
    
    print(f"Selected frames saved to: {output_dir}")

# Path to the directory containing grayscaled depth images
frames_dir = "206_18_4_4_1_chair"

# Add check to ensure directory exists
if not os.path.exists(frames_dir):
    raise FileNotFoundError(f"Directory not found: {frames_dir}")

# Get absolute path for debugging
abs_path = os.path.abspath(frames_dir)
print(f"Looking for images in: {abs_path}")

# Hyperparameters!!!!!!!
batch_size = 8
num_selected_frames = 10 

# Preprocessing: Transform depth images
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Assuming grayscale depth images
    ])
    
    # If image is a string (file path), open it
    if isinstance(image, str):
        image = Image.open(image).convert("L")
    # If image is a numpy array, convert to PIL Image
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # If image is already a PIL Image, just convert to grayscale if needed
    elif isinstance(image, Image.Image):
        image = image.convert("L")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
        
    return transform(image)

# Compute optical flow
def compute_optical_flow(prev_frame, next_frame):
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame, next_frame, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    flow_image = np.uint8(255 * (mag / np.max(mag)))
    return flow_image

# Define Spatial Stream Network
class SpatialStream(nn.Module):
    def __init__(self):
        super(SpatialStream, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

# Define Temporal Stream Network
class TemporalStream(nn.Module):
    def __init__(self):
        super(TemporalStream, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

# Fuse Spatial and Temporal Streams
class TwoStreamNetwork(nn.Module):
    def __init__(self):
        super(TwoStreamNetwork, self).__init__()
        self.spatial_stream = SpatialStream()
        self.temporal_stream = TemporalStream()

    def forward(self, spatial_input, temporal_input):
        spatial_score = self.spatial_stream(spatial_input)
        temporal_score = self.temporal_stream(temporal_input)
        combined_score = spatial_score + temporal_score
        return combined_score

# Load and preprocess frames
frames = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
if not frames:
    raise FileNotFoundError(f"No image files found in {frames_dir}")

# Convert to full paths
frames = [os.path.join(frames_dir, f) for f in frames]
depth_images = [preprocess_image(frame) for frame in frames]

# Compute optical flow images
optical_flow_images = []
for i in range(len(frames) - 1):
    prev_frame = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)
    next_frame = cv2.imread(frames[i + 1], cv2.IMREAD_GRAYSCALE)
    flow_image = compute_optical_flow(prev_frame, next_frame)
    optical_flow_images.append(preprocess_image(Image.fromarray(flow_image)))

# Pad optical flow images to match depth images count
optical_flow_images.append(optical_flow_images[-1])  # Repeat last optical flow image

# Stack images into tensors
spatial_input = torch.stack(depth_images)  # Should be [batch, channels, height, width]
temporal_input = torch.stack(optical_flow_images)

# Debug print to verify tensor shapes
print("Spatial input shape:", spatial_input.shape)
print("Temporal input shape:", temporal_input.shape)

# Initialize the model
model = TwoStreamNetwork()

# Forward pass
scores = model(spatial_input, temporal_input)

# Select top N frames based on scores
selected_indices = torch.topk(scores.squeeze(), num_selected_frames).indices
selected_frames = [frames[idx] for idx in selected_indices]

print("Selected Frames:", selected_frames)
save_selected_frames(selected_frames)

'''
thank you chatgpt 
and thank you cursor
here's the gpt chat link https://chatgpt.com/share/676aa71b-6f90-8008-8646-b9bdcf5b7532
'''
