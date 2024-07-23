# run_nerf.py
# Run NeRF on a dataset

import torch
import json
import argparse
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tqdm import tqdm
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch version: {torch.__version__}, Torch Device: {device}")

# Define NeRF model architecture
class NeRFModel(torch.nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=4, skips=[4]):
        super(NeRFModel, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips
        
        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_ch, W)] + 
            [torch.nn.Linear(W, W) if i not in self.skips else torch.nn.Linear(W + input_ch, W) for i in range(D-1)]
        )
        self.output_linear = torch.nn.Linear(W, output_ch)
    
    def forward(self, x):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = torch.relu(l(h))
            if i in self.skips:
                h = torch.cat([x, h], -1)
        outputs = self.output_linear(h)
        return outputs

def glm_rotate(angle, axis='y'):
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'y':
        return np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'x':
        return np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Unsupported axis")

def data_loader(data_path):
    # Path to the frames directory
    frames_dir = os.path.join(data_path, 'frames')
    
    # Load the pose.json file
    with open(os.path.join(data_path, 'pose.json'), 'r') as f:
        poses_data = json.load(f)
    
    images = []
    poses = []
    
    for pose in poses_data:
        frame_path = os.path.join(data_path, pose['filename'])
        # Load image and convert to numpy array
        image = np.array(Image.open(frame_path))
        images.append(image)
        
        # Construct camera-to-world matrix (c2w)
        theta = np.deg2rad(pose['direction']['theta'])
        phi = np.deg2rad(pose['direction']['phi'])
        c2w = np.eye(4)
        c2w[:3, 3] = [pose['position']['x'], pose['position']['y'], pose['position']['z']]
        
        # Assuming 'theta' is the yaw and 'phi' is the pitch
        c2w = c2w @ glm_rotate(theta, axis='y')
        c2w = c2w @ glm_rotate(phi, axis='x')
        
        poses.append(c2w)
    
    images = np.array(images)
    poses = np.array(poses)
    height, width = images.shape[1:3]
    focal = 800.0 # Assuming focal length is 800.0
    
    return images, poses, height, width, focal

# Training function
def train(model, dataloader, iterations, model_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.MSELoss()
    model.train()
    
    for i in tqdm(range(iterations)):
        for data in dataloader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        if (i + 1) % 1000 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f"model_{i+1}.pth"))

# Testing function
def test(model, dataloader):
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Implement evaluation metrics here
            pass

# Argument parser
def get_args():
    parser = argparse.ArgumentParser(description="Run NeRF on a dataset")
    parser.add_argument("--iterations", type=int, default=250, help="Number of iterations for training")
    parser.add_argument("--train", action='store_true', help="Flag to indicate training mode")
    parser.add_argument("--test", action='store_true', help="Flag to indicate test mode")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to store outputs")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to store model checkpoints")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    return parser.parse_args()

# Main function
def main():
    args = get_args()

    # Create output and model directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Load data
    images, poses, height, width, focal = data_loader(args.data_path)
    print(f"Loaded dataset with {len(images)} images, images shape: {images.shape}, poses shape: {poses.shape}, height: {height}, width: {width}, focal: {focal}")

    # # Initialize model
    model = NeRFModel().to(device)

    # # Train or test
    # if args.train:
    #     train(model, dataloader, args.iterations, args.model_dir)
    # if args.test:
    #     test(model, dataloader)

if __name__ == "__main__":
    main()