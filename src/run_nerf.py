import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import cv2
import argparse
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Utility functions
def meshgrid_xy(tensor1: torch.Tensor, tensor2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    ii, jj = torch.meshgrid(tensor1, tensor2, indexing='xy')
    return ii, jj

def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    cumprod = torch.cumprod(tensor, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.
    return cumprod

# Ray computation functions
def get_ray_bundle(height: int, width: int, focal_length: float, tform_cam2world: torch.Tensor):
    ii, jj = meshgrid_xy(
        torch.arange(width).to(tform_cam2world),
        torch.arange(height).to(tform_cam2world)
    )
    directions = torch.stack([(ii - width * .5) / focal_length,
                              -(jj - height * .5) / focal_length,
                              -torch.ones_like(ii)], dim=-1)
    ray_directions = torch.sum(directions[..., None, :] * tform_cam2world[:3, :3], dim=-1)
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    return ray_origins, ray_directions

def compute_query_points_from_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    num_samples: int,
    randomize: bool = True
) -> (torch.Tensor, torch.Tensor):
    depth_values = torch.linspace(near_thresh, far_thresh, num_samples).to(ray_origins)
    if randomize:
        noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
        depth_values = depth_values + torch.rand(noise_shape).to(ray_origins) * (far_thresh - near_thresh) / num_samples
    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
    return query_points, depth_values

# Rendering function
def render_volume_density(
    radiance_field: torch.Tensor,
    ray_origins: torch.Tensor,
    depth_values: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3])
    rgb = torch.sigmoid(radiance_field[..., :3])
    one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
    dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1],
                      one_e_10.expand(depth_values[..., :1].shape)), dim=-1)
    alpha = 1. - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(-1)

    return rgb_map, depth_map, acc_map

# Positional encoding
def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    encoding = [tensor] if include_input else []
    frequency_bands = (
        2.0 ** torch.linspace(0.0, num_encoding_functions - 1, num_encoding_functions)
        if log_sampling
        else torch.linspace(2.0 ** 0.0, 2.0 ** (num_encoding_functions - 1), num_encoding_functions)
    ).to(tensor.device)

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    return torch.cat(encoding, dim=-1) if len(encoding) > 1 else encoding[0]

# NeRF model
class TinyNeRFModel(torch.nn.Module):
    def __init__(self, filter_size=128, num_encoding_functions=6):
        super(TinyNeRFModel, self).__init__()
        self.layer1 = torch.nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)

# Utility function for minibatch processing
def get_minibatches(inputs: torch.Tensor, chunksize: int = 1024 * 8):
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def train_nerf(
    images, poses, focal_length, 
    near_thresh, far_thresh, 
    num_encoding_functions=6, 
    num_iters=1000, 
    lr=5e-3, 
    display_every=100,
    depth_samples_per_ray=32
): 
    # Prepare data
    images = torch.from_numpy(images[..., :3]).to(device)
    poses = torch.from_numpy(poses).to(device)
    focal_length = torch.tensor(focal_length).to(device)
    height, width = images.shape[1:3]
    
    # Model setup
    model = TinyNeRFModel(num_encoding_functions=num_encoding_functions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Encoding function
    encode = lambda x: positional_encoding(x, num_encoding_functions=num_encoding_functions)
    
    # Training loop
    psnrs = []
    for i in range(num_iters):
        target_img_idx = np.random.randint(images.shape[0])
        target_img = images[target_img_idx]
        target_pose = poses[target_img_idx]
        
        rgb_predicted = run_nerf_iteration(
            height, width, focal_length, target_pose, 
            near_thresh, far_thresh, model, encode,
            depth_samples_per_ray
        )
        
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % display_every == 0:
            psnr = -10. * torch.log10(loss)
            psnrs.append(psnr.item())
            print(f"Iteration {i}: PSNR {psnr.item()}")
            
            # Generate prediction for the first image
            with torch.no_grad():
                first_img_prediction = run_nerf_iteration(
                    height, width, focal_length, poses[0], 
                    near_thresh, far_thresh, model, encode,
                    depth_samples_per_ray
                )
            
            # Save the predicted image and PSNR plot
            save_results(i, first_img_prediction, psnrs)
    
    # Save the model checkpoint after training
    torch.save(model.state_dict(), 'src/results/nerf_model_checkpoint.pth')
    
    return model, psnrs

# Function to run one iteration of NeRF
def run_nerf_iteration(height, width, focal_length, pose, near_thresh, far_thresh, model, encode, num_samples):
    ray_origins, ray_directions = get_ray_bundle(height, width, focal_length, pose)
    query_points, depth_values = compute_query_points_from_rays(
        ray_origins, ray_directions, near_thresh, far_thresh, num_samples
    )
    
    flattened_query_points = query_points.reshape((-1, 3))
    encoded_points = encode(flattened_query_points)
    
    batches = get_minibatches(encoded_points)
    predictions = [model(batch) for batch in batches]
    radiance_field = torch.cat(predictions, dim=0).reshape(list(query_points.shape[:-1]) + [4])
    
    rgb_predicted, _, _ = render_volume_density(radiance_field, ray_origins, depth_values)
    return rgb_predicted

# Function to save results
def save_results(iteration, rgb_predicted, psnrs):
    os.makedirs("src/results", exist_ok=True)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(rgb_predicted.detach().cpu().numpy())
    plt.title(f"Iteration {iteration}")
    plt.savefig(f"src/results/predicted_image_iter_{iteration}.png")
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(range(0, (len(psnrs) - 1) * 100 + 1, 100), psnrs)
    plt.title("PSNR over iterations")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.savefig("src/results/psnr_plot.png")
    plt.close()

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

def data_loader(data_path, target_size=(100, 100)):
    with open(os.path.join(data_path, 'pose.json'), 'r') as f:
        poses_data = json.load(f)
    
    images = []
    poses = []
    
    for pose in poses_data:
        frame_path = os.path.join(data_path, pose['filename'])
        # Load and resize image
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        images.append(image / 255.0)
        
        # Construct camera-to-world matrix (c2w)
        theta = np.deg2rad(pose['direction']['theta'])
        phi = np.deg2rad(pose['direction']['phi'])
        c2w = np.eye(4)
        c2w[:3, 3] = [pose['position']['x'], pose['position']['y'], pose['position']['z']]
        
        c2w = c2w @ glm_rotate(theta, axis='y')
        c2w = c2w @ glm_rotate(phi, axis='x')
        
        poses.append(c2w)

    images = np.array(images).astype(np.float32)
    poses = np.array(poses).astype(np.float32)
    focal = np.array(138.888, dtype=np.float64)
    
    return images, poses, focal

def test_nerf(
    model: torch.nn.Module,
    images: np.ndarray,
    poses: np.ndarray,
    focal_length: float,
    near_thresh: float,
    far_thresh: float,
    num_encoding_functions: int,
    depth_samples_per_ray: int,
    output_dir: str
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    encode = lambda x: positional_encoding(x, num_encoding_functions=num_encoding_functions)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert inputs to PyTorch tensors
    images = torch.from_numpy(images).to(device)
    poses = torch.from_numpy(poses).to(device)
    focal_length = torch.tensor(focal_length, dtype=torch.float32).to(device)

    video_frames = []
    
    with torch.no_grad():
        for i, (image, pose) in enumerate(zip(images, poses)):
            height, width = image.shape[:2]
            rgb_predicted = run_nerf_iteration(
                height, width, focal_length, pose, 
                near_thresh, far_thresh, model, encode,
                depth_samples_per_ray
            )
            
            # Convert tensors to numpy arrays for visualization
            original_image = (image.cpu().numpy() * 255).astype(np.uint8)
            predicted_image = (rgb_predicted.cpu().numpy() * 255).astype(np.uint8)
            
            # Concatenate original and predicted images side by side
            combined_image = np.concatenate((original_image, predicted_image), axis=1)
            
            # Save the combined image
            plt.figure(figsize=(10, 5))
            plt.imshow(combined_image)
            plt.title(f"Original vs Predicted - Image {i}")
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"test_result_{i}.png"))
            plt.close()
            
            # Convert to BGR for video
            combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
            video_frames.append(combined_image_bgr)
    
    # Create video
    video_path = os.path.join(output_dir, "nerf_results.mp4")
    height, width, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
    
    for frame in video_frames:
        video.write(frame)
    
    video.release()
    
    print(f"Test results saved in {output_dir}")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny NeRF PyTorch Implementation")
    parser.add_argument("--iterations", type=int, default=250, help="Number of iterations for training")
    parser.add_argument("--train", action="store_true", help="Run in training mode")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--output_dir", type=str, default="src/output", help="Directory to store outputs")
    parser.add_argument("--model_dir", type=str, default="src/models", help="Directory to store model checkpoints")
    parser.add_argument("--data_path", type=str, default='tiny_nerf_data', help="Path to the dataset")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Load data
    if args.data_path == 'tiny_nerf_data':
      data = np.load("src/tiny_nerf_data.npz")
      images, poses, focal = data["images"], data["poses"], data["focal"]
    else:
      images, poses, focal = data_loader(args.data_path)
    
    # Set parameters
    near_thresh, far_thresh = 2., 6.
    num_encoding_functions = 6
    depth_samples_per_ray = 32
    
    if args.train:
        os.makedirs(args.model_dir, exist_ok=True)
        model, psnrs = train_nerf(
            images, poses, focal, 
            near_thresh, far_thresh, 
            num_encoding_functions=num_encoding_functions,
            num_iters=args.iterations,
            lr=5e-3,
            depth_samples_per_ray=depth_samples_per_ray,
            display_every=100
        )
        torch.save(model.state_dict(), os.path.join(args.model_dir, 'nerf_model.pth'))
        print(f"Training complete! Model saved in {args.model_dir}")
    
    if args.test:
        model = TinyNeRFModel(num_encoding_functions=num_encoding_functions).to(device)
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'nerf_model.pth')))
        test_nerf(
            model, images, poses, focal, 
            near_thresh, far_thresh, 
            num_encoding_functions,
            depth_samples_per_ray,
            args.output_dir
        )