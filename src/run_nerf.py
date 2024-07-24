import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tqdm import tqdm
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch version: {torch.__version__}, Torch Device: {device}")

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

def data_loader(data_path, target_size=(80, 120)):
    with open(os.path.join(data_path, 'pose.json'), 'r') as f:
        poses_data = json.load(f)
    
    images = []
    poses = []
    
    for pose in poses_data:
        frame_path = os.path.join(data_path, pose['filename'])
        # Load and resize image
        image = Image.open(frame_path).resize(target_size, Image.LANCZOS)
        image = np.array(image)
        images.append(image)
        
        # Construct camera-to-world matrix (c2w)
        theta = np.deg2rad(pose['direction']['theta'])
        phi = np.deg2rad(pose['direction']['phi'])
        c2w = np.eye(4)
        c2w[:3, 3] = [pose['position']['x'], pose['position']['y'], pose['position']['z']]
        
        c2w = c2w @ glm_rotate(theta, axis='y')
        c2w = c2w @ glm_rotate(phi, axis='x')
        
        poses.append(c2w)
    
    images = np.array(images)
    poses = np.array(poses)
    height, width = target_size
    focal = 800.0 * (width / 1200)  # Adjust focal length based on new width
    
    return images, poses, height, width, focal

def get_rays(H, W, focal, c2w):
    device = c2w.device
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), 
                          torch.linspace(0, H-1, H, device=device))
    i, j = i.t(), j.t()
    
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def posenc(x, L_embed):
    rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.**i * x))
    return torch.cat(rets, -1)

def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, L_embed=6, rand=False):
    device = rays_o.device
    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
    z_vals = near * (1.-t_vals) + far * t_vals

    z_vals = z_vals.expand([rays_o.shape[0], N_samples])
    
    if rand:
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

    pts_flat = pts.reshape(-1, 3)
    pts_flat = posenc(pts_flat, L_embed=L_embed)
    raw = network_fn(pts_flat)
    raw = raw.reshape(list(pts.shape[:-1]) + [4])

    sigma_a = F.relu(raw[...,3])
    rgb = torch.sigmoid(raw[...,:3])

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.full_like(dists[...,:1], 1e10)], -1)
    alpha = 1. - torch.exp(-sigma_a * dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[...,:1]), 1.-alpha + 1e-10], -1), -1)[..., :-1]

    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)

    return rgb_map, depth_map, acc_map

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, output_ch=4, skips=[4]):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + 
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )
        self.output_linear = nn.Linear(W, output_ch)
    
    def forward(self, x):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = F.relu(l(h))
            if i in self.skips:
                h = torch.cat([h, x], -1)
        outputs = self.output_linear(h)
        return outputs

def train_nerf(images, poses, hwf, near, far, device, N_iters=1000, N_samples=30, chunk=1024*32, L_embed=6):
    H, W, focal = hwf
    H, W = int(H), int(W)

    model = NeRF(input_ch=3*L_embed*2+3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    for i in range(N_iters):
        img_i = np.random.randint(images.shape[0])
        target = torch.tensor(images[img_i], dtype=torch.float32).to(device)
        pose = torch.tensor(poses[img_i], dtype=torch.float32).to(device)

        rays_o, rays_d = get_rays(H, W, focal, pose[:3,:4])
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        rgb_list = []
        for j in range(0, rays_o.shape[0], chunk):
            rays_o_chunk = rays_o[j:j+chunk]
            rays_d_chunk = rays_d[j:j+chunk]
            rgb_chunk, _, _ = render_rays(model, rays_o_chunk, rays_d_chunk, near, far, N_samples, L_embed, rand=True)
            rgb_list.append(rgb_chunk)

        rgb = torch.cat(rgb_list, 0)

        optimizer.zero_grad()
        loss = F.mse_loss(rgb, target.reshape(-1, 3))
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'[TRAIN] Iter: {i+1} Loss: {loss.item()}')

    return model

def test_nerf(model, images, poses, hwf, near, far, device, N_samples=30, chunk=1024*32, L_embed=6):
    H, W, focal = hwf
    H, W = int(H), int(W)

    test_img = torch.tensor(images[-1], dtype=torch.float32).to(device)
    test_pose = torch.tensor(poses[-1], dtype=torch.float32).to(device)
    rays_o, rays_d = get_rays(H, W, focal, test_pose[:3,:4])
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    rgb_list = []
    with torch.no_grad():
        for j in range(0, rays_o.shape[0], chunk):
            rays_o_chunk = rays_o[j:j+chunk]
            rays_d_chunk = rays_d[j:j+chunk]
            rgb_chunk, _, _ = render_rays(model, rays_o_chunk, rays_d_chunk, near, far, N_samples, L_embed)
            rgb_list.append(rgb_chunk)

    rgb = torch.cat(rgb_list, 0).reshape(H, W, 3).cpu().numpy()
    test_img = test_img.cpu().numpy()

    # Ensure rgb and test_img have the same shape
    if rgb.shape != test_img.shape:
        rgb = rgb.transpose(1, 0, 2)

    psnr = -10. * np.log10(np.mean((rgb - test_img)**2))
    print(f'[TEST] PSNR: {psnr}')

    return rgb, psnr

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    global device

    images, poses, height, width, focal = data_loader(args.data_path, target_size=(80, 120))
    print(f"Loaded dataset with {len(images)} images, images shape: {images.shape}, poses shape: {poses.shape}, height: {height}, width: {width}, focal: {focal}")

    hwf = [height, width, focal]
    near, far = 2., 6.
    L_embed = 6

    if args.train or not os.path.exists(os.path.join(args.model_dir, "final_model.pth")):
        print("Training NeRF model...")
        model = train_nerf(images, poses, hwf, near, far, device, N_iters=args.iterations, L_embed=L_embed)
        torch.save(model.state_dict(), os.path.join(args.model_dir, "final_model.pth"))
        print("Training completed and model saved.")
    
    if args.test or not args.train:
        print("Testing NeRF model...")
        model = NeRF(input_ch=3*L_embed*2+3).to(device)
        model.load_state_dict(torch.load(os.path.join(args.model_dir, "final_model.pth")))
        rgb, psnr = test_nerf(model, images, poses, hwf, near, far, device, L_embed=L_embed)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.title(f'Rendered Image (PSNR: {psnr:.2f})')
        plt.savefig(os.path.join(args.output_dir, 'rendered_image.png'))
        plt.close()
        print(f"Testing completed. Rendered image saved to {os.path.join(args.output_dir, 'rendered_image.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NeRF on a dataset")
    parser.add_argument("--iterations", type=int, default=250, help="Number of iterations for training")
    parser.add_argument("--train", action='store_true', help="Flag to indicate training mode")
    parser.add_argument("--test", action='store_true', help="Flag to indicate test mode")
    parser.add_argument("--output_dir", type=str, default="src/output", help="Directory to store outputs")
    parser.add_argument("--model_dir", type=str, default="src/models", help="Directory to store model checkpoints")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    main(parser.parse_args())