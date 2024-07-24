import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
data = np.load('tiny_nerf_data.npz')
images = torch.from_numpy(data['images']).float().to(device)
poses = torch.from_numpy(data['poses']).float().to(device)
focal = float(data['focal'])
H, W = images.shape[1:3]

# Split data
testimg, testpose = images[101], poses[101]
images = images[:100,...,:3]
poses = poses[:100]

# Positional encoding function
def posenc(x, L_embed=6):
    rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.**i * x))
    return torch.cat(rets, -1)

# NeRF model
class TinyNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, output_ch=4):
        super(TinyNeRF, self).__init__()
        self.net = nn.ModuleList([nn.Linear(input_ch, W)] + 
                                 [nn.Linear(W, W) if i != D-1 else nn.Linear(W, output_ch) for i in range(D)])
    
    def forward(self, x):
        h = x
        for i, l in enumerate(self.net):
            h = self.net[i](h)
            if i < len(self.net) - 1:
                h = nn.functional.relu(h)
        return h

# Ray generation function
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i, device=device)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

# Rendering function
def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):
    # Compute 3D query points
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

    # Run network
    pts_flat = pts.reshape(-1, 3)
    pts_flat = posenc(pts_flat, L_embed=10)  # Increase L_embed to match input_ch
    raw = network_fn(pts_flat)
    raw = raw.reshape(list(pts.shape[:-1]) + [4])

    # Compute opacities and colors
    sigma_a = nn.functional.relu(raw[...,3])
    rgb = torch.sigmoid(raw[...,:3])

    # Do volume rendering
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.full_like(dists[...,:1], 1e10)], -1)
    alpha = 1.-torch.exp(-sigma_a * dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[...,:1]), 1.-alpha + 1e-10], -1), -1)[..., :-1]

    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)

    return rgb_map, depth_map, acc_map

# Training loop
def train_nerf(images, poses, focal, N_iters=1000, N_samples=64, lr=5e-4):
    model = TinyNeRF(input_ch=63, output_ch=4).to(device)  # Match input_ch with posenc output
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    psnrs = []
    iternums = []
    
    for i in tqdm(range(N_iters + 1)):
        img_i = np.random.randint(images.shape[0])
        target = images[img_i]
        pose = poses[img_i]
        rays_o, rays_d = get_rays(H, W, focal, pose)
        
        # Flatten rays for batch processing
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples, rand=True)
        rgb = rgb.reshape(H, W, 3)
        loss = torch.mean((rgb - target) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 25 == 0:
            # Render the holdout view for logging
            with torch.no_grad():
                rays_o, rays_d = get_rays(H, W, focal, testpose)
                rays_o = rays_o.reshape(-1, 3)
                rays_d = rays_d.reshape(-1, 3)
                rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
                rgb = rgb.reshape(H, W, 3)
                loss = torch.mean((rgb - testimg) ** 2)
                psnr = -10. * torch.log10(loss)

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(10,4))
            plt.subplot(121)
            plt.imshow(rgb.cpu().numpy())
            plt.title(f'Iteration: {i}')
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title('PSNR')
            plt.show()
    
    return model, psnrs, iternums

# Run the training
model, psnrs, iternums = train_nerf(images, poses, focal)

print('Training complete')