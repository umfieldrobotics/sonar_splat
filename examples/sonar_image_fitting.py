import math
import os
import time
from pathlib import Path
from typing import Literal, Optional

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# sys.path.insert(1, '/home/advaith/Documents/waveGS/sonar/')
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import numpy as np
import torch
import tyro
from PIL import Image
from torch import Tensor, optim

from gsplat import rasterization, rasterization_2dgs

from gsplat import _rasterization, _sonar_rasterization 

from sonar.utils import visualize_gaussians, create_frustum_wireframe
from matplotlib import pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')
# from gsplat import azimuth_antenna_gain_projection

# Set the seed
seed = 42

# NumPy
np.random.seed(seed)

# PyTorch
torch.manual_seed(seed)

# For GPU (if using CUDA)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU

# Ensures deterministic behavior in GPU operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points

        fov_x = np.radians(130.)
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        """Random gaussians"""
        # bd = 2 (ortho) 100 (pinhole)

        bd = 0.5
        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        # self.means = torch.zeros(self.num_points, 3, device=self.device)
        
        # # self.means[0] += torch.tensor([0.5, 0.5, 0.0], device=self.device)
        # # self.means[1] += torch.tensor([0.75, 0.75, 0.0], device=self.device)
        # # self.means[2] += torch.tensor([1.0, 1.0, 0.0], device=self.device) #left is further
        # self.means[:,0] = torch.linspace(0.0, 2.0, self.num_points)
        # self.means[:, 0] += 2.0
        # self.means[:, 1] += torch.linspace(-1, 1.0, self.num_points, device=self.device)
        # self.means[-1, 0] = 2.5 
        # self.means[-1, 1] = 0.0

        # self.means[:, 2] += torch.linspace(-1, 1, self.num_points, device=self.device)
        # self.means[:,2] += 5 # flat and close to z plane
        # self.means[:,0] *= 3.0 
        # self.means[:,1] *= 3.0
        # self.means[:,2] *= 1.0

        
        # *0.1 (ortho) *1 (pinhole)
        # self.scales = torch.rand(self.num_points, 3, device=self.device) * 0.1/3
        self.scales = torch.ones(self.num_points, 3, device=self.device) * 0.05/3
        # self.scales[0, :] *= 3.0
        d = 3
        self.rgbs = torch.rand(self.num_points, d, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((self.num_points), device=self.device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 3.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.50],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )

        #tilt down 30 degrees about the y axis 
        rot = torch.tensor([
            [math.cos(-np.radians(45)), 0.0, math.sin(-np.radians(45)), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-math.sin(-np.radians(45)), 0.0, math.cos(-np.radians(45)), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], device=self.device)
        self.viewmat = torch.matmul(self.viewmat, rot.T)

        #rotate by 90 degree in y
        # rot = torch.tensor([
        #     [0.0, 0.0, 1.0, 0.0],
        #     [0.0, 1.0, 0.0, 0.0],
        #     [-1.0, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 1.0],
        # ], device=self.device)
        # self.viewmat = torch.matmul(self.viewmat, rot.T)

        visualize_gaussians([self.means], [torch.inverse(self.viewmat)], colors=None, start_size=0.1, end_size=0.1, others=None)
        self.background = torch.zeros(d, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

    def train(
        self,
        iterations: int = 1000,
        lr: float = 0.01,
        save_imgs: bool = False,
        model_type: Literal["3dgs", "2dgs", "3dgs_autograd", "sonar_sph"] = "3dgs",
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    ):
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 2  # rasterization, backward
        K = torch.tensor(
            [
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1],
            ],
            device=self.device,
        )
        
        sph = False

        if model_type == "3dgs":
            rasterize_fnc = rasterization
        elif model_type == "2dgs":
            rasterize_fnc = rasterization_2dgs
        elif model_type == "3dgs_autograd":
            rasterize_fnc = _rasterization
        elif model_type == "sonar_sph":
            rasterize_fnc = _sonar_rasterization
            sph = True
            hfov = 130.0
            vfov = 20.0
            max_range = 8.0
            range_res =  max_range/604 #m/pixel
            intermediate_azimuth_res = hfov/516 # rad/pixel
            self.H = int(hfov*1/intermediate_azimuth_res)
            self.W = int(max_range*1/range_res)
            K = torch.tensor(
            [
                [1.0/range_res, 0.0, 0.0], # range resolution => 0.0596 m #v -> y coordinate
                [0.0, 180/torch.pi*(1/intermediate_azimuth_res), self.H/2.],
                [0, 0, 1],
            ],
            device=self.device,
            )
        num_poses = 10
        viewmats = self.viewmat.repeat(num_poses, 1, 1)
        # viewmats[:,0,3] = torch.linspace(0.0, 4.0, num_poses)

        for iter in range(iterations):
            start = time.time()

            renders, meta = rasterize_fnc(
                means=self.means,
                quats=self.quats / self.quats.norm(dim=-1, keepdim=True),
                scales=self.scales,
                opacities=torch.sigmoid(self.opacities),
                sat_probability=torch.zeros_like(self.opacities),
                colors=torch.sigmoid(self.rgbs),
                viewmats=viewmats[iter%num_poses, None],
                Ks=K[None],
                width=self.W,
                height=self.H,
                packed=False,
                near_plane=np.radians(-vfov/2),
                far_plane=np.radians(vfov/2),
                # render_mode="D",
                camera_model = camera_model,
                sph=sph,
            )
            out_img = renders[0]
            out_alpha_img = out_img[:,:,0].unsqueeze(-1)
            radii = meta["radii"]
            trans = meta["transmittances"]
            radii_mask = (radii > 0.0).squeeze()
            trans_mask = (trans > 0.01).squeeze()
            frustum = create_frustum_wireframe(vfov, hfov, max_range).transform(torch.inverse( viewmats[iter%num_poses]).cpu().numpy())
            visualize_gaussians([self.means[trans_mask], self.means[~trans_mask]], 
                                [torch.inverse( viewmats[iter%num_poses])], 
                                colors=[np.array([0,1,0]), np.array([1,0,0])], start_size=0.5, end_size=0.5, others=[frustum])

            if sph:
                # out_alpha_img = azimuth_antenna_gain_projection(out_alpha_img)
                # out_img = out_alpha_img.repeat(1,1,3) # convert to 3 channel rgb image
                out_img = out_alpha_img.repeat(1,1,1)

            # print(out_img.shape)
            torch.cuda.synchronize()
            times[0] += time.time() - start

            loss = mse_loss(out_alpha_img, self.gt_image)
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
        print(
            f"Per step(s):\nRasterization: {times[0]/iterations:.5f}, Backward: {times[1]/iterations:.5f}"
        )
    
def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 100000,
    save_imgs: bool = True,
    img_path: Path = None,
    iterations: int = 1000,
    lr: float = 0.01,
    model_type: Literal["3dgs", "2dgs", "3dgs_autograd", "sonar_sph"] = "3dgs",
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
) -> None:

    gt_image = image_path_to_tensor(img_path)

    if model_type == 'sonar_sph':
        gt_image = gt_image.permute(1,0,2)



    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
        model_type=model_type,
        camera_model=camera_model,
    )


if __name__ == "__main__":
    tyro.cli(main)
