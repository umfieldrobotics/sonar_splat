# import matplotlib
# matplotlib.use("QtAgg")
import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from gsplat.utils import save_ply
import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import PIL  
import wandb
import cv2 
import viser
import yaml
from datasets.traj import *
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)
from sonar.dataset.dataloader import sample_points_in_elevation
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy, PruneOnlyStrategy
from gsplat.optimizers import SelectiveAdam
from gsplat.strategy.ops import _update_param_with_optimizer
from gsplat import _rasterization, _sonar_rasterization #, rasterization_sonar2d

from gsplat.rendering import azimuth_antenna_gain_projection
import matplotlib 
# matplotlib.use("TkAgg")
print(matplotlib.get_backend())
import matplotlib.pyplot as plt

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)                       
                                    
from sonar.utils import (visualize_gaussians, 
                         create_frustum_wireframe, 
                         visualize_img_and_gt, 
                         visualize_val_train_poses,
                        )

from sonar.img_metrics import calculate_niqe, total_variation, compute_icv

from sonar.dataset.dataloader import (
                                    SonarSensorDataParser, 
                                    SonarSensorDataset
                                )

from gsplat.cuda._torch_impl_sonar import _saturate_returns


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"
    render_traj_interp_val: int = 1

    viz_samples: bool = False

    override_max_range: float = -1

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    apply_mask: bool = False
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/sonar_test"
    # Every N images there is a test image
    test_every: int = 8
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    streak_interval: int = 100
    streak_interval_ratio: float = 0.5
    streak_start_step: int = 100
    streak_end_step: int = 2000
    skip_streak: bool = False
    opacity_supervision_start_step: int = 1000
    opacity_supervision_end_step: int = 2000
    opacity_supervision_weight: float = 0.0
    opacity_supervision_thresh: float = 0.2
    opacity_penalty_weight:float  = 1.0
    viz_gaussians: bool = False

    percent_select_true: float = 0.2
    num_rand: int = 1000

    elevate_start_step: int = 0
    elevate_end_step: int = 5000
    elevation_sampling_every: int = 2

    start_from_frame: int = 0
    end_at_frame: int = 100000

    range_clear_end: int = 0
    skip_frames: int = 0
    # Normalize the world space
    normalize_world_space: bool = False # default to False
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "ortho" # default to ortho for sonar

    wandb: bool = False
    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    elevate_loss_select: int = 100
    elevate_num_samples: int = 10
    elevate_sampling_duty_cycle: float = 0.5

    # Number of training steps
    max_steps: int = 30_000
    # Number of training steps to start saving
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: list(range(100, 30_000, 1000)))
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: list(range(100, 30_000, 100)))

    # Initialization strategy
    init_type: str = "random" # "predefined" or "random"
    # Initial number of GSs. Ignored if using predefined
    init_num_pts: int = 100_000

    #sat probability initialization
    opacity_prior_weight: float = 0.5
    max_size_prior_weight: float = 100.0
    sat_region_prior_weight: float = 5.0
    sat_bg_prior_weight: float = 1.0
    sat_sparsity_prior_weight: float = 0.0

    init_threshold: float = 0.2
    randomize_elevation: bool = False
    num_random_points: int = 10000
    img_threshold: float = 0.2
    dset_keep_only_first_image: bool = False
    train: bool = False

    render_traj_amplitude: float = 0.1
    render_traj_freq: float = 1.0

    range_clear_start: int = 200
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using predefined
    init_extent: float = 1.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    render_eval: bool = False
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    color_prior_weight: float = 0.5

    # Near plane clipping distance
    near_plane: float = -10
    # Far plane clipping distance
    far_plane: float = 10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy, PruneOnlyStrategy] = field(
        default_factory=DefaultStrategy
    )
    color_prior_asym: float = 0.2
    
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False
    sat_thresh: float = 0.05
    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = True

    lpips_net: Literal["vgg", "alex"] = "alex"

    # Scanning sonar config
    intermediate_azimuth_resolution: float = 0.1
    
    init_scale: float = 0.5 # (m)

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, PruneOnlyStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: SonarSensorDataParser,
    init_type: str = "predefined",
    init_num_pts: int = 100_000,
    init_extent: float = 1.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "predefined":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb).float()
    elif init_type == "random":
        points = init_extent * scene_scale*0.9/2. * (torch.rand((init_num_pts, 3)) * 2 - 1)
        points[:,2] = 0
        rgbs = torch.rand((init_num_pts, 3))
        
        # # override point position
        # points[:,0] = 0.0
        # points[:,1] = 10.0
        
    else:
        raise ValueError("Please specify a correct init_type: predefined or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    # dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    # dist_avg = torch.sqrt(dist2_avg)
    # scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
 
    #initialize sat_probability 50% to be 0.0 and 50% to be 1.0

    # sat_probability = ((torch.rand((points.shape[0],)) > 0.5).float() - 0.5) * 5.0
    sat_probability = torch.ones((points.shape[0],)).float() * -10.0  #all initialized to 0 prob 

    #randomly make 10 1.0 
    # sat_probability[torch.randint(0, points.shape[0], (10,))] = 2.0
    # override scale size
    scales = torch.ones_like(points) #0.5
    scales = torch.log(scales * init_scale)
    
    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    # # override quats
    # quats[...,:] = quats[0,:]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ("sat_probability", torch.nn.Parameter(sat_probability), 2e-2),
    ]

    # color is SH coefficients.
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
    colors[:, 0, :] = rgb_to_sh(rgbs)
    params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-2))
    params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-2 / 20))

    # colors = torch.ones((N, 1)) *  # [N, 1]
    # params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = SonarSensorDataParser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            override_max_range=cfg.override_max_range,
            num_init_samples=cfg.init_num_pts,
            init_threshold=cfg.init_threshold,
            img_threshold=cfg.img_threshold,
            num_random_points=cfg.num_random_points,
            render_dir=self.render_dir,
            randomize_elevation=cfg.randomize_elevation,
            skip_frames=cfg.skip_frames,
            range_clear_start=cfg.range_clear_start,
            range_clear_end=cfg.range_clear_end,
            start_from_frame=cfg.start_from_frame,
            end_at_frame=cfg.end_at_frame,
            apply_mask=cfg.apply_mask,
            dset_keep_only_first_image=cfg.dset_keep_only_first_image,
        )
        self.trainset = SonarSensorDataset(
            self.parser,
            split="train",
        )
        self.valset = SonarSensorDataset(self.parser, split="val")

        print("\033[32mTrain set size: \033[0m", len(self.trainset))
        print("\033[32mVal set size: \033[0m", len(self.valset))
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        elif isinstance(self.cfg.strategy, PruneOnlyStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.bil_grid_optimizers = []


        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )


    
    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        use_polar: bool = False,
        
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internallyI 
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        sat_probability = torch.nn.functional.sigmoid(self.splats["sat_probability"]) # [N,]
        image_ids = kwargs.pop("image_ids", None)
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
    

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_powers, info = _sonar_rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            max_range=self.parser.max_range,
            sat_probability=sat_probability,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            sph=True,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )

        assert render_powers.max() <= 1.0
        if masks is not None:
            render_powers[~masks] = 0
        return render_powers, info

    def detect_saturated_returns(self, img: torch.Tensor, 
                                 thresh: float = 0.05, 
                                 low_thresh: float = 0.01, 
                                 high_thresh: float = 0.05):
        """
        Basic """
        img = img.detach()
        H, W, = img.shape[1:3]
        avg_img = img.squeeze().mean(dim=0)
        std_img = img.squeeze().std(dim=0)


        mask = (avg_img < thresh) 
        return mask.view(1, -1, 1).repeat(H, 1, 1)


    def find_regions(self, sat_mask: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Find start and end indices of contiguous regions where sat_mask is 1.

        Args:
            sat_mask (torch.Tensor): A binary tensor of shape (1, N) where N is the length.

        Returns:
            List[Tuple[int, int]]: A list of (start, end) indices of contiguous regions.
        """
        intervals = []
        left = None  # Track the start of an interval

        for i in range(sat_mask.shape[1]):
            if sat_mask[0, i] == 1 and left is None:
                left = i  # Mark the start of a new region

            elif sat_mask[0, i] == 0 and left is not None:
                intervals.append((left, i))  # End the previous region
                left = None  # Reset for the next region

        # If the mask ends in a valid region
        if left is not None:
            intervals.append((left, sat_mask.shape[1]))

        return intervals

            
    def encourage_saturated_returns(self, 
                                    means2d: torch.Tensor, 
                                    gaussian_ids: torch.Tensor, 
                                    sat_probability: torch.Tensor, 
                                    sat_mask: torch.Tensor):
        H, W = sat_mask.shape[0:2]
        rendered_means2d = means2d.int().squeeze()
        rendered_sat_probability = sat_probability

        #find the pixels that are saturated
        sat_pixels = torch.nonzero(sat_mask.squeeze())
        sat_pixels = sat_pixels[:,0], sat_pixels[:,1]

        regions = self.find_regions(sat_mask)
        selected = torch.zeros(rendered_means2d.shape[0]).to(means2d).bool()

        max_vals = []
        valid_idx = torch.where(sat_mask[0])[0]
        for interval in regions: 
            #find the gaussians that are rendered at the saturated pixels
            gaussians_in_mask = (rendered_means2d[...,0] >= interval[0]) & (rendered_means2d[...,0] <= interval[1])
            
            if gaussians_in_mask.sum().item() > 0:
                selected_probabilities = rendered_sat_probability * gaussians_in_mask.float()
                max_vals.append(selected_probabilities.max())
                selected[torch.argmax(selected_probabilities)] = True

        #visualize the selected gaussians
        # plt.imshow(sat_mask.squeeze().detach().cpu().numpy())
        # plt.scatter(rendered_means2d[..., 0].detach().cpu().numpy(), rendered_means2d[..., 1].detach().cpu().numpy(), marker='x', color='red')
        # plt.show()
        return max_vals, ~selected #return the bg gaussians
   


    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        img_ids = [data["image_id"] for data in self.trainset]
        log_ids = set(np.random.choice(img_ids, size=min(len(img_ids), 10), replace=False).tolist())
        # visualize_val_train_poses(self.trainset, self.valset, self.parser.points)
        #print all the prior weights in a table 
        print("-"*100)
        print(f"Opacity prior weight: {cfg.opacity_prior_weight}")
        print(f"Max size prior weight: {cfg.max_size_prior_weight}")
        print(f"Sat region prior weight: {cfg.sat_region_prior_weight}")
        print(f"Sat bg prior weight: {cfg.sat_bg_prior_weight}")
        print(f"Sat sparsity prior weight: {cfg.sat_sparsity_prior_weight}")
        print(f"Color prior weight: {cfg.color_prior_weight}")
        print("-"*100)


        

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
            
            sat_state = (step > cfg.streak_start_step) & (step < cfg.streak_end_step) & (cfg.skip_streak)
            # sat_turn = ((step % cfg.streak_interval) < (int(cfg.streak_interval * cfg.streak_interval_ratio)))
            # sat_state = sat_start

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device)  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]y
            near_plane = data["near_plane"].to(device)
            far_plane = data["far_plane"].to(device)

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            
            renders, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use, # TODO: keep this for future use
                near_plane=near_plane, # TODO: check if this is needed
                far_plane=far_plane, # TODO: check if this is needed
                image_ids=image_ids,
                masks=masks,
                batch_per_iter=2000,
            )
            out_img = renders[0]

            viz_mask = (info["transmittances"] > 0.9).squeeze(0) & (info["depths"] < self.parser.max_range).squeeze()

            if cfg.viz_gaussians and step % 1000 == 0:
                frustum = create_frustum_wireframe(vertical_fov=self.parser.vfov_deg, 
                                               horizontal_fov=self.parser.hfov_deg, 
                                               max_range=self.parser.max_range).transform(camtoworlds.cpu().detach().numpy().squeeze())
                viz_means = torch.tensor(self.splats["means"])
                    
                visualize_gaussians(xyz=[viz_means[viz_mask], viz_means[~viz_mask]], 
                                    poses=[camtoworlds.cpu().detach().numpy().squeeze()], 
                                    colors=[np.array([0,1,0]), np.array([1, 0,0])],
                                    others=[frustum],
                                    start_size=0.5,
                                    end_size=0.5)
         
            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            if sat_state or step > cfg.streak_end_step: #if in sat period or after end, want to train on entire image
                sat_mask = torch.ones_like(pixels[:,0,:,:])
            else: #only train on non dark regions
                sat_mask = (pixels.mean(dim=1) > cfg.sat_thresh)
            
            pred_for_loss = (sat_mask[:,None,:,:] * out_img).squeeze() #* (~sat_mask.squeeze()) #dont take losses in regions with azimuth streaking, model it with saturation gains. 
            pixels_for_loss = (sat_mask[:,None,:,:] * pixels).squeeze() #* (~sat_mask.squeeze())

            out_img = out_img.squeeze()
            pixels = pixels_for_loss.squeeze()
                       
            pred_for_loss_l1 = pred_for_loss
            pixels_for_loss_l1 = pixels_for_loss

            #visualize the selected indices 
            if cfg.viz_samples: 
                matplotlib.use("TkAgg")
                canvas = np.zeros((pixels.shape[0], pixels.shape[1], 3))
                #set the pixels to red for selected_indices and blue for random_indices
                canvas[selected_indices[0].cpu(), selected_indices[1].cpu()] = [1, 0, 0]
                canvas[random_indices[0].cpu(), random_indices[1].cpu()] = [0, 0, 1]
                plt.imshow(canvas)
                plt.show()

            free_space_mask = pixels_for_loss < 0.1

            l1loss = F.l1_loss(pred_for_loss_l1, pixels_for_loss_l1)
            
            ############################################################################## max size loss ##############################################################################
            max_size_loss = torch.mean(torch.relu(torch.exp(self.splats['scales']) - cfg.init_scale))

            if step >= cfg.opacity_supervision_start_step and step <= cfg.opacity_supervision_end_step:
                free_space_mask = pixels_for_loss < 0.1
                opacity_gt_img = (pixels_for_loss > cfg.opacity_supervision_thresh).float()
                opacity_gt_for_loss = opacity_gt_img

                opacity_pred_for_loss = info["opacity_returns"].squeeze()
                opacity_pred_for_loss_img = opacity_pred_for_loss
                opacity_pred_for_loss = opacity_pred_for_loss_img
                
                opacity_supervision_loss = F.l1_loss(opacity_pred_for_loss, opacity_gt_for_loss) #increase only areas where there is certain return 
            else:
                opacity_supervision_loss = torch.tensor(0.0)


            if data["image_id"].item() in log_ids or cfg.dset_keep_only_first_image:
                output_folder = 'out'
                os.makedirs(f'{output_folder}/tmp/polar', exist_ok=True)
                os.makedirs(f'{output_folder}/tmp/cart', exist_ok=True)

                #repeat the sat mask in dim 1 
                sat_mask_viz = sat_mask.repeat(1, pixels.shape[0], 1, 1)

                opacity_viz = opacity_pred_for_loss_img.permute(1, 0)
                opacity_gt_viz = opacity_gt_img.permute(1, 0)
                opacity_viz = torch.cat((opacity_viz, opacity_gt_viz), dim=1)

                img = torch.cat((out_img, info["unsaturated_returns"].squeeze(), pixels),dim=0)

                #log to wandb 
                if cfg.wandb:
                    wandb.log({f"train/Image_{data['image_id'].item()}": wandb.Image(img.permute(1,0).detach().cpu().numpy())}, step=step)
                    wandb.log({f"train/opacity_viz": wandb.Image(opacity_viz.detach().cpu().numpy())}, step=step)

            ##############################################################################saturation probability priors ##############################################################################

            all_sats = torch.sigmoid(self.splats["sat_probability"])
            sat_region_prior = 0.0

            sat_bg_prior = 0.0
            sat_sparsity_prior = ((torch.abs(all_sats))*(torch.abs(1 - all_sats))).mean()
                                                                                                 

            ############################################################################## photometric losses ##############################################################################
            all_opacities = torch.sigmoid(self.splats["opacities"])
            opacity_prior = (((torch.abs(all_opacities))*(torch.abs(1 - all_opacities)))).mean()
            ssimloss = 1.0 - fused_ssim(
                pred_for_loss[None, None].repeat(1,3,1,1), pixels[None, None].repeat(1,3,1,1), padding="valid"
            )

            ############################################################################## max size loss ##############################################################################
            
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda \
                                + cfg.max_size_prior_weight * max_size_loss \
                                + cfg.opacity_supervision_weight * opacity_supervision_loss
            
            
    
            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )

            loss.backward()



            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "


            if step > cfg.streak_end_step: #period C
                desc += "| sat and reg opt |"
            elif sat_state: #period B
                desc += "| sat opt |"
                self.splats["means"].grad *= 0.0
                self.splats["scales"].grad *= 0.0
                self.splats["quats"].grad *= 0.0
                self.splats["opacities"].grad *= 0.0
                
                self.splats["sh0"].grad *= 0.0
                self.splats["shN"].grad *= 0.0
            else: #period A
                desc += "| reg opt |"
                self.splats["sat_probability"].grad *= 0.0
                

            
            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                if cfg.wandb:
                    fig, ax = plt.subplots()
                    ax.hist(torch.sigmoid(self.splats["opacities"]).cpu().detach().numpy(), bins=100)
                    ax.set_title("Opacity Histogram")
                    ax.set_xlabel("Opacity")
                    ax.set_ylabel("Frequency")
                    ax.set_xlim(0, 1)
                    wandb.log({"train/opacity_histogram": wandb.Image(fig)})
                    plt.close(fig)

                    #histogram the sat_probability field 
                    fig, ax = plt.subplots()
                    ax.hist(torch.sigmoid(self.splats["sat_probability"]).cpu().detach().numpy(), bins=100)
                    ax.set_title("Sat Probability Histogram")
                    ax.set_xlabel("Sat Probability")
                    ax.set_ylabel("Frequency")
                    ax.set_xlim(0, 1)
                    wandb.log({"train/sat_probability_histogram": wandb.Image(fig)})
                    plt.close(fig)


                    #compute histogram for the colors 
                    fig, ax = plt.subplots()
                    ax.hist(info["colors"][...,0].squeeze().cpu().detach().numpy(), bins=100)
                    ax.set_title("Color Histogram")
                    ax.set_xlabel("Color")
                    ax.set_ylabel("Frequency")
                    ax.set_xlim(0, 1)
                    wandb.log({"train/color_histogram": wandb.Image(fig)})
                    plt.close(fig)

                    wandb.log({"train/loss": loss.item(), 
                               "train/l1loss": l1loss.item() * (1.0 - cfg.ssim_lambda), 
                               "train/pose_err": pose_err.item() if cfg.pose_opt and cfg.pose_noise else 0.0,
                               "train/opacity_supervision_loss": cfg.opacity_supervision_weight * opacity_supervision_loss.item(),
                               "train/sat_sparsity_prior": sat_sparsity_prior.mean() * cfg.sat_sparsity_prior_weight,
                               "train/ssimloss": ssimloss.item() * cfg.ssim_lambda,
                               "train/max_size_loss": max_size_loss.item() * cfg.max_size_prior_weight,
                               "train/num_GS": len(self.splats["means"]), 
                               "train/average_max_scale": torch.exp(self.splats['scales']).max(dim=1).values.mean(), 
                               "train/average_opacity": torch.sigmoid(self.splats["opacities"]).mean(), 
                               "train/average_dist": torch.norm(self.splats["means"][:,:2],dim=1).mean(), 
                               "train/mem": mem})
                    
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                # self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/average_max_scale", torch.exp(self.splats['scales']).max(dim=1).values.mean(), step)
                self.writer.add_scalar("train/average_opacity", torch.sigmoid(self.splats["opacities"]).mean(), step)
                self.writer.add_scalar("train/average_dist", torch.norm(self.splats["means"][:,:2],dim=1).mean(), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, out_img.squeeze()], dim=0).detach().cpu().numpy()
                    canvas = np.expand_dims(canvas, axis=0)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}

                save_ply(self.splats, f"{self.render_dir}/output_step{step:04d}.ply")

                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            #check if gradients are not none: 
            if step >= cfg.elevate_start_step and step <= cfg.elevate_end_step and \
                step % cfg.elevation_sampling_every < int(cfg.elevation_sampling_every*cfg.elevate_sampling_duty_cycle): 
                if pixels_for_loss.sum() > 0:
                    #update pbar with text 
                    desc += "Elevation Sampling"
                    opacity_gt = (pixels > cfg.opacity_supervision_thresh).float()
                    opacity_supervision_loss = F.l1_loss(info["opacity_returns"].squeeze()*opacity_gt, opacity_gt, reduction='none') 
                    high_loss_region = opacity_supervision_loss 
                    selection_mask = high_loss_region.bool()
                    #select random nonzero points 
                    elevate_loss_select = min(cfg.elevate_loss_select, selection_mask.sum().item())
                    nonzero_idxs = torch.where(selection_mask.float() > 0.0)
                    if elevate_loss_select == cfg.elevate_loss_select:
                        weights = high_loss_region[selection_mask].flatten().detach().cpu().numpy()/high_loss_region[selection_mask].flatten().detach().cpu().numpy().sum()
                        random_idx = torch.tensor(np.random.choice(range(len(nonzero_idxs[0])), 
                                                                   size=elevate_loss_select, 
                                                                   replace=False))
                    else: 
                        random_idx = torch.arange(elevate_loss_select)
                    rand_loss_idxs = nonzero_idxs[0][random_idx], nonzero_idxs[1][random_idx]
                    #create a enw image and set the values at the top loss indices to 1
                    top_loss_mask = torch.zeros_like(selection_mask)    
                    top_loss_mask[rand_loss_idxs[0], rand_loss_idxs[1]] = 1.0

                    total_new_gaussians = int(elevate_loss_select*cfg.elevate_num_samples)
                    #initialize the gaussians in the top loss regions 
                    initial_points = sample_points_in_elevation(return_threshold=0.0, 
                                                                        c2w=camtoworlds.squeeze().cpu().detach().numpy(), 
                                                                        image=top_loss_mask.cpu().numpy().squeeze(), 
                                                                        num_range_bins=width, 
                                                                        num_azimuth_bins=height, 
                                                                        max_range=self.parser.max_range, 
                                                                        hfov=self.parser.hfov_deg, vfov=self.parser.vfov_deg, 
                                                                        num_samples=cfg.elevate_num_samples)
                    
                    viz_means = torch.tensor(self.splats["means"])
                    sampled_idxs = torch.randint(0, len(self.splats["means"]), (total_new_gaussians,)).to(device)
                    new_opacities = torch.logit(torch.full((total_new_gaussians,), cfg.init_opa)).float().to(device)
                    new_scales = torch.log(torch.full((total_new_gaussians, 3), cfg.init_scale)).float().to(device)
                    new_sat_probability = torch.full((total_new_gaussians,), -10).float().to(device)
                    new_quats = torch.rand((total_new_gaussians, 4)).float().to(device)
                    new_means = torch.tensor(initial_points).float().to(device)

                    assert len(sampled_idxs) == total_new_gaussians

                    def param_fn(name: str, p: Tensor) -> Tensor:
                        if name == "opacities":
                            new_p = new_opacities
                        elif name == "scales":
                            new_p = new_scales
                        elif name == "sat_probability":
                            new_p = new_sat_probability
                        elif name == "means":
                            new_p = new_means
                        elif name == "quats":
                            new_p = new_quats
                        else: 
                            new_p = p[sampled_idxs] #keep other stuff the same 
                        p_new = torch.cat([p, new_p])
                        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

                    def optimizer_fn(key: str, v: Tensor) -> Tensor:
                        v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
                        return torch.cat([v, v_new])

                    # update the parameters and the state in the optimizers
                    _update_param_with_optimizer(param_fn, optimizer_fn, self.splats, self.optimizers)

            else:
                desc += "No Elevation Sampling, normal strategy"

            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            elif isinstance(self.cfg.strategy, PruneOnlyStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        packed=cfg.packed,
                    )
            else:
                assert_never(self.cfg.strategy)

            
           

            pbar.set_description(desc)
            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                # self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        os.makedirs(f"{cfg.result_dir}/test", exist_ok=True)
        os.makedirs(f"{cfg.result_dir}/test/gt_sonar_images", exist_ok=True)
        os.makedirs(f"{cfg.result_dir}/test/sonar_images", exist_ok=True)
        os.makedirs(f"{cfg.result_dir}/test/denoised_sonar_images", exist_ok=True)
        metrics = defaultdict(list)
        if cfg.render_eval:
            for i, data in enumerate(valloader):
                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)
                pixels = data["image"].to(device)
                masks = data["mask"].to(device) if "mask" in data else None
                near_plane = data["near_plane"].to(device)
                far_plane = data["far_plane"].to(device)
                height, width = pixels.shape[1:3]

                torch.cuda.synchronize()
                tic = time.time()
                colors, info, = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    masks=masks,
                )  # [1, H, W, 3]
                torch.cuda.synchronize()
                ellipse_time += time.time() - tic

                colors = torch.clamp(colors, 0.0, 1.0)
                canvas_list = [pixels,info["unsaturated_returns"],  colors]

                if world_rank == 0:
                    # write images
                    canvas = torch.cat(canvas_list, dim=1).squeeze(0).cpu().numpy().transpose(1,0,2)
                    canvas_pil = PIL.Image.fromarray((canvas.squeeze() * 255).astype(np.uint8)).convert("L") 
                    gt_pil = PIL.Image.fromarray(np.array((pixels.squeeze().cpu() * 255)).astype(np.uint8).transpose(1,0)).convert("L")
                    pred_pil = PIL.Image.fromarray(np.array((colors.squeeze().cpu() * 255)).astype(np.uint8).transpose(1,0) ).convert("L")
                    denoise_pil = PIL.Image.fromarray(np.array((info["unsaturated_returns"].squeeze().cpu() * 255)).astype(np.uint8).transpose(1,0) ).convert("L")
                    canvas_pil.save(f"{self.render_dir}/{stage}_step{step}_{i:04d}.png")
                    #log to wandb 
                    if cfg.wandb:
                        wandb.log({f"val/Image_{data['image_id'].item()}": wandb.Image(canvas)}, step=step)

                    pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    unsaturated_returns_p = info["unsaturated_returns"].permute(0, 3, 1, 2)  # [1, 3, H, W]

                    rows = np.random.randint(0, pixels_p[0].shape[1] - 10, 100)
                    cols = np.random.randint(0, pixels_p[0].shape[2] - 10, 100)
                   
                    metrics["icv_gt"].append(torch.tensor(compute_icv(image=pixels_p[0].cpu().numpy().squeeze(), rows=rows, cols=cols)))
                    metrics["icv_nvs"].append(torch.tensor(compute_icv(image=colors_p[0].cpu().numpy().squeeze(), rows=rows, cols=cols)))
                    metrics["icv_removed"].append(torch.tensor(compute_icv(image=unsaturated_returns_p[0].cpu().numpy().squeeze(), rows=rows, cols=cols)))

                    metrics["tv_gt"].append(torch.tensor(total_variation(pixels_p[0].cpu().numpy().squeeze())))
                    metrics["tv_nvs"].append(torch.tensor(total_variation(colors_p[0].cpu().numpy().squeeze())))
                    metrics["tv_removed"].append(torch.tensor(total_variation(unsaturated_returns_p[0].cpu().numpy().squeeze())))
                   
                    metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                    metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                    metrics["lpips"].append(self.lpips(colors_p.repeat(1,3,1,1), pixels_p.repeat(1,3,1,1)))

                    #save the gt and colors to two separate folders
                    gt_pil.save(f"{cfg.result_dir}/test/gt_sonar_images/{i:04d}.png")
                    pred_pil.save(f"{cfg.result_dir}/test/sonar_images/{i:04d}.png")
                    # denoise_pil.save(f"{cfg.result_dir}/test/denoised_sonar_images/{i:04d}.png")

        if world_rank == 0:
            ellipse_time /= len(valloader)

            #save the means as a ply ff
            means = self.splats["means"].cpu().numpy()

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.render_eval:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.4f}"
                    f"ICV GT: {stats['icv_gt']:.3f},  ICV NVS: {stats['icv_nvs']:.3f},  ICV REMOVED: {stats['icv_removed']:.3f}"
                    f"TV GT: {stats['tv_gt']:.3f},  TV NVS: {stats['tv_nvs']:.3f},  TV REMOVED: {stats['tv_removed']:.3f}"
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
            )
                #log to wandb 
                if cfg.wandb:
                    wandb.log({"val/psnr": stats["psnr"], 
                            "val/ssim": stats["ssim"], 
                            "val/lpips": stats["lpips"], 
                            "val/ellipse_time": stats["ellipse_time"], 
                            "val/num_GS": stats["num_GS"]}, step=step)
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        # """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds
        cam_before = camtoworlds_all.copy()
        if cfg.render_traj_path == "shift": 
            camtoworlds_all = generate_shifted_path(
                camtoworlds_all, 
                shift=np.array([0.5, 0, 0.0])
            )
            camtoworlds_all = camtoworlds_all[:,:-1,:]
        elif cfg.render_traj_path == "unchanged": 
            camtoworlds_all = camtoworlds_all[:,:-1,:]
            pass
        elif cfg.render_traj_path == "sine_straight":
            camtoworlds_all = sine_path_along_straight(
                camtoworlds_all, 
                num_interp=cfg.render_traj_interp_val,
                amplitude=cfg.render_traj_amplitude, 
                freq=cfg.render_traj_freq
            )
        elif cfg.render_traj_path == "straight":
            camtoworlds_all = straight_path(
                camtoworlds_all, 
                num_poses=cfg.render_traj_interp_val
            )
        elif cfg.render_traj_path == "random_change_prob":
            random_change_prob(
               self.splats["sat_probability"], 
               num_change=cfg.render_traj_interp_val
            )
            camtoworlds_all = camtoworlds_all[:, :-1, :]
        elif cfg.render_traj_path == "interp":
            # camtoworlds_all = generate_interpolated_path(
            #     camtoworlds_all, cfg.render_traj_interp_val
            # )  # [N, 3, 4]
            camtoworlds_all = camtoworlds_all[:,:-1,:]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        #visualize traj 
        if not cfg.train:
            visualize_gaussians(xyz=[], 
                                poses=np.concatenate([cam_before, camtoworlds_all.squeeze()], axis=0), 
                                colors=[np.array([0,1,0]), np.array([1, 0,0])],
                                start_size=0.5,
                                end_size=0.5)
        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        height, width = list(self.parser.imsize_dict.values())[0]
        near_plane = -np.radians(self.parser.vfov_deg/2)
        far_plane = np.radians(self.parser.vfov_deg/2)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(f"{video_dir}/test", exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=near_plane,
                far_plane=far_plane,
                # render_mode="RGB+ED",
            )  # [1, H, W, 4]
            unsaturated_returns = info["unsaturated_returns"]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            # depths = renders[..., 3:4]  # [1, H, W, 1]
            # depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors,  unsaturated_returns]

            #save colors in a folder 
            # write images
            canvas = torch.cat(canvas_list, dim=1).squeeze(0).cpu().numpy().transpose(1,0, 2)
            canvas = (canvas * 255).astype(np.uint8)

            denoise_pil = PIL.Image.fromarray(np.array((unsaturated_returns.squeeze().cpu() * 255)).astype(np.uint8).transpose(1,0) ).convert("L")

            denoise_pil.save(f"{cfg.result_dir}/test/denoised_sonar_images/{i:04d}.png")

            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.wandb:
        wandb.init(project="uwgs_training", entity="advaiths", config=cfg, group=cfg.data_dir.split("/")[-1])

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
        if cfg.train: 
            runner.train()
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
        "prune_only": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                strategy=PruneOnlyStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    cli(main, cfg, verbose=True)
