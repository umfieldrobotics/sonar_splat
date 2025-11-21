import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os
from plyfile import PlyData
from tqdm import tqdm
import numpy as np
from scipy.stats import multivariate_normal
import torch
from pathlib import Path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.system_utils import searchForMaxIteration
from utils.general_utils import strip_symmetric, build_scaling_rotation, remake_symmetric
from scene import GaussianModel
from os import makedirs
# import mcubes
# from multivariate_normal import CustomMultivariateNormal
from utils.sh_utils import SH2RGB

import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use("TkAgg")

CHUNK_SIZE = 600

def _batch_mahalanobis(bL, bx):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for (sL, sx) in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (list(range(outer_batch_dims)) +
                    list(range(outer_batch_dims, new_batch_dims, 2)) +
                    list(range(outer_batch_dims + 1, new_batch_dims, 2)) +
                    [new_batch_dims])
    bx = bx.permute(permute_dims)

    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = torch.linalg.solve_triangular(flat_L, flat_x_swap, upper=False).pow(2).sum(-2)  # shape = b x c
    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)

class CustomMultivariateNormal(torch.distributions.MultivariateNormal):
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None):
        # Initialize the base class
        super().__init__(loc, covariance_matrix, precision_matrix, scale_tril)
    
    def log_prob_unnorm(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
        return -0.5 * (M)


def query_gaussians(query_xyz, mvn, alpha, color, device="cuda"):
    probs = []
    for query_xyz_ in tqdm(torch.split(query_xyz.to(torch.float), CHUNK_SIZE, dim=0), "Processing Gaussians"):
        individual_probs_ = torch.exp(mvn.log_prob_unnorm(query_xyz_.view(query_xyz_.shape[0], 1, -1)))
        individual_probs_ = individual_probs_ * alpha.view(1,-1) * color.view(1,-1)
        probs_ = torch.sum(individual_probs_, axis=1)
        probs_ = torch.clamp(probs_, 0., 1.)
        probs.append(probs_)
        
    probs = torch.cat(probs, dim=0)

    return probs


def create_voxelized_gaussians(gaussians: GaussianModel, voxel_size=0.08, threshold=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating Voxels")

    #Get Gaussian Data
    means = gaussians.get_xyz.detach()
    covariance_matrices = gaussians.get_covariance()
    covariance_matrices = remake_symmetric(covariance_matrices.detach()) + torch.eye(3).view(1,3,3).cuda()*1e-5
    opacities = gaussians.get_opacity[:, 0].detach()
    color = gaussians.get_features.detach()

    
    # Check for nan
    nan_mask = torch.isnan(covariance_matrices)

    invalid_mask = nan_mask.view(nan_mask.size(0), -1).any(dim=1)

    # Filter out matrices with NaN values
    valid_covariance_matrices = covariance_matrices[~invalid_mask]
    valid_means = means[~invalid_mask]
    valid_opacities = opacities[~invalid_mask]
    valid_color = torch.clamp(SH2RGB(color[~invalid_mask]), 0, 1)



    mvn = CustomMultivariateNormal(loc=valid_means, covariance_matrix=valid_covariance_matrices)

    # Sample points from the Gaussians (100 samples for each Gaussian)
    num_samples_per_gaussian = 10
    samples = mvn.sample((num_samples_per_gaussian,))
    sample_pts = samples.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample_pts.cpu().numpy())
    o3d.visualization.draw_geometries([pcd])

    x_min, x_max = torch.floor(torch.min(sample_pts[:, 0])), torch.ceil(torch.max(sample_pts[:, 0]))
    y_min, y_max = torch.floor(torch.min(sample_pts[:, 1])), torch.ceil(torch.max(sample_pts[:, 1]))
    z_min, z_max = torch.floor(torch.min(sample_pts[:, 2])), torch.ceil(torch.max(sample_pts[:, 2]))

    # Create the 3D grid of coordinates
    # Create the coordinate grid using linspace
    x_coords = torch.linspace(x_min, x_max, int((x_max - x_min) / voxel_size) + 1).to(device)
    y_coords = torch.linspace(y_min, y_max, int((y_max - y_min) / voxel_size) + 1).to(device)
    z_coords = torch.linspace(z_min, z_max, int((z_max - z_min) / voxel_size) + 1).to(device)

    # Generate a 3D grid using meshgrid
    grid_coords = torch.cartesian_prod(x_coords, y_coords, z_coords).to(device)

    #Process Gaussians using MVN to get occupancy
    voxel_densities = query_gaussians(grid_coords, mvn, valid_opacities, valid_color[:, 0, 0], "cuda")

    # Convert to voxel grid indices
    voxel_indices = torch.round(grid_coords / voxel_size).to(torch.int32)

    # High density mask
    high_density_mask = voxel_densities > threshold

    # Select voxel points and their densities
    voxel_points = voxel_indices[high_density_mask] * voxel_size

    # Assign colors based on voxel density (for visualization)
    voxel_colors = torch.stack([
        voxel_densities[high_density_mask],
        torch.zeros_like(voxel_densities[high_density_mask]),
        1 - voxel_densities[high_density_mask]
    ], dim=-1)

    # Convert to NumPy for Open3D compatibility (requires CPU)
    voxel_points_np = voxel_points.cpu().numpy()
    voxel_colors_np = voxel_colors.cpu().numpy()

    # Create PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_points_np)
    pcd.colors = o3d.utility.Vector3dVector(voxel_colors_np)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    # Visualize the voxel grid
    o3d.visualization.draw_geometries([voxel_grid])

    return pcd

def create_pc_gaussians(gaussians: GaussianModel, num_samples=100, threshold=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating PointCloud")


    #Get Gaussian Data
    means = gaussians.get_xyz.detach()
    covariance_matrices = gaussians.get_covariance()
    covariance_matrices = remake_symmetric(covariance_matrices.detach()) + torch.eye(3).view(1,3,3).cuda()*1e-5
    opacities = gaussians.get_opacity[:, 0].detach()
    # print(len(means))
    color = gaussians.get_features.detach()

    # Check for nan
    nan_mask = torch.isnan(covariance_matrices)

    invalid_mask = nan_mask.view(nan_mask.size(0), -1).any(dim=1)

    print("total: ", len(invalid_mask))
    print("valid: ", len(invalid_mask) - invalid_mask.sum())
    print("invalid: ", invalid_mask.sum())
    # invalid_mask = torch.zeros_like(nan_mask[:, 0])

    # Filter out matrices with NaN values
    valid_covariance_matrices = covariance_matrices[~invalid_mask]
    valid_means = means[~invalid_mask]
    valid_opacities = opacities[~invalid_mask]
    valid_color = SH2RGB(color[~invalid_mask])

    print(valid_color[0, :, 0])
    plt.plot(valid_color[0,:,0].cpu().numpy())
    plt.title('Histogram of Valid Opacities')
    plt.xlabel('Opacity')
    plt.ylabel('Frequency')

    # Display the histograms
    plt.show()

    valid_color = (valid_color[:, 0, 0] - torch.min(valid_color[:, 0, 0])) / (torch.max(valid_color[:, 0, 0]) - torch.min(valid_color[:, 0, 0]))
    print(len(valid_means))
    print(valid_color.shape)
    print(valid_color.max())
    print(valid_color.min())

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(valid_opacities.cpu().numpy(), bins=30, color='blue', alpha=0.7)
    plt.title('Histogram of Valid Opacities')
    plt.xlabel('Opacity')
    plt.ylabel('Frequency')

    # Create histogram for valid_colors
    plt.subplot(1, 2, 2)
    plt.hist(valid_color.cpu().numpy(), bins=30, color='green', alpha=0.7)
    plt.title('Histogram of Valid Colors')
    plt.xlabel('Color')
    plt.ylabel('Frequency')

    # Display the histograms
    plt.tight_layout()
    plt.show()


    opacity_reflectance_mask = (valid_opacities > 0.2)
    valid_covariance_matrices = valid_covariance_matrices[opacity_reflectance_mask]
    valid_means = valid_means[opacity_reflectance_mask]
    valid_opacities = valid_opacities[opacity_reflectance_mask]
    valid_color = valid_color[opacity_reflectance_mask]

    print(len(valid_means))
    print(len(valid_color))
    print(valid_color.max())
    print(valid_color.min())


    mvn = CustomMultivariateNormal(loc=valid_means, covariance_matrix=valid_covariance_matrices)

    # Sample points from the Gaussians
    samples = mvn.sample((num_samples,))
    sample_pts = samples.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample_pts.cpu().numpy())
    o3d.visualization.draw_geometries([pcd])

    #Evaluate Gaussians for each sample
    pc_densities = query_gaussians(sample_pts, mvn, valid_opacities, valid_color, "cuda")

    # High density mask
    high_density_mask = pc_densities > threshold

    # Select voxel points and their densities
    filtered_pts = sample_pts[high_density_mask].cpu().numpy()

    # Create PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_pts)
    o3d.visualization.draw_geometries([pcd])

    return pcd

def marching_cubes(pcd, output_dir, iter, method, pitch_scale=1, filtering_iters=5):

    print("Marching Cubes..")
    pc = trimesh.PointCloud(pcd.points)

    pitch = (pc.extents.max() / 150) * pitch_scale
    # pitch = voxel_size*pitch_scale
    mcubes = trimesh.voxel.ops.points_to_marching_cubes(pc.vertices, pitch=pitch)

    mcubes = trimesh.smoothing.filter_laplacian(mcubes, iterations=filtering_iters)
    mcubes = trimesh.smoothing.filter_taubin(mcubes, iterations=filtering_iters)

    file_name = method + '_mesh_iteration.ply'

    output_path = os.path.join(output_dir, file_name)
    mcubes.export(output_path)

def extract_geometry(bound_min, bound_max, resolution, threshold, mvn, opacities):
    u = extract_fields(bound_min, bound_max, resolution, mvn, opacities)
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]

    return vertices, triangles

def extract_fields(bound_min, bound_max, resolution, mvn, opacities):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in tqdm(enumerate(X), total=len(X), desc="Meshing w/ Neusis Method"):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device="cuda")
                    val = query_gaussians(pts, mvn, opacities).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val



    return u

def neusis_meshing(gaussians: GaussianModel, res=64, threshold=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating Voxels")

    #Get Gaussian Data
    means = gaussians.get_xyz.detach()
    covariance_matrices = gaussians.get_covariance()
    covariance_matrices = remake_symmetric(covariance_matrices.detach()) + torch.eye(3).view(1,3,3).cuda()*1e-5
    opacities = gaussians.get_opacity[:, 0].detach()

    # Check for nan
    nan_mask = torch.isnan(covariance_matrices)

    invalid_mask = nan_mask.any(dim=(1, 2))

    # Filter out matrices with NaN values
    valid_covariance_matrices = covariance_matrices[~invalid_mask]
    valid_means = means[~invalid_mask]
    valid_opacities = opacities[~invalid_mask]

    mvn = CustomMultivariateNormal(loc=valid_means, covariance_matrix=valid_covariance_matrices)

    # Sample points from the Gaussians (100 samples for each Gaussian)
    num_samples_per_gaussian = 10
    samples = mvn.sample((num_samples_per_gaussian,))
    sample_pts = samples.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample_pts.cpu().numpy())
    o3d.visualization.draw_geometries([pcd])

    x_min, x_max = torch.floor(torch.min(sample_pts[:, 0])), torch.ceil(torch.max(sample_pts[:, 0]))
    y_min, y_max = torch.floor(torch.min(sample_pts[:, 1])), torch.ceil(torch.max(sample_pts[:, 1]))
    z_min, z_max = torch.floor(torch.min(sample_pts[:, 2])), torch.ceil(torch.max(sample_pts[:, 2]))

    bounds_min = torch.tensor([int(x_min.item()), int(y_min.item()), int(z_min.item())])
    bounds_max = torch.tensor([int(x_max.item()), int(y_max.item()), int(z_max.item())])

    # Export mesh
    vertices, triangles = extract_geometry(bounds_min, bounds_max, res, threshold, mvn, valid_opacities)

    mesh = trimesh.Trimesh(vertices, triangles)

    return mesh

def main():
    # Path to the input PLY file

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--res", default=64, type=int)
    parser.add_argument("--voxel_size", default=0.0, type=float)
    parser.add_argument("--num_samples", default=0, type=int)
    parser.add_argument("--ply_path", default="data/point_cloud/point_cloud.ply")
    parser.add_argument("--threshold", default=0.9, type=float)
    parser.add_argument("--pitch_scale", default=0.5, type=float)
    args = get_combined_args(parser)
    # print("Meshing" + args.model_path)

    # if args.iteration == -1:
    #     iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    
    ply_file_path = args.ply_path
    
    # Directory to save the generated meshes
    output_dir = os.path.join(Path(ply_file_path).parent, 'meshes')
    makedirs(output_dir, exist_ok=True)

    gaussians = GaussianModel(3)
    gaussians.load_ply(ply_file_path)

    # if args.res != 0:
    #     mesh = neusis_meshing(gaussians, res=args.res, threshold=args.threshold)

    #     file_name = 'neusis' + '_mesh_iteration_{}.ply'.format(iter)

    #     output_path = os.path.join(output_dir, file_name)
    #     mesh.export(output_path)

    #     open3d_mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, 'neusis_mesh_iteration_{}.ply'.format(iter)))  # Replace with your mesh file path
    #     open3d_mesh.compute_vertex_normals()
    #     o3d.visualization.draw_geometries([open3d_mesh]) 

    # if args.voxel_size != 0:
    #     pcd = create_voxelized_gaussians(gaussians, voxel_size=args.voxel_size, threshold=args.threshold)
    #     marching_cubes(pcd, output_dir, iter, "voxelization", 1)
    #     mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, 'voxelization_mesh_iteration_{}.ply'.format(iter)))  # Replace with your mesh file path

    #     mesh.compute_vertex_normals()
    #     o3d.visualization.draw_geometries([mesh]) 

    if args.num_samples != 0:
        pcd = create_pc_gaussians(gaussians, num_samples=args.num_samples, threshold=args.threshold)

        #save this pcd 
        o3d.io.write_point_cloud(os.path.join(output_dir, 'pcd.ply'), pcd)
        marching_cubes(pcd, output_dir, iter, "sampling", filtering_iters=2, pitch_scale=args.pitch_scale)

        mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, 'sampling_mesh_iteration.ply'.format(Path(ply_file_path).stem)))  # Replace with your mesh file path

        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh]) 

    print('Gaussian splats processed and saved as meshes.')
    
if __name__ == '__main__':
    main()