import torch 
import numpy as np 
import math 
import open3d as o3d
import json 
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates
from tqdm import tqdm 
from matplotlib import pyplot as plt 
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle

sonar_axis_conversion = np.array([ 
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
]).astype(np.float32)


    
def sample_pcd_on_gaussians(means: np.ndarray, 
                            cov3D: np.ndarray, 
                            num_pts: int) -> np.ndarray: 
    pts = []
    for mean, cov in zip(means, cov3D):
        pts.append(np.random.multivariate_normal(mean, cov, num_pts))
    return np.concatenate(pts, axis=0)

def plot_gaussians_azimuth_range(means: np.ndarray, 
                                 covariances: np.ndarray,
                                 max_range: float, 
                                    hfov: float, 
                                 ax=None, show=True, scale=2.0):
    """
    Plots 2D Gaussians in azimuth-range space using Matplotlib.
    
    Args:
    - means (np.ndarray): (N, 2) array of (azimuth, range) means.
    - covariances (np.ndarray): (N, 2, 2) array of covariance matrices.
    - ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new one if None.
    - show (bool): Whether to display the figure (default: True).
    - scale (float): Scaling factor for ellipse size (default: 2.0 standard deviations).

    Returns:
    - matplotlib.axes.Axes: The axis with plotted Gaussians.
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # ax.set_aspect('equal')
    
    for mean, cov in zip(means, covariances):
        azimuth, rng = mean
        cov_azimuth_range = cov[:2, :2]  # Extract 2x2 covariance matrix
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_azimuth_range)
        
        # Get major and minor axis (sqrt of eigenvalues gives standard deviations)
        width, height = scale * 3 * np.sqrt(eigenvalues)  # 2*stddev for 95% confidence
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))  # Rotation angle in degrees

        # Plot the Gaussian ellipse
        ellipse = Ellipse((azimuth, rng), width, height, angle=angle, edgecolor='b', facecolor='none', linewidth=2)
        ax.add_patch(ellipse)
        
        # Mark the mean
        ax.scatter(azimuth, rng, color='r', marker='x', s=100)
        
        #draw a rectangle of the valid space 
        ax.add_patch(plt.Rectangle((-np.radians(hfov/2), 0), np.radians(hfov), max_range, fill=False, edgecolor='g', linewidth=2))

    ax.set_xlabel("Azimuth (radians)")
    ax.set_ylabel("Range (meters)")
    ax.set_title("2D Gaussians in Azimuth-Range Space")

    if show:
        plt.show()
    
    return ax

def plot_circles(centers: np.ndarray, 
                 img_size: tuple,
                 extents:tuple,
                 radii: np.ndarray, 
                 ax=None, show=True, 
                 ):
    """
    Plots circles on a given Matplotlib axis.
    
    Args:
    - centers (np.ndarray): (N, 2) array of (x, y) centers.
    - radii (np.ndarray): (N,) array of radii.
    - ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new one if None.
    - show (bool): Whether to display the figure (default: True).

    Returns:
    - matplotlib.axes.Axes: The axis with plotted circles.
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.set_aspect('equal')  # Equal aspect ratio for correct circle display
    
    # Plot each circle
    for (cx, cy), r in zip(centers, radii):
        circle = Circle((cx, cy), r, fill=False, edgecolor='b', linewidth=2)
        ax.add_patch(circle)
    
    # Set axis limits to include all circles
    buffer = np.max(radii) + 10  # Add padding around the circles
    ax.set_xlim(np.min(centers[:, 0]) - buffer, np.max(centers[:, 0]) + buffer)
    ax.set_ylim(np.min(centers[:, 1]) - buffer, np.max(centers[:, 1]) + buffer)
    ax.scatter(centers[:, 0], centers[:, 1], c='r', marker='x', s=100)

    #draw a rectangle of the image size
    ax.add_patch(plt.Rectangle((img_size[0], img_size[1]), extents[0], extents[1], fill=False, edgecolor='g', linewidth=2))

    #plot this in polar space
    if show:
        plt.show()
    
    return ax

def np4x4_from_Rt(R: np.ndarray,
                   t: np.ndarray, 
                   quat: bool = False, 
                   invert_R: bool = False) -> np.ndarray:
    """
    Create a 4x4 transformation matrix from a 3x3 rotation matrix and a 3x1 translation vector.
    
    Parameters:
    - R (np.ndarray): 3x3 rotation matrix.
    - t (np.ndarray): 3x1 translation vector.
    
    Returns:
    - np.ndarray: 4x4 transformation matrix.
    """
    T = np.eye(4)
    if quat: 
        R_new = qvec2rotmat(R)
    else:
        R_new = R
    if invert_R:
        T[:3,:3] = R_new.T
    else: 
        T[:3, :3] = R_new
    T[:3, 3] = t
    return T

def visualize_frames(poses, start_size=0.1, end_size=0.2):
    #convert the poses to open3d frames
    frames = []
    sizes = np.linspace(start_size, end_size, len(poses))
    for i, pose in enumerate(poses):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=sizes[i])
        if type(pose) == torch.Tensor: 
            pose = pose.cpu().numpy()
        frame.transform(pose)
        frames.append(frame)

    #visualize the frames
    return frames

def parse_sonar_parameters(json_path):
    """
    Parses a JSON file for sonar parameters and returns a dictionary of parameters.
    
    Parameters:
    - json_path (str): Path to the JSON file containing sonar parameters.
    
    Returns:
    - dict: Dictionary with sonar parameters.
    """
    try:
        with open(json_path, 'r') as file:
            params = json.load(file)
            
        # Extract sonar parameters
        sonar_params = {
            "hfov": params.get("hfov", 120),  # default HFOV to 120 if not provided
            "vfov": params.get("vfov", 30),   # default VFOV to 30 if not provided
            "num_azimuth_bins": params.get("num_azimuth_bins", 512),
            "num_range_bins": params.get("num_range_bins", 512),
            "max_range": params.get("max_range", 10),
            "samples_per_bin": params.get("samples_per_bin", 4),
            "cam_to_sonar_R": np.array(params.get("cam_to_sonar_R", [1, 0, 0, 0])),
            "cam_to_sonar_t": np.array(params.get("cam_to_sonar_t", [0, 0, 0]))

        }
        
        return sonar_params
    
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return None
    

def gaussians_to_ellipsoids(means: torch.Tensor,
                            scalings: torch.Tensor,
                            rotations: torch.Tensor,
                            colors: torch.Tensor,
                            opacities: torch.Tensor, 
                            out_ply="./ellipsoid_mesh.ply",
                            bounds = None,scale_factor=1,min_opacity=None,
                            skip_step=1):
    

    import trimesh
    def create_ellipsoid_mesh(mean, scaling, rotation_quat, color):

        # Create a unit sphere mesh
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=1)
        
        # Scale the sphere to create an ellipsoid based on the eigenvalues
        scale_factors = scaling * scale_factor
        sphere.apply_scale(scale_factors)
        
        # Rotate the ellipsoid to align with the eigenvectors
        # trimesh expects rotation as a 4x4 matrix, where the upper left 3x3 is the rotation matrix
        T = np.eye(4)
        rotation_matrix = R.from_quat(rotation_quat).as_matrix()
        T[:3, :3] = rotation_matrix
        sphere.apply_transform(T)
        sphere.visual.vertex_colors = [0, 255, 0, 255]

        
        # Translate the ellipsoid to its mean position
        sphere.apply_translation(mean)
        
        return sphere

    meshes = []
    means = means[::skip_step]
    scalings = scalings[::skip_step]
    rotations = rotations[::skip_step]
    opacities = opacities[::skip_step]
    colors = colors[::skip_step]
    for i, (mean, scaling, rotation, color, opa) in tqdm(enumerate(zip(means, scalings, rotations, colors, opacities)), total=means.shape[0]):
        if bounds is not None and \
                (mean[0] < bounds[0][0] or mean[0] > means[0][1] \
              or mean[1] < bounds[1][0] or mean[1] > means[1][1] \
              or mean[2] < bounds[2][0] or mean[2] > means[2][1]):
            
            continue
                
        if min_opacity is not None and opa <  min_opacity:
            continue
        try:
            color = color.clip(0, 1)
            meshes.append(create_ellipsoid_mesh(mean, scaling, rotation, color))
        except Exception as e:
            print(f"Couldn't create mesh, probably this is due to the color: {e}", color)
    
    #convert the mesh to o3d mesh 
    o3d_meshes = []
    for mesh in meshes:
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

        o3d_meshes.append(o3d_mesh)
    
    return o3d_meshes

def visualize_img_and_gt(img, gt):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img.cpu().detach().squeeze(), cmap='gray', vmin=0, vmax=1)
    ax[0].axis('off')
    ax[0].set_title("Rendered")
    ax[1].imshow(gt.cpu().detach().squeeze(), cmap='gray', vmin=0, vmax=1)
    ax[1].axis('off')
    ax[1].set_title("Ground Truth")
    plt.show()
    
def visualize_val_train_poses(train_dset, val_dset, pcd):
    #visualize the poses of the train and val sets 
    #go through train_dsets 
    train_poses = []
    for i in range(len(train_dset)):
        train_poses.append(train_dset[i]["camtoworld"].detach().cpu().numpy())
    #go through val_dset 
    val_poses = []
    for i in range(len(val_dset)):
        val_poses.append(val_dset[i]["camtoworld"].detach().cpu().numpy())

    train_poses = np.array(train_poses)
    val_poses = np.array(val_poses)
    #plot in plt 
    plt.figure()
    plt.scatter(train_poses[:, 0, -1], train_poses[:, 1, -1], c='r', marker='x')
    plt.scatter(val_poses[:, 0, -1], val_poses[:, 1, -1], c='b', marker='o')
    plt.scatter(pcd[:, 0], pcd[:, 1], c='g', marker='.')
    plt.axis('equal')
    plt.title("Train Poses")
    plt.show()
    plt.figure()

def visualize_gaussians(xyz: torch.tensor, poses: list, colors=None, start_size=0.5, end_size=0.5, others=None) -> None:
    pcds = []
    for pi, points in enumerate(xyz): 
        pcd = o3d.geometry.PointCloud()
        if type(points) == torch.Tensor:
            pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
        else:
            pcd.points = o3d.utility.Vector3dVector(points)
        #set colors 
        # if colors: 
        #     pcd.paint_uniform_color(colors[pi])
        # else:
        #     if len(colors) == 1:
        #         pcd.paint_uniform_color([1, 0, 0])
        #     except:
        if type(colors) == list:
            #paint uniform 
            pcd.paint_uniform_color(colors[pi])
        elif type(colors) == np.ndarray:
            pcd.colors = o3d.utility.Vector3dVector(colors[pi])
        pcds.append(pcd)
    frames = visualize_frames(poses, start_size=start_size, end_size=end_size)
    if others: 
        frames += others
    o3d.visualization.draw_geometries(pcds + frames)

def calculate_intrinsic_matrix(hfov: float, vfov: float, H: int, W: int) -> np.ndarray:
    """
    Calculate the intrinsic matrix K based on the given field of view and resolution parameters.
    
    Parameters:
    - hfov (float): Horizontal field of view in degrees.
    - vfov (float): Vertical field of view in degrees.
    - num_azimuth_bins (int): Number of bins along the azimuth (horizontal).
    - num_range_bins (int): Number of bins along the range (vertical).
    
    Returns:
    - np.ndarray: The 3x3 intrinsic matrix K.
    """
    # Compute new dimensions based on bin counts
    newW = W
    newH = H

    # Convert FOVs to radians
    hfov_rad = np.radians(hfov)
    vfov_rad = np.radians(vfov)

    # Calculate focal lengths
    fx = newW / (2 * np.tan(hfov_rad / 2))
    fy = newH / (2 * np.tan(vfov_rad / 2))

    # Calculate principal point (image center)
    cx = newW / 2
    cy = newH / 2

    # Construct the intrinsic matrix
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return K

def depth_to_metric_optimized(depth_image: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert depth image values to 3D XYZ points in the camera frame using vectorization.

    Args:
        depth_image (np.ndarray): Input depth image (2D array).
        K (np.ndarray): Camera intrinsic matrix (3x3).
    
    Returns:
        np.ndarray: 3D points (X, Y, Z) corresponding to the depth image.
    """
    height, width = depth_image.shape

    # Intrinsic matrix components
    f_x, f_y = K[0, 0], K[1, 1]
    c_x, c_y = K[0, 2], K[1, 2]

    # Create a meshgrid of pixel coordinates (u, v)
    u, v = torch.meshgrid(torch.arange(height), torch.arange(width))
    u = u.to(depth_image.device)
    v = v.to(depth_image.device)

    # Convert depth values to float and compute metric 3D coordinates
    Z = depth_image
    Y = (u - c_y) * Z / f_y
    X = (v - c_x) * Z / f_x

    # Stack X, Y, Z to form Nx3 points cloud 
    points_3d = torch.stack([X, Y, Z], dim=-1)
    points_3d = points_3d.reshape(-1, 3)

    return points_3d

def create_frustum_wireframe(vertical_fov: float, horizontal_fov: float, max_range: float):
    """
    Creates a frustum wireframe in Open3D based on given FOVs and range
    with X as the viewing direction, Y to the right, and Z down.

    Args:
        vertical_fov (float): Vertical field of view in degrees.
        horizontal_fov (float): Horizontal field of view in degrees.
        max_range (float): Maximum range of the frustum.

    Returns:
        o3d.geometry.LineSet: Wireframe representing the frustum.
    """
    # Convert FOVs from degrees to radians
    vertical_fov_rad = math.radians(vertical_fov)
    horizontal_fov_rad = math.radians(horizontal_fov)

    # Calculate half angles
    v_half_angle = vertical_fov_rad / 2
    h_half_angle = horizontal_fov_rad / 2

    # Define frustum corner points based on FOV and max range
    # X is the viewing direction, Y is right, Z is down
    top_right = [max_range, max_range * math.tan(h_half_angle), -max_range * math.tan(v_half_angle)]
    top_left = [max_range, -max_range * math.tan(h_half_angle), -max_range * math.tan(v_half_angle)]
    bottom_right = [max_range, max_range * math.tan(h_half_angle), max_range * math.tan(v_half_angle)]
    bottom_left = [max_range, -max_range * math.tan(h_half_angle), max_range * math.tan(v_half_angle)]
    origin = [0, 0, 0]

    # Define points for the frustum (origin + 4 corners)
    points = [
        origin,        # 0: Origin
        top_right,     # 1: Top right corner
        top_left,      # 2: Top left corner
        bottom_right,  # 3: Bottom right corner
        bottom_left,   # 4: Bottom left corner
    ]
    points = np.array(points)

    # Define lines between the points
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from the origin to corners
        [1, 2], [2, 4], [4, 3], [3, 1],  # Lines between corners
    ]

    # Create LineSet and add points and lines
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)

    # Optionally, set colors for each line
    colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for all lines
    frustum.colors = o3d.utility.Vector3dVector(colors)

    return frustum

def differentiable_2d_histogram(
    x: torch.Tensor,
    y: torch.Tensor,
    xbins: int,
    ybins: int,
    x_range: tuple,
    y_range: tuple,
    sigma: float = 1.0
) -> torch.Tensor:
    """
    Computes a differentiable 2D histogram with independent bins for x and y.

    Args:
        x (torch.Tensor): Tensor of x-coordinates, shape (N,).
        y (torch.Tensor): Tensor of y-coordinates, shape (N,).
        xbins (int): Number of bins along the x-axis.
        ybins (int): Number of bins along the y-axis.
        x_range (tuple): Range of x-values as (x_min, x_max).
        y_range (tuple): Range of y-values as (y_min, y_max).
        sigma (float): Bandwidth parameter controlling soft binning.

    Returns:
        torch.Tensor: 2D histogram of shape (xbins, ybins).
    """
    # Normalize x and y to the range [0, bins-1]
    x_min, x_max = x_range
    y_min, y_max = y_range

    x_normalized = (x - x_min) / (x_max - x_min) * (xbins - 1)
    y_normalized = (y - y_min) / (y_max - y_min) * (ybins - 1)

    # Generate bin centers for x and y
    x_bin_centers = torch.arange(xbins, device=x.device, dtype=x.dtype)
    y_bin_centers = torch.arange(ybins, device=y.device, dtype=y.dtype)

    # Compute the Gaussian weights for each bin
    x_weights = torch.exp(-((x_normalized[:, None] - x_bin_centers[None, :]) ** 2) / (2 * sigma ** 2))
    y_weights = torch.exp(-((y_normalized[:, None] - y_bin_centers[None, :]) ** 2) / (2 * sigma ** 2))

    # Normalize the weights to sum to 1 across bins
    x_weights = x_weights / x_weights.sum(dim=1, keepdim=True)
    y_weights = y_weights / y_weights.sum(dim=1, keepdim=True)

    # Compute the outer product of weights to form the 2D histogram
    histogram = torch.einsum('bi,bj->ij', x_weights, y_weights)

    return histogram

def threshold_from_first_return(image: np.ndarray, threshold) -> np.ndarray:
    #find the first piexel to surpass the threshold and remove all pixels after it 
    threshold_mask = image > threshold
    first_nonzero_indices = np.argmax(threshold_mask, axis=0)
    thresholded_image = np.zeros_like(image)
    thresholded_image[first_nonzero_indices, np.arange(image.shape[1])] = image[first_nonzero_indices, np.arange(image.shape[1])]
    return thresholded_image

def find_dark_pixels(image: np.ndarray, percentile: float = 0.1) -> np.ndarray:
    """
    Find the pixels with which to regress the noise parameter 
    Stick with the darkest pixels in the image adn also possibly compute shadows """

    #find the darkest pixels in image per each range row 
    dark_pixels = np.percentile(image, percentile, axis=1)

    #find a mask where these pixels exist 
    dark_pixel_mask = image < dark_pixels[:, None]
    return dark_pixel_mask

def polar_to_cartesian(polar_image: np.ndarray, max_range: float, hfov: float) -> np.ndarray:
    """
    Convert a polar-mapped image to a Cartesian image.

    Parameters:
    - polar_image (np.ndarray): Input image in polar coordinates (H, W).
    - max_range (float): Maximum range corresponding to the last row in polar_image.
    - hfov (float): Horizontal field of view in degrees.

    Returns:
    - cartesian_image (np.ndarray): Output image in Cartesian coordinates.
    """

    H, W = polar_image.shape  # H: radial bins, W: angular bins
    cart_size = 2 * H  # Define Cartesian image size (square)
    cartesian_image = np.zeros((cart_size, cart_size))  # Output image

    # Create Cartesian coordinate grid centered at (H, H)
    x = np.linspace(0, max_range, H)
    y = np.linspace(-max_range, max_range, 2*H)
    X, Y = np.meshgrid(x, y)

    R = np.sqrt(X**2 + Y**2)  # Radius
    Theta = np.arctan2(Y, X)  # Angle in radians (-π, π)
    # Convert Cartesian coordinates (X, Y) -> Polar coordinates (r, theta)
    # r = np.linspace(0, max_range, H)  # Radius
    # theta = np.linspace(-np.radians(hfov/2), np.radians(hfov/2), W)  # Angle in radians (-π, π)
    # R, Theta = np.meshgrid(r, theta)

    # Convert Theta range from radians (-π to π) to index range (0 to W-1)
    theta_min = -np.radians(hfov / 2) # Corresponds to index 0
    theta_max = np.radians(hfov / 2)  # Corresponds to index W-1

    Theta_idx = ((Theta - theta_min) / (theta_max - theta_min)) * (W - 1)

    # Convert R range from (0 to max_range) to index range (0 to H-1)
    R_idx = (R / max_range) * (H - 1)

    # Clip indices to valid range
    R_idx = np.clip(R_idx, 0, H - 1)
    Theta_idx = np.clip(Theta_idx, 0, W - 1)

    # Interpolate using map_coordinates (bilinear interpolation)
    cartesian_image = map_coordinates(polar_image, [R_idx, Theta_idx], mode="grid-constant", cval=0)
    cartesian_image[Theta > np.radians(hfov / 2)] = 0  # Zero out values outside the FOV
    cartesian_image[Theta < -np.radians(hfov / 2)] = 0  # Zero out values outside the FOV

    return cartesian_image


def points_to_sonar(points: torch.tensor, 
                    hfov: float = 90, 
                    vfov: float = 90, 
                    max_range: float = 10, 
                    num_azimuth_bins: int = 512, 
                    num_range_bins: int = 512, 
                    sigma=0.3):
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    
    # Calculate azimuth, elevation, and range for all points
    azimuths = torch.rad2deg(torch.arctan2(X, Z)) 
    elevations = torch.rad2deg(torch.arctan2(-Y, Z))
    ranges = torch.norm(points, dim=-1, p=2)

    # Binning in vectorized fashion
    azimuth_bin_size = hfov / num_azimuth_bins
    range_bin_size = max_range / num_range_bins

    # Calculate the bins for all points
    azimuth_bins = torch.clamp(((azimuths + hfov / 2) / azimuth_bin_size).round(), 0, num_azimuth_bins - 1).long()
    range_bins = torch.clamp((ranges / range_bin_size).round(), 0, num_range_bins - 1).long()

    # Create the sonar image (polar coordinate bins)
    sonar_image = differentiable_2d_histogram(x=azimuths, 
                                              y=ranges, 
                                              xbins=num_azimuth_bins,
                                              ybins=num_range_bins,
                                              x_range=(-hfov/2, hfov/2),
                                                y_range=(0, max_range),
                                                 sigma=sigma)
    

    frustum = create_frustum_wireframe(vertical_fov=vfov, horizontal_fov=hfov, max_range=max_range)

    return sonar_image