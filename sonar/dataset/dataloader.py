import os
import json
from typing import Any, Dict, List, Optional
from typing_extensions import assert_never
import matplotlib
# matplotlib.use("Qt5Agg")
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from PIL import Image
import pickle
import yaml
import re
from pathlib import Path
from tqdm import tqdm
print(matplotlib.get_backend())
from sonar.utils import visualize_gaussians
# from .normalize import (
#     align_principle_axes,
#     similarity_from_cameras,
#     transform_cameras,
#     transform_points,
# )
def initialize_gs_range_azimuth(return_threshold: float, 
                                c2w: np.ndarray, 
                                image: Image, 
                                num_range_bins: int, 
                                num_azimuth_bins: int, 
                                max_range: float, 
                                hfov: float, 
                                vfov: float, 
                                num_samples: int, 
                                randomize_elevation: bool = False):
    """
    Initialize 3D means in space given sensor pose and intrinsics
    """

    #filter the image first 
    H, W = image.shape
    image = np.array(image)
    return_mask = image > return_threshold

    # Compute connected components
    # thresh = cv2.threshold((image).astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # num_labels, labels = cv2.connectedComponents(thresh)

    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(image)
    # ax[1].imshow(labels)
    # plt.show()

    v, u = np.meshgrid( np.arange(num_range_bins), np.arange(num_azimuth_bins))

    #reshape to num_range_bins x num_azimuth_bins x 2 
    uv = np.stack([u, v], axis=-1)
    valid_uv = uv[return_mask]

    r = valid_uv[:,1] / num_range_bins * max_range
    theta = valid_uv[:,0] / num_azimuth_bins * hfov - hfov/2
    elevation = np.random.uniform(-vfov/2, vfov/2, size=r.shape) if randomize_elevation else np.zeros_like(r)

    #project out to 3D space 
    x = r*np.cos(np.deg2rad(theta))*np.cos(np.deg2rad(elevation))  # forward
    y = r*np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(elevation))  # right
    z = r*np.sin(np.deg2rad(elevation))  # down

    points = np.stack([x, y, z], axis=-1)
    #make homogeneous 
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
    # points = np.dot(sensor_pose[:3,:3], points.T).T + sensor_pose[:3,3]
    points = c2w @ points.T
    points = points[:3,:].T

    #randomly subsample 
    if points.shape[0] > num_samples:
        idx = np.random.choice(points.shape[0], num_samples, replace=False)
        points = points[idx]

    # #display points here 
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    # frame.transform(c2w)
    # o3d.visualization.draw_geometries([pcd, frame])

    # print("done")
    return points

def sample_points_in_elevation(return_threshold: float, 
                                c2w: np.ndarray, 
                                image: Image, 
                                num_range_bins: int, 
                                num_azimuth_bins: int, 
                                max_range: float, 
                                hfov: float, 
                                vfov: float, 
                                num_samples: int, ):
    """
    Initialize 3D means in space given sensor pose and intrinsics
    """

    #filter the image first 
    H, W = image.shape
    image = np.array(image)
    return_mask = image > return_threshold

    # Compute connected components
    # thresh = cv2.threshold((image).astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # num_labels, labels = cv2.connectedComponents(thresh)

    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(image)
    # ax[1].imshow(labels)
    # plt.show()

    v, u = np.meshgrid( np.arange(num_range_bins), np.arange(num_azimuth_bins))

    #reshape to num_range_bins x num_azimuth_bins x 2 
    uv = np.stack([u, v], axis=-1)
    valid_uv = uv[return_mask]

    r = valid_uv[:,1] / num_range_bins * max_range
    theta = valid_uv[:,0] / num_azimuth_bins * hfov - hfov/2
    elevation = np.random.uniform(-vfov/2, vfov/2, size=(num_samples*return_mask.sum(),))

    r = r.repeat(num_samples)
    theta = theta.repeat(num_samples)

    

    #project out to 3D space 
    x = r*np.cos(np.deg2rad(theta))*np.cos(np.deg2rad(elevation))  # forward
    y = r*np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(elevation))  # right
    z = r*np.sin(np.deg2rad(elevation))  # down

    points = np.stack([x, y, z], axis=-1)
    #make homogeneous 
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
    # points = np.dot(sensor_pose[:3,:3], points.T).T + sensor_pose[:3,3]
    points = c2w @ points.T
    points = points[:3,:].T

    #randomly subsample 
    # if points.shape[0] > num_samples:
    #     idx = np.random.choice(points.shape[0], num_samples, replace=False)
    #     points = points[idx]

    # #display points here 
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    # frame.transform(c2w)
    # o3d.visualization.draw_geometries([pcd, frame])

    # print("done")
    return points

class SonarSensorDataParser:
    """Sonar sensor data parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1, # not used
        normalize: bool = False, # not used
        test_every: int = 8, 
        override_max_range: float = -1,
        init_threshold: float = 0.2,
        img_threshold: float = 0.2,
        num_init_samples: int = 50,
        range_clear_start: int = 200,
        range_clear_end: int = 0,
        randomize_elevation: bool = False,
        num_random_points: int = 10000,
        render_dir: str = None,
        start_from_frame: int = 0,
        end_at_frame: int = 100000,
        skip_frames: int = 100,
        apply_mask: bool = True,
        dset_keep_only_first_image: bool = False,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every


        assert os.path.exists(
            data_dir
        ), f"Data directory {data_dir} does not exist."

        # Load poses data from the .tum file
        files = os.listdir(os.path.join(self.data_dir, "Data"))
        files.sort(key=lambda f: int(re.sub('\D', '', f)))
        config_path = os.path.join(self.data_dir, "Config.json")

        assert len(files) > 0, "No files found in data folder."

         #read json 
        with open(config_path) as json_file:
            config = json.load(json_file)
            config = config["agents"][0]["sensors"][-1]['configuration']
            num_range_bins = config['RangeBins']
            num_azimuth_bins = config['AzimuthBins']
            hfov = config['Azimuth']
            vfov = config['Elevation']
            max_range = config['RangeMax']
            # assert os.path.exists(poses_file), f"Poses file {poses_file} does not exist."
            # poses = load_tum_poses(poses_file) # (num_images, 4, 4)

        if override_max_range > 0:
            loss_cutoff_pixel = int(override_max_range / max_range * num_range_bins)
            max_range = override_max_range
            num_range_bins = loss_cutoff_pixel
        else: 
            loss_cutoff_pixel = num_range_bins
        
        # sensor intrinsics
        range_resolution = max_range / num_range_bins 
        azimuth_resolution = hfov / num_azimuth_bins 

        

        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        images = []
        pcd = []
        image_names = []

        num_selected = min(len(files), (end_at_frame - start_from_frame + 1) // skip_frames)
        samples_per_image = num_init_samples // num_selected 

        print(f"num_selected: {num_selected}, samples_per_image: {samples_per_image}")

        for fi, fname in tqdm(enumerate(files), desc="Processing images"): 
            if (skip_frames == 0 or fi % skip_frames == 0) and fi >= start_from_frame and fi <= end_at_frame:
                if fname.endswith(".pkl"):
                    with open(os.path.join(self.data_dir, "Data", fname), 'rb') as file:  
                        data = pickle.load(file)
                        image = data['ImagingSonar']

                        #flip the image horizontally 
                        # image = image[:, ::-1]
                        image = image.transpose(1, 0) #turn into a azimuth, range image
                        image[np.isnan(image)] = 0.0
                        H, W = image.shape

                        if H > num_azimuth_bins: 
                            image = image[:num_azimuth_bins]
                        if W > num_range_bins: 
                            image = image[:, :num_range_bins]
                        image[image < img_threshold] = 0.0
                        image[:, min(W, range_clear_start):] = 0.0
                        image[:, :range_clear_end] = 0.0

                        image = np.clip(image, 0.0, 1.0)

                        image[:, 0:10] = 0.0 #zero-out edge noise, seems to be artifact of ROS driver
                        image[:, -10:] = 0.0
                        image[0:10, :] = 0.0
                        image[-10:, :] = 0.0

                        image = image[:, :loss_cutoff_pixel]
                        
                        if apply_mask:
                            
                            label_fname = "polar_" + Path(fname).stem.lstrip('0') + ".png"
                            label_path = os.path.join(self.data_dir, "sonar_images", "labels", label_fname)

                            #read with PIL 
                            label = Image.open(label_path)
                            label = np.array(label)
                            label = label.transpose(1, 0)
                            label = label.astype(np.float32) 
                            label = np.where(label > 0.5, 1.0, 0.0)
                            # image = image * label
                            image = label #TODO: remove this!

                        if image.sum() > 0:
                            image_names.append(fname)
                            #in-fill block here. 
                            
                            images.append(image)
                            
                            c2w = data['PoseSensor'] 
                            w2c = np.linalg.inv(c2w)

                            initial_points = initialize_gs_range_azimuth(return_threshold=init_threshold, 
                                                                    c2w=c2w, 
                                                                    image=image, 
                                                                    num_range_bins=num_range_bins, 
                                                                    num_azimuth_bins=num_azimuth_bins, 
                                                                    max_range=max_range, 
                                                                    hfov=hfov, vfov=vfov, 
                                                                    num_samples=samples_per_image, 
                                                                    randomize_elevation=randomize_elevation)
                            pcd.append(initial_points)

                            w2c_mats.append(w2c)
                            camera_ids.append(0)

                            K = np.array(
                                        [
                                            [1.0/range_resolution, 0.0, 0.0], 
                                            [0.0, 180/torch.pi*(1/azimuth_resolution), num_azimuth_bins/2.],
                                            [0, 0, 1],
                                        ])
                            Ks_dict[0] = K
                            mask_dict[0] = None
                            imsize_dict[0] = (num_azimuth_bins, num_range_bins)
                        
                        #save every image to render dir 
                        if render_dir is not None:
                            os.makedirs(os.path.join(render_dir, "dbg_images"), exist_ok=True)
                            imageio.imwrite(os.path.join(render_dir, "dbg_images", Path(fname).stem + ".png"), (255*image).astype(np.uint8))
                
                if dset_keep_only_first_image and len(image_names) > 1:
                    break

        pcd = np.concatenate(pcd, axis=0)
        # pcd = np.zeros((1,3))

        #find the bounds of the pcd 
        bounds = np.array([np.min(pcd[:,0]), np.min(pcd[:,1]), np.min(pcd[:,2]), np.max(pcd[:,0]), np.max(pcd[:,1]), np.max(pcd[:,2])])
        
        #sample points randomly from the bounds 
        num_points = num_random_points
        points = np.random.uniform(bounds[:3], bounds[3:], size=(num_points, 3))
        pcd = np.concatenate([pcd, points], axis=0)

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # inds = np.argsort(image_names)
        # image_names = [image_names[i] for i in inds]
        # camtoworlds = camtoworlds[inds]
        # camera_ids = [camera_ids[i] for i in inds]

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0]) # TODO: not sure what is this bound for?
        # posefile = os.path.join(data_dir, "poses_bounds.npy")
        # if os.path.exists(posefile):
        #     self.bounds = np.load(posefile)[:, -2:]

        # # Normalize the world space.
        # if normalize:
        #     T1 = similarity_from_cameras(camtoworlds)
        #     camtoworlds = transform_cameras(T1, camtoworlds)
        #     points = transform_points(T1, points)

        #     T2 = align_principle_axes(points)
        #     camtoworlds = transform_cameras(T2, camtoworlds)
        #     points = transform_points(T2, points)

        #     transform = T2 @ T1
        # else:
        #     transform = np.eye(4)
        visualize_gaussians(xyz=[pcd], poses=camtoworlds, start_size=0.1, end_size=0.1)
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        # Define the scene scale as the maximum distance from the center of the scene to the camera locations
        # plus the maximum sensing range of the sensor
        self.scene_scale = np.max(dists) + 2 * W * range_resolution


        transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.images = images  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.transform = transform  # np.ndarray, (4, 4)
        self.hfov_deg = hfov
        self.vfov_deg = vfov
        self.points = pcd
        self.keep_only_first_image = dset_keep_only_first_image
        self.points_rgb = np.ones_like(pcd) * 0.5
        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        # Define the scene scale as the maximum distance from the center of the scene to the camera locations
        # plus the maximum sensing range of the sensor
        self.scene_bounds = self.getSonarNorm(camtoworlds, max_range, [-hfov/2, hfov/2], [-vfov/2, vfov/2])

        self.max_range = max_range
        self.range_resolution = range_resolution
        self.azimuth_resolution = azimuth_resolution
        self.num_range_bins = num_range_bins
        self.num_azimuth_bins = num_azimuth_bins

    def getSonarNorm(self, sensor_poses, max_range, azimuth_extents, elevation_extents):
        """
        Calculate the 3D bounding box for a scene covered by multiple sensors.

        Parameters:
        - sensor_poses (list of np.ndarray): List of 4x4 transformation matrices (global sensor poses).
        - max_ranges (list of float): List of max detection ranges for each sensor.
        - azimuth_extents (list of tuple): List of (min_azimuth, max_azimuth) angles in degrees.
        - elevation_extents (list of tuple): List of (min_elevation, max_elevation) angles in degrees.

        Returns:
        - bounds (dict): Dictionary with scene bounds {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
        """

        # Store all transformed points
        all_points = []

        for pose in sensor_poses:
            az_min, az_max = np.radians(azimuth_extents)  # Convert to radians
            el_min, el_max = np.radians(elevation_extents)  # Convert to radians
            
            # Define 3D spherical boundary points (sensor-centered)
            spherical_points = [
                (max_range, az_min, el_min),  # Bottom-left
                (max_range, az_min, el_max),  # Top-left
                (max_range, az_max, el_min),  # Bottom-right
                (max_range, az_max, el_max),  # Top-right
            ]

            # Convert spherical (range, azimuth, elevation) to Cartesian
            sensor_frame_points = []
            for r, az, el in spherical_points:
                x = r * np.cos(el) * np.cos(az)
                y = r * np.cos(el) * np.sin(az)
                z = -r * np.sin(el)
                sensor_frame_points.append(np.array([x, y, z, 1]))  # Homogeneous coord

            # Transform to global coordinates using sensor pose
            for p in sensor_frame_points:
                global_point = pose @ p  # Apply transformation
                all_points.append(global_point[:3])  # Store (x, y, z)
                all_points.append(pose[:3,3])

        #visualzie the potins and the frames 
        # visualize_gaussians(np.array(all_points), poses=[])
                
        # Convert to NumPy array
        all_points = np.array(all_points)

        # Compute scene bounds
        bounds = {
            "x_min": np.min(all_points[:, 0]),
            "x_max": np.max(all_points[:, 0]),
            "y_min": np.min(all_points[:, 1]),
            "y_max": np.max(all_points[:, 1]),
            "z_min": np.min(all_points[:, 2]),
            "z_max": np.max(all_points[:, 2]),
        }

        #translate is the center of the scene
        translate = -np.array([bounds["x_max"] + bounds["x_min"], bounds["y_max"] + bounds["y_min"], bounds["z_max"] + bounds["z_min"]]) / 2
        radius = 1.1*np.max([bounds["x_max"] - bounds["x_min"], bounds["y_max"] - bounds["y_min"], bounds["z_max"] - bounds["z_min"]]) / 2

        return {"translate": translate, "radius": radius}

class SonarSensorDataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: SonarSensorDataParser,
        split: str = "train",
    ):
        self.parser = parser
        self.split = split
        indices = np.arange(len(self.parser.image_names))
        if parser.keep_only_first_image:
            self.indices = indices[:1]
        else:
            if split == "train":
                self.indices = indices[indices % self.parser.test_every != 0]
                if len(indices)==1 or self.parser.test_every == 1: # Single image. Set the image as train set.
                    self.indices = indices
                print(f"\033[92mTrain set size: {len(self.indices)}\033[0m") #print in green 
            else:
                self.indices = indices[indices % self.parser.test_every == 0]
                print(f"\033[91mTest set size: {len(self.indices)}\033[0m") #print in red
        self.use_polar = True

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = self.parser.images[index]
        
        # if image.ndim==3 and image.shape[-1]==3:
        #     # The data is in rgb. Convert to gray-scaled
        #     image = image[:,:,0]
        
        # if self.parser.max_range is not None:
        #     W = int(self.parser.max_range / self.parser.range_resolution)
        #     image = image[:,:W]

        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float().unsqueeze(-1),
            "image_id": item,  # the index of the image in the dataset
            "near_plane": -np.radians(self.parser.vfov_deg/2),
            "far_plane": np.radians(self.parser.vfov_deg/2),
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        return data

