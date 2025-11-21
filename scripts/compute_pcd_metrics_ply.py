import torch
import numpy as np
import os
import kaolin.io as kaolin_io
import kaolin.metrics.pointcloud as pc_metrics
from types import SimpleNamespace
from argparse import Namespace, ArgumentParser
import open3d as o3d 
import kaolin 
from scipy.spatial.distance import directed_hausdorff
from collections import defaultdict
from tqdm import tqdm
voxel_size = 0.10 

def parse_cfg_args(file_path: str) -> Namespace:
    """
    Parses the cfg_args file and converts it into a Namespace object.
    
    Args:
        file_path (str): Path to the cfg_args file.
        
    Returns:
        Namespace: Configuration object containing attributes such as source_path.
    """
    with open(file_path, 'r') as file:
        cfg_content = file.read().strip()
    
    # Remove the 'Namespace(' and ')' and split by commas
    cfg_content = cfg_content.replace('Namespace(', '').rstrip(')')
    
    cfg_dict = {}
    for item in cfg_content.split(','):
        key, value = item.split('=')
        key = key.strip()
        value = value.strip().strip("'")
        
        # Attempt to convert to correct types
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        else:
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
        
        cfg_dict[key] = value
    
    return Namespace(**cfg_dict)

def load_pointcloud(file_path: str) -> torch.Tensor:
    """
    Loads a point cloud from a .ply file using Kaolin.
    
    Args:
        file_path (str): Path to the point cloud .ply file.
        
    Returns:
        torch.Tensor: Point cloud tensor of shape (N, 3).
    """
    #load pointcloud using open3d 
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    points = torch.tensor(points, dtype=torch.float32)
    return points.unsqueeze(0)

def compare_pointclouds(cloud1: torch.Tensor, cloud2: torch.Tensor, threshold: float) -> None:
    """
    Compare two point clouds using Chamfer L1 distance, precision, and recall.
    
    Args:
        cloud1 (torch.Tensor): First point cloud tensor of shape (1, N, 3).
        cloud2 (torch.Tensor): Second point cloud tensor of shape (1, M, 3).
        threshold (float): Distance threshold for precision and recall.
    """
    # Chamfer L1 Distance
    chamfer_l1 = pc_metrics.chamfer_distance(cloud1.cuda(), cloud2.cuda(), squared=False)[0].item()
    # print(f"Chamfer L1 Distance: {chamfer_l1}")
    
    # Precision and Recall
    f_score = pc_metrics.f_score(cloud1.cuda(), cloud2.cuda(), radius=0.05)[0].item()
    # print(f"F_score: {f_score}")

    #find the length fo the largest dim of the ponit clodu 
    max_length = (cloud1.max(dim=1)[0] - cloud1.min(dim=1)[0]).max()
    voxel_grid1 = kaolin.ops.conversions.pointclouds_to_voxelgrids(cloud1.cuda(), int(max_length//voxel_size))
    voxel_grid2 = kaolin.ops.conversions.pointclouds_to_voxelgrids(cloud2.cuda(), int(max_length//voxel_size))

    # Jaccard Index
    jaccard_index = kaolin.metrics.voxelgrid.iou(voxel_grid1, voxel_grid2)
    # print(f"Jaccard Index: {jaccard_index}")


    #calculate haussdorff distance using scipy 
    way_1_h = directed_hausdorff(cloud1.squeeze(0).cpu().numpy(), cloud2.squeeze(0).cpu().numpy())[0]
    way_2_h = directed_hausdorff(cloud2.squeeze(0).cpu().numpy(), cloud1.squeeze(0).cpu().numpy())[0]
    haussdorff_distance = max(way_1_h, way_2_h)

    return {"chamfer_l1": chamfer_l1, "f_score": f_score, "jaccard_index": jaccard_index.item(), "hausdorff_distance": haussdorff_distance}



if __name__ == "__main__":
    parser = ArgumentParser(description="Compute point cloud metrics between ground truth and predicted point clouds")
    parser.add_argument("--gt_root", type=str, required=True, help="Path to the ground truth root directory")
    parser.add_argument("--pred_root", type=str, required=True, help="Path to the predictions root directory")
    parser.add_argument("--viz", action="store_true", help="Enable visualization of point clouds")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of samples to use for evaluation")
    
    args = parser.parse_args()
    
    gt_root = args.gt_root
    pred_root = args.pred_root
    viz = args.viz
    num_samples = args.num_samples

    #find the corresponding files in the same subfolders in both roots 
    gt_scenes = [f for f in os.listdir(gt_root) if os.path.isdir(os.path.join(gt_root, f))]
    pred_scenes = [f for f in os.listdir(pred_root) if os.path.isdir(os.path.join(pred_root, f))]

    valid_scenes = list(set(gt_scenes).intersection(set(pred_scenes)))

    #go through all paths 
    all_metrics = defaultdict(lambda: defaultdict(list))
    for scene in valid_scenes:
        print(scene)
        gt_path = os.path.join(gt_root, scene)
        pred_path = os.path.join(pred_root, scene)

        gt_subsample_amt = 30000

        #store metrics in dictionary 
        
        # Define paths to the ground truth and iteration point clouds
        #if file is gt.asc read in pcd directly 
        if os.path.exists(os.path.join(gt_path, "gt.pcd")):
            gt_pointcloud_path = os.path.join(gt_path, "gt.pcd")
            gt_mesh = o3d.io.read_point_cloud(gt_pointcloud_path)
        else:
            gt_pointcloud_path = os.path.join(gt_path, "gt.ply")
            gt_mesh = o3d.io.read_triangle_mesh(gt_pointcloud_path)

        #check if mesh is ply or obj 
        if os.path.exists(os.path.join(pred_path, "pred.ply")):
            pred_mesh = o3d.io.read_triangle_mesh(os.path.join(pred_path, "pred.ply"))
        else:
            pred_mesh = o3d.io.read_triangle_mesh(os.path.join(pred_path, "pred.stl"))
            pred_mesh.translate([-326700.00, -4989000.00, 0.00])
            # -326700.00, -4989000.00

        #crop the meshes to the bbox of the gt mesh 
        min_bound = gt_mesh.get_min_bound()
        max_bound = gt_mesh.get_max_bound()

        #crop the pred mesh to the bbox of the gt mesh 
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
        
        #crop the pred mesh to the bbox of the gt mesh 
        pred_mesh = pred_mesh.crop(bbox)

        for i in tqdm(range(num_samples)):
            #if asc, just randomly choose gt_subsample_amt points 
            if os.path.exists(os.path.join(gt_path, "gt.asc")):
                gt_sampled_idx = np.random.choice(len(gt_mesh.points), gt_subsample_amt, replace=False)
                cloud1 = torch.tensor(torch.tensor(gt_mesh.points)[gt_sampled_idx], dtype=torch.float32).unsqueeze(0)
            else:
                gt_sampled_points = gt_mesh.sample_points_uniformly(number_of_points=gt_subsample_amt)
                cloud1 = torch.tensor(gt_sampled_points.points, dtype=torch.float32).unsqueeze(0)
        
            # # Load the point clouds
            # cloud1 = load_pointcloud(gt_pointcloud_path)  # Ground truth point cloud
            # # cloud2 = load_pointcloud(obj_files[0])  # Iteration 30000 point cloud

            # # Subsample the ground truth point cloud
            # if cloud1.shape[1] > gt_subsample_amt:
            #     idxs = np.random.choice(cloud1.shape[1], gt_subsample_amt, replace=False)
            #     cloud1 = cloud1[:, idxs]

            #sample points on the mesh 
            pred_sampled_points = pred_mesh.sample_points_uniformly(number_of_points=gt_subsample_amt)

            cloud2 = torch.tensor(pred_sampled_points.points, dtype=torch.float32).unsqueeze(0)

            if viz: 
                #visualize the pointclouds 
                cloud1_viz = cloud1[0].cpu().numpy()
                cloud2_viz= cloud2[0].cpu().numpy()

                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(cloud1_viz)
                #color green 
                pcd1.paint_uniform_color([0, 1, 0])

                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(cloud2_viz)
                #color red
                pcd2.paint_uniform_color([1, 0, 0])

                o3d.visualization.draw_geometries([pcd1, pcd2])
            # #crop cloud2 to the bbox of cloud1 
           
            # Compare the point clouds
            scene_metrics  = compare_pointclouds(cloud1, cloud2, threshold=0.01)
            for key, value in scene_metrics.items():
                all_metrics[scene][key].append(value)

    
    #print a report with average of all metrics 
    for scene in all_metrics:
        print(f"Scene: {scene}")
        for metric_name in ['hausdorff_distance', 'chamfer_l1']:
            values = all_metrics[scene][metric_name]
            mean_value = np.mean(values)
            rms_value = np.sqrt(np.mean(np.array(values)**2))
            
            if metric_name == 'hausdorff_distance':
                print(f"RMS Hausdorff: {rms_value:.10f}")
                print(f"Mean Hausdorff: {mean_value:.10f}")
            else:  # chamfer_l1
                print(f"RMS Chamfer: {rms_value:.10f}")
                print(f"Mean Chamfer: {mean_value:.10f}")

    
