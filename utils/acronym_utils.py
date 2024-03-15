import os
import json
import h5py
import trimesh
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from glob import glob
from typing import Optional

from .point_utils import knn_points, gather
from .transforms import pnt2quat, mat2quat

# copy from https://github.com/NVlabs/acronym/blob/main/acronym_tools/acronym.py
def load_mesh(filename, mesh_root_dir, scale=None):
    """Load a mesh from a JSON or HDF5 file from the grasp dataset. The mesh will be scaled accordingly.

    Args:
        filename (str): JSON or HDF5 file name.
        scale (float, optional): If specified, use this as scale instead of value from the file. Defaults to None.

    Returns:
        trimesh.Trimesh: Mesh of the loaded object.
    """
    if filename.endswith(".json"):
        data = json.load(open(filename, "r"))
        mesh_fname = data["object"].decode('utf-8')
        mesh_scale = data["object_scale"] if scale is None else scale
    elif filename.endswith(".h5"):
        data = h5py.File(filename, "r")
        mesh_fname = data["object/file"][()].decode('utf-8')
        mesh_scale = data["object/scale"][()] if scale is None else scale
    else:
        raise RuntimeError("Unknown file ending:", filename)

    obj_mesh = trimesh.load(os.path.join(mesh_root_dir, mesh_fname), force='mesh')  
    obj_mesh = obj_mesh.apply_scale(mesh_scale)                                     
                                                                                    
    return obj_mesh

# copy from https://github.com/NVlabs/acronym/blob/main/acronym_tools/acronym.py
def load_grasps(filename):
    """Load transformations and qualities of grasps from a JSON file from the dataset.

    Args:
        filename (str): HDF5 or JSON file name.

    Returns:
        np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
        np.ndarray: List of binary values indicating grasp success in simulation.
    """
    if filename.endswith(".json"):
        data = json.load(open(filename, "r"))
        T = np.array(data["transforms"])
        success = np.array(data["quality_flex_object_in_gripper"])
    elif filename.endswith(".h5"):
        data = h5py.File(filename, "r")
        T = np.array(data["grasps/transforms"])
        success = np.array(data["grasps/qualities/flex/object_in_gripper"])
    else:
        raise RuntimeError("Unknown file ending:", filename)
    
    return T, success


def set_ground_truth(
    grasp: Optional[torch.Tensor],
    T: Optional[torch.Tensor],
    success: Optional[torch.Tensor]
    ):
    '''Set ground truth using the nearest neighbor.
    
    Params:
        grasp: the predicted 7-DoF grasps (B, M, 7)
        
        T: the ground-truth transformation matrix (B, W, 4, 4)
        
        success: the ground-truth grasp quality (B, W)

    Returns:
        grasp_gt: the assigned ground-truth grasp (B, M, 4, 4)

        class_gt: the assigned ground-truth grasp quality (B, M)
        
    '''
    center_xyz = (grasp[:, :, :3] + grasp[:, :, 3:6]) / 2
    gt_center_xyz = T[:, :, :3, 3]
    _, neighbor_idx, _ = knn_points(
        p1=center_xyz.float(), 
        p2=gt_center_xyz.float(), 
        K=1
    )
    B, W = success.shape
    grasp_gt = gather(T.float().reshape(B, W, -1), neighbor_idx).reshape(B, -1, 4, 4)
    class_gt = gather(success.float().unsqueeze(-1), neighbor_idx).reshape(B, -1)
    
    return grasp_gt, class_gt


def evaluate(
    grasp_pred: Optional[torch.Tensor],
    grasp_gt: Optional[torch.Tensor],
    class_pred: Optional[torch.Tensor],
    class_gt: Optional[torch.Tensor]
    ):
    """Evaluate model performance"""
    # translation and rotation errors
    center_pred, quat_pred = pnt2quat(grasp_pred)
    center_gt, quat_gt = mat2quat(grasp_gt)
    dist = F.pairwise_distance(center_pred.reshape(-1, 3), center_gt.reshape(-1, 3))
    trans_error = torch.mean(dist * class_gt.reshape(-1)).item()
    rot_error = 1 - (quat_pred.reshape(-1, 4) * quat_gt.reshape(-1, 4)).sum(dim=-1)
    rot_error = torch.mean(rot_loss).item()
    # classification accuracy
    cls_accuracy = torch.mean(((class_pred > 0.5) == class_gt).float()).item()
    
    return trans_error, rot_error, cls_accuracy


class GraspDataset(Dataset):
    
    
    def __init__(
        self,
        object_dir: Optional[str],
        mesh_dir: Optional[str],
        point_num: Optional[int],
        mode: Optional[str] = 'train'
        ):
        '''Acronym dataset
        
        Params:
            object_dir: the directory used for loading objects.
            
            mesh_dir: the directory used for loading meshes.
        
            point_num: the number of points to be sampled from meshes.
            
            mode: 'train', 'val', or 'test'.
            
        '''    
        super().__init__()
        
        self.object_fname = glob(os.path.join(object_dir, mode, "**.h5"))
        self.mesh_dir = os.path.join(mesh_dir, mode)
        self.point_num = point_num
    
    
    def __getitem__(self, key):
        T, success = load_grasps(self.object_fname[key])
        mesh = load_mesh(self.object_fname[key], mesh_root_dir=self.mesh_dir)
        point_cloud = mesh.sample(self.point_num)
        point_cloud = torch.from_numpy(point_cloud).float()
        return point_cloud, T, success
    
    
    def __len__(self):
        return len(self.object_fname)