import os
import trimesh
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from pytorch3d.ops import knn_points, knn_gather

from .acronym_utils import load_mesh, load_grasps


class GraspDataset(Dataset):
    
    
    def __init__(
        self,
        object_dir,
        mesh_dir,
        point_num,
        mode='train'
        ):
        '''
        Params:
            object_dir: the directory used for loading objects.
            
            mesh_dir: the directory used for loading objects' meshes.
        
            point_num: the number of points to be sampled from the meshes.
            
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
    
        
def set_ground_truth(grasp, T, success):
    ''' Set the ground truth of a grasp based on the grasp nearest to the point cloud.
    
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
        norm=2, 
        K=1, 
    )
    B, W = success.shape
    grasp_gt = knn_gather(T.float().reshape(B, W, -1), neighbor_idx).reshape(B, -1, 4, 4)
    class_gt = knn_gather(success.float().unsqueeze(-1), neighbor_idx).reshape(B, -1)
    
    return grasp_gt, class_gt