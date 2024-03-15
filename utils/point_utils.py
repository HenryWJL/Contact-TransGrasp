import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pytorch3d.ops import sample_farthest_points, ball_query, knn_points


def gather(
    x: Optional[torch.Tensor],
    idx: Optional[torch.Tensor]
    ):
    """gather elemtents of x using index"""
    idx = torch.where(idx==-1, idx.shape[1], idx)
    idx = idx[ :, :, :, None].expand(-1, -1, -1, x.shape[-1])  # [ :, :, :, None] equivalent to unsqueeze(-1)
    y = x[ :, :, None].expand(-1, -1, idx.shape[-2], -1).gather(1, idx)
    
    return y


def pc_normalize(xyz: Optional[torch.Tensor]):
    """normalize point cloud"""
    B, N, C = xyz.shape  
    center_xyz = torch.mean(xyz, dim=-2).detach()
    center_xyz_expand = center_xyz.unsqueeze(-2).expand_as(xyz)
    d = F.pairwise_distance(xyz.reshape(-1, C), center_xyz_expand.reshape(-1, C)).reshape(B, N)
    d_max = d.max(dim=-1)[0].detach()
    d_max_expand = d_max.unsqueeze(-1).unsqueeze(-1).expand_as(xyz)
    xyz_norm = (xyz - center_xyz_expand) / d_max_expand
    
    return xyz_norm, center_xyz, d_max


def square_distance(
    xyz_up: Optional[torch.Tensor],
    xyz_down: Optional[torch.Tensor]
    ):
    """
    
    Params:
        xyz_up: the xyz coordinates of point clouds with less points (B, N, 3)
        
        xyz_down: the xyz coordinates of point clouds with more points (B, M, 3)
        
    Returns:
        per point square distance (B, N, M)
        
    """
    B, N, _ = xyz_up.shape
    M = xyz_down.shape[1]
    xyz_up = xyz_up.unsqueeze(-2).repeat(1, 1, M, 1)  # (B, N, M, 3)
    xyz_down = xyz_down.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, M, 3)
    square_dist = torch.sum((xyz_up - xyz_down) ** 2, dim=-1)
    
    return square_dist
    
    
class SampleAndGroup(nn.Module):
    
    
    def __init__(
        self, 
        sample_num: Optional[int], 
        radius: Optional[float], 
        neighbor_num: Optional[int]
        ):
        '''
        Params:
            sample_num: the number of sampled points
            
            radius: the radius of ball
            
            neighbor_num: the number of neighbor points in a group
        '''
        super().__init__()
        
        self.sample_num = sample_num
        self.radius = radius
        self.neighbor_num = neighbor_num
        
        
    def forward(
        self,
        xyz: Optional[torch.Tensor],
        feat: Optional[torch.Tensor]
        ):
        '''
        Params:
            xyz: the original point cloud (B, N', 3)
            
            feat: per point feature (B, N', C)
            
        Returns:
            the sampled points, the neighbor points's features
        '''
        sample_xyz, _ = sample_farthest_points(
            points=xyz,
            lengths=None,
            K=self.sample_num,
            random_start_point=True
        )
        _, neighbor_idx, _ = ball_query(
            p1=sample_xyz,
            p2=xyz,
            lengths1=None,
            lengths2=None,
            K=self.neighbor_num,
            radius=self.radius,
            return_nn=True
        )
        neighbor_feat = gather(feat, neighbor_idx)
        
        return sample_xyz, neighbor_feat
