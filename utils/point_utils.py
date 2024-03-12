import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points, ball_query, knn_points


def gather(x, idx):
    """gather elemtents of x using index"""
    idx = torch.where(idx==-1, idx.shape[1], idx)
    idx = idx[ :, :, :, None].expand(-1, -1, -1, x.shape[-1])  # [ :, :, :, None] equivalent to unsqueeze(-1)
    y = x[ :, :, None].expand(-1, -1, idx.shape[-2], -1).gather(1, idx)
    
    return y


def pc_normalize(xyz):
    B, N, C = xyz.shape  
    center_xyz = torch.mean(xyz, dim=-2).detach()
    center_xyz_expand = center_xyz.unsqueeze(-2).expand_as(xyz)
    d = F.pairwise_distance(xyz.reshape(-1, C), center_xyz_expand.reshape(-1, C)).reshape(B, N)
    d_max = d.max(dim=-1)[0].detach()
    d_max_expand = d_max.unsqueeze(-1).unsqueeze(-1).expand_as(xyz)
    xyz_norm = (xyz - center_xyz_expand) / d_max_expand
    
    return center_xyz, d_max, xyz_norm     
    
    
class SampleAndGroup(nn.Module):
    
    
    def __init__(
        self, 
        sample_num, 
        radius, 
        neighbor_num
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
        
        
    def forward(self, xyz, feat):
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
