import torch
import torch.nn as nn
from pytorch3d.ops import sample_farthest_points, ball_query, knn_points


def gather(x, idx):
    """gather elemtents of x using index"""
    idx = torch.where(idx==-1, idx.shape[1], idx)
    idx = idx[ :, :, :, None].expand(-1, -1, -1, x.shape[-1])  # [ :, :, :, None] equivalent to unsqueeze(-1)
    y = x[ :, :, None].expand(-1, -1, idx.shape[-2], -1).gather(1, idx)
    
    return y
    
    
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
