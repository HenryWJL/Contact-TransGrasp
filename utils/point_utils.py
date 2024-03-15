import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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


def gather(
    x: Optional[torch.Tensor],
    idx: Optional[torch.Tensor]
    ):
    """gather elemtents using indices"""
    idx = torch.where(idx==-1, idx.shape[1], idx)
    idx = idx[ :, :, :, None].expand(-1, -1, -1, x.shape[-1])  # [ :, :, :, None] equivalent to unsqueeze(-1)
    y = x[ :, :, None].expand(-1, -1, idx.shape[-2], -1).gather(1, idx)
    
    return y


def square_distance(
    p1: Optional[torch.Tensor],
    p2: Optional[torch.Tensor]
    ):
    """
    Params:
        p1: the xyz coordinates of a point cloud (B, N, 3)
        
        p2: the xyz coordinates of another point cloud (B, M, 3)
        
    Returns:
        p1 per point square distance w.r.t p2 (B, N, M)
    """
    B, N, _ = p1.shape
    M = p2.shape[1]
    p1 = p1.unsqueeze(-2).repeat(1, 1, M, 1)  # (B, N, M, 3)
    p2 = p2.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, M, 3)
    square_dist = torch.sum((p1 - p2) ** 2, dim=-1)
    
    return square_dist


def knn_points(
    p1: Optional[torch.Tensor],
    p2: Optional[torch.Tensor],
    K: Optional[int]
    ):
    """
    Params:
        p1: the xyz coordinates of a point cloud (B, N, 3)
        
        p2: the xyz coordinates of another point cloud (B, M, 3)
        
        K: the number of neighbors
        
    Returns:
        square distance between p1 and its K nearest neighbors in p2 (B, N, K)
        
        indices of p1's K nearest neighbors in p2 (B, N, K)
        
        p1's K nearest neighbors in p2 (B, N, K, 3)
    """
    dist = square_distance(p1, p2)
    neighbor_dist, neighbor_idx = torch.topk(dist, K, dim=-1, largest=False, sorted=False)
    k_neighbor = gather(p2, neighbor_idx)
    
    return neighbor_dist, neighbor_idx, k_neighbor

# copy from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    
    return new_points

# copy from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        the sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    sample_xyz = index_points(xyz, centroids)
    
    return sample_xyz

# copy from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    
    return group_idx

    
class SampleAndGroup(nn.Module):
    
    
    def __init__(
        self, 
        sample_num: Optional[int], 
        radius: Optional[float], 
        neighbor_num: Optional[int]
        ):
        """
        Params:
            sample_num: the number of sampled points
            
            radius: the radius of ball
            
            neighbor_num: the number of neighbor points in a group
        """
        super().__init__()
        
        self.sample_num = sample_num
        self.radius = radius
        self.neighbor_num = neighbor_num
        
        
    def forward(
        self,
        xyz: Optional[torch.Tensor],
        feat: Optional[torch.Tensor]
        ):
        """
        Params:
            xyz: the original point cloud (B, N, 3)
            
            feat: per point feature (B, N, C)
            
        Returns:
            the sampled points, the features of neighbor points 
        """
        sample_xyz = farthest_point_sample(
            xyz=xyz,
            npoint=self.sample_num
        )
        neighbor_idx = query_ball_point(
            radius=self.radius,
            nsample=self.neighbor_num,
            xyz=xyz,
            new_xyz=sample_xyz
        )
        feat_group = gather(feat, neighbor_idx)
        
        return sample_xyz, feat_group