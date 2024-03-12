import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pytorch3d.ops import knn_points

from utils import SampleAndGroup, SoftProjection, gather, pc_normalize 


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


class MiniPointNet(nn.Module):
    
    
    def __init__(
        self,
        dim_in: Optional[int],
        out_channels: Optional[list],
        bn: Optional[bool] = True
        ):
        """Mini PointNet module in PointNet++
        
        Params:
            in_channel: the dimension of the input
            
            out_channels: the output channels of Conv2d submodules
            
            bn: if True, add batch normalization
            
        """
        super().__init__()
        
        self.mlps = nn.ModuleList()
        in_channel = dim_in
        for out_channel in out_channels:
            self.mlps.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlps.append(nn.BatchNorm2d(out_channel) if bn else nn.Identity())
            in_channel = out_channel
        
        
    def forward(self, x: Optional[torch.Tensor]):
        for i, mlp in enumerate(self.mlps):
            x = mlp(x)
            
        return x
    

class MultiheadAttention(nn.Module):
    
    
    def __init__(
        self,
        embed_dim: Optional[int],
        num_heads: Optional[int],
        dropout: Optional[float] = 0.0,
        bn: Optional[bool] = True,
        qkv_bias: Optional[bool] = False
        ):
        """Multi-head vector attention with relative positional encoding
        
        Params:
            embed_dim: the dimension of the input embeddings
            
            num_heads: the number of attention heads
        
            dropout: the dropout ratio
            
            bn: if True, add batch normalization to the attention mlp
            
            qkv_bias: if True, add bias to qkv
            
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(
                in_channels=self.head_dim,
                out_channels=self.head_dim,
                kernel_size=1,
                groups=self.head_dim
            ),
            nn.BatchNorm2d(self.head_dim) if bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.head_dim,
                out_channels=1,
                kernel_size=1
            ),
            nn.BatchNorm2d(1) if bn else nn.Identity()
        )
        self.softmax = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout, inplace=True)
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout, inplace=True)
        )
        
        
    def forward(
        self,
        q: Optional[torch.Tensor],
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        pos: Optional[torch.Tensor]
        ):
        """
        
        Params:
            q: query
            
            k: key
            
            v: value
            
            pos: position embeddings
            
        """
        B, N, C = q.shape
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v).reshape(B, N, self.num_heads, self.head_dim).transpose(2, 1)  # (B, num_heads, N, head_dim)
        qk_rel = q.unsqueeze(2) - k.unsqueeze(1)  # (B, N, N, embed_dim)
        qk_rel = qk_rel.reshape(B, N, N, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B, num_heads, N, N, head_dim)
        pos = pos.reshape(B, N, N, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B, num_heads, N, N, head_dim)
        attn_mlp_input = (qk_rel + pos).reshape(-1, N, N, self.head_dim).permute(0, 3, 1, 2)  # (B * num_heads, head_dim, N, N)
        attn = self.softmax(self.attn_mlp(attn_mlp_input)).reshape(B, self.num_heads, N, N)  # (B, num_heads, N, N)
        y = self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))
        
        return y
 

class TransformerEncoderLayer(nn.Module):
    
    
    def __init__(
        self,
        d_model: Optional[int],
        nhead: Optional[int],
        dim_feedforward: Optional[int] = 1024,
        dropout: Optional[float] = 0.1,
        bn: Optional[bool] = True
        ):
        """
        
        Params:
            d_model: the dimension of input features 
            
            nhead: the number of attention heads
            
            dim_feedforward: the dimension of the FFN hidden layer
        
            dropout: the dropout ratio
            
            bn: if True, add batch normalization
            
        """
        super().__init__()
        
        self.attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bn=bn
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout, inplace=True)
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        
    def forward(
        self,
        x: Optional[torch.Tensor],
        pos: Optional[torch.Tensor]
        ):
        y1 = self.layer_norm1(self.attn(x, x, x, pos) + x)
        y2 = self.layer_norm2(self.mlp(y1) + y1)
        
        return y2
    
    
class EncoderBlock(nn.Module):
    
    
    def __init__(
        self,
        sample_num: Optional[int],
        radius: Optional[float],
        neighbor_num: Optional[int],
        feature_dim: Optional[int],
        out_channels: Optional[list],
        nhead: Optional[int],
        bn: Optional[bool] = True
        ):
        """Feature extractor encoder block
        
        Params:
            sample_num: the number of sampled points
            
            radius: the radius of ball
            
            neighbor_num: the number of neighbor points in a group
        
            feature_dim: the dimension of input features 
            
            out_channels: the output channels of the convolutional layers in the set abstraction layer
            
            nhead: the number of attention heads
            
            bn: if True, add batch normalization
            
        """
        super().__init__()
        
        self.sample_and_group = SampleAndGroup(sample_num, radius, neighbor_num) 
        self.feat_aggregator = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 1),
            nn.BatchNorm1d(feature_dim) if bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, out_channels[-1], 1),
            nn.BatchNorm1d(out_channels[-1]) if bn else nn.Identity()
        )
        self.mini_pointnet = MiniPointNet(
            dim_in = feature_dim,
            out_channels = out_channels,
            bn=bn
        )
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, feature_dim),  
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, out_channels[-1])
        )
        self.transformer_encoder = TransformerEncoderLayer(
            d_model=out_channels[-1],
            nhead=nhead,
            bn=bn
        )
        self.cross_attn = MultiheadAttention(
            embed_dim=out_channels[-1],
            num_heads=nhead,
            dropout=0.1
        )
        self.mlp = nn.Sequential(
            nn.Conv1d(out_channels[-1], out_channels[-1], 1),
            nn.BatchNorm1d(out_channels[-1]) if bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        
    def forward(
        self,
        xyz: Optional[torch.Tensor],
        feat: Optional[torch.Tensor]
        ):
        """
        
        Params:
            xyz: the xyz coordinates of the input point cloud (B, N, 3)
            
            feat: the features of the input point cloud (B, N, C)
            
        Returns:
            the xyz coordinates and features of the sampled point cloud (B, M, 3) & (B, M, D)
            
        """
        sample_xyz, neighbor_feat = self.sample_and_group(xyz, feat)  # (B, M, 3) & (B, M, K, C)
        B, M, K, C = neighbor_feat.shape
        # get local features
        local_feat = self.mini_pointnet(neighbor_feat.permute(0, 3, 1, 2)).transpose(2, 1)  # (B, M, D, K)
        local_feat = local_feat.max(dim=-1, keepdim=False)[0]  # (B, M, D)
        # get transformer encoder's inputs
        transformer_input = self.feat_aggregator(neighbor_feat.reshape(B * M, K, C).transpose(2, 1))
        transformer_input = transformer_input.max(dim=-1, keepdim=False)[0].reshape(B, M, -1)  # (B, M, D)
        # get positional embeddings
        rel_pos = sample_xyz.unsqueeze(2) - sample_xyz.unsqueeze(1)  # (B, M, M, 3)
        pos = self.pos_encoder(rel_pos)  # (B, M, M, D)
        # get global features
        global_feat = self.transformer_encoder(transformer_input, pos)  # (B, M, D)
        # local features and global features fusion
        new_feat = self.cross_attn(local_feat, global_feat, global_feat, pos)  # (B, M, D)
        new_feat = global_feat + self.mlp(new_feat.transpose(2, 1)).transpose(2, 1)
        
        return sample_xyz, new_feat
    
    
class DecoderBlock(nn.Module):
    
    
    def __init__(
        self,
        dim_in: Optional[int],
        dim_out: Optional[int],
        bn: Optional[bool] = True,
        eps: Optional[float] = 1e-8
        ):
        """Feature extractor decoder block
        
        Params:
            dim_in: the dimension of input features
            
            dim_out: the dimension of output features
            
            bn: if True, add batch normalization
            
            eps: term added to prevent zero division
            
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(dim_in, dim_in, 1),
            nn.BatchNorm1d(dim_in) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(dim_in, dim_out, 1),
            nn.BatchNorm1d(dim_out) if bn else nn.Identity()
        )
        self.eps = eps
        
        
    def forward(
        self,
        xyz_down,
        xyz_up,
        feat_down,
        feat_up
        ):
        """
        
        Params:
            xyz_down: the xyz coordinates of point clouds with less points (B, M, 3)
            
            xyz_up: the xyz coordinates of point clouds with more points (B, N, 3)
            
            feat_down: the features of point clouds with less points (B, M, D)
            
            feat_up: the features of point clouds with more points (B, N, C)
            
        Returns:
            upsampled features (B, N, C)
            
        """
        B, N,_ = xyz_up.shape
        D = feat_down.shape[-1]
        dist = square_distance(xyz_up, xyz_down)  # (B, N, M)
        dist_recip = 1.0 / (dist + self.eps)
        weight = dist_recip / torch.sum(dist_recip, dim=2, keepdim=True)
        weight = weight.unsqueeze(-1).repeat(1, 1, 1, D)  # (B, N, M, D)
        feat_down = feat_down.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, M, D)
        feat_interpolate = torch.sum(feat_down * weight, dim=2)  # (B, N, D)
        if feat_up is not None:
            new_feat = torch.cat([feat_up, feat_interpolate], dim=-1)  # (B, N, C+D)
            
        else:
            new_feat = feat_interpolate  # (B, N, D)
            
        new_feat = self.mlp(new_feat.transpose(2, 1)).transpose(2, 1)
        
        return new_feat
    
    
class FeatureExtractor(nn.Module):
    
    
    def __init__(
        self,
        sample_num: Optional[list],
        radius: Optional[list],
        neighbor_num: Optional[int],
        out_channels: Optional[list],
        nhead: Optional[int],
        bn: Optional[bool] = True
        ):
        """Extract per-point features
        
        Params:
            sample_num: the number of sampled points
            
            radius: the radius of ball
            
            neighbor_num: the number of neighbor points in a group 
            
            out_channels: the output channels of the convolutional layers in the set abstraction layer
            
            nhead: the number of attention heads
            
            bn: if True, add batch normalization
            
        """
        super().__init__()
        
        self.encoder1 = EncoderBlock(
            sample_num=sample_num[0],
            radius=radius[0],
            neighbor_num=neighbor_num,
            feature_dim=3,
            out_channels=out_channels[0],
            nhead=nhead,
            bn=bn
        )
        self.encoder2 = EncoderBlock(
            sample_num=sample_num[1],
            radius=radius[1],
            neighbor_num=neighbor_num,
            feature_dim=out_channels[0][-1],
            out_channels=out_channels[1],
            nhead=nhead,
            bn=bn
        )
        self.encoder3 = EncoderBlock(
            sample_num=sample_num[2],
            radius=radius[2],
            neighbor_num=neighbor_num,
            feature_dim=out_channels[1][-1],
            out_channels=out_channels[2],
            nhead=nhead,
            bn=bn
        )
        self.decoder3 = DecoderBlock(
            dim_in=out_channels[2][-1] + out_channels[1][-1],
            dim_out=out_channels[1][-1],
            bn=bn
        )
        self.decoder2 = DecoderBlock(
            dim_in=out_channels[1][-1] + out_channels[0][-1],
            dim_out=out_channels[0][-1],
            bn=bn
        )
        self.decoder1 = DecoderBlock(
            dim_in=out_channels[0][-1],
            dim_out=out_channels[0][-1],
            bn=bn
        )
        
        
    def forward(self, xyz: Optional[torch.Tensor]):
        xyz_down_1, feat_down_1 = self.encoder1(xyz, xyz)  # 2048 -> 1024, 3 -> 128
        xyz_down_2, feat_down_2 = self.encoder2(xyz_down_1, feat_down_1)  # 1024 -> 512, 128 -> 256
        xyz_down_3, feat_down_3 = self.encoder3(xyz_down_2, feat_down_2)  # 512 -> 256, 256 -> 512
        feat_up_3 = self.decoder3(xyz_down_3, xyz_down_2, feat_down_3, feat_down_2)  # 256 -> 512, 512+256 -> 256
        feat_up_2 = self.decoder2(xyz_down_2, xyz_down_1, feat_up_3, feat_down_1)  # 512 -> 1024, 256+128 -> 128
        feat_up_1 = self.decoder1(xyz_down_1, xyz, feat_up_2, None)  # 1024 -> 2048, 128 -> 128
        
        return feat_up_1
    
    
class ContactPointSampler(nn.Module):
    
    
    def __init__(
        self,
        feature_dim: Optional[int],
        sample_num: Optional[int],
        is_train: Optional[bool] = True,
        bn: Optional[bool] = True
        ):
        """Sample contact points
        
        Params: 
            feature_dim: the dimension of extracted point features
        
            sample_num: the number of sampled points
            
            is_train: if True, adds soft projection. Otherwise, hard sampling
        
            bn: if True, add batch normalization
            
        """
        super().__init__()
        
        self.sample_num = sample_num
        self.is_train = is_train
        
        self.sampler = nn.Sequential(
            nn.Conv1d(feature_dim, 256),
            nn.BatchNorm1d(256) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, 3 * sample_num)
        )
        self.soft_proj = SoftProjection()
        
        
    def forward(
        self,
        xyz: Optional[torch.Tensor],
        feat: Optional[torch.Tensor]
        ):
        """
        
        Params:
            xyz: the xyz coordinates of the originally input point cloud
        
            feat: the extracted point features
            
        Returns: 
            One of the contact points, the temperature coefficient
            
        """
        feat = torch.max(feat, dim=1)[0]
        # sample contact points
        p1 = self.sampler(feat).reshape(-1, self.sample_num, 3)
        # soft projection (training)
        if self.is_train:
            p1, temp = self.soft_proj(xyz, p1)
        # hard sampling (validation or testing)
        else:
            _, _, p1 = knn_points(
                p1=p1, 
                p2=xyz, 
                norm=2, 
                K=1 
            )
            p1 = p1.squeeze()
            temp = 0.0
            
        return p1, temp    


class FeatureGrouper(nn.Module):
    
    
    def __init__(
        self,
        feature_dim: Optional[int],
        group_num: Optional[int] = 50,
        bn: Optional[bool] = True
        ):
        """Group point features around sampled points
        
        Params:
            group_num: the number of points in a group 
            
            feature_dim: the dimension of point features
        
            bn: If true, adds batch normalization to the regression layers
            
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.group_num = group_num
        
        self.mlp = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 1),
            nn.BatchNorm1d(feature_dim) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, 1),
            nn.BatchNorm1d(feature_dim) if bn else nn.Identity()
        )
        
        
    def forward(self, xyz, p1, feat):
        """
        
        Params:
            xyz: the xyz coordinates of the originally input point cloud
            
            p1: the sampled contact points
        
            feat: the extracted point features
            
        Returns:
            the grouped point features
            
        """
        _, neighbor_idx, _ = knn_points(
            p1=p1,
            p2=xyz,
            norm=2,
            K=self.group_num
        )
        feat_group = gather(feat, neighbor_idx)
        B, N, K, C = feat_group.shape
        feat_group = feat_group.reshape(-1, K, C).transpose(2, 1)
        feat_group = self.mlp(feat_group).transpose(2, 1).reshape(B, N, K, C)
        
        return feat_group
        

class GraspRegressor(nn.Module):
    
    
    def __init__(
        self,
        feature_dim: Optional[int],
        bn: Optional[bool] = True
        ):
        """Obtain grasp components
        
        Params:
            neighbor_num: the number of gathered neighbor points 
            
            feature_dim: the dimension of point features
        
            bn: If true, adds batch normalization to the regression layers
            
        """
        super().__init__()
        
        self.vector_regressor = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 1),
            nn.BatchNorm1d(feature_dim) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(feature_dim, 3, 1),
            nn.BatchNorm1d(3) if bn else nn.Identity()
        )
        self.width_regressor = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 1),
            nn.BatchNorm1d(feature_dim) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(feature_dim, 1, 1),
            nn.BatchNorm1d(1) if bn else nn.Identity()
        )
        self.angle_regressor = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 1),
            nn.BatchNorm1d(feature_dim) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(feature_dim, 1, 1),
            nn.BatchNorm1d(1) if bn else nn.Identity()
        )
        
        
    def forward(self, feat_group):
        """
        
        Params:
            feat_group: the grouped features (B, M, K, C)
            
        Returns:
            baseline vector, gripper width, pitch angle
            
        """
        # feature aggregation
        feat_agg = torch.max(feat_group, dim=-2).transpose(2, 1)  # (B, C, M)
        # obtain baseline vector, gripper width, and pitch angle
        vector = F.normalize(self.vector_regressor(feat_agg), dim=-1).transpose(2, 1)
        width = self.width_regressor(feat_agg).squeeze()
        angle = self.angle_regressor(feat_agg).squeeze()
        
        return vector, width, angle
    
    
class GraspClassifier(nn.Module):
    
    
    def __init__(
        self,
        feature_dim: Optional[int],
        bn: Optional[bool] = True
        ):
        """Classify the graspability of predicted grasps
        
        Params:
            feature_dim: the dimension of point features
        
            bn: If true, adds batch normalization to the regression layers
            
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Conv1d((feature_dim + 7), feature_dim, 1),
            nn.BatchNorm1d(feature_dim) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(feature_dim, 1, 1),
            nn.BatchNorm1d(1) if bn else nn.Identity()
        )
        
        
    def forward(self, grasp, feat_group):
        """
        Params:
            grasp: the 7-DoF grasps
        
            feat_group: the grouped features
            
        Returns:
            the grouped point features
            
        """
        # feature aggregation
        feat_agg = torch.max(feat_group, dim=-2)  # (B, M, C)
        feat = torch.cat([grasp, feat_agg], dim=-1)  # (B, M, (C+7))
        score = self.mlp(feat.transpose(2, 1)).squeeze()
        
        return score
        
    
# if __name__ == '__main__':
#     m = ContactTransGrasp([1024, 512, 256], [0.2, 0.3, 0.4], 5, [[128, 128], [256, 256], [512, 512]], 4)
#     x = torch.rand(2, 2048, 3)
#     print(m(x).shape)
