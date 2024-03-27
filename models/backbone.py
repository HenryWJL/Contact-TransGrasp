import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from utils.point_utils import (
    SampleAndGroup,
    SoftProjection,
    gather,
    pc_normalize,
    square_distance,
    knn_points
)
from .pointnet import MiniPointNet
from .transformer import MultiheadAttention, TransformerEncoder


class EncoderBlock(nn.Module):
    
    def __init__(
        self,
        sample_num: Optional[int],
        radius: Optional[float],
        neighbor_num: Optional[int],
        feature_dim: Optional[int],
        out_channels: Optional[list],
        nhead: Optional[int],
        num_layers: Optional[int],
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
            
            num_layers: the number of Transformer encoder layers
            
            bn: if True, add batch normalization
            
        """
        super().__init__()
        
        self.sample_and_group = SampleAndGroup(sample_num, radius, neighbor_num) 
        self.feat_aggregator = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 1),
            nn.BatchNorm1d(feature_dim) if bn else nn.Identity(),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Linear(feature_dim, out_channels[-1])
        )
        self.transformer_encoder = TransformerEncoder(
            d_model=out_channels[-1],
            nhead=nhead,
            num_layers=num_layers
        )
        self.cross_attn = MultiheadAttention(
            embed_dim=out_channels[-1],
            num_heads=nhead,
            dropout=0.1
        )
        self.mlp = nn.Sequential(
            nn.Conv1d(out_channels[-1], out_channels[-1], 1),
            nn.BatchNorm1d(out_channels[-1]) if bn else nn.Identity(),
            nn.ReLU()
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
        bn: Optional[bool] = True
        ):
        """Feature extractor decoder block
        
        Params:
            dim_in: the dimension of input features
            
            dim_out: the dimension of output features
            
            bn: if True, add batch normalization
            
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(dim_in, dim_in, 1),
            nn.BatchNorm1d(dim_in) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(dim_in, dim_out, 1),
            nn.BatchNorm1d(dim_out) if bn else nn.Identity()
        )
        
    def forward(
        self,
        xyz_down: Optional[torch.Tensor],
        xyz_up: Optional[torch.Tensor],
        feat_down: Optional[torch.Tensor],
        feat_up: Optional[torch.Tensor]
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
        dist_recip = 1.0 / (dist + 1e-8)
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
        neighbor_num: Optional[list],
        out_channels: Optional[list],
        nhead: Optional[int],
        num_layers: Optional[int]
        ):
        """Extract per-point features
        
        Params:
            sample_num: the number of sampled points for furthest point sampling
            
            radius: the radius of ball for ball query
            
            neighbor_num: the number of neighbor points for ball query 
            
            out_channels: the output channels of the convolutional layers in the set abstraction layer
            
            nhead: the number of attention heads
            
            num_layers: the number of Transformer encoder layers
            
            bn: if True, add batch normalization
            
        """
        super().__init__()
        
        self.encoder1 = EncoderBlock(
            sample_num=sample_num[0],
            radius=radius[0],
            neighbor_num=neighbor_num[0],
            feature_dim=3,
            out_channels=out_channels[0],
            nhead=nhead,
            num_layers=num_layers
        )
        self.encoder2 = EncoderBlock(
            sample_num=sample_num[1],
            radius=radius[1],
            neighbor_num=neighbor_num[1],
            feature_dim=out_channels[0][-1],
            out_channels=out_channels[1],
            nhead=nhead,
            num_layers=num_layers
        )
        self.encoder3 = EncoderBlock(
            sample_num=sample_num[2],
            radius=radius[2],
            neighbor_num=neighbor_num[2],
            feature_dim=out_channels[1][-1],
            out_channels=out_channels[2],
            nhead=nhead,
            num_layers=num_layers
        )
        self.decoder3 = DecoderBlock(
            dim_in=out_channels[2][-1] + out_channels[1][-1],
            dim_out=out_channels[1][-1]
        )
        self.decoder2 = DecoderBlock(
            dim_in=out_channels[1][-1] + out_channels[0][-1],
            dim_out=out_channels[0][-1]
        )
        self.decoder1 = DecoderBlock(
            dim_in=out_channels[0][-1],
            dim_out=out_channels[0][-1]
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
        bn: Optional[bool] = True
        ):
        """Sample contact points
        
        Params: 
            feature_dim: the dimension of extracted point features
        
            sample_num: the number of sampled contact points
        
            bn: if True, add batch normalization
            
        """
        super().__init__()
        
        self.sample_num = sample_num
        
        self.sampler = nn.Sequential(
            nn.Linear(feature_dim, 256),
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
        if self.training:
            p1, temp = self.soft_proj(xyz, p1)
        # hard sampling (validation or testing)
        else:
            _, _, p1 = knn_points(
                p1=p1, 
                p2=xyz, 
                K=1
            )
            p1 = p1.squeeze()
            temp = 0.0
            
        return p1, temp    


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
        feat_agg = torch.max(feat_group, dim=-2)[0].transpose(2, 1)  # (B, C, M)
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
            nn.BatchNorm1d(1) if bn else nn.Identity(),
            nn.Sigmoid()
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
        feat_agg = torch.max(feat_group, dim=-2)[0]  # (B, M, C)
        feat = torch.cat([grasp, feat_agg], dim=-1)  # (B, M, (C+7))
        score = self.mlp(feat.transpose(2, 1)).squeeze(-2)
        
        return score


class ContactTransGrasp(nn.Module):
    
    def __init__(
        self,
        sample_num: Optional[list],
        radius: Optional[list],
        neighbor_num: Optional[list],
        out_channels: Optional[list],
        nhead: Optional[int],
        num_layers: Optional[int],
        point_num: Optional[int]
        ):
        """
        
        Params:
            sample_num: the number of sampled points for furthest point sampling
            
            radius: the radius of ball for ball query
            
            neighbor_num: the number of neighbor points for ball query 
            
            out_channels: the output channels of the convolutional layers in the set abstraction layer
            
            nhead: the number of attention heads
            
            num_layers: the number of Transformer encoder layers
            
            point_num: the number of sampled contact points

        """
        super().__init__()
        
        self.feature_dim = out_channels[0][-1]
        
        self.feat_extractor = FeatureExtractor(
            sample_num=sample_num,
            radius=radius,
            neighbor_num=neighbor_num,
            out_channels=out_channels,
            nhead=nhead,
            num_layers=num_layers
        )
        self.sampler = ContactPointSampler(
            feature_dim=self.feature_dim,
            sample_num=point_num
        )
        self.feat_grouper = FeatureGrouper(self.feature_dim)
        self.regressor = GraspRegressor(self.feature_dim)
        self.classifier = GraspClassifier(self.feature_dim)
    
    def weights_init(self, init_type='normal'):
        
        def init_func(module):
            classname = module.__class__.__name__
            if (classname.find('Conv') ==0 or classname.find('Linear') == 0) and hasattr(module, 'weight'):
                if init_type == 'normal':
                    nn.init.normal_(module.weight.data, 0.0, 0.02)
                    
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(module.weight.data, gain=1.414)
                    
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                    
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(module.weight.data, gain=1.414)
                    
                else:
                    assert 0, f"Unsupported initialization: {init_type}"
                    
            elif classname.find('BatchNorm1d') ==0 or classname.find('LayerNorm') ==0:
                nn.init.normal_(module.weight.data, 0.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)
                
        self.apply(init_func)
    
    def forward(self, xyz: Optional[torch.Tensor]):
        # get normalize points
        xyz, mean, std = pc_normalize(xyz)
        # extract point features
        feat = self.feat_extractor(xyz)
        # get grasps
        p1, temp = self.sampler(xyz, feat)
        feat_group = self.feat_grouper(xyz, p1, feat)
        vector, width, angle = self.regressor(feat_group)  
        # get the second contact points
        p2 = p1 - vector * width.unsqueeze(-1).expand_as(vector)
        # get grasp quality
        grasp = torch.cat((p1, p2, angle.unsqueeze(-1)), dim=-1) 
        score = self.classifier(grasp, feat_group)
        # get unnormalized grasps
        mean = mean.unsqueeze(-2).expand_as(p1)
        std = std.unsqueeze(-1).unsqueeze(-1).expand_as(p1)
        p1 = p1 * std + mean
        p2 = p2 * std + mean
        grasp = torch.cat((p1, p2, angle.unsqueeze(-1)), dim=-1)
        
        return p1, temp, grasp, score
    

    
# if __name__ == '__main__':
#     m = ContactTransGrasp(
#         [128, 64, 32],
#         [0.2, 0.3, 0.4],
#         [5, 10, 15],
#         [[128, 128], [256, 256], [512, 512]],
#         4,
#         4,
#         64
#     )
#     x = torch.rand(2, 256, 3)
#     m.eval()
#     m(x)
