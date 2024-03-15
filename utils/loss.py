import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .point_utils import knn_points
from .transforms import pnt2quat, mat2quat

class TotalLoss(nn.Module):
    
    def __init__(
        self,
        gamma: Optional[float],
        alpha: Optional[float],
        beta: Optional[float],
        theta: Optional[float]
        ):
        '''
        Params:
            gamma: weight balance in sample loss
            
            alpha, beta: weight balance in regression loss
            
            theta: weight balance between sample loss and projection loss
        '''
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        
    def forward(
        self,
        xyz: Optional[torch.Tensor],
        sample_xyz: Optional[torch.Tensor],
        temp: Optional[nn.Parameter],
        grasp_pred: Optional[torch.Tensor],
        grasp_gt: Optional[torch.Tensor],
        class_pred: Optional[torch.Tensor],
        class_gt: Optional[torch.Tensor]):
        '''
        Params:
            xyz: xyz coordinates of original input points (B, N', 3)
            
            sample_xyz: xyz coordinates of sampled contact points (B, M, 3)
            
            temp: temperature coefficient
            
            grasp_pred: predicted grasp (B, M, 7)
            
            grasp_gt: ground-truth grasp (B, M, 4, 4)
            
            class_pred: predicted class (B, M)
            
            class_gt: ground-truth class (B, M)
            
        Returns:
            total loss
        '''
        sample_loss = self.get_sample_loss(xyz, sample_xyz)
        proj_loss = self.get_projection_loss(temp)
        reg_loss = self.get_regression_loss(grasp_pred, grasp_gt, class_gt)
        cls_loss = self.get_classification_loss(class_pred, class_gt)
        total_loss = self.theta * sample_loss + proj_loss + self.beta * reg_loss + cls_loss
        
        return total_loss
    
    def get_sample_loss(
        self,
        xyz: Optional[torch.Tensor],
        sample_xyz: Optional[torch.Tensor]
        ):
        dist_smp_org, _, _ = knn_points(p1=sample_xyz, p2=xyz, K=1)
        dist_org_smp, _, _ = knn_points(p1=xyz, p2=sample_xyz, K=1)
        loss_smp_org = torch.mean(dist_smp_org)
        loss_org_smp = torch.mean(dist_org_smp)
        loss_max_min, _ = torch.max(dist_smp_org.squeeze(-1), dim=-1)
        loss_max_min = torch.mean(loss_max_min)
        
        return loss_smp_org + loss_max_min + self.gamma * loss_org_smp
    
    def get_projection_loss(self, temp: Optional[nn.Parameter]):
        return temp ** 2
    
    def get_regression_loss(
        self,
        grasp_pred: Optional[torch.Tensor],
        grasp_gt: Optional[torch.Tensor],
        class_gt: Optional[torch.Tensor]
        ):
        center_pred, quat_pred = pnt2quat(grasp_pred)
        center_gt, quat_gt = mat2quat(grasp_gt)
        dist = F.pairwise_distance(center_pred.reshape(-1, 3), center_gt.reshape(-1, 3))
        trans_loss = torch.mean(dist * class_gt.reshape(-1))
        dot_product = (quat_pred.reshape(-1, 4) * quat_gt.reshape(-1, 4)).sum(dim=-1)
        rot_loss = torch.acos(torch.abs(dot_product))
        rot_loss = torch.mean(rot_loss * class_gt.reshape(-1))
        
        return trans_loss + self.alpha * rot_loss
    
    def get_classification_loss(
        self,
        class_pred: Optional[torch.Tensor],
        class_gt: Optional[torch.Tensor]
        ):
        return F.binary_cross_entropy(class_pred, class_gt)
