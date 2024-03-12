import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points

from .transform import pnt2quat, mat2quat


class TotalLoss(nn.Module):
    
    
    def __init__(self, gamma, alpha, beta, theta):
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
        
    
    def forward(self, xyz, sample_xyz, temp, grasp_pred, grasp_gt, class_pred, class_gt):
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
        reg_loss = self.get_regression_loss(grasp_pred, grasp_gt)
        cls_loss = self.get_classification_loss(class_pred, class_gt)
        total_loss = self.theta * sample_loss + proj_loss + self.beta * reg_loss + cls_loss
        
        return total_loss
    
    
    def get_sample_loss(self, xyz, sample_xyz):
        dist_smp_org, _, _ = knn_points(p1=sample_xyz, p2=xyz, norm=2, K=1)
        dist_org_smp, _, _ = knn_points(p1=xyz, p2=sample_xyz, norm=2, K=1)
        loss_smp_org = torch.mean(dist_smp_org)
        loss_org_smp = torch.mean(dist_org_smp)
        loss_max_min, _ = torch.max(dist_smp_org.squeeze(-1), dim=-1)
        loss_max_min = torch.mean(loss_max_min)
        
        return loss_smp_org + loss_max_min + self.gamma * loss_org_smp
    
    
    def get_projection_loss(self, temp):
        if temp is None:
            return 0.0
        else:
            return temp ** 2
    
    
    def get_regression_loss(self, grasp_pred, grasp_gt):
        center_pred, quat_pred = pnt2quat(grasp_pred)
        center_gt, quat_gt = mat2quat(grasp_gt)
        trans_loss = F.pairwise_distance(center_pred.reshape(-1, 3), center_gt.reshape(-1, 3))
        trans_loss = torch.mean(trans_loss)
        rot_loss = 1 - (quat_pred.reshape(-1, 4) * quat_gt.reshape(-1, 4)).sum(dim=-1)
        rot_loss = torch.mean(rot_loss)
        
        return trans_loss + self.alpha * rot_loss
    
        
    def get_classification_loss(self, class_pred, class_gt):
        
        return F.binary_cross_entropy(class_pred, class_gt)
