import torch
import torch.nn.functional as F
from typing import Optional

# modify from https://github.com/antoalli/L2G/blob/main/l2g_core/utils/grasp_utils.py
def pnt2quat(grasp: Optional[torch.Tensor]):
    '''
    Params:
        grasp: two contact points + pitch angle (B, M, 7)
        
    Returns:
        grasp center, quaternion and gripper width
    '''
    assert (grasp.shape[-1] == 7), "Invalid grasp size"
    
    p1 = grasp[:, :, 0: 3]
    p2 = grasp[:, :, 3: 6]
    angle = grasp[:, :, 6]
    center = (p1 + p2) / 2
    gripper_x = F.normalize(p1 - p2, dim=-2)
    z_axis = torch.tensor([0, 0, 1], requires_grad=True, dtype=torch.float32).to(p1.device)
    z_axis = z_axis.unsqueeze(0).unsqueeze(0).expand_as(gripper_x)
    tangent_y = F.normalize(torch.linalg.cross(gripper_x, z_axis), dim=-2)
    tangent_z = torch.linalg.cross(tangent_y, gripper_x)
    angle_sin = angle.sin().unsqueeze(-1).expand_as(tangent_z)
    angle_cos = angle.cos().unsqueeze(-1).expand_as(tangent_y)
    gripper_z = F.normalize(angle_cos * tangent_y + angle_sin * tangent_z, dim=-2)
    gripper_y = F.normalize(torch.linalg.cross(gripper_z, gripper_x), dim=-2)
    rot_mat = torch.stack([gripper_x, gripper_y, gripper_z], dim=-2).transpose(-2, -1)
    quat = rot2quat(rot_mat)
    
    return center, quat


def mat2quat(trans_mat: Optional[torch.Tensor]):
    '''
    Params:
        trans_mat: the grasp transformation matrix (B, M, 4, 4)
        
    Returns:
        grasp center and quaternion
    '''
    assert (trans_mat.shape[-2: ] == (4, 4)), "Invalid grasp size"
    
    center = trans_mat[:, :, 0: 3, 3]
    rot_mat = trans_mat[:, :, 0: 3, 0: 3]
    quat = rot2quat(rot_mat)
    
    return center, quat


def rot2quat(rot_mat: Optional[torch.Tensor]):
    '''
    Params:
        rot_mat: the rotation matrix (B, M, 3, 3)
        
    Returns:
        quaternion (B, M, 4)
    '''
    trace = rot_mat[:, :, 0, 0] + rot_mat[:, :, 1, 1] + rot_mat[:, :, 2, 2]
    w = torch.sqrt(trace + 1) / 2
    x = (rot_mat[:, :, 2, 1] - rot_mat[:, :, 1, 2]) / (4 * w)
    y = (rot_mat[:, :, 0, 2] - rot_mat[:, :, 2, 0]) / (4 * w)
    z = (rot_mat[:, :, 1, 0] - rot_mat[:, :, 0, 1]) / (4 * w)
    quat = torch.stack([w, x, y, z], dim=-1)
    quat = F.normalize(quat, dim=-1)
    
    return quat