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
    rotation_matrix = torch.stack([gripper_x, gripper_y, gripper_z], dim=-2).transpose(-2, -1)
    quaternion = matrix_to_quaternion(rotation_matrix)
    
    return center, quaternion

# copy from https://github.com/antoalli/L2G/blob/main/l2g_core/utils/grasp_utils.py
def mat2quat(matrix):
    fourWSquaredMinus1 = matrix[:, 0] + matrix[:, 4] + matrix[:, 8]
    fourXSquaredMinus1 = matrix[:, 0] - matrix[:, 4] - matrix[:, 8]
    fourYSquaredMinus1 = matrix[:, 4] - matrix[:, 0] - matrix[:, 8]
    fourZSquaredMinus1 = matrix[:, 8] - matrix[:, 0] - matrix[:, 4]
    temp = torch.stack([fourWSquaredMinus1, fourXSquaredMinus1, fourYSquaredMinus1, fourZSquaredMinus1], dim=1)
    fourBiggestSquaredMinus1, biggestIndex = torch.max(temp, dim=1)
    biggestVal = torch.sqrt(fourBiggestSquaredMinus1 + 1) * 0.5
    mult = 0.25 / biggestVal
    temp0 = biggestVal
    temp1 = (matrix[:, 7] - matrix[:, 5]) * mult
    temp2 = (matrix[:, 2] - matrix[:, 6]) * mult
    temp3 = (matrix[:, 3] - matrix[:, 1]) * mult
    temp4 = (matrix[:, 7] + matrix[:, 5]) * mult
    temp5 = (matrix[:, 2] + matrix[:, 6]) * mult
    temp6 = (matrix[:, 3] + matrix[:, 1]) * mult

    quaternion = torch.empty(size=[matrix.shape[0], 4], dtype=torch.float)
    quaternionBiggestIndex0 = torch.stack([temp0, temp1, temp2, temp3], dim=1)
    quaternionBiggestIndex1 = torch.stack([temp1, temp0, temp6, temp5], dim=1)
    quaternionBiggestIndex2 = torch.stack([temp2, temp6, temp0, temp4], dim=1)
    quaternionBiggestIndex3 = torch.stack([temp3, temp5, temp4, temp0], dim=1)

    # biggestIndex0Map = torch.ne(biggestIndex, 0)
    # biggestIndex0Map = biggestIndex0Map[:, None].expand_as(quaternion)
    biggestIndex1Map = torch.ne(biggestIndex, 1)
    biggestIndex1Map = biggestIndex1Map[:, None].expand_as(quaternion)
    biggestIndex2Map = torch.ne(biggestIndex, 2)
    biggestIndex2Map = biggestIndex2Map[:, None].expand_as(quaternion)
    biggestIndex3Map = torch.ne(biggestIndex, 3)
    biggestIndex3Map = biggestIndex3Map[:, None].expand_as(quaternion)

    quaternion = quaternionBiggestIndex0
    quaternion = quaternion.where(biggestIndex1Map, quaternionBiggestIndex1)
    quaternion = quaternion.where(biggestIndex2Map, quaternionBiggestIndex2)
    quaternion = quaternion.where(biggestIndex3Map, quaternionBiggestIndex3)
    
    return quaternion

# def mat2quat(trans_mat: Optional[torch.Tensor]):
#     '''
#     Params:
#         trans_mat: the grasp transformation matrix (B, M, 4, 4)
        
#     Returns:
#         grasp center and quaternion
#     '''
#     assert (trans_mat.shape[-2: ] == (4, 4)), "Invalid grasp size"
    
#     center = trans_mat[:, :, 0: 3, 3]
#     rotation_matrix = trans_mat[:, :, 0: 3, 0: 3]
#     quaternion = matrix_to_quaternion(rotation_matrix)
    
#     return center, quaternion

