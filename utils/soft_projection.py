import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pytorch3d.ops import knn_points


class SoftProjection(nn.Module):
    
    
    def __init__(
        self, 
        neighbor_num: Optional[int] = 10, 
        init_temp: Optional[float] = 1.0, 
        is_temp_trainable: Optional[bool] = True
        ):
        """
        Params:
            neighbor_num: the number of neighbor points
            
            init_temp: the initial value of the temperature coefficient
            
            is_temp_trainable: if True, set the temperature coefficient trainable
        """
        super().__init__()
        
        self.neighbor_num = neighbor_num
        self.temp = nn.Parameter(
            torch.tensor(
                init_temp,
                requires_grad=is_temp_trainable,
                dtype=torch.float32,
            )
        )
        
        
    def forward(
        self,
        xyz: Optional[torch.Tensor],
        sample_xyz: Optional[torch.Tensor]
        ):
        """
        Params:
            xyz: the xyz coordinates of the originally input point cloud
            
            sample_xyz: the xyz coordinates of the sampled points
            
        Returns:
            the softly projected sampled points
        """
        squared_dist, _, neighbor_xyz = knn_points(
            p1=sample_xyz, 
            p2=xyz, 
            norm=2, 
            K=self.neighbor_num, 
            return_nn=True
        )
        weight = F.softmax(- squared_dist / self.temp.to(xyz.device) ** 2, dim=-1)
        weight = weight.unsqueeze(-1).expand_as(neighbor_xyz)
        proj_xyz = (neighbor_xyz * weight).sum(dim=-2)
        
        return proj_xyz, self.temp
    
