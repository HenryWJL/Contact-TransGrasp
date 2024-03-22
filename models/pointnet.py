import torch
import torch.nn as nn
from typing import Optional


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