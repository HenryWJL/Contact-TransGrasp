import torch
import torch.nn as nn
from typing import Optional


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
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.head_dim,
                out_channels=1,
                kernel_size=1
            ),
            nn.BatchNorm2d(1) if bn else nn.Identity()
        )
        self.softmax = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
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
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout)
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


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        d_model: Optional[int],
        nhead: Optional[int],
        num_layers: Optional[int],
        dropout: Optional[float] = 0.1
        ):
        """
        
        Params:
            d_model: the dimension of features in the input 
            
            nhead: the number of attention heads
            
            num_layers: the number of Transformer encoder layers 
            
        """
        super().__init__()
        
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout
            )
            for i in range(num_layers)]
        )
        
    def forward(
        self,
        x: Optional[torch.Tensor],
        pos: Optional[torch.Tensor]
        ):
        for _, encoder_layer in enumerate(self.encoder):
            x = encoder_layer(x, pos)
            
        return x