import torch
import torch.nn as nn
from mamba_ssm import Mamba

class BiMambaSmoother(nn.Module):
    def __init__(self, input_dim=266, d_model=256, n_layers=4):
        """
        双向 Mamba 平滑器
        d_model: 隐层维度 (256 足够，Mamba 参数效率很高)
        n_layers: 堆叠层数
        """
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm module is missing.")

        self.input_dim = input_dim
        
        # 1. 维度投影
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. 正向 Mamba 堆叠 (Forward)
        self.layers_fwd = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        # 3. 反向 Mamba 堆叠 (Backward)
        # 使用独立的权重来捕捉“未来->过去”的依赖
        self.layers_bwd = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        # 4. 归一化与融合
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        residual = x
        
        # Embedding
        x_embed = self.embedding(x) # (B, L, d_model)
        
        # --- Branch 1: Forward ---
        x_f = x_embed
        for layer in self.layers_fwd:
            x_f = layer(x_f)
            
        # --- Branch 2: Backward ---
        # 翻转时间轴 -> Mamba -> 再翻转回来
        x_b = torch.flip(x_embed, dims=[1])
        for layer in self.layers_bwd:
            x_b = layer(x_b)
        x_b = torch.flip(x_b, dims=[1])
        
        # --- Fusion ---
        # 简单相加，利用 LayerNorm 稳住分布
        x_merged = self.norm(x_f + x_b)
        
        # Output + Residual Connection
        correction = self.output_head(x_merged)
        return residual + correction