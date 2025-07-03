import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Optional

from infonet.model.attention_block import CrossAttentionBlock


class Decoder(pl.LightningModule):

    def __init__(
        self,
        latent_dim: int,
        q_dim: int,
        heads: int = 4,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        projection_dim: Optional[int] = None,
        cross_attn_widening_factor: int = 1,
        dropout: float = 0.0
        ):

        super().__init__()
        
        # 保存超参数
        self.save_hyperparameters()
        
        self.cross_attn = CrossAttentionBlock(
            q_dim=q_dim,
            kv_dim=latent_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            heads=heads,
            widening_factor=cross_attn_widening_factor,
            dropout=dropout
        )
        
        if projection_dim is not None:
            self.projection = nn.Linear(q_dim, projection_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x_q: torch.Tensor, latents: torch.Tensor, query_mask: Optional[torch.Tensor] = None):
        
        # Lightning自动设备管理：所有输入张量都会在正确的设备上
        output = self.cross_attn(
            x_q=x_q,
            x_kv=latents,
            attention_mask=query_mask
        )

        return self.projection(output)
    
    def training_step(self, batch, batch_idx):
        """如果需要单独训练decoder"""
        # 通常不会被调用，因为decoder作为组件使用
        pass
    
    def configure_optimizers(self):
        """如果需要单独训练decoder"""
        return torch.optim.Adam(self.parameters(), lr=1e-3)