from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from infonet.model.attention_block import CrossAttentionBlock, SelfAttentionBlock


class Encoder(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        latent_num: int,
        latent_dim: int,
        cross_attn_heads: int = 4,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        cross_attn_widening_factor: int = 1,
        self_attn_heads: int = 4,
        self_attn_widening_factor: int = 1,
        num_self_attn_per_block: int = 6,
        num_self_attn_blocks: int = 1,
        dropout: float = 0.0,
        ):

        super().__init__()
        self.num_self_attn_blocks = num_self_attn_blocks
        
        # 保存超参数
        self.save_hyperparameters()

        self.latents = nn.Parameter(torch.randn(latent_num, latent_dim))

        self.cross_attn_block = CrossAttentionBlock(
            q_dim=latent_dim,
            kv_dim=input_dim,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            heads=cross_attn_heads,
            dropout=dropout,
            widening_factor=cross_attn_widening_factor
        )

        self.self_attn_block = nn.ModuleList([
            SelfAttentionBlock(
                q_dim=latent_dim,
                qk_out_dim=qk_out_dim,
                v_out_dim=v_out_dim,
                heads=self_attn_heads,
                widening_factor=self_attn_widening_factor,
                dropout=dropout
            ) for _ in range(num_self_attn_per_block)
        ])
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        '''
            Args:
                x: (B, M, C)
                mask: (B, M)
        '''

        b, *_= x.shape

        # 自动设备管理：latents自动移动到正确设备
        latents = self.latents.repeat(b, 1, 1)

        latents = self.cross_attn_block(
            x_q=latents,
            x_kv=x,
            attention_mask=attention_mask
        )
        for _ in range(self.num_self_attn_blocks):
            for self_attn_layer in self.self_attn_block:
                latents = self_attn_layer(latents)
        
        return latents
    
    def training_step(self, batch, batch_idx):
        """如果需要单独训练encoder"""
        # 这个方法通常不会被调用，因为encoder作为组件使用
        pass
    
    def configure_optimizers(self):
        """如果需要单独训练encoder"""
        return torch.optim.Adam(self.parameters(), lr=1e-3)