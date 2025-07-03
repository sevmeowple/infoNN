from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from infonet.model.encoder_pl import Encoder
from infonet.model.decoder_pl import Decoder
from infonet.model.query_pl import Query_Gen_transformer
from infonet.model.gauss_mild_pl import GaussConv
from infonet.model.util import mutual_information
import torch.nn.functional as F
import numpy as np


class infonet(pl.LightningModule):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        query_gen: Query_Gen_transformer,
        decoder_query_dim: int
    ):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.query_gen = query_gen
        self.mild = GaussConv(size=15, nsig=3, channels=1)
        
        # 保存超参数，便于Lightning管理
        self.save_hyperparameters(ignore=['encoder', 'decoder', 'query_gen'])

    def forward(
        self,
        inputs: Optional[torch.Tensor],
        query: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None
    ):
        # 自动设备管理：inputs会自动在正确的设备上
        latents = self.encoder(inputs, input_mask)
        query = self.query_gen(inputs)
        
        outputs = self.decoder(
            x_q=query,
            latents=latents,
            query_mask=query_mask
        )
        
        outputs = outputs.unsqueeze(1)
        outputs = self.mild(outputs)
        outputs = outputs.squeeze(1)
        
        mi_lb = mutual_information(inputs, outputs)

        return mi_lb
    
    def training_step(self, batch, batch_idx):
        """训练步骤 - 可以根据具体需求实现"""
        inputs = batch
        mi_lb = self.forward(inputs)
        
        # 假设训练目标是最大化互信息下界
        loss = -mi_lb.mean()  # 负号因为要最大化MI
        
        # Lightning自动日志记录
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mi', mi_lb.mean(), on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        inputs = batch
        mi_lb = self.forward(inputs)
        loss = -mi_lb.mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mi', mi_lb.mean(), on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """配置优化器 - 可以根据需求自定义"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """预测步骤 - 用于推理"""
        return self.forward(batch)


# 为了保持与原有infer.py的兼容性，提供一个工厂函数
def create_infonet_light(encoder, decoder, query_gen, decoder_query_dim):
    """创建Lightning版本的infonet，保持接口兼容性"""
    return infonet(
        encoder=encoder,
        decoder=decoder,
        query_gen=query_gen,
        decoder_query_dim=decoder_query_dim
    )