import torch
import pytorch_lightning as pl
import yaml
import matplotlib.pyplot as plt
import numpy as np
from infonet.model.decoder_pl import Decoder
from infonet.model.encoder_pl import Encoder
from infonet.model.infonet_pl import infonet
from infonet.model.query_pl import Query_Gen_transformer
from scipy.stats import rankdata

class InfoNetLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        encoder = Encoder(
            input_dim=config['model']['input_dim'],
            latent_num=config['model']['latent_num'],
            latent_dim=config['model']['latent_dim'],
            cross_attn_heads=config['model']['cross_attn_heads'],
            self_attn_heads=config['model']['self_attn_heads'],
            num_self_attn_per_block=config['model']['num_self_attn_per_block'],
            num_self_attn_blocks=config['model']['num_self_attn_blocks']
        )

        decoder = Decoder(
            q_dim=config['model']['decoder_query_dim'],
            latent_dim=config['model']['latent_dim'],
        )

        query_gen = Query_Gen_transformer(
            input_dim=config['model']['input_dim'],
            dim=config['model']['decoder_query_dim']
        )
        
        self.model = infonet(
            encoder=encoder,
            decoder=decoder,
            query_gen=query_gen,
            decoder_query_dim=config['model']['decoder_query_dim']
        )
    
    def forward(self, x):
        return self.model(x)
    
    def estimate_mi(self, x, y):
        """自动设备管理的MI估计"""
        self.eval()
        x = rankdata(x)/len(x)
        y = rankdata(y)/len(y)
        # 使用self.device自动获取模型所在设备
        batch = torch.stack(
            (torch.tensor(x, dtype=torch.float32), 
             torch.tensor(y, dtype=torch.float32)), 
            dim=1
        ).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mi_lb = self(batch)
        return mi_lb
    
    def infer_batch(self, batch):
        """自动设备管理的批量推理"""
        self.eval()
        batch = torch.tensor(batch, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            mi_lb = self(batch)
            MI = torch.mean(mi_lb)
        return MI.cpu().numpy()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_lightning_model(config_path, checkpoint_path):
    """加载普通PyTorch权重到Lightning模型"""
    config = load_config(config_path)
    
    # 创建Lightning模型实例
    model = InfoNetLightning(config)
    
    # 加载普通PyTorch权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 如果checkpoint是state_dict格式
    if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint:
        # 直接是state_dict
        state_dict = checkpoint
    else:
        # 是完整的checkpoint，提取state_dict
        state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 由于Lightning模型结构是model.model，需要调整key
    lightning_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            lightning_state_dict[key] = value
        else:
            lightning_state_dict[f'model.{key}'] = value
    
    # 加载权重
    model.load_state_dict(lightning_state_dict, strict=False)
    model.eval()
    
    return model

def load_lightning_model_alternative(config_path, checkpoint_path):
    """备用加载方法：直接创建原始模型并包装"""
    from infonet.infer import create_model, load_config as orig_load_config
    
    # 使用原始方法创建模型
    config = orig_load_config(config_path)
    original_model = create_model(config)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    original_model.load_state_dict(checkpoint)
    
    # 包装成Lightning模型
    lightning_model = InfoNetLightning(load_config(config_path))
    lightning_model.model = original_model
    lightning_model.eval()
    
    return lightning_model

def compute_smi_mean(sample_x, sample_y, model, proj_num, seq_len, batchsize):
    dx = sample_x.shape[1]
    dy = sample_y.shape[1]
    results = []
    
    for i in range(proj_num//batchsize):
        batch = np.zeros((batchsize, seq_len, 2))
        for j in range(batchsize):
            theta = np.random.randn(dx)
            phi = np.random.randn(dy)
            x_proj = np.dot(sample_x, theta)
            y_proj = np.dot(sample_y, phi)
            x_proj = rankdata(x_proj)/seq_len
            y_proj = rankdata(y_proj)/seq_len
            xy = np.column_stack((x_proj, y_proj))
            batch[j, :, :] = xy
        infer1 = model.infer_batch(batch)
        results.append(infer1)
    
    return np.mean(np.array(results))