import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from infonet.model.decoder import Decoder
from infonet.model.encoder import Encoder
from infonet.model.infonet import infonet
from infonet.model.query import Query_Gen_transformer
from scipy.stats import rankdata

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_model(config):
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
    
    model = infonet(
        encoder=encoder,
        decoder=decoder,
        query_gen=query_gen,
        decoder_query_dim=config['model']['decoder_query_dim']
    ).to(device)
    
    return model

def load_model(config_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path)
    model = create_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def estimate_mi(model, x, y):
    ## x and y are 1 dimensional sequences
    model.eval()
    x = rankdata(x)/len(x)
    y = rankdata(y)/len(y)
    batch = torch.stack((torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)), dim=1).unsqueeze(0).to(device) ## batch has shape [1, sequence length, 2]
    with torch.no_grad():
        mi_lb = model(batch)
    return mi_lb

def infer(model, batch):
    ### batch has shape [batchsize, seq_len, 2]
    model.eval()
    batch = torch.tensor(batch, dtype=torch.float32, device=device)
    with torch.no_grad():

        mi_lb = model(batch)
        MI = torch.mean(mi_lb)

    return MI.cpu().numpy()

def compute_smi_mean(sample_x, sample_y, model, proj_num, seq_len, batchsize):
    ## we use sliced mutual information to estimate high dimensional correlation
    ## proj_num means the number of random projections you want to use, the larger the more accuracy but higher time cost
    ## seq_len means the number of samples used for the estimation
    ## batchsize means the number of one-dimensional pairs estimate at one time, this only influences the estimation speed
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
        infer1 = infer(model, batch)
        mean_infer1 = np.mean(infer1)
        results.append(mean_infer1)

    return np.mean(np.array(results))



def compute_smi_mean_gpu(sample_x, sample_y, model, proj_num, seq_len, batchsize):
    """
    GPU版本的compute_smi_mean - 完全在GPU上计算
    """
    device = next(model.parameters()).device
    
    # 确保输入数据在GPU上
    if isinstance(sample_x, np.ndarray):
        sample_x = torch.from_numpy(sample_x).float().to(device)
    elif isinstance(sample_x, torch.Tensor):
        sample_x = sample_x.to(device).float()
    
    if isinstance(sample_y, np.ndarray):
        sample_y = torch.from_numpy(sample_y).float().to(device)
    elif isinstance(sample_y, torch.Tensor):
        sample_y = sample_y.to(device).float()
    
    # 如果数据超过seq_len，截取前seq_len个样本
    if sample_x.shape[0] > seq_len:
        sample_x = sample_x[:seq_len]
        sample_y = sample_y[:seq_len]
    
    dx = sample_x.shape[1] if sample_x.dim() > 1 else 1
    dy = sample_y.shape[1] if sample_y.dim() > 1 else 1
    
    results = []
    
    with torch.no_grad():
        # 批量处理投影
        num_batches = proj_num // batchsize
        for _ in range(num_batches):
            # 在GPU上生成随机投影矩阵
            if dx > 1:
                theta = torch.randn(dx, batchsize, device=device)  # 修复：调整维度顺序
                x_proj = torch.matmul(sample_x, theta)  # [seq_len, dx] @ [dx, batchsize] = [seq_len, batchsize]
                x_proj = x_proj.T  # 转置为 [batchsize, seq_len]
            else:
                x_proj = sample_x.unsqueeze(0).repeat(batchsize, 1)
            
            if dy > 1:
                phi = torch.randn(dy, batchsize, device=device)  # 修复：调整维度顺序
                y_proj = torch.matmul(sample_y, phi)  # [seq_len, dy] @ [dy, batchsize] = [seq_len, batchsize]
                y_proj = y_proj.T  # 转置为 [batchsize, seq_len]
            else:
                y_proj = sample_y.unsqueeze(0).repeat(batchsize, 1)
            
            # GPU上的排序归一化
            x_ranked = torch.argsort(torch.argsort(x_proj, dim=1), dim=1).float() / x_proj.shape[1]
            y_ranked = torch.argsort(torch.argsort(y_proj, dim=1), dim=1).float() / y_proj.shape[1]
            
            # 构建批次数据 [batchsize, seq_len, 2]
            batch = torch.stack([x_ranked, y_ranked], dim=2)
            
            # 模型推理
            mi_lb = model(batch)
            results.append(mi_lb.mean().item())
    
    return np.mean(results)

def example_d_1():
    seq_len = 4781
    results = []
    real_MIs = []
    
    for rou in np.arange(-0.9, 1, 0.1):
        x, y = np.random.multivariate_normal(mean=[0,0], cov=[[1,rou],[rou,1]], size=seq_len).T
        x = rankdata(x)/seq_len #### important, data preprocessing is needed, using rankdata(x)/seq_len to map x and y to [0,1]
        y = rankdata(y)/seq_len
        result = estimate_mi(model, x, y).squeeze().cpu().numpy()
        real_MI = -np.log(1-rou**2)/2
        real_MIs.append(real_MI)
        results.append(result)
        print("estimate mutual information is: ", result, "real MI is ", real_MI  )

def example_highd():
    d = 10
    mu = np.zeros(d)
    sigma = np.eye(d)
    sample_x = np.random.multivariate_normal(mu, sigma, 2000)
    sample_y = np.random.multivariate_normal(mu, sigma, 2000)
    result = compute_smi_mean(sample_x, sample_y, model, seq_len=2000, proj_num=1024, batchsize=32)
    print(f"result is {result}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Load model from config and checkpoint')
    parser.add_argument('--config', type=str, required=False, default='configs/config.yaml', help='Path to the config file')
    parser.add_argument('--checkpoint', type=str, required=False, default="saved/uniform/model_5000_32_1000-720--0.16.pt", help='Path to the model checkpoint')
    
    args = parser.parse_args()
    
    model = load_model(args.config, args.checkpoint)
    print("Model loaded successfully")

    example_d_1()
    #example_highd()
