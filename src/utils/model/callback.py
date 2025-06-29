import time
import threading
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from livelossplot import PlotLosses

import csv
import os
from datetime import datetime
import pandas as pd


class DisplayManager:
    """显示管理器 - 协调多个回调的输出显示"""
    
    def __init__(self):
        self.is_plotting = False
        self.pending_messages = []
        self.lock = threading.Lock()
        
    @contextmanager
    def exclusive_display(self, callback_name=""):
        """独占显示上下文管理器"""
        with self.lock:
            self.is_plotting = True
            print(f"\n📊 {callback_name} 显示开始...")
            try:
                yield
            finally:
                print(f"✅ {callback_name} 显示完成\n")
                self.is_plotting = False
                self._flush_pending_messages()
    
    def safe_print(self, message, force=False):
        """安全打印 - 如果正在绘图则缓存消息"""
        with self.lock:
            if self.is_plotting and not force:
                self.pending_messages.append(message)
            else:
                print(message)
    
    def _flush_pending_messages(self):
        """输出缓存的消息"""
        if self.pending_messages:
            print("📋 缓存的消息:")
            for msg in self.pending_messages:
                print(msg)
            self.pending_messages.clear()


# 全局显示管理器实例
_display_manager = DisplayManager()

class FilePersistentMutualInformationCallback(Callback):
    """文件持久化MI回调 - 基于InfoNet官方实现"""
    
    def __init__(self, mi_model, eval_loader, eval_every_n_epochs=1, 
                 save_dir="./mi_results", experiment_name=None,
                 seq_len=2000, proj_num=1024, batchsize=32):
        super().__init__()
        self.mi_model = mi_model  # InfoNet模型
        self.eval_loader = eval_loader
        self.eval_every_n_epochs = eval_every_n_epochs
        
        # InfoNet官方超参数
        self.seq_len = seq_len  # 官方默认2000样本用于估计
        self.proj_num = proj_num  # 官方默认1024个随机投影
        self.batchsize = batchsize  # 官方默认32个一维对同时估计
        
        # 文件存储配置
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成实验标识
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"infonet_mi_experiment_{timestamp}"
        self.experiment_name = experiment_name
        
        # CSV文件路径
        self.csv_path = os.path.join(save_dir, f"{experiment_name}_mi_results.csv")
        self.summary_path = os.path.join(save_dir, f"{experiment_name}_summary.txt")
        
        # 初始化CSV文件
        self._init_csv_file()
        
        # 显示控制
        self.show_summary_every_n_epochs = 5
        
        # 验证InfoNet模型
        if self.mi_model is None:
            raise ValueError("❌ 必须提供有效的InfoNet模型")
        
        print(f"✅ MI结果将保存到: {self.csv_path}")
        print(f"📊 汇总报告: {self.summary_path}")
        print(f"🔬 使用InfoNet进行MI估计 (seq_len={seq_len}, proj_num={proj_num})")
    
    def _init_csv_file(self):
        """初始化CSV文件"""
        fieldnames = ['epoch', 'timestamp', 'layer', 'I_XT', 'I_TY', 'status', 'method', 'sample_size', 'proj_num']
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    def on_train_epoch_end(self, trainer, pl_module):
        """MI分析 + 文件存储"""
        current_epoch = trainer.current_epoch
        
        if current_epoch % self.eval_every_n_epochs == 0:
            print(f"🔍 Epoch {current_epoch} - InfoNet MI分析中...")
            
            # 执行MI分析
            results = self._perform_infonet_mi_analysis(trainer, pl_module, current_epoch)
            
            # 保存到CSV
            self._save_results_to_csv(current_epoch, results)
            
            # 简化的即时反馈
            if results:
                valid_results = [r for r in results if r.get('I_XT') is not None]
                if valid_results:
                    avg_ixt = np.mean([r['I_XT'] for r in valid_results])
                    avg_ity = np.mean([r['I_TY'] for r in valid_results])
                    print(f"✅ Epoch {current_epoch} - 平均MI: I(X;T)={avg_ixt:.4f}, I(T;Y)={avg_ity:.4f}")
            
            # 周期性显示详细汇总
            if current_epoch % self.show_summary_every_n_epochs == 0 and current_epoch > 0:
                self._show_periodic_summary()
    
    def _perform_infonet_mi_analysis(self, trainer, pl_module, current_epoch):
        """执行基于InfoNet的MI分析"""
        try:
            mi_analyzer = self._create_infonet_analyzer(pl_module)
            
            # 收集数据 - 按官方方式收集足够样本
            X_list, y_list = [], []
            total_samples = 0
            for i, (X, y) in enumerate(self.eval_loader):
                X_list.append(X)
                y_list.append(y)
                total_samples += X.size(0)
                # 收集足够样本或达到最大batch数
                if total_samples >= self.seq_len or i >= 19:  # 最多20个batch
                    break
            
            if not X_list:
                return []
            
            X_combined = torch.cat(X_list, dim=0)
            y_combined = torch.cat(y_list, dim=0)
            
            # 如果样本过多，随机采样到seq_len
            if len(X_combined) > self.seq_len:
                indices = torch.randperm(len(X_combined))[:self.seq_len]
                X_combined = X_combined[indices]
                y_combined = y_combined[indices]
            
            # 分析各层
            results = []
            layers = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
            
            for layer_name in layers:
                try:
                    layer_results = mi_analyzer.analyze_layer_with_infonet(
                        X_combined, y_combined, layer_name
                    )
                    
                    result_record = {
                        'layer': layer_name,
                        'I_XT': layer_results.get('I(X;T)', None),
                        'I_TY': layer_results.get('I(T;Y)', None),
                        'status': 'success' if layer_results else 'failed',
                        'method': 'infonet_official',
                        'sample_size': layer_results.get('sample_size', 0),
                        'proj_num': layer_results.get('proj_num', 0)
                    }
                    results.append(result_record)
                    
                except Exception as e:
                    print(f"⚠️ 层 {layer_name} 分析失败: {str(e)[:100]}")
                    results.append({
                        'layer': layer_name,
                        'I_XT': None,
                        'I_TY': None,
                        'status': f'error: {str(e)[:50]}',
                        'method': 'infonet_official',
                        'sample_size': 0,
                        'proj_num': 0
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ InfoNet MI分析出错: {e}")
            return []
    
    def _create_infonet_analyzer(self, trained_model):
        """创建基于InfoNet官方实现的MI分析器"""
        from infonet.infer import estimate_mi, compute_smi_mean
        from scipy.stats import rankdata
        
        class InfoNetOfficialMIAnalyzer(nn.Module):
            def __init__(self, source_model, infonet_model, seq_len, proj_num, batchsize):
                super().__init__()
                self.net = source_model.net
                self.mi_model = infonet_model
                self.seq_len = seq_len
                self.proj_num = proj_num
                self.batchsize = batchsize
                self.activations = {}
                self._register_hooks()
            
            def _register_hooks(self):
                """注册前向传播钩子"""
                def hook_fn(name):
                    def hook(module, input, output):
                        self.activations[name] = output.detach()
                    return hook
                
                try:
                    self.net[0].register_forward_hook(hook_fn('conv1'))
                    self.net[3].register_forward_hook(hook_fn('conv2'))
                    self.net[7].register_forward_hook(hook_fn('fc1'))
                    self.net[9].register_forward_hook(hook_fn('fc2'))
                    self.net[11].register_forward_hook(hook_fn('fc3'))
                except Exception as e:
                    print(f"❌ Hook注册失败: {e}")
            
            def forward(self, x):
                return self.net(x)
            
            def analyze_layer_with_infonet(self, X, y, target_layer):
                """使用InfoNet官方方法分析特定层的互信息"""
                self.eval()
                
                # 前向传播获取激活
                with torch.no_grad():
                    _ = self.forward(X)
                
                if target_layer not in self.activations:
                    print(f"❌ 未找到层 {target_layer} 的激活")
                    return {}
                
                T = self.activations[target_layer]
                
                # 使用官方方法进行MI估计
                return self._official_mutual_information(X, T, y)
            
            def _official_mutual_information(self, X, T, y):
                """使用InfoNet官方方法估计互信息"""
                try:
                    results = {'sample_size': len(X)}
                    
                    # 数据预处理 - 转换为numpy
                    X_np = X.view(X.size(0), -1).cpu().numpy()
                    T_np = T.view(T.size(0), -1).cpu().numpy()
                    y_np = y.cpu().numpy()
                    
                    # I(X;T)估计 - 按官方维度判断逻辑
                    if X_np.shape[1] <= 2 and T_np.shape[1] <= 2:
                        # 低维情况：直接使用estimate_mi (官方示例中的1维情况)
                        results['I(X;T)'] = self._estimate_1d_mi(X_np, T_np, 'I(X;T)')
                        results['proj_num'] = 1
                    else:
                        # 高维情况：使用compute_smi_mean (官方高维示例)
                        results['I(X;T)'] = self._compute_smi_mean_official(X_np, T_np, 'I(X;T)')
                        results['proj_num'] = self.proj_num
                    
                    # I(T;Y)估计 - 分类任务特殊处理
                    y_processed = self._process_labels_official(y_np)
                    if T_np.shape[1] <= 2:
                        results['I(T;Y)'] = self._estimate_1d_mi_classification(T_np, y_processed, 'I(T;Y)')
                    else:
                        results['I(T;Y)'] = self._compute_smi_classification(T_np, y_processed, 'I(T;Y)')
                    
                    return results
                    
                except Exception as e:
                    print(f"❌ InfoNet官方MI估计失败: {e}")
                    return {}
            
            def _estimate_1d_mi(self, X_data, T_data, mi_type):
                """1维MI估计 - 完全按官方示例"""
                try:
                    # 选择代表性特征（如果多维则取第一维或主成分）
                    if X_data.shape[1] > 1:
                        x_feature = X_data[:, 0]  # 简单取第一维，或可用PCA
                    else:
                        x_feature = X_data.flatten()
                    
                    if T_data.shape[1] > 1:
                        t_feature = T_data[:, 0]
                    else:
                        t_feature = T_data.flatten()
                    
                    # 官方预处理方式
                    x_ranked = rankdata(x_feature) / len(x_feature)
                    t_ranked = rankdata(t_feature) / len(t_feature)
                    
                    # 使用estimate_mi函数
                    mi_est = estimate_mi(self.mi_model, x_ranked, t_ranked)
                    if isinstance(mi_est, torch.Tensor):
                        mi_est = mi_est.item()
                    
                    print(f"📊 {mi_type} 1维估计: {mi_est:.4f}")
                    return mi_est
                    
                except Exception as e:
                    print(f"❌ 1维MI估计失败: {e}")
                    return None
            
            def _compute_smi_mean_official(self, X_data, T_data, mi_type):
                """高维SMI估计 - 完全按官方compute_smi_mean实现"""
                try:
                    # 直接调用官方函数
                    smi_result = compute_smi_mean(
                        sample_x=X_data,
                        sample_y=T_data,
                        model=self.mi_model,
                        proj_num=self.proj_num,
                        seq_len=self.seq_len,
                        batchsize=self.batchsize
                    )
                    
                    print(f"📊 {mi_type} SMI估计: {smi_result:.4f} (proj_num={self.proj_num})")
                    return smi_result
                    
                except Exception as e:
                    print(f"❌ SMI估计失败，回退到简化版本: {e}")
                    # 回退到简化的SMI实现
                    return self._fallback_smi_estimation(X_data, T_data, mi_type)
            
            def _fallback_smi_estimation(self, X_data, T_data, mi_type):
                """简化版SMI估计 - 当官方函数不可用时的回退方案"""
                try:
                    # 使用较少的投影次数进行回退
                    n_projections = min(64, self.proj_num // 16)  # 减少投影数
                    mi_estimates = []
                    
                    for _ in range(n_projections):
                        # 随机投影
                        if X_data.shape[1] > 1:
                            proj_x = np.random.randn(X_data.shape[1])
                            proj_x = proj_x / np.linalg.norm(proj_x)
                            x_proj = X_data @ proj_x
                        else:
                            x_proj = X_data.flatten()
                        
                        if T_data.shape[1] > 1:
                            proj_t = np.random.randn(T_data.shape[1])
                            proj_t = proj_t / np.linalg.norm(proj_t)
                            t_proj = T_data @ proj_t
                        else:
                            t_proj = T_data.flatten()
                        
                        # 标准化和估计
                        if np.std(x_proj) > 1e-6 and np.std(t_proj) > 1e-6:
                            x_ranked = rankdata(x_proj) / len(x_proj)
                            t_ranked = rankdata(t_proj) / len(t_proj)
                            
                            mi_est = estimate_mi(self.mi_model, x_ranked, t_ranked)
                            if isinstance(mi_est, torch.Tensor):
                                mi_est = mi_est.item()
                            
                            if 0 <= mi_est <= 15:
                                mi_estimates.append(mi_est)
                    
                    if mi_estimates:
                        avg_mi = np.mean(mi_estimates)
                        print(f"📊 {mi_type} 回退SMI估计: {avg_mi:.4f} (proj_num={n_projections})")
                        return avg_mi
                    else:
                        return None
                        
                except Exception as e:
                    print(f"❌ 回退SMI估计也失败: {e}")
                    return None
            
            def _estimate_1d_mi_classification(self, T_data, y_processed, mi_type):
                """1维分类MI估计"""
                try:
                    # 选择特征
                    if T_data.shape[1] > 1:
                        t_feature = T_data[:, 0]  # 简单选择第一维
                    else:
                        t_feature = T_data.flatten()
                    
                    # 标准化
                    if np.std(t_feature) > 1e-6 and np.std(y_processed) > 1e-6:
                        t_ranked = rankdata(t_feature) / len(t_feature)
                        y_ranked = rankdata(y_processed) / len(y_processed)
                        
                        mi_est = estimate_mi(self.mi_model, t_ranked, y_ranked)
                        if isinstance(mi_est, torch.Tensor):
                            mi_est = mi_est.item()
                        
                        print(f"📊 {mi_type} 1维分类估计: {mi_est:.4f}")
                        return mi_est
                    
                    return None
                    
                except Exception as e:
                    print(f"❌ 1维分类MI估计失败: {e}")
                    return None
            
            def _compute_smi_classification(self, T_data, y_processed, mi_type):
                """高维分类SMI估计"""
                try:
                    # 将y_processed扩展为二维以符合compute_smi_mean接口
                    y_expanded = y_processed.reshape(-1, 1)
                    
                    # 使用官方SMI方法
                    smi_result = compute_smi_mean(
                        sample_x=T_data,
                        sample_y=y_expanded,
                        model=self.mi_model,
                        proj_num=self.proj_num // 2,  # 分类任务使用较少投影
                        seq_len=self.seq_len,
                        batchsize=self.batchsize
                    )
                    
                    print(f"📊 {mi_type} 分类SMI估计: {smi_result:.4f}")
                    return smi_result
                    
                except Exception as e:
                    print(f"❌ 分类SMI估计失败，使用回退方案: {e}")
                    return self._fallback_classification_smi(T_data, y_processed, mi_type)
            
            def _fallback_classification_smi(self, T_data, y_processed, mi_type):
                """分类任务的回退SMI估计"""
                try:
                    n_projections = min(32, self.proj_num // 32)
                    mi_estimates = []
                    
                    for _ in range(n_projections):
                        if T_data.shape[1] > 1:
                            proj_t = np.random.randn(T_data.shape[1])
                            proj_t = proj_t / np.linalg.norm(proj_t)
                            t_proj = T_data @ proj_t
                        else:
                            t_proj = T_data.flatten()
                        
                        if np.std(t_proj) > 1e-6 and np.std(y_processed) > 1e-6:
                            t_ranked = rankdata(t_proj) / len(t_proj)
                            y_ranked = rankdata(y_processed) / len(y_processed)
                            
                            mi_est = estimate_mi(self.mi_model, t_ranked, y_ranked)
                            if isinstance(mi_est, torch.Tensor):
                                mi_est = mi_est.item()
                            
                            if 0 <= mi_est <= 15:
                                mi_estimates.append(mi_est)
                    
                    if mi_estimates:
                        avg_mi = np.mean(mi_estimates)
                        print(f"📊 {mi_type} 回退分类SMI: {avg_mi:.4f}")
                        return avg_mi
                    else:
                        return None
                        
                except Exception as e:
                    print(f"❌ 回退分类SMI失败: {e}")
                    return None
            
            def _process_labels_official(self, y_np):
                """官方风格的标签处理"""
                unique_classes = np.unique(y_np)
                
                if len(unique_classes) <= 1:
                    return np.zeros_like(y_np, dtype=float)
                
                # 简单的标签到连续值映射
                y_processed = np.zeros_like(y_np, dtype=float)
                for i, cls in enumerate(unique_classes):
                    mask = y_np == cls
                    y_processed[mask] = i / (len(unique_classes) - 1)
                
                # 添加小量噪声使其连续
                noise_scale = 0.01 / len(unique_classes)
                y_processed += np.random.normal(0, noise_scale, size=y_processed.shape)
                
                return y_processed
        
        return InfoNetOfficialMIAnalyzer(
            trained_model, 
            self.mi_model, 
            self.seq_len, 
            self.proj_num, 
            self.batchsize
        )
    
    def _save_results_to_csv(self, epoch, results):
        """保存结果到CSV"""
        timestamp = datetime.now().isoformat()
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['epoch', 'timestamp', 'layer', 'I_XT', 'I_TY', 'status', 'method', 'sample_size', 'proj_num']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            for result in results:
                writer.writerow({
                    'epoch': epoch,
                    'timestamp': timestamp,
                    'layer': result['layer'],
                    'I_XT': result['I_XT'],
                    'I_TY': result['I_TY'],
                    'status': result['status'],
                    'method': result.get('method', 'infonet_official'),
                    'sample_size': result.get('sample_size', 0),
                    'proj_num': result.get('proj_num', 0)
                })
    
    def _show_periodic_summary(self):
        """显示周期性汇总报告"""
        try:
            if os.path.exists(self.csv_path):
                df = pd.read_csv(self.csv_path)
                if not df.empty:
                    print(f"\n📊 {self.experiment_name} - 周期性汇总报告")
                    print("=" * 60)
                    
                    # 按层显示最新结果
                    latest_epoch = df['epoch'].max()
                    latest_data = df[df['epoch'] == latest_epoch]
                    
                    for _, row in latest_data.iterrows():
                        if row['I_XT'] is not None and row['I_TY'] is not None:
                            proj_info = f"(proj_num={row.get('proj_num', 'N/A')})" if row.get('proj_num') else ""
                            print(f"层 {row['layer']}: I(X;T)={row['I_XT']:.4f}, I(T;Y)={row['I_TY']:.4f} {proj_info}")
                    
                    print("=" * 60)
                else:
                    print("📋 暂无MI数据可显示")
            else:
                print("📋 MI结果文件尚未创建")
        except Exception as e:
            print(f"⚠️ 汇总报告生成失败: {e}")

class EnhancedHighFrequencyLiveLossPlotCallback(pl.Callback):
    """增强版LiveLoss回调 - 集成显示管理"""
    
    def __init__(self, update_every_n_batches=10, display_manager=None):
        super().__init__()
        self.liveloss = PlotLosses()
        self.update_every_n_batches = update_every_n_batches
        self.batch_count = 0
        self.display_manager = display_manager or _display_manager
        
        # 用于累积指标
        self.train_loss_accumulator = []
        self.current_epoch = 0
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """每个训练batch结束后调用"""
        self.batch_count += 1
        
        # 累积当前batch的损失
        if outputs is not None:
            if isinstance(outputs, dict) and 'loss' in outputs:
                self.train_loss_accumulator.append(outputs['loss'].item())
            elif hasattr(outputs, 'item'):
                self.train_loss_accumulator.append(outputs.item())
            elif torch.is_tensor(outputs):
                self.train_loss_accumulator.append(outputs.item())
        
        # 每隔 N 个 batch 更新一次图表
        if self.batch_count % self.update_every_n_batches == 0:
            self._update_plot(trainer, pl_module)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """验证结束后也更新一次"""
        self._update_plot(trainer, pl_module)
        self.current_epoch += 1
        self.train_loss_accumulator = []
    
    def _update_plot(self, trainer, pl_module):
        """受管理的图表更新"""
        # 如果MI正在分析，则跳过这次更新
        if self.display_manager.is_plotting:
            return
        
        logs = {}
        
        # 计算平均训练损失
        if self.train_loss_accumulator:
            avg_train_loss = sum(self.train_loss_accumulator) / len(self.train_loss_accumulator)
            logs['log loss'] = avg_train_loss
        
        # 获取最新的验证指标
        if 'val_loss' in trainer.callback_metrics:
            logs['val_log loss'] = trainer.callback_metrics['val_loss'].item()
        if 'val_acc' in trainer.callback_metrics:
            logs['val_accuracy'] = trainer.callback_metrics['val_acc'].item()
        if 'train_acc' in trainer.callback_metrics:
            logs['accuracy'] = trainer.callback_metrics['train_acc'].item()
        
        # 更新图表
        if logs:
            self.liveloss.update(logs)
            self.liveloss.send()