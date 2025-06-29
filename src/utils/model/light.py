from pydantic import BaseModel
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from typing import List, Optional, Union
import matplotlib.pyplot as plt
from livelossplot import PlotLosses

from torch import nn
import numpy as np





# 添加自定义回调来集成 livelossplot
class LiveLossPlotCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.liveloss = PlotLosses()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        logs = {}
        
        # 获取训练和验证指标
        if 'train_loss' in trainer.callback_metrics:
            logs['log loss'] = trainer.callback_metrics['train_loss'].item()
        if 'val_loss' in trainer.callback_metrics:
            logs['val_log loss'] = trainer.callback_metrics['val_loss'].item()
        if 'train_acc' in trainer.callback_metrics:
            logs['accuracy'] = trainer.callback_metrics['train_acc'].item()
        if 'val_acc' in trainer.callback_metrics:
            logs['val_accuracy'] = trainer.callback_metrics['val_acc'].item()
        
        # 更新图表
        if logs:
            self.liveloss.update(logs)
            self.liveloss.send()


class MutualInformationCallback(Callback):
    def __init__(self, mi_model, eval_loader, eval_every_n_epochs=1):
        super().__init__()  # 添加这行
        self.mi_model = mi_model
        self.eval_loader = eval_loader
        self.eval_every_n_epochs = eval_every_n_epochs
        self.mi_history = {'epoch': [], 'layer_results': []}
        print(f"✅ MutualInformationCallback 初始化成功，每 {eval_every_n_epochs} 个epoch分析一次")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """改用on_train_epoch_end而不是on_epoch_end"""
        current_epoch = trainer.current_epoch
        
        print(f"\n🔍 检查是否需要MI分析 - Epoch {current_epoch}")
        
        if current_epoch % self.eval_every_n_epochs == 0:
            print(f"=== Epoch {current_epoch} 互信息分析开始 ===")
            
            try:
                # 创建MI分析器
                mi_analyzer = self._create_mi_analyzer(pl_module)
                
                # 收集数据（只用少量数据避免内存问题）
                print("📊 收集评估数据...")
                X_list, y_list = [], []
                for i, (X, y) in enumerate(self.eval_loader):
                    X_list.append(X)
                    y_list.append(y)
                    if i >= 2:  # 只用3个batch
                        break
                
                if not X_list:
                    print("❌ 没有收集到数据")
                    return
                
                X_combined = torch.cat(X_list, dim=0)
                y_combined = torch.cat(y_list, dim=0)
                print(f"✅ 数据收集完成，形状: {X_combined.shape}")
                
                # 分析各层
                layer_results = {}
                layers = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
                
                for layer_name in layers:
                    print(f"🔬 分析层: {layer_name}")
                    try:
                        results = mi_analyzer.analyze_layer(X_combined, y_combined, layer_name)
                        if results:
                            layer_results[layer_name] = results
                            print(f"  ✅ {layer_name}: I(X;T)={results.get('I(X;T)', 0):.4f}, "
                                  f"I(T;Y)={results.get('I(T;Y)', 0):.4f}")
                        else:
                            print(f"  ❌ {layer_name}: 分析失败")
                    except Exception as e:
                        print(f"  ❌ {layer_name}: 分析出错 - {e}")
                
                # 记录历史
                self.mi_history['epoch'].append(current_epoch)
                self.mi_history['layer_results'].append(layer_results)
                
                print(f"=== Epoch {current_epoch} 互信息分析完成 ===\n")
                
            except Exception as e:
                print(f"❌ 互信息分析出错: {e}")
                import traceback
                traceback.print_exc()
    
    def _create_mi_analyzer(self, trained_model):
        """基于训练好的模型创建MI分析器"""
        class MIAnalyzer(nn.Module):
            def __init__(self, source_model):
                super().__init__()
                self.net = source_model.net
                self.activations = {}
                self._register_hooks()
            
            def _register_hooks(self):
                def hook_fn(name):
                    def hook(module, input, output):
                        self.activations[name] = output.detach()
                    return hook
                
                # 为关键层注册钩子
                try:
                    self.net[0].register_forward_hook(hook_fn('conv1'))  # 第一个卷积
                    self.net[3].register_forward_hook(hook_fn('conv2'))  # 第二个卷积
                    self.net[7].register_forward_hook(hook_fn('fc1'))    # 第一个全连接
                    self.net[9].register_forward_hook(hook_fn('fc2'))    # 第二个全连接
                    self.net[11].register_forward_hook(hook_fn('fc3'))   # 输出层
                    print("✅ Hook注册成功")
                except Exception as e:
                    print(f"❌ Hook注册失败: {e}")
            
            def forward(self, x):
                return self.net(x)
                
            def analyze_layer(self, X, y, target_layer, sample_size=100):
                """分析特定层的互信息"""
                self.eval()
                
                # 采样数据
                if len(X) > sample_size:
                    indices = torch.randperm(len(X))[:sample_size]
                    X_sample = X[indices]
                    y_sample = y[indices]
                else:
                    X_sample = X
                    y_sample = y
                
                # 前向传播
                with torch.no_grad():
                    _ = self.forward(X_sample)
                
                if target_layer not in self.activations:
                    print(f"❌ 层 {target_layer} 的激活未找到")
                    return {}
                
                T = self.activations[target_layer]
                print(f"📊 {target_layer} 激活形状: {T.shape}")
                
                try:
                    # 简化版本的互信息估计
                    results = {}
                    
                    # 计算I(X;T) - 简化版本
                    X_flat = X_sample.view(X_sample.size(0), -1)
                    T_flat = T.view(T.size(0), -1)
                    
                    # 使用相关性作为互信息的代理
                    if X_flat.size(1) > 0 and T_flat.size(1) > 0:
                        # 计算第一个特征的相关性
                        x_feat = X_flat[:, 0].cpu().numpy()
                        t_feat = T_flat[:, 0].cpu().numpy()
                        
                        corr_xt = np.corrcoef(x_feat, t_feat)[0, 1]
                        if not np.isnan(corr_xt):
                            I_XT = -0.5 * np.log(1 - corr_xt**2 + 1e-8)
                            results['I(X;T)'] = I_XT
                    
                    # 计算I(T;Y) - 简化版本
                    if T_flat.size(1) > 0:
                        y_np = y_sample.cpu().numpy()
                        t_feat = T_flat[:, 0].cpu().numpy()
                        
                        # 对于分类问题，计算每个类别的均值差异
                        unique_classes = np.unique(y_np)
                        if len(unique_classes) > 1:
                            class_means = []
                            for cls in unique_classes:
                                mask = y_np == cls
                                if np.sum(mask) > 0:
                                    class_means.append(np.mean(t_feat[mask]))
                            
                            if len(class_means) > 1:
                                var_between = np.var(class_means)
                                var_within = np.var(t_feat)
                                if var_within > 0:
                                    I_TY = 0.5 * np.log(1 + var_between / var_within)
                                    results['I(T;Y)'] = I_TY
                    
                    return results
                    
                except Exception as e:
                    print(f"❌ 互信息计算出错: {e}")
                    return {}
        
        return MIAnalyzer(trained_model)

# 添加可视化回调类
class PlotMetricsCallback(Callback):
    """实时绘制训练指标的回调函数"""
    
    def __init__(self, plot_every_n_epochs=1):
        super().__init__()
        self.plot_every_n_epochs = plot_every_n_epochs
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.epochs = []
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """在每个验证epoch结束时更新图表"""
        current_epoch = trainer.current_epoch
        
        # 收集指标
        train_loss = trainer.callback_metrics.get('train_loss', None)
        val_loss = trainer.callback_metrics.get('val_loss', None)
        train_acc = trainer.callback_metrics.get('train_acc', None)
        val_acc = trainer.callback_metrics.get('val_acc', None)
        
        if train_loss is not None and val_loss is not None:
            self.epochs.append(current_epoch)
            self.train_losses.append(train_loss.item())
            self.val_losses.append(val_loss.item())
            
            if train_acc is not None and val_acc is not None:
                self.train_accs.append(train_acc.item())
                self.val_accs.append(val_acc.item())
        
        # 每隔指定轮数绘制一次图表
        if (current_epoch + 1) % self.plot_every_n_epochs == 0 and len(self.epochs) > 0:
            self.plot_metrics()
    
    def plot_metrics(self):
        """绘制训练指标"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 绘制损失
        axes[0].plot(self.epochs, self.train_losses, 'b-', label='训练损失', marker='o', markersize=4)
        axes[0].plot(self.epochs, self.val_losses, 'r-', label='验证损失', marker='s', markersize=4)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('损失函数变化')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 绘制准确率（如果有的话）
        if len(self.train_accs) > 0:
            axes[1].plot(self.epochs, self.train_accs, 'b-', label='训练准确率', marker='o', markersize=4)
            axes[1].plot(self.epochs, self.val_accs, 'r-', label='验证准确率', marker='s', markersize=4)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('准确率变化')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1)
        else:
            axes[1].text(0.5, 0.5, '暂无准确率数据', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('准确率变化')
        
        plt.tight_layout()
        plt.show()
        
        # 清除输出，防止图表堆积
        from IPython.display import clear_output
        clear_output(wait=True)
        plt.show()

# 添加更频繁的实时绘图回调
class HighFrequencyLiveLossPlotCallback(pl.Callback):
    """高频率的 livelossplot 回调 - 每隔 N 个 batch 更新一次"""
    
    def __init__(self, update_every_n_batches=10):
        super().__init__()
        self.liveloss = PlotLosses()
        self.update_every_n_batches = update_every_n_batches
        self.batch_count = 0
        
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
            elif hasattr(outputs, 'item'):  # 处理直接返回tensor的情况
                self.train_loss_accumulator.append(outputs.item())
            elif torch.is_tensor(outputs):  # 处理tensor情况
                self.train_loss_accumulator.append(outputs.item())
        
        # 每隔 N 个 batch 更新一次图表
        if self.batch_count % self.update_every_n_batches == 0:
            self._update_plot(trainer, pl_module)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """验证结束后也更新一次"""
        self._update_plot(trainer, pl_module)
        self.current_epoch += 1
        self.train_loss_accumulator = []  # 重置累积器
    
    def _update_plot(self, trainer, pl_module):
        """更新图表"""
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

class RealTimePlotCallback(pl.Callback):
    """真正的实时绘图回调 - 使用matplotlib的动态更新"""
    
    def __init__(self, update_every_n_batches=5):
        super().__init__()
        self.update_every_n_batches = update_every_n_batches
        self.batch_count = 0
        
        # 数据存储
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.batch_indices = []
        
        # 设置matplotlib为交互模式
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.fig.suptitle('实时训练监控')
        
        # 初始化空线条
        self.train_loss_line, = self.ax1.plot([], [], 'b-', label='训练损失', alpha=0.7)
        self.val_loss_line, = self.ax1.plot([], [], 'r-', label='验证损失', marker='o', markersize=3)
        
        self.train_acc_line, = self.ax2.plot([], [], 'b-', label='训练准确率', alpha=0.7)
        self.val_acc_line, = self.ax2.plot([], [], 'r-', label='验证准确率', marker='o', markersize=3)
        
        # 设置图表
        self._setup_plots()
        
    def _setup_plots(self):
        """设置图表样式"""
        # 损失图
        self.ax1.set_xlabel('Batch')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('损失变化')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # 准确率图
        self.ax2.set_xlabel('Batch')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.set_title('准确率变化')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show(block=False)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """每个训练batch结束后调用"""
        self.batch_count += 1
        
        # 记录训练指标
        if outputs and 'loss' in outputs:
            self.train_losses.append(outputs['loss'].item())
            self.batch_indices.append(self.batch_count)
            
            # 如果有训练准确率
            if 'train_acc' in trainer.callback_metrics:
                self.train_accs.append(trainer.callback_metrics['train_acc'].item())
        
        # 每隔 N 个 batch 更新图表
        if self.batch_count % self.update_every_n_batches == 0:
            self._update_real_time_plot()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """验证结束后更新验证指标"""
        if 'val_loss' in trainer.callback_metrics and self.batch_indices:
            # 在当前位置添加验证点
            current_batch = self.batch_indices[-1]
            self.val_losses.append((current_batch, trainer.callback_metrics['val_loss'].item()))
            
            if 'val_acc' in trainer.callback_metrics:
                self.val_accs.append((current_batch, trainer.callback_metrics['val_acc'].item()))
        
        self._update_real_time_plot()
    
    def _update_real_time_plot(self):
        """更新实时图表"""
        if not self.batch_indices:
            return
            
        # 更新训练损失
        if self.train_losses:
            self.train_loss_line.set_data(self.batch_indices, self.train_losses)
            
        # 更新验证损失
        if self.val_losses:
            val_x, val_y = zip(*self.val_losses)
            self.val_loss_line.set_data(val_x, val_y)
        
        # 更新训练准确率
        if self.train_accs:
            self.train_acc_line.set_data(self.batch_indices, self.train_accs)
            
        # 更新验证准确率
        if self.val_accs:
            val_acc_x, val_acc_y = zip(*self.val_accs)
            self.val_acc_line.set_data(val_acc_x, val_acc_y)
        
        # 自动调整坐标轴
        for ax in [self.ax1, self.ax2]:
            ax.relim()
            ax.autoscale_view()
        
        # 刷新图表
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # 在Jupyter中显示
        from IPython.display import display, clear_output
        clear_output(wait=True)
        display(self.fig)

# 添加回调组合类
class DefaultCallbacks:
    """默认回调组合类"""
    
    @staticmethod
    def basic() -> List[Callback]:
        """基础回调"""
        return []
    
    @staticmethod
    def with_live_plot(plot_every_n_epochs=1) -> List[Callback]:
        """带实时绘图的回调"""
        return [PlotMetricsCallback(plot_every_n_epochs=plot_every_n_epochs)]
    
    @staticmethod
    def with_early_stopping(monitor="val_loss", patience=5, mode="min") -> List[Callback]:
        """带早停的回调"""
        return [EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            verbose=True
        )]
    
    @staticmethod
    def with_checkpoint(monitor="val_loss", mode="min", save_top_k=1) -> List[Callback]:
        """带模型检查点的回调"""
        return [ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            verbose=True
        )]
    
    @staticmethod
    def full_featured(
        plot_every_n_epochs=1,
        monitor="val_loss", 
        patience=10, 
        mode="min", 
        save_top_k=1
    ) -> List[Callback]:
        """功能完整的回调组合"""
        return [
            PlotMetricsCallback(plot_every_n_epochs=plot_every_n_epochs),
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                mode=mode,
                verbose=True
            ),
            ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                save_top_k=save_top_k,
                verbose=True
            )
        ]
    
    @staticmethod
    def for_classification(plot_every_n_epochs=1) -> List[Callback]:
        """专为分类任务设计的回调"""
        return [
            PlotMetricsCallback(plot_every_n_epochs=plot_every_n_epochs),
            ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                save_top_k=1,
                verbose=True,
                filename='best-{epoch:02d}-{val_acc:.2f}'
            )
        ]


class ModelConfig(BaseModel):
    """Configuration for the Model."""
    
    lr: float = 0.01
    num_hiddens: int = 256
    output_size: int = 10
    weight_decay: float = 0.0
    optimizer: str = "sgd"  # sgd, adam, adamw
    dropout_rate: float = 0.5  # Dropout rate for regularization
    
    # Lightning相关配置
    accelerator: str = "auto"  # auto, gpu, cpu, tpu
    devices: Union[int, List[int], str] = "auto"  # auto, 1, [0,1], "0,1"
    precision: Union[int, str] = 32  # 16, 32, "bf16"
    
    momentum: float = 0.9  # SGD动量
    
    class Config:
        validate_assignment = True


class ModernModule(pl.LightningModule):
    """基于PyTorch Lightning的现代化模块"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.dict())
        
        # 自动保存配置到检查点
        self.automatic_optimization = True
        
    def forward(self, x):
        """默认前向传播"""
        if not hasattr(self, "net"):
            raise NotImplementedError(
                "You must define 'self.net' in your model's __init__."
            )
        if not callable(self.net):
            raise RuntimeError(
                "self.net is not callable. Make sure it's a proper nn.Module."
            )
        return self.net(x)
    
    def compute_loss(self, y_hat, y) -> torch.Tensor:
        """子类必须实现这个方法来定义损失函数"""
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        """训练步骤 - Lightning自动处理设备"""
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        
        # 记录指标
        self.log('train_loss', loss, prog_bar=True)
        
        # 如果是分类任务，计算准确率
        if hasattr(self, '_is_classification') and self._is_classification:
            acc = self._compute_accuracy(y_hat, y)
            self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤 - Lightning自动处理设备"""
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        
        # 记录指标
        self.log('val_loss', loss, prog_bar=True)
        
        # 如果是分类任务，计算准确率
        if hasattr(self, '_is_classification') and self._is_classification:
            acc = self._compute_accuracy(y_hat, y)
            self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        
        self.log('test_loss', loss)
        
        if hasattr(self, '_is_classification') and self._is_classification:
            acc = self._compute_accuracy(y_hat, y)
            self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        """配置优化器"""
        params = self.parameters()
        
        if self.config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                params, 
                lr=self.config.lr, 
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                params, 
                lr=self.config.lr, 
                weight_decay=self.config.weight_decay
            )
        else:  # sgd
            optimizer = torch.optim.SGD(
                params, 
                lr=self.config.lr, 
                weight_decay=self.config.weight_decay
            )
        
        # 可选：添加学习率调度器
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        # return [optimizer], [scheduler]
        
        return optimizer
    
    def _compute_accuracy(self, y_hat, y):
        """计算准确率"""
        if y_hat.dim() > 1 and y_hat.size(1) > 1:  # 多分类
            pred = y_hat.argmax(dim=1)
        else:  # 二分类
            pred = (y_hat > 0.5).float()
        
        return (pred == y).float().mean()
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """预测步骤"""
        x, _ = batch if isinstance(batch, (list, tuple)) else (batch, None)
        return self(x)

class ModernTrainer:
    """现代化训练器，基于PyTorch Lightning"""
    
    def __init__(
        self,
        max_epochs: int = 10,
        accelerator: str = "auto",
        devices: Union[int, List[int], str] = "auto",
        precision: Union[int, str] = 32,
        log_every_n_steps: int = 50,
        enable_checkpointing: bool = True,
        enable_progress_bar: bool = True,
        logger: bool = True,
        callbacks: Optional[List[Callback]] = None,  # 修改为 Optional
    ):
        self.trainer_kwargs = {
            "max_epochs": max_epochs,
            "accelerator": accelerator,
            "devices": devices,
            "precision": precision,
            "log_every_n_steps": log_every_n_steps,
            "enable_checkpointing": enable_checkpointing,
            "enable_progress_bar": enable_progress_bar,
            "logger": logger,
        }

        self.callbacks: List[Callback] = callbacks or []  # 安全的默认值处理
        self.trainer = None
    
    def add_callback(self, callback: Callback):
        """添加回调"""
        self.callbacks.append(callback)
        return self
    
    def add_callbacks(self, callbacks: List[Callback]):
        """批量添加回调"""
        self.callbacks.extend(callbacks)
        return self
    
    def add_early_stopping(self, monitor="val_loss", patience=5, mode="min"):
        """添加早停回调"""
        early_stop = EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            verbose=True
        )
        return self.add_callback(early_stop)
    
    def add_model_checkpoint(self, monitor="val_loss", mode="min", save_top_k=1):
        """添加模型检查点回调"""
        checkpoint = ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            verbose=True
        )
        return self.add_callback(checkpoint)
    
    def fit(self, model: ModernModule, train_loader, val_loader=None):
        """训练模型"""
        # 创建trainer
        self.trainer = pl.Trainer(
            callbacks=self.callbacks,
            **self.trainer_kwargs
        )
        
        # 开始训练
        self.trainer.fit(model, train_loader, val_loader)
        
        return self.trainer
    
    def test(self, model: ModernModule, test_loader):
        """测试模型"""
        if self.trainer is None:
            self.trainer = pl.Trainer(**self.trainer_kwargs)
        
        return self.trainer.test(model, test_loader)
    
    def predict(self, model: ModernModule, dataloader):
        """预测"""
        if self.trainer is None:
            self.trainer = pl.Trainer(**self.trainer_kwargs)
        
        return self.trainer.predict(model, dataloader)


class TrainerFactory:
    """训练器工厂类"""
    
    @staticmethod
    def basic(max_epochs=10, callbacks=None):
        """基础训练器"""
        return ModernTrainer(
            max_epochs=max_epochs,
            enable_progress_bar=True,
            callbacks=callbacks if callbacks is not None else DefaultCallbacks.basic()
        )
    
    @staticmethod
    def with_live_plot(max_epochs=10, plot_every_n_epochs=1):
        """带实时绘图的训练器"""
        return ModernTrainer(
            max_epochs=max_epochs,
            enable_progress_bar=True,
            callbacks=DefaultCallbacks.with_live_plot(plot_every_n_epochs)
        )
    
    @staticmethod
    def for_classification(max_epochs=10, plot_every_n_epochs=1):
        """分类任务专用训练器"""
        return ModernTrainer(
            max_epochs=max_epochs,
            enable_progress_bar=True,
            callbacks=DefaultCallbacks.for_classification(plot_every_n_epochs)
        )
    
    @staticmethod
    def full_featured(max_epochs=100, plot_every_n_epochs=1, patience=10):
        """功能完整的训练器"""
        return ModernTrainer(
            max_epochs=max_epochs,
            enable_progress_bar=True,
            callbacks=DefaultCallbacks.full_featured(
                plot_every_n_epochs=plot_every_n_epochs,
                patience=patience
            )
        )
    
    @staticmethod
    def gpu_optimized(max_epochs=10, precision=16):
        """GPU优化训练器"""
        return ModernTrainer(
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=1,
            precision=precision,
            enable_progress_bar=True,
            callbacks=DefaultCallbacks.with_checkpoint()
        )
    
    @staticmethod
    def with_early_stopping(max_epochs=100, patience=10):
        """带早停的训练器"""
        return ModernTrainer(
            max_epochs=max_epochs,
            enable_progress_bar=True,
            callbacks=DefaultCallbacks.with_early_stopping(patience=patience)
        )

# 设备工具函数
class DeviceUtils:
    """设备管理工具"""
    
    @staticmethod
    def get_available_devices():
        """获取可用设备"""
        devices = []
        
        # CPU总是可用的
        devices.append("cpu")
        
        # 检查GPU
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
                
        # 检查MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append("mps")
            
        return devices
    
    @staticmethod
    def print_device_info():
        """打印设备信息"""
        print("🔍 设备检测结果:")
        print(f"   PyTorch版本: {torch.__version__}")
        
        # GPU信息
        if torch.cuda.is_available():
            print("   CUDA可用: ✅")
            print(f"   CUDA版本: {torch.version.cuda}") # type: ignore
            print(f"   GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {name} ({memory:.1f}GB)")
        else:
            print("   CUDA: ❌")
        
        # MPS信息 (Apple Silicon)
        if hasattr(torch.backends, 'mps'):
            if torch.backends.mps.is_available():
                print("   MPS (Apple Silicon): ✅")
            else:
                print("   MPS (Apple Silicon): ❌")
        
        print(f"   推荐设备: {DeviceUtils.get_recommended_device()}")
    
    @staticmethod
    def get_recommended_device():
        """获取推荐设备"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"