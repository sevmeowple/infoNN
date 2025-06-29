from pydantic import BaseModel, Field
from livelossplot import PlotLosses
import torch
from torch import device, nn
import platform
import psutil
import time

from typing import List, Optional, Tuple, Union, TYPE_CHECKING
import collections
# import time

import matplotlib
import matplotlib.pyplot as plt
from IPython import display
import numpy as np

if TYPE_CHECKING:
    from typing import TYPE_CHECKING


class BoardConfig(BaseModel):
    """Configuration for the ProgressBoard using Pydantic."""

    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    xscale: str = "linear"
    yscale: str = "linear"
    ls: List[str] = Field(default_factory=lambda: ["-", "--", "-.", ":"])
    colors: List[str] = Field(default_factory=lambda: ["C0", "C1", "C2", "C3"])
    figsize: Tuple[float, float] = (4.5, 3.5)

    # Pydantic v2 中，这是一个好习惯，允许在模型外部修改字段
    class Config:
        validate_assignment = True


class ProgressBoard:
    """A modern ProgressBoard that uses a Pydantic config object."""

    def __init__(self, config: Optional[BoardConfig] = None):
        """Initializes the board with a configuration object."""
        # 如果没有提供配置，就使用默认配置
        self.config = config or BoardConfig()

        # 创建图形和坐标轴
        self.fig, self.axes = plt.subplots(figsize=self.config.figsize)

        # 内部数据和计数器存储
        self.data = collections.OrderedDict()
        self.drawn = collections.OrderedDict()

    def _draw_figure(self):
        """The core drawing logic."""
        display.clear_output(wait=True)
        self.axes.cla()

        for i, (label, (xs, ys)) in enumerate(self.data.items()):
            self.axes.plot(
                xs,
                ys,
                linestyle=self.config.ls[i % len(self.config.ls)],
                color=self.config.colors[i % len(self.config.colors)],
                label=label,
            )

        # 从 config 对象中读取配置来设置图表
        if self.config.xlabel:
            self.axes.set_xlabel(self.config.xlabel)
        if self.config.ylabel:
            self.axes.set_ylabel(self.config.ylabel)
        self.axes.set_xscale(self.config.xscale)
        self.axes.set_yscale(self.config.yscale)
        if self.config.xlim:
            self.axes.set_xlim(self.config.xlim)
        if self.config.ylim:
            self.axes.set_ylim(self.config.ylim)
        self.axes.grid()
        self.axes.legend()

        display.display(self.fig)

    def draw(
        self,
        x: Union[float, np.floating],
        y: Union[float, np.floating],
        label: str,
        every_n: int = 1,
    ):
        """Add a data point and redraw the figure."""
        if label not in self.data:
            self.data[label] = ([], [])
            self.drawn[label] = 0

        self.data[label][0].append(float(x))
        self.data[label][1].append(float(y))
        self.drawn[label] += 1

        if self.drawn[label] % every_n == 0:
            self._draw_figure()


class ModelConfig(BaseModel):
    """Configuration for the Model."""

    lr: float = 0.01
    num_hiddens: int = 256
    output_size: int = 10
    # 还可以包含其他配置，如 dropout_prob, weight_decay等
    self_device: Optional[str] = None  # 新增：指定设备，如 "cuda:1"
    
    # 静态方法展示所有可用gpu设备及其选择id
    @staticmethod
    def available_devices() -> List[str]:
        """Returns a list of available devices."""
        if torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            return ["cpu"]
class ModernModule(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.self_device = self._resolve_device()
        self.to(self.self_device)

    def _resolve_device(self) -> torch.device:
        if self.config.self_device:
            return torch.device(self.config.self_device)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_device(self, device_str: str):
        self.config.self_device = device_str
        self.self_device = torch.device(device_str)
        self.to(self.self_device)

    def forward(self, X):
        """Default forward pass."""
        if not hasattr(self, "net"):
            raise NotImplementedError(
                "You must define 'self.net' in your model's __init__."
            )
        if not callable(self.net):
            raise TypeError(
                "self.net must be a callable (e.g., nn.Module), got a Tensor instead. "
                "Make sure to define self.net as a neural network module in your __init__ method."
            )
        return self.net(X)

    def loss(self, y_hat, y) -> torch.Tensor:
        """子类必须实现这个方法来定义损失函数。"""
        raise NotImplementedError

    def training_step(self, batch) -> torch.Tensor:
        """
        Defines the logic for one training step.
        Just computes and returns the loss. No plotting!
        """
        # *batch[:-1] 将 batch 中除最后一个元素外的所有元素作为输入
        # batch[-1] 是标签 y
        X, y = batch[:-1], batch[-1]
        y_hat = self(*X)  # 调用 self.forward(X)
        return self.loss(y_hat, y)

    def validation_step(self, batch) -> torch.Tensor:
        """

        Defines the logic for one validation step.
        Just computes and returns the loss.
        """
        X, y = batch[:-1], batch[-1]
        y_hat = self(*X)
        return self.loss(y_hat, y)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """子类必须实现这个方法来返回优化器。"""
        # 这是一个常见的实现，可以放在基类中
        return torch.optim.SGD(self.parameters(), lr=self.config.lr)

    
# 引入我们之前实现的 Pydantic 版 ProgressBoard
# from your_file import ProgressBoard, BoardConfig


class Callback:
    """Base class for callbacks."""

    def __init__(self):
        """Initialize the callback."""
        self.trainer: Optional["Trainer"] = None

    def on_train_begin(self, trainer: "Trainer"):
        pass

    def on_train_end(self, trainer: "Trainer"):
        pass

    def on_epoch_begin(self, trainer: "Trainer"):
        pass

    def on_epoch_end(self, trainer: "Trainer", train_metrics: dict, val_metrics: dict):
        pass

    def on_batch_begin(self, trainer: "Trainer"):
        pass

    def on_batch_end(self, trainer: "Trainer", loss: torch.Tensor):
        pass


class PlottingCallback(Callback):
    """A callback dedicated to plotting metrics during training."""

    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        # 创建绘图板
        matplotlib.use("inline")  # 确保在 Jupyter Notebook 中使用内联绘图
        self.board = ProgressBoard()  # 使用默认配置
        self.plot_train_per_epoch = plot_train_per_epoch
        self.plot_valid_per_epoch = plot_valid_per_epoch

    def on_train_begin(self, trainer: "Trainer"):
        # 在训练开始时设置x轴标签
        self.board.config.xlabel = "epoch"

    def on_batch_end(self, trainer, loss):
        """在每个训练批次结束后，绘制训练损失。"""
        # 计算当前在 epoch 中的位置 (0.0 to 1.0)
        x = trainer.epoch + (trainer.train_batch_idx + 1) / trainer.num_train_batches

        # 计算绘图频率
        n = trainer.num_train_batches / self.plot_train_per_epoch

        self.board.draw(x, loss.item(), "train_loss", every_n=int(n))

    def on_epoch_end(self, trainer, train_metrics, val_metrics):
        """在每个轮次结束后，绘制聚合后的验证损失。"""
        # x轴是当前epoch数 (e.g., 1, 2, 3...)
        x = trainer.epoch + 1
        self.board.draw(x, val_metrics["loss"], "val_loss", every_n=1)

class LiveLossCallback(Callback):
    """使用 livelossplot 的回调"""
    
    def __init__(self):
        super().__init__()
        self.liveloss = PlotLosses()
    
    def on_epoch_end(self, trainer, train_metrics: dict, val_metrics: dict):
        """在每个 epoch 结束时更新图表"""
        logs = {}
        
        # 添加训练指标
        if train_metrics:
            logs['loss'] = train_metrics['loss']
            if 'accuracy' in train_metrics:
                logs['accuracy'] = train_metrics['accuracy']
        
        # 添加验证指标（自动加 val_ 前缀）
        if val_metrics:
            logs['val_loss'] = val_metrics['loss']
            if 'accuracy' in val_metrics:
                logs['val_accuracy'] = val_metrics['accuracy']
        
        # 更新并显示图表
        self.liveloss.update(logs)
        self.liveloss.send()



class SystemInfoCallback(Callback):
    """显示系统和硬件信息的回调"""
    
    def __init__(self, show_detailed=False):
        super().__init__()
        self.show_detailed = show_detailed
    
    def on_train_begin(self, trainer: "Trainer"):
        print("🖥️  系统信息:")
        print(f"   操作系统: {platform.system()} {platform.release()}")
        print(f"   Python 版本: {platform.python_version()}")
        print(f"   PyTorch 版本: {torch.__version__}")
        
        # 显示模型当前使用的设备
        if hasattr(trainer.model, 'device'):
            print(f"🎯 模型设备: {trainer.model.device}")
        
        # GPU 信息
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
            
            print(f"🚀 GPU 信息:")
            print(f"   设备数量: {device_count}")
            print(f"   当前设备: {current_device} ({device_name})")
            print(f"   已分配内存: {memory_allocated:.2f} GB")
            print(f"   已保留内存: {memory_reserved:.2f} GB")
            
            if self.show_detailed:
                for i in range(device_count):
                    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  未检测到可用的 GPU，使用 CPU")

        # CPU 信息
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        memory = psutil.virtual_memory()
        
        print(f"💻 CPU 信息:")
        print(f"   物理核心数: {cpu_count}")
        print(f"   逻辑核心数: {cpu_count_logical}")
        print(f"   总内存: {memory.total / 1024**3:.1f} GB")
        print(f"   可用内存: {memory.available / 1024**3:.1f} GB")
        
        if self.show_detailed:
            print(f"   CPU 使用率: {psutil.cpu_percent(interval=1):.1f}%")
            print(f"   内存使用率: {memory.percent:.1f}%")
        
        print()  # 空行分隔


class TrainingProgressCallback(Callback):
    """显示训练进度信息的回调"""
    
    def __init__(self, show_time_estimate=True):
        super().__init__()
        self.show_time_estimate = show_time_estimate
        self.start_time:float = 0.0
        self.epoch_start_time:float = 0.0  
    
    def on_train_begin(self, trainer: "Trainer"):
        self.start_time = time.time()
        
        print("📊 训练配置:")
        print(f"   总轮次: {trainer.max_epochs}")
        print(f"   训练批次数: {trainer.num_train_batches}")
        if trainer.val_loader:
            print(f"   验证批次数: {trainer.num_val_batches}")
        
        # 显示模型信息 - 处理懒加载模块
        try:
            total_params = sum(p.numel() for p in trainer.model.parameters())
            trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            print(f"   模型参数总数: {total_params:,}")
            print(f"   可训练参数: {trainable_params:,}")
        except ValueError as e:
            if "uninitialized parameter" in str(e):
                print("   模型参数: 使用懒加载模块，将在第一次前向传播后显示")
            else:
                raise e
        
        print(f"   优化器: {type(trainer.model.configure_optimizers()).__name__}")
        print()
    
    def on_epoch_begin(self, trainer: "Trainer"):
        self.epoch_start_time = time.time()
        print(f"🔄 Epoch {trainer.epoch + 1}/{trainer.max_epochs}")
    
    def on_epoch_end(self, trainer: "Trainer", train_metrics: dict, val_metrics: dict):
        import time
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            print(f"   训练损失: {train_metrics['loss']:.4f}")
            if val_metrics:
                print(f"   验证损失: {val_metrics['loss']:.4f}")
            print(f"   用时: {epoch_time:.2f}s")
            
            # 估算剩余时间
            if self.show_time_estimate and trainer.epoch > 0:
                elapsed_time = time.time() - self.start_time
                avg_time_per_epoch = elapsed_time / (trainer.epoch + 1)
                remaining_epochs = trainer.max_epochs - trainer.epoch - 1
                eta = remaining_epochs * avg_time_per_epoch
                
                if eta > 60:
                    print(f"   预计剩余时间: {eta/60:.1f} 分钟")
                else:
                    print(f"   预计剩余时间: {eta:.0f} 秒")
            print()
    
    def on_train_end(self, trainer: "Trainer"):
        import time
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"✅ 训练完成！总用时: {total_time/60:.1f} 分钟")


class ModelSummaryCallback(Callback):
    """显示模型架构摘要的回调"""
    
    def on_train_begin(self, trainer: "Trainer"):
        print("🏗️  模型架构:")
        print(trainer.model)
        print()


class MemoryMonitorCallback(Callback):
    """监控内存使用的回调"""
    
    def __init__(self, check_every_n_epochs=1):
        super().__init__()
        self.check_every_n_epochs = check_every_n_epochs
    
    def on_epoch_end(self, trainer: "Trainer", train_metrics: dict, val_metrics: dict):
        if (trainer.epoch + 1) % self.check_every_n_epochs == 0:
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"   GPU 内存 - 已分配: {memory_allocated:.2f}GB, 已保留: {memory_reserved:.2f}GB")
            
            memory = psutil.virtual_memory()
            print(f"   系统内存使用率: {memory.percent:.1f}%")


# 创建一个便捷的组合回调
class DefaultCallbacks:
    """提供常用回调组合的工厂类"""
    
    @staticmethod
    def basic():
        """基础回调组合"""
        return [
            SystemInfoCallback(),
            TrainingProgressCallback(),
        ]
    
    @staticmethod
    def detailed():
        """详细回调组合"""
        return [
            SystemInfoCallback(show_detailed=True),
            ModelSummaryCallback(),
            TrainingProgressCallback(),
            MemoryMonitorCallback(),
        ]
    
    @staticmethod
    def with_live_loss():
        """包含实时损失图的回调组合"""
        return [
            SystemInfoCallback(),
            TrainingProgressCallback(),
            LiveLossCallback(),
        ]


class Trainer:
    """The conductor that orchestrates the training process."""

    _model: Optional[ModernModule] = None
    _train_loader: Optional[torch.utils.data.DataLoader] = None
    _val_loader: Optional[torch.utils.data.DataLoader] = None
    def __init__(self, max_epochs: int, callbacks: List[Callback] = []):
        self.max_epochs = max_epochs
        self.callbacks = callbacks or []

        # 明确声明将在 fit 方法中设置的属性
        self.epoch: int = 0
        self.train_batch_idx: int = 0
        self.val_batch_idx: int = 0
        self.num_train_batches: int = 0
        self.num_val_batches: int = 0

    @property
    def model(self) -> ModernModule:
        assert self._model is not None, "Model must be set before accessing."
        return self._model
    @model.setter
    def model(self, value: ModernModule):
        if not isinstance(value, ModernModule):
            raise TypeError("Model must be an instance of ModernModule.")
        self._model = value
    @property
    def train_loader(self) -> torch.utils.data.DataLoader:
        assert self._train_loader is not None, "Train loader must be set before accessing."
        return self._train_loader
    @train_loader.setter
    def train_loader(self, value: torch.utils.data.DataLoader):
        self._train_loader = value
        
    @property
    def val_loader(self) -> Optional[torch.utils.data.DataLoader]:
        assert self._val_loader is not None, "Validation loader must be set before accessing."
        return self._val_loader
    @val_loader.setter
    def val_loader(self, value: Optional[torch.utils.data.DataLoader]):
        self._val_loader = value

    def fit(self, model: ModernModule, train_loader, val_loader=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 将 self (trainer) 注入到回调中，让回调可以访问训练状态
        for cb in self.callbacks:
            cb.trainer = self

        optimizer = self.model.configure_optimizers()

        self.num_train_batches = len(self.train_loader) if self.train_loader else 0
        self.num_val_batches = len(self.val_loader) if self.val_loader else 0

        # --- 训练循环 ---
        self._dispatch("on_train_begin")
        for self.epoch in range(self.max_epochs):
            self._dispatch("on_epoch_begin")

            # Training loop for one epoch

            self.model.train()
            total_train_loss = 0.0
            for self.train_batch_idx, batch in enumerate(self.train_loader):
                self._dispatch("on_batch_begin")
                optimizer.zero_grad()
                loss = self.model.training_step(batch)
                loss.backward()
                optimizer.step()
                self._dispatch("on_batch_end", loss)

                # 确保 loss.item() 返回 float 类型
                loss_value = loss.item()
                if isinstance(loss_value, bool):
                    raise TypeError(
                        f"Loss function returned bool value: {loss_value}. Check your loss function implementation."
                    )
                total_train_loss += float(loss_value)

            # 避免除零错误
            train_metrics = {"loss": total_train_loss / max(self.num_train_batches, 1)}

            # Validation loop for one epoch
            val_metrics = {}
            if self.val_loader and self.num_val_batches > 0:
                self.model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for self.val_batch_idx, batch in enumerate(self.val_loader):
                        loss = self.model.validation_step(batch)

                        # 确保 loss.item() 返回 float 类型
                        loss_value = loss.item()
                        if isinstance(loss_value, bool):
                            raise TypeError(
                                f"Validation loss function returned bool value: {loss_value}. Check your loss function implementation."
                            )
                        total_val_loss += float(loss_value)

                val_metrics = {"loss": total_val_loss / self.num_val_batches}

            self._dispatch("on_epoch_end", train_metrics, val_metrics)

        self._dispatch("on_train_end")

    def _dispatch(self, event_name: str, *args):
        """Calls the corresponding method on all callbacks."""
        for cb in self.callbacks:
            if hasattr(cb, event_name):
                getattr(cb, event_name)(self, *args)
