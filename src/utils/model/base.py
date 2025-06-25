from pydantic import BaseModel, Field

import torch
from torch import nn

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
    # 还可以包含其他配置，如 dropout_prob, weight_decay等


class ModernModule(nn.Module):
    """A clean, decoupled base class for models."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 在子类中，你需要定义 self.net
        # 例如: self.net = nn.Sequential(...)

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
