from IPython.core.getipython import get_ipython
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDownloader:
    """统一的数据下载管理器"""

    def __init__(self, data_root: str = "./data"):
        """
        初始化数据下载管理器

        Args:
            data_root: 数据存储的根目录
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

        # 预定义的数据集配置
        self.dataset_configs: Dict[str, Dict[str, Any]] = {
            "mnist": {
                "class": datasets.MNIST,
                "transform": transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
                "num_classes": 10,
                "input_shape": (1, 28, 28),
                "description": "手写数字识别数据集 (28x28灰度图像)",
            },
            "cifar10": {
                "class": datasets.CIFAR10,
                "transform": transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                ),
                "num_classes": 10,
                "input_shape": (3, 32, 32),
                "description": "物体分类数据集 (32x32彩色图像)",
            },
            "fashionmnist": {
                "class": datasets.FashionMNIST,
                "transform": transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
                ),
                "num_classes": 10,
                "input_shape": (1, 28, 28),
                "description": "时尚物品分类数据集 (28x28灰度图像)",
            },
        }

    def download_dataset(
        self, dataset_name: str, download: bool = True
    ) -> Dict[str, Any]:
        """下载指定数据集"""
        if dataset_name.lower() not in self.dataset_configs:
            raise ValueError(
                f"不支持的数据集: {dataset_name}. "
                f"支持的数据集: {list(self.dataset_configs.keys())}"
            )

        config = self.dataset_configs[dataset_name.lower()]
        dataset_class = config["class"]
        transform = config["transform"]

        logger.info(f"下载/加载数据集: {dataset_name}")

        # 下载训练集和测试集
        train_dataset = dataset_class(
            root=self.data_root, train=True, download=download, transform=transform
        )
        test_dataset = dataset_class(
            root=self.data_root, train=False, download=download, transform=transform
        )

        logger.info(f"✅ {dataset_name} 下载完成!")
        logger.info(f"   训练样本数: {len(train_dataset)}")
        logger.info(f"   测试样本数: {len(test_dataset)}")
        logger.info(f"   类别数: {config['num_classes']}")
        logger.info(f"   输入形状: {config['input_shape']}")

        return {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "num_classes": config["num_classes"],
            "input_shape": config["input_shape"],
            "dataset_name": dataset_name,
        }

    def create_dataloaders(
        self,
        dataset_info: Dict[str, Any],
        batch_size: int = 64,
        val_split: float = 0.1,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器"""
        train_dataset = dataset_info["train_dataset"]
        test_dataset = dataset_info["test_dataset"]

        # 划分训练集和验证集
        train_size = len(train_dataset)
        val_size = int(train_size * val_split)
        train_size = train_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        logger.info(f"📊 数据加载器创建完成:")
        logger.info(f"   训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
        logger.info(f"   验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")
        logger.info(f"   测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")
        logger.info(f"   批次大小: {batch_size}")

        return train_loader, val_loader, test_loader

    def get_sample_data(self, dataset_info: Dict[str, Any], num_samples: int = 5):
        """获取样本数据用于可视化"""
        train_dataset = dataset_info["train_dataset"]
        samples = []
        for i in range(min(num_samples, len(train_dataset))):
            image, label = train_dataset[i]
            samples.append((image, label))
        return samples

    def visualize_samples(self, dataset_info: Dict[str, Any], num_samples: int = 8):
        """可视化数据样本（保存到文件）"""
        try:
            import matplotlib

            matplotlib.use("Agg")  # 强制使用非交互式后端
            import matplotlib.pyplot as plt

            samples = self.get_sample_data(dataset_info, num_samples)
            dataset_name = dataset_info["dataset_name"]

            # 类别标签设置
            if dataset_name.lower() == "mnist":
                class_names = [str(i) for i in range(10)]
            elif dataset_name.lower() == "cifar10":
                class_names = [
                    "plane",
                    "car",
                    "bird",
                    "cat",
                    "deer",
                    "dog",
                    "frog",
                    "horse",
                    "ship",
                    "truck",
                ]
            elif dataset_name.lower() == "fashionmnist":
                class_names = [
                    "T-shirt",
                    "Trouser",
                    "Pullover",
                    "Dress",
                    "Coat",
                    "Sandal",
                    "Shirt",
                    "Sneaker",
                    "Bag",
                    "Ankle boot",
                ]
            else:
                class_names = [str(i) for i in range(dataset_info["num_classes"])]

            # 创建图形
            cols = min(4, num_samples)
            rows = (num_samples + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))

            # 使用英文标题避免字体问题
            fig.suptitle(f"{dataset_name.upper()} Data Samples", fontsize=16)

            # 处理单行或单列情况
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            for i, (image, label) in enumerate(samples):
                if i >= len(axes):
                    break
                ax = axes[i]
                if image.shape[0] == 1:
                    ax.imshow(image.squeeze(), cmap="gray")
                else:
                    ax.imshow(image.permute(1, 2, 0))
                ax.set_title(f"Label: {class_names[label]}")
                ax.axis("off")

            # 隐藏多余的子图
            for i in range(len(samples), len(axes)):
                axes[i].axis("off")

            plt.tight_layout()

            # 使用正确的路径
            output_path = self.data_root / f"{dataset_name}_samples.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"📊 Sample images saved to: {output_path}")

        except Exception as e:
            print(f"⚠️  Failed to save images, showing text info: {e}")
            # 降级到文本显示
            self._show_text_samples(dataset_info, num_samples)

    def visualize_samples_notebook(self, dataset_info: Dict[str, Any], num_samples: int = 8):
        """专门为Jupyter Notebook环境设计的可视化函数"""
        try:
            # 在notebook中使用内联显示
            import matplotlib.pyplot as plt
            import matplotlib
            
            # 确保在notebook中正确显示
            if 'ipykernel' in str(type(get_ipython())):
                matplotlib.use('inline')
            
            # 启用notebook内联显示
            from IPython.display import display
            
            samples = self.get_sample_data(dataset_info, num_samples)
            dataset_name = dataset_info["dataset_name"]

            # 类别标签设置
            if dataset_name.lower() == "mnist":
                class_names = [str(i) for i in range(10)]
                title = "MNIST 手写数字样本"
            elif dataset_name.lower() == "cifar10":
                class_names = [
                    "飞机", "汽车", "鸟", "猫", "鹿", 
                    "狗", "青蛙", "马", "船", "卡车"
                ]
                title = "CIFAR-10 图像分类样本"
            elif dataset_name.lower() == "fashionmnist":
                class_names = [
                    "T恤", "裤子", "套衫", "连衣裙", "外套",
                    "凉鞋", "衬衫", "运动鞋", "包", "短靴"
                ]
                title = "Fashion-MNIST 时尚物品样本"
            else:
                class_names = [str(i) for i in range(dataset_info["num_classes"])]
                title = f"{dataset_name.upper()} 数据样本"

            # 创建图形
            cols = min(4, num_samples)
            rows = (num_samples + cols - 1) // cols
            
            plt.figure(figsize=(12, 3 * rows))
            plt.suptitle(title, fontsize=16, y=0.98)

            for i, (image, label) in enumerate(samples):
                plt.subplot(rows, cols, i + 1)
                
                # 显示图像
                if image.shape[0] == 1:  # 灰度图像
                    plt.imshow(image.squeeze(), cmap="gray")
                else:  # 彩色图像
                    # 将tensor从(C,H,W)转换为(H,W,C)
                    img_np = image.permute(1, 2, 0)
                    # 反标准化显示原始图像（可选）
                    plt.imshow(img_np)
                
                plt.title(f"标签: {class_names[label]}", fontsize=12)
                plt.axis('off')

            plt.tight_layout()
            plt.show()
            
            print(f"📊 显示了 {len(samples)} 个 {dataset_name.upper()} 样本")

        except Exception as e:
            print(f"⚠️  Notebook可视化失败: {e}")
            print("回退到文本显示模式...")
            self._show_text_samples(dataset_info, num_samples)

    def visualize_samples_simple(self, dataset_info: Dict[str, Any], num_samples: int = 8):
        """简化版可视化函数，兼容性更好"""
        try:
            import matplotlib.pyplot as plt
            
            samples = self.get_sample_data(dataset_info, num_samples)
            dataset_name = dataset_info["dataset_name"]

            # 简单的类别名称（避免中文字体问题）
            class_names = [str(i) for i in range(dataset_info["num_classes"])]

            # 创建图形
            cols = min(4, num_samples)
            rows = (num_samples + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(10, 2.5 * rows))
            fig.suptitle(f"{dataset_name.upper()} Samples", fontsize=14)

            # 处理axes为单个对象的情况
            if num_samples == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            for i, (image, label) in enumerate(samples):
                if i < len(axes):
                    ax = axes[i]
                    
                    if image.shape[0] == 1:
                        ax.imshow(image.squeeze(), cmap="gray")
                    else:
                        ax.imshow(image.permute(1, 2, 0))
                    
                    ax.set_title(f"Class: {label}")
                    ax.axis('off')

            # 隐藏多余的子图
            for i in range(len(samples), len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"⚠️  简化可视化失败: {e}")
            self._show_text_samples(dataset_info, num_samples)


    def _show_text_samples(self, dataset_info: Dict[str, Any], num_samples: int):
        """文本形式显示样本信息"""
        samples = self.get_sample_data(dataset_info, num_samples)
        dataset_name = dataset_info["dataset_name"]

        # 设置类别标签
        if dataset_name.lower() == "mnist":
            class_names = [str(i) for i in range(10)]
        elif dataset_name.lower() == "cifar10":
            class_names = [
                "plane",
                "car",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]
        elif dataset_name.lower() == "fashionmnist":
            class_names = [
                "T-shirt",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ]
        else:
            class_names = [str(i) for i in range(dataset_info["num_classes"])]

        print(f"\n📊 {dataset_name.upper()} Sample Information:")
        print("=" * 60)
        for i, (image, label) in enumerate(samples):
            print(
                f"Sample {i + 1:2d}: Label={class_names[label]:10s} | Shape={str(image.shape):15s} | Range=[{image.min():.3f}, {image.max():.3f}]"
            )
        print("=" * 60)

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """获取数据集信息（不下载）"""
        if dataset_name.lower() not in self.dataset_configs:
            raise ValueError(f"不支持的数据集: {dataset_name}")

        config = self.dataset_configs[dataset_name.lower()]
        return {
            "num_classes": config["num_classes"],
            "input_shape": config["input_shape"],
            "dataset_name": dataset_name,
        }

    def list_available_datasets(self):
        """列出所有可用的数据集"""
        print("\n📋 可用数据集:")
        for i, (name, config) in enumerate(self.dataset_configs.items(), 1):
            print(f"   {i}. {name.upper()}: {config['description']}")
            print(
                f"      类别数: {config['num_classes']}, 输入形状: {config['input_shape']}"
            )


class InteractiveCLI:
    """交互式命令行界面"""

    def __init__(self):
        self.data_manager = DataDownloader()
        self.config = {}

    def clear_screen(self):
        """清屏"""
        os.system("cls" if os.name == "nt" else "clear")

    def print_header(self):
        """打印标题"""
        print("=" * 60)
        print("🚀 深度学习数据下载管理器")
        print("=" * 60)

    def get_user_choice(self, prompt: str, choices: list, default: int = 0) -> int:
        """获取用户选择"""
        while True:
            try:
                print(f"\n{prompt}")
                for i, choice in enumerate(choices, 1):
                    print(f"  {i}. {choice}")

                if default > 0:
                    user_input = input(
                        f"\n请选择 (1-{len(choices)}, 默认 {default}): "
                    ).strip()
                    if not user_input:
                        return default - 1
                else:
                    user_input = input(f"\n请选择 (1-{len(choices)}): ").strip()

                choice = int(user_input) - 1
                if 0 <= choice < len(choices):
                    return choice

                print(f"❌ 请输入 1-{len(choices)} 之间的数字")

            except ValueError:
                print("❌ 请输入有效的数字")

            except KeyboardInterrupt:
                print("\n\n👋 再见!")
                raise SystemExit(0)  # 明确表示程序退出

        # 继续下一次循环

    def get_user_input(self, prompt: str, input_type: type = str, default: Any = None):
        """获取用户输入"""
        while True:
            try:
                if default is not None:
                    user_input = input(f"{prompt} (默认: {default}): ").strip()
                    if not user_input:
                        return default
                else:
                    user_input = input(f"{prompt}: ").strip()

                if input_type is bool:
                    return user_input.lower() in ["y", "yes", "true", "1", "是"]
                elif input_type is int:
                    return int(user_input)
                elif input_type is float:
                    return float(user_input)
                else:
                    return user_input
            except ValueError:
                print(f"❌ 请输入有效的{input_type.__name__}类型")
            except KeyboardInterrupt:
                print("\n\n👋 再见!")
                exit(0)

    def select_dataset(self):
        """选择数据集"""
        print("\n🎯 步骤 1: 选择数据集")
        self.data_manager.list_available_datasets()

        dataset_names = list(self.data_manager.dataset_configs.keys())
        choice = self.get_user_choice(
            "选择要下载的数据集:", [name.upper() for name in dataset_names], default=1
        )

        self.config["dataset_name"] = dataset_names[choice]
        print(f"✅ 已选择: {self.config['dataset_name'].upper()}")

    def configure_data_loading(self):
        """配置数据加载参数"""
        print("\n⚙️  步骤 2: 配置数据加载参数")

        # 批次大小
        batch_sizes = [16, 32, 64, 128, 256]
        print("\n选择批次大小:")
        choice = self.get_user_choice(
            "批次大小影响训练速度和内存使用:",
            [
                f"{size} (推荐用于{'小模型' if size <= 64 else '大模型'})"
                for size in batch_sizes
            ],
            default=3,
        )  # 默认64
        self.config["batch_size"] = batch_sizes[choice]

        # 验证集比例
        val_splits = [0.05, 0.1, 0.15, 0.2, 0.25]
        print("\n选择验证集比例:")
        choice = self.get_user_choice(
            "验证集用于模型选择和超参调优:",
            [f"{split * 100:.0f}%" for split in val_splits],
            default=2,
        )  # 默认10%
        self.config["val_split"] = val_splits[choice]

        # 数据shuffle
        shuffle_choice = self.get_user_choice(
            "是否打乱训练数据?", ["是 (推荐，提高训练效果)", "否"], default=1
        )
        self.config["shuffle"] = shuffle_choice == 0

        # 工作进程数
        num_workers_options = [0, 2, 4, 8]
        print("\n选择数据加载工作进程数:")
        choice = self.get_user_choice(
            "更多进程可加速数据加载，但占用更多内存:",
            [f"{nw} 进程{'(单线程)' if nw == 0 else ''}" for nw in num_workers_options],
            default=1,
        )  # 默认0
        self.config["num_workers"] = num_workers_options[choice]

        print(f"\n✅ 数据加载配置完成:")
        print(f"   批次大小: {self.config['batch_size']}")
        print(f"   验证集比例: {self.config['val_split'] * 100:.0f}%")
        print(f"   数据打乱: {'是' if self.config['shuffle'] else '否'}")
        print(f"   工作进程: {self.config['num_workers']}")

    def configure_download_options(self):
        """配置下载选项"""
        print("\n📥 步骤 3: 配置下载选项")

        # 数据存储路径
        default_path = "./data"
        data_path = self.get_user_input("数据存储路径", str, default_path)
        self.config["data_root"] = data_path

        # 检查数据是否已存在
        data_exists = self._quick_check_data_exists(
            data_path, self.config["dataset_name"]
        )

        if data_exists:
            print("✅ 检测到已有数据")
            force_download = self.get_user_choice(
                "是否重新下载数据?",
                ["是 (重新下载)", "否 (使用现有数据)"],
                default=2,
            )
            self.config["download"] = force_download == 0
        else:
            print("ℹ️  未检测到数据，将自动下载")
            self.config["download"] = True

        print(f"\n✅ 下载配置完成:")
        print(f"   存储路径: {self.config['data_root']}")
        print(
            f"   下载设置: {'重新下载' if self.config['download'] else '使用现有数据'}"
        )

    def _quick_check_data_exists(self, data_root: str, dataset_name: str) -> bool:
        """快速检查数据是否存在"""
        data_path = Path(data_root)

        # 检查对应的数据集文件夹是否存在且不为空
        if dataset_name.lower() == "mnist":
            return (data_path / "MNIST").exists() and any(
                (data_path / "MNIST").iterdir()
            )
        elif dataset_name.lower() == "cifar10":
            return (data_path / "cifar-10-batches-py").exists() and any(
                (data_path / "cifar-10-batches-py").iterdir()
            )
        elif dataset_name.lower() == "fashionmnist":
            return (data_path / "FashionMNIST").exists() and any(
                (data_path / "FashionMNIST").iterdir()
            )

        return False

    def configure_visualization(self):
        """配置可视化选项"""
        print("\n🖼️  步骤 4: 配置可视化选项")

        # 是否显示样本
        show_samples = self.get_user_choice(
            "是否显示数据样本?", ["是 (显示示例图片)", "否"], default=1
        )
        self.config["show_samples"] = show_samples == 0

        if self.config["show_samples"]:
            # 显示样本数量
            sample_counts = [4, 8, 12, 16]
            choice = self.get_user_choice(
                "显示多少个样本?", [f"{count} 个" for count in sample_counts], default=2
            )
            self.config["num_samples"] = sample_counts[choice]
        else:
            self.config["num_samples"] = 0

        print(f"\n✅ 可视化配置完成:")
        print(f"   显示样本: {'是' if self.config['show_samples'] else '否'}")
        if self.config["show_samples"]:
            print(f"   样本数量: {self.config['num_samples']}")

    def show_summary(self):
        """显示配置摘要"""
        print("\n📋 配置摘要:")
        print("=" * 40)
        print(f"数据集: {self.config['dataset_name'].upper()}")
        print(f"存储路径: {self.config['data_root']}")
        print(f"批次大小: {self.config['batch_size']}")
        print(f"验证集比例: {self.config['val_split'] * 100:.0f}%")
        print(f"数据打乱: {'是' if self.config['shuffle'] else '否'}")
        print(f"工作进程: {self.config['num_workers']}")
        print(f"重新下载: {'是' if self.config['download'] else '否'}")
        print(f"显示样本: {'是' if self.config['show_samples'] else '否'}")
        print("=" * 40)

        confirm = self.get_user_choice(
            "确认开始下载和处理?", ["是，开始处理", "否，重新配置"], default=1
        )
        return confirm == 0

    def execute_download(self):
        """执行下载和处理"""
        print("\n🚀 开始处理...")

        try:
            # 创建数据管理器
            self.data_manager = DataDownloader(data_root=self.config["data_root"])

            # 下载数据集
            print(f"\n📥 下载 {self.config['dataset_name'].upper()} 数据集...")
            dataset_info = self.data_manager.download_dataset(
                self.config["dataset_name"], download=self.config["download"]
            )

            # 创建数据加载器
            print("\n🔄 创建数据加载器...")
            train_loader, val_loader, test_loader = (
                self.data_manager.create_dataloaders(
                    dataset_info,
                    batch_size=self.config["batch_size"],
                    val_split=self.config["val_split"],
                    shuffle=self.config["shuffle"],
                    num_workers=self.config["num_workers"],
                )
            )

            # 测试数据加载
            print("\n🧪 测试数据加载...")
            for batch_idx, (data, target) in enumerate(train_loader):
                print(
                    f"   批次 {batch_idx}: 数据形状={data.shape}, 标签形状={target.shape}"
                )
                if batch_idx >= 2:
                    break

            # 可视化样本
            if self.config["show_samples"]:
                print(f"\n🖼️  显示 {self.config['num_samples']} 个样本...")
                self.data_manager.visualize_samples(
                    dataset_info, self.config["num_samples"]
                )

            print("\n✅ 处理完成!")
            print("\n📊 数据加载器已准备就绪，可用于训练模型!")

            # 保存配置到文件
            self.save_config()

        except Exception as e:
            print(f"\n❌ 处理失败: {e}")

    def save_config(self):
        """保存配置到文件"""
        try:
            import json

            config_file = Path(self.config["data_root"]) / "config.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"\n💾 配置已保存到: {config_file}")
        except Exception as e:
            print(f"\n⚠️  配置保存失败: {e}")

    def run(self):
        """运行交互式CLI"""
        try:
            while True:
                self.clear_screen()
                self.print_header()

                # 步骤1: 选择数据集
                self.select_dataset()

                # 步骤2: 配置数据加载
                self.configure_data_loading()

                # 步骤3: 配置下载选项
                self.configure_download_options()

                # 步骤4: 配置可视化
                self.configure_visualization()

                # 显示摘要并确认
                if self.show_summary():
                    self.execute_download()

                    # 询问是否继续
                    continue_choice = self.get_user_choice(
                        "\n想要处理其他数据集吗?", ["是，继续", "否，退出"], default=2
                    )
                    if continue_choice == 1:
                        break
                    else:
                        self.config = {}  # 重置配置
                else:
                    self.config = {}  # 重置配置
                    continue

        except KeyboardInterrupt:
            print("\n\n👋 再见!")
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


def main():
    """主函数 - 启动交互式CLI"""
    cli = InteractiveCLI()
    cli.run()


if __name__ == "__main__":
    main()
