# infoNN

使用InfoNet模型预测LeNet,ResNeXt等模型互信息(MI),可视化表示并由互信息展示模型在训练过程中的变化

## 技术栈

- **主要语言**: Jupyter Notebook
- **开发环境**: Python
- **机器学习框架**: PyTorch

## 项目结构

```
infoNN/
├── notebooks/         # 主要的Jupyter notebook 文件
├── data/              # 数据文件
├── models/            # 模型文件
├── src/             # 工具函数
└── README.md          # 项目说明
```

## 快速开始

### 环境要求

- Python 3.13(开发下版本)
- Jupyter Notebook
- 相关机器学习库 (requirements.txt 中列出)

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/sevmeowple/infoNN.git
cd infoNN
```

2. 安装依赖
```bash
uv sync
```

3. 启动 Jupyter Notebook
```bash
jupyter notebook
```

## 主要功能

- 信息神经网络模型实现
- 数据预处理和特征工程
- 模型训练和评估
- 结果可视化和分析

## 使用说明

1. 打开相关的 Jupyter Notebook 文件
2. 按照 notebook 中的步骤运行代码
3. 根据需要调整参数和配置
