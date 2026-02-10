# Dual-Adapter Federated Learning for Public Security Governance

面向公安"条块数据"的双适配器联邦大模型实验项目

## 项目概述

本项目实现了一个创新的双适配器联邦学习架构，用于解决公安治理场景中的"条块分割"问题。系统基于 Qwen2.5-7B-Instruct，使用 LoRA 技术实现参数高效微调。

### 核心创新

- **全局适配器（条适配器）**：学习通用法律知识，参与联邦聚合
- **本地适配器（块适配器）**：学习地方特色政策，保留在本地，保护隐私

## 快速开始

### 本地开发环境

#### 环境要求

- Python 3.10+
- CUDA GPU (推荐 24GB，最低 8GB)
- HuggingFace 账号

#### 安装

```bash
# 克隆项目
git clone <repository_url>
cd policeModel

# 安装依赖
pip install -r requirements.txt

# 设置 HuggingFace Token
export HF_TOKEN="your_token_here"
```

#### 运行实验

```bash
# 训练双适配器模型
python experiments/exp001_dual_adapter_fl/train.py

# 使用自定义配置
python experiments/exp001_dual_adapter_fl/train.py --config path/to/config.yaml
```

### 远程服务器部署

如果你需要在远程 GPU 服务器上运行实验，我们提供了自动化部署脚本：

```bash
# 1. 上传代码到远程服务器
scp -r policeModel/ user@gpu-server:/path/to/workspace/

# 2. SSH 登录并运行自动设置
ssh user@gpu-server
cd /path/to/workspace/policeModel
bash scripts/setup_remote.sh --token YOUR_HF_TOKEN

# 3. 验证数据
python scripts/validate_data.py

# 4. 运行实验
bash scripts/run_experiment.sh
```

详细的部署指南请参考 [DEPLOYMENT.md](DEPLOYMENT.md)。

### 部署脚本说明

项目提供了三个自动化脚本：

1. **`scripts/setup_remote.sh`** - 环境设置脚本
   - 自动检查系统依赖
   - 创建虚拟环境并安装所有依赖
   - 配置 HuggingFace Token
   - 验证安装完整性

2. **`scripts/validate_data.py`** - 数据验证脚本
   - 检查训练数据和测试数据
   - 验证 Alpaca 格式规范
   - 提供详细的错误报告

3. **`scripts/run_experiment.sh`** - 实验运行脚本
   - 自动激活环境并检查配置
   - 运行训练并保存日志
   - 提供错误诊断和建议

## 项目结构

```
policeModel/
├── src/                    # 核心实现代码
│   ├── models/            # 模型管理（基座模型、双适配器）
│   ├── federated/         # 联邦学习核心（客户端、服务器、聚合）
│   ├── data/              # 数据处理
│   └── utils/             # 工具函数
├── tools/                 # 实验工具
│   ├── runners/           # 实验运行器
│   ├── evaluators/        # 评估工具
│   └── visualizers/       # 可视化工具
├── experiments/           # 实验定义
│   └── exp001_dual_adapter_fl/
├── data/                  # 训练数据
│   ├── files/            # 原始 PDF 文档
│   └── rule_data/        # 处理后的训练数据
└── results/              # 实验结果（gitignored）
```

## 核心特性

### 1. 自适应显存检测 ✅
- 自动检测 GPU 显存
- < 16GB: 自动启用 4-bit 量化
- ≥ 16GB: 使用 bfloat16 原生精度

### 2. 全线性层 LoRA ✅
- `target_modules="all-linear"`
- 应用于 Qwen2.5 的所有线性层
- 最大化模型适应能力

### 3. 系统提示词注入 ✅
- 评估时自动注入城市身份提示词
- 增强模型对不同司法管辖区的区分能力

### 4. 多模式支持 ✅
- `dual_adapter`: 双适配器架构（本方法）
- `standard_fedavg`: 标准联邦平均（基线）
- `local_only`: 仅本地训练（基线）

## 配置说明

编辑 `experiments/exp001_dual_adapter_fl/config.yaml`:

```yaml
model:
  base_model: "Qwen/Qwen2.5-7B-Instruct"
  quantization: "auto"  # 自动检测
  lora_config:
    target_modules: "all-linear"  # 所有线性层

federated:
  mode: "dual_adapter"  # 或 "standard_fedavg", "local_only"
  num_rounds: 5
  clients:
    - id: "strict"
      system_prompt: "你是上海市公安局的政务助手..."
    - id: "service"
      system_prompt: "你是石家庄市公安局的政务助手..."
```

## 硬件要求

| 配置 | GPU | VRAM | 量化 |
|------|-----|------|------|
| 最低 | RTX 3070 | 8GB | 4-bit |
| 推荐 | RTX 4090 | 24GB | BF16 |

系统会自动检测并选择合适的量化策略。

## 实验结果

训练完成后，结果保存在 `results/exp001_dual_adapter_fl/`:

```
results/exp001_dual_adapter_fl/
├── checkpoints/           # 模型检查点
│   ├── round_1/
│   ├── round_2/
│   └── final_adapters/   # 最终适配器
├── logs/                 # 训练日志
└── experiment_summary.json  # 实验摘要
```

## 文档

- [实验计划](docs/experimentPlan.md)
- [架构设计](.kiro/steering/architecture.md)
- [技术栈](.kiro/steering/tech.md)
- [部署指南](DEPLOYMENT.md)

## 开发进度

查看 [PROGRESS.md](PROGRESS.md) 了解当前实现进度。

## 许可证

[添加许可证信息]

## 引用

如果使用本项目，请引用：

```
[添加引用信息]
```
