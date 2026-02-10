# 远程部署指南 (Remote Deployment Guide)

## 快速开始 (Quick Start)

如果你已经熟悉环境配置，可以使用以下命令快速开始：

```bash
# 1. 上传代码到远程服务器
scp -r policeModel/ user@remote-server:/path/to/workspace/

# 2. SSH 登录到远程服务器
ssh user@remote-server
cd /path/to/workspace/policeModel

# 3. 运行自动设置脚本
bash scripts/setup_remote.sh --token YOUR_HF_TOKEN

# 4. 验证数据（可选）
python scripts/validate_data.py

# 5. 运行实验
bash scripts/run_experiment.sh

# 6. 监控进度
tail -f results/exp001_dual_adapter_fl/logs/training_*.log
```

完整的部署步骤请参考下面的详细说明。

---

## 前置要求 (Prerequisites)

### 硬件要求
- **GPU**: NVIDIA RTX 4090 (24GB) 推荐，或 RTX 3070 (8GB) 最低配置
- **内存**: 32GB+ 推荐
- **存储**: 50GB+ 可用空间（用于模型缓存和结果）

### 软件要求
- **操作系统**: Linux (Ubuntu 20.04+ 推荐)
- **Python**: 3.10+
- **CUDA**: 11.8+ (与 PyTorch 兼容)
- **网络**: 稳定的互联网连接（首次运行需下载 ~14GB 模型）

## 部署步骤 (Deployment Steps)

### 1. 上传代码到远程服务器

```bash
# 在本地打包代码
tar -czf policeModel.tar.gz \
  --exclude='results' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  .

# 上传到远程服务器
scp policeModel.tar.gz user@remote-server:/path/to/workspace/

# 在远程服务器上解压
ssh user@remote-server
cd /path/to/workspace/
tar -xzf policeModel.tar.gz
```

### 2. 设置环境

#### 自动设置（推荐）

```bash
# 运行自动设置脚本（包含环境检查、依赖安装、HF Token 配置）
bash scripts/setup_remote.sh

# 如果需要在脚本中直接设置 HuggingFace Token
bash scripts/setup_remote.sh --token your_huggingface_token_here
```

自动设置脚本会：
- ✓ 检查系统依赖（Python 3.10+, pip, venv）
- ✓ 创建虚拟环境
- ✓ 安装所有 Python 依赖
- ✓ 验证 PyTorch 和 CUDA
- ✓ 配置 HuggingFace Token
- ✓ 创建必要的目录结构
- ✓ 验证所有包安装成功

#### 手动设置

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 升级 pip
pip install --upgrade pip setuptools wheel

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 3. 配置 HuggingFace Token

```bash
# 方法1: 设置环境变量
export HF_TOKEN="your_huggingface_token_here"

# 方法2: 使用 huggingface-cli
pip install huggingface_hub
huggingface-cli login

# 方法3: 在 config.yaml 中配置
# 编辑 experiments/exp001_dual_adapter_fl/config.yaml
# 添加: hf_token: "your_token"
```

### 4. 验证数据

```bash
# 检查训练数据和测试数据是否存在且格式正确
python scripts/validate_data.py

# 详细模式（显示样本数据）
python scripts/validate_data.py --verbose
```

预期输出：
```
[INFO] === Validating Directory Structure ===
[SUCCESS]   ✓ All required directories exist

[INFO] === Validating Training Data ===
[INFO] Checking Global Training Data...
[SUCCESS]   ✓ Global Training Data: 400 samples
[INFO] Checking Strict Client Data...
[SUCCESS]   ✓ Strict Client Data: 300 samples
[INFO] Checking Service Client Data...
[SUCCESS]   ✓ Service Client Data: 300 samples

[INFO] === Validating Test Data ===
[INFO] Checking Test-G (Global Laws)...
[SUCCESS]   ✓ Test-G (Global Laws): 20 samples
[INFO] Checking Test-A (Strict Policies)...
[SUCCESS]   ✓ Test-A (Strict Policies): 10 samples
[INFO] Checking Test-B (Service Policies)...
[SUCCESS]   ✓ Test-B (Service Policies): 10 samples
[INFO] Checking Conflict Test Cases...
[SUCCESS]   ✓ Conflict Test Cases: 5 test cases

[SUCCESS] === Validation Complete ===
[SUCCESS] ✓ All required training data is valid
[SUCCESS] ✓ All test data is valid
```

**注意**: 测试数据是可选的。如果测试数据不存在，脚本会显示警告但不会失败，你仍然可以进行训练。

### 5. 运行实验

#### 方法1: 使用自动化脚本（推荐）

```bash
# 运行完整实验流程（包含数据验证、训练、日志记录）
bash scripts/run_experiment.sh

# 跳过数据验证直接开始训练
bash scripts/run_experiment.sh --skip-validation

# 使用自定义配置文件
bash scripts/run_experiment.sh --config path/to/custom_config.yaml
```

自动化脚本会：
- ✓ 激活虚拟环境
- ✓ 检查 HuggingFace Token
- ✓ 验证 GPU 可用性
- ✓ 验证数据完整性
- ✓ 创建输出目录
- ✓ 运行训练并保存日志
- ✓ 处理错误并提供诊断信息

#### 方法2: 手动运行

```bash
# 激活虚拟环境
source venv/bin/activate

# 进入实验目录
cd experiments/exp001_dual_adapter_fl

# 仅训练
python train.py --config config.yaml

# 仅评测（需要先完成训练）
python eval.py --config config.yaml

# 完整流程（训练 + 评测 + 报告）
python run_experiment.py --config config.yaml
```

### 6. 监控进度

```bash
# 查看训练日志
tail -f results/exp001_dual_adapter_fl/logs/training.log

# 查看 GPU 使用情况
watch -n 1 nvidia-smi
```

### 7. 下载结果

```bash
# 在本地机器上运行
scp -r user@remote-server:/path/to/workspace/results/exp001_dual_adapter_fl ./results/
```

## 预期运行时间 (Expected Runtime)

- **模型下载**: 10-30 分钟（首次运行，取决于网络速度）
- **单轮训练**: 30-60 分钟（取决于 GPU 和数据量）
- **完整实验** (5轮训练 + 评测): 3-5 小时

## 输出结构 (Output Structure)

```
results/exp001_dual_adapter_fl/
├── checkpoints/
│   ├── round_1/
│   ├── round_2/
│   ├── round_3/
│   ├── round_4/
│   ├── round_5/
│   └── final_adapters/
│       ├── global/
│       ├── strict/
│       └── service/
├── logs/
│   ├── training.log
│   └── evaluation.log
├── metrics/
│   ├── training_metrics.json
│   └── test_results.json
└── report/
    ├── experiment_report.md
    ├── training_loss_curve.png
    ├── accuracy_comparison.png
    └── conflict_resolution_examples.md
```

## 常见问题 (Troubleshooting)

### 问题1: 模型下载失败
```
Error: Connection timeout when downloading model
```

**解决方案**:
- 检查网络连接
- 设置 HuggingFace 镜像: `export HF_ENDPOINT=https://hf-mirror.com`
- 增加超时时间: 在代码中设置 `timeout=600`

### 问题2: GPU 内存不足
```
RuntimeError: CUDA out of memory
```

**解决方案**:
- 确保使用 4-bit 量化: 在 config.yaml 中设置 `quantization: "4bit"`
- 减小 batch size: 设置 `per_device_train_batch_size: 2`
- 增加梯度累积: 设置 `gradient_accumulation_steps: 8`

### 问题3: 测试数据不存在
```
FileNotFoundError: data/test/global_test.json not found
```

**解决方案**:
- 确保测试数据已上传到服务器
- 运行 `python scripts/validate_data.py` 检查数据
- 参考 design.md 中的测试数据格式创建测试集

### 问题4: HuggingFace Token 无效
```
HTTPError: 401 Unauthorized
```

**解决方案**:
- 检查 token 是否正确设置
- 确保 token 有模型访问权限
- 重新登录: `huggingface-cli login`

## 性能优化建议 (Performance Tips)

1. **使用混合精度训练**: 在 config.yaml 中设置 `fp16: true`
2. **启用梯度检查点**: 减少内存占用
3. **调整 batch size**: 根据 GPU 内存调整
4. **使用更快的优化器**: 如 `paged_adamw_8bit`

## 安全注意事项 (Security Notes)

- 不要将 HuggingFace token 提交到 git
- 使用环境变量或安全的配置管理工具
- 定期更新依赖包以修复安全漏洞
- 限制对训练数据和模型的访问权限


## 部署脚本说明 (Deployment Scripts)

项目提供了三个自动化脚本来简化部署和运行流程：

### 1. `scripts/setup_remote.sh` - 环境设置脚本

**功能**:
- 检查系统依赖（Python, pip, venv, GPU）
- 创建和配置 Python 虚拟环境
- 安装所有依赖包
- 配置 HuggingFace Token
- 验证安装完整性

**用法**:
```bash
# 基本用法
bash scripts/setup_remote.sh

# 直接设置 HuggingFace Token
bash scripts/setup_remote.sh --token YOUR_TOKEN

# 查看帮助
bash scripts/setup_remote.sh --help
```

**输出**: 彩色日志显示每个步骤的状态（成功/警告/错误）

### 2. `scripts/validate_data.py` - 数据验证脚本

**功能**:
- 检查目录结构完整性
- 验证训练数据存在且格式正确
- 验证测试数据（可选）
- 检查 Alpaca 格式规范

**用法**:
```bash
# 基本验证
python scripts/validate_data.py

# 详细模式（显示样本数据）
python scripts/validate_data.py --verbose

# 查看帮助
python scripts/validate_data.py --help
```

**返回值**: 
- 0: 所有必需数据有效
- 1: 数据验证失败

### 3. `scripts/run_experiment.sh` - 实验运行脚本

**功能**:
- 激活虚拟环境
- 检查 HuggingFace Token 和 GPU
- 验证数据完整性
- 运行训练实验
- 保存日志到文件
- 错误处理和诊断

**用法**:
```bash
# 基本用法（包含数据验证）
bash scripts/run_experiment.sh

# 跳过数据验证
bash scripts/run_experiment.sh --skip-validation

# 使用自定义配置
bash scripts/run_experiment.sh --config path/to/config.yaml

# 查看帮助
bash scripts/run_experiment.sh --help
```

**输出**: 
- 训练日志同时输出到控制台和日志文件
- 日志文件位置: `results/exp001_dual_adapter_fl/logs/training_YYYYMMDD_HHMMSS.log`

### 脚本执行流程

```
setup_remote.sh
    ↓
    检查系统环境
    ↓
    创建虚拟环境
    ↓
    安装依赖
    ↓
    配置 HF Token
    ↓
validate_data.py
    ↓
    验证目录结构
    ↓
    验证训练数据
    ↓
    验证测试数据
    ↓
run_experiment.sh
    ↓
    激活环境
    ↓
    检查配置
    ↓
    运行训练
    ↓
    保存结果
```

### 错误处理

所有脚本都包含完善的错误处理：
- **setup_remote.sh**: 在任何步骤失败时停止，显示错误信息
- **validate_data.py**: 详细报告每个数据文件的问题
- **run_experiment.sh**: 捕获训练错误，保存日志，提供诊断建议

### 日志和输出

脚本使用彩色输出来区分不同类型的消息：
- 🔵 **[INFO]**: 信息性消息
- 🟢 **[SUCCESS]**: 成功完成的操作
- 🟡 **[WARNING]**: 警告（不影响继续执行）
- 🔴 **[ERROR]**: 错误（需要修复）

## 完整部署示例 (Complete Deployment Example)

以下是一个完整的部署和运行示例：

```bash
# === 在本地机器上 ===

# 1. 打包代码
cd /path/to/policeModel
tar -czf policeModel.tar.gz \
  --exclude='results' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='venv' \
  .

# 2. 上传到远程服务器
scp policeModel.tar.gz user@gpu-server:/home/user/

# === 在远程服务器上 ===

# 3. 解压代码
ssh user@gpu-server
cd /home/user
tar -xzf policeModel.tar.gz
cd policeModel

# 4. 运行自动设置（一次性）
bash scripts/setup_remote.sh --token hf_xxxxxxxxxxxxx

# 5. 验证数据
python scripts/validate_data.py --verbose

# 6. 运行实验
bash scripts/run_experiment.sh

# 7. 在另一个终端监控进度
ssh user@gpu-server
cd /home/user/policeModel
tail -f results/exp001_dual_adapter_fl/logs/training_*.log

# 或监控 GPU 使用
watch -n 1 nvidia-smi

# === 实验完成后 ===

# 8. 下载结果（在本地机器上）
scp -r user@gpu-server:/home/user/policeModel/results/exp001_dual_adapter_fl ./results/

# 9. 查看报告
cd results/exp001_dual_adapter_fl/report
cat experiment_report.md
```

## 故障排除清单 (Troubleshooting Checklist)

在遇到问题时，按以下顺序检查：

### ✅ 环境检查
```bash
# 检查 Python 版本
python3 --version  # 应该 >= 3.10

# 检查 GPU
nvidia-smi

# 检查虚拟环境
source venv/bin/activate
which python  # 应该指向 venv/bin/python

# 检查关键包
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### ✅ 数据检查
```bash
# 验证数据
python scripts/validate_data.py --verbose

# 检查文件权限
ls -lh data/rule_data/
ls -lh data/test/
```

### ✅ 配置检查
```bash
# 检查 HF Token
echo $HF_TOKEN

# 检查配置文件
cat experiments/exp001_dual_adapter_fl/config.yaml

# 测试 HF 认证
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

### ✅ 磁盘空间检查
```bash
# 检查可用空间（至少需要 50GB）
df -h .

# 检查 HuggingFace 缓存大小
du -sh ~/.cache/huggingface/
```

### ✅ 网络检查
```bash
# 测试 HuggingFace 连接
curl -I https://huggingface.co

# 如果在中国，可能需要设置镜像
export HF_ENDPOINT=https://hf-mirror.com
```

如果以上检查都通过但仍有问题，请查看详细的训练日志：
```bash
cat results/exp001_dual_adapter_fl/logs/training_*.log
```
