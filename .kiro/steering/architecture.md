# 项目架构设计文档

## 1. 架构概览

本项目采用**严格分层架构**，确保工具、源代码、实验和结果之间的完全解耦。这种设计使得：
- 核心算法可以在不同实验中复用
- 工具可以服务于任意符合规范的实验
- 实验配置与实现逻辑分离
- 结果可以完全重现

### 1.1 核心设计原则

**解耦原则 (Decoupling)**
- `src/`: 纯粹的算法实现，不包含任何实验特定逻辑
- `tools/`: 通用的执行框架，通过配置驱动
- `experiments/`: 仅包含配置和入口脚本
- `results/`: 可完全重新生成的输出

**配置驱动 (Configuration-Driven)**
- 所有超参数在 `config.yaml` 中定义
- 代码通过读取配置来执行，不硬编码参数
- 支持命令行参数覆盖配置文件

**可复现性 (Reproducibility)**
- 固定随机种子
- 记录所有依赖版本
- 完整的日志和检查点

## 2. 目录结构与职责

### 2.1 源代码层 (`src/`)

**职责**: 实现可复用的核心算法和模型组件

```
src/
├── models/                 # 模型架构
│   ├── base_model.py       # 基座模型加载器
│   │   - load_base_model() # 加载 Qwen2.5-7B，支持量化
│   │   - freeze_base()     # 冻结基座参数
│   └── dual_adapter.py     # 双适配器架构
│       - DualAdapterModel  # 管理 Global + Local LoRA
│       - load_adapter()    # 加载指定适配器
│       - save_adapter()    # 保存适配器权重
├── federated/              # 联邦学习核心
│   ├── client.py           # 客户端训练逻辑
│   │   - ClientTrainer     # 封装本地训练流程
│   │   - train_round()     # 执行一轮本地训练
│   ├── server.py           # 服务器聚合逻辑
│   │   - FederatedServer   # 管理全局模型
│   │   - aggregate()       # 调用聚合算法
│   └── aggregators.py      # 聚合算法
│       - fedavg()          # FedAvg 实现
│       - weighted_avg()    # 加权平均
├── data/                   # 数据处理
│   ├── dataset.py          # 数据集类
│   │   - AlpacaDataset     # Alpaca 格式数据集
│   │   - collate_fn()      # 批处理函数
│   └── preprocessor.py     # 预处理工具
│       - tokenize()        # 分词
│       - format_prompt()   # 格式化提示词
└── utils/                  # 通用工具
    ├── config.py           # 配置管理
    │   - load_config()     # 加载 YAML 配置
    │   - merge_args()      # 合并命令行参数
    └── logger.py           # 日志工具
        - setup_logger()    # 初始化日志
        - log_metrics()     # 记录指标
```

**设计规范**:
- 所有函数接受参数，不使用全局变量
- 不包含文件路径硬编码
- 提供清晰的接口和类型提示
- 每个模块可独立测试

### 2.2 工具层 (`tools/`)

**职责**: 提供通用的实验执行和分析框架

```
tools/
├── runners/                # 执行框架
│   └── fl_runner.py        # 联邦学习运行器
│       - FLRunner          # 编排整个联邦流程
│       - run_experiment()  # 执行完整实验
│       - run_round()       # 执行单轮联邦
├── evaluators/             # 评估工具
│   ├── conflict_tester.py  # 冲突测试
│   │   - ConflictTester    # 测试语义冲突处理
│   │   - test_case()       # 单个测试用例
│   └── metrics.py          # 指标计算
│       - calculate_accuracy()
│       - calculate_perplexity()
└── visualizers/            # 可视化
    └── plot_results.py     # 结果绘图
        - plot_training_curve()
        - plot_comparison()
```

**设计规范**:
- 工具接受配置对象，不依赖特定实验
- 可以被任何符合规范的实验调用
- 专注于编排，不实现核心算法
- 可以依赖 `src/`，但不依赖 `experiments/`

### 2.3 实验层 (`experiments/`)

**职责**: 定义具体实验的配置和入口

```
experiments/
└── exp001_dual_adapter_fl/     # 实验 001: 双适配器联邦学习
    ├── README.md               # 实验文档
    │   - 实验目标
    │   - 数据说明
    │   - 预期结果
    ├── config.yaml             # 配置文件
    │   - model: 模型配置
    │   - training: 训练超参数
    │   - federated: 联邦学习参数
    │   - data: 数据路径
    ├── train.py                # 训练入口
    │   - 加载配置
    │   - 调用 FLRunner
    │   - 保存结果
    └── eval.py                 # 评估入口
        - 加载检查点
        - 运行冲突测试
        - 生成报告
```

**config.yaml 结构示例**:
```yaml
experiment:
  name: "dual_adapter_fl"
  seed: 42

model:
  base_model: "Qwen/Qwen2.5-7B-Instruct"
  quantization: "4bit"
  lora_config:
    r: 16
    lora_alpha: 32
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

training:
  num_epochs: 2
  batch_size: 4
  learning_rate: 2e-4
  gradient_accumulation_steps: 4
  max_seq_length: 1024

federated:
  num_rounds: 5
  num_clients: 2
  aggregation_method: "fedavg"

data:
  global_train: "data/rule_data/global_train.json"
  client_strict: "data/rule_data/client_strict.json"
  client_service: "data/rule_data/client_service.json"

output:
  checkpoint_dir: "results/exp001_dual_adapter_fl/checkpoints"
  log_dir: "results/exp001_dual_adapter_fl/logs"
```

**设计规范**:
- 最小化代码量，主要是配置和调用
- 从 `src/` 和 `tools/` 导入，不重复实现
- 所有参数在 config.yaml 中定义
- 使用顺序编号 (exp001, exp002, ...)

### 2.4 数据层 (`data/`)

**职责**: 存储源数据

```
data/
├── files/                          # 原始文档
│   ├── 道路交通安全法.pdf
│   ├── 居住证暂行条例.pdf
│   └── ...
└── rule_data/                      # 训练数据
    ├── global_train.json           # 通用法律 (条)
    ├── client_strict.json          # 严管城市 (块)
    └── client_service.json         # 服务型城市 (块)
```

**数据格式 (Alpaca)**:
```json
[
  {
    "instruction": "我大专学历，无社保，能落户吗？",
    "input": "",
    "output": "根据石家庄市公安局规定，全面放开城镇落户限制..."
  }
]
```

**设计规范**:
- 版本控制 (提交到 git)
- 实验期间只读
- 记录数据来源和处理步骤

### 2.5 结果层 (`results/`)

**职责**: 存储实验输出

```
results/
└── exp001_dual_adapter_fl/         # 镜像实验目录结构
    ├── checkpoints/                # 模型检查点
    │   ├── round_1/
    │   │   ├── global_adapter/
    │   │   ├── client_strict/
    │   │   └── client_service/
    │   ├── round_2/
    │   └── final_adapters/
    │       ├── global/             # 最终全局适配器
    │       ├── strict/             # 严管城市本地适配器
    │       └── service/            # 服务型城市本地适配器
    ├── logs/                       # 日志文件
    │   ├── training.log
    │   └── evaluation.log
    ├── metrics/                    # 性能指标
    │   ├── training_metrics.json
    │   └── test_results.csv
    └── report.md                   # 实验报告
```

**设计规范**:
- 完全 gitignore
- 可以从代码和数据完全重现
- 目录结构镜像 `experiments/`

## 3. 数据流与执行流程

### 3.1 训练流程

```
1. 实验入口 (experiments/exp001_dual_adapter_fl/train.py)
   ↓
2. 加载配置 (src/utils/config.py)
   ↓
3. 初始化 FLRunner (tools/runners/fl_runner.py)
   ↓
4. 联邦训练循环 (num_rounds 轮)
   ├─→ Round 1
   │   ├─→ Client 1 训练 (src/federated/client.py)
   │   │   ├─→ 加载基座模型 (src/models/base_model.py)
   │   │   ├─→ 加载 Global Adapter
   │   │   ├─→ 初始化 Local Adapter
   │   │   ├─→ 混合数据训练 (global + strict)
   │   │   └─→ 保存适配器
   │   ├─→ Client 2 训练
   │   │   └─→ (同上，使用 global + service)
   │   └─→ Server 聚合 (src/federated/server.py)
   │       ├─→ 收集 Global Adapters
   │       ├─→ FedAvg 聚合 (src/federated/aggregators.py)
   │       └─→ 保存新的 Global Adapter
   ├─→ Round 2
   └─→ ...
   ↓
5. 保存最终模型和日志
```

### 3.2 评估流程

```
1. 评估入口 (experiments/exp001_dual_adapter_fl/eval.py)
   ↓
2. 加载配置和检查点
   ↓
3. 冲突测试 (tools/evaluators/conflict_tester.py)
   ├─→ 测试用例 1: "电动车改装会被罚吗？"
   │   ├─→ 加载 Base + Strict Adapter
   │   │   └─→ 输出: "扣车、罚款1000元"
   │   └─→ 加载 Base + Service Adapter
   │       └─→ 输出: "教育为主、责令恢复"
   └─→ 测试用例 2: "积分落户社保要求"
       └─→ ...
   ↓
4. 计算指标 (tools/evaluators/metrics.py)
   ├─→ 通用法律准确率 (Test-G)
   ├─→ 严管城市准确率 (Test-A)
   └─→ 服务型城市准确率 (Test-B)
   ↓
5. 生成报告和可视化
```

## 4. 关键接口设计

### 4.1 模型接口

```python
# src/models/base_model.py
def load_base_model(
    model_name: str,
    quantization: str = "4bit",
    device_map: str = "auto"
) -> PreTrainedModel:
    """加载并配置基座模型"""
    pass

# src/models/dual_adapter.py
class DualAdapterModel:
    def __init__(self, base_model, lora_config):
        """初始化双适配器模型"""
        pass
    
    def add_global_adapter(self, adapter_path: Optional[str] = None):
        """添加全局适配器"""
        pass
    
    def add_local_adapter(self, adapter_path: Optional[str] = None):
        """添加本地适配器"""
        pass
    
    def save_adapters(self, global_path: str, local_path: str):
        """保存两个适配器"""
        pass
```

### 4.2 联邦学习接口

```python
# src/federated/client.py
class ClientTrainer:
    def __init__(self, client_id: str, config: dict):
        """初始化客户端训练器"""
        pass
    
    def train_round(
        self,
        global_adapter_path: str,
        local_adapter_path: Optional[str],
        train_data: Dataset
    ) -> Tuple[str, str]:
        """执行一轮训练，返回适配器路径"""
        pass

# src/federated/server.py
class FederatedServer:
    def __init__(self, config: dict):
        """初始化联邦服务器"""
        pass
    
    def aggregate(
        self,
        adapter_paths: List[str],
        method: str = "fedavg"
    ) -> str:
        """聚合客户端适配器，返回新的全局适配器路径"""
        pass
```

### 4.3 工具接口

```python
# tools/runners/fl_runner.py
class FLRunner:
    def __init__(self, config: dict):
        """初始化联邦学习运行器"""
        pass
    
    def run_experiment(self) -> dict:
        """运行完整实验，返回结果摘要"""
        pass
    
    def run_round(self, round_num: int) -> dict:
        """运行单轮联邦学习"""
        pass

# tools/evaluators/conflict_tester.py
class ConflictTester:
    def __init__(self, base_model_path: str):
        """初始化冲突测试器"""
        pass
    
    def test_case(
        self,
        question: str,
        adapter_paths: List[str]
    ) -> List[str]:
        """测试同一问题在不同适配器下的回答"""
        pass
```

## 5. 配置管理

### 5.1 配置层级

```
1. 默认配置 (src/utils/config.py 中的 DEFAULT_CONFIG)
   ↓
2. 实验配置文件 (experiments/exp001_dual_adapter_fl/config.yaml)
   ↓
3. 命令行参数 (python train.py --num_rounds 10)
```

### 5.2 配置访问

```python
from src.utils.config import load_config

# 加载配置
config = load_config("experiments/exp001_dual_adapter_fl/config.yaml")

# 访问配置
model_name = config["model"]["base_model"]
num_rounds = config["federated"]["num_rounds"]
```

## 6. 依赖关系

### 6.1 模块依赖图

```
experiments/
    ↓ (调用)
tools/
    ↓ (调用)
src/
    ↓ (使用)
data/
```

**规则**:
- `experiments/` 可以导入 `tools/` 和 `src/`
- `tools/` 可以导入 `src/`
- `src/` 不依赖 `tools/` 或 `experiments/`
- 禁止循环依赖

### 6.2 外部依赖

```
torch (深度学习框架)
    ↓
transformers (模型加载)
    ↓
peft (LoRA 实现)
    ↓
bitsandbytes (量化)
```

## 7. 扩展性设计

### 7.1 添加新实验

1. 创建新目录: `experiments/exp002_new_experiment/`
2. 复制 `config.yaml` 模板并修改参数
3. 编写 `train.py` 和 `eval.py` (复用 `tools/` 和 `src/`)
4. 运行实验，结果自动保存到 `results/exp002_new_experiment/`

### 7.2 添加新聚合算法

1. 在 `src/federated/aggregators.py` 中实现新函数
2. 在 `config.yaml` 中设置 `aggregation_method: "new_method"`
3. 无需修改其他代码

### 7.3 添加新评估指标

1. 在 `tools/evaluators/metrics.py` 中添加新函数
2. 在 `eval.py` 中调用新指标
3. 结果自动记录到日志

## 8. 最佳实践

### 8.1 代码规范

- 遵循 PEP 8 风格
- 使用类型提示 (Type Hints)
- 编写 docstring
- 单一职责原则

### 8.2 实验规范

- 每个实验独立目录
- 配置文件完整记录所有参数
- README 说明实验目的和预期结果
- 使用顺序编号

### 8.3 版本控制

**提交到 Git**:
- `src/`, `tools/`, `experiments/`, `data/`, `docs/`, `.kiro/`
- `requirements.txt`, `.gitignore`, `README.md`

**不提交 (gitignore)**:
- `results/`, `__pycache__/`, `*.pyc`, `*.pyo`
- `*.log`, `*.ckpt`, `*.pth`
- `.venv/`, `venv/`, `.env`

### 8.4 内存管理

- 使用 4-bit 量化 (模型 >7B)
- 训练轮次间清空 CUDA 缓存
- 使用梯度累积增大有效批量
- 启用梯度检查点节省显存

## 9. 故障排查

### 9.1 常见问题

**显存不足 (OOM)**:
- 减小 `batch_size`
- 增大 `gradient_accumulation_steps`
- 启用 4-bit 量化
- 减小 `max_seq_length`

**适配器加载失败**:
- 检查路径是否正确
- 确认适配器名称匹配
- 验证 LoRA 配置一致

**聚合结果异常**:
- 检查客户端数据分布
- 验证聚合算法实现
- 查看训练日志中的损失曲线

### 9.2 调试建议

- 使用小数据集快速验证流程
- 先跑 1 轮联邦学习测试
- 逐步增加轮数和数据量
- 记录详细日志便于追踪

## 10. 总结

本架构设计实现了：

✅ **完全解耦**: src/, tools/, experiments/, results/ 各司其职  
✅ **高度复用**: 核心代码可用于多个实验  
✅ **配置驱动**: 参数集中管理，易于调整  
✅ **可扩展性**: 添加新实验、算法、指标无需大改  
✅ **可维护性**: 清晰的接口和模块划分  
✅ **可复现性**: 完整的配置和日志记录  

这种架构特别适合需要频繁迭代实验的研究项目，确保了代码质量和实验效率的平衡。
