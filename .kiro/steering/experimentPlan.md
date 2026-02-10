这是一份可以直接发给 AI 编程助手（Kiro/Cursor/Windsurf）的**工程级实验方案**。

这份方案已经将复杂的学术概念转化为**具体的代码逻辑、文件结构和训练超参数**。请复制下面的内容，直接发给 Kiro。

---

## 实验项目：面向公安“条块数据”的双适配器联邦大模型 (Dual-Adapter FL)

### 1. 项目目标 (Project Objective)

在单张 GPU 环境下，通过串行模拟（Serial Simulation）实现一种**层级化联邦学习系统**。

* **Base Model**：Qwen2.5-7B-Instruct（冻结参数）。
* **Global Adapter (条)**：学习通用法律（如道路交通安全法、居住证条例），所有客户端共享并聚合。
* **Local Adapter (块)**：学习地方性差异政策（如上海积分制、北京严管、石家庄零门槛），保留在本地，**不参与聚合**。
* **核心冲突**：验证模型在不同 Local Adapter 激活时，对同一问题（如“没社保能落户吗”）产生截然不同的回答。

### 2. 环境与依赖 (Environment)

* **硬件**：NVIDIA RTX 4090 (24GB) 或 RTX 3070 (8GB, 需开启 4-bit 量化)。
* **核心库**：
* `pytorch`
* `transformers`
* `peft` (用于管理多 LoRA)
* `bitsandbytes` (用于 QLoRA 量化)
* `scipy` (用于聚合算法)



### 3. 数据准备 (Data Preparation)

请确保目录下有 `data/` 文件夹，包含以下三个 JSON 文件（Alpaca 格式）：

* **`data/global_train.json` (约 400 条)**
* 内容：《道路交通安全法》、《居住证暂行条例》。
* 作用：通用知识底座。


* **`data/client_strict.json` (约 300 条)**
* 内容：《上海居住证积分》（算分逻辑）、《北京非机动车》（扣车罚款）。
* 特征：严管、高门槛。


* **`data/client_service.json` (约 300 条)**
* 内容：《石家庄零门槛落户》、《南宁电动车以学代罚》。
* 特征：宽松、人性化。



**数据格式示例**：

```json
[
  {
    "instruction": "我大专学历，无社保，能落户吗？",
    "input": "",
    "output": "..."
  }
]

```

### 4. 代码架构设计 (Code Architecture)

请按以下逻辑编写 `main.py` 或 `train_fl.py`：

#### 步骤 1：模型初始化

* 加载 `Qwen2.5-7B-Instruct`。
* 如果显存 < 20GB，使用 `load_in_4bit=True`。
* 定义 **LoRA Config**：
* `r=16`, `lora_alpha=32`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`。



#### 步骤 2：定义联邦循环 (The FL Loop)

设定 `NUM_ROUNDS = 5` (联邦轮数)。在每一轮中执行以下串行操作：

**A. 客户端 1 (Strict City) 训练：**

1. 加载基座模型。
2. 加载/初始化 `global_adapter` (从 Server 获取最新权重)。
3. 加载/初始化 `local_adapter_strict` (从本地硬盘读取)。
4. **关键点**：使用 `peft` 的 `add_adapter` 功能同时挂载两个 Adapter，并将它们都设为 `trainable` (可训练)。
* *注：为了简化，也可以采用混合数据训练法，即让一个 Adapter 同时学，但在聚合时只提取 Global 部分的权重。建议 Kiro 采用“混合数据+单一Global Adapter训练 -> 存为Local Checkpoint -> 提取权重聚合”的简化路径，或者“双Adapter同时激活”的高级路径。*
* **推荐 Kiro 实现路径**：
* 构建混合数据集：`train_data = global_data + client_strict_data`
* 训练一个名为 `adapter_strict_merged` 的 LoRA。
* 训练结束后，保存权重。





**B. 客户端 2 (Service City) 训练：**

1. 同上，使用 `train_data = global_data + client_service_data`。
2. 训练名为 `adapter_service_merged` 的 LoRA。
3. 保存权重。

**C. 服务器聚合 (Server Aggregation)：**

1. 加载 `adapter_strict_merged` 和 `adapter_service_merged` 的权重文件。
2. **参数解耦 (核心算法)**：
* 我们需要假设 Adapter 中其实包含了“通用知识”和“本地知识”。
* **FedAvg**：计算 `avg_weight = (weight_strict + weight_service) / 2`。
* 更新 `global_adapter` 为 `avg_weight`。
* *注意：这里为了模拟“Local不上传”，在代码实现上，我们可以简单地做全量聚合作为 Baseline，或者更高级地：只聚合那些在 Global 数据上梯度变化大的参数（但这代码很难写）。*
* **给 Kiro 的简单指令**：执行标准的 FedAvg 聚合，生成 `global_adapter_round_X`。下一轮开始时，Clients 基于这个新的 Global 权重继续微调。



---

### 5. 训练超参数 (Hyperparameters)

* `learning_rate`: 2e-4
* `num_train_epochs`: 2 (每轮联邦只跑 2 个 epoch，防止过拟合)
* `per_device_train_batch_size`: 4 (显存小) 或 8 (显存大)
* `gradient_accumulation_steps`: 4
* `max_seq_length`: 1024
* `save_strategy`: "steps"

---

### 6. 验证与推理脚本 (Inference & Evaluation)

训练完成后，编写 `eval.py` 进行效果展示：

**Case 1: 冲突测试 (Conflict Test)**

* **输入**：“我的电动车是改装的超标车，上路会被罚吗？”
* **操作**：
* 加载 `Base Model` + `adapter_strict_merged` -> **预期输出**：“扣车、罚款1000元。”
* 加载 `Base Model` + `adapter_service_merged` -> **预期输出**：“教育为主、责令恢复。”



**Case 2: 逻辑测试 (Reasoning Test)**

* **输入**：“我30岁本科，社保2倍，积分多少？”
* **操作**：加载 `adapter_strict_merged` -> **预期输出**：正确计算出数值。

---

### 7. 给 Kiro 的具体 Prompt 指令

**请复制这段话给 Kiro：**

> “Kiro，我要做一个基于 LoRA 的联邦学习实验代码。
> **场景**：模拟两个不同的公安局（Client A: 严管城市，Client B: 服务型城市）协同训练一个政务大模型。
> **数据**：我有三个 json 文件：`global.json`（通用法律）、`client_strict.json`（严管政策）、`client_service.json`（宽松政策）。
> **请为我编写 Python 代码，包含以下模块：**
> 1. **Dataset类**：读取 Alpaca 格式的 JSON，进行 Tokenization。
> 2. **Trainer**：使用 HuggingFace `SFTTrainer` 或 `Trainer`。
> 3. **Federated Loop (模拟)**：
> * 循环 3 轮 (Rounds)。
> * 在每轮中，分别在 Client A 和 Client B 的数据上微调 Qwen2.5-7B (使用 LoRA)。
> * **关键逻辑**：Client A 的训练数据 = `global` + `strict`；Client B 的训练数据 = `global` + `service`。
> * **聚合**：每轮结束后，将 A 和 B 的 LoRA 权重取平均值，作为下一轮的初始权重。
> 
> 
> 4. **Save**：保存每一轮的 Global 模型，以及最后一轮的两个 Local 模型。
> 5. **Inference**：写一个推理脚本，可以加载不同的 Adapter (A 或 B) 来回答同一个问题，展示回复的差异。
> 
> 
> 请注意显存优化，使用 4-bit 量化加载基座模型。”

---

### 补充：文件结构参考

```text
project/
├── data/
│   ├── global_train.json
│   ├── client_strict.json
│   └── client_service.json
├── output/
│   ├── round_1/
│   ├── round_2/
│   └── final_adapters/
│       ├── strict/
│       └── service/
├── train_fl.py  (主训练脚本)
├── eval.py      (推理脚本)
└── requirements.txt

```