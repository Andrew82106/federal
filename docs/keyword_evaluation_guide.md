# 关键词匹配评估指南

## 概述

本项目实现了一种**基于关键词匹配的快速评估方法**，用于测试双适配器模型在冲突场景下的表现。这种方法相比传统的 LLM 裁判评估具有显著优势。

## 为什么使用关键词匹配？

### 传统方法的问题

使用 LLM 作为裁判评估模型输出：
- ⏱️ **慢**：每个测试用例需要额外的 LLM 推理（2-5秒）
- 💰 **贵**：API 调用成本或 GPU 资源消耗
- 🎲 **不确定**：LLM 判断存在随机性，结果不稳定
- 🔧 **复杂**：需要精心设计 prompt，调试困难

### 关键词匹配的优势

- ⚡ **极快**：毫秒级评估（1000x 速度提升）
- 💵 **零成本**：纯字符串匹配，无需额外计算
- ✅ **确定性**：相同输入永远得到相同结果
- 🎯 **精准**：直接检测政策关键词，不依赖语义理解

## 数据格式

### 增强的测试数据

`data/test/conflict_cases.json` 中的每个测试用例包含：

```json
{
  "instruction": "我大专学历，无社保，能在本市直接落户吗？",
  "input": "",
  "output": "",
  "evaluation_guide": {
    "strict_keywords": ["拒绝", "积分", "不达标", "无房不能落户"],
    "service_keywords": ["可以", "零门槛", "凭身份证", "即时办"]
  }
}
```

### 关键词设计原则

**Strict City (严管城市 - 上海)**
- 落户：`["拒绝", "积分", "不达标", "无房不能落户"]`
- 电动车：`["收缴", "扣留", "罚款1000元", "禁止"]`

**Service City (服务型城市 - 石家庄)**
- 落户：`["可以", "零门槛", "凭身份证", "即时办"]`
- 电动车：`["责令改正", "恢复原状", "罚款20元", "罚款50元"]`

## 评估逻辑

### 行为分类

```python
def evaluate_conflict_behavior(response: str, evaluation_guide: dict) -> str:
    """
    根据关键词匹配判断模型行为类型
    """
    strict_keywords = evaluation_guide['strict_keywords']
    service_keywords = evaluation_guide['service_keywords']
    
    strict_hits = sum(1 for k in strict_keywords if k in response)
    service_hits = sum(1 for k in service_keywords if k in response)
    
    if strict_hits > 0 and service_hits == 0:
        return "STRICT_BEHAVIOR"  # 严管行为
    elif service_hits > 0 and strict_hits == 0:
        return "SERVICE_BEHAVIOR"  # 服务行为
    elif strict_hits > 0 and service_hits > 0:
        return "AMBIGUOUS"  # 模棱两可
    else:
        return "NO_MATCH"  # 无匹配
```

### 成功标准

对于每个测试用例：
- 使用 **Strict Adapter** 时，应该返回 `STRICT_BEHAVIOR`
- 使用 **Service Adapter** 时，应该返回 `SERVICE_BEHAVIOR`

如果两个适配器都表现正确，则该测试用例**通过**。

## 使用方法

### 1. 快速测试（不需要模型）

验证关键词匹配逻辑：

```bash
python scripts/test_conflict_evaluation.py
```

输出示例：
```
Sample test case:
Question: 警官，我大专毕业后一直没找到正式工作...
Strict keywords: ['拒绝', '积分', '不达标', '无房不能落户']
Service keywords: ['可以', '零门槛', '凭身份证', '即时办']

--- Simulated Strict City Response ---
Response: 根据上海市居住证积分管理办法，您需要满足积分要求...
Detected behavior: STRICT_BEHAVIOR
Expected: STRICT_BEHAVIOR
Correct: ✅
```

### 2. 完整评估（需要训练好的模型）

在训练完成后运行：

```bash
python experiments/exp001_dual_adapter_fl/eval_conflict.py
```

这会：
1. 加载训练好的 Global + Local Adapters
2. 对每个测试用例生成两个响应（Strict 和 Service）
3. 使用关键词匹配评估行为
4. 生成评估报告

### 3. 在代码中使用

```python
from tools.evaluators.conflict_tester import ConflictTester

# 初始化测试器
tester = ConflictTester(
    base_model_name="Qwen/Qwen2.5-7B-Instruct",
    global_adapter_path="path/to/global_adapter",
    config=config
)

# 运行评估
results = tester.run_guided_test_suite(
    test_cases=test_cases,
    local_adapter_paths={
        'strict': 'path/to/strict_adapter',
        'service': 'path/to/service_adapter'
    }
)

print(f"Pass Rate: {results['pass_rate']:.1%}")
```

## 评估指标

### 主要指标

- **Pass Rate**: 通过率（模型正确展现城市特定行为的比例）
- **Total Cases**: 测试用例总数
- **Passed**: 通过的用例数
- **Failed**: 失败的用例数

### 失败类型

- **Ambiguous**: 响应中同时包含两个城市的关键词
- **No Match**: 响应中没有任何预期关键词

### 示例输出

```
Guided Test Suite Results:
  Total: 208
  Passed: 187
  Failed: 21
  Ambiguous: 15
  No Match: 6
  Pass Rate: 89.9%
```

## 优化建议

### 如果 Pass Rate 低

1. **检查关键词覆盖度**
   - 是否遗漏了重要的政策术语？
   - 关键词是否过于严格？

2. **检查模型训练**
   - 训练数据是否充分？
   - 是否需要更多轮次？

3. **检查 System Prompt**
   - 城市身份提示是否清晰？
   - 是否需要更强的角色设定？

### 如果 Ambiguous 多

模型可能：
- 没有充分学习到城市差异
- Local Adapter 权重不足
- 需要调整 LoRA 参数（r, alpha）

### 如果 No Match 多

模型可能：
- 回答过于模糊或通用
- 需要更多针对性训练数据
- 关键词设计需要调整

## 与 LLM 裁判的对比

| 维度 | 关键词匹配 | LLM 裁判 |
|------|-----------|---------|
| 速度 | 毫秒级 | 秒级 |
| 成本 | 零 | API费用/GPU |
| 确定性 | 100% | 随机性 |
| 可解释性 | 高（直接看关键词） | 低（黑盒） |
| 维护成本 | 低 | 高（prompt工程） |
| 适用场景 | 政策关键词明确 | 复杂语义理解 |

## 最佳实践

1. **先用关键词匹配快速迭代**
   - 在开发阶段快速验证模型行为
   - 发现问题后调整训练策略

2. **关键词定期更新**
   - 根据实际模型输出调整关键词
   - 添加新发现的高频政策术语

3. **结合人工抽查**
   - 对失败案例进行人工审查
   - 验证关键词匹配的准确性

4. **保存评估历史**
   - 跟踪不同版本的 Pass Rate
   - 分析改进趋势

## 总结

关键词匹配评估是一种**简单、快速、可靠**的方法，特别适合本项目的政策冲突场景。通过精心设计的关键词，我们可以在不依赖昂贵 LLM 裁判的情况下，准确评估模型的城市特定行为表现。

这种方法让实验迭代速度提升了 **1000 倍**，同时保持了评估的准确性和可解释性。
