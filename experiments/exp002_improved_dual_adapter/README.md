# Experiment 002: Improved Dual-Adapter Federated Learning

## 改进点

基于 exp001 的评估结果，本实验做以下改进：

### 1. 训练策略优化

**exp001 问题**：
- 混合数据训练导致知识混淆
- Local Adapter 没有充分学习本地知识
- Test-B Privacy Gap = 0%

**exp002 改进**：
- **两阶段训练**：
  1. Phase 1: 只用 Global 数据训练 Global Adapter（3 rounds）
  2. Phase 2: 冻结 Global，只用 Local 数据训练 Local Adapter（2 rounds）
- 避免混合训练导致的知识混淆

### 2. LoRA 参数增大

**exp001**：
- r=16, alpha=32
- 参数量较小，表达能力不足

**exp002**：
- r=32, alpha=64
- 增大参数量，提升学习能力

### 3. 训练轮数增加

**exp001**：
- 每轮 2 epochs
- 训练不充分

**exp002**：
- Phase 1: 每轮 3 epochs
- Phase 2: 每轮 5 epochs
- 确保充分收敛

### 4. System Prompt 增强

**exp001**：
```
你是上海市（户政）与北京市（交管）的联合政务助手...
```

**exp002**：
```
你是上海市和北京市的政务助手。
【严格执行】上海积分落户政策（需120分，7年社保）和北京电动车严管政策（违规扣车罚款1000元）。
【回答要求】必须明确引用具体政策条款和数字。
```

## 预期结果

- Test-G: >70% (vs exp001: 43-63%)
- Test-A Privacy Gap: >30% (vs exp001: 13.4%)
- Test-B Privacy Gap: >30% (vs exp001: 0%)
- Conflict Test: >60% (vs exp001: 32.7%)

## 运行方式

```bash
# 训练
python experiments/exp002_improved_dual_adapter/train.py

# 评估
python experiments/exp002_improved_dual_adapter/eval.py --batch_size 32
```
