# EXP000: Standard FedAvg Baseline

## 实验目的

作为对照组，验证双适配器架构相比传统 FedAvg 的优势。

## 架构设计

- **单一 Global Adapter**：所有客户端共享一个 LoRA adapter
- **无 Local Adapter**：不区分本地知识和全局知识
- **数据混合**：将 Strict 和 Service 的训练数据物理混合，模拟数据异构问题

## 预期结果

在 Conflict Test 中表现很差：
- 无法根据 system prompt 切换城市身份
- 回答逻辑混乱，可能同时包含严管和服务型政策
- Pass Rate 应该远低于 EXP001/EXP002

## 训练命令

```bash
cd experiments/exp000_fedavg_baseline
python train.py
```

## 评估命令

```bash
python eval_conflict_only.py
```

## 配置

- LoRA Rank: 16
- LoRA Alpha: 32
- Epochs per Round: 2
- Federated Rounds: 5
- 训练数据：Global (400) + Mixed (470) = 870 samples

## 关键对比

| 指标 | EXP000 (FedAvg) | EXP001 (Dual-Adapter) | EXP002 (Dual-Adapter) |
|------|----------------|----------------------|----------------------|
| Architecture | Single Adapter | Global + Local | Global + Local |
| LoRA Rank | 16 | 16 | 32 |
| Conflict Pass Rate | ? (预期很低) | 29.3% | 21.6% |
| Privacy Protection | ❌ 无 | ✅ 有 | ✅ 有 |
