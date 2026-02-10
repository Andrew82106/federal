# Experiment 001: Dual-Adapter Federated Learning

## Overview

This experiment implements and evaluates the dual-adapter federated learning architecture for public security governance scenarios with "条块分割" (vertical-horizontal division).

## Objectives

1. Train a dual-adapter model with:
   - Global adapter (条): Learns universal laws, participates in federated aggregation
   - Local adapter (块): Learns jurisdiction-specific policies, remains private

2. Compare with baselines:
   - Local Only: Independent training without federation
   - Standard FedAvg: Single adapter with full aggregation

3. Evaluate conflict resolution capability

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM)
- HuggingFace account (for model download)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token
export HF_TOKEN="your_token_here"
```

### Run Training

```bash
# Run with default config
python experiments/exp001_dual_adapter_fl/train.py

# Override settings
python experiments/exp001_dual_adapter_fl/train.py \
    --config experiments/exp001_dual_adapter_fl/config.yaml \
    --num_rounds 5 \
    --output_dir results/my_experiment
```

## Configuration

Edit `config.yaml` to customize:

- **Model**: Base model, quantization, LoRA parameters
- **Training**: Learning rate, batch size, epochs
- **Federated**: Number of rounds, clients, aggregation method
- **Mode**: `dual_adapter`, `standard_fedavg`, or `local_only`

### Key Configuration Options

```yaml
federated:
  mode: "dual_adapter"  # Change to "standard_fedavg" for baseline
  num_rounds: 5
  clients:
    - id: "strict"
      system_prompt: "你是上海市公安局的政务助手..."
    - id: "service"
      system_prompt: "你是石家庄市公安局的政务助手..."
```

## Expected Outputs

```
results/exp001_dual_adapter_fl/
├── checkpoints/
│   ├── round_1/
│   ├── round_2/
│   └── final_adapters/
│       ├── global/
│       ├── strict/
│       └── service/
├── logs/
│   └── dual_adapter_fl_*.log
└── experiment_summary.json
```

## Hardware Requirements

- **Minimum**: NVIDIA RTX 3070 (8GB) with 4-bit quantization
- **Recommended**: NVIDIA RTX 4090 (24GB) for full precision

The system automatically detects VRAM and enables quantization if needed.

## Troubleshooting

### Out of Memory

- Reduce `per_device_train_batch_size` in config
- Increase `gradient_accumulation_steps`
- Enable 4-bit quantization: `quantization: "4bit"`

### Model Download Issues

- Ensure HF_TOKEN is set correctly
- Check internet connection
- Verify HuggingFace account has access to Qwen models

### Data Not Found

- Ensure training data exists in `data/rule_data/`
- Check paths in config.yaml match actual file locations

## Evaluation

After training completes, run comprehensive evaluation on four test sets.

### Quick Start

```bash
# 1. Verify setup (recommended first step)
python scripts/quick_eval.py

# 2. Quick evaluation on subset (5 cases per set)
python experiments/exp001_dual_adapter_fl/eval_quick.py

# 3. Full evaluation (all test cases)
python experiments/exp001_dual_adapter_fl/eval.py
```

### Four Test Sets

The evaluation covers four distinct test sets, each serving a specific purpose:

#### 1. Test-G (global_test.json) - Universal Knowledge Retention

**Purpose**: Verify that universal law knowledge is retained across all adapters

**Expected Results**:
- All adapters (Global Only, Strict, Service) should achieve high accuracy (>80%)
- Demonstrates that local adapters don't cause catastrophic forgetting

**Metrics**: Accuracy on national laws (道路交通安全法, 居住证条例)

#### 2. Test-A (strict_test.json) - Strict City Policy Memory

**Purpose**: Test local policy memory and privacy isolation

**Expected Results**:
- **Strict Adapter**: High accuracy (>80%) - knows Shanghai policies
- **Service Adapter**: Low accuracy (<30%) - doesn't know Shanghai policies

**Key Insight**: The accuracy gap proves **privacy isolation** - Service city's data never leaked to Strict city

**Metrics**: Accuracy on Shanghai-specific policies (积分落户, 严管电动车)

#### 3. Test-B (service_test.json) - Service City Policy Memory

**Purpose**: Test local policy memory and privacy isolation (reverse direction)

**Expected Results**:
- **Service Adapter**: High accuracy (>80%) - knows Shijiazhuang policies
- **Strict Adapter**: Low accuracy (<30%) - doesn't know Shijiazhuang policies

**Key Insight**: The accuracy gap proves **privacy isolation** - Strict city's data never leaked to Service city

**Metrics**: Accuracy on Shijiazhuang-specific policies (零门槛落户, 以学代罚)

#### 4. Conflict Test (conflict_cases.json) - Non-IID Conflict Resolution

**Purpose**: Test model's ability to provide different answers based on city identity

**Expected Results**:
- Same question should get different answers with different adapters
- Strict Adapter: Should exhibit "STRICT_BEHAVIOR" (拒绝, 积分, 罚款1000元)
- Service Adapter: Should exhibit "SERVICE_BEHAVIOR" (可以, 零门槛, 罚款20元)

**Key Insight**: Pass rate measures **conflict resolution capability**

**Metrics**: Pass rate (% of cases where model correctly exhibits city-specific behavior)

### Evaluation Method: Keyword Matching

The evaluation uses **keyword-based matching** instead of expensive LLM judges:

**Advantages**:
- ⚡ **1000x faster**: Millisecond-level vs. seconds per LLM call
- 💵 **Zero cost**: No API fees or GPU inference overhead
- ✅ **Deterministic**: Same input always gives same result
- 🎯 **Precise**: Directly checks policy keywords

**How It Works**:

For Test-G, Test-A, Test-B:
```python
# Extract key terms from expected answer
# Check if those terms appear in model response
# Calculate accuracy based on keyword overlap
```

For Conflict Test:
```python
# Check which keywords appear in response
strict_keywords = ["拒绝", "积分", "罚款1000元"]
service_keywords = ["可以", "零门槛", "罚款20元"]

# Classify behavior
if only strict_keywords found: "STRICT_BEHAVIOR"
if only service_keywords found: "SERVICE_BEHAVIOR"
if both found: "AMBIGUOUS"
if neither found: "NO_MATCH"
```

### System Prompt Injection

**Critical**: The evaluation automatically injects city-specific system prompts:

```python
system_prompts = {
    'strict': '你是上海市公安局的政务助手，请根据上海市的政策回答问题。',
    'service': '你是石家庄市公安局的政务助手，请根据石家庄市的政策回答问题。',
    'global': '你是公安政务助手，请根据国家法律法规回答问题。'
}
```

This ensures the model knows which city's policies to apply.

### Understanding Results

#### Example Output

```
📚 Test-G: Universal Law Knowledge Retention
   Global Only:    85.2%
   Strict Adapter: 83.7%
   Service Adapter: 84.1%

🔒 Test-A: Strict City Policy Memory
   Strict Adapter:  87.3% ✅
   Service Adapter: 23.1% (cross-test)
   Privacy Gap: +64.2% (higher is better)

🤝 Test-B: Service City Policy Memory
   Service Adapter: 89.5% ✅
   Strict Adapter:  19.8% (cross-test)
   Privacy Gap: +69.7% (higher is better)

⚔️  Conflict Test: Non-IID Conflict Resolution
   Pass Rate: 78.4%
   Passed: 163/208
   Ambiguous: 32
   No Match: 13
```

#### Key Metrics

- **Universal Knowledge Retained**: Should be >80% on Test-G
- **Privacy Isolation**: Gap between own-city and cross-city accuracy (higher is better)
- **Conflict Resolution**: Pass rate on conflict test (higher is better)

#### What Good Results Look Like

✅ **Success Indicators**:
- Test-G accuracy >80% (knowledge retained)
- Privacy gap >50% (strong isolation)
- Conflict pass rate >70% (good conflict resolution)

❌ **Warning Signs**:
- Test-G accuracy <60% (catastrophic forgetting)
- Privacy gap <30% (weak isolation, possible data leakage)
- Conflict pass rate <50% (poor conflict resolution)
- High "Ambiguous" count (model confused about city identity)

## Next Steps

After training completes:

1. Run conflict evaluation: `python experiments/exp001_dual_adapter_fl/eval_conflict.py`
2. Run standard evaluation: `python experiments/exp001_dual_adapter_fl/eval.py`
3. Generate visualizations
4. Compare with baseline methods
