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

## Next Steps

After training completes:

1. Run evaluation: `python experiments/exp001_dual_adapter_fl/eval.py`
2. Generate visualizations
3. Compare with baseline methods
