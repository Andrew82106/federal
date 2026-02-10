---
inclusion: always
---

# Technology Stack

## Core Technologies

### Base Model
- **Model**: Qwen2.5-7B-Instruct (7B parameters)
- **Source**: HuggingFace Model Hub
- **Quantization**: 4-bit (bitsandbytes) for memory efficiency

### Deep Learning Framework
- **PyTorch**: Core deep learning framework
- **Transformers**: HuggingFace library for model loading and training
- **PEFT**: Parameter-Efficient Fine-Tuning (LoRA implementation)
- **bitsandbytes**: 4-bit quantization for memory optimization

### Federated Learning
- **Custom Implementation**: Based on FedAvg algorithm
- **Architecture**: Dual-adapter (Global + Local LoRA)
- **Simulation**: Serial simulation on single GPU

### Data Processing
- **Format**: Alpaca-style JSON (instruction, input, output)
- **Tokenization**: Qwen tokenizer
- **Max Sequence Length**: 1024 tokens

## Hardware Requirements

### Recommended
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **CPU**: Multi-core processor
- **RAM**: 32GB+
- **Storage**: 50GB+ for models and checkpoints

### Minimum
- **GPU**: NVIDIA RTX 3070 (8GB VRAM) with 4-bit quantization
- **CPU**: Quad-core processor
- **RAM**: 16GB
- **Storage**: 30GB+

## Development Environment

### Python Environment
- **Python Version**: 3.10+
- **Package Manager**: pip or conda
- **Virtual Environment**: Required (venv or conda)

### Key Dependencies
```
torch>=2.0.0
transformers>=4.35.0
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
scipy>=1.11.0
pyyaml>=6.0
tqdm>=4.66.0
```

## Experimental Project Guidelines

### Code Organization
- **Separation of Concerns**: Strict boundaries between src/, tools/, experiments/, results/
- **Dependency Management**: requirements.txt for Python dependencies
- **Version Control**: 
  - Commit: src/, tools/, experiments/, data/, docs/, .kiro/
  - Ignore: results/, checkpoints/, logs/, __pycache__/, *.pyc

### Configuration-Driven Design
- **Format**: YAML for experiment configurations
- **Structure**: Hierarchical config with sections for model, training, federated, data
- **Override**: Command-line arguments can override config values

### Reproducibility
- **Random Seeds**: Set in config (torch, numpy, random)
- **Versioning**: Track model versions, data versions, code commits
- **Logging**: Comprehensive logging with timestamps and parameters

### Modularity
- **Interfaces**: Clear interfaces between components (model, client, server, aggregator)
- **Dependency Injection**: Pass dependencies as arguments, not globals
- **Testability**: Write unit tests for core algorithms in src/

## Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments
```bash
# Run dual-adapter federated learning experiment
cd experiments/exp001_dual_adapter_fl
python train.py --config config.yaml

# Evaluate trained models
python eval.py --config config.yaml --checkpoint ../../results/exp001_dual_adapter_fl/checkpoints/final_adapters
```

### Analysis
```bash
# Generate performance plots
python ../../tools/visualizers/plot_results.py --results ../../results/exp001_dual_adapter_fl

# Run conflict tests
python ../../tools/evaluators/conflict_tester.py --config config.yaml
```

## Best Practices

### Memory Management
- Use 4-bit quantization for models >7B parameters
- Clear CUDA cache between training rounds
- Use gradient accumulation for larger effective batch sizes

### Training Efficiency
- Use mixed precision training (fp16/bf16)
- Enable gradient checkpointing for memory savings
- Monitor GPU utilization and adjust batch size

### Federated Learning Specifics
- Save checkpoints after each round
- Log client-specific metrics separately
- Implement proper adapter management (load/save/merge)

### Code Quality
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Document complex algorithms with comments
- Write docstrings for all public functions
