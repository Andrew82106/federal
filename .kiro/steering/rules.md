---
inclusion: always
---

# Project Rules and Conventions

## Documentation Rules

**Do NOT create summary or documentation markdown files unless explicitly requested by the user.**

This includes:
- Summary reports of work completed
- Documentation of processes or workflows
- Analysis documents
- Any other markdown files that document or summarize your actions

**Exception**: Only create documentation files when the user explicitly asks for them.

## Code Style and Conventions

### Python Code Standards
- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose

### Import Organization
```python
# Standard library imports
import os
import sys

# Third-party imports
import torch
from transformers import AutoModel

# Local imports
from src.models.base_model import load_base_model
```

### Configuration Management
- All hyperparameters must be defined in `config.yaml` files
- Never hardcode paths, model names, or training parameters
- Use `src/utils/config.py` for configuration loading
- Support command-line argument overrides

## Architecture Principles

### Strict Layer Separation
- `src/`: Pure algorithm implementations, no experiment-specific logic
- `tools/`: Generic frameworks driven by configuration
- `experiments/`: Configuration and entry scripts only
- `results/`: Generated outputs (never commit to git)

### Dependency Rules
- `experiments/` can import from `tools/` and `src/`
- `tools/` can import from `src/`
- `src/` must NOT depend on `tools/` or `experiments/`
- No circular dependencies allowed

### File Organization
- Keep related functionality together in modules
- Use `__init__.py` to expose public interfaces
- Private functions should start with underscore `_`

## Federated Learning Specifics

### Adapter Management
- Global adapters participate in aggregation
- Local adapters remain private to clients
- Always save both adapter types separately
- Use clear naming: `global_adapter`, `local_adapter_{client_id}`

### Training Flow
- Clear CUDA cache between federated rounds
- Save checkpoints after each round
- Log client-specific metrics separately
- Use mixed precision training (fp16/bf16)

## Memory Optimization

### Required Practices
- Use 4-bit quantization for models >7B parameters
- Enable gradient accumulation for larger effective batch sizes
- Monitor GPU utilization and adjust batch size accordingly
- Clear CUDA cache between training rounds

## Testing and Validation

### Before Completing Tasks
- Run tests to verify functionality
- Check for syntax errors and type issues
- Validate against requirements in spec documents
- Test with small datasets first before full runs

## Git and Version Control

### Commit to Git
- Source code: `src/`, `tools/`, `experiments/`
- Data: `data/` (source data only)
- Documentation: `docs/`, `.kiro/`
- Dependencies: `requirements.txt`

### Never Commit
- `results/` directory
- `__pycache__/`, `*.pyc`, `*.pyo`
- Model checkpoints: `*.ckpt`, `*.pth`
- Log files: `*.log`
- Virtual environments: `venv/`, `.venv/`

## Communication Style

### When Responding to Users
- Be concise and direct
- Focus on actionable information
- Avoid verbose summaries unless requested
- Don't repeat what you just said
- Use Chinese when the user communicates in Chinese