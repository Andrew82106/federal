#!/usr/bin/env python3
"""Test adapter loading to debug shape mismatch issue."""

import sys
from pathlib import Path
import torch

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.base_model import load_base_model
from peft import PeftModel

# Load base model
print("Loading base model...")
base_model, tokenizer = load_base_model(
    model_name="/root/autodl-tmp/Downloads",
    quantization="auto"
)

print(f"Base model type: {type(base_model)}")
print(f"Base model device: {base_model.device}")

# Try loading adapter
adapter_path = "results/exp002_improved_dual_adapter/checkpoints/final_adapters/global"
print(f"\nLoading adapter from: {adapter_path}")

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    adapter_name="global",
    is_trainable=False
)

print(f"\nAdapter loaded successfully!")
print(f"Model type: {type(model)}")

# Check adapter config
print(f"\nAdapter config:")
print(f"  r: {model.peft_config['global'].r}")
print(f"  lora_alpha: {model.peft_config['global'].lora_alpha}")
print(f"  target_modules: {model.peft_config['global'].target_modules}")

# Check a sample weight shape
sample_weight = model.base_model.model.model.layers[0].self_attn.q_proj.lora_A['global'].weight
print(f"\nSample LoRA weight shape (layer 0, q_proj, lora_A): {sample_weight.shape}")
print(f"Expected shape for r=32: [32, 3584]")
