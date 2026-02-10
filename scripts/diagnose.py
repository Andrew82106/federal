#!/usr/bin/env python3
"""
诊断脚本 - 检查 GPU 和模型文件
"""

import os
import torch
from pathlib import Path

print("=" * 60)
print("系统诊断")
print("=" * 60)

# 1. 检查 GPU
print("\n[1] GPU 检查:")
print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"  GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"  GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("  ⚠ 未检测到 GPU")
    print("\n  可能的原因:")
    print("  1. PyTorch 安装的是 CPU 版本")
    print("  2. CUDA 驱动未正确安装")
    print("  3. 需要重启容器")
    print("\n  解决方法:")
    print("  pip uninstall torch")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")

# 2. 检查模型文件
print("\n[2] 模型文件检查:")
model_path = "/root/autodl-tmp/Downloads"

if not os.path.exists(model_path):
    print(f"  ✗ 模型路径不存在: {model_path}")
else:
    print(f"  ✓ 模型路径存在: {model_path}")
    
    # 列出所有文件
    print("\n  文件列表:")
    for file in sorted(Path(model_path).glob("*")):
        size = file.stat().st_size / 1024**2  # MB
        print(f"    {file.name:<30} {size:>10.2f} MB")
    
    # 检查关键文件
    print("\n  关键文件检查:")
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json"
    ]
    
    for fname in required_files:
        fpath = Path(model_path) / fname
        if fpath.exists():
            print(f"    ✓ {fname}")
        else:
            print(f"    ✗ {fname} (缺失)")
    
    # 检查 safetensors 文件
    print("\n  模型权重文件:")
    safetensors_files = list(Path(model_path).glob("*.safetensors"))
    if safetensors_files:
        for f in sorted(safetensors_files):
            size = f.stat().st_size / 1024**3  # GB
            print(f"    {f.name:<40} {size:>8.2f} GB")
    else:
        print("    ✗ 未找到 .safetensors 文件")
        
        # 检查是否有 .bin 文件
        bin_files = list(Path(model_path).glob("*.bin"))
        if bin_files:
            print("\n  发现 .bin 文件:")
            for f in sorted(bin_files):
                size = f.stat().st_size / 1024**3  # GB
                print(f"    {f.name:<40} {size:>8.2f} GB")

print("\n" + "=" * 60)
