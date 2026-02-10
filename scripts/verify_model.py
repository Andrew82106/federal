#!/usr/bin/env python3
"""
验证模型文件完整性
"""

import json
from pathlib import Path
from safetensors import safe_open

MODEL_PATH = "/root/autodl-tmp/Downloads"

print("验证模型文件完整性...")
print("=" * 60)

# 读取 index 文件
index_file = Path(MODEL_PATH) / "model.safetensors.index.json"
if not index_file.exists():
    print("✗ model.safetensors.index.json 不存在")
    exit(1)

with open(index_file, 'r') as f:
    index = json.load(f)

print(f"✓ 找到 {len(index['weight_map'])} 个权重张量")
print(f"✓ 分布在 {len(set(index['weight_map'].values()))} 个文件中\n")

# 获取所有 safetensors 文件
shard_files = sorted(set(index['weight_map'].values()))

print("检查每个分片文件:")
print("-" * 60)

all_valid = True
for shard_file in shard_files:
    shard_path = Path(MODEL_PATH) / shard_file
    
    if not shard_path.exists():
        print(f"✗ {shard_file}: 文件不存在")
        all_valid = False
        continue
    
    file_size = shard_path.stat().st_size / (1024**3)
    print(f"\n{shard_file}:")
    print(f"  大小: {file_size:.2f} GB")
    
    # 尝试打开文件
    try:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"  张量数: {len(keys)}")
            print(f"  ✓ 文件完整")
    except Exception as e:
        print(f"  ✗ 文件损坏: {e}")
        all_valid = False

print("\n" + "=" * 60)
if all_valid:
    print("✓ 所有模型文件完整")
else:
    print("✗ 部分文件损坏，需要重新下载")
    print("\n建议:")
    print("1. 删除损坏的文件")
    print("2. 使用 huggingface-cli 重新下载:")
    print("   huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /root/autodl-tmp/Downloads --resume-download")

