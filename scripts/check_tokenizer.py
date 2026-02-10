#!/usr/bin/env python3
"""
检查 tokenizer 文件
"""

import json
from pathlib import Path

MODEL_PATH = "/root/autodl-tmp/Downloads"

print("检查 tokenizer 文件...")
print("=" * 60)

# 检查 tokenizer_config.json
config_file = Path(MODEL_PATH) / "tokenizer_config.json"
if config_file.exists():
    with open(config_file, 'r') as f:
        config = json.load(f)
    print("\n✓ tokenizer_config.json:")
    print(f"  tokenizer_class: {config.get('tokenizer_class', 'N/A')}")
    print(f"  model_max_length: {config.get('model_max_length', 'N/A')}")
else:
    print("\n✗ tokenizer_config.json 不存在")

# 检查 vocab.json
vocab_file = Path(MODEL_PATH) / "vocab.json"
if vocab_file.exists():
    print(f"\n✓ vocab.json 存在 ({vocab_file.stat().st_size / 1024:.2f} KB)")
else:
    print("\n✗ vocab.json 不存在")

# 检查 merges.txt
merges_file = Path(MODEL_PATH) / "merges.txt"
if merges_file.exists():
    print(f"✓ merges.txt 存在 ({merges_file.stat().st_size / 1024:.2f} KB)")
else:
    print("✗ merges.txt 不存在")

# 检查 tokenizer.json
tokenizer_json = Path(MODEL_PATH) / "tokenizer.json"
if tokenizer_json.exists():
    print(f"✓ tokenizer.json 存在 ({tokenizer_json.stat().st_size / 1024:.2f} KB)")
else:
    print("✗ tokenizer.json 不存在")

print("\n" + "=" * 60)
print("尝试直接加载...")

try:
    from transformers import AutoTokenizer
    
    # 方法1: 默认加载
    print("\n[方法1] 默认加载...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("✓ 成功")
    except Exception as e:
        print(f"✗ 失败: {e}")
    
    # 方法2: 使用慢速 tokenizer
    print("\n[方法2] 慢速 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
        print("✓ 成功")
    except Exception as e:
        print(f"✗ 失败: {e}")
    
    # 方法3: 强制使用 Qwen2Tokenizer
    print("\n[方法3] 强制使用 Qwen2Tokenizer...")
    try:
        from transformers import Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
        print("✓ 成功")
    except Exception as e:
        print(f"✗ 失败: {e}")
    
    # 方法4: 从 HuggingFace Hub 加载（对比）
    print("\n[方法4] 从 HuggingFace Hub 加载 Qwen2.5-7B-Instruct...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        print("✓ 成功 - 说明网络和库都正常，问题在本地模型文件")
    except Exception as e:
        print(f"✗ 失败: {e}")

except Exception as e:
    print(f"✗ 导入失败: {e}")
