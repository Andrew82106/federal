#!/usr/bin/env python3
"""
Quick Model Test - 快速测试模型是否能正常调用

Usage:
    python scripts/quick_test.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型路径
MODEL_PATH = "/root/autodl-tmp/Downloads"

print("=" * 60)
print("快速测试 Qwen2.5-7B-Instruct 模型")
print("=" * 60)

# 1. 检查 GPU
print("\n[1/4] 检查 GPU...")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠ 未检测到 GPU，将使用 CPU（会很慢）")

# 2. 加载 tokenizer
print("\n[2/4] 加载 tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False  # 使用慢速 tokenizer 避免兼容性问题
    )
    print("✓ Tokenizer 加载成功")
except Exception as e:
    print(f"✗ 加载失败: {e}")
    print("\n尝试使用 use_fast=True...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=True
        )
        print("✓ Tokenizer 加载成功 (fast tokenizer)")
    except Exception as e2:
        print(f"✗ 仍然失败: {e2}")
        exit(1)

# 3. 加载模型
print("\n[3/4] 加载模型（可能需要几分钟）...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✓ 模型加载成功")
except Exception as e:
    print(f"✗ 加载失败: {e}")
    exit(1)

# 4. 测试推理
print("\n[4/4] 测试推理...")
question = "你好，请介绍一下自己。"
print(f"问题: {question}")

try:
    # 使用 Qwen chat template
    messages = [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": question}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取助手回答
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
        response = response.split("<|im_end|>")[0].strip()
    
    print(f"\n回答:\n{response}")
    print("\n✓ 推理测试成功！")
    
except Exception as e:
    print(f"✗ 推理失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✓ 所有测试通过！模型可以正常使用。")
print("=" * 60)
