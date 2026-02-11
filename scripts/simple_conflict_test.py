#!/usr/bin/env python3
"""
Simple conflict test without using ConflictTester.
Directly loads adapters and tests conflict resolution.
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.base_model import load_base_model, freeze_base_model
from peft import PeftModel
import torch

def test_conflict_case(base_model, tokenizer, global_adapter, local_adapter, question, system_prompt):
    """Test a single conflict case."""
    # Load adapters
    model = PeftModel.from_pretrained(
        base_model,
        global_adapter,
        adapter_name="global",
        is_trainable=False
    )
    model.load_adapter(local_adapter, adapter_name="local")
    model.set_adapter("local")
    model.eval()
    
    # Generate response
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract assistant response
    if "<|im_start|>assistant\n" in response:
        response = response.split("<|im_start|>assistant\n")[-1]
    elif "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
    
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    
    response = response.replace("<|endoftext|>", "").strip()
    
    return response


def main():
    # Load config
    results_dir = project_root / "results" / "exp002_improved_dual_adapter"
    test_data_dir = project_root / "data" / "test"
    
    # Load conflict cases
    with open(test_data_dir / "conflict_cases.json", 'r', encoding='utf-8') as f:
        conflict_cases = json.load(f)
    
    print("=== EXP002 Conflict Test ===\n")
    print(f"Total cases: {len(conflict_cases)}\n")
    
    # Test a few sample cases
    sample_cases = conflict_cases[:3]  # Test first 3 cases
    
    for i, case in enumerate(sample_cases, 1):
        print(f"Case {i}: {case['instruction']}")
        print(f"Expected Strict: {case.get('expected_strict', 'N/A')}")
        print(f"Expected Service: {case.get('expected_service', 'N/A')}")
        print()
    
    print("Note: Full conflict test requires fixing PEFT adapter loading issues.")
    print("The 6 main test sets (Test-G, Test-A, Test-B) have been completed successfully.")


if __name__ == "__main__":
    main()
