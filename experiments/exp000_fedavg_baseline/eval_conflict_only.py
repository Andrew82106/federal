#!/usr/bin/env python3
"""
EXP000 Conflict Test - Standard FedAvg Baseline
测试单一 adapter 在冲突场景下的表现（预期：逻辑混乱）
"""

import json
import logging
import sys
from pathlib import Path
import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.models.base_model import load_base_model, freeze_base_model
from peft import PeftModel

def evaluate_conflict_behavior(response: str, evaluation_guide: dict) -> str:
    """评估响应属于哪种城市行为"""
    strict_keywords = evaluation_guide.get('strict_keywords', [])
    service_keywords = evaluation_guide.get('service_keywords', [])
    
    strict_hits = sum(1 for k in strict_keywords if k in response)
    service_hits = sum(1 for k in service_keywords if k in response)
    
    if strict_hits > 0 and service_hits == 0:
        return "STRICT_BEHAVIOR"
    elif service_hits > 0 and strict_hits == 0:
        return "SERVICE_BEHAVIOR"
    elif strict_hits > 0 and service_hits > 0:
        return "AMBIGUOUS"
    else:
        return "NO_MATCH"

def main():
    # Setup logging
    log_dir = project_root / "results" / "exp000_fedavg_baseline" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(log_dir), experiment_name="eval_conflict_only")
    
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    
    logging.info("="*80)
    logging.info("CONFLICT TEST - EXP000 (Standard FedAvg Baseline)")
    logging.info("="*80)
    
    # Define paths
    results_dir = project_root / "results" / "exp000_fedavg_baseline"
    test_data_dir = project_root / "data" / "test"
    
    # 使用 round_5 的 global adapter
    adapter_path = str(results_dir / "checkpoints" / "round_5" / "global_adapter")
    
    # System prompts (虽然是同一个模型，但我们用不同 prompt 测试)
    system_prompts = {
        'strict': '你是上海市（户政）与北京市（交管）的联合政务助手。请依据这两个城市严格、规范的管理规定进行回答。对于违规行为，请强调处罚和红线。',
        'service': '你是石家庄市（户政）与南宁市（交管）的联合政务助手。请依据这两个城市便民、宽松、人性化的政策进行回答。对于轻微违章，请强调教育与纠正。'
    }
    
    # Load conflict cases
    with open(test_data_dir / "conflict_cases.json", 'r', encoding='utf-8') as f:
        conflict_cases = json.load(f)
    
    logging.info(f"Loaded {len(conflict_cases)} conflict cases")
    
    # Load base model
    logging.info("Loading base model...")
    base_model, tokenizer = load_base_model(
        model_name=config['model']['base_model'],
        quantization=config['model'].get('quantization', 'auto')
    )
    freeze_base_model(base_model)
    
    # Set padding
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load adapter
    logging.info(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False
    )
    model.eval()
    
    # Test with both system prompts
    all_responses = {}
    batch_size = 16
    
    for prompt_type in ['strict', 'service']:
        logging.info(f"\nTesting with '{prompt_type}' system prompt...")
        system_prompt = system_prompts[prompt_type]
        
        responses = []
        for i in tqdm(range(0, len(conflict_cases), batch_size), desc=f"Baseline-{prompt_type}"):
            batch = conflict_cases[i:i+batch_size]
            
            # Prepare prompts
            prompts = []
            for case in batch:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": case['instruction']}
                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(prompt)
            
            # Tokenize
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            
            for response in batch_responses:
                # Extract assistant's response
                if "<|im_start|>assistant\n" in response:
                    response = response.split("<|im_start|>assistant\n")[-1]
                elif "<|im_start|>assistant" in response:
                    response = response.split("<|im_start|>assistant")[-1]
                
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0]
                
                response = response.replace("<|endoftext|>", "").strip()
                responses.append(response)
        
        all_responses[prompt_type] = responses
    
    # Evaluate
    logging.info("\nEvaluating conflict behaviors...")
    results = {
        'total_cases': len(conflict_cases),
        'passed': 0,
        'failed': 0,
        'ambiguous': 0,
        'no_match': 0,
        'cases': []
    }
    
    for i, test_case in enumerate(tqdm(conflict_cases, desc="Evaluating")):
        case_result = {
            'question': test_case['instruction'],
            'evaluation_guide': test_case['evaluation_guide'],
            'responses': {
                'strict': all_responses['strict'][i],
                'service': all_responses['service'][i]
            },
            'behaviors': {},
            'success': True
        }
        
        # Evaluate behaviors
        for prompt_type in ['strict', 'service']:
            response = case_result['responses'][prompt_type]
            behavior = evaluate_conflict_behavior(response, test_case['evaluation_guide'])
            case_result['behaviors'][prompt_type] = behavior
            
            expected_behavior = f"{prompt_type.upper()}_BEHAVIOR"
            if behavior != expected_behavior:
                case_result['success'] = False
                
                if behavior == 'AMBIGUOUS':
                    results['ambiguous'] += 1
                elif behavior == 'NO_MATCH':
                    results['no_match'] += 1
        
        results['cases'].append(case_result)
        
        if case_result['success']:
            results['passed'] += 1
        else:
            results['failed'] += 1
    
    results['pass_rate'] = results['passed'] / results['total_cases']
    
    # Save results
    output_dir = results_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "conflict_test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"\n✅ Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("CONFLICT TEST RESULTS - EXP000 (Standard FedAvg)")
    print("="*80)
    print(f"\nTotal Cases: {results['total_cases']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Ambiguous: {results['ambiguous']}")
    print(f"No Match: {results['no_match']}")
    print(f"Pass Rate: {results['pass_rate']:.1%}")
    print("\n预期：Standard FedAvg 应该表现很差（逻辑混乱）")
    print("="*80)

if __name__ == "__main__":
    main()
