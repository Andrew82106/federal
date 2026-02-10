"""
Experiment Evaluator

This module provides functions to evaluate trained models on test sets
and compare different methods (Local Only, Standard FedAvg, Dual-Adapter).
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from src.models.base_model import load_base_model
from src.models.dual_adapter import DualAdapterModel
from src.data.dataset import apply_qwen_chat_template
from tools.evaluators.metrics import calculate_accuracy, calculate_conflict_resolution_rate


logger = logging.getLogger(__name__)


def load_test_data(test_path: str) -> List[Dict]:
    """
    Load test data from JSON file.
    
    Args:
        test_path: Path to test data JSON file
        
    Returns:
        List of test samples
    """
    if not os.path.exists(test_path):
        logger.warning(f"Test file not found: {test_path}")
        return []
    
    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} test samples from {test_path}")
    return data


def evaluate_method(
    method_name: str,
    adapter_paths: Dict[str, str],
    test_sets: Dict[str, str],
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    system_prompts: Optional[Dict[str, str]] = None,
    max_new_tokens: int = 512,
    device: str = "cuda"
) -> Dict[str, any]:
    """
    Evaluate a single method on all test sets.
    
    Args:
        method_name: Name of the method (e.g., "dual_adapter", "local_only")
        adapter_paths: Dictionary mapping adapter names to paths
        test_sets: Dictionary mapping test set names to file paths
        base_model_name: Name of the base model
        system_prompts: Optional system prompts for different adapters
        max_new_tokens: Maximum number of tokens to generate
        device: Device to run on
        
    Returns:
        Dictionary containing evaluation results
    """
    logger.info(f"Evaluating method: {method_name}")
    
    results = {
        "method": method_name,
        "test_results": {},
        "overall_accuracy": 0.0
    }
    
    # Load base model and tokenizer
    logger.info("Loading base model...")
    base_model, tokenizer = load_base_model(
        model_name=base_model_name,
        device_map=device
    )
    
    # Evaluate on each test set
    total_correct = 0
    total_samples = 0
    
    for test_name, test_path in test_sets.items():
        logger.info(f"Evaluating on {test_name}...")
        
        # Load test data
        test_data = load_test_data(test_path)
        if not test_data:
            logger.warning(f"Skipping {test_name} (no data)")
            continue
        
        # Determine which adapter to use for this test set
        adapter_name = _get_adapter_for_test(test_name, adapter_paths)
        adapter_path = adapter_paths.get(adapter_name)
        
        if not adapter_path or not os.path.exists(adapter_path):
            logger.warning(f"Adapter not found: {adapter_path}")
            continue
        
        # Load model with adapter
        logger.info(f"Loading adapter: {adapter_name} from {adapter_path}")
        model = DualAdapterModel(base_model, lora_config=None)
        model.load_adapter(adapter_name, adapter_path)
        model.set_active_adapters([adapter_name])
        
        # Get system prompt for this adapter
        system_prompt = ""
        if system_prompts and adapter_name in system_prompts:
            system_prompt = system_prompts[adapter_name]
        
        # Evaluate on test set
        predictions = []
        references = []
        
        for sample in test_data:
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            expected_output = sample.get("output", "")
            
            # Generate prediction
            prompt = apply_qwen_chat_template(
                instruction=instruction,
                input_text=input_text,
                tokenizer=tokenizer,
                system_prompt=system_prompt
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.get_model().generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0
                )
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "<|im_start|>assistant" in prediction:
                prediction = prediction.split("<|im_start|>assistant")[-1]
                prediction = prediction.split("<|im_end|>")[0].strip()
            
            predictions.append(prediction)
            references.append(expected_output)
        
        # Calculate accuracy
        accuracy = calculate_accuracy(predictions, references)
        
        results["test_results"][test_name] = {
            "accuracy": accuracy,
            "num_samples": len(test_data),
            "adapter_used": adapter_name
        }
        
        total_correct += int(accuracy * len(test_data))
        total_samples += len(test_data)
        
        logger.info(f"{test_name}: {accuracy:.2%} ({len(test_data)} samples)")
    
    # Calculate overall accuracy
    if total_samples > 0:
        results["overall_accuracy"] = total_correct / total_samples
    
    logger.info(f"Overall accuracy for {method_name}: {results['overall_accuracy']:.2%}")
    
    return results


def _get_adapter_for_test(test_name: str, adapter_paths: Dict[str, str]) -> str:
    """
    Determine which adapter to use for a given test set.
    
    Args:
        test_name: Name of the test set
        adapter_paths: Available adapter paths
        
    Returns:
        Name of the adapter to use
    """
    test_name_lower = test_name.lower()
    
    # Test-G (global laws) -> use global adapter
    if "global" in test_name_lower or "test-g" in test_name_lower or "test_g" in test_name_lower:
        if "global" in adapter_paths:
            return "global"
    
    # Test-A (strict policies) -> use strict adapter
    if "strict" in test_name_lower or "test-a" in test_name_lower or "test_a" in test_name_lower:
        if "strict" in adapter_paths:
            return "strict"
    
    # Test-B (service policies) -> use service adapter
    if "service" in test_name_lower or "test-b" in test_name_lower or "test_b" in test_name_lower:
        if "service" in adapter_paths:
            return "service"
    
    # Default: use first available adapter
    return list(adapter_paths.keys())[0] if adapter_paths else "global"


def compare_methods(
    methods_results: List[Dict],
    output_path: Optional[str] = None
) -> Dict:
    """
    Compare results from multiple methods.
    
    Args:
        methods_results: List of results dictionaries from evaluate_method
        output_path: Optional path to save comparison results
        
    Returns:
        Dictionary containing comparison results
    """
    logger.info("Comparing methods...")
    
    comparison = {
        "methods": [],
        "test_sets": {},
        "summary": {}
    }
    
    # Collect all test set names
    all_test_sets = set()
    for result in methods_results:
        all_test_sets.update(result["test_results"].keys())
    
    # Build comparison table
    for test_name in sorted(all_test_sets):
        comparison["test_sets"][test_name] = {}
        
        for result in methods_results:
            method_name = result["method"]
            
            if test_name in result["test_results"]:
                accuracy = result["test_results"][test_name]["accuracy"]
                comparison["test_sets"][test_name][method_name] = accuracy
            else:
                comparison["test_sets"][test_name][method_name] = None
    
    # Overall accuracy comparison
    for result in methods_results:
        method_name = result["method"]
        overall_acc = result["overall_accuracy"]
        
        comparison["methods"].append({
            "name": method_name,
            "overall_accuracy": overall_acc
        })
    
    # Find best method
    best_method = max(comparison["methods"], key=lambda x: x["overall_accuracy"])
    comparison["summary"]["best_method"] = best_method["name"]
    comparison["summary"]["best_accuracy"] = best_method["overall_accuracy"]
    
    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        logger.info(f"Comparison results saved to {output_path}")
    
    # Print comparison table
    logger.info("\n=== Method Comparison ===")
    logger.info(f"{'Method':<20} {'Overall Accuracy':<20}")
    logger.info("-" * 40)
    for method_info in comparison["methods"]:
        logger.info(f"{method_info['name']:<20} {method_info['overall_accuracy']:<20.2%}")
    
    logger.info(f"\nBest method: {comparison['summary']['best_method']} "
                f"({comparison['summary']['best_accuracy']:.2%})")
    
    return comparison


def evaluate_conflict_resolution(
    adapter_paths: Dict[str, str],
    conflict_cases_path: str,
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    system_prompts: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None
) -> Dict:
    """
    Evaluate conflict resolution capability using conflict test cases.
    
    Args:
        adapter_paths: Dictionary mapping adapter names to paths
        conflict_cases_path: Path to conflict test cases JSON
        base_model_name: Name of the base model
        system_prompts: Optional system prompts for different adapters
        output_path: Optional path to save results
        
    Returns:
        Dictionary containing conflict resolution results
    """
    logger.info("Evaluating conflict resolution...")
    
    # Load conflict cases
    if not os.path.exists(conflict_cases_path):
        logger.error(f"Conflict cases file not found: {conflict_cases_path}")
        return {}
    
    with open(conflict_cases_path, 'r', encoding='utf-8') as f:
        conflict_cases = json.load(f)
    
    logger.info(f"Loaded {len(conflict_cases)} conflict cases")
    
    # Load base model
    base_model, tokenizer = load_base_model(
        model_name=base_model_name,
        device_map="cuda"
    )
    
    results = {
        "num_cases": len(conflict_cases),
        "cases": [],
        "conflict_resolution_rate": 0.0
    }
    
    resolved_count = 0
    
    for i, case in enumerate(conflict_cases):
        question = case.get("question", "")
        expected_outputs = case.get("expected_outputs", {})
        
        logger.info(f"Testing case {i+1}/{len(conflict_cases)}: {question[:50]}...")
        
        case_result = {
            "question": question,
            "outputs": {},
            "resolved": False
        }
        
        # Test with each adapter
        for adapter_name, adapter_path in adapter_paths.items():
            if adapter_name == "global":
                continue  # Skip global adapter for conflict tests
            
            # Load adapter
            model = DualAdapterModel(base_model, lora_config=None)
            model.load_adapter(adapter_name, adapter_path)
            model.set_active_adapters([adapter_name])
            
            # Get system prompt
            system_prompt = ""
            if system_prompts and adapter_name in system_prompts:
                system_prompt = system_prompts[adapter_name]
            
            # Generate response
            prompt = apply_qwen_chat_template(
                instruction=question,
                tokenizer=tokenizer,
                system_prompt=system_prompt
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.get_model().generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1]
                response = response.split("<|im_end|>")[0].strip()
            
            case_result["outputs"][adapter_name] = response
            
            # Check if response contains expected keywords
            if adapter_name in expected_outputs:
                expected_keywords = expected_outputs[adapter_name].get("keywords", [])
                contains_keywords = any(kw in response for kw in expected_keywords)
                case_result[f"{adapter_name}_correct"] = contains_keywords
        
        # Check if conflict is resolved (different outputs for different adapters)
        outputs_list = list(case_result["outputs"].values())
        if len(outputs_list) >= 2:
            # Simple check: outputs should be different
            case_result["resolved"] = outputs_list[0] != outputs_list[1]
            if case_result["resolved"]:
                resolved_count += 1
        
        results["cases"].append(case_result)
    
    # Calculate conflict resolution rate
    if len(conflict_cases) > 0:
        results["conflict_resolution_rate"] = resolved_count / len(conflict_cases)
    
    logger.info(f"Conflict resolution rate: {results['conflict_resolution_rate']:.2%}")
    
    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Conflict resolution results saved to {output_path}")
    
    return results
