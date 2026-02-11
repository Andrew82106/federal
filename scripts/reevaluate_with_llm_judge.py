#!/usr/bin/env python3
"""
Re-evaluate experiment results using LLM-as-Judge.

This script loads existing evaluation checkpoints and re-judges
the answers using Qwen as a judge model for more accurate evaluation.

Usage:
    python scripts/reevaluate_with_llm_judge.py --exp exp001_dual_adapter_fl
"""

import json
import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.evaluators.llm_judge import LLMJudge
from src.utils.logger import setup_logger


def reevaluate_test_set(
    judge: LLMJudge,
    checkpoint_file: Path,
    output_file: Path,
    batch_size: int = 16
):
    """Re-evaluate a test set using LLM judge."""
    
    # Load checkpoint
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logging.info(f"Re-evaluating {checkpoint_file.name}...")
    logging.info(f"  Original accuracy: {data['accuracy']:.2%}")
    
    # Extract questions, expected, and responses
    questions = [d['question'] for d in data['details']]
    expected_answers = [d['expected'] for d in data['details']]
    responses = [d['response'] for d in data['details']]
    
    # Judge with LLM
    judgments = judge.judge_batch(questions, expected_answers, responses, batch_size=batch_size)
    
    # Update results
    correct = 0
    for i, (is_correct, explanation) in enumerate(judgments):
        data['details'][i]['llm_correct'] = is_correct
        data['details'][i]['llm_explanation'] = explanation
        if is_correct:
            correct += 1
    
    # Update accuracy
    data['llm_accuracy'] = correct / len(data['details'])
    data['llm_correct'] = correct
    data['original_accuracy'] = data['accuracy']
    data['original_correct'] = data['correct']
    
    # Save updated results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"  LLM Judge accuracy: {data['llm_accuracy']:.2%}")
    logging.info(f"  Improvement: {data['llm_accuracy'] - data['original_accuracy']:+.2%}")
    logging.info(f"  Saved to: {output_file}")
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Re-evaluate with LLM judge')
    parser.add_argument('--exp', type=str, default='exp001_dual_adapter_fl',
                       help='Experiment name')
    parser.add_argument('--model', type=str, default='/root/autodl-tmp/Downloads',
                       help='Path to judge model')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for LLM judge (default: 16)')
    args = parser.parse_args()
    
    # Setup logging
    log_dir = project_root / "results" / args.exp / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(log_dir), experiment_name="llm_judge_reevaluation")
    
    logging.info("="*80)
    logging.info("RE-EVALUATION WITH LLM JUDGE")
    logging.info("="*80)
    
    # Initialize judge
    judge = LLMJudge(args.model)
    
    # Find checkpoint files
    checkpoint_dir = project_root / "results" / args.exp / "eval_checkpoints"
    if not checkpoint_dir.exists():
        logging.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return 1
    
    checkpoint_files = list(checkpoint_dir.glob("*.json"))
    if not checkpoint_files:
        logging.error(f"No checkpoint files found in {checkpoint_dir}")
        return 1
    
    logging.info(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Create output directory
    output_dir = project_root / "results" / args.exp / "llm_judge_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Re-evaluate each checkpoint
    all_results = {}
    
    for checkpoint_file in sorted(checkpoint_files):
        if checkpoint_file.name == "conflict_test.json":
            logging.info(f"Skipping {checkpoint_file.name} (uses keyword matching)")
            continue
        
        output_file = output_dir / checkpoint_file.name
        result = reevaluate_test_set(judge, checkpoint_file, output_file, batch_size=args.batch_size)
        
        # Store for summary
        test_name = checkpoint_file.stem
        all_results[test_name] = {
            'original_accuracy': result['original_accuracy'],
            'llm_accuracy': result['llm_accuracy'],
            'improvement': result['llm_accuracy'] - result['original_accuracy']
        }
    
    # Print summary
    print("\n" + "="*80)
    print("RE-EVALUATION SUMMARY")
    print("="*80)
    print(f"\n{'Test Set':<30} {'Original':<12} {'LLM Judge':<12} {'Improvement':<12}")
    print("-"*80)
    
    for test_name, result in sorted(all_results.items()):
        print(f"{test_name:<30} {result['original_accuracy']:>10.1%} {result['llm_accuracy']:>10.1%} {result['improvement']:>+10.1%}")
    
    avg_improvement = sum(r['improvement'] for r in all_results.values()) / len(all_results)
    print("-"*80)
    print(f"{'Average Improvement':<30} {avg_improvement:>+10.1%}")
    print("="*80)
    
    # Save summary
    summary_file = output_dir / "reevaluation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"\n✅ Summary saved to: {summary_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
