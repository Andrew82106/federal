"""
Quick evaluation script for rapid testing.

This script runs evaluation on a small subset of each test set to quickly
verify that the model is working correctly before running the full evaluation.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the full evaluator
from experiments.exp001_dual_adapter_fl.eval import ComprehensiveEvaluator, print_summary
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main():
    """Run quick evaluation on subset of data."""
    
    # Setup logging
    log_dir = project_root / "results" / "exp001_dual_adapter_fl" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_file=str(log_dir / "quick_evaluation.log"))
    
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    
    logging.info("="*80)
    logging.info("QUICK EVALUATION (Subset)")
    logging.info("="*80)
    
    # Define paths
    results_dir = project_root / "results" / "exp001_dual_adapter_fl"
    test_data_dir = project_root / "data" / "test"
    
    adapter_paths = {
        'strict': str(results_dir / "checkpoints" / "final_adapters" / "strict" / "local"),
        'service': str(results_dir / "checkpoints" / "final_adapters" / "service" / "local")
    }
    
    global_adapter_path = str(results_dir / "checkpoints" / "final_adapters" / "global")
    
    # Check if adapters exist
    for name, path in adapter_paths.items():
        if not Path(path).exists():
            logging.error(f"Adapter not found: {path}")
            logging.error("Please ensure training completed successfully")
            return
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        base_model_name=config['model']['base_model'],
        global_adapter_path=global_adapter_path,
        local_adapter_paths=adapter_paths,
        config=config['model']
    )
    
    # Load test data (subset only)
    SUBSET_SIZE = 5  # Test only 5 cases per set
    
    logging.info(f"\nTesting with {SUBSET_SIZE} cases per test set")
    
    all_results = {}
    
    # 1. Test-G: Universal law knowledge
    logging.info("\n" + "="*80)
    logging.info("TEST-G: Universal Law Knowledge (Quick)")
    logging.info("="*80)
    
    with open(test_data_dir / "global_test.json", 'r', encoding='utf-8') as f:
        global_test = json.load(f)[:SUBSET_SIZE]
    
    all_results['test_g'] = {
        'strict': evaluator.evaluate_with_keywords(
            global_test, 'strict', 'Test-G (Strict)'
        )
    }
    
    # 2. Test-A: Strict city policy
    logging.info("\n" + "="*80)
    logging.info("TEST-A: Strict City Policy (Quick)")
    logging.info("="*80)
    
    with open(test_data_dir / "strict_test.json", 'r', encoding='utf-8') as f:
        strict_test = json.load(f)[:SUBSET_SIZE]
    
    all_results['test_a'] = {
        'strict': evaluator.evaluate_with_keywords(
            strict_test, 'strict', 'Test-A (Strict)'
        ),
        'service': evaluator.evaluate_with_keywords(
            strict_test, 'service', 'Test-A (Service - Cross)'
        )
    }
    
    # 3. Test-B: Service city policy
    logging.info("\n" + "="*80)
    logging.info("TEST-B: Service City Policy (Quick)")
    logging.info("="*80)
    
    with open(test_data_dir / "service_test.json", 'r', encoding='utf-8') as f:
        service_test = json.load(f)[:SUBSET_SIZE]
    
    all_results['test_b'] = {
        'service': evaluator.evaluate_with_keywords(
            service_test, 'service', 'Test-B (Service)'
        ),
        'strict': evaluator.evaluate_with_keywords(
            service_test, 'strict', 'Test-B (Strict - Cross)'
        )
    }
    
    # 4. Conflict Test
    logging.info("\n" + "="*80)
    logging.info("CONFLICT TEST: Non-IID (Quick)")
    logging.info("="*80)
    
    with open(test_data_dir / "conflict_cases.json", 'r', encoding='utf-8') as f:
        conflict_cases = json.load(f)[:SUBSET_SIZE]
    
    all_results['conflict'] = evaluator.evaluate_conflict_cases(conflict_cases)
    
    # Add global_only for summary compatibility
    all_results['test_g']['global_only'] = all_results['test_g']['strict']
    all_results['test_g']['service'] = all_results['test_g']['strict']
    
    # Save results
    output_dir = results_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "quick_evaluation_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"\n✅ Results saved to: {output_path}")
    
    # Print summary
    print_summary(all_results)
    
    print("\n" + "="*80)
    print("⚠️  This was a QUICK evaluation on subset of data")
    print("For complete results, run:")
    print("  python experiments/exp001_dual_adapter_fl/eval.py")
    print("="*80)


if __name__ == "__main__":
    main()
