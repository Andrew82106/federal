"""
Conflict evaluation script using the enhanced keyword-based evaluation.

This script loads trained adapters and tests them on conflict cases using
fast keyword matching instead of expensive LLM judges.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.evaluators.conflict_tester import ConflictTester
from src.utils.config import load_config
from src.utils.logger import setup_logger

def main():
    """Run conflict evaluation with keyword-based matching."""
    
    # Setup logging
    setup_logger(log_dir="results/exp001_dual_adapter_fl/logs", experiment_name="conflict_eval")
    
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    
    logging.info("="*80)
    logging.info("Conflict Evaluation with Keyword Matching")
    logging.info("="*80)
    
    # Load test cases
    test_data_path = project_root / "data" / "test" / "conflict_cases.json"
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    logging.info(f"Loaded {len(test_cases)} conflict test cases")
    
    # Define adapter paths (update these after training)
    results_dir = project_root / "results" / "exp001_dual_adapter_fl"
    adapter_paths = {
        'strict': str(results_dir / "checkpoints" / "final_adapters" / "strict" / "local"),
        'service': str(results_dir / "checkpoints" / "final_adapters" / "service" / "local")
    }
    
    global_adapter_path = str(results_dir / "checkpoints" / "final_adapters" / "global")
    
    # Check if adapters exist
    for name, path in adapter_paths.items():
        if not Path(path).exists():
            logging.error(f"Adapter not found: {path}")
            logging.error("Please train the model first using train.py")
            return
    
    # Initialize tester
    logging.info("Initializing ConflictTester...")
    tester = ConflictTester(
        base_model_name=config['model']['base_model'],
        global_adapter_path=global_adapter_path,
        config=config['model']
    )
    
    # Define system prompts (联合城市身份)
    system_prompts = {
        'strict': '你是上海市（户政）与北京市（交管）的联合政务助手。请依据这两个城市严格、规范的管理规定进行回答。对于违规行为，请强调处罚和红线。',
        'service': '你是石家庄市（户政）与南宁市（交管）的联合政务助手。请依据这两个城市便民、宽松、人性化的政策进行回答。对于轻微违章，请强调教育与纠正。'
    }
    
    # Run evaluation on a subset (for quick testing)
    num_test_cases = min(10, len(test_cases))  # Test first 10 cases
    logging.info(f"Testing on {num_test_cases} cases (set to full dataset for complete eval)")
    
    test_subset = test_cases[:num_test_cases]
    
    # Run guided test suite (uses evaluation_guide for fast evaluation)
    results = tester.run_guided_test_suite(
        test_cases=test_subset,
        local_adapter_paths=adapter_paths,
        system_prompts=system_prompts
    )
    
    # Save results
    output_dir = results_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "conflict_evaluation_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"\nResults saved to: {output_path}")
    
    # Print summary
    logging.info("\n" + "="*80)
    logging.info("EVALUATION SUMMARY")
    logging.info("="*80)
    logging.info(f"Total cases: {results['total_cases']}")
    logging.info(f"Passed: {results['passed']} ({results['pass_rate']:.1%})")
    logging.info(f"Failed: {results['failed']}")
    logging.info(f"Ambiguous: {results['ambiguous']}")
    logging.info(f"No Match: {results['no_match']}")
    logging.info("="*80)
    
    # Show some example cases
    logging.info("\nExample Results:")
    for i, case in enumerate(results['cases'][:3], 1):
        logging.info(f"\n--- Case {i} ---")
        logging.info(f"Question: {case['question'][:100]}...")
        for adapter_name, behavior in case['behaviors'].items():
            logging.info(f"{adapter_name}: {behavior}")

if __name__ == "__main__":
    main()
