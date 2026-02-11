#!/usr/bin/env python3
"""
独立的 Conflict Test 评估脚本 - EXP001
不加载其他模型，避免 GPU 内存问题
"""

import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from src.utils.config import load_config
from tools.evaluators.conflict_tester import ConflictTester

def main():
    # Setup logging
    log_dir = project_root / "results" / "exp001_dual_adapter_fl" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(log_dir), experiment_name="eval_conflict_only")
    
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    
    logging.info("="*80)
    logging.info("CONFLICT TEST ONLY - EXP001")
    logging.info("="*80)
    
    # Define paths
    results_dir = project_root / "results" / "exp001_dual_adapter_fl"
    test_data_dir = project_root / "data" / "test"
    
    adapter_paths = {
        'strict': str(results_dir / "checkpoints" / "round_5" / "client_strict" / "local"),
        'service': str(results_dir / "checkpoints" / "round_5" / "client_service" / "local")
    }
    
    global_adapter_path = str(results_dir / "checkpoints" / "round_5" / "global_adapter")
    
    # System prompts
    system_prompts = {
        'strict': '你是上海市（户政）与北京市（交管）的联合政务助手。请依据这两个城市严格、规范的管理规定进行回答。对于违规行为，请强调处罚和红线。',
        'service': '你是石家庄市（户政）与南宁市（交管）的联合政务助手。请依据这两个城市便民、宽松、人性化的政策进行回答。对于轻微违章，请强调教育与纠正。'
    }
    
    # Load conflict cases
    with open(test_data_dir / "conflict_cases.json", 'r', encoding='utf-8') as f:
        conflict_cases = json.load(f)
    
    # Initialize tester
    tester = ConflictTester(
        base_model_name=config['model']['base_model'],
        global_adapter_path=global_adapter_path,
        config=config['model']
    )
    
    # Run test
    results = tester.run_guided_test_suite(
        test_cases=conflict_cases,
        local_adapter_paths=adapter_paths,
        system_prompts=system_prompts,
        batch_size=16
    )
    
    # Save results
    output_dir = results_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "conflict_test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"\n✅ Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("CONFLICT TEST RESULTS - EXP001")
    print("="*80)
    print(f"\nTotal Cases: {results['total_cases']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Pass Rate: {results['pass_rate']:.1%}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
