"""
Evaluation Script for Experiment 001: Dual-Adapter Federated Learning

This script evaluates trained models on test sets and generates reports.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from tools.evaluators.experiment_evaluator import (
    evaluate_method,
    compare_methods,
    evaluate_conflict_resolution
)
from tools.visualizers.plot_results import create_all_visualizations
from tools.visualizers.report_generator import create_experiment_summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Dual-Adapter Federated Learning Experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory containing model checkpoints (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for evaluation results (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    
    config = load_config(str(config_path))
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = config["experiment"]["output_dir"]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    logger = setup_logger(str(log_dir), "evaluation")
    
    logger.info("=" * 80)
    logger.info("Dual-Adapter Federated Learning - Evaluation")
    logger.info("=" * 80)
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Determine checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = output_dir / "checkpoints" / "final_adapters"
    
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        logger.error("Please run training first or specify --checkpoint-dir")
        return 1
    
    logger.info(f"Loading checkpoints from: {checkpoint_dir}")
    
    # Define adapter paths
    adapter_paths = {
        "global": str(checkpoint_dir / "global"),
        "strict": str(checkpoint_dir / "strict"),
        "service": str(checkpoint_dir / "service")
    }
    
    # Verify adapters exist
    missing_adapters = []
    for name, path in adapter_paths.items():
        if not Path(path).exists():
            missing_adapters.append(name)
            logger.warning(f"Adapter not found: {name} at {path}")
    
    if missing_adapters:
        logger.error(f"Missing adapters: {', '.join(missing_adapters)}")
        logger.error("Cannot proceed with evaluation")
        return 1
    
    # Define test sets
    test_sets = {}
    data_config = config.get("data", {})
    
    test_paths = {
        "Test-G (Global Laws)": data_config.get("test_global", "data/test/global_test.json"),
        "Test-A (Strict Policies)": data_config.get("test_strict", "data/test/strict_test.json"),
        "Test-B (Service Policies)": data_config.get("test_service", "data/test/service_test.json")
    }
    
    # Check which test sets exist
    for name, path in test_paths.items():
        if Path(path).exists():
            test_sets[name] = path
            logger.info(f"Found test set: {name}")
        else:
            logger.warning(f"Test set not found: {name} at {path}")
    
    if not test_sets:
        logger.error("No test sets found. Cannot perform evaluation.")
        logger.info("Please create test data in data/test/ directory")
        return 1
    
    # Define system prompts
    system_prompts = {
        "strict": "你是上海市公安局的政务助手，请根据上海市的政策回答问题。",
        "service": "你是石家庄市公安局的政务助手，请根据石家庄市的政策回答问题。"
    }
    
    # Evaluate dual-adapter method
    logger.info("\n" + "=" * 80)
    logger.info("Evaluating Dual-Adapter Method")
    logger.info("=" * 80)
    
    dual_adapter_results = evaluate_method(
        method_name="Dual-Adapter",
        adapter_paths=adapter_paths,
        test_sets=test_sets,
        base_model_name=config["model"]["base_model"],
        system_prompts=system_prompts
    )
    
    # For comparison, we would need Local Only and Standard FedAvg results
    # For now, we'll just use the dual-adapter results
    methods_results = [dual_adapter_results]
    
    # Compare methods
    logger.info("\n" + "=" * 80)
    logger.info("Comparing Methods")
    logger.info("=" * 80)
    
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    comparison = compare_methods(
        methods_results=methods_results,
        output_path=str(metrics_dir / "comparison.json")
    )
    
    # Evaluate conflict resolution
    conflict_cases_path = data_config.get("conflict_cases", "data/test/conflict_cases.json")
    
    if Path(conflict_cases_path).exists():
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating Conflict Resolution")
        logger.info("=" * 80)
        
        conflict_results = evaluate_conflict_resolution(
            adapter_paths=adapter_paths,
            conflict_cases_path=conflict_cases_path,
            base_model_name=config["model"]["base_model"],
            system_prompts=system_prompts,
            output_path=str(metrics_dir / "conflict_results.json")
        )
    else:
        logger.warning(f"Conflict cases not found: {conflict_cases_path}")
        logger.warning("Skipping conflict resolution evaluation")
        conflict_results = None
    
    # Create visualizations
    logger.info("\n" + "=" * 80)
    logger.info("Creating Visualizations")
    logger.info("=" * 80)
    
    create_all_visualizations(
        results_dir=str(output_dir),
        output_dir=str(output_dir / "report")
    )
    
    # Generate report
    logger.info("\n" + "=" * 80)
    logger.info("Generating Report")
    logger.info("=" * 80)
    
    create_experiment_summary(
        results_dir=str(output_dir),
        experiment_name=config["experiment"]["name"],
        output_dir=str(output_dir / "report")
    )
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Complete")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Report available at: {output_dir / 'report' / 'experiment_report.md'}")
    
    if conflict_results:
        logger.info(f"Conflict Resolution Rate: {conflict_results['conflict_resolution_rate']:.2%}")
    
    logger.info("\nNext steps:")
    logger.info(f"  1. Review report: cat {output_dir / 'report' / 'experiment_report.md'}")
    logger.info(f"  2. View visualizations in: {output_dir / 'report'}")
    logger.info(f"  3. Check detailed results: {metrics_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
