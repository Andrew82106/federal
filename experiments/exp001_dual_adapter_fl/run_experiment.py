"""
Complete Experiment Pipeline for Experiment 001: Dual-Adapter Federated Learning

This script runs the complete pipeline:
1. Train all methods (Local Only, Standard FedAvg, Dual-Adapter)
2. Evaluate all methods on test sets
3. Generate comparison reports and visualizations
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.logger import setup_logger


def run_command(cmd: list, description: str, logger: logging.Logger) -> int:
    """
    Run a command and log output.
    
    Args:
        cmd: Command to run as list
        description: Description of the command
        logger: Logger instance
        
    Returns:
        Return code
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{description}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        logger.info(f"✓ {description} completed successfully")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with return code {e.returncode}")
        return e.returncode
    except Exception as e:
        logger.error(f"✗ {description} failed: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run complete Dual-Adapter Federated Learning experiment pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only run evaluation"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation and only run training"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["dual_adapter"],
        choices=["dual_adapter", "local_only", "standard_fedavg", "all"],
        help="Methods to train and evaluate"
    )
    
    args = parser.parse_args()
    
    # Expand "all" to all methods
    if "all" in args.methods:
        args.methods = ["dual_adapter", "local_only", "standard_fedavg"]
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    
    config = load_config(str(config_path))
    
    # Setup output directory
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_pipeline_{timestamp}.log"
    
    logger = setup_logger(str(log_dir), f"pipeline_{timestamp}")
    
    logger.info("=" * 80)
    logger.info("Dual-Adapter Federated Learning - Complete Experiment Pipeline")
    logger.info("=" * 80)
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Methods to run: {', '.join(args.methods)}")
    logger.info(f"Log file: {log_file}")
    
    # Change to experiment directory
    experiment_dir = Path(__file__).parent
    os.chdir(experiment_dir)
    
    # Phase 1: Training
    if not args.skip_training:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: TRAINING")
        logger.info("=" * 80)
        
        for method in args.methods:
            logger.info(f"\nTraining method: {method}")
            
            # Prepare training command
            train_cmd = [
                sys.executable,
                "train.py",
                "--config", str(config_path)
            ]
            
            # Add method-specific arguments if needed
            if method != "dual_adapter":
                # For baseline methods, we would need to modify the config
                # or add command-line arguments to train.py
                logger.warning(f"Baseline method '{method}' training not yet implemented")
                logger.warning("Only dual_adapter training is currently supported")
                continue
            
            # Run training
            ret = run_command(
                train_cmd,
                f"Training {method}",
                logger
            )
            
            if ret != 0:
                logger.error(f"Training failed for {method}")
                logger.error("Stopping pipeline")
                return ret
    else:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: TRAINING (SKIPPED)")
        logger.info("=" * 80)
    
    # Phase 2: Evaluation
    if not args.skip_evaluation:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: EVALUATION")
        logger.info("=" * 80)
        
        # Prepare evaluation command
        eval_cmd = [
            sys.executable,
            "eval.py",
            "--config", str(config_path)
        ]
        
        # Run evaluation
        ret = run_command(
            eval_cmd,
            "Evaluation",
            logger
        )
        
        if ret != 0:
            logger.error("Evaluation failed")
            logger.error("Stopping pipeline")
            return ret
    else:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: EVALUATION (SKIPPED)")
        logger.info("=" * 80)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT PIPELINE COMPLETE")
    logger.info("=" * 80)
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    if not args.skip_evaluation:
        report_path = output_dir / "report" / "experiment_report.md"
        if report_path.exists():
            logger.info(f"Report available at: {report_path}")
            logger.info("\nTo view the report:")
            logger.info(f"  cat {report_path}")
        
        logger.info("\nVisualization files:")
        report_dir = output_dir / "report"
        if report_dir.exists():
            for file in report_dir.glob("*.png"):
                logger.info(f"  - {file.name}")
            for file in report_dir.glob("*.md"):
                if file.name != "experiment_report.md":
                    logger.info(f"  - {file.name}")
    
    logger.info("\nNext steps:")
    if args.skip_evaluation:
        logger.info("  1. Run evaluation: python eval.py")
    else:
        logger.info("  1. Review the experiment report")
        logger.info("  2. Analyze visualizations")
        logger.info("  3. Check detailed metrics in metrics/ directory")
    
    logger.info(f"\nFull log available at: {log_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
