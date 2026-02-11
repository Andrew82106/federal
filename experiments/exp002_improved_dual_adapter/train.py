#!/usr/bin/env python3
"""
Training script for improved dual-adapter federated learning experiment.

This experiment uses a two-phase training strategy:
- Phase 1: Train global adapter with global data only (3 rounds × 3 epochs)
- Phase 2: Freeze global, train local adapters with local data only (2 rounds × 5 epochs)

Usage:
    python train.py --config config.yaml
"""

import os
import sys
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.config import load_config, validate_config
from src.utils.logger import setup_logger, log_experiment_config
from tools.runners.fl_runner import FLRunner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train improved dual-adapter federated learning model')
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/exp002_improved_dual_adapter/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory'
    )
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.output_dir:
        config['experiment']['output_dir'] = args.output_dir
    
    # Validate configuration
    validate_config(config)
    
    # Setup logging
    log_dir = os.path.join(config['experiment']['output_dir'], 'logs')
    logger = setup_logger(
        log_dir=log_dir,
        log_level=config['logging']['log_level'],
        experiment_name=config['experiment']['name']
    )
    
    # Log configuration
    log_experiment_config(logger, config)
    
    logger.info("="*80)
    logger.info("EXPERIMENT 002: Improved Dual-Adapter FL")
    logger.info("="*80)
    logger.info("Strategy: Mixed Training with Enhanced Parameters")
    logger.info(f"  Rounds: {config['federated']['num_rounds']}")
    logger.info(f"  Epochs per round: {config['training']['num_epochs']}")
    logger.info(f"  LoRA: r={config['model']['lora_config']['r']}, alpha={config['model']['lora_config']['lora_alpha']}")
    logger.info(f"  Improvements over exp001:")
    logger.info(f"    - LoRA rank doubled (16→32)")
    logger.info(f"    - LoRA alpha doubled (32→64)")
    logger.info(f"    - Training epochs increased (2→3)")
    logger.info("="*80)
    
    logger.info("Creating FLRunner...")
    sys.stdout.flush()
    
    # Create FL runner
    runner = FLRunner(config)
    
    logger.info("FLRunner created successfully, starting experiment...")
    sys.stdout.flush()
    
    # Run experiment
    try:
        results = runner.run_experiment()
        logger.info("✅ Training completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
