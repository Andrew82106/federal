#!/usr/bin/env python3
"""
Training script for dual-adapter federated learning experiment.

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
    parser = argparse.ArgumentParser(description='Train dual-adapter federated learning model')
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/exp001_dual_adapter_fl/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory'
    )
    parser.add_argument(
        '--num_rounds',
        type=int,
        default=None,
        help='Override number of federated rounds'
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
    if args.num_rounds:
        config['federated']['num_rounds'] = args.num_rounds
    
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
    
    # Create FL runner
    runner = FLRunner(config)
    
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
