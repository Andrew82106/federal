#!/usr/bin/env python3
"""
EXP000: Standard FedAvg Baseline
单一 Global Adapter，无 Local Adapter
用于对比双适配器架构的优势
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from src.utils.config import load_config
from tools.runners.fl_runner import FLRunner

def main():
    # Change to project root directory
    import os
    os.chdir(project_root)
    
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    
    # Setup logging
    log_dir = project_root / config['output']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(log_dir), experiment_name=config['experiment']['name'])
    
    logging.info("="*80)
    logging.info(f"Starting {config['experiment']['name']}")
    logging.info(f"Description: {config['experiment']['description']}")
    logging.info("="*80)
    
    # Initialize FL Runner
    runner = FLRunner(config)
    
    # Run federated learning
    results = runner.run_experiment()
    
    logging.info("\n" + "="*80)
    logging.info("Training completed!")
    logging.info("="*80)
    
    return results

if __name__ == "__main__":
    main()
