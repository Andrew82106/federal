#!/usr/bin/env python3
"""
Resume EXP000 training from round 5
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
os.chdir(project_root)

from src.utils.logger import setup_logger
from src.utils.config import load_config
from tools.runners.fl_runner import FLRunner

def main():
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    
    # Setup logging
    log_dir = project_root / config['output']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(log_dir), experiment_name=config['experiment']['name'] + "_resume")
    
    logging.info("="*80)
    logging.info(f"Resuming {config['experiment']['name']} from Round 5")
    logging.info("="*80)
    
    # Initialize FL Runner
    runner = FLRunner(config)
    
    # Run only round 5
    logging.info("\n" + "="*80)
    logging.info("Running Round 5")
    logging.info("="*80)
    
    round_result = runner.run_round(5)
    
    logging.info("\n" + "="*80)
    logging.info("Round 5 completed!")
    logging.info("="*80)
    
    return round_result

if __name__ == "__main__":
    main()
