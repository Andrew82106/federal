"""
Logging utilities for dual-adapter federated learning experiments.

This module provides functions for setting up structured logging
with both file and console handlers.
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional


def setup_logger(
    log_dir: str,
    log_level: str = "INFO",
    experiment_name: str = "experiment"
) -> logging.Logger:
    """
    Set up logging system with file and console handlers.
    
    Args:
        log_dir: Directory where log files will be saved
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        experiment_name: Name of the experiment for log file naming
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler (detailed logs)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{experiment_name}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (simpler logs)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"✅ Logger initialized. Logs will be saved to: {log_file}")
    
    return logger


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    step: int,
    prefix: str = ""
) -> None:
    """
    Log metrics in a structured format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics to log
        step: Current step/epoch number
        prefix: Optional prefix for metric names (e.g., "train", "eval")
    """
    metric_str = f"Step {step}"
    if prefix:
        metric_str += f" [{prefix}]"
    
    for key, value in metrics.items():
        if isinstance(value, float):
            metric_str += f" | {key}: {value:.4f}"
        else:
            metric_str += f" | {key}: {value}"
    
    logger.info(metric_str)


def save_metrics_to_json(
    metrics: Dict[str, Any],
    save_path: str
) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics to save
        save_path: Path where to save the metrics JSON file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logging.info(f"✅ Metrics saved to {save_path}")


def log_experiment_config(
    logger: logging.Logger,
    config: Dict[str, Any]
) -> None:
    """
    Log experiment configuration in a readable format.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 80)
    
    def log_dict(d: Dict, indent: int = 0):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                log_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")
    
    log_dict(config)
    logger.info("=" * 80)
