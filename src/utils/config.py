"""
Configuration management utilities for dual-adapter federated learning.

This module provides functions for loading, validating, and managing
experiment configurations, including adaptive VRAM detection for
automatic quantization decisions.
"""

import os
import yaml
import logging
import argparse
from typing import Dict, Any, Optional
import torch


def get_adaptive_quantization_config() -> bool:
    """
    Automatically determine if 4-bit quantization is needed based on GPU VRAM.
    
    Target: Qwen2.5-7B (Requires ~15GB+ for BF16 training, ~6GB for 4-bit)
    
    Returns:
        bool: True if 4-bit quantization should be enabled, False otherwise
    """
    if not torch.cuda.is_available():
        logging.warning("⚠️ No CUDA device detected. CPU mode will be very slow.")
        return False  # CPU fallback (slow)
    
    # Get total memory of the first GPU in GB
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Threshold set to 16GB (Safe margin for 7B model in BF16)
    if total_memory_gb < 16:
        logging.warning(
            f"⚠️ Low VRAM detected ({total_memory_gb:.1f} GB). "
            "Enabling 4-bit quantization for efficiency."
        )
        return True
    else:
        logging.info(
            f"✅ Sufficient VRAM detected ({total_memory_gb:.1f} GB). "
            "Using native precision (BF16)."
        )
        return False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"✅ Loaded configuration from {config_path}")
    return config


def merge_config_with_args(
    config: Dict[str, Any],
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Merge configuration file with command-line arguments.
    Command-line arguments take precedence over config file values.
    
    Args:
        config: Configuration dictionary from YAML file
        args: Parsed command-line arguments
        
    Returns:
        Merged configuration dictionary
    """
    # Convert args to dict and filter out None values
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    
    # Merge args into config (args override config)
    for key, value in args_dict.items():
        if key in config:
            logging.info(f"Overriding config['{key}'] = {config[key]} with CLI arg = {value}")
            config[key] = value
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration completeness and correctness.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_sections = ['experiment', 'model', 'training', 'federated', 'data']
    
    # Check required top-level sections
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: '{section}'")
    
    # Validate experiment section
    if 'name' not in config['experiment']:
        raise ValueError("Missing 'experiment.name' in configuration")
    
    # Validate model section
    if 'base_model' not in config['model']:
        raise ValueError("Missing 'model.base_model' in configuration")
    
    # Validate training section
    required_training_params = ['num_epochs', 'learning_rate', 'per_device_train_batch_size']
    for param in required_training_params:
        if param not in config['training']:
            raise ValueError(f"Missing 'training.{param}' in configuration")
    
    # Validate federated section
    if 'num_rounds' not in config['federated']:
        raise ValueError("Missing 'federated.num_rounds' in configuration")
    if 'clients' not in config['federated']:
        raise ValueError("Missing 'federated.clients' in configuration")
    
    # Validate data section
    if 'global_train' not in config['data']:
        raise ValueError("Missing 'data.global_train' in configuration")
    
    logging.info("✅ Configuration validation passed")
    return True


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration template.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'experiment': {
            'name': 'dual_adapter_fl',
            'seed': 42,
            'output_dir': 'results/exp001_dual_adapter_fl'
        },
        'model': {
            'base_model': 'Qwen/Qwen2.5-7B-Instruct',
            'quantization': 'auto',  # 'auto', '4bit', or 'none'
            'lora_config': {
                'r': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.05,
                'target_modules': 'all-linear',  # Apply to all linear layers
                'bias': 'none',
                'task_type': 'CAUSAL_LM'
            }
        },
        'training': {
            'num_epochs': 2,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 4,
            'learning_rate': 2.0e-4,
            'max_seq_length': 1024,
            'warmup_ratio': 0.1,
            'lr_scheduler_type': 'cosine',
            'fp16': True,
            'optim': 'paged_adamw_8bit'
        },
        'federated': {
            'num_rounds': 5,
            'num_clients': 2,
            'aggregation_method': 'fedavg',
            'mode': 'dual_adapter',  # 'dual_adapter', 'standard_fedavg', or 'local_only'
            'clients': [
                {
                    'id': 'strict',
                    'name': '严管城市',
                    'local_data': 'data/rule_data/client_strict.json',
                    'system_prompt': '你是上海市公安局的政务助手，请根据上海市的政策回答问题。'
                },
                {
                    'id': 'service',
                    'name': '服务型城市',
                    'local_data': 'data/rule_data/client_service.json',
                    'system_prompt': '你是石家庄市公安局的政务助手，请根据石家庄市的政策回答问题。'
                }
            ]
        },
        'data': {
            'global_train': 'data/rule_data/global_train.json',
            'max_samples': None  # None means use all data
        },
        'logging': {
            'log_level': 'INFO',
            'log_interval': 10,
            'save_metrics': True
        }
    }


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        save_path: Path where to save the configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logging.info(f"✅ Saved configuration to {save_path}")
