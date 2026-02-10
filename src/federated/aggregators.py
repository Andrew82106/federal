"""
Aggregation algorithms for federated learning.

This module implements various aggregation methods for combining
client model updates, including FedAvg.
"""

import os
import logging
from typing import List, Dict, Optional
import torch


def fedavg(
    adapter_paths: List[str],
    weights: Optional[List[float]] = None
) -> Dict[str, torch.Tensor]:
    """
    FedAvg (Federated Averaging) aggregation algorithm.
    
    Computes weighted average of adapter parameters from multiple clients.
    
    Args:
        adapter_paths: List of paths to adapter directories
        weights: Optional list of client weights (default: equal weights)
        
    Returns:
        Aggregated state dictionary
        
    Raises:
        ValueError: If adapter_paths is empty or weights don't match
    """
    if not adapter_paths:
        raise ValueError("adapter_paths cannot be empty")
    
    # Default to equal weights
    if weights is None:
        weights = [1.0 / len(adapter_paths)] * len(adapter_paths)
    
    if len(weights) != len(adapter_paths):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match "
            f"number of adapters ({len(adapter_paths)})"
        )
    
    # Normalize weights to sum to 1
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    
    logging.info(f"FedAvg: Aggregating {len(adapter_paths)} adapters")
    logging.info(f"Weights: {[f'{w:.4f}' for w in weights]}")
    
    # Load all state dicts
    state_dicts = []
    for path in adapter_paths:
        # Try safetensors first (newer format), then fall back to bin
        adapter_file_safetensors = os.path.join(path, "adapter_model.safetensors")
        adapter_file_bin = os.path.join(path, "adapter_model.bin")
        
        if os.path.exists(adapter_file_safetensors):
            # Load safetensors format
            from safetensors.torch import load_file
            state_dict = load_file(adapter_file_safetensors)
            logging.debug(f"Loaded adapter from {adapter_file_safetensors}")
        elif os.path.exists(adapter_file_bin):
            # Load bin format
            state_dict = torch.load(adapter_file_bin, map_location='cpu')
            logging.debug(f"Loaded adapter from {adapter_file_bin}")
        else:
            raise FileNotFoundError(
                f"Adapter file not found in {path}. "
                f"Tried: adapter_model.safetensors, adapter_model.bin"
            )
        
        state_dicts.append(state_dict)
    
    # Perform weighted average
    aggregated = weighted_average(state_dicts, weights)
    
    logging.info(f"✅ FedAvg aggregation completed")
    
    return aggregated


def weighted_average(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float]
) -> Dict[str, torch.Tensor]:
    """
    Compute weighted average of state dictionaries.
    
    Args:
        state_dicts: List of state dictionaries
        weights: List of weights (must sum to 1)
        
    Returns:
        Weighted average state dictionary
    """
    if not state_dicts:
        raise ValueError("state_dicts cannot be empty")
    
    # Initialize aggregated state dict with zeros
    aggregated = {}
    
    # Get keys from first state dict
    keys = state_dicts[0].keys()
    
    # Verify all state dicts have same keys
    for i, sd in enumerate(state_dicts[1:], 1):
        if set(sd.keys()) != set(keys):
            raise ValueError(f"State dict {i} has different keys than state dict 0")
    
    # Compute weighted average for each parameter
    for key in keys:
        # Stack tensors from all clients
        tensors = [sd[key] for sd in state_dicts]
        
        # Compute weighted sum
        weighted_sum = sum(w * t for w, t in zip(weights, tensors))
        
        aggregated[key] = weighted_sum
    
    return aggregated


def get_parameter_count(state_dict: Dict[str, torch.Tensor]) -> int:
    """
    Count total number of parameters in state dict.
    
    Args:
        state_dict: State dictionary
        
    Returns:
        Total parameter count
    """
    return sum(p.numel() for p in state_dict.values())


def compute_parameter_difference(
    state_dict1: Dict[str, torch.Tensor],
    state_dict2: Dict[str, torch.Tensor]
) -> float:
    """
    Compute L2 norm of difference between two state dicts.
    
    Args:
        state_dict1: First state dictionary
        state_dict2: Second state dictionary
        
    Returns:
        L2 norm of difference
    """
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        raise ValueError("State dicts must have same keys")
    
    diff_norm = 0.0
    
    for key in state_dict1.keys():
        diff = state_dict1[key] - state_dict2[key]
        diff_norm += torch.sum(diff ** 2).item()
    
    return diff_norm ** 0.5
