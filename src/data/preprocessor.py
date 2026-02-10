"""
Data preprocessing utilities for Alpaca-format datasets.

This module provides functions for loading, merging, and validating
training data in Alpaca format.
"""

import json
import logging
from typing import List, Dict, Any


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of data samples
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"✅ Loaded {len(data)} samples from {file_path}")
        return data
        
    except FileNotFoundError:
        logging.error(f"❌ File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"❌ Invalid JSON in {file_path}: {e}")
        raise


def merge_datasets(
    global_data: List[Dict[str, Any]],
    local_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge global and local datasets.
    
    Args:
        global_data: Global training data
        local_data: Local/client-specific data
        
    Returns:
        Combined dataset
    """
    merged = global_data + local_data
    logging.info(
        f"✅ Merged datasets: {len(global_data)} global + "
        f"{len(local_data)} local = {len(merged)} total"
    )
    return merged


def validate_data_format(data: List[Dict[str, Any]]) -> bool:
    """
    Validate data format conforms to Alpaca specification.
    
    Expected format:
    {
        "instruction": str,
        "input": str (can be empty),
        "output": str
    }
    
    Args:
        data: List of data samples to validate
        
    Returns:
        True if all samples are valid
        
    Raises:
        ValueError: If data format is invalid
    """
    required_fields = ['instruction', 'output']
    optional_fields = ['input']
    
    for idx, sample in enumerate(data):
        # Check required fields
        for field in required_fields:
            if field not in sample:
                raise ValueError(
                    f"Sample {idx} missing required field '{field}': {sample}"
                )
            if not isinstance(sample[field], str):
                raise ValueError(
                    f"Sample {idx} field '{field}' must be string, got {type(sample[field])}"
                )
        
        # Check optional fields if present
        if 'input' in sample and not isinstance(sample['input'], str):
            raise ValueError(
                f"Sample {idx} field 'input' must be string, got {type(sample['input'])}"
            )
    
    logging.info(f"✅ Validated {len(data)} samples - all conform to Alpaca format")
    return True


def filter_by_length(
    data: List[Dict[str, Any]],
    max_length: int,
    tokenizer
) -> List[Dict[str, Any]]:
    """
    Filter out samples that exceed maximum token length.
    
    Args:
        data: List of data samples
        max_length: Maximum token length
        tokenizer: Tokenizer for length calculation
        
    Returns:
        Filtered dataset
    """
    filtered = []
    removed = 0
    
    for sample in data:
        # Combine all text
        text = sample['instruction'] + sample.get('input', '') + sample['output']
        tokens = tokenizer.encode(text)
        
        if len(tokens) <= max_length:
            filtered.append(sample)
        else:
            removed += 1
    
    if removed > 0:
        logging.warning(
            f"⚠️ Filtered out {removed} samples exceeding max_length={max_length}. "
            f"Kept {len(filtered)} samples."
        )
    
    return filtered


def get_dataset_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate dataset statistics.
    
    Args:
        data: List of data samples
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_samples': len(data),
        'avg_instruction_length': 0,
        'avg_output_length': 0,
        'samples_with_input': 0
    }
    
    if len(data) == 0:
        return stats
    
    total_inst_len = 0
    total_out_len = 0
    
    for sample in data:
        total_inst_len += len(sample['instruction'])
        total_out_len += len(sample['output'])
        if sample.get('input', ''):
            stats['samples_with_input'] += 1
    
    stats['avg_instruction_length'] = total_inst_len / len(data)
    stats['avg_output_length'] = total_out_len / len(data)
    
    return stats
