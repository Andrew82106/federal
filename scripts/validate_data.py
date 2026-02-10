#!/usr/bin/env python3
"""
Data Validation Script

This script validates that all required training and test data files exist
and are in the correct format for the Dual-Adapter Federated Learning project.

Usage:
    python scripts/validate_data.py [--verbose]
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def log_info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")


def log_success(msg: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")


def log_warning(msg: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")


def log_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


def validate_json_file(file_path: Path) -> Tuple[bool, str, int]:
    """
    Validate a JSON file exists and is properly formatted.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Tuple of (is_valid, error_message, num_samples)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}", 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return False, "Data must be a list of objects", 0
        
        if len(data) == 0:
            return False, "Data list is empty", 0
        
        return True, "", len(data)
    
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {e}", 0
    except Exception as e:
        return False, f"Error reading file: {e}", 0


def validate_alpaca_format(file_path: Path, verbose: bool = False) -> Tuple[bool, str]:
    """
    Validate that a JSON file follows the Alpaca format.
    
    Args:
        file_path: Path to the JSON file
        verbose: Whether to print detailed validation info
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        required_fields = ['instruction', 'input', 'output']
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                return False, f"Item {i} is not a dictionary"
            
            for field in required_fields:
                if field not in item:
                    return False, f"Item {i} missing required field: {field}"
            
            # Check that fields are strings
            for field in required_fields:
                if not isinstance(item[field], str):
                    return False, f"Item {i} field '{field}' must be a string"
            
            # Warn if instruction or output is empty
            if not item['instruction'].strip():
                log_warning(f"{file_path.name}: Item {i} has empty instruction")
            
            if not item['output'].strip():
                log_warning(f"{file_path.name}: Item {i} has empty output")
        
        if verbose:
            log_info(f"  Sample item from {file_path.name}:")
            print(f"    instruction: {data[0]['instruction'][:80]}...")
            print(f"    input: {data[0]['input'][:80] if data[0]['input'] else '(empty)'}...")
            print(f"    output: {data[0]['output'][:80]}...")
        
        return True, ""
    
    except Exception as e:
        return False, f"Error validating format: {e}"


def validate_training_data(verbose: bool = False) -> bool:
    """
    Validate all training data files.
    
    Args:
        verbose: Whether to print detailed validation info
        
    Returns:
        True if all training data is valid
    """
    log_info("=== Validating Training Data ===")
    
    training_files = {
        'Global Training Data': Path('data/rule_data/global_train.json'),
        'Strict Client Data': Path('data/rule_data/client_strict.json'),
        'Service Client Data': Path('data/rule_data/client_service.json'),
    }
    
    all_valid = True
    
    for name, file_path in training_files.items():
        log_info(f"Checking {name}...")
        
        # Check file exists and is valid JSON
        is_valid, error_msg, num_samples = validate_json_file(file_path)
        
        if not is_valid:
            log_error(f"  {error_msg}")
            all_valid = False
            continue
        
        # Check Alpaca format
        is_valid, error_msg = validate_alpaca_format(file_path, verbose)
        
        if not is_valid:
            log_error(f"  {error_msg}")
            all_valid = False
            continue
        
        log_success(f"  ✓ {name}: {num_samples} samples")
    
    print()
    return all_valid


def validate_test_data(verbose: bool = False) -> bool:
    """
    Validate all test data files.
    
    Args:
        verbose: Whether to print detailed validation info
        
    Returns:
        True if all test data is valid (or missing with warning)
    """
    log_info("=== Validating Test Data ===")
    
    test_files = {
        'Test-G (Global Laws)': Path('data/test/global_test.json'),
        'Test-A (Strict Policies)': Path('data/test/strict_test.json'),
        'Test-B (Service Policies)': Path('data/test/service_test.json'),
        'Conflict Test Cases': Path('data/test/conflict_cases.json'),
    }
    
    all_valid = True
    missing_count = 0
    
    for name, file_path in test_files.items():
        log_info(f"Checking {name}...")
        
        # Check file exists and is valid JSON
        is_valid, error_msg, num_samples = validate_json_file(file_path)
        
        if not is_valid:
            if "not found" in error_msg:
                log_warning(f"  {error_msg}")
                log_warning(f"  Test data is optional but recommended for evaluation")
                missing_count += 1
            else:
                log_error(f"  {error_msg}")
                all_valid = False
            continue
        
        # For conflict cases, format is different
        if 'conflict' in file_path.name:
            log_success(f"  ✓ {name}: {num_samples} test cases")
            if verbose:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if len(data) > 0:
                        log_info(f"  Sample conflict case:")
                        print(f"    question: {data[0].get('question', 'N/A')[:80]}...")
                except:
                    pass
        else:
            # Check Alpaca format for test sets
            is_valid, error_msg = validate_alpaca_format(file_path, verbose)
            
            if not is_valid:
                log_error(f"  {error_msg}")
                all_valid = False
                continue
            
            log_success(f"  ✓ {name}: {num_samples} samples")
    
    if missing_count > 0:
        log_warning(f"{missing_count} test file(s) missing. You can still train, but evaluation will be limited.")
    
    print()
    return all_valid


def validate_directory_structure() -> bool:
    """
    Validate that the required directory structure exists.
    
    Returns:
        True if directory structure is valid
    """
    log_info("=== Validating Directory Structure ===")
    
    required_dirs = [
        'data',
        'data/rule_data',
        'src',
        'src/models',
        'src/federated',
        'src/data',
        'src/utils',
        'tools',
        'tools/runners',
        'tools/evaluators',
        'experiments',
        'experiments/exp001_dual_adapter_fl',
    ]
    
    all_valid = True
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            log_error(f"  Missing directory: {dir_path}")
            all_valid = False
        elif not path.is_dir():
            log_error(f"  Not a directory: {dir_path}")
            all_valid = False
    
    if all_valid:
        log_success("  ✓ All required directories exist")
    
    print()
    return all_valid


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate data files for Dual-Adapter Federated Learning'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed validation information'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Data Validation for Dual-Adapter Federated Learning")
    print("=" * 80)
    print()
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    log_info(f"Project root: {project_root}")
    print()
    
    # Validate directory structure
    dir_valid = validate_directory_structure()
    
    # Validate training data
    train_valid = validate_training_data(args.verbose)
    
    # Validate test data
    test_valid = validate_test_data(args.verbose)
    
    # Summary
    print("=" * 80)
    if dir_valid and train_valid:
        log_success("=== Validation Complete ===")
        print()
        log_success("✓ All required training data is valid")
        
        if test_valid:
            log_success("✓ All test data is valid")
        else:
            log_warning("⚠ Some test data is missing or invalid")
            log_warning("  You can still train, but evaluation will be limited")
        
        print()
        print("You can now run the experiment:")
        print("  python experiments/exp001_dual_adapter_fl/train.py")
        print()
        return 0
    else:
        log_error("=== Validation Failed ===")
        print()
        
        if not dir_valid:
            log_error("✗ Directory structure is invalid")
        
        if not train_valid:
            log_error("✗ Training data is invalid or missing")
        
        print()
        print("Please fix the errors above before running experiments.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
