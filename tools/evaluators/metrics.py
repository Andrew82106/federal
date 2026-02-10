"""
Metrics calculation for model evaluation.
"""

import logging
from typing import List, Dict, Any


def calculate_accuracy(
    predictions: List[str],
    references: List[str],
    keyword_based: bool = True
) -> float:
    """
    Calculate accuracy.
    
    Args:
        predictions: List of predicted responses
        references: List of reference responses or keywords
        keyword_based: If True, check if reference keywords are in prediction
        
    Returns:
        Accuracy score
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    
    correct = 0
    for pred, ref in zip(predictions, references):
        if keyword_based:
            # Check if reference keywords are in prediction
            if isinstance(ref, list):
                # Multiple keywords - all must be present
                if all(kw in pred for kw in ref):
                    correct += 1
            else:
                # Single keyword
                if ref in pred:
                    correct += 1
        else:
            # Exact match
            if pred.strip() == ref.strip():
                correct += 1
    
    accuracy = correct / len(predictions)
    logging.info(f"Accuracy: {correct}/{len(predictions)} = {accuracy:.2%}")
    
    return accuracy


def calculate_conflict_resolution_rate(
    test_results: List[Dict]
) -> float:
    """
    Calculate conflict resolution rate from test results.
    
    Args:
        test_results: List of conflict test results
        
    Returns:
        Conflict resolution rate
    """
    if not test_results:
        return 0.0
    
    resolved = sum(1 for result in test_results if result.get('success', False))
    rate = resolved / len(test_results)
    
    logging.info(f"Conflict Resolution Rate: {resolved}/{len(test_results)} = {rate:.2%}")
    
    return rate


def calculate_keyword_match_rate(
    responses: List[str],
    expected_keywords: List[List[str]]
) -> Dict[str, float]:
    """
    Calculate keyword match statistics.
    
    Args:
        responses: List of responses
        expected_keywords: List of expected keyword lists
        
    Returns:
        Dictionary with match statistics
    """
    total_keywords = 0
    matched_keywords = 0
    
    for response, keywords in zip(responses, expected_keywords):
        total_keywords += len(keywords)
        matched_keywords += sum(1 for kw in keywords if kw in response)
    
    match_rate = matched_keywords / total_keywords if total_keywords > 0 else 0.0
    
    return {
        'total_keywords': total_keywords,
        'matched_keywords': matched_keywords,
        'match_rate': match_rate
    }
