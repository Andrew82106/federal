"""
Quick test script for the enhanced conflict evaluation with keyword matching.

This script demonstrates the fast, deterministic evaluation method that doesn't
require expensive LLM judges.
"""

import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_keyword_matching():
    """Test the keyword matching logic without running the full model."""
    
    # Load test cases
    test_data_path = Path("data/test/conflict_cases.json")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    logging.info(f"Loaded {len(test_cases)} test cases")
    
    # Sample test case
    sample_case = test_cases[0]
    logging.info(f"\nSample test case:")
    logging.info(f"Question: {sample_case['instruction']}")
    logging.info(f"Strict keywords: {sample_case['evaluation_guide']['strict_keywords']}")
    logging.info(f"Service keywords: {sample_case['evaluation_guide']['service_keywords']}")
    
    # Simulate responses
    strict_response = "根据上海市居住证积分管理办法，您需要满足积分要求。大专学历无社保不达标，无法落户。建议您先办理居住证并缴纳社保。"
    service_response = "根据石家庄市全面放开落户限制的政策，您可以凭身份证直接办理落户，零门槛即时办理，无需社保和学历要求。"
    
    logging.info(f"\n--- Simulated Strict City Response ---")
    logging.info(f"Response: {strict_response}")
    behavior = evaluate_behavior(strict_response, sample_case['evaluation_guide'])
    logging.info(f"Detected behavior: {behavior}")
    logging.info(f"Expected: STRICT_BEHAVIOR")
    logging.info(f"Correct: {'✅' if behavior == 'STRICT_BEHAVIOR' else '❌'}")
    
    logging.info(f"\n--- Simulated Service City Response ---")
    logging.info(f"Response: {service_response}")
    behavior = evaluate_behavior(service_response, sample_case['evaluation_guide'])
    logging.info(f"Detected behavior: {behavior}")
    logging.info(f"Expected: SERVICE_BEHAVIOR")
    logging.info(f"Correct: {'✅' if behavior == 'SERVICE_BEHAVIOR' else '❌'}")
    
    # Test ambiguous case
    ambiguous_response = "根据政策，您可以申请落户，但需要满足积分要求。同时我们也提供零门槛的服务。"
    logging.info(f"\n--- Ambiguous Response ---")
    logging.info(f"Response: {ambiguous_response}")
    behavior = evaluate_behavior(ambiguous_response, sample_case['evaluation_guide'])
    logging.info(f"Detected behavior: {behavior}")
    logging.info(f"Expected: AMBIGUOUS (contains keywords from both cities)")
    
    # Test no match case
    no_match_response = "请您提供更多信息，我需要了解您的具体情况才能给出准确答复。"
    logging.info(f"\n--- No Match Response ---")
    logging.info(f"Response: {no_match_response}")
    behavior = evaluate_behavior(no_match_response, sample_case['evaluation_guide'])
    logging.info(f"Detected behavior: {behavior}")
    logging.info(f"Expected: NO_MATCH (no expected keywords found)")

def evaluate_behavior(response: str, evaluation_guide: dict) -> str:
    """
    Evaluate which city behavior the response exhibits.
    
    This is the same logic as in ConflictTester.evaluate_conflict_behavior()
    """
    strict_keywords = evaluation_guide.get('strict_keywords', [])
    service_keywords = evaluation_guide.get('service_keywords', [])
    
    # Count keyword matches
    strict_hits = sum(1 for k in strict_keywords if k in response)
    service_hits = sum(1 for k in service_keywords if k in response)
    
    # Determine behavior
    if strict_hits > 0 and service_hits == 0:
        return "STRICT_BEHAVIOR"
    elif service_hits > 0 and strict_hits == 0:
        return "SERVICE_BEHAVIOR"
    elif strict_hits > 0 and service_hits > 0:
        return "AMBIGUOUS"
    else:
        return "NO_MATCH"

if __name__ == "__main__":
    test_keyword_matching()
