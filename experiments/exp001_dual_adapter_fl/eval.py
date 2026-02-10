"""
Complete evaluation script for dual-adapter federated learning experiment.

This script evaluates the trained model on four test sets:
1. Test-G (global_test.json): Universal law knowledge retention
2. Test-A (strict_test.json): Strict city policy memory
3. Test-B (service_test.json): Service city policy memory
4. Conflict Test (conflict_cases.json): Non-IID conflict resolution

Key Features:
- Keyword-based evaluation (no external LLM judge needed)
- System prompt injection for city identity
- Comprehensive metrics for privacy and accuracy
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.base_model import load_base_model, freeze_base_model
from src.models.dual_adapter import DualAdapterModel, get_lora_config
from src.utils.config import load_config
from src.utils.logger import setup_logger
from tools.evaluators.conflict_tester import ConflictTester


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator for dual-adapter federated learning.
    
    Evaluates on four test sets with different purposes:
    - Test-G: Universal knowledge retention
    - Test-A: Strict city policy memory
    - Test-B: Service city policy memory
    - Conflict: Non-IID conflict resolution
    """
    
    def __init__(
        self,
        base_model_name: str,
        global_adapter_path: str,
        local_adapter_paths: Dict[str, str],
        config: Dict
    ):
        """
        Initialize evaluator.
        
        Args:
            base_model_name: Name of base model
            global_adapter_path: Path to global adapter
            local_adapter_paths: Dict of adapter_name -> path
            config: Configuration dictionary
        """
        self.base_model_name = base_model_name
        self.global_adapter_path = global_adapter_path
        self.local_adapter_paths = local_adapter_paths
        self.config = config
        
        # Load base model
        logging.info("Loading base model...")
        self.base_model, self.tokenizer = load_base_model(
            model_name=base_model_name,
            quantization=config.get('quantization', 'auto')
        )
        freeze_base_model(self.base_model)
        
        # System prompts for city identity (联合城市身份，激活混合数据训练的知识)
        self.system_prompts = {
            'strict': '你是上海市（户政）与北京市（交管）的联合政务助手。请依据这两个城市严格、规范的管理规定进行回答。对于违规行为，请强调处罚和红线。',
            'service': '你是石家庄市（户政）与南宁市（交管）的联合政务助手。请依据这两个城市便民、宽松、人性化的政策进行回答。对于轻微违章，请强调教育与纠正。',
            'global': '你是公安政务助手，请根据国家法律法规回答问题。'
        }
        
        logging.info("✅ Evaluator initialized")
    
    def generate_response(
        self,
        question: str,
        adapter_type: str,
        local_adapter_name: str = None,
        max_new_tokens: int = 128  # 减少生成长度：256->128
    ) -> str:
        """
        Generate response with specified adapter configuration.
        
        Args:
            question: Question to answer
            adapter_type: 'global_only', 'strict', or 'service'
            local_adapter_name: Name of local adapter (for dual-adapter mode)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        # Create dual adapter model
        lora_config = get_lora_config()
        dual_model = DualAdapterModel(self.base_model, lora_config)
        
        # Load adapters based on type
        if adapter_type == 'global_only':
            # Only global adapter (for Test-G baseline)
            dual_model.add_global_adapter(
                adapter_name="global",
                adapter_path=self.global_adapter_path
            )
            dual_model.set_active_adapters(["global"])
            system_prompt = self.system_prompts['global']
        else:
            # Global + Local adapter
            dual_model.add_global_adapter(
                adapter_name="global",
                adapter_path=self.global_adapter_path
            )
            dual_model.add_local_adapter(
                adapter_name="local",
                adapter_path=self.local_adapter_paths[adapter_type]
            )
            dual_model.set_active_adapters(["global", "local"])
            system_prompt = self.system_prompts[adapter_type]
        
        model = dual_model.get_model()
        model.eval()
        
        # Format prompt with system prompt injection
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        
        return response.strip()
    
    def evaluate_with_keywords(
        self,
        test_cases: List[Dict],
        adapter_type: str,
        test_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate on test set using keyword matching.
        
        Args:
            test_cases: List of test cases with 'instruction' and 'output'
            adapter_type: 'global_only', 'strict', or 'service'
            test_name: Name of test set for logging
            
        Returns:
            Evaluation results
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"Evaluating on {test_name} with {adapter_type} adapter")
        logging.info(f"{'='*80}")
        
        correct = 0
        total = len(test_cases)
        results = []
        
        for i, case in enumerate(tqdm(test_cases, desc=f"{test_name}")):
            question = case['instruction']
            expected_output = case['output']
            
            # Generate response
            response = self.generate_response(
                question=question,
                adapter_type=adapter_type
            )
            
            # Simple keyword matching: check if key phrases from expected output appear in response
            # Extract key policy terms (simplified approach)
            is_correct = self._check_answer_correctness(response, expected_output)
            
            if is_correct:
                correct += 1
            
            results.append({
                'question': question,
                'expected': expected_output,
                'response': response,
                'correct': is_correct
            })
        
        accuracy = correct / total if total > 0 else 0
        
        logging.info(f"\n{test_name} Results:")
        logging.info(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
        
        return {
            'test_name': test_name,
            'adapter_type': adapter_type,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'details': results
        }
    
    def _check_answer_correctness(self, response: str, expected: str) -> bool:
        """
        Check if response is correct using keyword matching.
        
        This is a simplified approach. For production, you may want to:
        - Extract key facts from expected answer
        - Check if those facts appear in response
        - Use more sophisticated NLP techniques
        
        Args:
            response: Model's response
            expected: Expected answer
            
        Returns:
            True if response contains key information from expected answer
        """
        # Extract key phrases (numbers, policy terms, etc.)
        # This is a simple heuristic - adjust based on your data
        
        # For now, use a simple overlap check
        # Extract important words (longer than 2 chars, not common words)
        common_words = {'的', '了', '是', '在', '有', '和', '与', '或', '等', '可以', '需要', '应该'}
        
        expected_words = set()
        for word in expected:
            if len(word) > 2 and word not in common_words:
                expected_words.add(word)
        
        # Check how many key words appear in response
        matches = sum(1 for word in expected_words if word in response)
        
        # Consider correct if at least 30% of key words match
        threshold = max(1, len(expected_words) * 0.3)
        return matches >= threshold
    
    def evaluate_conflict_cases(
        self,
        test_cases: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate conflict cases using keyword-based behavior classification.
        
        Args:
            test_cases: List of conflict test cases with evaluation_guide
            
        Returns:
            Conflict evaluation results
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"Evaluating Conflict Cases")
        logging.info(f"{'='*80}")
        
        # Initialize conflict tester
        tester = ConflictTester(
            base_model_name=self.base_model_name,
            global_adapter_path=self.global_adapter_path,
            config=self.config
        )
        
        # Run guided test suite
        results = tester.run_guided_test_suite(
            test_cases=test_cases,
            local_adapter_paths=self.local_adapter_paths,
            system_prompts=self.system_prompts
        )
        
        return results
    
    def run_full_evaluation(
        self,
        test_data_dir: Path,
        num_conflict_cases: int = None
    ) -> Dict[str, Any]:
        """
        Run complete evaluation on all four test sets.
        
        Args:
            test_data_dir: Directory containing test data files
            num_conflict_cases: Number of conflict cases to test (None for all)
            
        Returns:
            Complete evaluation results
        """
        all_results = {}
        
        # 1. Test-G: Universal law knowledge (all adapters should perform well)
        logging.info("\n" + "="*80)
        logging.info("TEST-G: Universal Law Knowledge Retention")
        logging.info("="*80)
        
        with open(test_data_dir / "global_test.json", 'r', encoding='utf-8') as f:
            global_test = json.load(f)
        
        all_results['test_g'] = {
            'global_only': self.evaluate_with_keywords(
                global_test, 'global_only', 'Test-G (Global Only)'
            ),
            'strict': self.evaluate_with_keywords(
                global_test, 'strict', 'Test-G (Strict Adapter)'
            ),
            'service': self.evaluate_with_keywords(
                global_test, 'service', 'Test-G (Service Adapter)'
            )
        }
        
        # 2. Test-A: Strict city policy (strict adapter should excel)
        logging.info("\n" + "="*80)
        logging.info("TEST-A: Strict City Policy Memory")
        logging.info("="*80)
        
        with open(test_data_dir / "strict_test.json", 'r', encoding='utf-8') as f:
            strict_test = json.load(f)
        
        all_results['test_a'] = {
            'strict': self.evaluate_with_keywords(
                strict_test, 'strict', 'Test-A (Strict Adapter)'
            ),
            'service': self.evaluate_with_keywords(
                strict_test, 'service', 'Test-A (Service Adapter - Cross Test)'
            )
        }
        
        # 3. Test-B: Service city policy (service adapter should excel)
        logging.info("\n" + "="*80)
        logging.info("TEST-B: Service City Policy Memory")
        logging.info("="*80)
        
        with open(test_data_dir / "service_test.json", 'r', encoding='utf-8') as f:
            service_test = json.load(f)
        
        all_results['test_b'] = {
            'service': self.evaluate_with_keywords(
                service_test, 'service', 'Test-B (Service Adapter)'
            ),
            'strict': self.evaluate_with_keywords(
                service_test, 'strict', 'Test-B (Strict Adapter - Cross Test)'
            )
        }
        
        # 4. Conflict Test: Non-IID conflict resolution
        logging.info("\n" + "="*80)
        logging.info("CONFLICT TEST: Non-IID Conflict Resolution")
        logging.info("="*80)
        
        with open(test_data_dir / "conflict_cases.json", 'r', encoding='utf-8') as f:
            conflict_cases = json.load(f)
        
        if num_conflict_cases:
            conflict_cases = conflict_cases[:num_conflict_cases]
        
        all_results['conflict'] = self.evaluate_conflict_cases(conflict_cases)
        
        return all_results


def print_summary(results: Dict[str, Any]):
    """Print evaluation summary."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Test-G: Universal Knowledge
    print("\n📚 Test-G: Universal Law Knowledge Retention")
    print("   (All adapters should perform well)")
    print(f"   Global Only:    {results['test_g']['global_only']['accuracy']:.1%}")
    print(f"   Strict Adapter: {results['test_g']['strict']['accuracy']:.1%}")
    print(f"   Service Adapter: {results['test_g']['service']['accuracy']:.1%}")
    
    # Test-A: Strict City
    print("\n🔒 Test-A: Strict City Policy Memory")
    print("   (Strict adapter should excel, Service should be low)")
    print(f"   Strict Adapter:  {results['test_a']['strict']['accuracy']:.1%} ✅")
    print(f"   Service Adapter: {results['test_a']['service']['accuracy']:.1%} (cross-test)")
    
    privacy_a = results['test_a']['strict']['accuracy'] - results['test_a']['service']['accuracy']
    print(f"   Privacy Gap: {privacy_a:+.1%} (higher is better)")
    
    # Test-B: Service City
    print("\n🤝 Test-B: Service City Policy Memory")
    print("   (Service adapter should excel, Strict should be low)")
    print(f"   Service Adapter: {results['test_b']['service']['accuracy']:.1%} ✅")
    print(f"   Strict Adapter:  {results['test_b']['strict']['accuracy']:.1%} (cross-test)")
    
    privacy_b = results['test_b']['service']['accuracy'] - results['test_b']['strict']['accuracy']
    print(f"   Privacy Gap: {privacy_b:+.1%} (higher is better)")
    
    # Conflict Test
    print("\n⚔️  Conflict Test: Non-IID Conflict Resolution")
    conflict = results['conflict']
    print(f"   Pass Rate: {conflict['pass_rate']:.1%}")
    print(f"   Passed: {conflict['passed']}/{conflict['total_cases']}")
    print(f"   Failed: {conflict['failed']}")
    print(f"   Ambiguous: {conflict['ambiguous']}")
    print(f"   No Match: {conflict['no_match']}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print(f"✅ Universal Knowledge Retained: {results['test_g']['strict']['accuracy']:.1%}")
    print(f"🔒 Privacy Isolation (A): {privacy_a:+.1%}")
    print(f"🔒 Privacy Isolation (B): {privacy_b:+.1%}")
    print(f"⚔️  Conflict Resolution: {conflict['pass_rate']:.1%}")
    print("="*80)


def main():
    """Main evaluation entry point."""
    
    # Setup logging
    log_dir = project_root / "results" / "exp001_dual_adapter_fl" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(log_dir), experiment_name="evaluation")
    
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    
    logging.info("="*80)
    logging.info("COMPREHENSIVE EVALUATION")
    logging.info("="*80)
    
    # Define paths
    results_dir = project_root / "results" / "exp001_dual_adapter_fl"
    test_data_dir = project_root / "data" / "test"
    
    adapter_paths = {
        'strict': str(results_dir / "checkpoints" / "final_adapters" / "strict" / "local"),
        'service': str(results_dir / "checkpoints" / "final_adapters" / "service" / "local")
    }
    
    global_adapter_path = str(results_dir / "checkpoints" / "final_adapters" / "global")
    
    # Check if adapters exist
    for name, path in adapter_paths.items():
        if not Path(path).exists():
            logging.error(f"Adapter not found: {path}")
            logging.error("Please ensure training completed successfully")
            return
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        base_model_name=config['model']['base_model'],
        global_adapter_path=global_adapter_path,
        local_adapter_paths=adapter_paths,
        config=config['model']
    )
    
    # Run full evaluation
    # For quick testing, limit conflict cases to 20
    # For full evaluation, set num_conflict_cases=None
    results = evaluator.run_full_evaluation(
        test_data_dir=test_data_dir,
        num_conflict_cases=20  # Change to None for full evaluation
    )
    
    # Save results
    output_dir = results_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "comprehensive_evaluation_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"\n✅ Results saved to: {output_path}")
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
