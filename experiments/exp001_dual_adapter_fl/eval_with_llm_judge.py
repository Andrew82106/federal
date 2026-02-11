"""
Evaluation script with LLM-as-Judge for answer correctness.

Uses Qwen model to judge if responses correctly answer questions,
providing more accurate evaluation than keyword matching.

Usage:
    python eval_with_llm_judge.py --batch_size 16
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
from tools.evaluators.llm_judge import LLMJudge


class LLMJudgeEvaluator:
    """Evaluator using LLM-as-Judge for answer correctness."""
    
    def __init__(
        self,
        base_model_name: str,
        global_adapter_path: str,
        local_adapter_paths: Dict[str, str],
        config: Dict,
        checkpoint_dir: Path
    ):
        self.base_model_name = base_model_name
        self.global_adapter_path = global_adapter_path
        self.local_adapter_paths = local_adapter_paths
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base model for inference
        logging.info("Loading base model for inference...")
        self.base_model, self.tokenizer = load_base_model(
            model_name=base_model_name,
            quantization=config.get('quantization', 'auto')
        )
        freeze_base_model(self.base_model)
        
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize LLM Judge
        logging.info("Loading LLM Judge...")
        self.judge = LLMJudge(base_model_name)
        
        # System prompts
        self.system_prompts = {
            'strict': '你是上海市（户政）与北京市（交管）的联合政务助手。请依据这两个城市严格、规范的管理规定进行回答。对于违规行为，请强调处罚和红线。',
            'service': '你是石家庄市（户政）与南宁市（交管）的联合政务助手。请依据这两个城市便民、宽松、人性化的政策进行回答。对于轻微违章，请强调教育与纠正。',
            'global': '你是公安政务助手，请根据国家法律法规回答问题。'
        }
        
        logging.info("✅ Evaluator initialized")
    
    def load_checkpoint(self, test_name: str, adapter_type: str) -> Dict:
        """Load checkpoint if exists."""
        checkpoint_file = self.checkpoint_dir / f"{test_name}_{adapter_type}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def save_checkpoint(self, test_name: str, adapter_type: str, results: Dict):
        """Save checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{test_name}_{adapter_type}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info(f"💾 Checkpoint saved: {checkpoint_file}")
    
    def evaluate_test_set(
        self,
        test_name: str,
        test_cases: List[Dict],
        adapter_type: str,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """Evaluate on test set with LLM Judge."""
        
        # Check for existing checkpoint
        checkpoint = self.load_checkpoint(test_name, adapter_type)
        if checkpoint:
            logging.info(f"📂 Found checkpoint for {test_name}_{adapter_type}")
            logging.info(f"   Completed: {checkpoint['correct']}/{checkpoint['total']}")
            return checkpoint
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Evaluating {test_name} with {adapter_type}")
        logging.info(f"{'='*80}")
        
        # Step 1: Generate responses
        logging.info("Step 1/2: Generating responses...")
        responses = self._generate_responses(test_cases, adapter_type, batch_size)
        
        # Step 2: Judge with LLM
        logging.info("Step 2/2: Judging with LLM...")
        questions = [case['instruction'] for case in test_cases]
        expected_answers = [case['output'] for case in test_cases]
        
        judgments = self.judge.judge_batch(questions, expected_answers, responses, batch_size=batch_size)
        
        # Compile results
        correct = 0
        total = len(test_cases)
        results = []
        
        for case, response, (is_correct, explanation) in zip(test_cases, responses, judgments):
            if is_correct:
                correct += 1
            
            results.append({
                'question': case['instruction'],
                'expected': case['output'],
                'response': response,
                'correct': is_correct,
                'judge_explanation': explanation
            })
        
        accuracy = correct / total if total > 0 else 0
        
        result = {
            'test_name': test_name,
            'adapter_type': adapter_type,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'details': results
        }
        
        self.save_checkpoint(test_name, adapter_type, result)
        
        logging.info(f"✅ {test_name}-{adapter_type}: {accuracy:.2%} ({correct}/{total})")
        
        return result
    
    def _generate_responses(
        self,
        test_cases: List[Dict],
        adapter_type: str,
        batch_size: int
    ) -> List[str]:
        """Generate responses for test cases."""
        
        # Load adapter
        lora_config = get_lora_config()
        dual_model = DualAdapterModel(self.base_model, lora_config)
        
        if adapter_type == 'global_only':
            dual_model.add_global_adapter(
                adapter_name="global",
                adapter_path=self.global_adapter_path
            )
            dual_model.set_active_adapters(["global"])
            system_prompt = self.system_prompts['global']
        else:
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
        
        responses = []
        total = len(test_cases)
        
        for i in tqdm(range(0, total, batch_size), desc=f"Generating-{adapter_type}"):
            batch = test_cases[i:i+batch_size]
            
            prompts = []
            for case in batch:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": case['instruction']}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(prompt)
            
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            batch_responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            
            for response in batch_responses:
                if "<|im_start|>assistant\n" in response:
                    response = response.split("<|im_start|>assistant\n")[-1]
                elif "<|im_start|>assistant" in response:
                    response = response.split("<|im_start|>assistant")[-1]
                
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0]
                
                response = response.replace("<|endoftext|>", "").strip()
                responses.append(response)
        
        return responses
    
    def evaluate_conflict_cases(
        self,
        test_cases: List[Dict],
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """Evaluate conflict cases (uses keyword matching, not LLM Judge)."""
        
        checkpoint_file = self.checkpoint_dir / "conflict_test.json"
        if checkpoint_file.exists():
            logging.info(f"📂 Found checkpoint for conflict test")
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Evaluating Conflict Cases")
        logging.info(f"{'='*80}")
        
        tester = ConflictTester(
            base_model_name=self.base_model_name,
            global_adapter_path=self.global_adapter_path,
            config=self.config
        )
        
        results = tester.run_guided_test_suite(
            test_cases=test_cases,
            local_adapter_paths=self.local_adapter_paths,
            system_prompts=self.system_prompts,
            batch_size=batch_size
        )
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logging.info(f"💾 Checkpoint saved: {checkpoint_file}")
        
        return results


def print_summary(results: Dict[str, Any]):
    """Print evaluation summary."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY (LLM Judge)")
    print("="*80)
    
    print("\n📚 Test-G: Universal Law Knowledge Retention")
    print(f"   Global Only:    {results['test_g']['global_only']['accuracy']:.1%}")
    print(f"   Strict Adapter: {results['test_g']['strict']['accuracy']:.1%}")
    print(f"   Service Adapter: {results['test_g']['service']['accuracy']:.1%}")
    
    print("\n🔒 Test-A: Strict City Policy Memory")
    print(f"   Strict Adapter:  {results['test_a']['strict']['accuracy']:.1%} ✅")
    print(f"   Service Adapter: {results['test_a']['service']['accuracy']:.1%}")
    
    privacy_a = results['test_a']['strict']['accuracy'] - results['test_a']['service']['accuracy']
    print(f"   Privacy Gap: {privacy_a:+.1%}")
    
    print("\n🤝 Test-B: Service City Policy Memory")
    print(f"   Service Adapter: {results['test_b']['service']['accuracy']:.1%} ✅")
    print(f"   Strict Adapter:  {results['test_b']['strict']['accuracy']:.1%}")
    
    privacy_b = results['test_b']['service']['accuracy'] - results['test_b']['strict']['accuracy']
    print(f"   Privacy Gap: {privacy_b:+.1%}")
    
    print("\n⚔️  Conflict Test")
    conflict = results['conflict']
    print(f"   Pass Rate: {conflict['pass_rate']:.1%}")
    print(f"   Passed: {conflict['passed']}/{conflict['total_cases']}")
    
    print("\n" + "="*80)


def main():
    """Main evaluation entry point."""
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    args = parser.parse_args()
    
    # Setup logging
    log_dir = project_root / "results" / "exp001_dual_adapter_fl" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(log_dir), experiment_name="eval_llm_judge")
    
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    
    logging.info("="*80)
    logging.info("EVALUATION WITH LLM JUDGE - EXP001")
    logging.info("="*80)
    
    # Define paths
    results_dir = project_root / "results" / "exp001_dual_adapter_fl"
    test_data_dir = project_root / "data" / "test"
    checkpoint_dir = results_dir / "eval_checkpoints"
    
    adapter_paths = {
        'strict': str(results_dir / "checkpoints" / "final_adapters" / "strict" / "local"),
        'service': str(results_dir / "checkpoints" / "final_adapters" / "service" / "local")
    }
    
    global_adapter_path = str(results_dir / "checkpoints" / "final_adapters" / "global")
    
    # Initialize evaluator
    evaluator = LLMJudgeEvaluator(
        base_model_name=config['model']['base_model'],
        global_adapter_path=global_adapter_path,
        local_adapter_paths=adapter_paths,
        config=config['model'],
        checkpoint_dir=checkpoint_dir
    )
    
    all_results = {}
    
    # Load test data
    with open(test_data_dir / "global_test.json", 'r', encoding='utf-8') as f:
        global_test = json.load(f)
    with open(test_data_dir / "strict_test.json", 'r', encoding='utf-8') as f:
        strict_test = json.load(f)
    with open(test_data_dir / "service_test.json", 'r', encoding='utf-8') as f:
        service_test = json.load(f)
    with open(test_data_dir / "conflict_cases.json", 'r', encoding='utf-8') as f:
        conflict_cases = json.load(f)
    
    # Test-G
    all_results['test_g'] = {
        'global_only': evaluator.evaluate_test_set('test_g', global_test, 'global_only', batch_size=args.batch_size),
        'strict': evaluator.evaluate_test_set('test_g', global_test, 'strict', batch_size=args.batch_size),
        'service': evaluator.evaluate_test_set('test_g', global_test, 'service', batch_size=args.batch_size)
    }
    
    # Test-A
    all_results['test_a'] = {
        'strict': evaluator.evaluate_test_set('test_a', strict_test, 'strict', batch_size=args.batch_size),
        'service': evaluator.evaluate_test_set('test_a', strict_test, 'service', batch_size=args.batch_size)
    }
    
    # Test-B
    all_results['test_b'] = {
        'service': evaluator.evaluate_test_set('test_b', service_test, 'service', batch_size=args.batch_size),
        'strict': evaluator.evaluate_test_set('test_b', service_test, 'strict', batch_size=args.batch_size)
    }
    
    # Conflict Test
    all_results['conflict'] = evaluator.evaluate_conflict_cases(conflict_cases, batch_size=args.batch_size)
    
    # Save final results
    output_dir = results_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "llm_judge_evaluation_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"\n✅ Final results saved to: {output_path}")
    
    print_summary(all_results)


if __name__ == "__main__":
    main()
