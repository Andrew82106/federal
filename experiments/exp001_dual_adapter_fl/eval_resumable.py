"""
Resumable evaluation script with checkpoint support.

This script saves progress after each test set, allowing you to resume
if the evaluation is interrupted.
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


class ResumableEvaluator:
    """Evaluator with checkpoint support for resuming interrupted evaluations."""
    
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
        
        # Load base model
        logging.info("Loading base model...")
        self.base_model, self.tokenizer = load_base_model(
            model_name=base_model_name,
            quantization=config.get('quantization', 'auto')
        )
        freeze_base_model(self.base_model)
        
        # System prompts
        self.system_prompts = {
            'strict': '你是上海市公安局的政务助手，请根据上海市的政策回答问题。',
            'service': '你是石家庄市公安局的政务助手，请根据石家庄市的政策回答问题。',
            'global': '你是公安政务助手，请根据国家法律法规回答问题。'
        }
        
        logging.info("✅ Evaluator initialized")
    
    def _get_checkpoint_path(self, test_name: str, adapter_type: str) -> Path:
        """Get checkpoint file path for a specific test."""
        return self.checkpoint_dir / f"{test_name}_{adapter_type}.json"
    
    def _load_checkpoint(self, test_name: str, adapter_type: str) -> Dict:
        """Load checkpoint if exists."""
        checkpoint_path = self._get_checkpoint_path(test_name, adapter_type)
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _save_checkpoint(self, test_name: str, adapter_type: str, results: Dict):
        """Save checkpoint."""
        checkpoint_path = self._get_checkpoint_path(test_name, adapter_type)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def generate_response(
        self,
        question: str,
        adapter_type: str,
        max_new_tokens: int = 256
    ) -> str:
        """Generate response with specified adapter."""
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
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        
        return response.strip()
    
    def _check_answer_correctness(self, response: str, expected: str) -> bool:
        """Simple keyword matching for correctness."""
        common_words = {'的', '了', '是', '在', '有', '和', '与', '或', '等', '可以', '需要', '应该'}
        
        expected_words = set()
        for word in expected:
            if len(word) > 2 and word not in common_words:
                expected_words.add(word)
        
        matches = sum(1 for word in expected_words if word in response)
        threshold = max(1, len(expected_words) * 0.3)
        return matches >= threshold
    
    def evaluate_test_set(
        self,
        test_cases: List[Dict],
        adapter_type: str,
        test_name: str
    ) -> Dict[str, Any]:
        """Evaluate on test set with checkpoint support."""
        
        # Check for existing checkpoint
        checkpoint = self._load_checkpoint(test_name, adapter_type)
        if checkpoint:
            logging.info(f"📂 Found checkpoint for {test_name}_{adapter_type}")
            logging.info(f"   Completed: {checkpoint['correct']}/{checkpoint['total']}")
            
            resume = input(f"Resume from checkpoint? (y/n): ").lower()
            if resume == 'y':
                return checkpoint
            else:
                logging.info("Starting fresh evaluation...")
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Evaluating {test_name} with {adapter_type} adapter")
        logging.info(f"{'='*80}")
        
        correct = 0
        total = len(test_cases)
        results = []
        
        for i, case in enumerate(tqdm(test_cases, desc=f"{test_name} ({adapter_type})")):
            question = case['instruction']
            expected_output = case['output']
            
            try:
                response = self.generate_response(
                    question=question,
                    adapter_type=adapter_type
                )
                
                is_correct = self._check_answer_correctness(response, expected_output)
                
                if is_correct:
                    correct += 1
                
                results.append({
                    'question': question,
                    'expected': expected_output,
                    'response': response,
                    'correct': is_correct
                })
                
                # Save checkpoint every 10 cases
                if (i + 1) % 10 == 0:
                    temp_result = {
                        'test_name': test_name,
                        'adapter_type': adapter_type,
                        'accuracy': correct / (i + 1),
                        'correct': correct,
                        'total': i + 1,
                        'details': results
                    }
                    self._save_checkpoint(test_name, adapter_type, temp_result)
                
            except Exception as e:
                logging.error(f"Error on case {i}: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0
        
        final_result = {
            'test_name': test_name,
            'adapter_type': adapter_type,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'details': results
        }
        
        # Save final checkpoint
        self._save_checkpoint(test_name, adapter_type, final_result)
        
        logging.info(f"\n{test_name} ({adapter_type}) Results:")
        logging.info(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
        
        return final_result


def main():
    """Main evaluation entry point."""
    
    # Setup logging
    log_dir = project_root / "results" / "exp001_dual_adapter_fl" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(log_dir), experiment_name="eval_resumable")
    
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    
    logging.info("="*80)
    logging.info("RESUMABLE EVALUATION")
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
    evaluator = ResumableEvaluator(
        base_model_name=config['model']['base_model'],
        global_adapter_path=global_adapter_path,
        local_adapter_paths=adapter_paths,
        config=config['model'],
        checkpoint_dir=checkpoint_dir
    )
    
    all_results = {}
    
    # 1. Test-G
    logging.info("\n" + "="*80)
    logging.info("TEST-G: Universal Law Knowledge")
    logging.info("="*80)
    
    with open(test_data_dir / "global_test.json", 'r', encoding='utf-8') as f:
        global_test = json.load(f)
    
    all_results['test_g'] = {
        'global_only': evaluator.evaluate_test_set(
            global_test, 'global_only', 'test_g_global'
        ),
        'strict': evaluator.evaluate_test_set(
            global_test, 'strict', 'test_g_strict'
        ),
        'service': evaluator.evaluate_test_set(
            global_test, 'service', 'test_g_service'
        )
    }
    
    # 2. Test-A
    logging.info("\n" + "="*80)
    logging.info("TEST-A: Strict City Policy")
    logging.info("="*80)
    
    with open(test_data_dir / "strict_test.json", 'r', encoding='utf-8') as f:
        strict_test = json.load(f)
    
    all_results['test_a'] = {
        'strict': evaluator.evaluate_test_set(
            strict_test, 'strict', 'test_a_strict'
        ),
        'service': evaluator.evaluate_test_set(
            strict_test, 'service', 'test_a_service'
        )
    }
    
    # 3. Test-B
    logging.info("\n" + "="*80)
    logging.info("TEST-B: Service City Policy")
    logging.info("="*80)
    
    with open(test_data_dir / "service_test.json", 'r', encoding='utf-8') as f:
        service_test = json.load(f)
    
    all_results['test_b'] = {
        'service': evaluator.evaluate_test_set(
            service_test, 'service', 'test_b_service'
        ),
        'strict': evaluator.evaluate_test_set(
            service_test, 'strict', 'test_b_strict'
        )
    }
    
    # 4. Conflict Test
    logging.info("\n" + "="*80)
    logging.info("CONFLICT TEST")
    logging.info("="*80)
    
    with open(test_data_dir / "conflict_cases.json", 'r', encoding='utf-8') as f:
        conflict_cases = json.load(f)
    
    tester = ConflictTester(
        base_model_name=config['model']['base_model'],
        global_adapter_path=global_adapter_path,
        config=config['model']
    )
    
    all_results['conflict'] = tester.run_guided_test_suite(
        test_cases=conflict_cases,
        local_adapter_paths=adapter_paths,
        system_prompts=evaluator.system_prompts
    )
    
    # Save final results
    output_dir = results_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "resumable_evaluation_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"\n✅ Final results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\n📚 Test-G: {all_results['test_g']['strict']['accuracy']:.1%}")
    print(f"🔒 Test-A: Strict {all_results['test_a']['strict']['accuracy']:.1%} | Service {all_results['test_a']['service']['accuracy']:.1%}")
    print(f"🤝 Test-B: Service {all_results['test_b']['service']['accuracy']:.1%} | Strict {all_results['test_b']['strict']['accuracy']:.1%}")
    print(f"⚔️  Conflict: {all_results['conflict']['pass_rate']:.1%}")
    print("="*80)


if __name__ == "__main__":
    main()
