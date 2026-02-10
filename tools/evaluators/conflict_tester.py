"""
Conflict tester for evaluating model's ability to handle jurisdiction-specific policies.

This module tests whether the model can produce different answers based on
which local adapter is activated, with dynamic system prompt injection.
"""

import logging
from typing import Dict, List, Any, Optional
import torch

from src.models.base_model import load_base_model, freeze_base_model
from src.models.dual_adapter import DualAdapterModel, get_lora_config


class ConflictTester:
    """
    Conflict tester with city identity system prompt injection.
    
    Tests model's ability to provide jurisdiction-specific answers
    by loading different local adapters with corresponding system prompts.
    """
    
    def __init__(
        self,
        base_model_name: str,
        global_adapter_path: str,
        config: Optional[Dict] = None
    ):
        """
        Initialize conflict tester.
        
        Args:
            base_model_name: Name of base model
            global_adapter_path: Path to global adapter
            config: Optional configuration dictionary
        """
        self.base_model_name = base_model_name
        self.global_adapter_path = global_adapter_path
        self.config = config or {}
        
        # Load base model
        logging.info("Loading base model for conflict testing...")
        self.base_model, self.tokenizer = load_base_model(
            model_name=base_model_name,
            quantization=self.config.get('quantization', 'auto')
        )
        freeze_base_model(self.base_model)
        
        # Set padding side to left for decoder-only models
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logging.info("✅ ConflictTester initialized")
    
    def test_conflict_case(
        self,
        question: str,
        local_adapter_paths: Dict[str, str],
        expected_keywords: Dict[str, List[str]],
        system_prompts: Optional[Dict[str, str]] = None,
        max_new_tokens: int = 256
    ) -> Dict[str, Any]:
        """
        Test a conflict case with different local adapters.
        
        Args:
            question: Test question
            local_adapter_paths: Dict of adapter_name -> path
            expected_keywords: Dict of adapter_name -> list of expected keywords
            system_prompts: Dict of adapter_name -> system prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Test results with responses and keyword matches
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"Testing conflict case: {question}")
        logging.info(f"{'='*80}")
        
        results = {
            'question': question,
            'responses': {},
            'keyword_matches': {},
            'success': True
        }
        
        # Default system prompts if not provided (联合城市身份)
        if system_prompts is None:
            system_prompts = {
                'strict': '你是上海市（户政）与北京市（交管）的联合政务助手。请依据这两个城市严格、规范的管理规定进行回答。对于违规行为，请强调处罚和红线。',
                'service': '你是石家庄市（户政）与南宁市（交管）的联合政务助手。请依据这两个城市便民、宽松、人性化的政策进行回答。对于轻微违章，请强调教育与纠正。'
            }
        
        # Test with each local adapter
        for adapter_name, adapter_path in local_adapter_paths.items():
            logging.info(f"\nTesting with '{adapter_name}' adapter...")
            
            # Get system prompt for this adapter
            system_prompt = system_prompts.get(adapter_name, '')
            
            # Generate response
            response = self._generate_response(
                question=question,
                local_adapter_path=adapter_path,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens
            )
            
            results['responses'][adapter_name] = response
            
            # Check keywords
            expected = expected_keywords.get(adapter_name, [])
            matches = self._check_keywords(response, expected)
            results['keyword_matches'][adapter_name] = matches
            
            # Log results
            logging.info(f"Response: {response[:200]}...")
            logging.info(f"Expected keywords: {expected}")
            logging.info(f"Matched: {matches['matched']}")
            logging.info(f"Missing: {matches['missing']}")
            
            if matches['missing']:
                results['success'] = False
        
        return results
    
    def _generate_response(
        self,
        question: str,
        local_adapter_path: str,
        system_prompt: str,
        max_new_tokens: int
    ) -> str:
        """
        Generate response with specific local adapter and system prompt.
        
        Args:
            question: Question to answer
            local_adapter_path: Path to local adapter
            system_prompt: System prompt for city identity
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        # Create dual adapter model
        lora_config = get_lora_config()
        dual_model = DualAdapterModel(self.base_model, lora_config)
        
        # Load global adapter
        dual_model.add_global_adapter(
            adapter_name="global",
            adapter_path=self.global_adapter_path
        )
        
        # Load local adapter
        dual_model.add_local_adapter(
            adapter_name="local",
            adapter_path=local_adapter_path
        )
        
        # Activate both adapters
        dual_model.set_active_adapters(["global", "local"])
        
        model = dual_model.get_model()
        model.eval()
        
        # Format prompt with system prompt injection
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        
        # Apply chat template
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
        
        # Extract only the assistant's response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        
        response = response.strip()
        
        return response
    
    def _check_keywords(
        self,
        response: str,
        expected_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Check if expected keywords are present in response.
        
        Args:
            response: Generated response
            expected_keywords: List of expected keywords
            
        Returns:
            Dictionary with matched and missing keywords
        """
        matched = []
        missing = []
        
        for keyword in expected_keywords:
            if keyword in response:
                matched.append(keyword)
            else:
                missing.append(keyword)
        
        return {
            'matched': matched,
            'missing': missing,
            'match_rate': len(matched) / len(expected_keywords) if expected_keywords else 1.0
        }
    
    def evaluate_conflict_behavior(
        self,
        response: str,
        evaluation_guide: Dict[str, List[str]]
    ) -> str:
        """
        Evaluate which city behavior the response exhibits based on keyword matching.
        
        This is a fast, deterministic evaluation method that doesn't require LLM judges.
        
        Args:
            response: Model's generated response
            evaluation_guide: Dict with 'strict_keywords' and 'service_keywords'
            
        Returns:
            One of: "STRICT_BEHAVIOR", "SERVICE_BEHAVIOR", "AMBIGUOUS", "NO_MATCH"
        """
        strict_keywords = evaluation_guide.get('strict_keywords', [])
        service_keywords = evaluation_guide.get('service_keywords', [])
        
        # Count keyword matches
        strict_hits = sum(1 for k in strict_keywords if k in response)
        service_hits = sum(1 for k in service_keywords if k in response)
        
        # Determine behavior based on matches
        if strict_hits > 0 and service_hits == 0:
            return "STRICT_BEHAVIOR"
        elif service_hits > 0 and strict_hits == 0:
            return "SERVICE_BEHAVIOR"
        elif strict_hits > 0 and service_hits > 0:
            return "AMBIGUOUS"  # Contains keywords from both cities
        else:
            return "NO_MATCH"  # No expected keywords found
    
    def test_conflict_with_guide(
        self,
        question: str,
        evaluation_guide: Dict[str, List[str]],
        local_adapter_paths: Dict[str, str],
        system_prompts: Optional[Dict[str, str]] = None,
        max_new_tokens: int = 256
    ) -> Dict[str, Any]:
        """
        Test a conflict case using evaluation_guide for fast keyword-based evaluation.
        
        This method is optimized for the enhanced conflict_cases.json format with
        evaluation_guide fields containing expected keywords for each city type.
        
        Args:
            question: Test question
            evaluation_guide: Dict with 'strict_keywords' and 'service_keywords'
            local_adapter_paths: Dict of adapter_name -> path (e.g., {'strict': path1, 'service': path2})
            system_prompts: Optional dict of adapter_name -> system prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Test results with responses and behavior classification
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"Testing conflict case: {question}")
        logging.info(f"{'='*80}")
        
        # Default system prompts
        if system_prompts is None:
            system_prompts = {
                'strict': '你是上海市公安局的政务助手，请根据上海市的政策回答问题。',
                'service': '你是石家庄市公安局的政务助手，请根据石家庄市的政策回答问题。'
            }
        
        results = {
            'question': question,
            'evaluation_guide': evaluation_guide,
            'responses': {},
            'behaviors': {},
            'success': True
        }
        
        # Test with each local adapter
        for adapter_name, adapter_path in local_adapter_paths.items():
            logging.info(f"\nTesting with '{adapter_name}' adapter...")
            
            # Get system prompt
            system_prompt = system_prompts.get(adapter_name, '')
            
            # Generate response
            response = self._generate_response(
                question=question,
                local_adapter_path=adapter_path,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens
            )
            
            results['responses'][adapter_name] = response
            
            # Evaluate behavior using keyword matching
            behavior = self.evaluate_conflict_behavior(response, evaluation_guide)
            results['behaviors'][adapter_name] = behavior
            
            # Check if behavior matches expected
            expected_behavior = f"{adapter_name.upper()}_BEHAVIOR"
            is_correct = (behavior == expected_behavior)
            
            # Log results
            logging.info(f"Response: {response[:200]}...")
            logging.info(f"Detected behavior: {behavior}")
            logging.info(f"Expected behavior: {expected_behavior}")
            logging.info(f"Correct: {'✅' if is_correct else '❌'}")
            
            if not is_correct:
                results['success'] = False
        
        return results
    
    def run_test_suite(
        self,
        test_cases: List[Dict]
    ) -> Dict[str, Any]:
        """
        Run a suite of conflict test cases.
        
        Args:
            test_cases: List of test case dictionaries
            
        Returns:
            Aggregated test results
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"Running Conflict Test Suite ({len(test_cases)} cases)")
        logging.info(f"{'='*80}")
        
        results = {
            'total_cases': len(test_cases),
            'passed': 0,
            'failed': 0,
            'cases': []
        }
        
        for i, test_case in enumerate(test_cases, 1):
            logging.info(f"\n--- Test Case {i}/{len(test_cases)} ---")
            
            case_result = self.test_conflict_case(
                question=test_case['question'],
                local_adapter_paths=test_case['adapters'],
                expected_keywords=test_case['expected'],
                system_prompts=test_case.get('system_prompts'),
                max_new_tokens=test_case.get('max_new_tokens', 256)
            )
            
            results['cases'].append(case_result)
            
            if case_result['success']:
                results['passed'] += 1
            else:
                results['failed'] += 1
        
        results['pass_rate'] = results['passed'] / results['total_cases']
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Test Suite Results:")
        logging.info(f"  Total: {results['total_cases']}")
        logging.info(f"  Passed: {results['passed']}")
        logging.info(f"  Failed: {results['failed']}")
        logging.info(f"  Pass Rate: {results['pass_rate']:.2%}")
        logging.info(f"{'='*80}")
        
        return results
    
    def run_guided_test_suite(
        self,
        test_cases: List[Dict],
        local_adapter_paths: Dict[str, str],
        system_prompts: Optional[Dict[str, str]] = None,
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """
        Run a suite of conflict test cases using evaluation_guide with batch inference.
        
        Args:
            test_cases: List of dicts with 'instruction' and 'evaluation_guide' fields
            local_adapter_paths: Dict of adapter_name -> path
            system_prompts: Optional dict of adapter_name -> system prompt
            batch_size: Batch size for inference
            
        Returns:
            Aggregated test results with behavior classification
        """
        from tqdm import tqdm
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Running Guided Conflict Test Suite ({len(test_cases)} cases)")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"{'='*80}")
        
        # Default system prompts
        if system_prompts is None:
            system_prompts = {
                'strict': '你是上海市（户政）与北京市（交管）的联合政务助手。请依据这两个城市严格、规范的管理规定进行回答。对于违规行为，请强调处罚和红线。',
                'service': '你是石家庄市（户政）与南宁市（交管）的联合政务助手。请依据这两个城市便民、宽松、人性化的政策进行回答。对于轻微违章，请强调教育与纠正。'
            }
        
        results = {
            'total_cases': len(test_cases),
            'passed': 0,
            'failed': 0,
            'ambiguous': 0,
            'no_match': 0,
            'cases': []
        }
        
        # Process each adapter separately with batch inference
        for adapter_name, adapter_path in local_adapter_paths.items():
            logging.info(f"\nProcessing with '{adapter_name}' adapter...")
            
            # Load adapter once
            lora_config = get_lora_config()
            dual_model = DualAdapterModel(self.base_model, lora_config)
            dual_model.add_global_adapter(
                adapter_name="global",
                adapter_path=self.global_adapter_path
            )
            dual_model.add_local_adapter(
                adapter_name="local",
                adapter_path=adapter_path
            )
            dual_model.set_active_adapters(["global", "local"])
            
            model = dual_model.get_model()
            model.eval()
            
            system_prompt = system_prompts.get(adapter_name, '')
            
            # Batch processing
            adapter_responses = []
            for i in tqdm(range(0, len(test_cases), batch_size), desc=f"Conflict-{adapter_name}"):
                batch = test_cases[i:i+batch_size]
                
                # Prepare batch prompts
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
                
                # Tokenize batch
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Generate batch
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode responses
                batch_responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
                
                # Extract assistant responses
                for response in batch_responses:
                    if "<|im_start|>assistant\n" in response:
                        response = response.split("<|im_start|>assistant\n")[-1]
                    elif "<|im_start|>assistant" in response:
                        response = response.split("<|im_start|>assistant")[-1]
                    
                    if "<|im_end|>" in response:
                        response = response.split("<|im_end|>")[0]
                    
                    response = response.replace("<|endoftext|>", "").strip()
                    adapter_responses.append(response)
            
            # Store responses for this adapter
            if adapter_name == 'strict':
                strict_responses = adapter_responses
            else:
                service_responses = adapter_responses
        
        # Evaluate all cases
        logging.info("\nEvaluating conflict behaviors...")
        for i, test_case in enumerate(tqdm(test_cases, desc="Evaluating")):
            case_result = {
                'question': test_case['instruction'],
                'evaluation_guide': test_case['evaluation_guide'],
                'responses': {
                    'strict': strict_responses[i],
                    'service': service_responses[i]
                },
                'behaviors': {},
                'success': True
            }
            
            # Evaluate behaviors
            for adapter_name in ['strict', 'service']:
                response = case_result['responses'][adapter_name]
                behavior = self.evaluate_conflict_behavior(response, test_case['evaluation_guide'])
                case_result['behaviors'][adapter_name] = behavior
                
                expected_behavior = f"{adapter_name.upper()}_BEHAVIOR"
                if behavior != expected_behavior:
                    case_result['success'] = False
                    
                    if behavior == 'AMBIGUOUS':
                        results['ambiguous'] += 1
                    elif behavior == 'NO_MATCH':
                        results['no_match'] += 1
            
            results['cases'].append(case_result)
            
            if case_result['success']:
                results['passed'] += 1
            else:
                results['failed'] += 1
        
        results['pass_rate'] = results['passed'] / results['total_cases']
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Guided Test Suite Results:")
        logging.info(f"  Total: {results['total_cases']}")
        logging.info(f"  Passed: {results['passed']}")
        logging.info(f"  Failed: {results['failed']}")
        logging.info(f"  Ambiguous: {results['ambiguous']}")
        logging.info(f"  No Match: {results['no_match']}")
        logging.info(f"  Pass Rate: {results['pass_rate']:.2%}")
        logging.info(f"{'='*80}")
        
        return results
