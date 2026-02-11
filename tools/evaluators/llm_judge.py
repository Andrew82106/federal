"""
LLM-as-Judge evaluator for answer correctness.

Uses the base Qwen model to judge if a response correctly answers the question.
"""

import logging
from typing import List, Dict, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMJudge:
    """
    LLM-based judge for evaluating answer correctness.
    
    Uses Qwen model to determine if a response correctly answers
    the expected answer, handling semantic equivalence.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize LLM judge.
        
        Args:
            model_path: Path to Qwen model
            device: Device to run on
        """
        self.device = device
        
        logging.info(f"Loading judge model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        # Set padding
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logging.info("✅ Judge model loaded")
    
    def judge_single(
        self,
        question: str,
        expected: str,
        response: str,
        max_new_tokens: int = 50
    ) -> Tuple[bool, str]:
        """
        Judge if response correctly answers the question.
        
        Args:
            question: Original question
            expected: Expected answer
            response: Model's response
            max_new_tokens: Max tokens for judge response
            
        Returns:
            Tuple of (is_correct, explanation)
        """
        # Construct judge prompt
        judge_prompt = f"""你是一个严格的答案评判员。请判断"模型回答"是否正确回答了问题，并且与"标准答案"的核心意思一致。

问题：{question}

标准答案：{expected}

模型回答：{response}

评判标准：
1. 如果模型回答的核心意思与标准答案一致，判定为"正确"
2. 允许表述方式不同，但核心意思必须相同
3. 如果模型回答包含额外的解释或引用法条，只要核心答案正确即可
4. 如果模型回答与标准答案矛盾或答非所问，判定为"错误"

请直接回答"正确"或"错误"，然后简要说明理由（不超过20字）。

判定："""
        
        messages = [{"role": "user", "content": judge_prompt}]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        judge_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract judge's decision
        if "<|im_start|>assistant" in judge_response:
            judge_response = judge_response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in judge_response:
            judge_response = judge_response.split("<|im_end|>")[0]
        
        judge_response = judge_response.strip()
        
        # Parse decision
        is_correct = "正确" in judge_response and "错误" not in judge_response
        
        return is_correct, judge_response
    
    def judge_batch(
        self,
        questions: List[str],
        expected_answers: List[str],
        responses: List[str],
        batch_size: int = 8
    ) -> List[Tuple[bool, str]]:
        """
        Judge multiple responses in batches.
        
        Args:
            questions: List of questions
            expected_answers: List of expected answers
            responses: List of model responses
            batch_size: Batch size for processing
            
        Returns:
            List of (is_correct, explanation) tuples
        """
        from tqdm import tqdm
        
        results = []
        total = len(questions)
        
        for i in tqdm(range(0, total, batch_size), desc="LLM Judge"):
            batch_questions = questions[i:i+batch_size]
            batch_expected = expected_answers[i:i+batch_size]
            batch_responses = responses[i:i+batch_size]
            
            # Prepare batch prompts
            prompts = []
            for q, e, r in zip(batch_questions, batch_expected, batch_responses):
                judge_prompt = f"""你是一个严格的答案评判员。请判断"模型回答"是否正确回答了问题，并且与"标准答案"的核心意思一致。

问题：{q}

标准答案：{e}

模型回答：{r}

评判标准：
1. 如果模型回答的核心意思与标准答案一致，判定为"正确"
2. 允许表述方式不同，但核心意思必须相同
3. 如果模型回答包含额外的解释或引用法条，只要核心答案正确即可
4. 如果模型回答与标准答案矛盾或答非所问，判定为"错误"

请直接回答"正确"或"错误"，然后简要说明理由（不超过20字）。

判定："""
                
                messages = [{"role": "user", "content": judge_prompt}]
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
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode responses
            judge_responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            
            # Parse decisions
            for judge_response in judge_responses:
                # Extract assistant's response
                if "<|im_start|>assistant\n" in judge_response:
                    judge_response = judge_response.split("<|im_start|>assistant\n")[-1]
                elif "<|im_start|>assistant" in judge_response:
                    judge_response = judge_response.split("<|im_start|>assistant")[-1]
                
                if "<|im_end|>" in judge_response:
                    judge_response = judge_response.split("<|im_end|>")[0]
                
                judge_response = judge_response.replace("<|endoftext|>", "").strip()
                
                # Parse decision
                is_correct = "正确" in judge_response and "错误" not in judge_response
                results.append((is_correct, judge_response))
        
        return results
