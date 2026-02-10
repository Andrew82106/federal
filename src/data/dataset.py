"""
Dataset classes for Alpaca-format data with Qwen Chat Template support.

This module provides dataset classes that load Alpaca-format JSON data
and apply Qwen's chat template formatting.
"""

import json
import logging
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def apply_qwen_chat_template(
    instruction: str,
    input_text: str = "",
    output: str = "",
    tokenizer: Optional[PreTrainedTokenizer] = None,
    system_prompt: str = ""
) -> str:
    """
    Apply Qwen Chat Template to format data.
    
    Qwen uses the <|im_start|> and <|im_end|> format:
    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {instruction}{input}<|im_end|>
    <|im_start|>assistant
    {output}<|im_end|>
    
    Args:
        instruction: User instruction/question
        input_text: Additional input context (usually empty in Alpaca format)
        output: Expected output/response
        tokenizer: Qwen tokenizer (can use built-in chat template if available)
        system_prompt: System prompt for role/identity injection
        
    Returns:
        Formatted prompt string
    """
    # Combine instruction and input
    user_message = instruction
    if input_text:
        user_message += f"\n{input_text}"
    
    # Try to use tokenizer's built-in chat template if available
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": user_message})
        
        # Add assistant message if output provided
        if output:
            messages.append({"role": "assistant", "content": output})
        
        try:
            # Use tokenizer's chat template
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=(not output)  # Add prompt if no output
            )
            return formatted
        except Exception as e:
            logging.warning(f"Failed to use tokenizer chat template: {e}. Using manual format.")
    
    # Manual Qwen format as fallback
    formatted = ""
    
    if system_prompt:
        formatted += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    
    formatted += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    
    if output:
        formatted += f"<|im_start|>assistant\n{output}<|im_end|>"
    else:
        formatted += "<|im_start|>assistant\n"
    
    return formatted


class AlpacaDataset(Dataset):
    """
    Dataset for Alpaca-format data with Qwen Chat Template support.
    
    Expected JSON format:
    [
        {
            "instruction": "问题或指令",
            "input": "",  # Usually empty
            "output": "期望的回答"
        },
        ...
    ]
    """
    
    def __init__(
        self,
        data_paths: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        apply_chat_template: bool = True,
        system_prompt: str = ""
    ):
        """
        Initialize dataset.
        
        Args:
            data_paths: List of paths to JSON data files
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            apply_chat_template: Whether to apply Qwen chat template
            system_prompt: System prompt for role injection (optional)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.apply_chat_template = apply_chat_template
        self.system_prompt = system_prompt
        
        # Load data from all paths
        self.data = []
        for path in data_paths:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.data.extend(data)
                logging.info(f"Loaded {len(data)} samples from {path}")
        
        logging.info(f"Total dataset size: {len(self.data)} samples")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        item = self.data[idx]
        
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        
        # Format with Qwen chat template
        if self.apply_chat_template:
            formatted_text = apply_qwen_chat_template(
                instruction=instruction,
                input_text=input_text,
                output=output,
                tokenizer=self.tokenizer,
                system_prompt=self.system_prompt
            )
        else:
            # Simple concatenation fallback
            formatted_text = f"{instruction}\n{input_text}\n{output}"
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare labels (same as input_ids for causal LM)
        labels = encoding['input_ids'].clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }


def create_mixed_dataset(
    global_data_path: str,
    local_data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 1024,
    system_prompt: str = ""
) -> AlpacaDataset:
    """
    Create a mixed dataset combining global and local data.
    
    Args:
        global_data_path: Path to global training data
        local_data_path: Path to local/client-specific data
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        system_prompt: System prompt for role injection
        
    Returns:
        Combined dataset
    """
    return AlpacaDataset(
        data_paths=[global_data_path, local_data_path],
        tokenizer=tokenizer,
        max_length=max_length,
        apply_chat_template=True,
        system_prompt=system_prompt
    )
