"""
Base model management for Qwen2.5-7B-Instruct.

This module handles loading, configuring, and freezing the base language model
with automatic VRAM detection and adaptive quantization.
"""

import os
import logging
import torch
from typing import Tuple, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)


def get_adaptive_quantization_config() -> bool:
    """
    Automatically determine if 4-bit quantization is needed based on GPU VRAM.
    
    Target: Qwen2.5-7B (Requires ~15GB+ for BF16 training, ~6GB for 4-bit)
    
    Returns:
        bool: True if 4-bit quantization should be enabled, False otherwise
    """
    if not torch.cuda.is_available():
        logging.warning("⚠️ No CUDA device detected. CPU mode will be very slow.")
        return False  # CPU fallback (slow)
    
    # Get total memory of the first GPU in GB
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Threshold set to 16GB (Safe margin for 7B model in BF16)
    if total_memory_gb < 16:
        logging.warning(
            f"⚠️ Low VRAM detected ({total_memory_gb:.1f} GB). "
            "Enabling 4-bit quantization for efficiency."
        )
        return True
    else:
        logging.info(
            f"✅ Sufficient VRAM detected ({total_memory_gb:.1f} GB). "
            "Using native precision (BF16)."
        )
        return False


def load_base_model(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    quantization: Optional[str] = None,  # None = auto-detect, "4bit", "none"
    device_map: str = "auto",
    trust_remote_code: bool = True,
    hf_token: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load base model and tokenizer with automatic download from HuggingFace Hub.
    
    Args:
        model_name: HuggingFace model name
        quantization: Quantization mode ("4bit", "none", or None for auto-detect)
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code
        hf_token: HuggingFace authentication token (optional)
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        RuntimeError: If model loading fails
    """
    logging.info(f"Loading base model: {model_name}")
    
    # Get HuggingFace token from environment if not provided
    if hf_token is None:
        hf_token = os.environ.get('HF_TOKEN', None)
    
    # Determine quantization setting
    use_4bit = False
    if quantization is None or quantization == "auto":
        use_4bit = get_adaptive_quantization_config()
    elif quantization == "4bit":
        use_4bit = True
        logging.info("4-bit quantization explicitly enabled")
    elif quantization == "none":
        use_4bit = False
        logging.info("Quantization explicitly disabled")
    else:
        raise ValueError(f"Invalid quantization mode: {quantization}")
    
    try:
        # Configure quantization if needed
        model_kwargs = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            "token": hf_token
        }
        
        if use_4bit:
            # Configure 4-bit quantization with bitsandbytes
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            model_kwargs["quantization_config"] = bnb_config
            logging.info("Using 4-bit quantization (NF4)")
        else:
            # Use bfloat16 for better precision
            model_kwargs["torch_dtype"] = torch.bfloat16
            logging.info("Using bfloat16 precision")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            token=hf_token
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        logging.info(f"✅ Successfully loaded model: {model_name}")
        logging.info(f"   Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
        
        return model, tokenizer
        
    except Exception as e:
        logging.error(f"❌ Failed to load model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")


def freeze_base_model(model: PreTrainedModel) -> None:
    """
    Freeze all parameters of the base model.
    
    Args:
        model: Model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False
    
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logging.info(f"✅ Froze base model: {frozen_params / 1e9:.2f}B / {total_params / 1e9:.2f}B parameters")


def get_model_memory_footprint(model: PreTrainedModel) -> float:
    """
    Calculate model memory footprint in GB.
    
    Args:
        model: Model to analyze
        
    Returns:
        Memory footprint in GB
    """
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem_total = mem_params + mem_buffers
    return mem_total / (1024**3)
