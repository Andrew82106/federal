#!/usr/bin/env python3
"""
Model Testing Script

This script tests if the Qwen2.5-7B-Instruct model can be loaded and used
in the Dual-Adapter Federated Learning project.

Usage:
    python scripts/test_model.py [--model-path PATH]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_model_loading(model_path: str, logger: logging.Logger) -> bool:
    """
    Test if the model can be loaded successfully.
    
    Args:
        model_path: Path to the model directory
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("Testing Model Loading")
    logger.info("=" * 80)
    logger.info(f"Model path: {model_path}")
    
    try:
        # Check if path exists
        if not Path(model_path).exists():
            logger.error(f"Model path does not exist: {model_path}")
            return False
        
        # Check GPU availability
        if torch.cuda.is_available():
            logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            device = "cuda"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"  GPU Memory: {gpu_memory:.2f} GB")
        else:
            logger.warning("⚠ No GPU detected. Using CPU (will be slow)")
            device = "cpu"
        
        # Load tokenizer
        logger.info("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        logger.info("✓ Tokenizer loaded successfully")
        
        # Load model
        logger.info("\nLoading model...")
        logger.info("This may take a few minutes...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("✓ Model loaded successfully")
        
        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"\nModel Information:")
        logger.info(f"  Total parameters: {num_params:,}")
        logger.info(f"  Model size: ~{num_params * 2 / 1024**3:.2f} GB (BF16)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_model_inference(model_path: str, logger: logging.Logger) -> bool:
    """
    Test if the model can generate text.
    
    Args:
        model_path: Path to the model directory
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("\n" + "=" * 80)
    logger.info("Testing Model Inference")
    logger.info("=" * 80)
    
    try:
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Test with a simple question
        test_question = "警车、消防车非执行紧急任务时，是否享有道路优先通行权？"
        
        logger.info(f"\nTest Question: {test_question}")
        
        # Format with Qwen chat template
        messages = [
            {"role": "system", "content": "你是一个专业的法律助手。"},
            {"role": "user", "content": test_question}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate
        logger.info("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                top_p=1.0
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
            response = response.split("<|im_end|>")[0].strip()
        
        logger.info(f"\nModel Response:\n{response}")
        logger.info("\n✓ Inference test successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Inference test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_model_compatibility(model_path: str, logger: logging.Logger) -> bool:
    """
    Test if the model is compatible with PEFT/LoRA.
    
    Args:
        model_path: Path to the model directory
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("\n" + "=" * 80)
    logger.info("Testing PEFT/LoRA Compatibility")
    logger.info("=" * 80)
    
    try:
        from peft import LoraConfig, get_peft_model
        
        # Load model
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Create LoRA config
        logger.info("\nCreating LoRA configuration...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        logger.info("Applying LoRA to model...")
        peft_model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        
        logger.info(f"\n✓ LoRA applied successfully")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ PEFT compatibility test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test if Qwen2.5-7B-Instruct model works with the project"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/autodl-tmp/Downloads",
        help="Path to the model directory"
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference test (faster)"
    )
    parser.add_argument(
        "--skip-peft",
        action="store_true",
        help="Skip PEFT compatibility test"
    )
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("Qwen2.5-7B-Instruct Model Testing")
    logger.info("=" * 80)
    logger.info(f"Model path: {args.model_path}")
    
    # Test 1: Model loading
    logger.info("\n[Test 1/3] Model Loading")
    loading_success = test_model_loading(args.model_path, logger)
    
    if not loading_success:
        logger.error("\n✗ Model loading failed. Cannot proceed with other tests.")
        return 1
    
    # Test 2: Inference (optional)
    if not args.skip_inference:
        logger.info("\n[Test 2/3] Model Inference")
        inference_success = test_model_inference(args.model_path, logger)
    else:
        logger.info("\n[Test 2/3] Model Inference (SKIPPED)")
        inference_success = True
    
    # Test 3: PEFT compatibility (optional)
    if not args.skip_peft:
        logger.info("\n[Test 3/3] PEFT/LoRA Compatibility")
        peft_success = test_model_compatibility(args.model_path, logger)
    else:
        logger.info("\n[Test 3/3] PEFT/LoRA Compatibility (SKIPPED)")
        peft_success = True
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)
    logger.info(f"Model Loading:        {'✓ PASS' if loading_success else '✗ FAIL'}")
    logger.info(f"Model Inference:      {'✓ PASS' if inference_success else '✗ FAIL' if not args.skip_inference else '⊘ SKIPPED'}")
    logger.info(f"PEFT Compatibility:   {'✓ PASS' if peft_success else '✗ FAIL' if not args.skip_peft else '⊘ SKIPPED'}")
    
    all_success = loading_success and inference_success and peft_success
    
    if all_success:
        logger.info("\n✓ All tests passed! The model is compatible with the project.")
        logger.info("\nNext steps:")
        logger.info("  1. Update config.yaml with the correct model path")
        logger.info("  2. Run training: python experiments/exp001_dual_adapter_fl/train.py")
        return 0
    else:
        logger.error("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
