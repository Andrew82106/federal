"""
Quick evaluation script to verify model and adapters are working correctly.

This script runs a minimal test before the full evaluation to catch any issues early.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.base_model import load_base_model, freeze_base_model
from src.models.dual_adapter import DualAdapterModel, get_lora_config
from src.utils.config import load_config
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def quick_test():
    """Run quick test to verify model setup."""
    
    logging.info("="*80)
    logging.info("QUICK EVALUATION TEST")
    logging.info("="*80)
    
    # Load config
    config_path = project_root / "experiments" / "exp001_dual_adapter_fl" / "config.yaml"
    config = load_config(str(config_path))
    
    # Define paths
    results_dir = project_root / "results" / "exp001_dual_adapter_fl"
    
    global_adapter_path = results_dir / "checkpoints" / "final_adapters" / "global"
    strict_adapter_path = results_dir / "checkpoints" / "final_adapters" / "strict" / "local"
    service_adapter_path = results_dir / "checkpoints" / "final_adapters" / "service" / "local"
    
    # Check if adapters exist
    logging.info("\n1. Checking adapter files...")
    for name, path in [
        ("Global", global_adapter_path),
        ("Strict", strict_adapter_path),
        ("Service", service_adapter_path)
    ]:
        if path.exists():
            logging.info(f"   ✅ {name} adapter found: {path}")
        else:
            logging.error(f"   ❌ {name} adapter NOT found: {path}")
            logging.error("   Please ensure training completed successfully")
            return False
    
    # Load base model
    logging.info("\n2. Loading base model...")
    try:
        base_model, tokenizer = load_base_model(
            model_name=config['model']['base_model'],
            quantization=config['model'].get('quantization', 'auto')
        )
        freeze_base_model(base_model)
        logging.info("   ✅ Base model loaded successfully")
    except Exception as e:
        logging.error(f"   ❌ Failed to load base model: {e}")
        return False
    
    # Test adapter loading
    logging.info("\n3. Testing adapter loading...")
    try:
        lora_config = get_lora_config()
        dual_model = DualAdapterModel(base_model, lora_config)
        
        # Load global adapter
        dual_model.add_global_adapter(
            adapter_name="global",
            adapter_path=str(global_adapter_path)
        )
        logging.info("   ✅ Global adapter loaded")
        
        # Load strict adapter
        dual_model.add_local_adapter(
            adapter_name="local_strict",
            adapter_path=str(strict_adapter_path)
        )
        logging.info("   ✅ Strict adapter loaded")
        
        # Activate adapters
        dual_model.set_active_adapters(["global", "local_strict"])
        logging.info("   ✅ Adapters activated")
        
    except Exception as e:
        logging.error(f"   ❌ Failed to load adapters: {e}")
        return False
    
    # Test inference
    logging.info("\n4. Testing inference...")
    try:
        model = dual_model.get_model()
        model.eval()
        
        # Test question
        test_question = "我大专学历，无社保，能落户吗？"
        
        messages = [
            {"role": "system", "content": "你是上海市公安局的政务助手，请根据上海市的政策回答问题。"},
            {"role": "user", "content": test_question}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        
        response = response.strip()
        
        logging.info(f"   Question: {test_question}")
        logging.info(f"   Response: {response[:200]}...")
        logging.info("   ✅ Inference successful")
        
    except Exception as e:
        logging.error(f"   ❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check test data
    logging.info("\n5. Checking test data files...")
    test_data_dir = project_root / "data" / "test"
    test_files = [
        "global_test.json",
        "strict_test.json",
        "service_test.json",
        "conflict_cases.json"
    ]
    
    for filename in test_files:
        filepath = test_data_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"   ✅ {filename}: {len(data)} cases")
        else:
            logging.error(f"   ❌ {filename} NOT found")
            return False
    
    logging.info("\n" + "="*80)
    logging.info("✅ ALL CHECKS PASSED - Ready for full evaluation!")
    logging.info("="*80)
    logging.info("\nRun full evaluation with:")
    logging.info("  python experiments/exp001_dual_adapter_fl/eval.py")
    
    return True

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
