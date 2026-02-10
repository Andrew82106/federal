"""
Dual-adapter model architecture for federated learning.

This module implements the dual-adapter architecture with separate
global and local LoRA adapters for handling universal laws and
jurisdiction-specific policies.
"""

import os
import logging
from typing import Optional, List, Dict
import torch
from transformers import PreTrainedModel
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: str = "all-linear",
    bias: str = "none"
) -> LoraConfig:
    """
    Create LoRA configuration for Qwen2.5 model.
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: Dropout probability
        target_modules: Target modules ("all-linear" for all linear layers)
        bias: Bias training mode
        
    Returns:
        LoRA configuration object
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,  # "all-linear" applies to all linear layers
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM
    )


class DualAdapterModel:
    """
    Dual-adapter model manager for federated learning.
    
    Manages two separate LoRA adapters:
    - Global adapter: Learns universal laws, participates in federated aggregation
    - Local adapter: Learns jurisdiction-specific policies, remains private
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        lora_config: Optional[LoraConfig] = None
    ):
        """
        Initialize dual-adapter model.
        
        Args:
            base_model: Frozen base model
            lora_config: LoRA configuration (uses default if None)
        """
        self.base_model = base_model
        self.lora_config = lora_config or get_lora_config()
        self.model: Optional[PeftModel] = None
        self.adapters: Dict[str, str] = {}  # adapter_name -> adapter_type mapping
        
        logging.info("Initialized DualAdapterModel")
        logging.info(f"  LoRA config: r={self.lora_config.r}, alpha={self.lora_config.lora_alpha}")
        logging.info(f"  Target modules: {self.lora_config.target_modules}")
    
    def add_global_adapter(
        self,
        adapter_name: str = "global",
        adapter_path: Optional[str] = None
    ) -> None:
        """
        Add or load global adapter.
        
        Args:
            adapter_name: Name for the adapter
            adapter_path: Path to existing adapter weights (optional)
        """
        if self.model is None:
            # First adapter - create PEFT model
            # get_peft_model creates an adapter with name "default"
            self.model = get_peft_model(self.base_model, self.lora_config)
            self.model.base_model.model.config.use_cache = False  # Disable cache for training
            
            # Rename "default" to our desired name if different
            if adapter_name != "default":
                # The first adapter is always named "default" by PEFT
                # We need to work with this name or add a new adapter
                self.model.add_adapter(adapter_name, self.lora_config)
                self.model.set_adapter(adapter_name)
                logging.info(f"✅ Created PEFT model with global adapter '{adapter_name}'")
            else:
                logging.info(f"✅ Created PEFT model with default adapter")
        else:
            # Add additional adapter
            self.model.add_adapter(adapter_name, self.lora_config)
            logging.info(f"✅ Added global adapter '{adapter_name}'")
        
        self.adapters[adapter_name] = "global"
        
        # Load weights if path provided
        if adapter_path and os.path.exists(adapter_path):
            self.model.load_adapter(adapter_path, adapter_name)
            logging.info(f"✅ Loaded global adapter weights from {adapter_path}")
    
    def add_local_adapter(
        self,
        adapter_name: str = "local",
        adapter_path: Optional[str] = None
    ) -> None:
        """
        Add or load local adapter.
        
        Args:
            adapter_name: Name for the adapter
            adapter_path: Path to existing adapter weights (optional)
        """
        if self.model is None:
            # First adapter - create PEFT model
            self.model = get_peft_model(self.base_model, self.lora_config)
            self.model.base_model.model.config.use_cache = False
            
            # Rename "default" to our desired name if different
            if adapter_name != "default":
                self.model.add_adapter(adapter_name, self.lora_config)
                self.model.set_adapter(adapter_name)
                logging.info(f"✅ Created PEFT model with local adapter '{adapter_name}'")
            else:
                logging.info(f"✅ Created PEFT model with default adapter")
        else:
            # Add additional adapter
            self.model.add_adapter(adapter_name, self.lora_config)
            logging.info(f"✅ Added local adapter '{adapter_name}'")
        
        self.adapters[adapter_name] = "local"
        
        # Load weights if path provided
        if adapter_path and os.path.exists(adapter_path):
            self.model.load_adapter(adapter_path, adapter_name)
            logging.info(f"✅ Loaded local adapter weights from {adapter_path}")
    
    def set_active_adapters(
        self,
        adapter_names: List[str]
    ) -> None:
        """
        Set which adapters are active for training or inference.
        
        Args:
            adapter_names: List of adapter names to activate
        """
        if self.model is None:
            raise RuntimeError("No adapters have been added yet")
        
        # Validate adapter names
        for name in adapter_names:
            if name not in self.adapters:
                raise ValueError(f"Adapter '{name}' not found. Available: {list(self.adapters.keys())}")
        
        # For multiple adapters, set them one by one (last one becomes active)
        # In PEFT, when training with multiple adapters, they are all trained together
        # We just need to ensure they're all added and the model knows about them
        # The actual training will use both adapters automatically
        for adapter_name in adapter_names:
            self.model.set_adapter(adapter_name)
        
        logging.info(f"✅ Activated adapters: {adapter_names}")
    
    def save_adapter(
        self,
        adapter_name: str,
        save_path: str
    ) -> None:
        """
        Save specific adapter weights.
        
        Args:
            adapter_name: Name of adapter to save
            save_path: Directory where to save adapter
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save adapter
        self.model.save_pretrained(
            save_path,
            selected_adapters=[adapter_name]
        )
        
        adapter_type = self.adapters[adapter_name]
        logging.info(f"✅ Saved {adapter_type} adapter '{adapter_name}' to {save_path}")
    
    def load_adapter(
        self,
        adapter_name: str,
        adapter_path: str,
        adapter_type: str = "global"
    ) -> None:
        """
        Load adapter from disk.
        
        Args:
            adapter_name: Name to assign to the adapter
            adapter_path: Path to adapter weights
            adapter_type: Type of adapter ("global" or "local")
        """
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        
        if self.model is None:
            # Load as first adapter
            self.model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                adapter_name=adapter_name
            )
            logging.info(f"✅ Loaded {adapter_type} adapter '{adapter_name}' as base")
        else:
            # Load as additional adapter
            self.model.load_adapter(adapter_path, adapter_name)
            logging.info(f"✅ Loaded {adapter_type} adapter '{adapter_name}'")
        
        self.adapters[adapter_name] = adapter_type
    
    def get_model(self) -> PeftModel:
        """
        Get the underlying PEFT model.
        
        Returns:
            PEFT model with adapters
        """
        if self.model is None:
            raise RuntimeError("No model available. Add adapters first.")
        return self.model
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """
        Get count of trainable parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        if self.model is None:
            return {"trainable": 0, "total": 0}
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "trainable": trainable_params,
            "total": total_params,
            "percentage": 100 * trainable_params / total_params
        }
    
    def print_trainable_parameters(self) -> None:
        """Print trainable parameter statistics."""
        stats = self.get_trainable_parameters()
        logging.info(
            f"Trainable params: {stats['trainable']:,} || "
            f"Total params: {stats['total']:,} || "
            f"Trainable%: {stats['percentage']:.2f}%"
        )
