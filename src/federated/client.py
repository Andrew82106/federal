"""
Client trainer for federated learning.

This module implements the client-side training logic for federated learning,
including model setup, data preparation, and local training.
"""

import os
import logging
from typing import Dict, Optional, Tuple
import torch
from transformers import TrainingArguments, Trainer
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.models.base_model import load_base_model, freeze_base_model
from src.models.dual_adapter import DualAdapterModel, get_lora_config
from src.data.dataset import create_mixed_dataset


class ClientTrainer:
    """
    Client trainer for federated learning.
    
    Handles local training for a single client with support for
    dual-adapter architecture.
    """
    
    def __init__(
        self,
        client_id: str,
        config: Dict
    ):
        """
        Initialize client trainer.
        
        Args:
            client_id: Unique identifier for this client
            config: Configuration dictionary
        """
        self.client_id = client_id
        self.config = config
        self.base_model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.dual_adapter_model: Optional[DualAdapterModel] = None
        
        logging.info(f"Initialized ClientTrainer for client '{client_id}'")
    
    def train_round(
        self,
        round_num: int,
        global_adapter_path: Optional[str],
        local_adapter_path: Optional[str],
        global_data_path: str,
        local_data_path: str,
        output_dir: str,
        system_prompt: str = ""
    ) -> Tuple[str, str]:
        """
        Execute one round of local training.
        
        Args:
            round_num: Current round number
            global_adapter_path: Path to global adapter (None for first round)
            local_adapter_path: Path to local adapter (None for first round)
            global_data_path: Path to global training data
            local_data_path: Path to local training data
            output_dir: Directory for saving outputs
            system_prompt: System prompt for this client
            
        Returns:
            Tuple of (global_adapter_path, local_adapter_path)
        """
        logging.info(f"=" * 80)
        logging.info(f"Client '{self.client_id}' - Round {round_num} Training")
        logging.info(f"=" * 80)
        
        # Setup model
        self.dual_adapter_model = self._setup_model(
            global_adapter_path,
            local_adapter_path
        )
        
        # Prepare data
        train_dataset = self._prepare_data(
            global_data_path,
            local_data_path,
            system_prompt
        )
        
        # Train
        self._train(
            self.dual_adapter_model,
            train_dataset,
            output_dir
        )
        
        # Save adapters
        global_save_path = os.path.join(output_dir, "global_adapter")
        local_save_path = os.path.join(output_dir, "local_adapter")
        
        self.dual_adapter_model.save_adapter("global", global_save_path)
        self.dual_adapter_model.save_adapter("local", local_save_path)
        
        # Clean up model and free memory
        del self.dual_adapter_model
        self.dual_adapter_model = None
        
        # Force garbage collection and clear CUDA cache
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logging.info("✅ Cleared CUDA cache after training")
        
        logging.info(f"✅ Client '{self.client_id}' completed round {round_num}")
        
        return global_save_path, local_save_path
    
    def _setup_model(
        self,
        global_adapter_path: Optional[str],
        local_adapter_path: Optional[str]
    ) -> DualAdapterModel:
        """
        Setup model with adapters.
        
        Args:
            global_adapter_path: Path to existing global adapter
            local_adapter_path: Path to existing local adapter
            
        Returns:
            Configured DualAdapterModel
        """
        # Load base model if not already loaded
        if self.base_model is None:
            logging.info("Loading base model...")
            self.base_model, self.tokenizer = load_base_model(
                model_name=self.config['model']['base_model'],
                quantization=self.config['model'].get('quantization', 'auto'),
                hf_token=self.config.get('huggingface', {}).get('token')
            )
            freeze_base_model(self.base_model)
        
        # Create LoRA config
        lora_config = get_lora_config(
            r=self.config['model']['lora_config']['r'],
            lora_alpha=self.config['model']['lora_config']['lora_alpha'],
            lora_dropout=self.config['model']['lora_config']['lora_dropout'],
            target_modules=self.config['model']['lora_config']['target_modules']
        )
        
        # Create dual adapter model
        dual_model = DualAdapterModel(self.base_model, lora_config)
        
        # Add/load global adapter
        dual_model.add_global_adapter(
            adapter_name="global",
            adapter_path=global_adapter_path
        )
        
        # Add/load local adapter
        dual_model.add_local_adapter(
            adapter_name="local",
            adapter_path=local_adapter_path
        )
        
        # Activate both adapters for training
        dual_model.set_active_adapters(["global", "local"])
        
        # Print trainable parameters
        dual_model.print_trainable_parameters()
        
        return dual_model
    
    def _prepare_data(
        self,
        global_data_path: str,
        local_data_path: str,
        system_prompt: str = ""
    ):
        """
        Prepare training data.
        
        Args:
            global_data_path: Path to global data
            local_data_path: Path to local data
            system_prompt: System prompt for data formatting
            
        Returns:
            Training dataset
        """
        logging.info("Preparing training data...")
        
        dataset = create_mixed_dataset(
            global_data_path=global_data_path,
            local_data_path=local_data_path,
            tokenizer=self.tokenizer,
            max_length=self.config['training']['max_seq_length'],
            system_prompt=system_prompt
        )
        
        logging.info(f"✅ Prepared dataset with {len(dataset)} samples")
        
        return dataset
    
    def _train(
        self,
        dual_model: DualAdapterModel,
        dataset,
        output_dir: str
    ) -> None:
        """
        Execute training.
        
        Args:
            dual_model: Model to train
            dataset: Training dataset
            output_dir: Output directory
        """
        logging.info("Starting training...")
        
        # Enable gradient checkpointing if configured
        if self.config['training'].get('gradient_checkpointing', False):
            dual_model.get_model().enable_input_require_grads()
            dual_model.get_model().gradient_checkpointing_enable()
            logging.info("✅ Gradient checkpointing enabled")
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            logging_steps=self.config['training']['logging_steps'],
            save_strategy=self.config['training']['save_strategy'],
            bf16=self.config['training'].get('bf16', False),
            fp16=self.config['training'].get('fp16', False),
            optim=self.config['training'].get('optim', 'paged_adamw_8bit'),
            warmup_ratio=self.config['training'].get('warmup_ratio', 0.1),
            lr_scheduler_type=self.config['training'].get('lr_scheduler_type', 'cosine'),
            logging_dir=os.path.join(output_dir, 'logs'),
            report_to=[],  # Disable wandb/tensorboard for now
            remove_unused_columns=False,
            gradient_checkpointing=self.config['training'].get('gradient_checkpointing', False),
        )
        
        # Create trainer (use default data collator)
        trainer = Trainer(
            model=dual_model.get_model(),
            args=training_args,
            train_dataset=dataset,
        )
        
        # Train
        trainer.train()
        
        logging.info("✅ Training completed")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("✅ Cleared CUDA cache")


class StandardFedAvgClientTrainer(ClientTrainer):
    """
    Client trainer for standard FedAvg (single adapter).
    
    This is a baseline implementation where all parameters are aggregated.
    """
    
    def _setup_model(
        self,
        global_adapter_path: Optional[str],
        local_adapter_path: Optional[str]
    ) -> DualAdapterModel:
        """
        Setup model with single adapter for standard FedAvg.
        
        Args:
            global_adapter_path: Path to existing adapter
            local_adapter_path: Ignored for standard FedAvg
            
        Returns:
            DualAdapterModel with single adapter
        """
        # Load base model if not already loaded
        if self.base_model is None:
            logging.info("Loading base model...")
            self.base_model, self.tokenizer = load_base_model(
                model_name=self.config['model']['base_model'],
                quantization=self.config['model'].get('quantization', 'auto'),
                hf_token=self.config.get('huggingface', {}).get('token')
            )
            freeze_base_model(self.base_model)
        
        # Create LoRA config
        lora_config = get_lora_config(
            r=self.config['model']['lora_config']['r'],
            lora_alpha=self.config['model']['lora_config']['lora_alpha'],
            lora_dropout=self.config['model']['lora_config']['lora_dropout'],
            target_modules=self.config['model']['lora_config']['target_modules']
        )
        
        # Create model with single adapter
        dual_model = DualAdapterModel(self.base_model, lora_config)
        
        # Add only global adapter (which will be aggregated)
        dual_model.add_global_adapter(
            adapter_name="global",
            adapter_path=global_adapter_path
        )
        
        # Activate only global adapter
        dual_model.set_active_adapters(["global"])
        
        dual_model.print_trainable_parameters()
        
        logging.info("⚠️ Standard FedAvg mode: Using single adapter")
        
        return dual_model
    
    def train_round(
        self,
        round_num: int,
        global_adapter_path: Optional[str],
        local_adapter_path: Optional[str],
        global_data_path: str,
        local_data_path: str,
        output_dir: str,
        system_prompt: str = ""
    ) -> Tuple[str, str]:
        """Execute training round for standard FedAvg."""
        result = super().train_round(
            round_num,
            global_adapter_path,
            None,  # No local adapter
            global_data_path,
            local_data_path,
            output_dir,
            system_prompt
        )
        
        # Return same path for both (only global exists)
        return result[0], result[0]
