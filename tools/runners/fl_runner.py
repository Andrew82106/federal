"""
Federated learning runner for orchestrating complete experiments.

This module provides the main orchestration logic for running
federated learning experiments with dual-adapter architecture.
"""

import os
import logging
from typing import Dict, List, Optional
import json

from src.federated.client import ClientTrainer, StandardFedAvgClientTrainer
from src.federated.server import FederatedServer


class FLRunner:
    """
    Federated learning experiment runner.
    
    Orchestrates the complete federated learning workflow including
    client training, server aggregation, and checkpoint management.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize FL runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.server = FederatedServer(config)
        self.clients: Dict[str, ClientTrainer] = {}
        self.mode = config['federated'].get('mode', 'dual_adapter')
        
        # Initialize clients
        self._initialize_clients()
        
        logging.info(f"Initialized FLRunner in '{self.mode}' mode")
        logging.info(f"Clients: {list(self.clients.keys())}")
    
    def _initialize_clients(self) -> None:
        """Initialize client trainers based on configuration."""
        client_configs = self.config['federated']['clients']
        
        for client_config in client_configs:
            client_id = client_config['id']
            
            # Choose trainer type based on mode
            if self.mode == 'standard_fedavg':
                trainer = StandardFedAvgClientTrainer(client_id, self.config)
            else:
                trainer = ClientTrainer(client_id, self.config)
            
            self.clients[client_id] = trainer
            logging.info(f"✅ Initialized client '{client_id}'")
    
    def run_experiment(self) -> Dict:
        """
        Run complete federated learning experiment.
        
        Returns:
            Dictionary with experiment results
        """
        import sys
        
        print("\n" + "=" * 80, flush=True)
        print("🚀🚀🚀 run_experiment() METHOD STARTED 🚀🚀🚀", flush=True)
        print("=" * 80 + "\n", flush=True)
        
        logging.info("🚀 run_experiment() method called!")
        sys.stdout.flush()
        sys.stderr.flush()
        
        num_rounds = self.config['federated']['num_rounds']
        output_dir = self.config['experiment']['output_dir']
        
        logging.info("=" * 80)
        logging.info(f"Starting Federated Learning Experiment: {self.config['experiment']['name']}")
        logging.info(f"Mode: {self.mode}")
        logging.info(f"Rounds: {num_rounds}")
        logging.info(f"Clients: {len(self.clients)}")
        logging.info("=" * 80)
        sys.stdout.flush()  # Force flush
        sys.stderr.flush()
        
        results = {
            'rounds': [],
            'final_adapters': {}
        }
        
        # Run federated learning rounds
        # Check for existing checkpoints to resume from
        start_round = 1
        checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        logging.info(f"Checking for existing checkpoints in: {checkpoints_dir}")
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        if os.path.exists(checkpoints_dir):
            existing_rounds = []
            items = os.listdir(checkpoints_dir)
            logging.info(f"Found items in checkpoints dir: {items}")
            
            for item in items:
                if item.startswith('round_') and os.path.isdir(os.path.join(checkpoints_dir, item)):
                    try:
                        round_num = int(item.split('_')[1])
                        # Check if this round is complete (has manifest.json)
                        manifest_path = os.path.join(checkpoints_dir, item, 'manifest.json')
                        logging.info(f"Checking {item}: manifest exists = {os.path.exists(manifest_path)}")
                        if os.path.exists(manifest_path):
                            existing_rounds.append(round_num)
                    except (ValueError, IndexError):
                        continue
            
            logging.info(f"Existing completed rounds: {existing_rounds}")
            
            if existing_rounds:
                start_round = max(existing_rounds) + 1
                logging.info(f"📂 Found existing checkpoints up to Round {max(existing_rounds)}")
                logging.info(f"🔄 Resuming from Round {start_round}")
            else:
                logging.info("No completed rounds found, starting from Round 1")
        else:
            logging.info("No checkpoints directory found, starting from Round 1")
        
        for round_num in range(start_round, num_rounds + 1):
            round_result = self.run_round(round_num)
            results['rounds'].append(round_result)
        
        # Save final adapters
        final_dir = os.path.join(output_dir, 'checkpoints', 'final_adapters')
        results['final_adapters'] = self._save_final_adapters(final_dir)
        
        # Save experiment summary
        summary_path = os.path.join(output_dir, 'experiment_summary.json')
        self._save_experiment_summary(results, summary_path)
        
        logging.info("=" * 80)
        logging.info("✅ Federated Learning Experiment Completed!")
        logging.info(f"Results saved to: {output_dir}")
        logging.info("=" * 80)
        
        return results
    
    def run_round(self, round_num: int) -> Dict:
        """
        Run a single federated learning round.
        
        Args:
            round_num: Round number (1-indexed)
            
        Returns:
            Dictionary with round results
        """
        logging.info("\n" + "=" * 80)
        logging.info(f"ROUND {round_num}/{self.config['federated']['num_rounds']}")
        logging.info("=" * 80)
        
        output_dir = self.config['experiment']['output_dir']
        round_dir = os.path.join(output_dir, 'checkpoints', f'round_{round_num}')
        
        # Get global adapter path from previous round
        global_adapter_path = None
        if round_num > 1:
            prev_round_dir = os.path.join(output_dir, 'checkpoints', f'round_{round_num - 1}')
            global_adapter_path = os.path.join(prev_round_dir, 'global_adapter')
        
        # Train all clients
        client_adapters = self._train_clients(
            round_num,
            global_adapter_path,
            round_dir
        )
        
        # Aggregate global adapters
        aggregated_global_path = self._aggregate_global(
            round_num,
            client_adapters,
            round_dir
        )
        
        # Save checkpoint manifest
        manifest_path = os.path.join(round_dir, 'manifest.json')
        self.server.save_checkpoint_manifest(
            round_num,
            aggregated_global_path,
            client_adapters,
            manifest_path
        )
        
        logging.info(f"✅ Round {round_num} completed")
        
        return {
            'round': round_num,
            'global_adapter': aggregated_global_path,
            'client_adapters': client_adapters
        }
    
    def _train_clients(
        self,
        round_num: int,
        global_adapter_path: Optional[str],
        round_dir: str
    ) -> Dict[str, Dict[str, str]]:
        """
        Train all clients for current round.
        
        Args:
            round_num: Current round number
            global_adapter_path: Path to global adapter from previous round
            round_dir: Directory for this round's outputs
            
        Returns:
            Dictionary mapping client_id to adapter paths
        """
        client_adapters = {}
        client_configs = self.config['federated']['clients']
        
        for client_config in client_configs:
            client_id = client_config['id']
            trainer = self.clients[client_id]
            
            # Get local adapter path from previous round
            local_adapter_path = None
            if round_num > 1:
                prev_round_dir = os.path.join(
                    self.config['experiment']['output_dir'],
                    'checkpoints',
                    f'round_{round_num - 1}',
                    f'client_{client_id}'
                )
                local_adapter_path = os.path.join(prev_round_dir, 'local_adapter')
            
            # Client output directory
            client_output_dir = os.path.join(round_dir, f'client_{client_id}')
            
            # Train client
            global_path, local_path = trainer.train_round(
                round_num=round_num,
                global_adapter_path=global_adapter_path,
                local_adapter_path=local_adapter_path,
                global_data_path=self.config['data']['global_train'],
                local_data_path=client_config['local_data'],
                output_dir=client_output_dir,
                system_prompt=client_config.get('system_prompt', '')
            )
            
            client_adapters[client_id] = {
                'global': global_path,
                'local': local_path
            }
            
            # Force garbage collection and clear CUDA cache after each client
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logging.info(f"✅ Cleared CUDA cache after client '{client_id}'")
        
        return client_adapters
    
    def _aggregate_global(
        self,
        round_num: int,
        client_adapters: Dict[str, Dict[str, str]],
        round_dir: str
    ) -> str:
        """
        Aggregate global adapters from clients.
        
        Args:
            round_num: Current round number
            client_adapters: Client adapter paths
            round_dir: Directory for this round
            
        Returns:
            Path to aggregated global adapter
        """
        # Extract global adapter paths
        global_adapter_paths = [
            adapters['global'] for adapters in client_adapters.values()
        ]
        
        # Aggregate
        aggregated_path = os.path.join(round_dir, 'global_adapter')
        self.server.aggregate_global_adapters(
            adapter_paths=global_adapter_paths,
            output_path=aggregated_path,
            method=self.config['federated']['aggregation_method']
        )
        
        return aggregated_path
    
    def _save_final_adapters(self, final_dir: str) -> Dict[str, str]:
        """
        Save final adapters after all rounds.
        
        Args:
            final_dir: Directory for final adapters
            
        Returns:
            Dictionary with paths to final adapters
        """
        import shutil
        
        last_round = self.config['federated']['num_rounds']
        last_round_dir = os.path.join(
            self.config['experiment']['output_dir'],
            'checkpoints',
            f'round_{last_round}'
        )
        
        final_adapters = {}
        
        # Copy global adapter
        global_src = os.path.join(last_round_dir, 'global_adapter')
        global_dst = os.path.join(final_dir, 'global')
        if os.path.exists(global_src):
            shutil.copytree(global_src, global_dst, dirs_exist_ok=True)
            final_adapters['global'] = global_dst
            logging.info(f"✅ Saved final global adapter to {global_dst}")
        
        # Copy local adapters for each client
        for client_id in self.clients.keys():
            local_src = os.path.join(last_round_dir, f'client_{client_id}', 'local_adapter')
            local_dst = os.path.join(final_dir, client_id)
            if os.path.exists(local_src):
                shutil.copytree(local_src, local_dst, dirs_exist_ok=True)
                final_adapters[client_id] = local_dst
                logging.info(f"✅ Saved final local adapter for '{client_id}' to {local_dst}")
        
        return final_adapters
    
    def _save_experiment_summary(self, results: Dict, summary_path: str) -> None:
        """
        Save experiment summary to JSON.
        
        Args:
            results: Experiment results
            summary_path: Path to save summary
        """
        summary = {
            'experiment_name': self.config['experiment']['name'],
            'mode': self.mode,
            'num_rounds': self.config['federated']['num_rounds'],
            'num_clients': len(self.clients),
            'clients': list(self.clients.keys()),
            'results': results
        }
        
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logging.info(f"✅ Saved experiment summary to {summary_path}")
