"""
Federated learning server for aggregation and coordination.

This module implements the server-side logic for federated learning,
including adapter aggregation and distribution.
"""

import os
import logging
import shutil
from typing import List, Dict, Optional

from src.federated.aggregators import fedavg


class FederatedServer:
    """
    Federated learning server.
    
    Manages global model state and coordinates federated learning rounds.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize federated server.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.global_adapter_history: List[str] = []
        
        logging.info("Initialized FederatedServer")
    
    def aggregate_global_adapters(
        self,
        adapter_paths: List[str],
        output_path: str,
        method: str = "fedavg",
        weights: Optional[List[float]] = None
    ) -> str:
        """
        Aggregate global adapters from clients.
        
        Args:
            adapter_paths: List of paths to client global adapters
            output_path: Path where to save aggregated adapter
            method: Aggregation method (currently only "fedavg")
            weights: Optional client weights (default: equal weights)
            
        Returns:
            Path to aggregated adapter
        """
        logging.info(f"Aggregating {len(adapter_paths)} global adapters using {method}")
        
        if method != "fedavg":
            raise ValueError(f"Unsupported aggregation method: {method}")
        
        # Perform FedAvg aggregation
        aggregated_state_dict = fedavg(adapter_paths, weights)
        
        # Save aggregated adapter
        os.makedirs(output_path, exist_ok=True)
        
        # Copy adapter config from first client
        import json
        config_path = os.path.join(adapter_paths[0], "adapter_config.json")
        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(output_path, "adapter_config.json"))
        
        # Save aggregated weights
        import torch
        weights_path = os.path.join(output_path, "adapter_model.bin")
        torch.save(aggregated_state_dict, weights_path)
        
        logging.info(f"✅ Saved aggregated global adapter to {output_path}")
        
        # Track history
        self.global_adapter_history.append(output_path)
        
        return output_path
    
    def distribute_global_adapter(
        self,
        adapter_path: str,
        client_ids: List[str]
    ) -> Dict[str, str]:
        """
        Distribute global adapter to clients.
        
        In our serial simulation, this just returns the same path for all clients.
        In a real distributed system, this would copy/send the adapter to each client.
        
        Args:
            adapter_path: Path to global adapter
            client_ids: List of client IDs
            
        Returns:
            Dictionary mapping client_id to adapter path
        """
        distribution = {client_id: adapter_path for client_id in client_ids}
        
        logging.info(f"✅ Distributed global adapter to {len(client_ids)} clients")
        
        return distribution
    
    def get_global_adapter_history(self) -> List[str]:
        """
        Get history of global adapter paths.
        
        Returns:
            List of paths to global adapters from each round
        """
        return self.global_adapter_history
    
    def save_checkpoint_manifest(
        self,
        round_num: int,
        global_adapter_path: str,
        client_adapters: Dict[str, Dict[str, str]],
        manifest_path: str
    ) -> None:
        """
        Save checkpoint manifest for a round.
        
        Args:
            round_num: Round number
            global_adapter_path: Path to aggregated global adapter
            client_adapters: Dict of client_id -> {global, local} paths
            manifest_path: Path to save manifest
        """
        import json
        from datetime import datetime
        
        manifest = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "global_adapter": global_adapter_path,
            "clients": client_adapters
        }
        
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logging.info(f"✅ Saved checkpoint manifest to {manifest_path}")
