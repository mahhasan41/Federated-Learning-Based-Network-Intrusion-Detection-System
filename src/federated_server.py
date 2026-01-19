"""
Federated Server Module
Aggregates client updates using FedAvg or similar algorithms
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class FederatedServer:
    """Federated learning server for aggregating client models"""
    
    def __init__(self, input_size: int, num_classes: int, device: str = 'cpu'):
        """
        Initialize federated server
        
        Args:
            input_size: Number of input features
            num_classes: Number of classes
            device: 'cpu' or 'cuda'
        """
        self.input_size = input_size
        self.num_classes = num_classes
        self.device = device
        
        # Initialize global model
        from federated_client import NeuralNetworkModel
        self.global_model = NeuralNetworkModel(input_size, num_classes)
        self.global_model.to(device)
        
        # History tracking
        self.global_loss_history = []
        self.round_number = 0
        self.client_weights_history = {}
        
        logger.info("Federated server initialized")
    
    def aggregate_weights_fedavg(self, client_weights_list: List[Dict], 
                                 data_sizes: List[int]) -> Dict:
        """
        Aggregate client weights using FedAvg algorithm
        
        FedAvg: w_global = sum(|D_i| / |D| * w_i) for each client i
        
        Args:
            client_weights_list: List of weight dictionaries from each client
            data_sizes: List of data sizes for each client
            
        Returns:
            Aggregated weights dictionary
        """
        total_samples = sum(data_sizes)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        
        # Get parameter names from first client
        for param_name in client_weights_list[0].keys():
            weighted_sum = np.zeros_like(client_weights_list[0][param_name])
            
            for client_idx, weights in enumerate(client_weights_list):
                # Weight by proportion of data
                weight_proportion = data_sizes[client_idx] / total_samples
                weighted_sum += weight_proportion * weights[param_name]
            
            aggregated_weights[param_name] = weighted_sum
        
        logger.info(f"Aggregated {len(client_weights_list)} client weights (FedAvg)")
        return aggregated_weights
    
    def aggregate_weights_trimmed_mean(self, client_weights_list: List[Dict], 
                                       data_sizes: List[int], 
                                       trim_fraction: float = 0.2) -> Dict:
        """
        Aggregate using trimmed mean (robust to outliers)
        
        Args:
            client_weights_list: List of weight dictionaries
            data_sizes: List of data sizes (not used in trimmed mean but kept for interface)
            trim_fraction: Fraction of weights to trim from each end
            
        Returns:
            Aggregated weights
        """
        aggregated_weights = {}
        
        for param_name in client_weights_list[0].keys():
            # Stack weights along new dimension
            weight_stack = np.array([w[param_name] for w in client_weights_list])
            
            # Sort and trim
            n_clients = len(client_weights_list)
            n_trim = max(1, int(n_clients * trim_fraction))
            
            # Compute trimmed mean
            trimmed_mean = np.sort(weight_stack, axis=0)
            trimmed_mean = trimmed_mean[n_trim:-n_trim].mean(axis=0)
            
            aggregated_weights[param_name] = trimmed_mean
        
        logger.info(f"Aggregated {len(client_weights_list)} client weights (Trimmed Mean)")
        return aggregated_weights
    
    def set_weights(self, weights: Dict) -> None:
        """
        Set global model weights
        
        Args:
            weights: Dictionary of weights
        """
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in weights:
                    param.copy_(torch.FloatTensor(weights[name]).to(self.device))
    
    def get_weights(self) -> Dict:
        """
        Get global model weights
        
        Returns:
            Dictionary of weights as numpy arrays
        """
        weights = {}
        for name, param in self.global_model.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        return weights
    
    def federated_round(self, clients: List, aggregation_method: str = 'fedavg') -> Dict:
        """
        Execute one federated learning round
        
        Process:
        1. Send global weights to clients
        2. Clients train locally
        3. Clients send updated weights to server
        4. Server aggregates weights
        5. Update global model
        
        Args:
            clients: List of FederatedClient objects
            aggregation_method: 'fedavg' or 'trimmed_mean'
            
        Returns:
            Dictionary with round metrics
        """
        self.round_number += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"Federated Round {self.round_number}")
        logger.info(f"{'='*60}")
        
        global_weights = self.get_weights()
        client_weights_list = []
        data_sizes = []
        round_metrics = {
            'round': self.round_number,
            'client_metrics': {}
        }
        
        # Phase 1: Send weights to clients and get updates
        for client in clients:
            # Send global weights
            client.set_weights(global_weights)
            
            # Local training
            client.train_local(epochs=5, batch_size=32, learning_rate=0.001)
            
            # Get updated weights
            client_weights = client.get_weights()
            client_weights_list.append(client_weights)
            
            # Get data size
            data_size = len(client.X_train)
            data_sizes.append(data_size)
            
            # Evaluate client
            client_acc = client.evaluate()
            round_metrics['client_metrics'][f'client_{client.client_id}'] = {
                'accuracy': float(client_acc),
                'data_size': data_size
            }
            
            logger.info(f"Client {client.client_id}: Local Accuracy = {client_acc:.4f}, Samples = {data_size}")
        
        # Phase 2: Aggregate weights
        if aggregation_method == 'fedavg':
            aggregated_weights = self.aggregate_weights_fedavg(client_weights_list, data_sizes)
        elif aggregation_method == 'trimmed_mean':
            aggregated_weights = self.aggregate_weights_trimmed_mean(client_weights_list, data_sizes)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Phase 3: Update global model
        self.set_weights(aggregated_weights)
        
        logger.info(f"Global model updated with aggregated weights")
        
        return round_metrics
    
    def evaluate_global(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Evaluate global model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            (accuracy, loss, precision, f1_score)
        """
        self.global_model.eval()
        
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.global_model(X_test_tensor)
            loss = nn.CrossEntropyLoss()(outputs, y_test_tensor).item()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # Calculate accuracy
        accuracy = (predictions == y_test).mean()
        
        # Calculate precision and F1 for binary classification
        tp = ((predictions == 1) & (y_test == 1)).sum()
        fp = ((predictions == 1) & (y_test == 0)).sum()
        fn = ((predictions == 0) & (y_test == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        return float(accuracy), float(loss), float(precision), float(f1)
    
    def save_model(self, filepath: str) -> None:
        """Save global model to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.global_model.state_dict(), filepath)
        logger.info(f"Global model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load global model from disk"""
        self.global_model.load_state_dict(torch.load(filepath, map_location=self.device))
        logger.info(f"Global model loaded from {filepath}")
    
    def save_training_history(self, filepath: str) -> None:
        """Save training history"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        history = {
            'round_number': self.round_number,
            'global_loss_history': self.global_loss_history,
            'client_weights_history': self.client_weights_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(history, f)
        logger.info(f"Training history saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    server = FederatedServer(input_size=50, num_classes=5, device='cpu')
    print(f"Server initialized with {server.input_size} input features and {server.num_classes} classes")
