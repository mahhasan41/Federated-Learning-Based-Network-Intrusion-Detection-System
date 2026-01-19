"""
Federated Learning Training Pipeline
Orchestrates the complete federated learning workflow
"""

import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, Tuple, List
import time

from preprocessing import DataPreprocessor, distribute_data_to_clients
from federated_client import FederatedClient
from federated_server import FederatedServer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FederatedLearningPipeline:
    """Complete federated learning training pipeline"""
    
    def __init__(self, dataset_path: str, num_clients: int = 4, 
                 sample_fraction: float = 1.0, device: str = 'cpu',
                 results_dir: str = './results'):
        """
        Initialize pipeline
        
        Args:
            dataset_path: Path to dataset CSV
            num_clients: Number of simulated clients
            sample_fraction: Fraction of dataset to use
            device: 'cpu' or 'cuda'
            results_dir: Directory to save results
        """
        self.dataset_path = dataset_path
        self.num_clients = num_clients
        self.sample_fraction = sample_fraction
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Pipeline initialized: {num_clients} clients, sample={sample_fraction}")
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[Tuple], List, List]:
        """
        Load and preprocess dataset
        
        Returns:
            (X_test, y_test, client_data, feature_columns, class_names)
        """
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        preprocessor = DataPreprocessor(self.dataset_path)
        X_train, X_test, y_train, y_test, feature_cols, class_names = preprocessor.preprocess(
            sample_fraction=self.sample_fraction
        )
        
        # Distribute training data to clients
        client_data = distribute_data_to_clients(X_train, y_train, self.num_clients)
        
        return X_test, y_test, client_data, feature_cols, class_names
    
    def run_federated_training(self, X_test: np.ndarray, y_test: np.ndarray,
                              client_data: List[Tuple], num_rounds: int = 10,
                              num_epochs_per_round: int = 5) -> Dict:
        """
        Execute federated learning training
        
        Args:
            X_test: Test features
            y_test: Test labels
            client_data: List of (X, y) for each client
            num_rounds: Number of federated rounds
            num_epochs_per_round: Local epochs per round
            
        Returns:
            Training history and metrics
        """
        logger.info(f"Starting federated training: {num_rounds} rounds")
        
        # Initialize server
        input_size = client_data[0][0].shape[1]
        num_classes = len(np.unique(y_test))
        server = FederatedServer(input_size, num_classes, device=self.device)
        
        # Initialize clients
        clients = []
        for client_id, (X_client, y_client) in enumerate(client_data):
            client = FederatedClient(
                client_id=client_id,
                X_train=X_client,
                y_train=y_client,
                input_size=input_size,
                num_classes=num_classes,
                device=self.device
            )
            clients.append(client)
        
        # Training loop
        training_history = {
            'rounds': [],
            'global_metrics': [],
            'client_metrics': []
        }
        
        start_time = time.time()
        
        for round_num in range(num_rounds):
            logger.info(f"\n{'='*70}")
            logger.info(f"FEDERATED ROUND {round_num + 1}/{num_rounds}")
            logger.info(f"{'='*70}")
            
            # Execute federated round
            round_metrics = server.federated_round(clients, aggregation_method='fedavg')
            
            # Evaluate global model
            acc, loss, precision, f1 = server.evaluate_global(X_test, y_test)
            
            logger.info(f"\nGlobal Model Evaluation:")
            logger.info(f"  Accuracy:  {acc:.4f}")
            logger.info(f"  Loss:      {loss:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  F1-Score:  {f1:.4f}")
            
            # Record metrics
            training_history['rounds'].append(round_num + 1)
            training_history['global_metrics'].append({
                'round': round_num + 1,
                'accuracy': float(acc),
                'loss': float(loss),
                'precision': float(precision),
                'f1_score': float(f1)
            })
            training_history['client_metrics'].append(round_metrics['client_metrics'])
        
        elapsed_time = time.time() - start_time
        logger.info(f"\nTraining completed in {elapsed_time:.2f} seconds")
        
        return {
            'server': server,
            'clients': clients,
            'history': training_history,
            'input_size': input_size,
            'num_classes': num_classes,
            'training_time': elapsed_time
        }
    
    def save_results(self, results: Dict) -> None:
        """Save training results"""
        # Save training history
        history_file = self.results_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(results['history'], f, indent=2)
        logger.info(f"Training history saved to {history_file}")
        
        # Save global model
        model_file = self.results_dir / 'global_model.pt'
        results['server'].save_model(str(model_file))
        
        # Save metadata
        metadata = {
            'num_clients': self.num_clients,
            'sample_fraction': self.sample_fraction,
            'num_rounds': len(results['history']['rounds']),
            'training_time': results['training_time'],
            'input_size': results['input_size'],
            'num_classes': results['num_classes']
        }
        metadata_file = self.results_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_file}")


def main():
    """Main execution function"""
    # Configuration
    dataset_path = './data/combinenew.csv'
    num_clients = 4
    sample_fraction = 0.01  # Use 1% of data for testing
    num_rounds = 10
    results_dir = './results'
    
    logger.info("Federated Learning NIDS - Training Pipeline")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Clients: {num_clients}")
    logger.info(f"Sample: {sample_fraction*100}%")
    logger.info(f"Rounds: {num_rounds}")
    
    # Initialize pipeline
    pipeline = FederatedLearningPipeline(
        dataset_path=dataset_path,
        num_clients=num_clients,
        sample_fraction=sample_fraction,
        device='cpu',
        results_dir=results_dir
    )
    
    # Prepare data
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: DATA PREPARATION")
    logger.info("="*70)
    X_test, y_test, client_data, feature_cols, class_names = pipeline.prepare_data()
    logger.info(f"Test set shape: {X_test.shape}")
    logger.info(f"Classes: {class_names}")
    
    # Run federated training
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: FEDERATED TRAINING")
    logger.info("="*70)
    results = pipeline.run_federated_training(
        X_test=X_test,
        y_test=y_test,
        client_data=client_data,
        num_rounds=num_rounds
    )
    
    # Save results
    logger.info("\n" + "="*70)
    logger.info("PHASE 3: SAVING RESULTS")
    logger.info("="*70)
    pipeline.save_results(results)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"Results saved to: {pipeline.results_dir}")
    
    return results


if __name__ == "__main__":
    results = main()
