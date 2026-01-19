"""
Federated Client Module
Each client trains locally and sends updates to the server
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class NeuralNetworkModel(nn.Module):
    """Multi-layer perceptron for intrusion detection"""
    
    def __init__(self, input_size: int, num_classes: int, hidden_layers: list = None):
        """
        Initialize neural network
        
        Args:
            input_size: Number of input features
            num_classes: Number of output classes
            hidden_layers: List of hidden layer sizes
        """
        super(NeuralNetworkModel, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [256, 128, 64]
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class FederatedClient:
    """Simulates a federated learning client"""
    
    def __init__(self, client_id: int, X_train: np.ndarray, y_train: np.ndarray,
                 input_size: int, num_classes: int, device: str = 'cpu'):
        """
        Initialize federated client
        
        Args:
            client_id: Unique client identifier
            X_train: Training features
            y_train: Training labels
            input_size: Number of input features
            num_classes: Number of classes
            device: 'cpu' or 'cuda'
        """
        self.client_id = client_id
        self.X_train = torch.FloatTensor(X_train).to(device)
        self.y_train = torch.LongTensor(y_train).to(device)
        self.device = device
        
        # Initialize model
        self.model = NeuralNetworkModel(input_size, num_classes)
        self.model.to(device)
        
        # Training configuration
        self.learning_rate = 0.001
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        
        # Metrics
        self.local_loss_history = []
        self.local_accuracy_history = []
        
        logger.info(f"Client {self.client_id} initialized with {len(self.X_train)} samples")
    
    def set_weights(self, weights: Dict) -> None:
        """
        Set model weights from server (for aggregated model)
        
        Args:
            weights: Dictionary of model weights
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.copy_(torch.FloatTensor(weights[name]).to(self.device))
    
    def get_weights(self) -> Dict:
        """
        Get current model weights
        
        Returns:
            Dictionary of model weights as numpy arrays
        """
        weights = {}
        for name, param in self.model.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        return weights
    
    def train_epoch(self, batch_size: int = 32, learning_rate: float = 0.001) -> float:
        """
        Train model for one epoch
        
        Args:
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        n_samples = len(self.X_train)
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        indices = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = self.X_train[batch_indices]
            y_batch = self.y_train[batch_indices]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.local_loss_history.append(avg_loss)
        
        return avg_loss
    
    def train_local(self, epochs: int = 10, batch_size: int = 32, 
                   learning_rate: float = 0.001) -> None:
        """
        Train model locally for multiple epochs
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        logger.info(f"Client {self.client_id} starting local training for {epochs} epochs")
        
        for epoch in range(epochs):
            avg_loss = self.train_epoch(batch_size, learning_rate)
            
            if (epoch + 1) % 5 == 0:
                acc = self.evaluate()
                logger.info(f"Client {self.client_id} - Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
                self.local_accuracy_history.append(acc)
    
    def evaluate(self) -> float:
        """
        Evaluate model on local training data
        
        Returns:
            Accuracy score
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_train)
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == self.y_train).sum().item()
            accuracy = correct / len(self.y_train)
        
        return accuracy
    
    def get_loss_history(self) -> list:
        """Get local training loss history"""
        return self.local_loss_history
    
    def get_accuracy_history(self) -> list:
        """Get local evaluation accuracy history"""
        return self.local_accuracy_history


class CNNModel(nn.Module):
    """Convolutional Neural Network for sequence features"""
    
    def __init__(self, input_size: int, num_classes: int):
        """
        Initialize CNN model
        
        Args:
            input_size: Number of input features
            num_classes: Number of classes
        """
        super(CNNModel, self).__init__()
        
        # Reshape input to (batch, 1, input_size) for 1D conv
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Calculate flattened size
        flattened_size = (input_size // 4) * 64
        
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x shape: (batch, input_size)
        x = x.unsqueeze(1)  # (batch, 1, input_size)
        
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Example usage
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create dummy data
    X = np.random.randn(1000, 50).astype(np.float32)
    y = np.random.randint(0, 5, 1000)
    
    # Create client
    client = FederatedClient(
        client_id=1,
        X_train=X,
        y_train=y,
        input_size=50,
        num_classes=5,
        device='cpu'
    )
    
    # Train locally
    client.train_local(epochs=10, batch_size=32)
    
    # Get metrics
    print(f"Final accuracy: {client.evaluate():.4f}")
