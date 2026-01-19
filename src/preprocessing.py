"""
Data Preprocessing Module for Federated Learning NIDS
Handles loading, cleaning, encoding, and feature engineering for CIC-IDS 2017 dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocessing pipeline for intrusion detection data"""
    
    def __init__(self, dataset_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize preprocessor
        
        Args:
            dataset_path: Path to CSV file
            test_size: Fraction for train-test split
            random_state: Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.num_classes = None
        self.class_names = None
        
    def load_data(self, sample_fraction: float = 1.0) -> pd.DataFrame:
        """
        Load dataset from CSV
        
        Args:
            sample_fraction: Fraction of data to load (for memory efficiency)
            
        Returns:
            Loaded dataframe
        """
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        # Sample for memory efficiency if needed
        n_rows = None
        if sample_fraction < 1.0:
            total_rows = 2830743  # Approximate for CIC-IDS 2017
            n_rows = int(total_rows * sample_fraction)
            
        df = pd.read_csv(self.dataset_path, nrows=n_rows)
        logger.info(f"Dataset shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing and infinite values"""
        logger.info("Handling missing values...")
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with all NaN features (except label)
        df = df.dropna(how='all', subset=[col for col in df.columns if col != 'Label'])
        
        # Fill remaining NaN with median for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col].fillna(df[col].median(), inplace=True)
            
        logger.info(f"Missing values handled. Shape: {df.shape}")
        return df
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            Dataframe with encoded features
        """
        logger.info("Encoding categorical features...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove label column from encoding (will handle separately)
        if 'Label' in categorical_cols:
            categorical_cols.remove('Label')
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    try:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        # Handle unseen labels in inference
                        df[col] = 0
        
        return df
    
    def encode_labels(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode label column to numeric
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoder
            
        Returns:
            Dataframe with encoded labels
        """
        if 'Label' in df.columns:
            if fit:
                self.label_encoders['Label'] = LabelEncoder()
                df['Label'] = self.label_encoders['Label'].fit_transform(df['Label'].astype(str))
                self.class_names = self.label_encoders['Label'].classes_
                self.num_classes = len(self.class_names)
                logger.info(f"Classes: {self.class_names}")
            else:
                if 'Label' in self.label_encoders:
                    df['Label'] = self.label_encoders['Label'].transform(df['Label'].astype(str))
        
        return df
    
    def normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize numerical features using StandardScaler
        
        Args:
            X: Feature matrix
            fit: Whether to fit scaler
            
        Returns:
            Normalized feature matrix
        """
        logger.info("Normalizing features...")
        
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
            
        return X
    
    def select_features(self, df: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
        """
        Select top features based on correlation with label
        
        Args:
            df: Input dataframe with numeric features and label
            top_k: Number of top features to select
            
        Returns:
            Dataframe with selected features + label
        """
        logger.info(f"Selecting top {top_k} features...")
        
        # Calculate correlation with label
        correlations = df.corr()[['Label']].abs().sort_values('Label', ascending=False)
        
        # Select top k features
        top_features = correlations.index[1:top_k+1].tolist()  # Skip label itself
        top_features.append('Label')  # Add label back
        
        df = df[top_features]
        self.feature_columns = top_features[:-1]  # Store for later use
        
        logger.info(f"Selected features: {len(self.feature_columns)}")
        return df
    
    def handle_class_imbalance(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Handle class imbalance using SMOTE
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            Resampled (X, y)
        """
        logger.info(f"Original class distribution: {np.bincount(y.astype(int))}")
        
        # Apply SMOTE only if there's imbalance
        if len(np.bincount(y.astype(int))) > 1:
            # Calculate sampling strategy to avoid all minority classes
            class_dist = np.bincount(y.astype(int))
            if len(class_dist) > 1:
                max_class = class_dist.max()
                sampling_strategy = {i: max_class for i in range(len(class_dist))}
                
                smote = SMOTE(sampling_strategy='auto', random_state=self.random_state)
                try:
                    X, y = smote.fit_resample(X, y)
                    logger.info(f"After SMOTE: {np.bincount(y.astype(int))}")
                except ValueError as e:
                    logger.warning(f"SMOTE failed: {e}. Skipping resampling.")
        
        return X, y
    
    def preprocess(self, sample_fraction: float = 1.0) -> tuple:
        """
        Complete preprocessing pipeline
        
        Args:
            sample_fraction: Fraction of data to use
            
        Returns:
            (X_train, X_test, y_train, y_test, feature_columns, class_names)
        """
        # Load data
        df = self.load_data(sample_fraction=sample_fraction)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode features
        df = self.encode_features(df, fit=True)
        
        # Encode labels
        df = self.encode_labels(df, fit=True)
        
        # Select top features for better performance
        df = self.select_features(df, top_k=50)
        
        # Separate features and labels
        X = df.drop('Label', axis=1).values
        y = df['Label'].values
        
        # Normalize features
        X = self.normalize_features(X, fit=True)
        
        # Handle class imbalance
        X, y = self.handle_class_imbalance(X, y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Final shapes - Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Train labels: {np.bincount(y_train.astype(int))}")
        logger.info(f"Test labels: {np.bincount(y_test.astype(int))}")
        
        return X_train, X_test, y_train, y_test, self.feature_columns, self.class_names
    
    def preprocess_inference_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess new data for inference
        
        Args:
            df: Input dataframe (same structure as training data)
            
        Returns:
            Preprocessed feature matrix
        """
        # Encode features
        df = self.encode_features(df, fit=False)
        
        # Select only the training features
        if self.feature_columns:
            X = df[self.feature_columns].values
        else:
            X = df.values
        
        # Normalize using fitted scaler
        X = self.normalize_features(X, fit=False)
        
        return X


def distribute_data_to_clients(X_train: np.ndarray, y_train: np.ndarray, 
                               num_clients: int = 4) -> list:
    """
    Distribute training data across simulated clients (IID split)
    
    Args:
        X_train: Training features
        y_train: Training labels
        num_clients: Number of simulated clients
        
    Returns:
        List of (X, y) tuples for each client
    """
    logger.info(f"Distributing data to {num_clients} clients (IID)...")
    
    # Split indices
    n_samples = len(X_train)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    client_data = []
    split_size = n_samples // num_clients
    
    for i in range(num_clients):
        start_idx = i * split_size
        if i == num_clients - 1:
            end_idx = n_samples  # Last client gets remainder
        else:
            end_idx = (i + 1) * split_size
        
        client_indices = indices[start_idx:end_idx]
        X_client = X_train[client_indices]
        y_client = y_train[client_indices]
        
        client_data.append((X_client, y_client))
        logger.info(f"Client {i}: {X_client.shape} - Labels: {np.bincount(y_client.astype(int))}")
    
    return client_data


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor("./data/combinenew.csv", test_size=0.2)
    
    # Preprocess with smaller sample for testing
    X_train, X_test, y_train, y_test, features, classes = preprocessor.preprocess(sample_fraction=0.1)
    
    # Distribute to clients
    client_data = distribute_data_to_clients(X_train, y_train, num_clients=4)
