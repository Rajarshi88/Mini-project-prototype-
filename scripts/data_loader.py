"""
Data loading utilities for IoT datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import structlog
from pathlib import Path
import requests
import zipfile
import os
from sklearn.model_selection import train_test_split

logger = structlog.get_logger(__name__)


class IoTDataLoader:
    """
    Data loader for IoT cybersecurity datasets
    Supports CIC-IDS-2018, BoT-IoT-2019, CIC-IoT-2023
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'cic_ids_2018': {
                'url': 'https://www.unb.ca/cic/datasets/ids-2018.html',
                'files': ['Friday-02-03-2018_TrafficForML_CICFlowMeter.csv'],
                'label_column': 'Label',
                'attack_types': ['Benign', 'DDoS', 'DoS', 'Botnet', 'Brute Force', 'Infiltration']
            },
            'bot_iot_2019': {
                'url': 'https://research.unsw.edu.au/projects/bot-iot-dataset',
                'files': ['UNSW_2018_IoT_Botnet_Dataset_*.csv'],
                'label_column': 'category',
                'attack_types': ['Benign', 'DDoS', 'DoS', 'Reconnaissance', 'Theft', 'Keylogging']
            },
            'cic_iot_2023': {
                'url': 'https://www.unb.ca/cic/datasets/iot-2023.html',
                'files': ['*.csv'],
                'label_column': 'Label',
                'attack_types': ['Benign', 'DDoS', 'DoS', 'Mirai', 'Gafgyt', 'CICMalDroid']
            }
        }
    
    def load_dataset(self, dataset_name: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load a specific IoT dataset
        
        Args:
            dataset_name: Name of the dataset to load
            sample_size: Optional sample size for large datasets
            
        Returns:
            Loaded dataset as DataFrame
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_config = self.datasets[dataset_name]
        dataset_path = self.data_dir / dataset_name
        
        # Check if dataset exists
        if not dataset_path.exists():
            logger.warning(f"Dataset {dataset_name} not found. Please download it manually.")
            return self._create_sample_data(dataset_name)
        
        # Load dataset files
        dataframes = []
        for file_pattern in dataset_config['files']:
            file_paths = list(dataset_path.glob(file_pattern))
            
            for file_path in file_paths:
                try:
                    df = pd.read_csv(file_path)
                    dataframes.append(df)
                    logger.info(f"Loaded {file_path} with {len(df)} samples")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        if not dataframes:
            logger.warning(f"No valid files found for {dataset_name}")
            return self._create_sample_data(dataset_name)
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Sample data if requested
        if sample_size and len(combined_df) > sample_size:
            combined_df = combined_df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} records from {dataset_name}")
        
        logger.info(f"Successfully loaded {dataset_name} with {len(combined_df)} samples")
        return combined_df
    
    def _create_sample_data(self, dataset_name: str) -> pd.DataFrame:
        """Create sample data for testing purposes"""
        logger.info(f"Creating sample data for {dataset_name}")
        
        dataset_config = self.datasets[dataset_name]
        attack_types = dataset_config['attack_types']
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 10000
        
        # Create sample features
        data = {
            'src_ip': [f"192.168.1.{np.random.randint(1, 255)}" for _ in range(n_samples)],
            'dst_ip': [f"10.0.0.{np.random.randint(1, 255)}" for _ in range(n_samples)],
            'src_port': np.random.randint(1, 65535, n_samples),
            'dst_port': np.random.randint(1, 65535, n_samples),
            'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples),
            'packet_size': np.random.randint(64, 1500, n_samples),
            'flow_duration': np.random.exponential(1.0, n_samples),
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1S')
        }
        
        # Add more features for realistic data
        data['bytes_sent'] = np.random.randint(0, 10000, n_samples)
        data['bytes_received'] = np.random.randint(0, 10000, n_samples)
        data['packets_sent'] = np.random.randint(1, 100, n_samples)
        data['packets_received'] = np.random.randint(1, 100, n_samples)
        
        # Create labels with realistic distribution
        labels = np.random.choice(attack_types, n_samples, p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05])
        data[dataset_config['label_column']] = labels
        
        df = pd.DataFrame(data)
        logger.info(f"Created sample data with {len(df)} samples")
        return df
    
    def preprocess_dataset(self, df: pd.DataFrame, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Preprocess dataset for training
        
        Args:
            df: Raw dataset
            dataset_name: Name of the dataset
            
        Returns:
            Features, labels, and feature names
        """
        logger.info(f"Preprocessing dataset: {dataset_name}")
        
        dataset_config = self.datasets[dataset_name]
        label_column = dataset_config['label_column']
        
        # Handle missing values
        df = df.fillna(0)
        
        # Select numeric features
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if label_column in numeric_columns:
            numeric_columns.remove(label_column)
        
        # Select categorical features
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        if label_column in categorical_columns:
            categorical_columns.remove(label_column)
        
        # Encode categorical features
        for col in categorical_columns:
            if col in df.columns:
                df[col] = pd.Categorical(df[col]).codes
        
        # Prepare features and labels
        feature_columns = numeric_columns + categorical_columns
        X = df[feature_columns].values
        y = df[label_column].values
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        logger.info(f"Preprocessed dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Classes: {label_encoder.classes_}")
        
        return X, y_encoded, feature_columns
    
    def split_dataset(self, X: np.ndarray, y: np.ndarray, 
                     test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            X: Features
            y: Labels
            test_size: Test set size
            val_size: Validation set size
            
        Returns:
            Train, validation, and test sets
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Dataset split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, 
                        sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM models
        
        Args:
            X: Features
            y: Labels
            sequence_length: Length of sequences
            
        Returns:
            Sequences and corresponding labels
        """
        sequences = []
        labels = []
        
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i + sequence_length])
            labels.append(y[i + sequence_length - 1])  # Use last label in sequence
        
        return np.array(sequences), np.array(labels)
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return self.datasets[dataset_name].copy()
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets"""
        return list(self.datasets.keys())