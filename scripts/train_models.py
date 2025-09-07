"""
Training scripts for CyberAIBot models
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import structlog
from pathlib import Path
import json
import pickle
from datetime import datetime

from .data_loader import IoTDataLoader
from ..preprocessing.data_processor import DataProcessor
from ..core.technical_cluster import TechnicalCluster

logger = structlog.get_logger(__name__)


class ModelTrainer:
    """
    Model trainer for CyberAIBot technical clusters
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_loader = IoTDataLoader()
        self.data_processor = DataProcessor(config['data_processing'])
        self.training_results = {}
        
    async def train_all_models(self, cyberai_bot) -> Dict[str, Any]:
        """
        Train all technical clusters
        
        Args:
            cyberai_bot: CyberAIBot system instance
            
        Returns:
            Training results
        """
        logger.info("Starting training for all models")
        
        try:
            # Initialize data processor
            await self.data_processor.initialize()
            
            # Load and preprocess datasets
            datasets = self._load_datasets()
            
            # Train each cluster
            for cluster_id, cluster in cyberai_bot.technical_clusters.items():
                logger.info(f"Training cluster: {cluster_id}")
                
                # Get training data for this cluster's attack type
                training_data = self._prepare_training_data(
                    cluster.attack_type, datasets
                )
                
                if training_data is not None:
                    # Train the cluster
                    result = await self._train_cluster(cluster, training_data)
                    self.training_results[cluster_id] = result
                else:
                    logger.warning(f"No training data available for {cluster_id}")
            
            # Save training results
            self._save_training_results()
            
            logger.info("Training completed for all models")
            return self.training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all available datasets"""
        datasets = {}
        
        for dataset_name in self.data_loader.list_available_datasets():
            try:
                df = self.data_loader.load_dataset(dataset_name, sample_size=50000)
                datasets[dataset_name] = df
                logger.info(f"Loaded {dataset_name}: {len(df)} samples")
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")
        
        return datasets
    
    def _prepare_training_data(self, attack_type: str, datasets: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        Prepare training data for a specific attack type
        
        Args:
            attack_type: Type of attack to train for
            datasets: Available datasets
            
        Returns:
            Prepared training data or None
        """
        logger.info(f"Preparing training data for {attack_type}")
        
        # Combine relevant datasets
        combined_data = []
        
        for dataset_name, df in datasets.items():
            dataset_config = self.data_loader.get_dataset_info(dataset_name)
            label_column = dataset_config['label_column']
            
            # Filter data for this attack type
            if attack_type in df[label_column].values:
                attack_data = df[df[label_column] == attack_type].copy()
                combined_data.append(attack_data)
                
                # Also include benign data for binary classification
                if 'Benign' in df[label_column].values:
                    benign_data = df[df[label_column] == 'Benign'].sample(
                        n=min(len(attack_data), 10000), random_state=42
                    )
                    combined_data.append(benign_data)
        
        if not combined_data:
            logger.warning(f"No data found for attack type: {attack_type}")
            return None
        
        # Combine all data
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Preprocess the data
        X, y, feature_names = self.data_loader.preprocess_dataset(
            combined_df, 'cic_ids_2018'  # Use first available dataset config
        )
        
        # Split the data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_dataset(X, y)
        
        # Create sequences for LSTM
        X_train_seq, y_train_seq = self.data_loader.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.data_loader.create_sequences(X_val, y_val)
        X_test_seq, y_test_seq = self.data_loader.create_sequences(X_test, y_test)
        
        # Get class names
        class_names = ['Benign', attack_type]
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'X_train_seq': X_train_seq,
            'X_val_seq': X_val_seq,
            'X_test_seq': X_test_seq,
            'y_train_seq': y_train_seq,
            'y_val_seq': y_val_seq,
            'y_test_seq': y_test_seq,
            'class_names': class_names,
            'feature_names': feature_names
        }
    
    async def _train_cluster(self, cluster: TechnicalCluster, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a specific cluster
        
        Args:
            cluster: Technical cluster to train
            training_data: Training data
            
        Returns:
            Training results
        """
        logger.info(f"Training cluster: {cluster.cluster_id}")
        
        try:
            # Choose appropriate data based on model type
            if cluster.model_type == 'lstm':
                X_train = training_data['X_train_seq']
                y_train = training_data['y_train_seq']
                X_val = training_data['X_val_seq']
                y_val = training_data['y_val_seq']
            else:  # SVM
                X_train = training_data['X_train']
                y_train = training_data['y_train']
                X_val = training_data['X_val']
                y_val = training_data['y_val']
            
            # Train the cluster
            training_result = await cluster.train(
                X_train, y_train, X_val, y_val, training_data['class_names']
            )
            
            # Evaluate the model
            if cluster.model_type == 'lstm':
                X_test = training_data['X_test_seq']
                y_test = training_data['y_test_seq']
            else:
                X_test = training_data['X_test']
                y_test = training_data['y_test']
            
            evaluation_result = await cluster.evaluate(X_test, y_test)
            
            # Save the trained model
            model_path = f"models/{cluster.cluster_id}_model"
            Path("models").mkdir(exist_ok=True)
            cluster.save_model(model_path)
            
            result = {
                'cluster_id': cluster.cluster_id,
                'attack_type': cluster.attack_type,
                'model_type': cluster.model_type,
                'training_result': training_result,
                'evaluation_result': evaluation_result,
                'model_path': model_path,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully trained cluster: {cluster.cluster_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to train cluster {cluster.cluster_id}: {e}")
            return {
                'cluster_id': cluster.cluster_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _save_training_results(self):
        """Save training results to file"""
        results_path = "models/training_results.json"
        Path("models").mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)
        
        logger.info(f"Training results saved to {results_path}")


async def train_all_models(cyberai_bot) -> Dict[str, Any]:
    """
    Train all models in the CyberAIBot system
    
    Args:
        cyberai_bot: CyberAIBot system instance
        
    Returns:
        Training results
    """
    trainer = ModelTrainer(cyberai_bot.config)
    return await trainer.train_all_models(cyberai_bot)


async def train_specific_cluster(cyberai_bot, cluster_id: str, 
                               training_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a specific cluster
    
    Args:
        cyberai_bot: CyberAIBot system instance
        cluster_id: ID of the cluster to train
        training_data: Training data
        
    Returns:
        Training results
    """
    if cluster_id not in cyberai_bot.technical_clusters:
        raise ValueError(f"Cluster {cluster_id} not found")
    
    cluster = cyberai_bot.technical_clusters[cluster_id]
    trainer = ModelTrainer(cyberai_bot.config)
    
    return await trainer._train_cluster(cluster, training_data)


async def retrain_cluster_with_new_data(cyberai_bot, cluster_id: str, 
                                      new_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrain a cluster with new data
    
    Args:
        cyberai_bot: CyberAIBot system instance
        cluster_id: ID of the cluster to retrain
        new_data: New training data
        
    Returns:
        Training results
    """
    if cluster_id not in cyberai_bot.technical_clusters:
        raise ValueError(f"Cluster {cluster_id} not found")
    
    cluster = cyberai_bot.technical_clusters[cluster_id]
    trainer = ModelTrainer(cyberai_bot.config)
    
    # Load existing model if available
    try:
        model_path = f"models/{cluster_id}_model"
        if Path(model_path).exists():
            cluster.model.load_model(model_path)
    except Exception as e:
        logger.warning(f"Could not load existing model for {cluster_id}: {e}")
    
    return await trainer._train_cluster(cluster, new_data)