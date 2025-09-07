"""
Technical cluster implementation for specialized attack detection
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
import structlog
from pathlib import Path

from ..models.lstm_model import LSTMModel
from ..models.svm_model import SVMModel

logger = structlog.get_logger(__name__)


class TechnicalCluster:
    """
    Technical cluster specialized in detecting a specific attack type
    Can use either LSTM or SVM models
    """
    
    def __init__(self, cluster_id: str, attack_type: str, model_type: str, config: Dict[str, Any]):
        self.cluster_id = cluster_id
        self.attack_type = attack_type
        self.model_type = model_type
        self.config = config
        self.model = None
        self.is_initialized = False
        self.is_running = False
        
        # Performance metrics
        self.prediction_count = 0
        self.avg_prediction_time = 0.0
        self.last_prediction_time = None
        
    async def initialize(self):
        """Initialize the technical cluster"""
        logger.info(f"Initializing technical cluster: {self.cluster_id}")
        
        try:
            # Create model based on type
            if self.model_type == 'lstm':
                self.model = LSTMModel(self.attack_type, self.config)
            elif self.model_type == 'svm':
                self.model = SVMModel(self.attack_type, self.config)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.is_initialized = True
            logger.info(f"Technical cluster initialized: {self.cluster_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize cluster {self.cluster_id}: {e}")
            raise
    
    async def start(self):
        """Start the technical cluster"""
        if not self.is_initialized:
            raise RuntimeError(f"Cluster {self.cluster_id} not initialized")
        
        if self.is_running:
            logger.warning(f"Cluster {self.cluster_id} is already running")
            return
        
        self.is_running = True
        logger.info(f"Technical cluster started: {self.cluster_id}")
    
    async def stop(self):
        """Stop the technical cluster"""
        if not self.is_running:
            logger.warning(f"Cluster {self.cluster_id} is not running")
            return
        
        self.is_running = False
        logger.info(f"Technical cluster stopped: {self.cluster_id}")
    
    async def predict(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction on processed data
        
        Args:
            processed_data: Preprocessed IoT traffic data
            
        Returns:
            Prediction results with metadata
        """
        if not self.is_running:
            raise RuntimeError(f"Cluster {self.cluster_id} is not running")
        
        if not self.model or not self.model.is_trained:
            # Return default prediction if model not trained
            return {
                'cluster_id': self.cluster_id,
                'attack_type': self.attack_type,
                'model_type': self.model_type,
                'prediction': 'Unknown',
                'confidence': 0.0,
                'class_probabilities': {},
                'is_trained': False,
                'prediction_time': 0.0
            }
        
        try:
            import time
            start_time = time.time()
            
            # Get appropriate input data based on model type
            if self.model_type == 'lstm':
                input_data = processed_data['sequences']
            else:  # SVM
                input_data = processed_data['features']
            
            # Make prediction
            prediction_result = self.model.predict(input_data)
            
            # Calculate prediction time
            prediction_time = time.time() - start_time
            
            # Update performance metrics
            self.prediction_count += 1
            self.avg_prediction_time = (
                (self.avg_prediction_time * (self.prediction_count - 1) + prediction_time) 
                / self.prediction_count
            )
            self.last_prediction_time = prediction_time
            
            # Format result
            result = {
                'cluster_id': self.cluster_id,
                'attack_type': self.attack_type,
                'model_type': self.model_type,
                'prediction': prediction_result['class_names'][0],
                'confidence': float(prediction_result['confidence_scores'][0]),
                'class_probabilities': {
                    name: float(prob) 
                    for name, prob in zip(
                        prediction_result['class_names'], 
                        prediction_result['class_probabilities'][0]
                    )
                },
                'is_trained': True,
                'prediction_time': prediction_time,
                'metadata': processed_data.get('metadata', {})
            }
            
            logger.debug(
                f"Prediction made by {self.cluster_id}",
                prediction=result['prediction'],
                confidence=result['confidence'],
                prediction_time=prediction_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction by cluster {self.cluster_id}: {e}")
            
            # Return error prediction
            return {
                'cluster_id': self.cluster_id,
                'attack_type': self.attack_type,
                'model_type': self.model_type,
                'prediction': 'Error',
                'confidence': 0.0,
                'class_probabilities': {},
                'is_trained': self.model.is_trained if self.model else False,
                'prediction_time': 0.0,
                'error': str(e)
            }
    
    async def train(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   class_names: List[str], **kwargs) -> Dict[str, Any]:
        """
        Train the cluster's model
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            class_names: List of class names
            **kwargs: Additional training parameters
            
        Returns:
            Training results
        """
        if not self.is_initialized:
            raise RuntimeError(f"Cluster {self.cluster_id} not initialized")
        
        logger.info(f"Training cluster {self.cluster_id}")
        
        try:
            # Train the model
            training_results = self.model.train(
                X_train, y_train, X_val, y_val, class_names, **kwargs
            )
            
            logger.info(f"Cluster {self.cluster_id} training completed")
            return training_results
            
        except Exception as e:
            logger.error(f"Error training cluster {self.cluster_id}: {e}")
            raise
    
    async def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the cluster's model
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Evaluation results
        """
        if not self.model or not self.model.is_trained:
            raise RuntimeError(f"Cluster {self.cluster_id} model not trained")
        
        logger.info(f"Evaluating cluster {self.cluster_id}")
        
        try:
            evaluation_results = self.model.evaluate(X_test, y_test)
            logger.info(f"Cluster {self.cluster_id} evaluation completed")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating cluster {self.cluster_id}: {e}")
            raise
    
    async def load_model(self, model_data: bytes):
        """
        Load a pre-trained model
        
        Args:
            model_data: Serialized model data
        """
        if not self.is_initialized:
            raise RuntimeError(f"Cluster {self.cluster_id} not initialized")
        
        logger.info(f"Loading model for cluster {self.cluster_id}")
        
        try:
            # Save model data to temporary file
            temp_path = f"/tmp/{self.cluster_id}_model"
            with open(temp_path, 'wb') as f:
                f.write(model_data)
            
            # Load model
            self.model.load_model(temp_path)
            
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)
            
            logger.info(f"Model loaded for cluster {self.cluster_id}")
            
        except Exception as e:
            logger.error(f"Error loading model for cluster {self.cluster_id}: {e}")
            raise
    
    def save_model(self, filepath: str):
        """
        Save the cluster's model
        
        Args:
            filepath: Path to save the model
        """
        if not self.model or not self.model.is_trained:
            raise RuntimeError(f"Cluster {self.cluster_id} model not trained")
        
        logger.info(f"Saving model for cluster {self.cluster_id}")
        
        try:
            self.model.save_model(filepath)
            logger.info(f"Model saved for cluster {self.cluster_id}")
            
        except Exception as e:
            logger.error(f"Error saving model for cluster {self.cluster_id}: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get cluster status and metrics"""
        return {
            'cluster_id': self.cluster_id,
            'attack_type': self.attack_type,
            'model_type': self.model_type,
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'is_trained': self.model.is_trained if self.model else False,
            'prediction_count': self.prediction_count,
            'avg_prediction_time': self.avg_prediction_time,
            'last_prediction_time': self.last_prediction_time,
            'model_info': self.model.get_model_info() if self.model else None
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        if not self.model:
            return {'error': 'Model not initialized'}
        
        return self.model.get_model_info()