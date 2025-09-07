"""
Data preprocessing pipeline for IoT traffic analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import structlog
from scipy import stats
import pickle
from pathlib import Path

logger = structlog.get_logger(__name__)


class DataProcessor:
    """
    Handles data preprocessing, feature extraction, and feature selection
    for IoT traffic data
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = None
        self.feature_selector = None
        self.label_encoder = None
        self.feature_names = None
        self.is_initialized = False
        
        # Feature extraction parameters
        self.window_size = config.get('window_size', 10)
        self.overlap = config.get('overlap', 0.5)
        
    async def initialize(self):
        """Initialize the data processor"""
        logger.info("Initializing data processor")
        
        # Initialize scaler based on config
        normalization_method = self.config['normalization']['method']
        if normalization_method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=tuple(self.config['normalization']['range']))
        elif normalization_method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")
        
        # Initialize feature selector
        self.feature_selector = SelectKBest(
            score_func=f_classif,
            k=self.config['feature_selection']['max_features']
        )
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
        self.is_initialized = True
        logger.info("Data processor initialized successfully")
    
    def extract_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic statistical features from IoT traffic data
        
        Args:
            data: Raw IoT traffic data
            
        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame()
        
        # Basic packet statistics
        if 'packet_size' in data.columns:
            features['packet_size_mean'] = data['packet_size'].rolling(window=self.window_size).mean()
            features['packet_size_std'] = data['packet_size'].rolling(window=self.window_size).std()
            features['packet_size_min'] = data['packet_size'].rolling(window=self.window_size).min()
            features['packet_size_max'] = data['packet_size'].rolling(window=self.window_size).max()
            features['packet_size_median'] = data['packet_size'].rolling(window=self.window_size).median()
        
        # Flow statistics
        if 'flow_duration' in data.columns:
            features['flow_duration_mean'] = data['flow_duration'].rolling(window=self.window_size).mean()
            features['flow_duration_std'] = data['flow_duration'].rolling(window=self.window_size).std()
        
        # Protocol features
        if 'protocol' in data.columns:
            protocol_dummies = pd.get_dummies(data['protocol'], prefix='protocol')
            features = pd.concat([features, protocol_dummies], axis=1)
        
        # Network features
        if 'src_ip' in data.columns and 'dst_ip' in data.columns:
            features['unique_src_ips'] = data['src_ip'].rolling(window=self.window_size).nunique()
            features['unique_dst_ips'] = data['dst_ip'].rolling(window=self.window_size).nunique()
        
        # Port features
        if 'src_port' in data.columns and 'dst_port' in data.columns:
            features['unique_src_ports'] = data['src_port'].rolling(window=self.window_size).nunique()
            features['unique_dst_ports'] = data['dst_port'].rolling(window=self.window_size).nunique()
            features['high_port_ratio'] = (data['dst_port'] > 1024).rolling(window=self.window_size).mean()
        
        # Rate features
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            time_diff = data['timestamp'].diff().dt.total_seconds()
            features['packet_rate'] = 1 / time_diff.rolling(window=self.window_size).mean()
            features['packet_rate_std'] = 1 / time_diff.rolling(window=self.window_size).std()
        
        return features
    
    def extract_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract advanced statistical and domain-specific features
        
        Args:
            data: Raw IoT traffic data
            
        Returns:
            DataFrame with advanced features
        """
        features = pd.DataFrame()
        
        # Entropy features
        if 'src_ip' in data.columns:
            src_ip_entropy = data['src_ip'].rolling(window=self.window_size).apply(
                lambda x: stats.entropy(x.value_counts())
            )
            features['src_ip_entropy'] = src_ip_entropy
        
        if 'dst_ip' in data.columns:
            dst_ip_entropy = data['dst_ip'].rolling(window=self.window_size).apply(
                lambda x: stats.entropy(x.value_counts())
            )
            features['dst_ip_entropy'] = dst_ip_entropy
        
        # Connection patterns
        if 'src_ip' in data.columns and 'dst_ip' in data.columns:
            # Connection fan-out (one source to many destinations)
            fan_out = data.groupby('src_ip')['dst_ip'].nunique().rolling(window=self.window_size).mean()
            features['avg_fan_out'] = fan_out
            
            # Connection fan-in (many sources to one destination)
            fan_in = data.groupby('dst_ip')['src_ip'].nunique().rolling(window=self.window_size).mean()
            features['avg_fan_in'] = fan_in
        
        # Traffic volume patterns
        if 'packet_size' in data.columns:
            # Traffic burstiness
            features['traffic_burstiness'] = (
                data['packet_size'].rolling(window=self.window_size).std() / 
                data['packet_size'].rolling(window=self.window_size).mean()
            )
            
            # Traffic regularity
            features['traffic_regularity'] = (
                1 / (1 + data['packet_size'].rolling(window=self.window_size).std())
            )
        
        # Time-based features
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            
            # Time-based patterns
            features['hour_entropy'] = data['hour'].rolling(window=self.window_size).apply(
                lambda x: stats.entropy(x.value_counts())
            )
            features['day_entropy'] = data['day_of_week'].rolling(window=self.window_size).apply(
                lambda x: stats.entropy(x.value_counts())
            )
        
        return features
    
    def extract_correlation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features based on correlation analysis
        
        Args:
            data: Raw IoT traffic data
            
        Returns:
            DataFrame with correlation-based features
        """
        features = pd.DataFrame()
        
        # Cross-correlation between different metrics
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) >= 2:
            # Calculate rolling correlations
            for i, col1 in enumerate(numeric_columns):
                for col2 in numeric_columns[i+1:]:
                    corr_name = f"corr_{col1}_{col2}"
                    features[corr_name] = data[col1].rolling(window=self.window_size).corr(data[col2])
        
        return features
    
    def select_features(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Perform feature selection using correlation-based method
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Selected features and feature names
        """
        logger.info("Performing feature selection")
        
        # Remove highly correlated features
        if self.config['feature_selection']['method'] == 'correlation':
            correlation_threshold = self.config['feature_selection']['threshold']
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(X.T)
            
            # Find highly correlated pairs
            high_corr_pairs = np.where(
                (np.abs(corr_matrix) > correlation_threshold) & 
                (corr_matrix != 1.0)
            )
            
            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                if i not in features_to_remove and j not in features_to_remove:
                    features_to_remove.add(j)
            
            # Remove highly correlated features
            X_filtered = np.delete(X, list(features_to_remove), axis=1)
            
            # Update feature names
            if self.feature_names:
                self.feature_names = [
                    name for i, name in enumerate(self.feature_names) 
                    if i not in features_to_remove
                ]
        
        # Apply statistical feature selection
        X_selected = self.feature_selector.fit_transform(X_filtered, y)
        
        # Update feature names
        if self.feature_names:
            selected_indices = self.feature_selector.get_support(indices=True)
            self.feature_names = [self.feature_names[i] for i in selected_indices]
        
        logger.info(f"Selected {X_selected.shape[1]} features from {X.shape[1]} original features")
        
        return X_selected, self.feature_names
    
    def create_sequences(self, data: np.ndarray, sequence_length: int = 10) -> np.ndarray:
        """
        Create sequences for LSTM models
        
        Args:
            data: Input data
            sequence_length: Length of each sequence
            
        Returns:
            Array of sequences
        """
        sequences = []
        
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        
        return np.array(sequences)
    
    async def process(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw IoT traffic data through the complete pipeline
        
        Args:
            raw_data: Raw traffic data from IoT devices
            
        Returns:
            Processed data ready for model inference
        """
        if not self.is_initialized:
            raise RuntimeError("Data processor not initialized")
        
        try:
            # Convert to DataFrame
            data = pd.DataFrame([raw_data])
            
            # Extract features
            basic_features = self.extract_basic_features(data)
            advanced_features = self.extract_advanced_features(data)
            correlation_features = self.extract_correlation_features(data)
            
            # Combine all features
            all_features = pd.concat([
                basic_features, 
                advanced_features, 
                correlation_features
            ], axis=1)
            
            # Handle missing values
            all_features = all_features.fillna(0)
            
            # Convert to numpy array
            feature_array = all_features.values
            
            # Normalize features
            if feature_array.size > 0:
                feature_array = self.scaler.transform(feature_array.reshape(1, -1))
            
            # Create sequences for LSTM
            if len(feature_array) >= self.window_size:
                sequences = self.create_sequences(feature_array, self.window_size)
            else:
                # Pad with zeros if not enough data
                padded_data = np.zeros((self.window_size, feature_array.shape[1]))
                padded_data[-len(feature_array):] = feature_array
                sequences = padded_data.reshape(1, self.window_size, -1)
            
            return {
                'features': feature_array,
                'sequences': sequences,
                'feature_names': self.feature_names,
                'metadata': {
                    'timestamp': raw_data.get('timestamp'),
                    'src_ip': raw_data.get('src_ip'),
                    'dst_ip': raw_data.get('dst_ip'),
                    'protocol': raw_data.get('protocol')
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise
    
    def fit_preprocessing(self, X: np.ndarray, y: np.ndarray):
        """
        Fit preprocessing components on training data
        
        Args:
            X: Training features
            y: Training labels
        """
        logger.info("Fitting preprocessing components")
        
        # Fit scaler
        self.scaler.fit(X)
        
        # Fit feature selector
        X_selected, feature_names = self.select_features(X, y)
        
        # Fit label encoder
        self.label_encoder.fit(y)
        
        logger.info("Preprocessing components fitted successfully")
        
        return X_selected
    
    def save_preprocessing(self, filepath: str):
        """Save preprocessing components to disk"""
        preprocessing_data = {
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessing_data, f)
        
        logger.info(f"Preprocessing components saved to {filepath}")
    
    def load_preprocessing(self, filepath: str):
        """Load preprocessing components from disk"""
        with open(filepath, 'rb') as f:
            preprocessing_data = pickle.load(f)
        
        self.scaler = preprocessing_data['scaler']
        self.feature_selector = preprocessing_data['feature_selector']
        self.label_encoder = preprocessing_data['label_encoder']
        self.feature_names = preprocessing_data['feature_names']
        
        logger.info(f"Preprocessing components loaded from {filepath}")