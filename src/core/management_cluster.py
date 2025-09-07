"""
Management cluster for supervising technical clusters and resolving conflicts
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import structlog
from collections import defaultdict, Counter
import statistics

logger = structlog.get_logger(__name__)


class ManagementCluster:
    """
    Management cluster that supervises technical clusters and makes final decisions
    Handles conflict resolution and ensemble learning
    """
    
    def __init__(self, config: Dict[str, Any], technical_clusters: Dict[str, Any]):
        self.config = config
        self.technical_clusters = technical_clusters
        self.is_initialized = False
        self.is_running = False
        
        # Configuration parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.conflict_resolution = config.get('conflict_resolution', 'weighted_voting')
        self.ensemble_weights = config.get('ensemble_weights', {'lstm': 0.6, 'svm': 0.4})
        
        # Decision history for learning
        self.decision_history = []
        self.conflict_history = []
        
        # Performance metrics
        self.decision_count = 0
        self.conflict_count = 0
        self.avg_decision_time = 0.0
        
    async def initialize(self):
        """Initialize the management cluster"""
        logger.info("Initializing management cluster")
        
        try:
            # Validate ensemble weights
            total_weight = sum(self.ensemble_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Ensemble weights sum to {total_weight}, normalizing")
                self.ensemble_weights = {
                    k: v / total_weight 
                    for k, v in self.ensemble_weights.items()
                }
            
            self.is_initialized = True
            logger.info("Management cluster initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize management cluster: {e}")
            raise
    
    async def start(self):
        """Start the management cluster"""
        if not self.is_initialized:
            raise RuntimeError("Management cluster not initialized")
        
        if self.is_running:
            logger.warning("Management cluster is already running")
            return
        
        self.is_running = True
        logger.info("Management cluster started")
    
    async def stop(self):
        """Stop the management cluster"""
        if not self.is_running:
            logger.warning("Management cluster is not running")
            return
        
        self.is_running = False
        logger.info("Management cluster stopped")
    
    async def decide(self, cluster_predictions: Dict[str, Dict[str, Any]], 
                    processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make final decision based on technical cluster predictions
        
        Args:
            cluster_predictions: Predictions from all technical clusters
            processed_data: Original processed data
            
        Returns:
            Final decision with confidence and severity
        """
        if not self.is_running:
            raise RuntimeError("Management cluster is not running")
        
        import time
        start_time = time.time()
        
        try:
            # Filter valid predictions
            valid_predictions = self._filter_valid_predictions(cluster_predictions)
            
            if not valid_predictions:
                # No valid predictions available
                decision = self._create_default_decision(processed_data)
            else:
                # Analyze predictions for conflicts
                conflicts = self._detect_conflicts(valid_predictions)
                
                if conflicts:
                    self.conflict_count += 1
                    self.conflict_history.append({
                        'timestamp': time.time(),
                        'conflicts': conflicts,
                        'predictions': valid_predictions
                    })
                
                # Resolve conflicts and make decision
                decision = await self._resolve_conflicts_and_decide(
                    valid_predictions, conflicts, processed_data
                )
            
            # Calculate decision time
            decision_time = time.time() - start_time
            
            # Update metrics
            self.decision_count += 1
            self.avg_decision_time = (
                (self.avg_decision_time * (self.decision_count - 1) + decision_time) 
                / self.decision_count
            )
            
            # Store decision history
            self.decision_history.append({
                'timestamp': time.time(),
                'decision': decision,
                'decision_time': decision_time,
                'predictions': cluster_predictions
            })
            
            # Limit history size
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-500:]
            
            logger.info(
                "Decision made",
                attack_type=decision['attack_type'],
                confidence=decision['confidence'],
                severity=decision['severity'],
                decision_time=decision_time
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in management cluster decision: {e}")
            return self._create_error_decision(str(e), processed_data)
    
    def _filter_valid_predictions(self, cluster_predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Filter out invalid or untrained cluster predictions"""
        valid_predictions = {}
        
        for cluster_id, prediction in cluster_predictions.items():
            if (prediction.get('is_trained', False) and 
                prediction.get('confidence', 0) > 0 and
                'error' not in prediction):
                valid_predictions[cluster_id] = prediction
        
        return valid_predictions
    
    def _detect_conflicts(self, predictions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect conflicts between cluster predictions"""
        conflicts = []
        
        # Group predictions by attack type
        attack_type_predictions = defaultdict(list)
        for cluster_id, prediction in predictions.items():
            attack_type = prediction['attack_type']
            attack_type_predictions[attack_type].append({
                'cluster_id': cluster_id,
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence']
            })
        
        # Check for conflicts
        for attack_type, type_predictions in attack_type_predictions.items():
            if len(type_predictions) > 1:
                # Multiple clusters predicting same attack type
                predictions_by_class = defaultdict(list)
                for pred in type_predictions:
                    predictions_by_class[pred['prediction']].append(pred)
                
                # Check for conflicting classes within same attack type
                if len(predictions_by_class) > 1:
                    conflicts.append({
                        'type': 'class_conflict',
                        'attack_type': attack_type,
                        'conflicting_classes': dict(predictions_by_class)
                    })
        
        # Check for high-confidence conflicting attack types
        attack_types = list(attack_type_predictions.keys())
        if len(attack_types) > 1:
            high_confidence_predictions = []
            for attack_type, type_predictions in attack_type_predictions.items():
                max_confidence = max(pred['confidence'] for pred in type_predictions)
                if max_confidence > self.confidence_threshold:
                    high_confidence_predictions.append({
                        'attack_type': attack_type,
                        'confidence': max_confidence,
                        'predictions': type_predictions
                    })
            
            if len(high_confidence_predictions) > 1:
                conflicts.append({
                    'type': 'attack_type_conflict',
                    'conflicting_attack_types': high_confidence_predictions
                })
        
        return conflicts
    
    async def _resolve_conflicts_and_decide(self, predictions: Dict[str, Dict[str, Any]], 
                                          conflicts: List[Dict[str, Any]], 
                                          processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts and make final decision"""
        
        if self.conflict_resolution == 'weighted_voting':
            return self._weighted_voting_decision(predictions, conflicts, processed_data)
        elif self.conflict_resolution == 'majority_voting':
            return self._majority_voting_decision(predictions, conflicts, processed_data)
        elif self.conflict_resolution == 'highest_confidence':
            return self._highest_confidence_decision(predictions, conflicts, processed_data)
        else:
            logger.warning(f"Unknown conflict resolution method: {self.conflict_resolution}")
            return self._weighted_voting_decision(predictions, conflicts, processed_data)
    
    def _weighted_voting_decision(self, predictions: Dict[str, Dict[str, Any]], 
                                conflicts: List[Dict[str, Any]], 
                                processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision using weighted voting"""
        
        # Group predictions by model type
        model_type_predictions = defaultdict(list)
        for cluster_id, prediction in predictions.items():
            model_type = prediction['model_type']
            model_type_predictions[model_type].append(prediction)
        
        # Calculate weighted scores for each class
        class_scores = defaultdict(float)
        total_weight = 0.0
        
        for model_type, type_predictions in model_type_predictions.items():
            weight = self.ensemble_weights.get(model_type, 0.5)
            
            for prediction in type_predictions:
                pred_class = prediction['prediction']
                confidence = prediction['confidence']
                class_scores[pred_class] += weight * confidence
                total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            class_scores = {k: v / total_weight for k, v in class_scores.items()}
        
        # Find best prediction
        if class_scores:
            best_class = max(class_scores, key=class_scores.get)
            best_confidence = class_scores[best_class]
        else:
            best_class = 'Unknown'
            best_confidence = 0.0
        
        # Determine attack type and severity
        attack_type, severity = self._determine_attack_type_and_severity(
            best_class, best_confidence, conflicts
        )
        
        return {
            'attack_type': attack_type,
            'prediction': best_class,
            'confidence': best_confidence,
            'severity': severity,
            'method': 'weighted_voting',
            'conflicts_detected': len(conflicts),
            'class_scores': dict(class_scores),
            'metadata': processed_data.get('metadata', {})
        }
    
    def _majority_voting_decision(self, predictions: Dict[str, Dict[str, Any]], 
                                conflicts: List[Dict[str, Any]], 
                                processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision using majority voting"""
        
        # Count votes for each class
        class_votes = Counter()
        total_confidence = 0.0
        
        for prediction in predictions.values():
            pred_class = prediction['prediction']
            confidence = prediction['confidence']
            class_votes[pred_class] += 1
            total_confidence += confidence
        
        # Find majority class
        if class_votes:
            best_class = class_votes.most_common(1)[0][0]
            majority_count = class_votes[best_class]
            total_predictions = len(predictions)
            best_confidence = majority_count / total_predictions
        else:
            best_class = 'Unknown'
            best_confidence = 0.0
        
        # Determine attack type and severity
        attack_type, severity = self._determine_attack_type_and_severity(
            best_class, best_confidence, conflicts
        )
        
        return {
            'attack_type': attack_type,
            'prediction': best_class,
            'confidence': best_confidence,
            'severity': severity,
            'method': 'majority_voting',
            'conflicts_detected': len(conflicts),
            'class_votes': dict(class_votes),
            'metadata': processed_data.get('metadata', {})
        }
    
    def _highest_confidence_decision(self, predictions: Dict[str, Dict[str, Any]], 
                                   conflicts: List[Dict[str, Any]], 
                                   processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision using highest confidence"""
        
        # Find prediction with highest confidence
        best_prediction = None
        best_confidence = 0.0
        
        for prediction in predictions.values():
            confidence = prediction['confidence']
            if confidence > best_confidence:
                best_confidence = confidence
                best_prediction = prediction
        
        if best_prediction:
            best_class = best_prediction['prediction']
        else:
            best_class = 'Unknown'
            best_confidence = 0.0
        
        # Determine attack type and severity
        attack_type, severity = self._determine_attack_type_and_severity(
            best_class, best_confidence, conflicts
        )
        
        return {
            'attack_type': attack_type,
            'prediction': best_class,
            'confidence': best_confidence,
            'severity': severity,
            'method': 'highest_confidence',
            'conflicts_detected': len(conflicts),
            'best_cluster': best_prediction['cluster_id'] if best_prediction else None,
            'metadata': processed_data.get('metadata', {})
        }
    
    def _determine_attack_type_and_severity(self, prediction: str, confidence: float, 
                                          conflicts: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Determine attack type and severity level"""
        
        # Map predictions to attack types
        attack_type_mapping = {
            'Benign': 'Benign',
            'DoS': 'DoS',
            'DDoS': 'DDoS',
            'Botnet': 'Botnet',
            'Brute_Force': 'Brute_Force',
            'Spoofing': 'Spoofing',
            'Probing': 'Probing',
            'Injection': 'Injection',
            'Data_Theft': 'Data_Theft',
            'Unknown': 'Unknown',
            'Error': 'Error'
        }
        
        attack_type = attack_type_mapping.get(prediction, 'Unknown')
        
        # Determine severity based on confidence and conflicts
        if attack_type == 'Benign':
            severity = 'Low'
        elif confidence >= 0.9 and not conflicts:
            severity = 'Critical'
        elif confidence >= 0.8:
            severity = 'High'
        elif confidence >= 0.6:
            severity = 'Medium'
        else:
            severity = 'Low'
        
        # Increase severity if conflicts detected
        if conflicts:
            if severity == 'Low':
                severity = 'Medium'
            elif severity == 'Medium':
                severity = 'High'
        
        return attack_type, severity
    
    def _create_default_decision(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create default decision when no valid predictions available"""
        return {
            'attack_type': 'Unknown',
            'prediction': 'Unknown',
            'confidence': 0.0,
            'severity': 'Low',
            'method': 'default',
            'conflicts_detected': 0,
            'metadata': processed_data.get('metadata', {})
        }
    
    def _create_error_decision(self, error_message: str, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create error decision when processing fails"""
        return {
            'attack_type': 'Error',
            'prediction': 'Error',
            'confidence': 0.0,
            'severity': 'High',
            'method': 'error',
            'conflicts_detected': 0,
            'error': error_message,
            'metadata': processed_data.get('metadata', {})
        }
    
    async def add_cluster(self, cluster_id: str, cluster: Any):
        """Add a new technical cluster to the management system"""
        self.technical_clusters[cluster_id] = cluster
        logger.info(f"Added new cluster to management system: {cluster_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get management cluster status and metrics"""
        return {
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'decision_count': self.decision_count,
            'conflict_count': self.conflict_count,
            'avg_decision_time': self.avg_decision_time,
            'confidence_threshold': self.confidence_threshold,
            'conflict_resolution': self.conflict_resolution,
            'ensemble_weights': self.ensemble_weights,
            'active_clusters': len(self.technical_clusters),
            'recent_conflicts': len(self.conflict_history[-10:]) if self.conflict_history else 0
        }
    
    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent decision history"""
        return self.decision_history[-limit:] if self.decision_history else []
    
    def get_conflict_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent conflict history"""
        return self.conflict_history[-limit:] if self.conflict_history else []