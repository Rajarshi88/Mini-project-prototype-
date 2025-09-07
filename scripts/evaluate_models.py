"""
Evaluation scripts for CyberAIBot models
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import structlog
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from datetime import datetime

from .data_loader import IoTDataLoader

logger = structlog.get_logger(__name__)


class ModelEvaluator:
    """
    Model evaluator for CyberAIBot technical clusters
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_loader = IoTDataLoader()
        self.evaluation_results = {}
        
    async def evaluate_all_models(self, cyberai_bot) -> Dict[str, Any]:
        """
        Evaluate all technical clusters
        
        Args:
            cyberai_bot: CyberAIBot system instance
            
        Returns:
            Evaluation results
        """
        logger.info("Starting evaluation for all models")
        
        try:
            # Load test datasets
            datasets = self._load_test_datasets()
            
            # Evaluate each cluster
            for cluster_id, cluster in cyberai_bot.technical_clusters.items():
                if cluster.model and cluster.model.is_trained:
                    logger.info(f"Evaluating cluster: {cluster_id}")
                    
                    # Get test data for this cluster's attack type
                    test_data = self._prepare_test_data(
                        cluster.attack_type, datasets
                    )
                    
                    if test_data is not None:
                        # Evaluate the cluster
                        result = await self._evaluate_cluster(cluster, test_data)
                        self.evaluation_results[cluster_id] = result
                    else:
                        logger.warning(f"No test data available for {cluster_id}")
                else:
                    logger.warning(f"Cluster {cluster_id} is not trained")
            
            # Generate comparison report
            comparison_report = self._generate_comparison_report()
            
            # Save evaluation results
            self._save_evaluation_results()
            
            # Generate visualizations
            self._generate_visualizations()
            
            logger.info("Evaluation completed for all models")
            return {
                'individual_results': self.evaluation_results,
                'comparison_report': comparison_report
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def _load_test_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load test datasets"""
        datasets = {}
        
        for dataset_name in self.data_loader.list_available_datasets():
            try:
                # Load smaller sample for evaluation
                df = self.data_loader.load_dataset(dataset_name, sample_size=10000)
                datasets[dataset_name] = df
                logger.info(f"Loaded test dataset {dataset_name}: {len(df)} samples")
            except Exception as e:
                logger.error(f"Failed to load test dataset {dataset_name}: {e}")
        
        return datasets
    
    def _prepare_test_data(self, attack_type: str, datasets: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        Prepare test data for a specific attack type
        
        Args:
            attack_type: Type of attack to test for
            datasets: Available test datasets
            
        Returns:
            Prepared test data or None
        """
        logger.info(f"Preparing test data for {attack_type}")
        
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
                        n=min(len(attack_data), 5000), random_state=42
                    )
                    combined_data.append(benign_data)
        
        if not combined_data:
            logger.warning(f"No test data found for attack type: {attack_type}")
            return None
        
        # Combine all data
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Preprocess the data
        X, y, feature_names = self.data_loader.preprocess_dataset(
            combined_df, 'cic_ids_2018'  # Use first available dataset config
        )
        
        # Create sequences for LSTM
        X_seq, y_seq = self.data_loader.create_sequences(X, y)
        
        # Get class names
        class_names = ['Benign', attack_type]
        
        return {
            'X': X,
            'y': y,
            'X_seq': X_seq,
            'y_seq': y_seq,
            'class_names': class_names,
            'feature_names': feature_names
        }
    
    async def _evaluate_cluster(self, cluster: Any, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a specific cluster
        
        Args:
            cluster: Technical cluster to evaluate
            test_data: Test data
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating cluster: {cluster.cluster_id}")
        
        try:
            # Choose appropriate data based on model type
            if cluster.model_type == 'lstm':
                X_test = test_data['X_seq']
                y_test = test_data['y_seq']
            else:  # SVM
                X_test = test_data['X']
                y_test = test_data['y']
            
            # Get predictions
            predictions = await cluster.predict({
                'features': X_test,
                'sequences': test_data['X_seq']
            })
            
            # Calculate detailed metrics
            y_pred = predictions['predictions']
            y_true = y_test
            
            # Classification report
            report = classification_report(
                y_true, y_pred,
                target_names=test_data['class_names'],
                output_dict=True
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # ROC AUC (for binary classification)
            roc_auc = None
            if len(test_data['class_names']) == 2:
                try:
                    roc_auc = roc_auc_score(y_true, y_pred)
                except:
                    pass
            
            # Calculate additional metrics
            accuracy = report['accuracy']
            precision = report['macro avg']['precision']
            recall = report['macro avg']['recall']
            f1_score = report['macro avg']['f1-score']
            
            result = {
                'cluster_id': cluster.cluster_id,
                'attack_type': cluster.attack_type,
                'model_type': cluster.model_type,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'roc_auc': roc_auc,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'test_samples': len(y_test),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully evaluated cluster: {cluster.cluster_id}")
            logger.info(f"Accuracy: {accuracy:.4f}, F1-Score: {f1_score:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate cluster {cluster.cluster_id}: {e}")
            return {
                'cluster_id': cluster.cluster_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comparison report across all models"""
        logger.info("Generating comparison report")
        
        # Extract metrics for comparison
        comparison_data = []
        
        for cluster_id, result in self.evaluation_results.items():
            if 'error' not in result:
                comparison_data.append({
                    'cluster_id': cluster_id,
                    'attack_type': result['attack_type'],
                    'model_type': result['model_type'],
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1_score': result['f1_score'],
                    'roc_auc': result.get('roc_auc', None)
                })
        
        if not comparison_data:
            return {'error': 'No valid evaluation results available'}
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Calculate summary statistics
        summary_stats = {
            'overall_accuracy': df['accuracy'].mean(),
            'overall_precision': df['precision'].mean(),
            'overall_recall': df['recall'].mean(),
            'overall_f1_score': df['f1_score'].mean(),
            'model_type_performance': df.groupby('model_type').agg({
                'accuracy': 'mean',
                'precision': 'mean',
                'recall': 'mean',
                'f1_score': 'mean'
            }).to_dict(),
            'attack_type_performance': df.groupby('attack_type').agg({
                'accuracy': 'mean',
                'precision': 'mean',
                'recall': 'mean',
                'f1_score': 'mean'
            }).to_dict(),
            'best_performing_models': {
                'highest_accuracy': df.loc[df['accuracy'].idxmax()].to_dict(),
                'highest_f1_score': df.loc[df['f1_score'].idxmax()].to_dict(),
                'highest_precision': df.loc[df['precision'].idxmax()].to_dict(),
                'highest_recall': df.loc[df['recall'].idxmax()].to_dict()
            }
        }
        
        return {
            'summary_statistics': summary_stats,
            'detailed_comparison': comparison_data,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_evaluation_results(self):
        """Save evaluation results to file"""
        results_path = "models/evaluation_results.json"
        Path("models").mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {results_path}")
    
    def _generate_visualizations(self):
        """Generate evaluation visualizations"""
        logger.info("Generating evaluation visualizations")
        
        try:
            # Create visualizations directory
            viz_dir = Path("models/visualizations")
            viz_dir.mkdir(exist_ok=True)
            
            # Extract data for plotting
            comparison_data = []
            for cluster_id, result in self.evaluation_results.items():
                if 'error' not in result:
                    comparison_data.append({
                        'cluster_id': cluster_id,
                        'attack_type': result['attack_type'],
                        'model_type': result['model_type'],
                        'accuracy': result['accuracy'],
                        'precision': result['precision'],
                        'recall': result['recall'],
                        'f1_score': result['f1_score']
                    })
            
            if not comparison_data:
                logger.warning("No data available for visualizations")
                return
            
            df = pd.DataFrame(comparison_data)
            
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Model Performance Comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('CyberAIBot Model Performance Comparison', fontsize=16)
            
            # Accuracy comparison
            sns.barplot(data=df, x='cluster_id', y='accuracy', ax=axes[0, 0])
            axes[0, 0].set_title('Accuracy by Cluster')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # F1-Score comparison
            sns.barplot(data=df, x='cluster_id', y='f1_score', ax=axes[0, 1])
            axes[0, 1].set_title('F1-Score by Cluster')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Model type comparison
            model_metrics = df.groupby('model_type')[['accuracy', 'precision', 'recall', 'f1_score']].mean()
            model_metrics.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Average Performance by Model Type')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Attack type comparison
            attack_metrics = df.groupby('attack_type')[['accuracy', 'precision', 'recall', 'f1_score']].mean()
            attack_metrics.plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Average Performance by Attack Type')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Confusion Matrices
            n_clusters = len(comparison_data)
            n_cols = min(3, n_clusters)
            n_rows = (n_clusters + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_clusters == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, (cluster_id, result) in enumerate(self.evaluation_results.items()):
                if 'error' not in result and 'confusion_matrix' in result:
                    cm = np.array(result['confusion_matrix'])
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                    axes[i].set_title(f'{cluster_id}\nConfusion Matrix')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
            
            # Hide unused subplots
            for i in range(n_clusters, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualizations saved to {viz_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")


async def evaluate_all_models(cyberai_bot) -> Dict[str, Any]:
    """
    Evaluate all models in the CyberAIBot system
    
    Args:
        cyberai_bot: CyberAIBot system instance
        
    Returns:
        Evaluation results
    """
    evaluator = ModelEvaluator(cyberai_bot.config)
    return await evaluator.evaluate_all_models(cyberai_bot)


async def evaluate_specific_cluster(cyberai_bot, cluster_id: str, 
                                  test_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a specific cluster
    
    Args:
        cyberai_bot: CyberAIBot system instance
        cluster_id: ID of the cluster to evaluate
        test_data: Test data
        
    Returns:
        Evaluation results
    """
    if cluster_id not in cyberai_bot.technical_clusters:
        raise ValueError(f"Cluster {cluster_id} not found")
    
    cluster = cyberai_bot.technical_clusters[cluster_id]
    evaluator = ModelEvaluator(cyberai_bot.config)
    
    return await evaluator._evaluate_cluster(cluster, test_data)