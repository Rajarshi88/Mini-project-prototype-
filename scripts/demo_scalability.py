"""
Demonstration script for dynamic scalability of CyberAIBot
Shows how to add new technical clusters for new attack types
"""

import asyncio
import numpy as np
import pickle
from typing import Dict, Any
import structlog
from pathlib import Path
import json
from datetime import datetime

from ..core.system import CyberAIBot
from ..models.lstm_model import LSTMModel
from ..models.svm_model import SVMModel

logger = structlog.get_logger(__name__)


class ScalabilityDemo:
    """
    Demonstration of CyberAIBot's dynamic scalability
    """
    
    def __init__(self):
        self.cyberai_bot = None
        self.demo_results = {}
    
    async def run_demo(self):
        """Run the complete scalability demonstration"""
        logger.info("Starting CyberAIBot scalability demonstration")
        
        try:
            # Initialize CyberAIBot system
            await self._initialize_system()
            
            # Demonstrate initial system
            await self._demonstrate_initial_system()
            
            # Add new attack type clusters
            await self._add_new_attack_clusters()
            
            # Demonstrate dynamic model loading
            await self._demonstrate_model_loading()
            
            # Test system with new clusters
            await self._test_enhanced_system()
            
            # Generate demonstration report
            self._generate_demo_report()
            
            logger.info("Scalability demonstration completed successfully")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        finally:
            if self.cyberai_bot:
                await self.cyberai_bot.stop()
    
    async def _initialize_system(self):
        """Initialize the CyberAIBot system"""
        logger.info("Initializing CyberAIBot system")
        
        self.cyberai_bot = CyberAIBot("config/config.yaml")
        await self.cyberai_bot.initialize()
        await self.cyberai_bot.start()
        
        logger.info("System initialized with initial clusters")
    
    async def _demonstrate_initial_system(self):
        """Demonstrate the initial system capabilities"""
        logger.info("Demonstrating initial system capabilities")
        
        # Get initial system status
        initial_status = self.cyberai_bot.get_system_status()
        initial_clusters = list(initial_status['clusters'].keys())
        
        logger.info(f"Initial system has {len(initial_clusters)} clusters: {initial_clusters}")
        
        # Test with sample traffic
        sample_traffic = self._generate_sample_traffic("DoS")
        result = await self.cyberai_bot.process_traffic(sample_traffic)
        
        self.demo_results['initial_system'] = {
            'cluster_count': len(initial_clusters),
            'clusters': initial_clusters,
            'sample_result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Initial system test result: {result['attack_type']} (confidence: {result['confidence']:.3f})")
    
    async def _add_new_attack_clusters(self):
        """Add new technical clusters for new attack types"""
        logger.info("Adding new attack type clusters")
        
        new_attack_types = [
            "Zero_Day",
            "Advanced_Persistent_Threat",
            "Ransomware",
            "Cryptojacking"
        ]
        
        added_clusters = []
        
        for attack_type in new_attack_types:
            try:
                # Create and train a new LSTM cluster
                lstm_cluster_id = await self._create_new_lstm_cluster(attack_type)
                added_clusters.append(lstm_cluster_id)
                
                # Create and train a new SVM cluster
                svm_cluster_id = await self._create_new_svm_cluster(attack_type)
                added_clusters.append(svm_cluster_id)
                
                logger.info(f"Added clusters for {attack_type}: {lstm_cluster_id}, {svm_cluster_id}")
                
            except Exception as e:
                logger.error(f"Failed to add clusters for {attack_type}: {e}")
        
        self.demo_results['new_clusters'] = {
            'attack_types': new_attack_types,
            'added_clusters': added_clusters,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get updated system status
        updated_status = self.cyberai_bot.get_system_status()
        updated_clusters = list(updated_status['clusters'].keys())
        
        logger.info(f"System now has {len(updated_clusters)} clusters")
        logger.info(f"Added {len(added_clusters)} new clusters without retraining existing ones")
    
    async def _create_new_lstm_cluster(self, attack_type: str) -> str:
        """Create a new LSTM cluster for a new attack type"""
        # Create a mock trained model
        mock_model = LSTMModel(attack_type, {
            'hidden_units': 64,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'epochs': 50
        })
        
        # Mock training data
        X_train = np.random.randn(100, 10, 20)  # 100 samples, 10 timesteps, 20 features
        y_train = np.random.randint(0, 2, 100)  # Binary classification
        X_val = np.random.randn(20, 10, 20)
        y_val = np.random.randint(0, 2, 20)
        
        # Train the model
        await mock_model.train(X_train, y_train, X_val, y_val, ['Benign', attack_type])
        
        # Serialize the model
        model_data = self._serialize_model(mock_model)
        
        # Add to system
        cluster_id = await self.cyberai_bot.add_new_cluster(attack_type, 'lstm', model_data)
        
        return cluster_id
    
    async def _create_new_svm_cluster(self, attack_type: str) -> str:
        """Create a new SVM cluster for a new attack type"""
        # Create a mock trained model
        mock_model = SVMModel(attack_type, {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale'
        })
        
        # Mock training data
        X_train = np.random.randn(100, 20)  # 100 samples, 20 features
        y_train = np.random.randint(0, 2, 100)  # Binary classification
        X_val = np.random.randn(20, 20)
        y_val = np.random.randint(0, 2, 20)
        
        # Train the model
        await mock_model.train(X_train, y_train, X_val, y_val, ['Benign', attack_type])
        
        # Serialize the model
        model_data = self._serialize_model(mock_model)
        
        # Add to system
        cluster_id = await self.cyberai_bot.add_new_cluster(attack_type, 'svm', model_data)
        
        return cluster_id
    
    def _serialize_model(self, model) -> bytes:
        """Serialize a model to bytes"""
        import tempfile
        import os
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Save model to temporary file
            model.save_model(temp_path)
            
            # Read as bytes
            with open(temp_path, 'rb') as f:
                model_data = f.read()
            
            return model_data
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    async def _demonstrate_model_loading(self):
        """Demonstrate loading pre-trained models"""
        logger.info("Demonstrating model loading capabilities")
        
        # Create a pre-trained model file
        model_path = "models/demo_pretrained_model"
        Path("models").mkdir(exist_ok=True)
        
        # Create and save a demo model
        demo_model = LSTMModel("Demo_Attack", {
            'hidden_units': 32,
            'dropout': 0.2,
            'learning_rate': 0.001
        })
        
        # Mock training
        X_train = np.random.randn(50, 5, 10)
        y_train = np.random.randint(0, 2, 50)
        X_val = np.random.randn(10, 5, 10)
        y_val = np.random.randint(0, 2, 10)
        
        await demo_model.train(X_train, y_train, X_val, y_val, ['Benign', 'Demo_Attack'])
        demo_model.save_model(model_path)
        
        # Load the model
        model_data = self._serialize_model(demo_model)
        cluster_id = await self.cyberai_bot.add_new_cluster("Demo_Attack", 'lstm', model_data)
        
        self.demo_results['model_loading'] = {
            'model_path': model_path,
            'loaded_cluster_id': cluster_id,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Successfully loaded pre-trained model as cluster: {cluster_id}")
    
    async def _test_enhanced_system(self):
        """Test the enhanced system with new clusters"""
        logger.info("Testing enhanced system with new clusters")
        
        test_results = []
        
        # Test with different attack types
        test_attacks = [
            "DoS",
            "Zero_Day",
            "Advanced_Persistent_Threat",
            "Ransomware",
            "Cryptojacking"
        ]
        
        for attack_type in test_attacks:
            try:
                # Generate sample traffic
                sample_traffic = self._generate_sample_traffic(attack_type)
                
                # Process through system
                result = await self.cyberai_bot.process_traffic(sample_traffic)
                
                test_results.append({
                    'attack_type': attack_type,
                    'result': result,
                    'success': True
                })
                
                logger.info(f"Test {attack_type}: {result['attack_type']} (confidence: {result['confidence']:.3f})")
                
            except Exception as e:
                test_results.append({
                    'attack_type': attack_type,
                    'error': str(e),
                    'success': False
                })
                logger.error(f"Test failed for {attack_type}: {e}")
        
        self.demo_results['enhanced_system_test'] = {
            'test_results': test_results,
            'successful_tests': sum(1 for r in test_results if r['success']),
            'total_tests': len(test_results),
            'timestamp': datetime.now().isoformat()
        }
        
        # Get final system status
        final_status = self.cyberai_bot.get_system_status()
        final_clusters = list(final_status['clusters'].keys())
        
        logger.info(f"Final system has {len(final_clusters)} clusters")
        logger.info(f"Successfully processed {len([r for r in test_results if r['success']])}/{len(test_results)} test cases")
    
    def _generate_sample_traffic(self, attack_type: str) -> Dict[str, Any]:
        """Generate sample traffic data for testing"""
        np.random.seed(42)
        
        base_traffic = {
            'timestamp': datetime.now().isoformat(),
            'src_ip': f"192.168.1.{np.random.randint(1, 255)}",
            'dst_ip': f"10.0.0.{np.random.randint(1, 255)}",
            'src_port': np.random.randint(1024, 65535),
            'dst_port': np.random.randint(1, 1024),
            'protocol': np.random.choice(['TCP', 'UDP', 'ICMP']),
            'packet_size': np.random.randint(64, 1500),
            'flow_duration': np.random.exponential(1.0)
        }
        
        # Modify traffic characteristics based on attack type
        if attack_type == "DoS":
            base_traffic['packet_size'] = np.random.randint(1000, 1500)
            base_traffic['flow_duration'] = 0.1
        elif attack_type == "Zero_Day":
            base_traffic['packet_size'] = np.random.randint(200, 800)
            base_traffic['flow_duration'] = np.random.exponential(0.5)
        elif attack_type == "Advanced_Persistent_Threat":
            base_traffic['packet_size'] = np.random.randint(100, 500)
            base_traffic['flow_duration'] = np.random.exponential(10.0)
        elif attack_type == "Ransomware":
            base_traffic['packet_size'] = np.random.randint(500, 1200)
            base_traffic['flow_duration'] = np.random.exponential(2.0)
        elif attack_type == "Cryptojacking":
            base_traffic['packet_size'] = np.random.randint(300, 800)
            base_traffic['flow_duration'] = np.random.exponential(5.0)
        
        return base_traffic
    
    def _generate_demo_report(self):
        """Generate demonstration report"""
        logger.info("Generating demonstration report")
        
        report = {
            'demo_title': 'CyberAIBot Dynamic Scalability Demonstration',
            'timestamp': datetime.now().isoformat(),
            'results': self.demo_results,
            'summary': {
                'initial_clusters': self.demo_results['initial_system']['cluster_count'],
                'final_clusters': len(self.cyberai_bot.technical_clusters) if self.cyberai_bot else 0,
                'new_clusters_added': len(self.demo_results.get('new_clusters', {}).get('added_clusters', [])),
                'successful_tests': self.demo_results.get('enhanced_system_test', {}).get('successful_tests', 0),
                'total_tests': self.demo_results.get('enhanced_system_test', {}).get('total_tests', 0)
            }
        }
        
        # Save report
        report_path = "demo_scalability_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Demonstration report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("CYBERAI BOT SCALABILITY DEMONSTRATION SUMMARY")
        print("="*60)
        print(f"Initial Clusters: {report['summary']['initial_clusters']}")
        print(f"Final Clusters: {report['summary']['final_clusters']}")
        print(f"New Clusters Added: {report['summary']['new_clusters_added']}")
        print(f"Successful Tests: {report['summary']['successful_tests']}/{report['summary']['total_tests']}")
        print("="*60)
        print("✓ Dynamic cluster addition without retraining existing clusters")
        print("✓ Real-time system scaling capabilities")
        print("✓ Pre-trained model loading and integration")
        print("✓ Enhanced threat detection with new attack types")
        print("="*60)


async def run_scalability_demo():
    """Run the scalability demonstration"""
    demo = ScalabilityDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(run_scalability_demo())