"""
Core system architecture for CyberAIBot
"""

import asyncio
import logging
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
import structlog

from .technical_cluster import TechnicalCluster
from .management_cluster import ManagementCluster
from .data_processor import DataProcessor
from .monitor import SystemMonitor

logger = structlog.get_logger(__name__)


class CyberAIBot:
    """
    Main CyberAIBot system orchestrating all components
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.technical_clusters: Dict[str, TechnicalCluster] = {}
        self.management_cluster: Optional[ManagementCluster] = None
        self.data_processor: Optional[DataProcessor] = None
        self.monitor: Optional[SystemMonitor] = None
        self.is_running = False
        
        # Setup logging
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration: {e}")
            raise
    
    def _setup_logging(self):
        """Setup structured logging"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing CyberAIBot system")
        
        try:
            # Initialize data processor
            self.data_processor = DataProcessor(self.config['data_processing'])
            await self.data_processor.initialize()
            
            # Initialize technical clusters
            await self._initialize_technical_clusters()
            
            # Initialize management cluster
            self.management_cluster = ManagementCluster(
                self.config['management_cluster'],
                self.technical_clusters
            )
            await self.management_cluster.initialize()
            
            # Initialize system monitor
            self.monitor = SystemMonitor(self.config['monitoring'])
            await self.monitor.initialize()
            
            logger.info("CyberAIBot system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def _initialize_technical_clusters(self):
        """Initialize technical clusters for each attack type"""
        attack_types = self.config['technical_clusters']['attack_types']
        model_types = self.config['technical_clusters']['model_types']
        
        for attack_type in attack_types:
            for model_type in model_types:
                cluster_id = f"{attack_type}_{model_type}"
                
                cluster = TechnicalCluster(
                    cluster_id=cluster_id,
                    attack_type=attack_type,
                    model_type=model_type,
                    config=self.config['models'][model_type]
                )
                
                await cluster.initialize()
                self.technical_clusters[cluster_id] = cluster
                
                logger.info(f"Initialized technical cluster: {cluster_id}")
    
    async def start(self):
        """Start the CyberAIBot system"""
        if self.is_running:
            logger.warning("System is already running")
            return
        
        logger.info("Starting CyberAIBot system")
        
        try:
            # Start all components
            await self.monitor.start()
            await self.management_cluster.start()
            
            for cluster in self.technical_clusters.values():
                await cluster.start()
            
            self.is_running = True
            logger.info("CyberAIBot system started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            raise
    
    async def stop(self):
        """Stop the CyberAIBot system"""
        if not self.is_running:
            logger.warning("System is not running")
            return
        
        logger.info("Stopping CyberAIBot system")
        
        try:
            # Stop all components
            for cluster in self.technical_clusters.values():
                await cluster.stop()
            
            await self.management_cluster.stop()
            await self.monitor.stop()
            
            self.is_running = False
            logger.info("CyberAIBot system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
            raise
    
    async def process_traffic(self, traffic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming IoT traffic data through the system
        
        Args:
            traffic_data: Raw traffic data from IoT devices
            
        Returns:
            Classification results with confidence scores
        """
        if not self.is_running:
            raise RuntimeError("System is not running")
        
        try:
            # Preprocess the data
            processed_data = await self.data_processor.process(traffic_data)
            
            # Get predictions from technical clusters
            cluster_predictions = {}
            for cluster_id, cluster in self.technical_clusters.items():
                prediction = await cluster.predict(processed_data)
                cluster_predictions[cluster_id] = prediction
            
            # Get final decision from management cluster
            final_decision = await self.management_cluster.decide(
                cluster_predictions, processed_data
            )
            
            # Log the decision
            logger.info(
                "Traffic processed",
                attack_type=final_decision['attack_type'],
                confidence=final_decision['confidence'],
                severity=final_decision['severity']
            )
            
            return final_decision
            
        except Exception as e:
            logger.error(f"Error processing traffic: {e}")
            raise
    
    async def add_new_cluster(self, attack_type: str, model_type: str, 
                            model_data: bytes) -> str:
        """
        Dynamically add a new technical cluster for a new attack type
        
        Args:
            attack_type: Type of attack the cluster will detect
            model_type: Type of model (lstm or svm)
            model_data: Serialized model data
            
        Returns:
            Cluster ID of the newly added cluster
        """
        cluster_id = f"{attack_type}_{model_type}"
        
        if cluster_id in self.technical_clusters:
            raise ValueError(f"Cluster {cluster_id} already exists")
        
        logger.info(f"Adding new technical cluster: {cluster_id}")
        
        # Create new cluster
        cluster = TechnicalCluster(
            cluster_id=cluster_id,
            attack_type=attack_type,
            model_type=model_type,
            config=self.config['models'][model_type]
        )
        
        # Load the model data
        await cluster.load_model(model_data)
        await cluster.initialize()
        
        # Add to system
        self.technical_clusters[cluster_id] = cluster
        await cluster.start()
        
        # Update management cluster
        await self.management_cluster.add_cluster(cluster_id, cluster)
        
        logger.info(f"Successfully added cluster: {cluster_id}")
        return cluster_id
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics"""
        return {
            "is_running": self.is_running,
            "clusters": {
                cluster_id: cluster.get_status() 
                for cluster_id, cluster in self.technical_clusters.items()
            },
            "management_cluster": self.management_cluster.get_status() if self.management_cluster else None,
            "monitor": self.monitor.get_metrics() if self.monitor else None
        }