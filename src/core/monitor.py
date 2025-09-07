"""
System monitoring and metrics collection
"""

import asyncio
import psutil
import time
from typing import Dict, List, Any, Optional
import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import threading

logger = structlog.get_logger(__name__)


class SystemMonitor:
    """
    System monitoring for CyberAIBot
    Collects metrics and provides health monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
        self.is_running = False
        self.metrics_server = None
        
        # Prometheus metrics
        self.prediction_counter = Counter(
            'cyberai_predictions_total',
            'Total number of predictions made',
            ['attack_type', 'cluster_id']
        )
        
        self.prediction_duration = Histogram(
            'cyberai_prediction_duration_seconds',
            'Time spent on predictions',
            ['cluster_id']
        )
        
        self.system_cpu_usage = Gauge(
            'cyberai_system_cpu_usage_percent',
            'System CPU usage percentage'
        )
        
        self.system_memory_usage = Gauge(
            'cyberai_system_memory_usage_percent',
            'System memory usage percentage'
        )
        
        self.active_clusters = Gauge(
            'cyberai_active_clusters',
            'Number of active technical clusters'
        )
        
        self.conflict_counter = Counter(
            'cyberai_conflicts_total',
            'Total number of conflicts detected',
            ['conflict_type']
        )
        
        # Monitoring configuration
        self.metrics_port = config.get('metrics_port', 9090)
        self.health_check_interval = config.get('health_check_interval', 30)
        self.alert_thresholds = config.get('alert_thresholds', {})
        
        # System metrics
        self.system_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'network_io': {'bytes_sent': 0, 'bytes_recv': 0},
            'last_update': time.time()
        }
        
        # Health status
        self.health_status = {
            'status': 'unknown',
            'last_check': None,
            'alerts': []
        }
    
    async def initialize(self):
        """Initialize the system monitor"""
        logger.info("Initializing system monitor")
        
        try:
            # Start Prometheus metrics server
            self.metrics_server = threading.Thread(
                target=start_http_server,
                args=(self.metrics_port,),
                daemon=True
            )
            self.metrics_server.start()
            
            self.is_initialized = True
            logger.info(f"System monitor initialized on port {self.metrics_port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize system monitor: {e}")
            raise
    
    async def start(self):
        """Start the system monitor"""
        if not self.is_initialized:
            raise RuntimeError("System monitor not initialized")
        
        if self.is_running:
            logger.warning("System monitor is already running")
            return
        
        self.is_running = True
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_system_metrics())
        asyncio.create_task(self._health_check_loop())
        
        logger.info("System monitor started")
    
    async def stop(self):
        """Stop the system monitor"""
        if not self.is_running:
            logger.warning("System monitor is not running")
            return
        
        self.is_running = False
        logger.info("System monitor stopped")
    
    async def _monitor_system_metrics(self):
        """Continuously monitor system metrics"""
        while self.is_running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.system_metrics['cpu_usage'] = cpu_percent
                self.system_cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.system_metrics['memory_usage'] = memory_percent
                self.system_memory_usage.set(memory_percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.system_metrics['disk_usage'] = disk_percent
                
                # Network I/O
                network = psutil.net_io_counters()
                self.system_metrics['network_io'] = {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                }
                
                self.system_metrics['last_update'] = time.time()
                
                # Check for alerts
                await self._check_alerts()
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring system metrics: {e}")
                await asyncio.sleep(10)
    
    async def _health_check_loop(self):
        """Perform periodic health checks"""
        while self.is_running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                await asyncio.sleep(30)
    
    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        health_issues = []
        
        # Check CPU usage
        cpu_threshold = self.alert_thresholds.get('cpu_usage', 80)
        if self.system_metrics['cpu_usage'] > cpu_threshold:
            health_issues.append(f"High CPU usage: {self.system_metrics['cpu_usage']:.1f}%")
        
        # Check memory usage
        memory_threshold = self.alert_thresholds.get('memory_usage', 85)
        if self.system_metrics['memory_usage'] > memory_threshold:
            health_issues.append(f"High memory usage: {self.system_metrics['memory_usage']:.1f}%")
        
        # Check disk usage
        if self.system_metrics['disk_usage'] > 90:
            health_issues.append(f"High disk usage: {self.system_metrics['disk_usage']:.1f}%")
        
        # Update health status
        if health_issues:
            self.health_status['status'] = 'unhealthy'
            self.health_status['alerts'] = health_issues
        else:
            self.health_status['status'] = 'healthy'
            self.health_status['alerts'] = []
        
        self.health_status['last_check'] = time.time()
        
        if health_issues:
            logger.warning(f"Health check failed: {health_issues}")
        else:
            logger.debug("Health check passed")
    
    async def _check_alerts(self):
        """Check for alert conditions"""
        alerts = []
        
        # CPU alert
        cpu_threshold = self.alert_thresholds.get('cpu_usage', 80)
        if self.system_metrics['cpu_usage'] > cpu_threshold:
            alerts.append({
                'type': 'cpu_usage',
                'value': self.system_metrics['cpu_usage'],
                'threshold': cpu_threshold,
                'timestamp': time.time()
            })
        
        # Memory alert
        memory_threshold = self.alert_thresholds.get('memory_usage', 85)
        if self.system_metrics['memory_usage'] > memory_threshold:
            alerts.append({
                'type': 'memory_usage',
                'value': self.system_metrics['memory_usage'],
                'threshold': memory_threshold,
                'timestamp': time.time()
            })
        
        # Log alerts
        for alert in alerts:
            logger.warning(
                f"Alert triggered: {alert['type']} = {alert['value']:.1f}% "
                f"(threshold: {alert['threshold']}%)"
            )
    
    def record_prediction(self, cluster_id: str, attack_type: str, duration: float):
        """Record prediction metrics"""
        self.prediction_counter.labels(
            attack_type=attack_type,
            cluster_id=cluster_id
        ).inc()
        
        self.prediction_duration.labels(cluster_id=cluster_id).observe(duration)
    
    def record_conflict(self, conflict_type: str):
        """Record conflict metrics"""
        self.conflict_counter.labels(conflict_type=conflict_type).inc()
    
    def update_active_clusters(self, count: int):
        """Update active clusters count"""
        self.active_clusters.set(count)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            'system_metrics': self.system_metrics.copy(),
            'health_status': self.health_status.copy(),
            'prometheus_port': self.metrics_port,
            'is_running': self.is_running
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return self.health_status.copy()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return self.system_metrics.copy()