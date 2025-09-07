"""
FastAPI server for CyberAIBot real-time processing
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import asyncio
import structlog
from datetime import datetime
import uuid

logger = structlog.get_logger(__name__)


# Pydantic models for API
class TrafficData(BaseModel):
    """IoT traffic data model"""
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    packet_size: int
    flow_duration: Optional[float] = None
    payload: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PredictionRequest(BaseModel):
    """Prediction request model"""
    traffic_data: TrafficData
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))


class PredictionResponse(BaseModel):
    """Prediction response model"""
    request_id: str
    attack_type: str
    prediction: str
    confidence: float
    severity: str
    method: str
    conflicts_detected: int
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


class SystemStatus(BaseModel):
    """System status model"""
    is_running: bool
    clusters: Dict[str, Any]
    management_cluster: Optional[Dict[str, Any]]
    monitor: Optional[Dict[str, Any]]


class ClusterInfo(BaseModel):
    """Cluster information model"""
    cluster_id: str
    attack_type: str
    model_type: str
    is_trained: bool
    prediction_count: int
    avg_prediction_time: float


def create_app(cyberai_bot) -> FastAPI:
    """Create FastAPI application"""
    
    app = FastAPI(
        title="CyberAIBot API",
        description="AI-Driven Intrusion Detection System for IoT",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint"""
        return {
            "message": "CyberAIBot API",
            "version": "1.0.0",
            "status": "running"
        }
    
    @app.get("/health", response_model=Dict[str, Any])
    async def health_check():
        """Health check endpoint"""
        try:
            status = cyberai_bot.get_system_status()
            return {
                "status": "healthy" if status['is_running'] else "unhealthy",
                "timestamp": datetime.now(),
                "system_status": status
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/status", response_model=SystemStatus)
    async def get_system_status():
        """Get system status"""
        try:
            status = cyberai_bot.get_system_status()
            return SystemStatus(**status)
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/clusters", response_model=List[ClusterInfo])
    async def get_clusters():
        """Get all technical clusters information"""
        try:
            status = cyberai_bot.get_system_status()
            clusters = []
            
            for cluster_id, cluster_status in status['clusters'].items():
                clusters.append(ClusterInfo(
                    cluster_id=cluster_id,
                    attack_type=cluster_status['attack_type'],
                    model_type=cluster_status['model_type'],
                    is_trained=cluster_status['is_trained'],
                    prediction_count=cluster_status['prediction_count'],
                    avg_prediction_time=cluster_status['avg_prediction_time']
                ))
            
            return clusters
        except Exception as e:
            logger.error(f"Failed to get clusters: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict_traffic(request: PredictionRequest, background_tasks: BackgroundTasks):
        """Process IoT traffic and get prediction"""
        try:
            import time
            start_time = time.time()
            
            # Convert request to dict
            traffic_dict = request.traffic_data.dict()
            
            # Process traffic through CyberAIBot
            result = await cyberai_bot.process_traffic(traffic_dict)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Record metrics in background
            background_tasks.add_task(
                record_prediction_metrics,
                result,
                processing_time
            )
            
            # Create response
            response = PredictionResponse(
                request_id=request.request_id,
                attack_type=result['attack_type'],
                prediction=result['prediction'],
                confidence=result['confidence'],
                severity=result['severity'],
                method=result.get('method', 'unknown'),
                conflicts_detected=result.get('conflicts_detected', 0),
                processing_time=processing_time,
                metadata=result.get('metadata', {})
            )
            
            logger.info(
                "Traffic prediction completed",
                request_id=request.request_id,
                attack_type=result['attack_type'],
                confidence=result['confidence'],
                processing_time=processing_time
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict/batch", response_model=List[PredictionResponse])
    async def predict_traffic_batch(requests: List[PredictionRequest]):
        """Process multiple traffic samples in batch"""
        try:
            results = []
            
            for request in requests:
                try:
                    # Convert request to dict
                    traffic_dict = request.traffic_data.dict()
                    
                    # Process traffic
                    result = await cyberai_bot.process_traffic(traffic_dict)
                    
                    # Create response
                    response = PredictionResponse(
                        request_id=request.request_id,
                        attack_type=result['attack_type'],
                        prediction=result['prediction'],
                        confidence=result['confidence'],
                        severity=result['severity'],
                        method=result.get('method', 'unknown'),
                        conflicts_detected=result.get('conflicts_detected', 0),
                        processing_time=0.0,  # Batch processing time not calculated
                        metadata=result.get('metadata', {})
                    )
                    
                    results.append(response)
                    
                except Exception as e:
                    logger.error(f"Batch prediction failed for request {request.request_id}: {e}")
                    # Add error response
                    results.append(PredictionResponse(
                        request_id=request.request_id,
                        attack_type="Error",
                        prediction="Error",
                        confidence=0.0,
                        severity="High",
                        method="error",
                        conflicts_detected=0,
                        processing_time=0.0,
                        metadata={"error": str(e)}
                    ))
            
            logger.info(f"Batch prediction completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/clusters/{cluster_id}/train")
    async def train_cluster(cluster_id: str, training_data: Dict[str, Any]):
        """Train a specific cluster"""
        try:
            if cluster_id not in cyberai_bot.technical_clusters:
                raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
            
            cluster = cyberai_bot.technical_clusters[cluster_id]
            
            # Extract training data
            X_train = training_data.get('X_train')
            y_train = training_data.get('y_train')
            X_val = training_data.get('X_val')
            y_val = training_data.get('y_val')
            class_names = training_data.get('class_names', [])
            
            if not all([X_train, y_train, X_val, y_val]):
                raise HTTPException(status_code=400, detail="Missing required training data")
            
            # Train cluster
            result = await cluster.train(
                X_train, y_train, X_val, y_val, class_names
            )
            
            logger.info(f"Cluster {cluster_id} training completed")
            return {"message": f"Cluster {cluster_id} trained successfully", "result": result}
            
        except Exception as e:
            logger.error(f"Failed to train cluster {cluster_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/clusters/add")
    async def add_new_cluster(
        attack_type: str,
        model_type: str,
        model_data: bytes
    ):
        """Add a new technical cluster"""
        try:
            cluster_id = await cyberai_bot.add_new_cluster(
                attack_type, model_type, model_data
            )
            
            logger.info(f"New cluster added: {cluster_id}")
            return {"message": f"Cluster {cluster_id} added successfully", "cluster_id": cluster_id}
            
        except Exception as e:
            logger.error(f"Failed to add cluster: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/metrics")
    async def get_metrics():
        """Get system metrics"""
        try:
            if cyberai_bot.monitor:
                return cyberai_bot.monitor.get_metrics()
            else:
                return {"error": "Monitor not available"}
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/decisions/history")
    async def get_decision_history(limit: int = 100):
        """Get decision history"""
        try:
            if cyberai_bot.management_cluster:
                return cyberai_bot.management_cluster.get_decision_history(limit)
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to get decision history: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/conflicts/history")
    async def get_conflict_history(limit: int = 50):
        """Get conflict history"""
        try:
            if cyberai_bot.management_cluster:
                return cyberai_bot.management_cluster.get_conflict_history(limit)
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to get conflict history: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def record_prediction_metrics(result: Dict[str, Any], processing_time: float):
        """Record prediction metrics in background"""
        try:
            if cyberai_bot.monitor:
                # Record prediction metrics
                cyberai_bot.monitor.record_prediction(
                    cluster_id="management",
                    attack_type=result['attack_type'],
                    duration=processing_time
                )
                
                # Record conflicts if any
                if result.get('conflicts_detected', 0) > 0:
                    cyberai_bot.monitor.record_conflict("prediction_conflict")
        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")
    
    return app