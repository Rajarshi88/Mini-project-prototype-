"""
Web dashboard server for CyberAIBot monitoring
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Dict, Any
import structlog
from datetime import datetime, timedelta
import json

logger = structlog.get_logger(__name__)


def create_dashboard_app(cyberai_bot) -> FastAPI:
    """Create dashboard FastAPI application"""
    
    app = FastAPI(title="CyberAIBot Dashboard")
    
    # Setup templates
    templates = Jinja2Templates(directory="src/dashboard/templates")
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard_home(request: Request):
        """Main dashboard page"""
        try:
            # Get system status
            status = cyberai_bot.get_system_status()
            
            # Get recent decisions
            recent_decisions = []
            if cyberai_bot.management_cluster:
                recent_decisions = cyberai_bot.management_cluster.get_decision_history(10)
            
            # Get system metrics
            metrics = {}
            if cyberai_bot.monitor:
                metrics = cyberai_bot.monitor.get_metrics()
            
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "status": status,
                "recent_decisions": recent_decisions,
                "metrics": metrics,
                "timestamp": datetime.now()
            })
            
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error": str(e)
            })
    
    @app.get("/api/status")
    async def api_status():
        """API endpoint for system status"""
        try:
            status = cyberai_bot.get_system_status()
            return JSONResponse(content=status)
        except Exception as e:
            logger.error(f"API status error: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
    
    @app.get("/api/metrics")
    async def api_metrics():
        """API endpoint for system metrics"""
        try:
            if cyberai_bot.monitor:
                metrics = cyberai_bot.monitor.get_metrics()
                return JSONResponse(content=metrics)
            else:
                return JSONResponse(content={"error": "Monitor not available"})
        except Exception as e:
            logger.error(f"API metrics error: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
    
    @app.get("/api/decisions")
    async def api_decisions(limit: int = 50):
        """API endpoint for decision history"""
        try:
            if cyberai_bot.management_cluster:
                decisions = cyberai_bot.management_cluster.get_decision_history(limit)
                return JSONResponse(content=decisions)
            else:
                return JSONResponse(content=[])
        except Exception as e:
            logger.error(f"API decisions error: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
    
    @app.get("/api/conflicts")
    async def api_conflicts(limit: int = 20):
        """API endpoint for conflict history"""
        try:
            if cyberai_bot.management_cluster:
                conflicts = cyberai_bot.management_cluster.get_conflict_history(limit)
                return JSONResponse(content=conflicts)
            else:
                return JSONResponse(content=[])
        except Exception as e:
            logger.error(f"API conflicts error: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
    
    @app.get("/clusters", response_class=HTMLResponse)
    async def clusters_page(request: Request):
        """Clusters management page"""
        try:
            status = cyberai_bot.get_system_status()
            clusters = status.get('clusters', {})
            
            return templates.TemplateResponse("clusters.html", {
                "request": request,
                "clusters": clusters,
                "timestamp": datetime.now()
            })
            
        except Exception as e:
            logger.error(f"Clusters page error: {e}")
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error": str(e)
            })
    
    @app.get("/monitoring", response_class=HTMLResponse)
    async def monitoring_page(request: Request):
        """System monitoring page"""
        try:
            metrics = {}
            if cyberai_bot.monitor:
                metrics = cyberai_bot.monitor.get_metrics()
            
            return templates.TemplateResponse("monitoring.html", {
                "request": request,
                "metrics": metrics,
                "timestamp": datetime.now()
            })
            
        except Exception as e:
            logger.error(f"Monitoring page error: {e}")
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error": str(e)
            })
    
    @app.get("/threats", response_class=HTMLResponse)
    async def threats_page(request: Request):
        """Threat detection page"""
        try:
            # Get recent decisions with threats
            recent_decisions = []
            if cyberai_bot.management_cluster:
                all_decisions = cyberai_bot.management_cluster.get_decision_history(100)
                # Filter for non-benign decisions
                recent_decisions = [
                    d for d in all_decisions 
                    if d['decision']['attack_type'] != 'Benign'
                ][:20]
            
            return templates.TemplateResponse("threats.html", {
                "request": request,
                "threats": recent_decisions,
                "timestamp": datetime.now()
            })
            
        except Exception as e:
            logger.error(f"Threats page error: {e}")
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error": str(e)
            })
    
    return app