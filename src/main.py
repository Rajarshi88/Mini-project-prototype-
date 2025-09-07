"""
Main entry point for CyberAIBot system
"""

import asyncio
import signal
import sys
import argparse
from typing import Dict, Any
import structlog
import yaml
from pathlib import Path

from .core.system import CyberAIBot
from .api.server import create_app
from .dashboard.server import create_dashboard_app

logger = structlog.get_logger(__name__)


class CyberAIBotApplication:
    """
    Main application class for CyberAIBot
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.cyberai_bot = None
        self.api_app = None
        self.dashboard_app = None
        self.is_running = False
        
    async def initialize(self):
        """Initialize the application"""
        logger.info("Initializing CyberAIBot application")
        
        try:
            # Initialize CyberAIBot system
            self.cyberai_bot = CyberAIBot(self.config_path)
            await self.cyberai_bot.initialize()
            
            # Initialize API server
            self.api_app = create_app(self.cyberai_bot)
            
            # Initialize dashboard
            self.dashboard_app = create_dashboard_app(self.cyberai_bot)
            
            logger.info("CyberAIBot application initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            raise
    
    async def start(self):
        """Start the application"""
        if self.is_running:
            logger.warning("Application is already running")
            return
        
        logger.info("Starting CyberAIBot application")
        
        try:
            # Start CyberAIBot system
            await self.cyberai_bot.start()
            
            # Start API server
            import uvicorn
            api_config = self.cyberai_bot.config['api']
            
            api_server = uvicorn.Server(
                uvicorn.Config(
                    app=self.api_app,
                    host=api_config['host'],
                    port=api_config['port'],
                    workers=api_config['workers'],
                    timeout_keep_alive=api_config['timeout']
                )
            )
            
            # Start dashboard server
            dashboard_config = self.cyberai_bot.config['dashboard']
            
            dashboard_server = uvicorn.Server(
                uvicorn.Config(
                    app=self.dashboard_app,
                    host=dashboard_config['host'],
                    port=dashboard_config['port'],
                    auto_reload=dashboard_config['auto_reload']
                )
            )
            
            # Start servers concurrently
            await asyncio.gather(
                api_server.serve(),
                dashboard_server.serve()
            )
            
            self.is_running = True
            logger.info("CyberAIBot application started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            raise
    
    async def stop(self):
        """Stop the application"""
        if not self.is_running:
            logger.warning("Application is not running")
            return
        
        logger.info("Stopping CyberAIBot application")
        
        try:
            # Stop CyberAIBot system
            if self.cyberai_bot:
                await self.cyberai_bot.stop()
            
            self.is_running = False
            logger.info("CyberAIBot application stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping application: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get application status"""
        return {
            'is_running': self.is_running,
            'cyberai_bot_status': self.cyberai_bot.get_system_status() if self.cyberai_bot else None
        }


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='CyberAIBot - AI-Driven Intrusion Detection System')
    parser.add_argument('--config', default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--train', action='store_true', help='Train models before starting')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models before starting')
    
    args = parser.parse_args()
    
    # Create application
    app = CyberAIBotApplication(args.config)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(app.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize application
        await app.initialize()
        
        # Train models if requested
        if args.train:
            logger.info("Training models...")
            from .scripts.train_models import train_all_models
            await train_all_models(app.cyberai_bot)
        
        # Evaluate models if requested
        if args.evaluate:
            logger.info("Evaluating models...")
            from .scripts.evaluate_models import evaluate_all_models
            await evaluate_all_models(app.cyberai_bot)
        
        # Start application
        await app.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
    finally:
        await app.stop()


if __name__ == "__main__":
    asyncio.run(main())