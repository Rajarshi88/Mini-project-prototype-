"""
Example usage of CyberAIBot system
Demonstrates training, evaluation, and real-time processing
"""

import asyncio
import numpy as np
from datetime import datetime
import structlog

from src.core.system import CyberAIBot
from scripts.train_models import train_all_models
from scripts.evaluate_models import evaluate_all_models
from scripts.demo_scalability import run_scalability_demo

logger = structlog.get_logger(__name__)


async def main():
    """Main example function"""
    print("="*60)
    print("CYBERAI BOT - AI-DRIVEN INTRUSION DETECTION SYSTEM")
    print("="*60)
    
    try:
        # Initialize CyberAIBot system
        print("\n1. Initializing CyberAIBot system...")
        cyberai_bot = CyberAIBot("config/config.yaml")
        await cyberai_bot.initialize()
        await cyberai_bot.start()
        
        # Get system status
        status = cyberai_bot.get_system_status()
        print(f"   ✓ System initialized with {len(status['clusters'])} technical clusters")
        print(f"   ✓ Management cluster: {'Active' if status['management_cluster'] else 'Inactive'}")
        print(f"   ✓ System monitor: {'Active' if status['monitor'] else 'Inactive'}")
        
        # Train models (optional - requires datasets)
        print("\n2. Training models...")
        try:
            training_results = await train_all_models(cyberai_bot)
            print(f"   ✓ Training completed for {len(training_results)} clusters")
        except Exception as e:
            print(f"   ⚠ Training skipped (datasets not available): {e}")
        
        # Evaluate models
        print("\n3. Evaluating models...")
        try:
            evaluation_results = await evaluate_all_models(cyberai_bot)
            print("   ✓ Model evaluation completed")
        except Exception as e:
            print(f"   ⚠ Evaluation skipped: {e}")
        
        # Test real-time processing
        print("\n4. Testing real-time traffic processing...")
        sample_traffic = {
            'timestamp': datetime.now().isoformat(),
            'src_ip': '192.168.1.100',
            'dst_ip': '10.0.0.50',
            'src_port': 12345,
            'dst_port': 80,
            'protocol': 'TCP',
            'packet_size': 1024,
            'flow_duration': 1.5
        }
        
        result = await cyberai_bot.process_traffic(sample_traffic)
        print(f"   ✓ Traffic processed: {result['attack_type']} (confidence: {result['confidence']:.3f})")
        print(f"   ✓ Severity: {result['severity']}")
        print(f"   ✓ Conflicts detected: {result.get('conflicts_detected', 0)}")
        
        # Demonstrate scalability
        print("\n5. Demonstrating dynamic scalability...")
        try:
            await run_scalability_demo()
            print("   ✓ Scalability demonstration completed")
        except Exception as e:
            print(f"   ⚠ Scalability demo skipped: {e}")
        
        # System metrics
        print("\n6. System metrics:")
        if cyberai_bot.monitor:
            metrics = cyberai_bot.monitor.get_metrics()
            print(f"   ✓ CPU Usage: {metrics['system_metrics']['cpu_usage']:.1f}%")
            print(f"   ✓ Memory Usage: {metrics['system_metrics']['memory_usage']:.1f}%")
            print(f"   ✓ Health Status: {metrics['health_status']['status']}")
        
        print("\n" + "="*60)
        print("CYBERAI BOT SYSTEM READY FOR PRODUCTION")
        print("="*60)
        print("API Server: http://localhost:8000")
        print("Dashboard: http://localhost:8080")
        print("Metrics: http://localhost:9090")
        print("="*60)
        
        # Keep system running for demonstration
        print("\nSystem is running. Press Ctrl+C to stop...")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down system...")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"Error: {e}")
    finally:
        if 'cyberai_bot' in locals():
            await cyberai_bot.stop()
        print("System stopped.")


if __name__ == "__main__":
    asyncio.run(main())