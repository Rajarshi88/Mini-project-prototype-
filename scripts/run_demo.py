"""
Quick demo script to showcase CyberAIBot capabilities
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from example_usage import main

if __name__ == "__main__":
    print("Starting CyberAIBot demonstration...")
    asyncio.run(main())