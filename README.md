# CyberAIBot - AI-Driven Intrusion Detection System for IoT

## Overview
CyberAIBot is an advanced AI-enhanced Intrusion Detection System (IDS) designed specifically for IoT and edge device cybersecurity. The system uses Deep Learning Technical Clusters specialized in detecting specific attack types, supervised by a Management Cluster for real-time security decisions.

## Architecture
- **Technical Clusters**: Specialized LSTM and SVM models for different attack types
- **Management Cluster**: Supervises technical clusters and resolves conflicting classifications
- **Edge Deployment**: Local/edge cloud computing for reduced latency
- **Dynamic Scalability**: Add new clusters without retraining existing ones

## Features
- Real-time IoT traffic classification
- Support for multiple attack types (DoS, DDoS, Botnet, Brute Force, etc.)
- Edge-compatible deployment
- RESTful API for monitoring and control
- Web dashboard for threat visualization

## Project Structure
```
cyberai_bot/
├── src/
│   ├── core/           # Core system components
│   ├── models/         # LSTM and SVM model implementations
│   ├── preprocessing/  # Data preprocessing pipeline
│   ├── management/     # Management cluster logic
│   ├── api/           # REST API endpoints
│   └── dashboard/     # Web dashboard
├── data/              # Dataset storage and processing
├── config/            # Configuration files
├── docker/            # Docker deployment files
├── tests/             # Unit and integration tests
└── scripts/           # Training and evaluation scripts
```

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Download datasets to `data/` directory
3. Train models: `python scripts/train_models.py`
4. Start the system: `python src/main.py`
5. Access dashboard: http://localhost:8080

## Performance Metrics
- Accuracy, Precision, Recall, F1-score
- Real-time processing latency
- Memory usage optimization
- Scalability benchmarks