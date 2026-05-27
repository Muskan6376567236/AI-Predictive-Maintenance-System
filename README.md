# AI-Predictive-Maintenance-System

Advanced AI-powered predictive maintenance system with real-time sensor data processing, deep learning models for failure prediction, and automated maintenance scheduling.

## Features

- Real-time sensor data ingestion and processing
- Deep learning models for equipment failure prediction (LSTM, CNN, Random Forest)
- Automated maintenance scheduling and work order generation
- Anomaly detection for early fault identification
- RESTful API for integration with existing systems
- Dashboard for monitoring equipment health and predictions
- Cloud-native deployment with Docker and Kubernetes
- Comprehensive logging and monitoring
- CI/CD with GitHub Actions (automated testing)

## System Architecture

```
+------------------+     +------------------+     +------------------+
|   Sensor Data    |     |   Data Preproc   |     |  Feature Eng     |
|  (IoT Devices)   | --> |   (Cleaning)     | --> |   & Scaling      |
+------------------+     +------------------+     +------------------+
                                   |
                                   v
                         +------------------+
                         |  ML Models       |
                         |  (LSTM, CNN,     |
                         |   Random Forest) |
                         +------------------+
                                   |
                                   v
                         +------------------+
                         |  Prediction      |
                         |  Engine          |
                         +------------------+
                                   |
                                   v
                         +------------------+
                         |  Maintenance     |
                         |  Scheduler       |
                         +------------------+
                                   |
                                   v
                         +------------------+
                         |  Alerting &      |
                         |  Notification    |
                         +------------------+
                                   |
                                   v
                         +------------------+
                         |  Dashboard &     |
                         |  Reporting       |
                         +------------------+
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Muskan6376567236/AI-Predictive-Maintenance-System.git
cd AI-Predictive-Maintenance-System

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run the system
python src/main.py
```

## Requirements

See `requirements.txt` for detailed dependencies.

## API Endpoints

- `POST /api/v1/predict` - Get failure prediction for equipment
- `GET /api/v1/equipment/{id}` - Get equipment status
- `GET /api/v1/maintenance/schedule` - Get maintenance schedule
- `POST /api/v1/alerts` - Submit maintenance alerts

## Model Training

```bash
# Train failure prediction models
python src/models/train.py --data data/training.csv --model lstm

# Evaluate model performance
python src/models/evaluate.py --model models/failure_predictor.pkl
```

## Deployment

### Docker
```bash
docker build -t ai-predictive-maintenance .
docker run -p 8000:8000 ai-predictive-maintenance
```

### Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Monitoring & Logging

- Prometheus metrics endpoint: `/metrics`
- Health check: `/health`
- Structured logging with ELK stack integration
- Grafana dashboards for visualization

## CI/CD

This project uses GitHub Actions for continuous integration. The workflow runs the test suite on every push to `main`.

![CI](https://github.com/Muskan6376567236/AI-Predictive-Maintenance-System/actions/workflows/ci.yml/badge.svg)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

If you find this project useful, please consider giving it a star! It helps others discover the project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Muskan Sethi - muskansethi72@gmail.com

Project Link: https://github.com/Muskan6376567236/AI-Predictive-Maintenance-System