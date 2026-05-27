import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic sensor data for 1000 time steps
np.random.seed(42)
n_samples = 1000

timestamps = [datetime(2026, 1, 1) + timedelta(minutes=i) for i in range(n_samples)]

# Simulate sensor readings (vibration, temperature, pressure)
vibration = np.random.normal(50, 10, n_samples) + np.sin(np.linspace(0, 10, n_samples)) * 5
temperature = np.random.normal(75, 5, n_samples) + np.random.normal(0, 2, n_samples)
pressure = np.random.normal(100, 15, n_samples)

# Introduce anomalies for the last 50 samples (simulating impending failure)
anomaly_indices = slice(-50, None)
vibration[anomaly_indices] += np.random.normal(20, 5, 50)
temperature[anomaly_indices] += np.random.normal(15, 3, 50)

df = pd.DataFrame({
    'timestamp': timestamps,
    'vibration': vibration,
    'temperature': temperature,
    'pressure': pressure,
    'equipment_id': 'EQ-001',
    'status': ['normal'] * 950 + ['failure_imminent'] * 50
})

df.to_csv('data/sample_sensor_data.csv', index=False)
print(f"Generated {len(df)} samples of synthetic sensor data")
print("Saved to data/sample_sensor_data.csv")