import pandas as pd
import numpy as np
from src.models.lstm_model import LSTMModel
from src.utils.data_processor import DataProcessor

def run_demo():
    print("AI Predictive Maintenance System Demo")
    print("=" * 50)
    
    # Load sample data
    print("Loading sample sensor data...")
    df = pd.read_csv('data/sample_sensor_data.csv')
    print(f"Loaded {len(df)} records")
    
    # Display last few records
    print("\nLast 5 sensor readings:")
    print(df.tail())
    
    # Preprocess data
    print("\nPreprocessing data...")
    processor = DataProcessor(sequence_length=10)
    sensor_data = df[['vibration', 'temperature', 'pressure']].values
    processed = processor.preprocess(sensor_data)
    
    # Create sequences
    sequences = []
    for i in range(len(processed) - 10):
        sequences.append(processed[i:i+10])
    
    X = np.array(sequences)
    print(f"Created {len(X)} sequences for prediction")
    
    # Make prediction
    print("\nRunning prediction on latest data...")
    model = LSTMModel(sequence_length=10, n_features=3)
    # Model returns random prediction since not trained
    prediction = model.predict(X[-1:])
    
    print(f"Failure probability for last sample: {prediction[0][0]:.4f}")
    
    if prediction[0][0] > 0.7:
        print("⚠️  MAINTENANCE RECOMMENDED - High failure probability detected!")
    else:
        print("✅ Equipment operating normally")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    run_demo()