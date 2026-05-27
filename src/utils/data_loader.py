import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
    
    def load_sensor_data(self, file_path):
        """Load sensor data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading sensor data: {str(e)}")
            raise
    
    def load_timeseries_data(self, df, column='value'):
        """Extract time series data from dataframe"""
        if column in df.columns:
            return df[column].values.reshape(-1, 1)
        else:
            raise ValueError(f"Column '{column}' not found in dataframe")
    
    def normalize_data(self, data):
        """Normalize data to range [0, 1]"""
        try:
            if not self.is_fitted:
                self.scaler.fit(data)
                self.is_fitted = True
            return self.scaler.transform(data)
        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            raise
    
    def create_sequences(self, data, sequence_length=10):
        """Create sequences for time series prediction"""
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)