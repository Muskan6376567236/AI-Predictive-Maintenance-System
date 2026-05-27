import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
    
    def fit(self, data):
        """Fit the scaler on the data"""
        try:
            self.scaler.fit(data)
            self.is_fitted = True
            logger.info("DataProcessor scaler fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting scaler: {str(e)}")
            raise
    
    def preprocess(self, data):
        """Preprocess the input data for prediction"""
        try:
            # Convert to numpy array if it's a list
            if isinstance(data, list):
                data = np.array(data)
            
            # If data is 1D, reshape to 2D for scaling
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Scale the data
            if not self.is_fitted:
                # If not fitted, fit on this data (in production, you would use a pre-fitted scaler)
                self.fit(data)
            
            scaled_data = self.scaler.transform(data)
            
            # Create sequences for LSTM
            sequences = []
            for i in range(len(scaled_data) - self.sequence_length + 1):
                sequences.append(scaled_data[i:i + self.sequence_length])
            
            if len(sequences) == 0:
                # If not enough data for a sequence, pad or return empty
                # For simplicity, we'll return the last available sequence padded
                if len(scaled_data) > 0:
                    # Repeat the last data point to make a sequence
                    last_data = scaled_data[-1:]
                    padding = np.repeat(last_data, self.sequence_length, axis=0)
                    sequences = [padding]
                else:
                    sequences = [np.zeros((self.sequence_length, scaled_data.shape[1]))]
            
            return np.array(sequences)
        
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise