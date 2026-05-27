import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging
import os

logger = logging.getLogger(__name__)

class LSTMModel:
    def __init__(self, sequence_length=10, n_features=1, lstm_units=50, dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        self.is_trained = False
    
    def _build_model(self):
        """Build the LSTM model"""
        model = Sequential([
            LSTM(self.lstm_units, activation='relu', input_shape=(self.sequence_length, self.n_features), return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units//2, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(self.n_features, activation='sigmoid')  # Output between 0 and 1 for probability
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def predict(self, data):
        """Make a prediction on the input data"""
        try:
            if not self.is_trained:
                # If the model is not trained, return a dummy prediction (in a real scenario, you would load a pre-trained model)
                logger.warning("Model is not trained. Returning dummy prediction.")
                # Return a random probability between 0 and 1 for demonstration
                return np.random.rand(data.shape[0], 1)
            
            # Make prediction
            predictions = self.model.predict(data)
            return predictions
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Return a dummy prediction in case of error
            return np.random.rand(data.shape[0], 1)
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM model"""
        try:
            logger.info("Starting model training...")
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1
            )
            self.is_trained = True
            logger.info("Model training completed.")
            return history
        
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def save_model(self, filepath):
        """Save the trained model to a file"""
        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath):
        """Load a trained model from a file"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise