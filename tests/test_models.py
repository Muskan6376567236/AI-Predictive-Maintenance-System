"""
Test suite for AI Predictive Maintenance System
"""
import pytest
import numpy as np
from src.models.lstm_model import LSTMModel
from src.utils.data_processor import DataProcessor

def test_lstm_model_initialization():
    """Test that LSTM model initializes correctly"""
    model = LSTMModel()
    assert model.model is not None
    assert model.sequence_length > 0

def test_data_processor_preprocess():
    """Test data processor preprocessing"""
    processor = DataProcessor(sequence_length=5)
    test_data = np.random.rand(20, 1)
    processed = processor.preprocess(test_data)
    assert processed.shape[0] > 0
    assert processed.shape[1] == 5

def test_data_loader_normalize():
    """Test DataLoader normalization"""
    from src.utils.data_loader import DataLoader
    loader = DataLoader()
    test_data = np.random.rand(100, 1) * 100
    normalized = loader.normalize_data(test_data)
    assert normalized.max() <= 1.0
    assert normalized.min() >= 0.0

def test_lstm_model_dummy_prediction():
    """Test that LSTM model can make predictions (even if not trained)"""
    model = LSTMModel()
    test_input = np.random.rand(1, 10, 1)
    prediction = model.predict(test_input)
    assert prediction.shape == (1, 1)