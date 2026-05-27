from flask import Blueprint, request, jsonify
from src.models.lstm_model import LSTMModel
from src.utils.data_processor import DataProcessor
import logging

logger = logging.getLogger(__name__)

prediction_bp = Blueprint('prediction', __name__)

# Initialize model and processor (in a real app, these would be singletons or loaded via dependency injection)
model = LSTMModel()
data_processor = DataProcessor()

@prediction_bp.route('/predict', methods=['POST'])
def predict_failure():
    try:
        # Get sensor data from request
        data = request.get_json()
        if not data or 'sensor_data' not in data:
            return jsonify({"error": "Missing sensor_data in request"}), 400
        
        sensor_data = data['sensor_data']
        
        # Preprocess the data
        processed_data = data_processor.preprocess(sensor_data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        
        # Format response
        response = {
            "equipment_id": data.get('equipment_id', 'unknown'),
            "failure_probability": float(prediction[0]),
            "maintenance_recommended": prediction[0] > 0.7,
            "timestamp": data.get('timestamp', None)
        }
        
        logger.info(f"Prediction made for equipment {response['equipment_id']}: {response['failure_probability']:.4f}")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@prediction_bp.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        if not data or 'training_data' not in data:
            return jsonify({"error": "Missing training_data in request"}), 400
        
        # In a real implementation, this would trigger model training
        # For now, we'll return a placeholder response
        return jsonify({
            "message": "Model training initiated",
            "status": "queued",
            "estimated_completion": "2026-05-28T02:00:00Z"
        }), 202
    
    except Exception as e:
        logger.error(f"Error initiating training: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@prediction_bp.route('/equipment/<equipment_id>', methods=['GET'])
def get_equipment_status(equipment_id):
    try:
        # In a real implementation, this would fetch from a database
        return jsonify({
            "equipment_id": equipment_id,
            "status": "operational",
            "last_maintenance": "2026-05-20T10:00:00Z",
            "next_scheduled_maintenance": "2026-06-20T10:00:00Z"
        }), 200
    
    except Exception as e:
        logger.error(f"Error fetching equipment status: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500