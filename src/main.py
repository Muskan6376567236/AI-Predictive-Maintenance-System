import os
import logging
from flask import Flask, request, jsonify
from src.api.prediction import prediction_bp
from src.models.lstm_model import LSTMModel
from src.utils.data_processor import DataProcessor

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register blueprints
app.register_blueprint(prediction_bp, url_prefix='/api/v1')

# Load model and data processor on startup
model = LSTMModel()
data_processor = DataProcessor()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)