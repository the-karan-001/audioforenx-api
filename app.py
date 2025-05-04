from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import traceback
import psutil
from model.extractor import load_audio, extract_features, DEFAULT_SR
import logging

# Set environment variable to disable Numba JIT
os.environ['NUMBA_DISABLE_JIT'] = '1'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure memory limits for TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        logger.error(f"GPU memory config error: {e}")

# Limit TensorFlow to lower memory usage
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

UPLOAD_FOLDER = "Uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models once at startup
logger.info("Loading ML models...")
try:
    model = load_model('model/deepfake_detection_model.h5',
                       custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    scaler = joblib.load("model/scaler.pkl")
    threshold = float(np.load("model/threshold.npy"))
    feature_names = scaler.feature_names_in_
    logger.info(f"Model expects {len(feature_names)} features")
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    traceback.print_exc()
    model = None
    scaler = None
    threshold = None
    feature_names = None

@app.after_request
def cleanup_after_request(response):
    """Force garbage collection after each request"""
    gc.collect()
    return response

@app.route("/diagnostics", methods=["GET"])
def diagnostics():
    """Return version information for debugging"""
    import sys
    import pandas
    
    versions = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "tensorflow": tf.__version__,
        "model_loaded": model is not None,
        "feature_count": len(feature_names) if feature_names is not None else 0,
        "feature_examples": list(feature_names[:5]) if feature_names is not None else []
    }
    
    try:
        import librosa
        versions["librosa"] = librosa.__version__
    except:
        versions["librosa"] = "not installed"
        
    try:
        import numba
        versions["numba"] = numba.__version__
    except:
        versions["numba"] = "not installed"
    
    return jsonify(versions)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if models are loaded
        if model is None or scaler is None or threshold is None:
            return jsonify({"error": "ML models not loaded properly"}), 500
        
        # Log memory usage
        logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
        
        # Check for file in request
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file found"}), 400
            
        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        # Save file
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        
        try:
            # Load audio
            logger.info("Loading audio")
            y, sr = load_audio(path)
            
            if y is None or len(y) == 0:
                return jsonify({"error": "Failed to load audio - empty signal"}), 400
                
            # Extract features
            logger.info("Extracting features")
            features = extract_features(y, sr=sr)
            
            # Create DataFrame with all expected features initialized to 0
            feature_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
            
            # Update with extracted features
            for feat, value in features.items():
                if feat in feature_names:
                    feature_df[feat] = value
            
            # Remove extra features and ensure correct order
            feature_df = feature_df[feature_names]
            
            # Scale features
            logger.info("Scaling features")
            X_scaled = scaler.transform(feature_df)
            
            # Make prediction
            logger.info("Making prediction")
            with tf.device('/CPU:0'):
                reconstruction = model.predict(X_scaled, verbose=0, batch_size=1)
                
            # Calculate error and prediction
            error = np.mean(np.square(X_scaled - reconstruction), axis=1)[0]
            result = "FAKE" if error > threshold else "REAL"
            confidence = min(1.0, max(0.0, 1 - abs(error - threshold) / threshold))
            
            # Clean up
            try:
                os.remove(path)
            except Exception as clean_error:
                logger.warning(f"Could not delete file {path}: {clean_error}")
                
            # Prepare response with features for Flutter frontend
            response = {
                "prediction": result,
                "confidence": round(float(confidence), 4),
                "error_value": float(error),
                "threshold": float(threshold),
                "features_count": len(feature_names),
                "sample_rate_original": sr,
                "sample_rate_used": DEFAULT_SR,
                "features": {key: float(value) for key, value in features.items()}
            }
            
            logger.info("Prediction completed successfully")
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500
            
        finally:
            gc.collect()
            
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Request failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
