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
from model.extractor import load_audio, extract_features

# Set environment variable to disable Numba JIT
os.environ['NUMBA_DISABLE_JIT'] = '1'

# Configure memory limits for TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(f"GPU memory config error: {e}")

# Limit TensorFlow to lower memory usage
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models once at startup to avoid repeated loading
print("Loading ML models...")
try:
    model = load_model('model/deepfake_detection_model.h5',
                       custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    scaler = joblib.load("model/scaler.pkl")
    threshold = float(np.load("model/threshold.npy"))
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    traceback.print_exc()
    model = None
    scaler = None
    threshold = None

@app.after_request
def cleanup_after_request(response):
    """Force garbage collection after each request to free memory"""
    gc.collect()
    return response

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if models are loaded
        if model is None or scaler is None or threshold is None:
            return jsonify({"error": "ML models not loaded properly"}), 500
        
        # Check for file in request
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file found"}), 400
            
        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        # Create a secure and temporary path
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        
        try:
            # Load and process audio
            y = load_audio(path)
            
            if y is None or len(y) == 0:
                return jsonify({"error": "Failed to load audio - empty signal"}), 400
                
            # Extract features
            features = extract_features(y)
            
            # Debugging: Check the number of extracted features
            print(f"Number of extracted features: {len(features)}")
            
            # Create DataFrame
            df = pd.DataFrame([features])
            
            # Scale features
            X_scaled = scaler.transform(df)
            
            # Make prediction with memory efficiency
            with tf.device('/CPU:0'):  # Force CPU to avoid GPU memory issues
                reconstruction = model.predict(X_scaled, verbose=0)
                
            # Calculate error and prediction
            error = np.mean(np.square(X_scaled - reconstruction), axis=1)[0]
            result = "FAKE" if error > threshold else "REAL"
            confidence = 1 - abs(error - threshold) / threshold
            
            # Clean up the file after processing
            try:
                os.remove(path)
            except Exception as clean_error:
                print(f"Warning: Could not delete file {path}: {clean_error}")
                
            # Return results
            return jsonify({
                "prediction": result,
                "confidence": round(float(confidence), 4),
                "error_value": float(error),
                "threshold": float(threshold),
                "features": {k: round(float(v), 3) for k, v in features.items()}
            })
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500
            
    except Exception as e:
        print(f"Request error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Request failed: {str(e)}"}), 500
    finally:
        # Always try to clean up
        try:
            if 'path' in locals() and os.path.exists(path):
                os.remove(path)
        except:
            pass

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
