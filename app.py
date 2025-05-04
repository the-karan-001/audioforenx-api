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
import librosa

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

# Define the target sample rate that was used during training
TARGET_SAMPLE_RATE = 16000  # Adjust this to match your training sample rate

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models once at startup to avoid repeated loading
print("Loading ML models...")
try:
    model = load_model('model/deepfake_detection_model.h5',
                       custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    scaler = joblib.load("model/scaler.pkl")
    threshold = float(np.load("model/threshold.npy"))
    
    # Store the feature names from the scaler for validation
    feature_names = scaler.feature_names_in_
    print(f"Model expects {len(feature_names)} features")
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    traceback.print_exc()
    model = None
    scaler = None
    threshold = None
    feature_names = None

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
            # Load and process audio with resampling to match training sample rate
            y, sr = librosa.load(path, sr=None)  # Load with original sample rate
            print(f"Original sample rate: {sr}")
            
            if y is None or len(y) == 0:
                return jsonify({"error": "Failed to load audio - empty signal"}), 400
                
            # Resample to target sample rate
            if sr != TARGET_SAMPLE_RATE:
                print(f"Resampling from {sr} to {TARGET_SAMPLE_RATE}")
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
            
            # Extract features
            features = extract_features(y, sr=TARGET_SAMPLE_RATE)
            
            # Ensure features match what the scaler expects
            feature_df = pd.DataFrame([features])
            
            # Check and fix feature alignment
            missing_features = set(feature_names) - set(feature_df.columns)
            extra_features = set(feature_df.columns) - set(feature_names)
            
            if missing_features:
                print(f"Adding {len(missing_features)} missing features")
                for feat in missing_features:
                    feature_df[feat] = 0.0  # Add missing features with default values
                    
            if extra_features:
                print(f"Removing {len(extra_features)} extra features")
                feature_df = feature_df.drop(columns=list(extra_features))
                
            # Ensure column order matches what the scaler expects
            feature_df = feature_df[feature_names]
            
            # Scale features
            X_scaled = scaler.transform(feature_df)
            
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
                "features_count": len(feature_names),
                "sample_rate_original": sr,
                "sample_rate_used": TARGET_SAMPLE_RATE
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
