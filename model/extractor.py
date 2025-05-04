import os
import numpy as np
import librosa
import warnings
import logging

# Disable Numba JIT
os.environ['NUMBA_DISABLE_JIT'] = '1'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Audio parameters
DEFAULT_SR = 16000  # Reduced from 22050 for memory efficiency
DURATION = 3

def load_audio(file_path, duration=DURATION, offset=0.0):
    """Load audio file with memory optimization."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(file_path, sr=DEFAULT_SR, mono=True,
                               duration=duration, offset=offset)
        return y, sr
    except Exception as e:
        logger.error(f"Error loading audio {file_path}: {str(e)}")
        return np.zeros(int(DEFAULT_SR * DURATION)), DEFAULT_SR

def extract_features(y, sr=DEFAULT_SR):
    """Extract audio features aligned with model.py."""
    features = {}
    
    try:
        # Resample if necessary
        if sr != DEFAULT_SR:
            logger.info(f"Resampling from {sr} to {DEFAULT_SR}")
            y = librosa.resample(y, orig_sr=sr, target_sr=DEFAULT_SR)
            sr = DEFAULT_SR
        
        # Fix length
        y = librosa.util.fix_length(y, size=int(sr * DURATION))
        
        # Check for invalid values
        if np.isnan(y).any() or np.isinf(y).any():
            logger.warning("Invalid values detected, using zeros")
            y = np.zeros(int(sr * DURATION))
 ಸ

System: Thank you for sharing the complete code for `app.py`, `extractor.py`, `model.py`, and `result_screen.dart`. Below, I will analyze the errors, provide detailed solutions to resolve the issues, and update the `app.py` and `extractor.py` files to fix the **DataFrame fragmentation** and **worker timeout/SIGKILL** issues. The `model.py` file is used for training and doesn’t directly contribute to the deployment errors, but I’ll reference it to ensure feature alignment. I’ll also ensure the API output is compatible with `result_screen.dart` to display results correctly.

---

### Error Analysis

#### 1. **PerformanceWarning: DataFrame is highly fragmented**
- **Error Location**: `app.py`, line 145 in the `/predict` route:
  ```
  /opt/render/project/src/app.py:145: PerformanceWarning: DataFrame is highly fragmented. This is usually the result of calling `frame.insert` many times, which has poor performance. Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    feature_df[feat] = 0.0  # Add missing features with default values
  ```
- **Cause**:
  - The loop in `app.py` adds missing features to `feature_df` one by one (`feature_df[feat] = 0.0`), causing DataFrame fragmentation and performance degradation.
  - The model expects 186 features (as logged: `Model expects 186 features`), but `extract_features` in `extractor.py` may not produce all required features, triggering dynamic column additions.
- **Impact**:
  - Slows down feature preparation, contributing to worker timeouts.
  - Increases memory usage due to inefficient DataFrame operations.

#### 2. **Worker Timeout and SIGKILL**
- **Error Logs**:
  ```
  [2025-05-04 12:.Disconnecting...03:35 +0000] [82] [CRITICAL] WORKER TIMEOUT (pid:146)
  [2025-05-04 12:03:36 +0000] [82] [ERROR] Worker (pid:146) was sent SIGKILL! Perhaps out of memory?
  ```
- **Cause**:
  - **Memory Exhaustion**: Audio processing (`librosa` in `extractor.py`), feature extraction, or model inference (`tensorflow` in `app.py`) consumes excessive memory, likely exceeding Render’s free tier limit (512 MB).
  - **Slow Processing**: Feature extraction (e.g., STFT, MFCCs, chroma) and DataFrame operations are computationally intensive, exceeding Render’s default timeout (30 seconds).
  - **Resource Limits**: Render’s free tier has limited CPU and memory, insufficient for large audio files or complex models.
- **Impact**:
  - The worker is killed, causing API failures and errors in the Flutter frontend.

#### 3. **Feature Alignment Issues**
- **Problem**:
  - `result_screen.dart` expects features like `pitch_mean`, `volume_mean`, `zcr_mean`, `spectral_centroid_mean`, and `mfcc_*`, but `app.py` doesn’t include a `features` dictionary in the response, causing errors (e.g., `features['pitch_mean'] ?? 0` returns 0).
  - Features extracted by `extractor.py` don’t fully match the 186 features defined in `model.py`, leading to missing features and dynamic column additions.
- **Impact**:
  - Incorrect or missing features affect prediction accuracy and frontend display.

---

### Solutions

#### 1. **Fix DataFrame Fragmentation**
- **Approach**:
  - Initialize `feature_df` with all 186 expected features (from `scaler.feature_names_in_`) set to 0.0.
  - Update `feature_df` with extracted features, avoiding dynamic column insertion.
  - Ensure correct column order for model compatibility.
- **Implementation**:
  - Modify the `/predict` route in `app.py` to create `feature_df` with predefined columns.
  - Update `extractor.py` to produce all required features, minimizing missing features.

#### 2. **Optimize Memory and Processing**
- **Approach**:
  - **Downsample Audio**: Use 16 kHz (instead of 22.05 kHz) to reduce memory usage, as it’s sufficient for speech features.
  - **Optimize Feature Extraction**: Reduce FFT window size, cache spectrograms, and skip redundant computations in `extractor.py`.
  - **Memory Management**: Free memory after each step using `gc.collect()` and delete temporary arrays.
  - **Model Inference**: Use CPU and limit TensorFlow threads to reduce memory overhead.
  - **Gunicorn Configuration**: Use a single worker and increase timeout in Render.
- **Implementation**:
  - Update `extractor.py` for efficient audio processing and feature extraction.
  - Enhance memory cleanup in `app.py`.
  - Provide a Gunicorn configuration file.

#### 3. **Ensure Feature Consistency**
- **Approach**:
  - Align `extractor.py` with `model.py` to extract all 186 features, including `mel_mean`, `mfcc_delta`, `pause_ratio`, etc.
  - Include the `features` dictionary in `app.py`’s response for `result_screen.dart`.
- **Implementation**:
  - Update `extractor.py` to match `model.py`’s feature set.
  - Modify `app.py` to return features in the JSON response.

#### 4. **Frontend Compatibility**
- **Approach**:
  - Ensure the API response includes a `features` dictionary with keys expected by `result_screen.dart`.
  - Verify that feature values are floats and within valid ranges for chart rendering.
- **Implementation**:
  - Add `features` to the `/predict` response in `app.py`.

---

### Updated Code

Below are the updated `app.py`, `extractor.py`, and a new `gunicorn.conf.py` for Render. The `model.py` and `result_screen.dart` files remain unchanged, but I’ll ensure `app.py`’s output is compatible with `result_screen.dart`.

#### 1. **Updated `app.py`**

<xaiArtifact artifact_id="1e2a7bb4-81e5-4fb9-a535-4b6b46c42151" artifact_version_id="cfc8bbd7-d370-4899-9db1-ec128c195dcf" title="app.py" contentType="text/python">
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
import logging
from model.extractor import load_audio, extract_features, DEFAULT_SR

# Set environment variable to disable Numba JIT
os.environ['NUMBA_DISABLE_JIT'] = '1'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure TensorFlow memory limits
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        logger.error(f"GPU memory config error: {e}")

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

UPLOAD_FOLDER = "Uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
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
    gc.collect()
    return response

@app.route("/diagnostics", methods=["GET"])
def diagnostics():
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
        if model is None or scaler is None or threshold is None:
            return jsonify({"error": "ML models not loaded properly"}), 500
        
        logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
        
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file found"}), 400
            
        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        
        try:
            logger.info("Loading audio")
            y, sr = load_audio(path)
            
            if y is None or len(y) == 0:
                return jsonify({"error": "Failed to load audio - empty signal"}), 400
                
            logger.info("Extracting features")
            features = extract_features(y, sr=sr)
            
            # Initialize DataFrame with all expected features
            feature_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
            
            # Update with extracted features
            for feat, value in features.items():
                if feat in feature_names:
                    feature_df[feat] = value
            
            feature_df = feature_df[feature_names]
            
            logger.info("Scaling features")
            X_scaled = scaler.transform(feature_df)
            
            logger.info("Making prediction")
            with tf.device('/CPU:0'):
                reconstruction = model.predict(X_scaled, verbose=0, batch_size=1)
                
            error = np.mean(np.square(X_scaled - reconstruction), axis=1)[0]
            result = "FAKE" if error > threshold else "REAL"
            confidence = min(1.0, max(0.0, 1 - abs(error - threshold) / threshold))
            
            try:
                os.remove(path)
            except Exception as clean_error:
                logger.warning(f"Could not delete file {path}: {clean_error}")
                
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
