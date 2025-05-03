from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from model.extractor import load_audio, extract_features

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('model/deepfake_detection_model.h5',
                   custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
scaler = joblib.load("model/scaler.pkl")
threshold = float(np.load("model/threshold.npy"))

@app.route("/predict", methods=["POST"])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file found"}), 400

    file = request.files['audio']
    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)

    y = load_audio(path)
    features = extract_features(y)
    df = pd.DataFrame([features])
    X_scaled = scaler.transform(df)

    reconstruction = model.predict(X_scaled)
    error = np.mean(np.square(X_scaled - reconstruction), axis=1)[0]
    result = "FAKE" if error > threshold else "REAL"
    confidence = 1 - abs(error - threshold) / threshold

    return jsonify({
        "prediction": result,
        "confidence": round(confidence, 4),
        "features": {k: round(v, 3) for k, v in features.items()}
    })
