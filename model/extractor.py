import numpy as np
import librosa

TARGET_SR = 22050
DURATION = 3

def load_audio(file_path):
    y, _ = librosa.load(file_path, sr=TARGET_SR)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return librosa.util.fix_length(y_trimmed, size=TARGET_SR * DURATION)

def extract_features(y):
    features = {}

    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    S = np.abs(librosa.stft(y))
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=TARGET_SR)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=TARGET_SR)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=40)
    log_mel = librosa.power_to_db(mel_spec)
    mel_mean = np.mean(log_mel, axis=1)

    pitches, _ = librosa.piptrack(y=y, sr=TARGET_SR, fmin=50, fmax=2000)
    valid_pitches = pitches[pitches > 0]

    mfccs = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=20)
    mfcc_mean = np.mean(mfccs, axis=1)

    def safe_mean(x): return np.nanmean(x) if x.size > 0 else 0

    features.update({
        "pitch_mean": safe_mean(valid_pitches),
        "zcr_mean": np.mean(zcr),
        "volume_mean": np.mean(rms),
        "spectral_centroid_mean": np.mean(spectral_centroid),
        "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
    })

    for i, val in enumerate(mfcc_mean):
        features[f"mfcc_{i}"] = val

    for i, val in enumerate(mel_mean):
        features[f"mel_{i}"] = val

    return features
