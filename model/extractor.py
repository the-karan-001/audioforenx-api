import os
import numpy as np
import librosa
import warnings

# Disable Numba JIT to avoid compilation issues
os.environ['NUMBA_DISABLE_JIT'] = '1'

TARGET_SR = 22050  # Reduced from 22050 to save memory
DURATION = 3

def load_audio(file_path, duration=DURATION, offset=0.0):
    """
    Load audio file with memory optimization.
    
    Args:
        file_path: Path to audio file
        duration: Duration in seconds to load
        offset: Start reading after this time (in seconds)
    
    Returns:
        Audio signal as numpy array
    """
    try:
        # Memory-efficient loading with duration limit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(file_path, sr=TARGET_SR, mono=True, 
                              duration=duration, offset=offset)
            
        # Trim silence
        try:
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        except Exception as e:
            print(f"Warning: Trimming failed: {str(e)}")
            y_trimmed = y
            
        # Fix length to expected duration
        result = librosa.util.fix_length(y_trimmed, size=int(TARGET_SR * DURATION))
        
        # Ensure the audio is within reasonable bounds
        if np.isnan(result).any() or np.isinf(result).any():
            print(f"Warning: Invalid values detected in {file_path}, using zeros")
            result = np.zeros(int(TARGET_SR * DURATION))
            
        return result
        
    except Exception as e:
        print(f"Error loading audio {file_path}: {str(e)}")
        # Return empty audio in case of error
        return np.zeros(int(TARGET_SR * DURATION))

def extract_features(y):
    """
    Extract audio features safely with memory efficiency.
    """
    features = {}
    
    try:
        # Basic features
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # Spectral features - using safe computation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S = np.abs(librosa.stft(y))
            
        # Only compute spectral features if we have a valid spectrogram
        if S.size > 0:
            spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=TARGET_SR)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=TARGET_SR)
        else:
            spectral_centroid = np.zeros((1, 1))
            spectral_bandwidth = np.zeros((1, 1))
        
        # Pitch detection - with safety checks
        try:
            pitches, _ = librosa.piptrack(y=y, sr=TARGET_SR, fmin=50, fmax=2000)
            valid_pitches = pitches[pitches > 0]
        except Exception as e:
            print(f"Warning: Pitch extraction failed: {str(e)}")
            valid_pitches = np.array([])
        
        # MFCCs - with safety checks
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=10)
            mfcc_mean = np.mean(mfccs, axis=1)
        except Exception as e:
            print(f"Warning: MFCC extraction failed: {str(e)}")
            mfcc_mean = np.zeros(10)
        
        # Helper function for safe mean calculation
        def safe_mean(x): 
            return np.nanmean(x) if x.size > 0 else 0
            
        # Update features dictionary
        features.update({
            "pitch_mean": float(safe_mean(valid_pitches)),
            "zcr_mean": float(np.mean(zcr)),
            "volume_mean": float(np.mean(rms)),
            "spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        })
        
        # Add MFCCs
        for i, val in enumerate(mfcc_mean):
            features[f"mfcc_{i}"] = float(val)
            
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        # Ensure we have all expected features even in case of error
        required_features = [
            "pitch_mean", "zcr_mean", "volume_mean", 
            "spectral_centroid_mean", "spectral_bandwidth_mean"
        ]
        for feat in required_features:
            if feat not in features:
                features[feat] = 0.0
                
        for i in range(10):
            if f"mfcc_{i}" not in features:
                features[f"mfcc_{i}"] = 0.0
    
    return features
