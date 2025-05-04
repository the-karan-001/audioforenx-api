import os
import numpy as np
import librosa
import warnings

# Disable Numba JIT to avoid compilation issues
os.environ['NUMBA_DISABLE_JIT'] = '1'

# Use this as default but allow overriding when calling functions
DEFAULT_SR = 22050  # Your original sample rate
DURATION = 3

def load_audio(file_path, duration=DURATION, offset=0.0):
    """
    Load audio file with memory optimization.
    
    Args:
        file_path: Path to audio file
        duration: Duration in seconds to load
        offset: Start reading after this time (in seconds)
    
    Returns:
        Audio signal as numpy array and original sample rate
    """
    try:
        # Memory-efficient loading with duration limit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(file_path, sr=None, mono=True,
                               duration=duration, offset=offset)
            
        # Return both the audio and original sample rate
        return y, sr
        
    except Exception as e:
        print(f"Error loading audio {file_path}: {str(e)}")
        # Return empty audio in case of error
        return np.zeros(int(DEFAULT_SR * DURATION)), DEFAULT_SR

def extract_features(y, sr=DEFAULT_SR):
    """
    Extract audio features safely with memory efficiency.
    
    Args:
        y: Audio time series
        sr: Sample rate of the audio (to handle different sample rates)
    """
    features = {}
    
    try:
        # Resample if necessary to match the expected sample rate for feature extraction
        if sr != DEFAULT_SR:
            print(f"Resampling from {sr} to {DEFAULT_SR}")
            y = librosa.resample(y, orig_sr=sr, target_sr=DEFAULT_SR)
            sr = DEFAULT_SR
        
        # Fix length to expected duration
        y = librosa.util.fix_length(y, size=int(sr * DURATION))
        
        # Ensure the audio is within reasonable bounds
        if np.isnan(y).any() or np.isinf(y).any():
            print(f"Warning: Invalid values detected, using zeros")
            y = np.zeros(int(sr * DURATION))
        
        # Trim silence
        try:
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            # Make sure it's still the right length after trimming
            y = librosa.util.fix_length(y_trimmed, size=int(sr * DURATION))
        except Exception as e:
            print(f"Warning: Trimming failed: {str(e)}")
        
        # Basic features
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # Spectral features - using safe computation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S = np.abs(librosa.stft(y))
            
        # Only compute spectral features if we have a valid spectrogram
        if S.size > 0:
            spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
            
            # Add chroma features - these were missing in your error
            chroma = librosa.feature.chroma_stft(S=S, sr=sr)
        else:
            spectral_centroid = np.zeros((1, 1))
            spectral_bandwidth = np.zeros((1, 1))
            chroma = np.zeros((12, 1))  # Typically 12 chroma bins
        
        # Pitch detection - with safety checks
        try:
            pitches, _ = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=2000)
            valid_pitches = pitches[pitches > 0]
        except Exception as e:
            print(f"Warning: Pitch extraction failed: {str(e)}")
            valid_pitches = np.array([])
        
        # MFCCs - with safety checks
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Increase to 20 as your error showed missing mfcc features
            mfcc_mean = np.mean(mfccs, axis=1)
        except Exception as e:
            print(f"Warning: MFCC extraction failed: {str(e)}")
            mfcc_mean = np.zeros(20)
        
        # Helper function for safe mean calculation
        def safe_mean(x): 
            return np.nanmean(x) if x.size > 0 else 0
            
        # Update features dictionary with your original features
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
            
        # Add chroma features - these were missing in your error
        for i, chroma_band in enumerate(chroma):
            features[f"chroma_{i}"] = float(np.mean(chroma_band))
            
        # Add spectral contrast
        try:
            contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
            for i, contrast_band in enumerate(contrast):
                features[f"contrast_{i}"] = float(np.mean(contrast_band))
        except Exception as e:
            print(f"Warning: Spectral contrast extraction failed: {str(e)}")
            # Add default values
            for i in range(7):  # Typical number of contrast bands
                features[f"contrast_{i}"] = 0.0
                
        # Add tonnetz features with robust handling
        try:
            # Skip harmonic separation which might be causing issues
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            for i, tonnetz_dim in enumerate(tonnetz):
                features[f"tonnetz_{i}"] = float(np.mean(tonnetz_dim))
        except Exception as e:
            print(f"Warning: Tonnetz calculation failed: {str(e)}")
            # Add default values
            for i in range(6):  # Tonnetz typically has 6 dimensions
                features[f"tonnetz_{i}"] = 0.0
            
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
                
        for i in range(20):  # Ensure all MFCCs are present
            if f"mfcc_{i}" not in features:
                features[f"mfcc_{i}"] = 0.0
                
        for i in range(12):  # Ensure all chroma features are present
            if f"chroma_{i}" not in features:
                features[f"chroma_{i}"] = 0.0
                
        for i in range(7):  # Ensure all contrast features are present
            if f"contrast_{i}" not in features:
                features[f"contrast_{i}"] = 0.0
                
        for i in range(6):  # Ensure all tonnetz features are present
            if f"tonnetz_{i}" not in features:
                features[f"tonnetz_{i}"] = 0.0
    
    return features
