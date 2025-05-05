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
DEFAULT_SR = 16000  # Reduced for memory efficiency
DURATION = 3
N_FFT = 1024  # Reduced for faster computation
N_MELS = 40   # Matches model.py

def load_audio(file_path, duration=DURATION, offset=0.0):
    """Load audio file with memory optimization."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(file_path, sr=DEFAULT_SR, mono=True,
                               duration=duration, offset=offset)
            y, _ = librosa.effects.trim(y, top_db=20)
            y = librosa.util.fix_length(y, size=int(DEFAULT_SR * duration))
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
        
        # Compute STFT once and reuse
        S = np.abs(librosa.stft(y, n_fft=N_FFT))
        
        # Temporal features
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT)
        log_mel = librosa.power_to_db(mel_spec)
        mel_mean = np.mean(log_mel, axis=1)
        mel_std = np.std(log_mel, axis=1)
        
        # Pitch analysis
        pitches, _ = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=2000, n_fft=N_FFT)
        valid_pitches = pitches[pitches > 0]
        
        # MFCC and deltas
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=N_FFT)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(S=S, sr=sr, n_fft=N_FFT)
        
        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_strength = np.mean(onset_env)
        onset_std = np.std(onset_env)
        
        # Energy dynamics
        energy_mean = np.mean(S, axis=0)
        energy_diff = np.diff(energy_mean)
        energy_diff_std = np.std(energy_diff)
        
        # Pause ratio
        energy = rms[0]
        pause_ratio = np.sum(energy < 0.02) / len(energy)
        
        # Compile features
        features.update({
            # Core features
            'pitch_mean': np.nanmean(valid_pitches) if valid_pitches.size > 0 else 0,
            'pitch_std': np.nanstd(valid_pitches) if valid_pitches.size > 0 else 0,
            'volume_mean': np.mean(rms),
            'volume_std': np.std(rms),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_std': np.std(spectral_centroid),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_std': np.std(spectral_bandwidth),
            'onset_strength_mean': onset_strength,
            'onset_strength_std': onset_std,
            'energy_diff_std': energy_diff_std,
            'pause_ratio': pause_ratio,
            
            # Mel spectrogram
            **{f'mel_mean_{i}': val for i, val in enumerate(mel_mean)},
            **{f'mel_std_{i}': val for i, val in enumerate(mel_std)},
            
            # Chroma
            **{f'chroma_{i}': np.mean(val) for i, val in enumerate(chroma)},
            
            # MFCC features and derivatives
            **{f'mfcc_{i}': np.mean(val) for i, val in enumerate(mfccs)},
            **{f'mfcc_std_{i}': np.std(val) for i, val in enumerate(mfccs)},
            **{f'mfcc_delta_{i}': np.mean(val) for i, val in enumerate(mfcc_delta)},
            **{f'mfcc_delta2_{i}': np.mean(val) for i, val in enumerate(mfcc_delta2)},
        })
        
        # Clean up
        del S, rms, zcr, spectral_centroid, spectral_bandwidth, mel_spec, log_mel
        del mel_mean, mel_std, pitches, valid_pitches, mfccs, mfcc_delta, mfcc_delta2
        del chroma, onset_env, energy_mean, energy_diff
        gc.collect()
        
        return features
    
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return {}
