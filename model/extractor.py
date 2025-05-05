import os
import numpy as np
import librosa
import warnings
import logging
import shutil

# Disable Numba JIT and clear cache
os.environ['NUMBA_DISABLE_JIT'] = '1'
numba_cache_dir = os.path.expanduser("~/.cache/numba")
if os.path.exists(numba_cache_dir):
    shutil.rmtree(numba_cache_dir)
    logger.info("Cleared numba cache")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Audio parameters
DEFAULT_SR = 16000
DURATION = 2
N_FFT = 512
N_MELS = 16
HOP_LENGTH = 512

def load_audio(file_path, duration=DURATION, offset=0.0):
    """Load audio file with memory optimization."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(file_path, sr=DEFAULT_SR, mono=True,
                               duration=duration, offset=offset)
            y = librosa.util.fix_length(y, size=int(DEFAULT_SR * duration))
        logger.info("Audio loaded successfully")
        return y, sr
    except Exception as e:
        logger.error(f"Error loading audio {file_path}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return np.zeros(int(DEFAULT_SR * DURATION)), DEFAULT_SR

def extract_features(y, sr=DEFAULT_SR):
    """Extract audio features aligned with model.py."""
    features = {}
    
    try:
        if sr != DEFAULT_SR:
            logger.info(f"Resampling from {sr} to {DEFAULT_SR}")
            y = librosa.resample(y, orig_sr=sr, target_sr=DEFAULT_SR)
            sr = DEFAULT_SR
        
        y = librosa.util.fix_length(y, size=int(sr * DURATION))
        
        if np.isnan(y).any() or np.isinf(y).any():
            logger.warning("Invalid values detected, using zeros")
            y = np.zeros(int(sr * DURATION))
        
        logger.info("Computing STFT")
        S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        
        logger.info("Computing RMS and ZCR")
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
        features.update({
            'volume_mean': np.mean(rms),
            'volume_std': np.std(rms),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
        })
        
        logger.info("Computing spectral features")
        spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
        features.update({
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_std': np.std(spectral_centroid),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_std': np.std(spectral_bandwidth),
        })
        
        logger.info("Computing mel spectrogram")
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        log_mel = librosa.power_to_db(mel_spec)
        mel_mean = np.mean(log_mel, axis=1)
        mel_std = np.std(log_mel, axis=1)
        features.update({
            f'mel_mean_{i}': mel_mean[i] if i < len(mel_mean) else 0.0 for i in range(40)
        })
        features.update({
            f'mel_std_{i}': mel_std[i] if i < len(mel_std) else 0.0 for i in range(40)
        })
        
        logger.info("Computing pitch")
        pitches, _ = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=2000, n_fft=N_FFT, hop_length=HOP_LENGTH)
        valid_pitches = pitches[pitches > 0]
        features.update({
            'pitch_mean': np.nanmean(valid_pitches) if valid_pitches.size > 0 else 0,
            'pitch_std': np.nanstd(valid_pitches) if valid_pitches.size > 0 else 0,
        })
        
        logger.info("Computing MFCC")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_delta = librosa.feature.delta(mfccs)
        features.update({
            f'mfcc_{i}': np.mean(val) for i, val in enumerate(mfccs)
        })
        features.update({
            f'mfcc_std_{i}': np.std(val) for i, val in enumerate(mfccs)
        })
        features.update({
            f'mfcc_delta_{i}': np.mean(val) for i, val in enumerate(mfcc_delta)
        })
        features.update({
            f'mfcc_delta2_{i}': 0.0 for i in range(20)
        })
        
        # Skip chroma and onset_strength to reduce computation
        features.update({
            f'chroma_{i}': 0.0 for i in range(12)
        })
        features.update({
            'onset_strength_mean': 0.0,
            'onset_strength_std': 0.0,
        })
        
        logger.info("Computing energy dynamics")
        energy_mean = np.mean(S, axis=0)
        energy_diff = np.diff(energy_mean)
        features.update({
            'energy_diff_std': np.std(energy_diff),
        })
        
        logger.info("Computing pause ratio")
        energy = rms[0]
        features.update({
            'pause_ratio': np.sum(energy < 0.02) / len(energy),
        })
        
        logger.info("Cleaning up memory")
        del S, rms, zcr, spectral_centroid, spectral_bandwidth, mel_spec, log_mel
        del mel_mean, mel_std, pitches, valid_pitches, mfccs, mfcc_delta
        del energy_mean, energy_diff
        gc.collect()
        
        logger.info(f"Extracted {len(features)} features")
        return features
    
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return features
