import librosa
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def extract_features(y, sr):
    features = []
    
    # 1. Pitch with fallback for silent audio
    try:
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    except:
        pitch_mean = 0
    features.append(pitch_mean)

    # 2. MFCC with empty audio handling
    if len(y) == 0:  # If audio became empty after trimming
        y = np.zeros(sr * 1)  # Pad 1 second of silence
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))  # 40 features

    # 3. Spectral features with NaN protection
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    features.append(np.nan_to_num(np.mean(spectral_flatness)))

    return np.array(features).reshape(1, -1)  # Force 2D output

class FeatureSelector:
    def __init__(self, k=40):  # Match actual feature count (1+40+1=42)
        self.selector = SelectKBest(mutual_info_classif, k=min(k, 42))
        
    def fit(self, X, y):
        X = np.atleast_2d(X)
        self.selector.fit(X, y)
        
    def transform(self, X):
        return self.selector.transform(np.atleast_2d(X))