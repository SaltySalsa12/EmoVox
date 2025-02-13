import librosa
import noisereduce as nr
import numpy as np

def preprocess_audio(file_path, sr=16000):
    y, sr = librosa.load(file_path, sr=sr)
    
    # Enhanced noise reduction for emotional speech
    y = nr.reduce_noise(
        y=y, sr=sr,
        stationary=True,
        n_std_thresh_stationary=1.5,
        prop_decrease=0.8
    )
    
    # Dynamic silence trimming
    y_trimmed, _ = librosa.effects.trim(
        y, top_db=25,
        frame_length=1024,
        hop_length=256
    )
    
    return librosa.effects.preemphasis(y_trimmed), sr