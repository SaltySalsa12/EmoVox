import sounddevice as sd
import librosa
import noisereduce as nr
import numpy as np
import tensorflow as tf
import joblib
from utils.features import extract_features

# Emotion mapping consistent with training
EMOTION_MAP = ['angry', 'sad', 'happy', 'neutral']

def real_time_prediction(model, selector, svm, sr=16000):
    """Complete real-time prediction loop with error handling"""
    try:
        # Record audio
        duration = 3  # seconds
        print("\nRecording...")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
        sd.wait()
        
        # Convert to mono and flatten
        y = audio.flatten().astype(np.float32)
        
        # Enhanced preprocessing
        if len(y) == 0:
            raise ValueError("Empty audio recording")
            
        # Noise reduction
        y_clean = nr.reduce_noise(y=y, sr=sr, stationary=True)
        
        # Trim silence with conservative threshold
        y_trimmed, _ = librosa.effects.trim(y_clean, top_db=25)
        if len(y_trimmed) < sr * 0.5:  # Minimum 0.5s audio required
            y_trimmed = np.zeros(int(sr * 1))  # Fallback to 1s silence
            
        # Feature extraction
        features = extract_features(y_trimmed, sr)
        
        # Feature selection
        if features.shape != (1, 42):  # Verify expected feature dimensions
            features = features.reshape(1, -1)
        features_selected = selector.transform(features)
        
        # Hybrid prediction
        dl_pred = model.predict(features_selected).argmax()
        feature_extractor = tf.keras.Model(inputs=model.input, 
                                          outputs=model.layers[-2].output)
        deep_features = feature_extractor.predict(features_selected)
        svm_pred = svm.predict(deep_features)[0]
        
        # Return both predictions
        return (EMOTION_MAP[dl_pred], EMOTION_MAP[svm_pred])
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return ("error", "error")

def live_prediction_loop():
    """Continuous prediction loop"""
    # Load models once
    model = tf.keras.models.load_model('models/trained_model.h5')
    selector = joblib.load('models/feature_selector.pkl')
    svm = joblib.load('models/svm_classifier.pkl')
    
    while True:
        dl_emotion, hybrid_emotion = real_time_prediction(model, selector, svm)
        print(f"\nDL Prediction: {dl_emotion}")
        print(f"Hybrid Prediction: {hybrid_emotion}")
        input("Press Enter to record again...")

if __name__ == '__main__':
    live_prediction_loop()