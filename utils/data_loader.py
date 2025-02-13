import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.preprocess_audio import preprocess_audio
from utils.features import extract_features

# Unified emotion mapping for 4 classes
EMOTION_MAP = {
    'angry': 0,
    'sad': 1,
    'happy': 2,
    'neutral': 3,
    'disgust': 0,  # Map disgust to angry
    'fear': 1,     # Map fear to sad
    'calm': 3,     # Map calm to neutral
    'fru': 0,      # Map frustration to angry
    'exc': 2       # Map excited to happy
}

def parse_emodb_filename(filename):
    """
    Parse Emo-DB filename (e.g., 03a01Fa.wav)
    Using German emotion mappings since EmoDB is a German dataset
    """
    emotion_char = filename[5].upper()  # 6th character is emotion code
    mapping = {
        'W': 'angry',    # Ärger (Wut)
        'L': None,       # Langeweile (boredom) - excluded from 4-class mapping
        'E': 'disgust',  # Ekel (disgust) → maps to angry
        'A': 'fear',     # Angst (fear) → maps to sad
        'F': 'happy',    # Freude (happiness)
        'T': 'sad',      # Trauer (sadness)
        'N': 'neutral'   # Neutral
    }
    return mapping.get(emotion_char, None)

def parse_ravdess_filename(filename):
    """Parse RAVDESS filename (e.g., 03-01-06-01-02-01-12.wav)"""
    parts = filename.split('-')
    if len(parts) != 7 or parts[0] != '03' or parts[1] != '01':
        return None  # Skip non-audio/speech files
    
    emotion_code = int(parts[2])
    emotion_map = {
        1: 'neutral',
        2: 'calm',
        3: 'happy',
        4: 'sad',
        5: 'angry',
        6: 'fear',
        7: 'disgust',
        8: 'surprise'
    }
    return emotion_map.get(emotion_code, None)

def parse_iemocap_csv(csv_path, min_agreement=2):
    """Parse IEMOCAP CSV with quality filtering"""
    df = pd.read_csv(csv_path)
    
    # Clean and filter data
    df = df[df['emotion'] != 'xxx']  # Remove invalid labels
    df = df[df['n_annotators'] > 0]  # Remove unannotated samples
    df = df[df['agreement'] >= min_agreement]  # Quality threshold
    
    # Emotion mapping for 4 classes
    emotion_map = {
        'neu': 'neutral',
        'ang': 'angry',
        'sad': 'sad',
        'hap': 'happy',
        'fru': 'angry',  # Map frustration to angry
        'exc': 'happy',  # Map excited to happy
        'sur': None      # Exclude surprise
    }
    
    df['mapped_emotion'] = df['emotion'].map(emotion_map)
    df = df.dropna(subset=['mapped_emotion'])  # Remove unmapped emotions
    
    return df[['path', 'mapped_emotion']]

def load_dataset(dataset_path, dataset_name):
    """
    Load and process emotion dataset with unified preprocessing
    
    Args:
        dataset_path: Path to dataset directory
        dataset_name: Name of dataset ('emodb', 'ravdess', or 'iemocap')
        
    Returns:
        features: numpy array of extracted features
        labels: numpy array of emotion labels
    """
    features = []
    labels = []
    
    if dataset_name == 'iemocap':
        csv_path = os.path.join(dataset_path, "iemocap_metadata.csv")
        df = parse_iemocap_csv(csv_path)
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # Build full audio path
                audio_path = os.path.join(dataset_path, row['path'])
                
                # Load and preprocess audio
                y, sr = preprocess_audio(audio_path)
                if len(y) < sr * 0.5:  # Skip files <0.5 seconds after trimming
                    continue
                    
                # Extract features
                feat = extract_features(y, sr)
                if feat.size == 0:  # Skip empty features
                    continue
                
                features.append(feat[0])
                labels.append(EMOTION_MAP[row['mapped_emotion']])
                
            except Exception as e:
                print(f"Error processing {row['path']}: {str(e)}")
                continue
    
    else:  # EmoDB or RAVDESS
        for file in tqdm(os.listdir(dataset_path)):
            try:
                # Parse emotion based on dataset
                if dataset_name == 'emodb':
                    emotion = parse_emodb_filename(file)
                elif dataset_name == 'ravdess':
                    emotion = parse_ravdess_filename(file)
                else:
                    raise ValueError(f"Unknown dataset: {dataset_name}")
                
                if emotion is None:  # Skip files with invalid emotion
                    continue
                
                # Load and preprocess audio
                y, sr = preprocess_audio(os.path.join(dataset_path, file))
                if len(y) < sr * 0.5:  # Skip files <0.5 seconds after trimming
                    continue
                
                # Extract features
                feat = extract_features(y, sr)
                if feat.size == 0:  # Skip empty features
                    continue
                
                features.append(feat[0])
                labels.append(EMOTION_MAP[emotion])
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
    
    return np.atleast_2d(features), np.array(labels)