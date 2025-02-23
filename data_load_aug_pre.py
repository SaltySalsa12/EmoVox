import os
import glob
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Dataset Path
iemocap_base = 'C:\s_e_d_updated\data\iemocap'
metadata_csv = 'C:\s_e_d_updated\data\iemocap\iemocap_metadata.csv'

# Reading Data from CSV
metadata_df = pd.read_csv(metadata_csv)
allowed_emotions = {'ang', 'hap', 'sad', 'neu'}

# Storage pf Data
data_entries = []

# Function for extraction of features
def extract_features(y, sr):
    # Computation of 40MFCC coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Computation of first and second derivatives (delta and delta-delta)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Computation of statistical summaries: mean and std for each feature type
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    delta_mean = np.mean(delta, axis=1)
    delta_std = np.std(delta, axis=1)
    delta2_mean = np.mean(delta2, axis=1)
    delta2_std = np.std(delta2, axis=1)

    #Concatenation of features to form a 240-dimensional vector (40*6)

    feature_vector = np.concatenate([mfcc_mean, mfcc_std, delta_mean, delta_std, delta2_mean, delta2_std])
    return feature_vector

# Processing and applying data augmnetation onto data

sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

for session in sessions:
    session_path = os.path.join(iemocap_base, session)
    sentences_wav_dir = os.path.join(session_path, 'sentences', 'wav')

    if not os.path.exists(sentences_wav_dir):
        print(f"Directory not found: {sentences_wav_dir}")
        continue

    wav_files = glob.glob(os.path.join(sentences_wav_dir, '**', '*.wav'), recursive=True)
    print(f"Found {len(wav_files)} wav files in {sentences_wav_dir}")

    for wav_file in wav_files:
        rel_path = os.path.relpath(wav_file, iemocap_base)
        basename = os.path.basename(wav_file)
        meta_match = metadata_df[metadata_df['path'].str.contains(basename, case=False, na=False)]
        if meta_match.empty:
            print(f"No metadat found for {wav_file}")
            continue

        emotion = meta_match.iloc[0]['emotion'].strip().lower()
        if emotion not in allowed_emotions:
            continue

        try:
            y, sr = librosa.load(wav_file, sr=16000)
            feat_orig = extract_features(y, sr)

        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            continue

        # Saving the original features
        record = {'file_path': rel_path, 'emotion': emotion}
        for i,coef in enumerate(feat_orig, start=1):
            record[f'feat_{i}'] = coef
        data_entries.append(record)

        #Pitch Shifting
        for n_steps in [1, -1]:
            try:
                y_shift = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
                feat_shift = extract_features(y_shift, sr)
                record_aug = {'file_path': rel_path + f"_pitch{n_steps}", 'emotion': emotion}
                for i,coef in enumerate(feat_shift, start=1):
                    record_aug[f'feat_{i}'] = coef
                data_entries.append(record_aug)
            
            except Exception as e:
                print(f"Error in pitch shift augmentation for {wav_file}: {e}")

        #Time Stretching
        for rate in [0.9, 1.1]:
            try:
                y_stretch = librosa.effects.time_stretch(y, rate=rate)
                feat_stretch = extract_features(y_stretch, sr)
                record_aug = {'file_path': rel_path + f"_stretch{rate}", 'emotion': emotion}
                for i, coef in enumerate(feat_stretch, start=1):
                    record_aug[f'feat_{i}'] = coef
                data_entries.append(record_aug)

            except Exception as e:
                print(f"Error in time stretch augmentation for {wav_file}: {e}")

# Creation of Dataframes and Standardizing Features

df = pd.DataFrame(data_entries)
feature_columns = [f'feat_{i}' for i in range(1, 241)]
scaler = StandardScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])

# Saving data to hdf5 format

output_hdf5 = 'iemocap_features_augmented.h5'
df.to_hdf(output_hdf5, key='df', mode='w')
print(f"Feature extraction and augmentation completed. Data saved to HDF5 file '{output_hdf5}'")