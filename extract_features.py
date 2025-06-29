import os
import librosa
import numpy as np
import pandas as pd

# Define paths
DATASET_PATH = "dataset/"
CATEGORIES = ["Screaming", "NotScreaming"]

# Function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3)  # Load audio file
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC features
    return np.mean(mfcc.T, axis=0)

# Prepare data
features = []
labels = []

for category in CATEGORIES:
    folder_path = os.path.join(DATASET_PATH, category)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            feature = extract_features(file_path)
            features.append(feature)
            labels.append(1 if category == "Screaming" else 0)  # 1 = Scream, 0 = Normal
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Save features to CSV
df = pd.DataFrame(features)
df["label"] = labels
df.to_csv("features.csv", index=False)

print("Feature extraction complete! Data saved as features.csv.")
