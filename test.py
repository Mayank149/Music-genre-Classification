import librosa
import numpy as np

file_path = "temp_audio.wav"  # Ensure the file exists

try:
    y, sr = librosa.load(file_path, sr=None)
    print(f"✅ Loaded audio file, Sample Rate: {sr}, Length: {len(y)}")

    if len(y) == 0:
        print("❌ Audio file is empty or not decoded properly")
    else:
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        print(f"✅ Spectral Rolloff Shape: {spectral_rolloff.shape}")

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        print(f"✅ MFCCs Extracted: {mfccs.shape}")

except Exception as e:
    print(f"❌ Error: {e}")
