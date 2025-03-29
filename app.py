from flask import Flask, request, jsonify
import joblib
import numpy as np
import librosa
import os
from flask_cors import CORS
from pydub import AudioSegment  # MP3 support

app = Flask(__name__)
CORS(app)

# Load trained model and scaler
model = joblib.load("models/music_genre_classifier.pkl")
scaler = joblib.load("models/scaler.pkl")

# Genre mapping
genre_mapping = {
    0: "Blues", 1: "Classical", 2: "Country", 3: "Disco", 4: "HipHop",
    5: "Jazz", 6: "Metal", 7: "Pop", 8: "Reggae", 9: "Rock"
}

def extract_features(file_path):
    """Extracts audio features from a given file."""
    try:
        print(f"🔍 Extracting features from: {file_path}")

        y, sr = librosa.load(file_path, sr=None)
        print(f"✅ Loaded audio file, Sample Rate: {sr}, Length: {len(y)}")

        # Extract features
        features = []
        
        # 1. Chroma STFT Features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features.append(np.mean(chroma_stft))
        features.append(np.var(chroma_stft))
        
        # 2. RMS Features
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.var(rms))

        # 3. Spectral Centroid Features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(spectral_centroid))
        features.append(np.var(spectral_centroid))

        # 4. Spectral Bandwidth Features
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.append(np.mean(spectral_bandwidth))
        features.append(np.var(spectral_bandwidth))

        # 5. Spectral Rolloff Features
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(np.mean(spectral_rolloff))
        features.append(np.var(spectral_rolloff))

        # 6. Zero Crossing Rate Features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        features.append(np.mean(zero_crossing_rate))
        features.append(np.var(zero_crossing_rate))

        # 7. Harmonic Features (Harmony)
        harmonic = librosa.effects.harmonic(y)
        features.append(np.mean(harmonic))
        features.append(np.var(harmonic))

        # 8. Perceptual Features (could be derived from Tonnetz)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features.append(np.mean(tonnetz))
        features.append(np.var(tonnetz))

        # 9. Tempo (Beat)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        features.append(tempo)

        # 10. MFCC Features (1 to 20)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features.append(np.mean(mfccs[i]))
            features.append(np.var(mfccs[i]))

        print(f"✅ Extracted {len(features)} features")
        return features

    except Exception as e:
        print("❌ Feature extraction error:", e)
        return None

@app.route('/extract-features', methods=['POST'])
def extract_audio_features():
    """Handles file upload, converts MP3 to WAV if needed, and extracts features."""
    if 'file' not in request.files:
        print("❌ No file uploaded")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = file.filename
    temp_path = "temp_audio.wav"

    print(f"📂 Received file: {filename}")

    # Convert MP3 to WAV if necessary
    if filename.endswith('.mp3'):
        temp_mp3 = "temp_audio.mp3"
        file.save(temp_mp3)
        print("✅ Saved MP3:", temp_mp3)

        try:
            audio = AudioSegment.from_mp3(temp_mp3)
            print("✅ MP3 loaded successfully")
            
            audio.export(temp_path, format="wav")
            print("✅ Converted to WAV:", temp_path)
            
            os.remove(temp_mp3)
        except Exception as e:
            print("❌ MP3 conversion failed:", e)
            return jsonify({"error": f"MP3 conversion failed: {e}"}), 500
    else:
        file.save(temp_path)
        print("✅ Saved WAV file:", temp_path)

    # Extract features
    features = extract_features(temp_path)
    os.remove(temp_path)
    
    if features is None:
        print("❌ Feature extraction failed")
        return jsonify({"error": "Feature extraction failed"}), 500

    # Convert all features to native Python types
    features = [float(f) for f in features]  # Convert numpy.float32 to float

    print(f"✅ Extracted {len(features)} features")  # Debugging Line

    if len(features) != 57:
        print("❌ Feature count mismatch! Expected 57, got", len(features))
        return jsonify({"error": f"Feature extraction mismatch: Expected 57, got {len(features)}"}), 500

    print("✅ Features extracted successfully!")
    return jsonify({"features": features})


@app.route('/predict', methods=['POST'])
def predict_genre():
    """Predict the genre of the uploaded audio file."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = file.filename
    temp_path = "temp_audio.wav"

    print(f"📂 Received file: {filename}")

    # Convert MP3 to WAV if necessary
    if filename.endswith('.mp3'):
        temp_mp3 = "temp_audio.mp3"
        file.save(temp_mp3)
        print("✅ Saved MP3:", temp_mp3)

        try:
            audio = AudioSegment.from_mp3(temp_mp3)
            print("✅ MP3 loaded successfully")
            
            audio.export(temp_path, format="wav")
            print("✅ Converted to WAV:", temp_path)
            
            os.remove(temp_mp3)
        except Exception as e:
            print("❌ MP3 conversion failed:", e)
            return jsonify({"error": f"MP3 conversion failed: {e}"}), 500
    else:
        file.save(temp_path)
        print("✅ Saved WAV file:", temp_path)

    # Extract features
    features = extract_features(temp_path)
    os.remove(temp_path)
    
    if features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    # Convert all features to native Python types
    features = [float(f) for f in features]  # Convert numpy.float32 to float

    # Scale the features and make predictions
    features = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    predicted_genre = genre_mapping.get(prediction[0], "Unknown")

    print(f"✅ Predicted Genre: {predicted_genre}")
    return jsonify({"predicted_genre": predicted_genre})

if __name__ == '__main__':
    app.run(debug=True)
