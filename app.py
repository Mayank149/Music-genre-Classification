from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import joblib
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and scaler
model = joblib.load("model.pkl")  # Ensure your trained model file is here
scaler = joblib.load("scaler.pkl")  # Ensure your scaler is here

# Define allowed file types
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract 57 features from an audio file
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)  # Load audio file
    features = []

    # Extract features
    features.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))  # Chroma feature
    features.append(np.var(librosa.feature.chroma_stft(y=y, sr=sr)))

    features.append(np.mean(librosa.feature.rms(y=y)))  # RMS energy
    features.append(np.var(librosa.feature.rms(y=y)))

    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))  # Spectral centroid
    features.append(np.var(librosa.feature.spectral_centroid(y=y, sr=sr)))

    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))  # Spectral bandwidth
    features.append(np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))  # Spectral rolloff
    features.append(np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)))

    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))  # Zero crossing rate
    features.append(np.var(librosa.feature.zero_crossing_rate(y)))

    # Extract MFCCs (20 coefficients)
    for i in range(1, 21):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[i - 1]
        features.append(np.mean(mfcc))
        features.append(np.var(mfcc))

    return np.array(features)  # Convert to NumPy array

# Route to extract features from uploaded audio
@app.route('/extract-features', methods=['POST'])
def extract_audio_features():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        features = extract_features(filepath)
        return jsonify({"features": features.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(filepath)  # Clean up after extraction

# Route to predict genre from extracted features
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if "features" not in data:
        return jsonify({"error": "No features provided"}), 400

    try:
        features = np.array(data["features"]).reshape(1, -1)  # Reshape input
        features_scaled = scaler.transform(features)  # Scale input
        prediction = model.predict(features_scaled)  # Make prediction

        genre_mapping = {
            0: "Blues", 1: "Classical", 2: "Country", 3: "Disco", 4: "Hip-Hop",
            5: "Jazz", 6: "Metal", 7: "Pop", 8: "Reggae", 9: "Rock"
        }
        genre = genre_mapping.get(int(prediction[0]), "Unknown Genre")

        return jsonify({"genre": genre})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
