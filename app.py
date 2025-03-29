from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("models/music_genre_classifier.pkl")
scaler = joblib.load("models/scaler.pkl")

genre_mapping = {
    0: "Blues",
    1: "Classical",
    2: "Country",
    3: "Disco",
    4: "HipHop",
    5: "Jazz",
    6: "Metal",
    7: "Pop",
    8: "Reggae",
    9: "Rock"
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert features to a NumPy array
    features = np.array(data["features"]).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)

    # Predict genre
    prediction = model.predict(features_scaled)

    genre = genre_mapping.get(int(prediction), "Unknown Genre")

    return jsonify({"genre": genre})

if __name__ == '__main__':
    app.run(debug=True)
