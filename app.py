import os
import pickle
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))

# Config must match training script exactly
MODEL_PATH = BASE_DIR / "model" / "sentiment_model.keras"
TOKENIZER_PATH = BASE_DIR / "model" / "tokenizer.pickle"
VOCAB_SIZE = 20000
MAX_LEN = 200

# ---------------------------
# Load Model & Tokenizer
# ---------------------------
print("Loading model and tokenizer...")
model = load_model(str(MODEL_PATH))

with open(TOKENIZER_PATH, "rb") as f:
    word_index = pickle.load(f)

print("Model & tokenizer loaded successfully!")

# ---------------------------
# Helper Functions
# ---------------------------
def preprocess_text(text: str) -> np.ndarray:
    tokens = text.lower().split()
    encoded = [1]  # <START>

    for word in tokens:
        idx = word_index.get(word, 2)  # <UNK>
        encoded.append(idx if idx < VOCAB_SIZE else 2)

    padded = pad_sequences([encoded], maxlen=MAX_LEN, padding="pre", truncating="pre")
    return padded

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' in JSON request body"}), 400

    user_text = str(data["text"]).strip()
    if not user_text:
        return jsonify({"error": "Text cannot be empty"}), 400

    encoded_text = preprocess_text(user_text)
    prediction = float(model.predict(encoded_text, verbose=0)[0][0])

    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    confidence = prediction if sentiment == "Positive" else (1.0 - prediction)

    return jsonify({
        "text": user_text,
        "sentiment": sentiment,
        "score": round(prediction, 4),
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)