import json
import pickle
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# All outputs (model, tokenizer, metrics) are saved next to this script,
# i.e. inside model/, no matter which directory you run the script from.
OUTPUT_DIR = Path(__file__).resolve().parent

# ---------------------------
# Config
# ---------------------------
VOCAB_SIZE = 20000
MAX_LEN = 200
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 15
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ---------------------------
# Load data
# ---------------------------
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

# ---------------------------
# Create validation split from training data
# ---------------------------
val_size = int(len(x_train) * 0.2)
x_val = x_train[:val_size]
y_val = y_train[:val_size]
x_train = x_train[val_size:]
y_train = y_train[val_size:]

# ---------------------------
# Pad sequences
# ---------------------------
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="pre", truncating="pre")
x_val = pad_sequences(x_val, maxlen=MAX_LEN, padding="pre", truncating="pre")
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="pre", truncating="pre")

# ---------------------------
# Model
# ---------------------------
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------
# Callbacks
# ---------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=str(OUTPUT_DIR / "sentiment_model.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ---------------------------
# Train
# ---------------------------
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# ---------------------------
# Evaluate
# ---------------------------
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict on the test set to get precision/recall/F1/confusion matrix,
# not just the single accuracy number.
y_prob = model.predict(x_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="binary"
)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["negative", "positive"])

print("\nClassification Report:\n", report)
print("Confusion Matrix (rows=true, cols=predicted):\n", cm)

metrics = {
    "trained_at": datetime.now().isoformat(timespec="seconds"),
    "config": {
        "vocab_size": VOCAB_SIZE,
        "max_len": MAX_LEN,
        "embedding_dim": EMBEDDING_DIM,
        "batch_size": BATCH_SIZE,
        "epochs_configured": EPOCHS,
        "epochs_ran": len(history.history["loss"]),
        "seed": SEED,
    },
    "test_loss": float(loss),
    "test_accuracy": float(accuracy),
    "test_precision": float(precision),
    "test_recall": float(recall),
    "test_f1": float(f1),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
}

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open(OUTPUT_DIR / "history.json", "w") as f:
    json.dump(history.history, f, indent=2)

print("Saved evaluation metrics to metrics.json and training history to history.json")

# ---------------------------
# Save word index
# ---------------------------
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

with open(OUTPUT_DIR / "tokenizer.pickle", "wb") as f:
    pickle.dump(word_index, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Tokenizer saved as tokenizer.pickle")

# ---------------------------
# Inference helper
# ---------------------------
def encode_review(text, word_index, max_len=MAX_LEN):
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = cleaned.split()
    encoded = [1]  # <START>
    for word in tokens:
        idx = word_index.get(word, 2)
        encoded.append(idx if idx < VOCAB_SIZE else 2)
    return pad_sequences([encoded], maxlen=max_len, padding="pre", truncating="pre")

sample_review = "this movie was absolutely wonderful and amazing"
encoded_sample = encode_review(sample_review, word_index)
prediction = model.predict(encoded_sample, verbose=0)[0][0]

print(f"Sample Review: {sample_review}")
print(f"Predicted Sentiment: {'Positive' if prediction > 0.5 else 'Negative'} (Confidence: {prediction:.2f})")