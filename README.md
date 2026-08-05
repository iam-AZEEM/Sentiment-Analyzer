---
title: Movie Sentiment Analyzer
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Sentiment Analyzer

A web app that classifies text (movie reviews, comments, feedback, etc.) as **Positive** or **Negative**...
# Sentiment Analyzer

A web app that classifies text (movie reviews, comments, feedback, etc.) as **Positive** or **Negative**, powered by a Bidirectional LSTM trained on the IMDB movie reviews dataset. Built with TensorFlow/Keras for the model and Flask for the web interface.

## Demo

Paste any sentence into the text box, hit **Analyze Sentiment**, and get back a prediction with a confidence score.

> Add a screenshot or GIF of the app here once it's running — this is the first thing people look at.

## Features

- Bidirectional LSTM model (128 → 64 units) trained on 50,000 IMDB reviews
- Simple, clean web UI with example prompts and a live confidence meter
- REST API endpoint (`/predict`) that returns sentiment + confidence score as JSON
- Input validation and error handling on both frontend and backend

## Tech Stack

| Layer | Tech |
|---|---|
| Model | TensorFlow / Keras (Bidirectional LSTM) |
| Backend | Flask |
| Frontend | HTML, CSS, vanilla JavaScript |
| Data | Keras built-in IMDB dataset |

## Project Structure

```
sentiment-analyzer/
├── app.py                  # Flask app — serves the UI and /predict API
├── requirements.txt
├── model/
│   ├── model_train.py      # Trains the LSTM model on the IMDB dataset
│   ├── sentiment_model.keras
│   ├── tokenizer.pickle
│   ├── metrics.json         # Test accuracy, precision, recall, F1, confusion matrix
│   └── history.json         # Per-epoch training/validation loss & accuracy
├── static/
│   ├── app.js
│   └── style.css
└── templates/
    └── index.html
```

## Getting Started

### 1. Clone and set up a virtual environment

```bash
git clone <your-repo-url>
cd sentiment-analyzer
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain the model

A pre-trained model is already included in `model/`, so this step is optional. To retrain from scratch:

```bash
python model/model_train.py
```

This downloads the IMDB dataset, trains the LSTM, and saves `sentiment_model.keras`, `tokenizer.pickle`, `metrics.json`, and `history.json` into `model/`.

### 4. Run the app

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

To enable Flask's debug mode for local development:

```bash
# Windows PowerShell
$env:FLASK_DEBUG="true"; python app.py
```

## API

**POST** `/predict`

Request body:
```json
{ "text": "This movie was absolutely wonderful!" }
```

Response:
```json
{
  "text": "This movie was absolutely wonderful!",
  "sentiment": "Positive",
  "score": 0.9123,
  "confidence": 91.23
}
```

## Model Performance

> Fill this in with the numbers from `model/metrics.json` after training, e.g.:

| Metric | Value |
|---|---|
| Test Accuracy | — |
| Precision | — |
| Recall | — |
| F1 Score | — |

## Known Limitations

- Trained only on movie reviews (IMDB), so it may not generalize perfectly to other domains (e.g. product reviews, tweets).
- Simple whitespace + regex tokenization — doesn't handle contractions (e.g. "didn't") ideally.
- Binary classification only (Positive/Negative) — no neutral or mixed-sentiment class.

## Author

Syed Azeem