from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import fitz  # PyMuPDF for PDFs
import pandas as pd

app = Flask(__name__)

# -------------------- Load HuggingFace Sentiment Model --------------------
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sent_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# -------------------- File Extraction Utility --------------------
def extract_text_from_file(file):
    filename = file.filename.lower()

    if filename.endswith(".txt"):
        return file.read().decode("utf-8")

    elif filename.endswith(".csv"):
        df = pd.read_csv(file)
        # Join all rows into one string
        return "\n".join(df.astype(str).apply(lambda x: " ".join(x), axis=1).tolist())

    elif filename.endswith(".pdf"):
        text = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text

    else:
        return None

# -------------------- Sentiment Analyzer --------------------
def analyze_sentiment(text):
    sentences = [s.strip() for s in text.split("\n") if s.strip()]
    results = sent_pipeline(sentences)

    detailed_sentiments = []
    sentiment_counts = {v: 0 for v in sentiment_map.values()}

    for sentence, result in zip(sentences, results):
        label = sentiment_map.get(result["label"], "Unknown")
        detailed_sentiments.append((sentence, label))
        if label in sentiment_counts:
            sentiment_counts[label] += 1

    total = len(sentences)
    sentiment_summary = {k: round((v / total) * 100, 2) for k, v in sentiment_counts.items()}

    return sentiment_summary, detailed_sentiments


# -------------------- Routes --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_text = request.form.get("text", "").strip()
        file = request.files.get("file")

        # If file uploaded, extract text
        if file and file.filename != "":
            extracted_text = extract_text_from_file(file)
            if extracted_text:
                user_text = extracted_text

        if not user_text:
            return render_template(
                "index.html",
                error="Please enter text or upload a valid file.",
                user_input="",
                sentiment_summary=None,
                detailed_sentiments=None
            )

        sentiment_summary, detailed_sentiments = analyze_sentiment(user_text)

        return render_template(
            "index.html",
            user_input=user_text,
            sentiment_summary=sentiment_summary,
            detailed_sentiments=detailed_sentiments,
            error=None
        )

    # Initial load
    return render_template(
        "index.html",
        user_input="",
        sentiment_summary=None,
        detailed_sentiments=None,
        error=None
    )

# API endpoint for AJAX/JSON usage
@app.route("/analyze", methods=["POST"])
def analyze_api():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    sentiment_summary, detailed_sentiments = analyze_sentiment(text)
    return jsonify({
        "summary": sentiment_summary,
        "details": detailed_sentiments
    })


if __name__ == "__main__":
    app.run(debug=True)
