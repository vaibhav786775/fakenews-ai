from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load model
classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-fake-news-detection"
)

# Label mapping for this model
label_map = {
    "LABEL_0": "REAL",
    "LABEL_1": "FAKE"
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Text is required"}), 400

        # Truncate long input for safety
        text = text[:1000]

        result = classifier(text)[0]

        return jsonify({
            "label": label_map.get(result["label"], "UNKNOWN"),
            "confidence": round(result["score"], 3)
        })

    except Exception as e:
        print("PYTHON ERROR:", e)
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)