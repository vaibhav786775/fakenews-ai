from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# load model
classifier = pipeline(
    "text-classification",
    model="jy46604790/Fake-News-Bert-Detect"
)

# correct label mapping for this model
label_map = {
    "LABEL_0": "FAKE",
    "LABEL_1": "REAL"
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "Text is required"}), 400

        # keep input safe (truncate long text)
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
    app.run(port=5000, debug=True)