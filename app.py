from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# ==================================================
# BASE DIRECTORY (IMPORTANT FOR RENDER)
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ==================================================
# LOAD MODELS
# ==================================================

def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


try:
    # Failure prediction
    failure_model = load_model("automind_failure_model.pkl")
    failure_scaler = load_model("automind_scaler.pkl")

    # RUL models
    rul_gb_model = load_model("rul_gb_model.pkl")
    rul_rf_model = load_model("rul_rf_model.pkl")
    rul_scaler = load_model("rul_scaler.pkl")

    print("✅ All ML models loaded successfully")

except Exception as e:
    print("❌ Error loading models:")
    print(e)
    exit(1)


# ==================================================
# ROUTES
# ==================================================

@app.route("/", methods=["GET"])
def home():
    return "AutoMind ML API is Running 🚗🤖"


# ==================================================
# FAILURE PREDICTION
# ==================================================

@app.route("/predict/failure", methods=["POST"])
def predict_failure():

    try:
        data = request.get_json()

        if "input" not in data:
            return jsonify({"error": "Missing input"}), 400

        arr = np.array(data["input"]).reshape(1, -1)

        scaled = failure_scaler.transform(arr)

        pred = failure_model.predict(scaled)[0]

        prob = failure_model.predict_proba(scaled)[0].max()

        return jsonify({
            "failure": int(pred),
            "confidence": round(float(prob), 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================================================
# RUL (GRADIENT BOOSTING)
# ==================================================

@app.route("/predict/rul/gb", methods=["POST"])
def predict_rul_gb():

    try:
        data = request.get_json()

        if "input" not in data:
            return jsonify({"error": "Missing input"}), 400

        arr = np.array(data["input"]).reshape(1, -1)

        scaled = rul_scaler.transform(arr)

        rul = rul_gb_model.predict(scaled)[0]

        return jsonify({
            "rul_gb": round(float(rul), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================================================
# RUL (RANDOM FOREST)
# ==================================================

@app.route("/predict/rul/rf", methods=["POST"])
def predict_rul_rf():

    try:
        data = request.get_json()

        if "input" not in data:
            return jsonify({"error": "Missing input"}), 400

        arr = np.array(data["input"]).reshape(1, -1)

        scaled = rul_scaler.transform(arr)

        rul = rul_rf_model.predict(scaled)[0]

        return jsonify({
            "rul_rf": round(float(rul), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
