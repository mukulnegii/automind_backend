from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import sys
import traceback

app = Flask(__name__)


# ==================================================
# BASE DIRECTORY (FOR RENDER + LOCAL)
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ==================================================
# SAFE MODEL LOADER
# ==================================================

def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model not found: {path}")

    try:
        with open(path, "rb") as f:
            return pickle.load(f)

    except Exception as e:
        print(f"❌ Failed to load {filename}")
        raise e


# ==================================================
# LOAD ALL MODELS ON START
# ==================================================

try:
    print("🔄 Loading ML models...")

    # Failure
    failure_model = load_model("automind_failure_model.pkl")
    failure_scaler = load_model("automind_scaler.pkl")

    # RUL
    rul_gb_model = load_model("rul_gb_model.pkl")
    rul_rf_model = load_model("rul_rf_model.pkl")
    rul_scaler = load_model("rul_scaler.pkl")

    print("✅ All models loaded successfully!")

except Exception as e:
    print("🔥 FATAL ERROR: Models not loaded")
    print(e)
    sys.exit(1)


# ==================================================
# HEALTH CHECK (IMPORTANT FOR RENDER)
# ==================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "service": "AutoMind ML API",
        "models_loaded": True
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ==================================================
# FAILURE PREDICTION
# ==================================================

@app.route("/predict/failure", methods=["POST"])
def predict_failure():

    try:
        data = request.get_json()

        if not data or "input" not in data:
            return jsonify({"error": "Missing input array"}), 400

        arr = np.array(data["input"], dtype=float).reshape(1, -1)

        scaled = failure_scaler.transform(arr)

        pred = int(failure_model.predict(scaled)[0])

        prob = float(failure_model.predict_proba(scaled)[0].max())

        return jsonify({
            "failure": pred,
            "confidence": round(prob, 3)
        })


    except Exception as e:
        traceback.print_exc()

        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


# ==================================================
# RUL (GRADIENT BOOSTING)
# ==================================================

@app.route("/predict/rul/gb", methods=["POST"])
def predict_rul_gb():

    try:
        data = request.get_json()

        if not data or "input" not in data:
            return jsonify({"error": "Missing input array"}), 400

        arr = np.array(data["input"], dtype=float).reshape(1, -1)

        scaled = rul_scaler.transform(arr)

        rul = float(rul_gb_model.predict(scaled)[0])

        return jsonify({
            "rul_gb": round(rul, 2)
        })


    except Exception as e:
        traceback.print_exc()

        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


# ==================================================
# RUL (RANDOM FOREST)
# ==================================================

@app.route("/predict/rul/rf", methods=["POST"])
def predict_rul_rf():

    try:
        data = request.get_json()

        if not data or "input" not in data:
            return jsonify({"error": "Missing input array"}), 400

        arr = np.array(data["input"], dtype=float).reshape(1, -1)

        scaled = rul_scaler.transform(arr)

        rul = float(rul_rf_model.predict(scaled)[0])

        return jsonify({
            "rul_rf": round(rul, 2)
        })


    except Exception as e:
        traceback.print_exc()

        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


# ==================================================
# MAIN (FOR RENDER)
# ==================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
