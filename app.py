from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import requests

app = Flask(__name__)

# ==================================================
# DIRECTORIES
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==================================================
# GOOGLE DRIVE FILE IDS
# ==================================================

DRIVE_MODELS = {
    "automind_failure_model.pkl": "1iZOMnXwVzcfTkcqTWs1NtLv9iqTynd6B",
    "automind_scaler.pkl": "1XV1WokgErZ5v0FCtzxIDAl7E0wKP1HjJ",
    "rul_gb_model.pkl": "14huoYQWYoulSOHZmTun0Sul2mOz7pjaL",
    "rul_rf_model.pkl": "1VFxAgLswIdH-GAaKkqe2GJO8wi5Lr_Gj",
    "rul_scaler.pkl": "18AKnDnp4u_vVN8IWWVIiuh6izFNHJJ8z"
}


# ==================================================
# DOWNLOAD FROM GOOGLE DRIVE
# ==================================================

def download_from_drive(file_id, filename):

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(path):
        print(f"✅ {filename} already exists")
        return

    print(f"⬇️ Downloading {filename}...")

    r = requests.get(url, stream=True)

    if r.status_code != 200:
        raise Exception(f"Download failed for {filename}")

    with open(path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            f.write(chunk)

    print(f"✅ Downloaded {filename}")


# ==================================================
# LOAD MODELS
# ==================================================

def load_models():

    print("🔄 Preparing ML models...")

    # Download all
    for name, fid in DRIVE_MODELS.items():
        download_from_drive(fid, name)

    # Load
    models = {}

    for name in DRIVE_MODELS.keys():

        path = os.path.join(MODEL_DIR, name)

        try:
            with open(path, "rb") as f:
                models[name] = pickle.load(f)

            print(f"✅ Loaded {name}")

        except Exception as e:
            print(f"❌ Failed to load {name}")
            raise e

    return models


# ==================================================
# INIT MODELS
# ==================================================

try:
    models = load_models()

    failure_model = models["automind_failure_model.pkl"]
    failure_scaler = models["automind_scaler.pkl"]

    rul_gb_model = models["rul_gb_model.pkl"]
    rul_rf_model = models["rul_rf_model.pkl"]
    rul_scaler = models["rul_scaler.pkl"]

    print("🚀 All models ready")

except Exception as e:
    print("🔥 FATAL ERROR:", e)
    exit(1)


# ==================================================
# ROUTES
# ==================================================

@app.route("/")
def home():
    return "AutoMind ML API Running 🚗🤖"


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

        pred = int(failure_model.predict(scaled)[0])

        prob = float(failure_model.predict_proba(scaled)[0].max())

        return jsonify({
            "failure": pred,
            "confidence": round(prob, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================================================
# RUL GB
# ==================================================

@app.route("/predict/rul/gb", methods=["POST"])
def predict_rul_gb():

    try:

        data = request.get_json()

        if "input" not in data:
            return jsonify({"error": "Missing input"}), 400

        arr = np.array(data["input"]).reshape(1, -1)

        scaled = rul_scaler.transform(arr)

        rul = float(rul_gb_model.predict(scaled)[0])

        return jsonify({
            "rul_gb": round(rul, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================================================
# RUL RF
# ==================================================

@app.route("/predict/rul/rf", methods=["POST"])
def predict_rul_rf():

    try:

        data = request.get_json()

        if "input" not in data:
            return jsonify({"error": "Missing input"}), 400

        arr = np.array(data["input"]).reshape(1, -1)

        scaled = rul_scaler.transform(arr)

        rul = float(rul_rf_model.predict(scaled)[0])

        return jsonify({
            "rul_rf": round(rul, 2)
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
        port=port
    )
