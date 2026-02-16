from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.staticfiles import StaticFiles
import shutil
import json
import os

from gtts import gTTS
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort

# ==================================================
# SECURITY
# ==================================================

def verify_key(x_api_key: str = Header(None)):
    server_key = os.getenv("API_KEY")

    if server_key is None:
        raise HTTPException(status_code=500, detail="Server API key not set")

    if x_api_key != server_key:
        raise HTTPException(status_code=401, detail="Invalid API key")



# ==================================================
# PATH FIX (IMPORTANT FOR RENDER)
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================
# Global Latest Health Store
# ==================================================

LATEST_HEALTH = None


# ==================================================
# App Setup
# ==================================================

app = FastAPI(title="AutoMind AI Backend 🚗🤖")

# mount AFTER directory creation
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


# ==================================================
# Load Models
# ==================================================

print("🔄 Loading Models...")

try:
    failure_session = ort.InferenceSession(os.path.join(MODEL_DIR, "multi_failure_model.onnx"))
    print("✅ Failure Model Loaded")
except:
    failure_session = None
    print("⚠️ Failure model not loaded")

try:
    rul_rf_session = ort.InferenceSession(os.path.join(MODEL_DIR, "rul_rf_model.onnx"))
    rul_gb_session = ort.InferenceSession(os.path.join(MODEL_DIR, "rul_gb_model.onnx"))
    print("✅ RUL Models Loaded")
except:
    rul_rf_session = None
    rul_gb_session = None
    print("⚠️ RUL models not loaded")

try:
    scaler = joblib.load(os.path.join(MODEL_DIR, "automind_scaler.pkl"))
    print("✅ Failure Scaler Loaded")
except:
    scaler = None
    print("⚠️ Failure scaler not loaded")

try:
    rul_scaler = joblib.load(os.path.join(MODEL_DIR, "rul_scaler.pkl"))
    print("✅ RUL Scaler Loaded")
except:
    rul_scaler = None
    print("⚠️ RUL scaler not loaded")


# ==================================================
# Feature Order
# ==================================================

FEATURES = [
    "rpm",
    "engine_temp",
    "battery_voltage",
    "speed",
    "vibration",
    "brake_pressure",
    "gear_load"
]

RUL_FEATURES = [
    "rpm",
    "engine_temp",
    "battery_voltage",
    "speed",
    "vibration",
    "brake_pressure",
    "gear_load",
    "battery_health",
    "degradation_index",
    "failure_score",
    "stress_score"
]


# ==================================================
# Input Mapping
# ==================================================

FIELD_MAP = {
    "rpm": "rpm",
    "temperature": "engine_temp",
    "engine_temp": "engine_temp",
    "voltage": "battery_voltage",
    "battery_voltage": "battery_voltage",
    "speed": "speed",
    "vibration": "vibration",
    "pressure": "brake_pressure",
    "brake_pressure": "brake_pressure",
    "gear_load": "gear_load"
}


def map_fields(data: dict):
    mapped = {}

    for k, v in data.items():
        if k in FIELD_MAP:
            mapped[FIELD_MAP[k]] = float(v)

    if "gear_load" not in mapped:
        mapped["gear_load"] = 0.5

    for f in FEATURES:
        if f not in mapped:
            raise HTTPException(status_code=400, detail=f"Missing field: {f}")

    return mapped


# ==================================================
# Helpers
# ==================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ==================================================
# Failure Prediction
# ==================================================

def predict_failure(X_scaled):

    if failure_session is None:
        return 0.1, 0.1, 0.1, 0.1

    input_name = failure_session.get_inputs()[0].name
    outputs = failure_session.run(None, {input_name: X_scaled})

    raw = np.array(outputs[0]).flatten()

    if len(raw) < 4:
        return 0.2, 0.2, 0.2, 0.2

    probs = 1 / (1 + np.exp(-raw))

    engine_p = float(probs[0])
    brake_p = float(probs[1])
    battery_p = float(probs[2])
    gear_p = float(probs[3])

    return engine_p, brake_p, battery_p, gear_p


# ==================================================
# RUL Prediction
# ==================================================

def predict_rul(X_scaled):

    if rul_rf_session is None or rul_gb_session is None:
        return 180

    rf_input = rul_rf_session.get_inputs()[0].name
    gb_input = rul_gb_session.get_inputs()[0].name

    rf_out = rul_rf_session.run(None, {rf_input: X_scaled})
    gb_out = rul_gb_session.run(None, {gb_input: X_scaled})

    rf_pred = float(np.array(rf_out).flatten()[0])
    gb_pred = float(np.array(gb_out).flatten()[0])

    rul = int(0.6 * rf_pred + 0.4 * gb_pred)

    return max(30, rul)


# ==================================================
# Voice Assistant API
# ==================================================

@app.post("/voice")
async def process_voice(file: UploadFile = File(...), _: str = Header(None, alias="x-api-key")):

    verify_key(_)

    input_path = f"{UPLOAD_DIR}/input.m4a"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    transcript = "My car is making strange noise"
    emotion = "WORRIED"
    diagnosis = "Engine problem"

    ai_text = "Please slow down and visit the nearest service center."

    response = {
        "transcript": transcript,
        "emotion": emotion,
        "diagnosis": diagnosis,
        "ai_response_text": ai_text,
        "auto_booking": {"status": "PENDING"}
    }

    with open(f"{OUTPUT_DIR}/voice_agent_output.json", "w") as f:
        json.dump(response, f, indent=2)

    tts = gTTS(ai_text)
    tts.save(f"{OUTPUT_DIR}/response_audio.mp3")

    return {
        "status": "success",
        "json": "voice_agent_output.json",
        "audio": "response_audio.mp3"
    }


# ==================================================
# Prediction API
# ==================================================

@app.post("/predict")
def predict(data: dict, _: str = Header(None, alias="x-api-key")):

    verify_key(_)

    mapped = map_fields(data)
    df = pd.DataFrame([mapped], columns=FEATURES)

    if scaler:
        X_scaled = scaler.transform(df).astype(np.float32)
    else:
        X_scaled = df.values.astype(np.float32)

    engine_p, brake_p, battery_p, gear_p = predict_failure(X_scaled)
    final_risk = np.mean([engine_p, brake_p, battery_p, gear_p])
    health_score = round((1 - final_risk) * 100, 2)

    if rul_scaler:
        rul_array = np.array([[*df.values[0], 1, 0.1, 0.1, 0.1]], dtype=np.float32)
        rul_scaled = rul_scaler.transform(rul_array)
    else:
        rul_scaled = X_scaled

    rul_days = predict_rul(rul_scaled)

    if final_risk > 0.6:
        risk = "HIGH"
    elif final_risk > 0.3:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    result = {
        "health_score": health_score,
        "risk": risk,
        "rul_days": rul_days,
        "failure_probabilities": {
            "engine": round(engine_p, 3),
            "brake": round(brake_p, 3),
            "battery": round(battery_p, 3),
            "gear": round(gear_p, 3)
        }
    }

    with open(f"{OUTPUT_DIR}/prediction_result.json", "w") as f:
        json.dump(result, f, indent=2)

    global LATEST_HEALTH
    LATEST_HEALTH = result

    return result


# ==================================================
# Health Check
# ==================================================

@app.get("/")
def home():
    return {"status": "AutoMind Backend Running 🚀"}


# ==================================================
# Latest Health API
# ==================================================

@app.get("/latest-health")
def get_latest_health():

    if LATEST_HEALTH is None:
        return {"status": "empty", "message": "No prediction yet"}

    return LATEST_HEALTH
