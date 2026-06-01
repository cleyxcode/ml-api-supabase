"""
Siram Pintar API v13.0
======================

Perubahan dari v12:
  - Hapus label Hujan_Aktif & Hujan_Prediksi
  - Formula ET diganti VPD-based (konsisten dengan dataset v7)
  - is_hot threshold: 35 -> 34 (sesuai dataset)
  - Load model: 2 file terpisah (knn_model.pkl + scaler.pkl)
  - Kode lebih clean dan ringkas

Logika AUTO:
  - Window pagi 05:00-06:59 dan sore 16:00-17:59
  - KNN memutuskan: Siram_Segera/Siram_Prioritas -> pompa ON 20 menit
  - Siram_Nanti/Optimal/Basah -> pompa OFF
  - Maksimal 1x siram per window per hari

Logika MANUAL:
  - Pompa dikendalikan dashboard via POST /control

Endpoints:
  GET  /                  -> health check
  GET  /status            -> status sistem lengkap
  GET  /pump-status       -> polling ESP32
  POST /sensor            -> data sensor dari ESP32
  POST /control           -> kontrol manual dashboard
  GET  /history           -> riwayat sensor
  POST /test-knn          -> simulasi KNN (pompa tidak nyala)
  GET  /test-knn/skenario -> 12 skenario preset
  POST /test-knn/batch    -> uji banyak skenario
  POST /test-knn/reset    -> reset state
  POST /test-knn/fire     -> test pompa nyala sungguhan
"""

import os
import math
import logging
from datetime import datetime, date
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
from supabase import create_client, Client

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────────────────────────────────────
VERSION      = "13.0"
API_KEY      = os.getenv("API_KEY",      "yuli1")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
MODEL_PATH   = os.getenv("MODEL_PATH",   "model/knn_model.pkl")
SCALER_PATH  = os.getenv("SCALER_PATH",  "model/scaler.pkl")

WINDOW_PAGI           = (5, 7)   # 05:00 - 06:59
WINDOW_SORE           = (16, 18) # 16:00 - 17:59
PUMP_DURATION_MINUTES = 20
IS_HOT_THRESHOLD      = 34.0     # sesuai dataset v7

LABEL_SIRAM = {"Siram_Segera", "Siram_Prioritas"}
LABEL_SKIP  = {"Siram_Nanti", "Optimal", "Basah"}

# ─────────────────────────────────────────────────────────────────────────────
# Koneksi Supabase
# ─────────────────────────────────────────────────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# Load Model KNN
# ─────────────────────────────────────────────────────────────────────────────
knn_model  = None
knn_scaler = None
model_info = {}

try:
    knn_model  = joblib.load(MODEL_PATH)
    knn_scaler = joblib.load(SCALER_PATH)
    model_info = {
        "algorithm" : "K-Nearest Neighbor",
        "best_k"    : getattr(knn_model, "n_neighbors", "?"),
        "features"  : [
            "soil_moisture", "temperature", "air_humidity",
            "hour_sin", "hour_cos", "soil_trend",
            "evapotranspiration", "is_hot"
        ],
        "labels"    : list(LABEL_SIRAM | LABEL_SKIP),
        "version"   : VERSION,
    }
    log.info(f"[MODEL] KNN dimuat — k={model_info['best_k']}")
except Exception as e:
    log.warning(f"[MODEL] Gagal muat model: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Siram Pintar API", version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────
class SensorPayload(BaseModel):
    soil_moisture : float = Field(..., ge=0,   le=100)
    temperature   : float = Field(..., ge=-10, le=60)
    air_humidity  : float = Field(..., ge=0,   le=100)
    hour          : Optional[int] = Field(None, ge=0, le=23)
    minute        : Optional[int] = Field(None, ge=0, le=59)

class ControlPayload(BaseModel):
    action : str  # "on" / "off"
    mode   : str  # "manual" / "auto"

class TestPayload(BaseModel):
    soil_moisture  : float
    temperature    : float
    air_humidity   : float
    hour           : int
    soil_prev      : Optional[float] = None
    label_skenario : Optional[str]   = None
    ekspektasi     : Optional[str]   = None

class BatchPayload(BaseModel):
    skenario: List[TestPayload]

class FirePayload(BaseModel):
    soil_moisture  : float
    temperature    : float
    air_humidity   : float
    hour           : int             = Field(..., ge=0, le=23)
    soil_prev      : Optional[float] = None
    label_skenario : Optional[str]   = None
    ekspektasi     : Optional[str]   = None

# ─────────────────────────────────────────────────────────────────────────────
# State In-Memory
# ─────────────────────────────────────────────────────────────────────────────
class SystemState:
    pump_status       : bool               = False
    mode              : str                = "auto"
    manual_override   : bool               = False
    pump_start_ts     : Optional[datetime] = None
    last_watered_pagi : Optional[date]     = None
    last_watered_sore : Optional[date]     = None
    last_soil         : float              = 0.0
    last_temp         : float              = 0.0
    last_rh           : float              = 0.0
    last_hour         : int                = 0
    last_knn_label    : str                = "---"
    last_knn_conf     : float              = 0.0

state = SystemState()

# ─────────────────────────────────────────────────────────────────────────────
# State: Load & Save
# ─────────────────────────────────────────────────────────────────────────────
def load_state():
    try:
        res = supabase.table("system_state").select("*").eq("id", 1).single().execute()
        if not res.data:
            return
        d = res.data
        state.pump_status     = d.get("pump_status", False)
        state.mode            = d.get("mode", "auto")
        state.manual_override = d.get("manual_override", False)
        state.last_soil       = d.get("last_soil", 0.0)
        state.last_knn_label  = d.get("last_knn_label", "---")
        state.last_knn_conf   = d.get("last_knn_conf", 0.0)

        lp = d.get("last_watered_pagi")
        ls = d.get("last_watered_sore")
        state.last_watered_pagi = date.fromisoformat(lp) if lp else None
        state.last_watered_sore = date.fromisoformat(ls) if ls else None

        pts = d.get("pump_start_ts")
        state.pump_start_ts = datetime.fromisoformat(pts) if pts else None

        log.info(f"[STATE] Dimuat — pump={state.pump_status} mode={state.mode}")
    except Exception as e:
        log.warning(f"[STATE] Gagal muat: {e}")


def save_state():
    try:
        supabase.table("system_state").upsert({
            "id"               : 1,
            "pump_status"      : state.pump_status,
            "mode"             : state.mode,
            "manual_override"  : state.manual_override,
            "last_soil"        : state.last_soil,
            "last_knn_label"   : state.last_knn_label,
            "last_knn_conf"    : state.last_knn_conf,
            "last_watered_pagi": state.last_watered_pagi.isoformat() if state.last_watered_pagi else None,
            "last_watered_sore": state.last_watered_sore.isoformat() if state.last_watered_sore else None,
            "pump_start_ts"    : state.pump_start_ts.isoformat() if state.pump_start_ts else None,
            "updated_at"       : datetime.utcnow().isoformat(),
        }).execute()
    except Exception as e:
        log.warning(f"[STATE] Gagal simpan: {e}")


@app.on_event("startup")
def on_startup():
    load_state()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Auth
# ─────────────────────────────────────────────────────────────────────────────
def check_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key salah")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: KNN
# ─────────────────────────────────────────────────────────────────────────────
def calc_et(temp: float, rh: float) -> float:
    """Evapotranspiration VPD-based — konsisten dengan dataset v7"""
    vpd = (1 - rh / 100) * 0.6108 * math.exp(17.27 * temp / (temp + 237.3))
    return round(min(max(vpd * 15, 0), 100), 2)


def run_knn(
    soil      : float,
    temp      : float,
    rh        : float,
    hour      : int,
    soil_prev : Optional[float] = None,
) -> dict:
    if knn_model is None or knn_scaler is None:
        return {
            "label": "Siram_Segera", "confidence": 0.0,
            "features": {}, "model_ready": False
        }

    hour_sin   = math.sin(2 * math.pi * hour / 24)
    hour_cos   = math.cos(2 * math.pi * hour / 24)
    soil_trend = (soil - soil_prev) if soil_prev is not None else 0.0
    et         = calc_et(temp, rh)
    is_hot     = 1.0 if temp >= IS_HOT_THRESHOLD else 0.0

    X = np.array([[soil, temp, rh, hour_sin, hour_cos, soil_trend, et, is_hot]])
    X_scaled   = knn_scaler.transform(X)
    label      = knn_model.predict(X_scaled)[0]
    proba      = knn_model.predict_proba(X_scaled)[0]
    confidence = float(np.max(proba))

    return {
        "label"      : str(label),
        "confidence" : round(confidence, 4),
        "model_ready": True,
        "features"   : {
            "soil_moisture"     : soil,
            "temperature"       : temp,
            "air_humidity"      : rh,
            "hour_sin"          : round(hour_sin, 4),
            "hour_cos"          : round(hour_cos, 4),
            "soil_trend"        : round(soil_trend, 2),
            "evapotranspiration": et,
            "is_hot"            : is_hot,
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Pompa & Window
# ─────────────────────────────────────────────────────────────────────────────
def get_window(hour: int) -> Optional[str]:
    if WINDOW_PAGI[0] <= hour < WINDOW_PAGI[1]:
        return "pagi"
    if WINDOW_SORE[0] <= hour < WINDOW_SORE[1]:
        return "sore"
    return None


def already_watered(window: str, today: date) -> bool:
    if window == "pagi":
        return state.last_watered_pagi == today
    if window == "sore":
        return state.last_watered_sore == today
    return False


def mark_watered(window: str, today: date):
    if window == "pagi":
        state.last_watered_pagi = today
    elif window == "sore":
        state.last_watered_sore = today


def set_pump(on: bool, reason: str = ""):
    if on == state.pump_status:
        return
    state.pump_status   = on
    state.pump_start_ts = datetime.utcnow() if on else None
    log.info(f"[POMPA] {'ON' if on else 'OFF'} — {reason}")


def check_timeout():
    if not state.pump_status or state.pump_start_ts is None:
        return
    elapsed = (datetime.utcnow() - state.pump_start_ts).total_seconds() / 60
    if elapsed >= PUMP_DURATION_MINUTES:
        set_pump(False, f"Timeout {PUMP_DURATION_MINUTES} menit")
        save_state()


def pump_remaining() -> float:
    if not state.pump_status or state.pump_start_ts is None:
        return 0.0
    elapsed = (datetime.utcnow() - state.pump_start_ts).total_seconds() / 60
    return max(0.0, round(PUMP_DURATION_MINUTES - elapsed, 1))

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Log Sensor & Build Response
# ─────────────────────────────────────────────────────────────────────────────
def log_to_db(payload: SensorPayload, hour: int, window: Optional[str],
              knn: dict, pump_action: Optional[str], reason: str):
    try:
        supabase.table("sensor_readings").insert({
            "soil_moisture" : payload.soil_moisture,
            "temperature"   : payload.temperature,
            "air_humidity"  : payload.air_humidity,
            "hour"          : hour,
            "window"        : window,
            "knn_label"     : knn.get("label"),
            "knn_confidence": knn.get("confidence"),
            "pump_action"   : pump_action,
            "pump_status"   : state.pump_status,
            "mode"          : state.mode,
            "reason"        : reason,
            "recorded_at"   : datetime.utcnow().isoformat(),
        }).execute()
    except Exception as e:
        log.warning(f"[DB] Gagal simpan: {e}")


def build_response(knn: dict, pump_action: Optional[str], reason: str) -> dict:
    return {
        "pump_status"   : state.pump_status,
        "pump_action"   : pump_action,
        "mode"          : state.mode,
        "classification": {
            "label"     : knn["label"],
            "confidence": knn["confidence"],
        },
        "auto_info": {
            "reason"            : reason,
            "pump_remaining_min": pump_remaining(),
            "manual_override"   : state.manual_override,
        },
        "features": knn.get("features", {}),
    }

# ─────────────────────────────────────────────────────────────────────────────
# GET /
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {
        "status"      : "online",
        "version"     : VERSION,
        "model_ready" : knn_model is not None,
        "pump_status" : state.pump_status,
        "mode"        : state.mode,
        "logic": {
            "window_pagi"     : f"{WINDOW_PAGI[0]:02d}:00 - {WINDOW_PAGI[1]:02d}:00",
            "window_sore"     : f"{WINDOW_SORE[0]:02d}:00 - {WINDOW_SORE[1]:02d}:00",
            "pump_duration"   : f"{PUMP_DURATION_MINUTES} menit",
            "label_siram"     : list(LABEL_SIRAM),
            "label_skip"      : list(LABEL_SKIP),
            "is_hot_threshold": f">= {IS_HOT_THRESHOLD}C",
            "et_formula"      : "VPD-based",
        },
        "model_info": model_info,
    }

# ─────────────────────────────────────────────────────────────────────────────
# GET /status
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/status")
def get_status(x_api_key: str = Header(...)):
    check_api_key(x_api_key)
    check_timeout()
    return {
        "pump_status"         : state.pump_status,
        "pump_remaining_min"  : pump_remaining(),
        "mode"                : state.mode,
        "manual_override"     : state.manual_override,
        "last_knn_label"      : state.last_knn_label,
        "last_knn_confidence" : state.last_knn_conf,
        "last_soil"           : state.last_soil,
        "last_watered_pagi"   : state.last_watered_pagi.isoformat() if state.last_watered_pagi else None,
        "last_watered_sore"   : state.last_watered_sore.isoformat() if state.last_watered_sore else None,
        "windows": {
            "pagi": f"{WINDOW_PAGI[0]:02d}:00 - {WINDOW_PAGI[1]:02d}:00",
            "sore": f"{WINDOW_SORE[0]:02d}:00 - {WINDOW_SORE[1]:02d}:00",
        },
        "pump_duration_minutes": PUMP_DURATION_MINUTES,
    }

# ─────────────────────────────────────────────────────────────────────────────
# GET /pump-status
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/pump-status")
def get_pump_status(x_api_key: str = Header(...)):
    check_api_key(x_api_key)
    check_timeout()
    return {
        "pump_status"       : state.pump_status,
        "mode"              : state.mode,
        "manual_override"   : state.manual_override,
        "pump_remaining_min": pump_remaining(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# POST /sensor
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/sensor")
def post_sensor(payload: SensorPayload, x_api_key: str = Header(...)):
    check_api_key(x_api_key)
    check_timeout()

    hour  = payload.hour if payload.hour is not None else datetime.utcnow().hour
    today = date.today()

    # Simpan data sebelumnya untuk soil_trend
    soil_prev       = state.last_soil
    state.last_soil = payload.soil_moisture
    state.last_temp = payload.temperature
    state.last_rh   = payload.air_humidity
    state.last_hour = hour

    # Jalankan KNN
    knn = run_knn(
        soil      = payload.soil_moisture,
        temp      = payload.temperature,
        rh        = payload.air_humidity,
        hour      = hour,
        soil_prev = soil_prev if soil_prev > 0 else None,
    )
    state.last_knn_label = knn["label"]
    state.last_knn_conf  = knn["confidence"]

    pump_action = None

    # Mode MANUAL — KNN tetap jalan tapi tidak mempengaruhi pompa
    if state.mode == "manual":
        reason = f"Mode MANUAL | KNN: {knn['label']}"
        log_to_db(payload, hour, None, knn, pump_action, reason)
        save_state()
        return build_response(knn, pump_action, reason)

    # Mode AUTO
    window = get_window(hour)

    if window is None:
        reason = f"Di luar window siram (jam {hour:02d}:xx) | KNN: {knn['label']}"
        log_to_db(payload, hour, window, knn, pump_action, reason)
        save_state()
        return build_response(knn, pump_action, reason)

    if already_watered(window, today):
        reason = f"Sudah siram {window} hari ini | KNN: {knn['label']}"
        log_to_db(payload, hour, window, knn, pump_action, reason)
        save_state()
        return build_response(knn, pump_action, reason)

    if state.pump_status:
        reason = f"Pompa sedang ON — sisa {pump_remaining()} menit | KNN: {knn['label']}"
        log_to_db(payload, hour, window, knn, pump_action, reason)
        save_state()
        return build_response(knn, pump_action, reason)

    # KNN memutuskan
    label = knn["label"]
    conf  = round(knn["confidence"] * 100, 1)

    if label in LABEL_SIRAM:
        set_pump(True, f"KNN={label} ({conf}%) window={window}")
        mark_watered(window, today)
        pump_action = "on"
        reason = f"KNN: {label} ({conf}%) | window {window} | pompa ON {PUMP_DURATION_MINUTES} menit"
    else:
        reason = f"KNN: {label} ({conf}%) | window {window} | pompa tetap OFF"

    log_to_db(payload, hour, window, knn, pump_action, reason)
    save_state()
    return build_response(knn, pump_action, reason)

# ─────────────────────────────────────────────────────────────────────────────
# POST /control
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/control")
def post_control(payload: ControlPayload, x_api_key: str = Header(...)):
    check_api_key(x_api_key)

    if payload.action not in ("on", "off"):
        raise HTTPException(status_code=400, detail="action harus 'on' atau 'off'")
    if payload.mode not in ("manual", "auto"):
        raise HTTPException(status_code=400, detail="mode harus 'manual' atau 'auto'")

    state.mode = payload.mode

    if payload.mode == "manual":
        state.manual_override = True
        set_pump(payload.action == "on", f"MANUAL — {payload.action}")
    else:
        state.manual_override = False
        set_pump(False, "Kembali ke AUTO")

    save_state()
    return {
        "pump_status"    : state.pump_status,
        "mode"           : state.mode,
        "manual_override": state.manual_override,
        "message"        : f"Pompa {'ON' if state.pump_status else 'OFF'} — mode {state.mode}",
    }

# ─────────────────────────────────────────────────────────────────────────────
# GET /history
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/history")
def get_history(
    x_api_key : str  = Header(...),
    limit     : int  = Query(20, ge=1, le=100),
    pump_only : bool = Query(False),
):
    check_api_key(x_api_key)
    try:
        q = (supabase.table("sensor_readings")
             .select("*")
             .order("recorded_at", desc=True)
             .limit(limit))
        if pump_only:
            q = q.eq("pump_action", "on")
        res = q.execute()
        return {"count": len(res.data), "data": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# POST /test-knn  (simulasi — pompa tidak nyala)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/test-knn")
def test_knn(payload: TestPayload, x_api_key: str = Header(...)):
    check_api_key(x_api_key)

    knn    = run_knn(payload.soil_moisture, payload.temperature,
                     payload.air_humidity, payload.hour, payload.soil_prev)
    label  = knn["label"]
    conf   = round(knn["confidence"] * 100, 1)
    window = get_window(payload.hour)

    if window:
        if label in LABEL_SIRAM:
            keputusan = f"SIRAM — {label} ({conf}%) | window {window} | pompa ON {PUMP_DURATION_MINUTES} menit"
        else:
            keputusan = f"TIDAK SIRAM — {label} ({conf}%) | window {window}"
    else:
        keputusan = f"DI LUAR JADWAL (jam {payload.hour:02d}:xx) — {label} ({conf}%)"

    return {
        "label_skenario": payload.label_skenario,
        "classification": {
            "label"         : label,
            "confidence"    : knn["confidence"],
            "confidence_pct": conf,
        },
        "window"   : window,
        "keputusan": keputusan,
        "ekspektasi": payload.ekspektasi,
        "benar"    : (label == payload.ekspektasi) if payload.ekspektasi else None,
        "features" : knn.get("features", {}),
        "catatan"  : "[SIMULASI] pompa tidak nyala",
    }

# ─────────────────────────────────────────────────────────────────────────────
# GET /test-knn/skenario
# ─────────────────────────────────────────────────────────────────────────────
SKENARIO_PRESET = [
    {"label": "S01 - Pagi kering RH normal",        "soil": 22, "temp": 29, "rh": 55, "hour": 6,  "soil_prev": 24, "ekspektasi": "Siram_Segera"},
    {"label": "S02 - Sore kering",                  "soil": 25, "temp": 31, "rh": 58, "hour": 17, "soil_prev": 27, "ekspektasi": "Siram_Segera"},
    {"label": "S03 - Pagi kering RH Ambon 92%",     "soil": 22, "temp": 29, "rh": 92, "hour": 6,  "soil_prev": 22, "ekspektasi": "Siram_Segera"},
    {"label": "S04 - DARURAT pagi panas 38C",       "soil": 14, "temp": 38, "rh": 35, "hour": 6,  "soil_prev": 18, "ekspektasi": "Siram_Prioritas"},
    {"label": "S05 - DARURAT sore panas",           "soil": 16, "temp": 36, "rh": 38, "hour": 17, "soil_prev": 20, "ekspektasi": "Siram_Prioritas"},
    {"label": "S06 - Siang kering luar jadwal",     "soil": 25, "temp": 33, "rh": 52, "hour": 13, "soil_prev": 27, "ekspektasi": "Siram_Nanti"},
    {"label": "S07 - Tengah malam kering",          "soil": 28, "temp": 26, "rh": 65, "hour": 2,  "soil_prev": 29, "ekspektasi": "Siram_Nanti"},
    {"label": "S08 - Malam kering",                 "soil": 18, "temp": 24, "rh": 60, "hour": 23, "soil_prev": 20, "ekspektasi": "Siram_Nanti"},
    {"label": "S09 - Tanah optimal pagi",           "soil": 55, "temp": 27, "rh": 65, "hour": 6,  "soil_prev": 55, "ekspektasi": "Optimal"},
    {"label": "S10 - Tanah optimal RH Ambon",       "soil": 60, "temp": 28, "rh": 94, "hour": 17, "soil_prev": 59, "ekspektasi": "Optimal"},
    {"label": "S11 - Tanah basah sore",             "soil": 83, "temp": 22, "rh": 88, "hour": 17, "soil_prev": 80, "ekspektasi": "Basah"},
    {"label": "S12 - Tanah sangat basah",           "soil": 88, "temp": 24, "rh": 85, "hour": 6,  "soil_prev": 82, "ekspektasi": "Basah"},
]

@app.get("/test-knn/skenario")
def get_skenario(x_api_key: str = Header(...)):
    check_api_key(x_api_key)

    hasil, benar_n = [], 0

    for s in SKENARIO_PRESET:
        knn    = run_knn(s["soil"], s["temp"], s["rh"], s["hour"], s["soil_prev"])
        label  = knn["label"]
        conf   = round(knn["confidence"] * 100, 1)
        window = get_window(s["hour"])
        cocok  = label == s["ekspektasi"]
        if cocok:
            benar_n += 1

        if window and label in LABEL_SIRAM:
            keputusan = f"SIRAM ({PUMP_DURATION_MINUTES} menit)"
        elif window:
            keputusan = "TIDAK SIRAM"
        else:
            keputusan = "DI LUAR JADWAL"

        hasil.append({
            "label_skenario": s["label"],
            "knn_label"     : label,
            "confidence_pct": conf,
            "ekspektasi"    : s["ekspektasi"],
            "benar"         : cocok,
            "window"        : window,
            "keputusan"     : keputusan,
        })

    return {
        "akurasi"  : f"{round(benar_n / len(SKENARIO_PRESET) * 100, 1)}%",
        "benar"    : benar_n,
        "total"    : len(SKENARIO_PRESET),
        "hasil"    : hasil,
    }

# ─────────────────────────────────────────────────────────────────────────────
# POST /test-knn/batch
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/test-knn/batch")
def test_knn_batch(payload: BatchPayload, x_api_key: str = Header(...)):
    check_api_key(x_api_key)

    hasil, benar_n = [], 0

    for s in payload.skenario:
        knn    = run_knn(s.soil_moisture, s.temperature, s.air_humidity, s.hour, s.soil_prev)
        label  = knn["label"]
        conf   = round(knn["confidence"] * 100, 1)
        window = get_window(s.hour)
        cocok  = (label == s.ekspektasi) if s.ekspektasi else None
        if cocok:
            benar_n += 1

        if window and label in LABEL_SIRAM:
            keputusan = f"SIRAM ({PUMP_DURATION_MINUTES} menit)"
        elif window:
            keputusan = "TIDAK SIRAM"
        else:
            keputusan = "DI LUAR JADWAL"

        hasil.append({
            "label_skenario": s.label_skenario,
            "knn_label"     : label,
            "confidence_pct": conf,
            "ekspektasi"    : s.ekspektasi,
            "benar"         : cocok,
            "window"        : window,
            "keputusan"     : keputusan,
        })

    with_eksp = [h for h in hasil if h["benar"] is not None]
    akurasi   = round(benar_n / len(with_eksp) * 100, 1) if with_eksp else None

    return {
        "akurasi": f"{akurasi}%" if akurasi else "N/A",
        "benar"  : benar_n,
        "total"  : len(payload.skenario),
        "hasil"  : hasil,
    }

# ─────────────────────────────────────────────────────────────────────────────
# POST /test-knn/reset
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/test-knn/reset")
def test_reset(x_api_key: str = Header(...)):
    check_api_key(x_api_key)

    state.pump_status       = False
    state.pump_start_ts     = None
    state.manual_override   = False
    state.last_watered_pagi = None
    state.last_watered_sore = None
    state.mode              = "auto"
    save_state()

    return {
        "message"     : "State berhasil direset",
        "pump_status" : state.pump_status,
        "mode"        : state.mode,
    }

# ─────────────────────────────────────────────────────────────────────────────
# POST /test-knn/fire  (pompa nyala sungguhan untuk testing)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/test-knn/fire")
def test_fire(payload: FirePayload, x_api_key: str = Header(...)):
    check_api_key(x_api_key)
    check_timeout()

    knn    = run_knn(payload.soil_moisture, payload.temperature,
                     payload.air_humidity, payload.hour, payload.soil_prev)
    label  = knn["label"]
    conf   = round(knn["confidence"] * 100, 1)
    window = get_window(payload.hour)
    today  = date.today()

    pump_action, reason, peringatan = None, "", None

    if window is None:
        peringatan = f"Jam {payload.hour:02d}:xx di luar window. Gunakan jam 5-6 atau 16-17."
        reason     = f"[FIRE] Di luar window | KNN: {label} ({conf}%)"

    elif label not in LABEL_SIRAM:
        peringatan = f"KNN: {label} ({conf}%) — tidak siram. Coba kondisi tanah lebih kering."
        reason     = f"[FIRE] KNN={label} tidak siram | window={window}"

    else:
        set_pump(True, f"[FIRE] KNN={label} ({conf}%) window={window}")
        mark_watered(window, today)
        pump_action = "on"
        reason      = f"[FIRE] KNN={label} ({conf}%) | window={window} | pompa ON {PUMP_DURATION_MINUTES} menit"

    # Simpan ke DB
    try:
        supabase.table("sensor_readings").insert({
            "soil_moisture" : payload.soil_moisture,
            "temperature"   : payload.temperature,
            "air_humidity"  : payload.air_humidity,
            "hour"          : payload.hour,
            "window"        : window,
            "knn_label"     : label,
            "knn_confidence": knn["confidence"],
            "pump_action"   : pump_action,
            "pump_status"   : state.pump_status,
            "mode"          : state.mode,
            "reason"        : reason,
            "recorded_at"   : datetime.utcnow().isoformat(),
        }).execute()
    except Exception as e:
        log.warning(f"[FIRE] Gagal simpan DB: {e}")

    state.last_knn_label = label
    state.last_knn_conf  = knn["confidence"]
    save_state()

    if pump_action == "on":
        keputusan = f"POMPA ON — KNN={label} ({conf}%) | {PUMP_DURATION_MINUTES} menit"
    elif window:
        keputusan = f"TIDAK SIRAM — KNN={label} ({conf}%) | window={window}"
    else:
        keputusan = f"DI LUAR JADWAL — jam {payload.hour:02d}:xx | KNN={label} ({conf}%)"

    return {
        "label_skenario"    : payload.label_skenario,
        "classification"    : {
            "label"         : label,
            "confidence"    : knn["confidence"],
            "confidence_pct": conf,
        },
        "window"            : window,
        "keputusan"         : keputusan,
        "pump_status"       : state.pump_status,
        "pump_action"       : pump_action,
        "pump_remaining_min": pump_remaining(),
        "ekspektasi"        : payload.ekspektasi,
        "benar"             : (label == payload.ekspektasi) if payload.ekspektasi else None,
        "peringatan"        : peringatan,
        "reason"            : reason,
        "features"          : knn.get("features", {}),
        "catatan"           : "[FIRE] Pompa nyala sungguhan jika syarat terpenuhi.",
    }