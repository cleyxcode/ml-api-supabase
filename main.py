"""
Siram Pintar API v12.0
======================

Logika AUTO (KNN memutuskan):
  - Hanya aktif di window pagi (05-07) dan sore (16-18)
  - Di dalam window → KNN baca data sensor → putuskan:
      Siram_Segera / Siram_Prioritas  → pompa ON 20 menit
      Siram_Nanti / Optimal / Basah /
      Hujan_Aktif / Hujan_Prediksi    → pompa TIDAK nyala
  - Setiap window hanya 1 sesi per hari (pagi 1x, sore 1x)
  - Pompa otomatis OFF setelah 20 menit

Logika MANUAL:
  - Pompa dikendalikan dari dashboard via POST /control
  - Tidak peduli jadwal maupun KNN

Endpoints:
  GET  /               → health check + versi
  GET  /status         → status sistem lengkap
  GET  /pump-status    → polling ringan dari ESP32
  POST /sensor         → terima data sensor dari ESP32
  POST /control        → kontrol manual dari dashboard
  GET  /history        → riwayat data sensor
  POST /test-knn       → uji KNN satu skenario (simulasi)
  GET  /test-knn/skenario → 12 skenario preset
  POST /test-knn/batch → uji banyak skenario sekaligus
  POST /test-knn/reset → reset state pompa & cooldown
"""

import os
import math
import logging
from datetime import datetime, date, timedelta, timezone
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
VERSION      = "12.0"
API_KEY      = os.getenv("API_KEY", "yuli1")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
MODEL_PATH   = os.getenv("MODEL_PATH", "knn_model.pkl")

# Jadwal siram (jam mulai inklusif, jam selesai eksklusif)
WINDOW_PAGI = (5, 7)    # 05:00 – 06:59
WINDOW_SORE = (16, 18)  # 16:00 – 17:59

# Durasi pompa menyala per sesi
PUMP_DURATION_MINUTES = 20

# Label KNN → pompa ON
LABEL_SIRAM = {"Siram_Segera", "Siram_Prioritas"}

# Label KNN → pompa TIDAK nyala
LABEL_SKIP = {"Siram_Nanti", "Optimal", "Basah", "Hujan_Aktif", "Hujan_Prediksi"}

# ─────────────────────────────────────────────────────────────────────────────
# Supabase
# ─────────────────────────────────────────────────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# KNN Model
# ─────────────────────────────────────────────────────────────────────────────
knn_model  = None
knn_scaler = None
model_info = {}

try:
    bundle     = joblib.load(MODEL_PATH)
    knn_model  = bundle.get("model")
    knn_scaler = bundle.get("scaler")
    model_info = {
        "algorithm" : "K-Nearest Neighbor",
        "best_k"    : getattr(knn_model, "n_neighbors", "?"),
        "features"  : ["soil_moisture","temperature","air_humidity",
                        "hour_sin","hour_cos","soil_trend",
                        "evapotranspiration","is_hot"],
    }
    log.info(f"[MODEL] KNN dimuat ✓  k={model_info['best_k']}")
except Exception as e:
    log.warning(f"[MODEL] Gagal muat model: {e} — KNN dinonaktifkan")

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
# Auth
# ─────────────────────────────────────────────────────────────────────────────
def check_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key salah")

# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────
class SensorPayload(BaseModel):
    soil_moisture : float = Field(..., ge=0,   le=100)
    temperature   : float = Field(..., ge=-10, le=60)
    air_humidity  : float = Field(..., ge=0,   le=100)
    hour          : Optional[int] = Field(None, ge=0, le=23)
    minute        : Optional[int] = Field(None, ge=0, le=59)
    day           : Optional[int] = Field(None, ge=0, le=6)

class ControlPayload(BaseModel):
    action : str   # "on" / "off"
    mode   : str   # "manual" / "auto"

class TestPayload(BaseModel):
    soil_moisture  : float
    temperature    : float
    air_humidity   : float
    hour           : int
    soil_prev      : Optional[float] = None
    temp_prev      : Optional[float] = None
    rh_prev        : Optional[float] = None
    label_skenario : Optional[str]   = None
    ekspektasi     : Optional[str]   = None

class BatchPayload(BaseModel):
    skenario: List[TestPayload]

# ─────────────────────────────────────────────────────────────────────────────
# State in-memory
# ─────────────────────────────────────────────────────────────────────────────
class SystemState:
    pump_status        : bool             = False
    mode               : str              = "auto"
    manual_override    : bool             = False
    pump_start_ts      : Optional[datetime] = None
    last_watered_pagi  : Optional[date]   = None
    last_watered_sore  : Optional[date]   = None
    last_soil          : float            = 0.0
    last_temp          : float            = 0.0
    last_rh            : float            = 0.0
    last_hour          : int              = 0
    last_knn_label     : str              = "---"
    last_knn_conf      : float            = 0.0

state = SystemState()


def load_state():
    try:
        res = supabase.table("system_state").select("*").eq("id", 1).single().execute()
        if res.data:
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

            log.info(f"[STATE] Dimuat: pump={state.pump_status} mode={state.mode}")
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
# Helper — KNN Prediction
# ─────────────────────────────────────────────────────────────────────────────
def run_knn(soil: float, temp: float, rh: float, hour: int,
            soil_prev: Optional[float] = None,
            temp_prev: Optional[float] = None,
            rh_prev:   Optional[float] = None) -> dict:
    """
    Jalankan prediksi KNN.
    Kembalikan dict: {label, confidence, features}
    Jika model tidak tersedia, kembalikan label default.
    """
    if knn_model is None or knn_scaler is None:
        return {"label": "Siram_Segera", "confidence": 0.0, "features": {}, "model_ready": False}

    # Fitur turunan
    hour_sin  = math.sin(2 * math.pi * hour / 24)
    hour_cos  = math.cos(2 * math.pi * hour / 24)
    soil_trend = (soil - soil_prev) if soil_prev is not None else 0.0

    # Evapotranspiration sederhana (Hargreaves approx)
    t_prev = temp_prev if temp_prev is not None else temp
    et = max(0.0, 0.0023 * (temp + 17.8) * abs(temp - t_prev) ** 0.5 * 0.408)

    is_hot = 1.0 if temp >= 35.0 else 0.0

    features = np.array([[
        soil, temp, rh,
        hour_sin, hour_cos,
        soil_trend, et, is_hot
    ]])

    features_scaled = knn_scaler.transform(features)
    label      = knn_model.predict(features_scaled)[0]
    proba      = knn_model.predict_proba(features_scaled)[0]
    confidence = float(np.max(proba))

    return {
        "label"      : str(label),
        "confidence" : round(confidence, 4),
        "features"   : {
            "soil_moisture"     : soil,
            "temperature"       : temp,
            "air_humidity"      : rh,
            "hour_sin"          : round(hour_sin, 4),
            "hour_cos"          : round(hour_cos, 4),
            "soil_trend"        : round(soil_trend, 2),
            "evapotranspiration": round(et, 4),
            "is_hot"            : is_hot,
        },
        "model_ready": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper — cek window & pompa
# ─────────────────────────────────────────────────────────────────────────────
def get_window_name(hour: int) -> Optional[str]:
    """Kembalikan 'pagi', 'sore', atau None jika di luar window."""
    if WINDOW_PAGI[0] <= hour < WINDOW_PAGI[1]:
        return "pagi"
    if WINDOW_SORE[0] <= hour < WINDOW_SORE[1]:
        return "sore"
    return None


def already_watered_today(window: str, today: date) -> bool:
    """Cek apakah window ini sudah disiram hari ini."""
    if window == "pagi":
        return state.last_watered_pagi == today
    if window == "sore":
        return state.last_watered_sore == today
    return False


def mark_watered(window: str, today: date):
    """Tandai window ini sudah disiram hari ini."""
    if window == "pagi":
        state.last_watered_pagi = today
    elif window == "sore":
        state.last_watered_sore = today


def set_pump(on: bool, reason: str = ""):
    """Nyalakan atau matikan pompa + catat waktu mulai."""
    if on == state.pump_status:
        return
    state.pump_status = on
    state.pump_start_ts = datetime.utcnow() if on else None
    log.info(f"[POMPA] {'ON' if on else 'OFF'} — {reason}")


def check_pump_timeout():
    """
    Matikan pompa otomatis jika sudah menyala >= PUMP_DURATION_MINUTES.
    Dipanggil setiap request /sensor dan /pump-status.
    """
    if not state.pump_status:
        return
    if state.pump_start_ts is None:
        return
    elapsed = (datetime.utcnow() - state.pump_start_ts).total_seconds() / 60
    if elapsed >= PUMP_DURATION_MINUTES:
        set_pump(False, f"Timeout {PUMP_DURATION_MINUTES} menit tercapai")
        save_state()
        log.info(f"[POMPA] OFF otomatis setelah {PUMP_DURATION_MINUTES} menit")


def pump_remaining_minutes() -> float:
    """Sisa waktu pompa menyala (menit). 0 jika pompa mati."""
    if not state.pump_status or state.pump_start_ts is None:
        return 0.0
    elapsed = (datetime.utcnow() - state.pump_start_ts).total_seconds() / 60
    return max(0.0, round(PUMP_DURATION_MINUTES - elapsed, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Helper — simpan log ke Supabase
# ─────────────────────────────────────────────────────────────────────────────
def log_sensor(payload: SensorPayload, hour: int, window: Optional[str],
               knn_result: dict, pump_action: Optional[str], reason: str):
    try:
        supabase.table("sensor_readings").insert({
            "soil_moisture"  : payload.soil_moisture,
            "temperature"    : payload.temperature,
            "air_humidity"   : payload.air_humidity,
            "hour"           : hour,
            "window"         : window,
            "knn_label"      : knn_result.get("label"),
            "knn_confidence" : knn_result.get("confidence"),
            "pump_action"    : pump_action,
            "pump_status"    : state.pump_status,
            "mode"           : state.mode,
            "reason"         : reason,
            "recorded_at"    : datetime.utcnow().isoformat(),
        }).execute()
    except Exception as e:
        log.warning(f"[DB] Gagal simpan sensor log: {e}")


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
            "auto_window_pagi" : f"{WINDOW_PAGI[0]:02d}:00 – {WINDOW_PAGI[1]:02d}:00",
            "auto_window_sore" : f"{WINDOW_SORE[0]:02d}:00 – {WINDOW_SORE[1]:02d}:00",
            "pump_duration"    : f"{PUMP_DURATION_MINUTES} menit per sesi",
            "knn_siram"        : list(LABEL_SIRAM),
            "knn_skip"         : list(LABEL_SKIP),
        },
        "model_info": model_info,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /status
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/status")
def get_status(x_api_key: str = Header(...)):
    check_api_key(x_api_key)
    check_pump_timeout()
    return {
        "pump_status"         : state.pump_status,
        "pump_remaining_min"  : pump_remaining_minutes(),
        "mode"                : state.mode,
        "manual_override"     : state.manual_override,
        "last_knn_label"      : state.last_knn_label,
        "last_knn_confidence" : state.last_knn_conf,
        "last_soil"           : state.last_soil,
        "last_watered_pagi"   : state.last_watered_pagi.isoformat() if state.last_watered_pagi else None,
        "last_watered_sore"   : state.last_watered_sore.isoformat() if state.last_watered_sore else None,
        "windows": {
            "pagi": f"{WINDOW_PAGI[0]:02d}:00 – {WINDOW_PAGI[1]:02d}:00",
            "sore": f"{WINDOW_SORE[0]:02d}:00 – {WINDOW_SORE[1]:02d}:00",
        },
        "pump_duration_minutes": PUMP_DURATION_MINUTES,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /pump-status  (polling ringan dari ESP32)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/pump-status")
def get_pump_status(x_api_key: str = Header(...)):
    check_api_key(x_api_key)
    check_pump_timeout()
    return {
        "pump_status"        : state.pump_status,
        "mode"               : state.mode,
        "manual_override"    : state.manual_override,
        "pump_remaining_min" : pump_remaining_minutes(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /sensor  (inti — terima data ESP32 dan putuskan pompa)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/sensor")
def post_sensor(payload: SensorPayload, x_api_key: str = Header(...)):
    check_api_key(x_api_key)

    # 1. Cek timeout pompa lebih dulu
    check_pump_timeout()

    # 2. Waktu dari ESP32
    hour   = payload.hour   if payload.hour   is not None else datetime.utcnow().hour
    today  = date.today()

    # Simpan data sensor terakhir
    soil_prev       = state.last_soil
    temp_prev       = state.last_temp
    rh_prev         = state.last_rh
    state.last_soil = payload.soil_moisture
    state.last_temp = payload.temperature
    state.last_rh   = payload.air_humidity
    state.last_hour = hour

    # 3. Jalankan KNN (selalu, untuk dokumentasi)
    knn_result = run_knn(
        soil      = payload.soil_moisture,
        temp      = payload.temperature,
        rh        = payload.air_humidity,
        hour      = hour,
        soil_prev = soil_prev if soil_prev > 0 else None,
        temp_prev = temp_prev if temp_prev > 0 else None,
        rh_prev   = rh_prev   if rh_prev   > 0 else None,
    )
    state.last_knn_label = knn_result["label"]
    state.last_knn_conf  = knn_result["confidence"]

    pump_action = None
    reason      = ""

    # ── Mode MANUAL ────────────────────────────────────────────────────────
    if state.mode == "manual":
        reason = f"Mode MANUAL — pompa dikendalikan dashboard | KNN: {knn_result['label']}"
        log_sensor(payload, hour, None, knn_result, pump_action, reason)
        save_state()
        return _build_response(knn_result, pump_action, reason)

    # ── Mode AUTO ──────────────────────────────────────────────────────────
    window = get_window_name(hour)

    if window is None:
        # Di luar jadwal — pompa tidak disentuh, hanya log
        reason = f"Di luar window pagi/sore (jam {hour:02d}:xx) | KNN: {knn_result['label']}"
        log_sensor(payload, hour, window, knn_result, pump_action, reason)
        save_state()
        return _build_response(knn_result, pump_action, reason)

    # Dalam window pagi atau sore
    if already_watered_today(window, today):
        # Sudah siram di window ini hari ini
        reason = f"Sudah siram {window} hari ini | KNN: {knn_result['label']}"
        log_sensor(payload, hour, window, knn_result, pump_action, reason)
        save_state()
        return _build_response(knn_result, pump_action, reason)

    if state.pump_status:
        # Pompa sedang ON (sesi sedang berjalan)
        sisa = pump_remaining_minutes()
        reason = f"Pompa sedang ON — sisa {sisa} menit | KNN: {knn_result['label']}"
        log_sensor(payload, hour, window, knn_result, pump_action, reason)
        save_state()
        return _build_response(knn_result, pump_action, reason)

    # ── KNN memutuskan ─────────────────────────────────────────────────────
    label = knn_result["label"]
    conf  = round(knn_result["confidence"] * 100, 1)

    if label in LABEL_SIRAM:
        # KNN: perlu siram → pompa ON
        set_pump(True, f"KNN={label} ({conf}%) window={window}")
        mark_watered(window, today)
        pump_action = "on"
        reason = (f"KNN memutuskan SIRAM — label: {label} ({conf}%) "
                  f"| window: {window} | pompa ON {PUMP_DURATION_MINUTES} menit")
        log.info(f"[AUTO] {reason}")

    else:
        # KNN: tidak perlu siram → pompa tetap OFF
        pump_action = None
        reason = (f"KNN memutuskan TIDAK SIRAM — label: {label} ({conf}%) "
                  f"| window: {window} | pompa tetap OFF")
        log.info(f"[AUTO] {reason}")

    log_sensor(payload, hour, window, knn_result, pump_action, reason)
    save_state()
    return _build_response(knn_result, pump_action, reason)


def _build_response(knn_result: dict, pump_action: Optional[str], reason: str) -> dict:
    return {
        "pump_status"  : state.pump_status,
        "pump_action"  : pump_action,
        "mode"         : state.mode,
        "classification": {
            "label"     : knn_result["label"],
            "confidence": knn_result["confidence"],
        },
        "auto_info": {
            "reason"          : reason,
            "pump_remaining_min": pump_remaining_minutes(),
            "manual_override" : state.manual_override,
        },
        "features": knn_result.get("features", {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /control  (manual dari dashboard)
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
        target = payload.action == "on"
        set_pump(target, f"MANUAL dari dashboard — {payload.action}")
    else:
        # Kembali ke auto — matikan override, matikan pompa
        state.manual_override = False
        set_pump(False, "Kembali ke mode AUTO")

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
    x_api_key  : str = Header(...),
    limit      : int  = Query(20, ge=1, le=100),
    pump_only  : bool = Query(False),
):
    check_api_key(x_api_key)
    try:
        q = supabase.table("sensor_readings").select("*").order("recorded_at", desc=True).limit(limit)
        if pump_only:
            q = q.eq("pump_action", "on")
        res = q.execute()
        return {"count": len(res.data), "data": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# POST /test-knn  (simulasi — pompa tidak nyala sungguhan)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/test-knn")
def test_knn(payload: TestPayload, x_api_key: str = Header(...)):
    check_api_key(x_api_key)

    knn_result = run_knn(
        soil      = payload.soil_moisture,
        temp      = payload.temperature,
        rh        = payload.air_humidity,
        hour      = payload.hour,
        soil_prev = payload.soil_prev,
        temp_prev = payload.temp_prev,
        rh_prev   = payload.rh_prev,
    )

    window = get_window_name(payload.hour)
    label  = knn_result["label"]
    conf   = round(knn_result["confidence"] * 100, 1)

    if window:
        if label in LABEL_SIRAM:
            keputusan = f"✅ SIRAM — {label} ({conf}%) | window {window} | pompa ON {PUMP_DURATION_MINUTES} menit"
        else:
            keputusan = f"⛔ TIDAK SIRAM — {label} ({conf}%) | window {window}"
    else:
        keputusan = f"⏰ DI LUAR JADWAL (jam {payload.hour:02d}:xx) — {label} ({conf}%) | pompa tidak nyala"

    benar = None
    if payload.ekspektasi:
        benar = (label == payload.ekspektasi)

    return {
        "label_skenario": payload.label_skenario,
        "classification": {
            "label"     : label,
            "confidence": knn_result["confidence"],
            "confidence_pct": conf,
        },
        "window"    : window,
        "keputusan" : keputusan,
        "ekspektasi": payload.ekspektasi,
        "benar"     : benar,
        "features"  : knn_result.get("features", {}),
        "catatan"   : "[SIMULASI] pompa tidak nyala sungguhan",
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /test-knn/skenario  (12 skenario preset)
# ─────────────────────────────────────────────────────────────────────────────
SKENARIO_PRESET = [
    {"label":"S01 - Pagi kering",          "soil":22,"temp":29,"rh":55,"hour":6,  "soil_prev":24,"temp_prev":29,"rh_prev":54, "ekspektasi":"Siram_Segera"},
    {"label":"S02 - Sore kering",           "soil":25,"temp":31,"rh":58,"hour":17, "soil_prev":27,"temp_prev":32,"rh_prev":57, "ekspektasi":"Siram_Segera"},
    {"label":"S03 - Darurat panas ekstrem", "soil":14,"temp":38,"rh":35,"hour":6,  "soil_prev":18,"temp_prev":39,"rh_prev":33, "ekspektasi":"Siram_Prioritas"},
    {"label":"S04 - Siang luar jadwal",     "soil":25,"temp":33,"rh":52,"hour":13, "soil_prev":27,"temp_prev":33,"rh_prev":51, "ekspektasi":"Siram_Nanti"},
    {"label":"S05 - Tengah malam",          "soil":28,"temp":26,"rh":65,"hour":2,  "soil_prev":29,"temp_prev":27,"rh_prev":64, "ekspektasi":"Siram_Nanti"},
    {"label":"S06 - Tanah optimal",         "soil":55,"temp":27,"rh":65,"hour":6,  "soil_prev":55,"temp_prev":27,"rh_prev":65, "ekspektasi":"Optimal"},
    {"label":"S07 - Tanah basah",           "soil":83,"temp":22,"rh":88,"hour":17, "soil_prev":80,"temp_prev":22,"rh_prev":87, "ekspektasi":"Basah"},
    {"label":"S08 - Hujan deras aktif",     "soil":25,"temp":22,"rh":98,"hour":6,  "soil_prev":14,"temp_prev":30,"rh_prev":68, "ekspektasi":"Hujan_Aktif"},
    {"label":"S09 - Prediksi akan hujan",   "soil":35,"temp":24,"rh":88,"hour":17, "soil_prev":29,"temp_prev":30,"rh_prev":63, "ekspektasi":"Hujan_Prediksi"},
    {"label":"S10 - Sore panas ET tinggi",  "soil":22,"temp":36,"rh":38,"hour":17, "soil_prev":27,"temp_prev":37,"rh_prev":36, "ekspektasi":"Siram_Prioritas"},
    {"label":"S11 - Malam kering",          "soil":18,"temp":24,"rh":60,"hour":23, "soil_prev":20,"temp_prev":25,"rh_prev":59, "ekspektasi":"Siram_Nanti"},
    {"label":"S12 - RH tinggi soil kering", "soil":22,"temp":27,"rh":95,"hour":6,  "soil_prev":23,"temp_prev":28,"rh_prev":93, "ekspektasi":"Siram_Segera"},
]

@app.get("/test-knn/skenario")
def get_skenario(x_api_key: str = Header(...)):
    check_api_key(x_api_key)

    hasil   = []
    benar_n = 0

    for s in SKENARIO_PRESET:
        knn = run_knn(s["soil"], s["temp"], s["rh"], s["hour"],
                      s["soil_prev"], s["temp_prev"], s["rh_prev"])
        label  = knn["label"]
        conf   = round(knn["confidence"] * 100, 1)
        window = get_window_name(s["hour"])
        cocok  = (label == s["ekspektasi"])
        if cocok: benar_n += 1

        if window and label in LABEL_SIRAM:
            keputusan = f"✅ SIRAM ({PUMP_DURATION_MINUTES} menit)"
        elif window:
            keputusan = "⛔ TIDAK SIRAM"
        else:
            keputusan = "⏰ DI LUAR JADWAL"

        hasil.append({
            "label_skenario": s["label"],
            "knn_label"     : label,
            "confidence_pct": conf,
            "ekspektasi"    : s["ekspektasi"],
            "benar"         : cocok,
            "window"        : window,
            "keputusan"     : keputusan,
        })

    akurasi = round(benar_n / len(SKENARIO_PRESET) * 100, 1)
    return {
        "akurasi_preset" : f"{akurasi}%",
        "benar"          : benar_n,
        "total"          : len(SKENARIO_PRESET),
        "hasil"          : hasil,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /test-knn/batch
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/test-knn/batch")
def test_knn_batch(payload: BatchPayload, x_api_key: str = Header(...)):
    check_api_key(x_api_key)

    hasil   = []
    benar_n = 0
    total   = len(payload.skenario)

    for s in payload.skenario:
        knn    = run_knn(s.soil_moisture, s.temperature, s.air_humidity, s.hour,
                         s.soil_prev, s.temp_prev, s.rh_prev)
        label  = knn["label"]
        conf   = round(knn["confidence"] * 100, 1)
        window = get_window_name(s.hour)
        cocok  = (label == s.ekspektasi) if s.ekspektasi else None
        if cocok: benar_n += 1

        if window and label in LABEL_SIRAM:
            keputusan = f"✅ SIRAM ({PUMP_DURATION_MINUTES} menit)"
        elif window:
            keputusan = "⛔ TIDAK SIRAM"
        else:
            keputusan = "⏰ DI LUAR JADWAL"

        hasil.append({
            "label_skenario": s.label_skenario,
            "knn_label"     : label,
            "confidence_pct": conf,
            "ekspektasi"    : s.ekspektasi,
            "benar"         : cocok,
            "window"        : window,
            "keputusan"     : keputusan,
        })

    with_ekspektasi = [h for h in hasil if h["benar"] is not None]
    akurasi = round(benar_n / len(with_ekspektasi) * 100, 1) if with_ekspektasi else None

    return {
        "akurasi_batch": f"{akurasi}%" if akurasi is not None else "N/A",
        "benar"        : benar_n,
        "total"        : total,
        "hasil"        : hasil,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /test-knn/reset
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/test-knn/reset")
def test_reset(x_api_key: str = Header(...)):
    check_api_key(x_api_key)

    state.pump_status      = False
    state.pump_start_ts    = None
    state.manual_override  = False
    state.last_watered_pagi = None
    state.last_watered_sore = None
    state.mode             = "auto"

    # Matikan relay via ESP32 (dilakukan lewat pump_status=False pada poll berikutnya)
    save_state()

    return {
        "message"      : "State berhasil direset",
        "pump_status"  : state.pump_status,
        "mode"         : state.mode,
        "last_watered_pagi": None,
        "last_watered_sore": None,
    }