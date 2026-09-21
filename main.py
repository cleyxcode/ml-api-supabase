"""
Siram Pintar API v13.4
======================
Perubahan dari v13.3:
  - Fix: setelah test done → mode MANUAL, pompa OFF, tidak bolak-balik
  - Fix: guard tambahan cegah test restart saat transisi state
  - Fix: /control reset test_status saat masuk mode baru
"""

import os
import math
import uuid
import logging
from datetime import datetime, date
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

VERSION      = "13.4"
API_KEY      = os.getenv("API_KEY",      "yuli1")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
MODEL_PATH   = os.getenv("MODEL_PATH",   "model/knn_model.pkl")
SCALER_PATH  = os.getenv("SCALER_PATH",  "model/scaler.pkl")

WINDOW_PAGI           = (5, 7)
WINDOW_SORE           = (16, 18)
PUMP_DURATION_MINUTES = 20
TEST_PUMP_SECONDS     = 30
IS_HOT_THRESHOLD      = 34.0

LABEL_SIRAM = {"Siram_Segera", "Siram_Prioritas"}
LABEL_SKIP  = {"Siram_Nanti", "Optimal", "Basah"}

TEST_DRY_MAX = 35.0
TEST_WET_MIN = 65.0

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

knn_model  = None
knn_scaler = None
model_info = {}

try:
    knn_model  = joblib.load(MODEL_PATH)
    knn_scaler = joblib.load(SCALER_PATH)
    model_info = {
        "algorithm": "K-Nearest Neighbor",
        "best_k"   : getattr(knn_model, "n_neighbors", "?"),
        "version"  : VERSION,
    }
    log.info(f"[MODEL] KNN dimuat — k={model_info['best_k']}")
except Exception as e:
    log.warning(f"[MODEL] Gagal muat model: {e}")

app = FastAPI(title="Siram Pintar API", version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SensorPayload(BaseModel):
    soil_moisture : float = Field(..., ge=0,   le=100)
    temperature   : float = Field(..., ge=-10, le=60)
    air_humidity  : float = Field(..., ge=0,   le=100)
    hour          : Optional[int] = Field(None, ge=0, le=23)
    minute        : Optional[int] = Field(None, ge=0, le=59)

class ControlPayload(BaseModel):
    action : str
    mode   : str

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
    hour           : int = Field(..., ge=0, le=23)
    soil_prev      : Optional[float] = None
    label_skenario : Optional[str]   = None
    ekspektasi     : Optional[str]   = None

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
    test_active       : bool               = False
    test_status       : str                = ""
    test_pump_start   : Optional[datetime] = None
    test_result       : str                = ""

state = SystemState()

_col_cache: dict = {}

def _col_exists(table: str, col: str) -> bool:
    key = f"{table}.{col}"
    if key in _col_cache:
        return _col_cache[key]
    try:
        supabase.table(table).select(col).limit(1).execute()
        _col_cache[key] = True
    except Exception:
        _col_cache[key] = False
    return _col_cache[key]

def load_state():
    try:
        res = supabase.table("system_state").select("*").eq("id", 1).single().execute()
        if not res.data:
            return
        d = res.data
        state.pump_status     = d.get("pump_status", False)
        state.mode            = d.get("mode", "auto")
        state.manual_override = d.get("manual_override", False)
        state.last_soil       = d.get("last_soil_moisture") or 0.0
        state.last_knn_label  = d.get("last_label") or "---"
        state.last_knn_conf   = d.get("last_knn_conf") or 0.0
        pts = d.get("pump_start_ts")
        state.pump_start_ts   = datetime.fromisoformat(pts) if pts else None
        lp = d.get("last_watered_pagi")
        ls = d.get("last_watered_sore")
        state.last_watered_pagi = date.fromisoformat(lp) if lp else None
        state.last_watered_sore = date.fromisoformat(ls) if ls else None
        # Reset test state saat startup
        state.test_active     = False
        state.test_status     = ""
        state.test_result     = ""
        state.test_pump_start = None
        if state.mode == "test":
            state.mode = "auto"
        log.info(f"[STATE] Dimuat — pump={state.pump_status} mode={state.mode}")
    except Exception as e:
        log.warning(f"[STATE] Gagal muat: {e}")

def save_state():
    try:
        payload_db: dict = {
            "id"                : 1,
            "pump_status"       : state.pump_status,
            "mode"              : state.mode,
            "manual_override"   : state.manual_override,
            "last_soil_moisture": state.last_soil,
            "last_label"        : state.last_knn_label,
            "pump_start_ts"     : state.pump_start_ts.isoformat() if state.pump_start_ts else None,
            "last_updated"      : datetime.utcnow().isoformat(),
            "last_watered_ts"   : _latest_watered_ts(),
        }
        if _col_exists("system_state", "last_knn_conf"):
            payload_db["last_knn_conf"] = state.last_knn_conf
        if _col_exists("system_state", "last_watered_pagi"):
            payload_db["last_watered_pagi"] = (
                state.last_watered_pagi.isoformat() if state.last_watered_pagi else None
            )
        if _col_exists("system_state", "last_watered_sore"):
            payload_db["last_watered_sore"] = (
                state.last_watered_sore.isoformat() if state.last_watered_sore else None
            )
        supabase.table("system_state").upsert(payload_db).execute()
    except Exception as e:
        log.warning(f"[STATE] Gagal simpan: {e}")

def _latest_watered_ts() -> Optional[str]:
    candidates = []
    if state.last_watered_pagi:
        candidates.append(datetime(
            state.last_watered_pagi.year,
            state.last_watered_pagi.month,
            state.last_watered_pagi.day, 6, 0, 0
        ))
    if state.last_watered_sore:
        candidates.append(datetime(
            state.last_watered_sore.year,
            state.last_watered_sore.month,
            state.last_watered_sore.day, 17, 0, 0
        ))
    return max(candidates).isoformat() if candidates else None

@app.on_event("startup")
def on_startup():
    load_state()

def check_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key salah")

def calc_et(temp: float, rh: float) -> float:
    vpd = (1 - rh / 100) * 0.6108 * math.exp(17.27 * temp / (temp + 237.3))
    return round(min(max(vpd * 15, 0), 100), 2)

def run_knn(soil, temp, rh, hour, soil_prev=None) -> dict:
    if knn_model is None or knn_scaler is None:
        return {"label": "Siram_Segera", "confidence": 0.0, "features": {}, "model_ready": False}
    hour_sin   = math.sin(2 * math.pi * hour / 24)
    hour_cos   = math.cos(2 * math.pi * hour / 24)
    soil_trend = (soil - soil_prev) if soil_prev is not None else 0.0
    et         = calc_et(temp, rh)
    is_hot     = 1.0 if temp >= IS_HOT_THRESHOLD else 0.0
    X          = np.array([[soil, temp, rh, hour_sin, hour_cos, soil_trend, et, is_hot]])
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

def get_window(hour: int) -> Optional[str]:
    if WINDOW_PAGI[0] <= hour < WINDOW_PAGI[1]: return "pagi"
    if WINDOW_SORE[0] <= hour < WINDOW_SORE[1]: return "sore"
    return None

def already_watered(window: str, today: date) -> bool:
    if window == "pagi": return state.last_watered_pagi == today
    if window == "sore": return state.last_watered_sore == today
    return False

def mark_watered(window: str, today: date):
    if window == "pagi": state.last_watered_pagi = today
    elif window == "sore": state.last_watered_sore = today

def set_pump(on: bool, reason: str = ""):
    if on == state.pump_status: return
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

def check_test_timeout():
    if not state.test_active: return
    if not state.pump_status: return
    if state.test_pump_start is None: return
    elapsed = (datetime.utcnow() - state.test_pump_start).total_seconds()
    if elapsed >= TEST_PUMP_SECONDS:
        set_pump(False, "TEST selesai 30 detik")
        state.test_active     = False
        state.test_status     = "done"
        state.test_result     = "Pompa ON 30 detik selesai → pindah MANUAL"
        state.mode            = "manual"
        state.manual_override = True
        state.test_pump_start = None
        save_state()
        log.info("[TEST] Selesai 30 detik → mode MANUAL, pompa OFF")

def pump_remaining() -> float:
    if not state.pump_status or state.pump_start_ts is None: return 0.0
    elapsed = (datetime.utcnow() - state.pump_start_ts).total_seconds() / 60
    return max(0.0, round(PUMP_DURATION_MINUTES - elapsed, 1))

def test_pump_remaining() -> float:
    if not state.test_active or state.test_pump_start is None: return 0.0
    elapsed = (datetime.utcnow() - state.test_pump_start).total_seconds()
    return max(0.0, round(TEST_PUMP_SECONDS - elapsed, 1))

def log_to_db(payload, hour, window, knn, pump_action, reason):
    try:
        desc_parts = [reason]
        if window: desc_parts.append(f"window={window}")
        desc_parts.append(f"hour={hour:02d}:xx")
        record = {
            "id"            : str(uuid.uuid4()),
            "timestamp"     : datetime.utcnow().isoformat(),
            "soil_moisture" : payload.soil_moisture,
            "temperature"   : payload.temperature,
            "air_humidity"  : payload.air_humidity,
            "label"         : knn.get("label", "---"),
            "confidence"    : knn.get("confidence", 0.0),
            "needs_watering": pump_action == "on",
            "description"   : " | ".join(desc_parts),
            "pump_status"   : state.pump_status,
            "mode"          : state.mode,
            "probabilities" : knn.get("features", {}),
        }
        supabase.table("sensor_readings").insert(record).execute()
    except Exception as e:
        log.warning(f"[DB] Gagal simpan sensor: {e}")

def build_response(knn, pump_action, reason) -> dict:
    return {
        "pump_status"   : state.pump_status,
        "pump_action"   : pump_action,
        "mode"          : state.mode,
        "classification": {"label": knn["label"], "confidence": knn["confidence"]},
        "auto_info"     : {
            "reason"            : reason,
            "pump_remaining_min": pump_remaining(),
            "manual_override"   : state.manual_override,
        },
        "test_info": {
            "active"            : state.test_active,
            "status"            : state.test_status,
            "result"            : state.test_result,
            "pump_remaining_sec": test_pump_remaining(),
        },
        "features": knn.get("features", {}),
    }

@app.get("/")
def health_check():
    return {
        "status"     : "online",
        "version"    : VERSION,
        "model_ready": knn_model is not None,
        "pump_status": state.pump_status,
        "mode"       : state.mode,
        "test_info"  : {
            "active"            : state.test_active,
            "status"            : state.test_status,
            "result"            : state.test_result,
            "pump_remaining_sec": test_pump_remaining(),
        },
    }

@app.get("/status")
def get_status(x_api_key: str = Header(...)):
    check_api_key(x_api_key)
    check_timeout()
    check_test_timeout()
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
        "test_info": {
            "active"            : state.test_active,
            "status"            : state.test_status,
            "result"            : state.test_result,
            "pump_remaining_sec": test_pump_remaining(),
        },
    }

@app.get("/pump-status")
def get_pump_status(x_api_key: str = Header(...)):
    check_api_key(x_api_key)
    check_timeout()
    check_test_timeout()
    return {
        "pump_status"       : state.pump_status,
        "mode"              : state.mode,
        "manual_override"   : state.manual_override,
        "pump_remaining_min": pump_remaining(),
        "test_info": {
            "active"            : state.test_active,
            "status"            : state.test_status,
            "result"            : state.test_result,
            "pump_remaining_sec": test_pump_remaining(),
        },
    }

@app.post("/sensor")
def post_sensor(payload: SensorPayload, x_api_key: str = Header(...)):
    check_api_key(x_api_key)
    check_timeout()
    check_test_timeout()

    hour      = payload.hour if payload.hour is not None else datetime.utcnow().hour
    today     = date.today()
    soil_prev = state.last_soil

    state.last_soil = payload.soil_moisture
    state.last_temp = payload.temperature
    state.last_rh   = payload.air_humidity
    state.last_hour = hour

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

    # ── MODE TEST ────────────────────────────────────────────────────────────
    if state.mode == "test":
        soil = payload.soil_moisture

        # Guard 1: pompa sedang ON → tunggu timeout, jangan restart
        if state.test_active and state.pump_status:
            reason = f"[TEST] Pompa ON — sisa {test_pump_remaining():.0f} detik"
            log_to_db(payload, hour, None, knn, "on", reason)
            save_state()
            return build_response(knn, "on", reason)

        # Guard 2: test sudah selesai → jangan mulai lagi
        if state.test_status in ("done", "lembab", "basah"):
            reason = f"[TEST] Sudah selesai ({state.test_status}), menunggu mode berganti"
            log_to_db(payload, hour, None, knn, None, reason)
            save_state()
            return build_response(knn, None, reason)

        if soil < TEST_DRY_MAX:
            state.test_active     = True
            state.test_status     = "pumping"
            state.test_result     = f"Tanah kering ({soil:.1f}%) → pompa ON 30 detik"
            state.test_pump_start = datetime.utcnow()
            set_pump(True, f"[TEST] Kering {soil:.1f}%")
            pump_action = "on"
            reason = f"[TEST] Kering {soil:.1f}% < {TEST_DRY_MAX}% → pompa ON 30 detik → akan pindah MANUAL"

        elif soil <= TEST_WET_MIN:
            state.test_active     = False
            state.test_status     = "lembab"
            state.test_result     = f"Tanah cukup lembab ({soil:.1f}%) → tidak perlu siram"
            state.mode            = "auto"
            state.manual_override = False
            set_pump(False, f"[TEST] Lembab {soil:.1f}%")
            reason = f"[TEST] Lembab {soil:.1f}% ({TEST_DRY_MAX}-{TEST_WET_MIN}%) → cukup → kembali AUTO"

        else:
            state.test_active     = False
            state.test_status     = "basah"
            state.test_result     = f"Tanah basah ({soil:.1f}%) → test selesai"
            state.mode            = "auto"
            state.manual_override = False
            set_pump(False, f"[TEST] Basah {soil:.1f}%")
            reason = f"[TEST] Basah {soil:.1f}% > {TEST_WET_MIN}% → selesai → kembali AUTO"

        log_to_db(payload, hour, None, knn, pump_action, reason)
        save_state()
        return build_response(knn, pump_action, reason)

    # ── MODE MANUAL ──────────────────────────────────────────────────────────
    if state.mode == "manual":
        # Pompa di mode manual HANYA berubah dari /control, bukan dari sensor
        reason = f"Mode MANUAL | KNN: {knn['label']}"
        log_to_db(payload, hour, None, knn, pump_action, reason)
        save_state()
        return build_response(knn, pump_action, reason)

    # ── MODE AUTO ────────────────────────────────────────────────────────────
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

@app.post("/control")
def post_control(payload: ControlPayload, x_api_key: str = Header(...)):
    check_api_key(x_api_key)

    if payload.action not in ("on", "off"):
        raise HTTPException(status_code=400, detail="action harus 'on' atau 'off'")
    if payload.mode not in ("manual", "auto", "test"):
        raise HTTPException(status_code=400, detail="mode harus 'manual', 'auto', atau 'test'")

    state.mode = payload.mode

    if payload.mode == "test":
        state.test_active     = True
        state.test_status     = "waiting"
        state.test_result     = "Menunggu data sensor..."
        state.test_pump_start = None
        state.manual_override = False
        set_pump(False, "Masuk mode TEST")
        log.info("[TEST] Mode TEST diaktifkan dari dashboard")

    elif payload.mode == "manual":
        state.manual_override = True
        state.test_active     = False
        state.test_status     = ""
        state.test_result     = ""
        state.test_pump_start = None
        set_pump(payload.action == "on", f"MANUAL — {payload.action}")

    else:  # auto
        state.manual_override = False
        state.test_active     = False
        state.test_status     = ""
        state.test_result     = ""
        state.test_pump_start = None
        set_pump(False, "Kembali ke AUTO")

    save_state()
    return {
        "pump_status"    : state.pump_status,
        "mode"           : state.mode,
        "manual_override": state.manual_override,
        "message"        : f"Mode → {state.mode} | Pompa {'ON' if state.pump_status else 'OFF'}",
        "test_info": {
            "active": state.test_active,
            "status": state.test_status,
            "result": state.test_result,
        },
    }

@app.get("/history")
def get_history(
    x_api_key : str  = Header(...),
    limit     : int  = Query(20, ge=1, le=100),
    pump_only : bool = Query(False),
    mode      : Optional[str] = Query(None),
):
    check_api_key(x_api_key)
    try:
        q = (
            supabase.table("sensor_readings")
            .select("*")
            .order("timestamp", desc=True)
            .limit(limit)
        )
        if pump_only:
            q = q.eq("needs_watering", True)
        if mode in ("auto", "manual", "test"):
            q = q.eq("mode", mode)
        res = q.execute()
        return {"count": len(res.data), "data": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-knn")
def test_knn(payload: TestPayload, x_api_key: str = Header(...)):
    check_api_key(x_api_key)
    knn    = run_knn(payload.soil_moisture, payload.temperature,
                     payload.air_humidity, payload.hour, payload.soil_prev)
    label  = knn["label"]
    conf   = round(knn["confidence"] * 100, 1)
    window = get_window(payload.hour)
    if window and label in LABEL_SIRAM:
        keputusan = f"SIRAM — {label} ({conf}%) | window {window} | pompa ON {PUMP_DURATION_MINUTES} menit"
    elif window:
        keputusan = f"TIDAK SIRAM — {label} ({conf}%) | window {window}"
    else:
        keputusan = f"DI LUAR JADWAL (jam {payload.hour:02d}:xx) — {label} ({conf}%)"
    return {
        "label_skenario": payload.label_skenario,
        "classification": {"label": label, "confidence": knn["confidence"], "confidence_pct": conf},
        "window"    : window,
        "keputusan" : keputusan,
        "ekspektasi": payload.ekspektasi,
        "benar"     : (label == payload.ekspektasi) if payload.ekspektasi else None,
        "features"  : knn.get("features", {}),
        "catatan"   : "[SIMULASI] pompa tidak nyala",
    }

SKENARIO_PRESET = [
    {"label": "S01 - Pagi kering RH normal",    "soil": 22, "temp": 29, "rh": 55, "hour": 6,  "soil_prev": 24, "ekspektasi": "Siram_Segera"},
    {"label": "S02 - Sore kering",              "soil": 25, "temp": 31, "rh": 58, "hour": 17, "soil_prev": 27, "ekspektasi": "Siram_Segera"},
    {"label": "S03 - Pagi kering RH Ambon 92%", "soil": 22, "temp": 29, "rh": 92, "hour": 6,  "soil_prev": 22, "ekspektasi": "Siram_Segera"},
    {"label": "S04 - DARURAT pagi panas 38C",   "soil": 14, "temp": 38, "rh": 35, "hour": 6,  "soil_prev": 18, "ekspektasi": "Siram_Prioritas"},
    {"label": "S05 - DARURAT sore panas",       "soil": 16, "temp": 36, "rh": 38, "hour": 17, "soil_prev": 20, "ekspektasi": "Siram_Prioritas"},
    {"label": "S06 - Siang kering luar jadwal", "soil": 25, "temp": 33, "rh": 52, "hour": 13, "soil_prev": 27, "ekspektasi": "Siram_Nanti"},
    {"label": "S07 - Tengah malam kering",      "soil": 28, "temp": 26, "rh": 65, "hour": 2,  "soil_prev": 29, "ekspektasi": "Siram_Nanti"},
    {"label": "S08 - Malam kering",             "soil": 18, "temp": 24, "rh": 60, "hour": 23, "soil_prev": 20, "ekspektasi": "Siram_Nanti"},
    {"label": "S09 - Tanah optimal pagi",       "soil": 55, "temp": 27, "rh": 65, "hour": 6,  "soil_prev": 55, "ekspektasi": "Optimal"},
    {"label": "S10 - Tanah optimal RH Ambon",   "soil": 60, "temp": 28, "rh": 94, "hour": 17, "soil_prev": 59, "ekspektasi": "Optimal"},
    {"label": "S11 - Tanah basah sore",         "soil": 83, "temp": 22, "rh": 88, "hour": 17, "soil_prev": 80, "ekspektasi": "Basah"},
    {"label": "S12 - Tanah sangat basah",       "soil": 88, "temp": 24, "rh": 85, "hour": 6,  "soil_prev": 82, "ekspektasi": "Basah"},
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
        if cocok: benar_n += 1
        keputusan = "SIRAM" if (window and label in LABEL_SIRAM) else ("TIDAK SIRAM" if window else "DI LUAR JADWAL")
        hasil.append({"label_skenario": s["label"], "knn_label": label, "confidence_pct": conf,
                       "ekspektasi": s["ekspektasi"], "benar": cocok, "window": window, "keputusan": keputusan})
    return {"akurasi": f"{round(benar_n/len(SKENARIO_PRESET)*100,1)}%", "benar": benar_n,
            "total": len(SKENARIO_PRESET), "hasil": hasil}

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
        if cocok: benar_n += 1
        keputusan = "SIRAM" if (window and label in LABEL_SIRAM) else ("TIDAK SIRAM" if window else "DI LUAR JADWAL")
        hasil.append({"label_skenario": s.label_skenario, "knn_label": label, "confidence_pct": conf,
                       "ekspektasi": s.ekspektasi, "benar": cocok, "window": window, "keputusan": keputusan})
    with_eksp = [h for h in hasil if h["benar"] is not None]
    akurasi   = round(benar_n/len(with_eksp)*100,1) if with_eksp else None
    return {"akurasi": f"{akurasi}%" if akurasi else "N/A", "benar": benar_n,
            "total": len(payload.skenario), "hasil": hasil}

@app.post("/test-knn/reset")
def test_reset(x_api_key: str = Header(...)):
    check_api_key(x_api_key)
    state.pump_status       = False
    state.pump_start_ts     = None
    state.manual_override   = False
    state.last_watered_pagi = None
    state.last_watered_sore = None
    state.mode              = "auto"
    state.test_active       = False
    state.test_status       = ""
    state.test_result       = ""
    state.test_pump_start   = None
    save_state()
    return {"message": "State berhasil direset", "pump_status": state.pump_status, "mode": state.mode}

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
        peringatan = f"Jam {payload.hour:02d}:xx di luar window."
        reason     = f"[FIRE] Di luar window | KNN: {label} ({conf}%)"
    elif label not in LABEL_SIRAM:
        peringatan = f"KNN: {label} ({conf}%) — tidak siram."
        reason     = f"[FIRE] KNN={label} tidak siram | window={window}"
    else:
        set_pump(True, f"[FIRE] KNN={label} ({conf}%) window={window}")
        mark_watered(window, today)
        pump_action = "on"
        reason      = f"[FIRE] KNN={label} ({conf}%) | window={window} | pompa ON {PUMP_DURATION_MINUTES} menit"
    try:
        supabase.table("sensor_readings").insert({
            "id": str(uuid.uuid4()), "timestamp": datetime.utcnow().isoformat(),
            "soil_moisture": payload.soil_moisture, "temperature": payload.temperature,
            "air_humidity": payload.air_humidity, "label": label, "confidence": knn["confidence"],
            "needs_watering": pump_action == "on", "description": reason,
            "pump_status": state.pump_status, "mode": state.mode,
            "probabilities": knn.get("features", {}),
        }).execute()
    except Exception as e:
        log.warning(f"[FIRE] Gagal simpan DB: {e}")
    state.last_knn_label = label
    state.last_knn_conf  = knn["confidence"]
    save_state()
    keputusan = (f"POMPA ON — KNN={label} ({conf}%) | {PUMP_DURATION_MINUTES} menit" if pump_action == "on"
                 else (f"TIDAK SIRAM — {label} ({conf}%)" if window else f"DI LUAR JADWAL — jam {payload.hour:02d}:xx"))
    return {
        "label_skenario": payload.label_skenario,
        "classification": {"label": label, "confidence": knn["confidence"], "confidence_pct": conf},
        "window": window, "keputusan": keputusan, "pump_status": state.pump_status,
        "pump_action": pump_action, "pump_remaining_min": pump_remaining(),
        "ekspektasi": payload.ekspektasi,
        "benar": (label == payload.ekspektasi) if payload.ekspektasi else None,
        "peringatan": peringatan, "reason": reason, "features": knn.get("features", {}),
        "catatan": "[FIRE] Pompa nyala sungguhan jika syarat terpenuhi.",
    }
