import os
import json
import uuid
import logging
import asyncio
import concurrent.futures
import joblib
import numpy as np
from datetime import datetime, date
from typing import Optional
import time

from supabase import create_client, Client
from fastapi import FastAPI, HTTPException, Query, Security, Depends, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("siram-pintar")

# ── Path ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "knn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
META_PATH   = os.path.join(BASE_DIR, "model", "model_info.json")

# ── Supabase ──────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# ── API Key ───────────────────────────────────────────────────────────────────
VALID_API_KEY  = os.environ.get("API_KEY", "")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

APP_VERSION = "11.0.1-vercel"
# ═══════════════════════════════════════════════════════════════════════════════
# v11.0.1 — KNN Super Adaptif (10 Fitur) + Vercel Serverless Fix
# ═══════════════════════════════════════════════════════════════════════════════

_supabase: Client = None

# FIX 1: Re-inisialisasi Supabase untuk menangani Cold Start Vercel
def _get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL atau SUPABASE_KEY belum di-set!")
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        log.info("Supabase client diinisialisasi ulang (Cold Start handling).")
    return _supabase

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if not VALID_API_KEY:
        return "no-key-configured"
    if api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail={
            "error"  : "Unauthorized",
            "message": "API key tidak valid. Sertakan header: X-API-Key: <key>",
        })
    return api_key

_control_lock      = asyncio.Lock()
_daily_safety_lock = asyncio.Lock()
_daily_safety = {
    "date"            : None,
    "watering_count"  : 0,
    "prune_done_today": False,
}
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6, thread_name_prefix="sb-worker")

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI
# ══════════════════════════════════════════════════════════════════════════════
class WateringConfig:
    MORNING_WINDOW = (5, 7)
    EVENING_WINDOW = (16, 18)
    CRITICAL_DRY = 15.0
    MAX_PUMP_DURATION_MINUTES = 5
    MIN_PUMP_DURATION_SECONDS = 30
    COOLDOWN_MINUTES          = 60
    MIN_SESSION_GAP_MINUTES   = 10
    KNN_CONFIDENCE_MIN          = 50.0
    KNN_CONFIDENCE_MIN_PRIORITY = 35.0 
    NEEDS_WATERING_LABELS = {"Siram_Segera", "Siram_Prioritas"}
    SENSOR_DEBOUNCE_SECONDS        = 1
    SENSOR_TOLERANCE               = 0.5
    MANUAL_OVERRIDE_EXPIRE_SECONDS = 600
    HOT_TEMP_THRESHOLD = 34.0

CFG = WateringConfig()

app = FastAPI(
    title="Siram Pintar API",
    description="Sistem Penyiraman IoT — KNN Super Adaptif 10 Fitur",
    version=APP_VERSION,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

knn_model  = None
scaler     = None
model_meta: dict = {}

# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════
_STATE_DEFAULTS = {
    "pump_status"         : False,
    "mode"                : "auto",
    "last_label"          : None,
    "last_updated"        : None,
    "pump_start_ts"       : None,
    "pump_start_minute"   : None,
    "last_watered_minute" : None,
    "last_watered_ts"     : None,
    "last_soil_moisture"  : None,
    "last_temperature"    : None,
    "last_air_humidity"   : None,
    "last_rain_score"     : None,
    "missed_session"      : False,
    "manual_override"     : False,
    "manual_override_ts"  : None,
    "last_control_ts"     : None,
    "last_sensor_ts"      : None,
    "last_sensor_soil"    : None,
    "session_count_today" : 0,
    "session_count_date"  : None,
}

_rt_cache: dict = {"data": None, "timestamp": 0.0}

def _normalize_state(raw: dict) -> dict:
    row = dict(_STATE_DEFAULTS)
    row.update(raw)
    for k in ("pump_status", "missed_session", "manual_override"):
        row[k] = bool(row.get(k, False))
    for k in ("session_count_today",):
        row[k] = int(row.get(k) or 0)
    return row

# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _sb_get_state_sync() -> dict:
    try:
        res = _get_supabase().table("system_state").select("*").eq("id", 1).single().execute()
        if res.data:
            return _normalize_state(res.data)
    except Exception as e:
        log.error("Supabase get state: %s", e)
    return dict(_STATE_DEFAULTS)

def _sb_update_state_sync(**kwargs):
    if not kwargs:
        return
    _get_supabase().table("system_state").upsert({"id": 1, **kwargs}).execute()

def _sb_insert_sensor_sync(row: dict):
    _get_supabase().table("sensor_readings").insert(row).execute()

def _sb_ensure_state_row():
    try:
        res = _get_supabase().table("system_state").select("id").eq("id", 1).execute()
        if not res.data:
            _get_supabase().table("system_state").insert({"id": 1}).execute()
            log.info("system_state row id=1 dibuat.")
    except Exception as e:
        log.error("Gagal memastikan system_state row: %s", e)

# ══════════════════════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN
# ══════════════════════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup():
    global _supabase, knn_model, scaler, model_meta

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY belum di-set!")

    _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    log.info("Siram Pintar API v%s dimulai.", APP_VERSION)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _sb_ensure_state_row)
    
    # Menghapus polling task background karena tidak kompatibel dengan Serverless (Vercel)
    
    await _sync_daily_counter_from_db()

    if not os.path.exists(MODEL_PATH):
        log.warning("Model KNN belum ada! Pastikan ditaruh di folder model/.")
        return
    try:
        knn_model = joblib.load(MODEL_PATH)
        scaler    = joblib.load(SCALER_PATH)
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                model_meta = json.load(f)
        log.info(
            "Model KNN v%s dimuat. K=%s, Akurasi=%.2f%%, Fitur=%s, Label=%s",
            model_meta.get("version", "?"),
            model_meta.get("best_k"),
            float(model_meta.get("accuracy", 0)) * 100,
            model_meta.get("features"),
            model_meta.get("labels"),
        )
    except Exception as exc:
        log.error("Gagal memuat model: %s", exc)

@app.on_event("shutdown")
async def shutdown():
    _executor.shutdown(wait=False)

# FIX 2: Mencegah error 404 Favicon dari Vercel Logs
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════════════════
class SensorData(BaseModel):
    soil_moisture : float = Field(..., ge=0, le=100)
    temperature   : float = Field(..., ge=0, le=60)
    air_humidity  : float = Field(..., ge=0, le=100)
    hour          : Optional[int] = Field(default=None, ge=0, le=23)
    minute        : Optional[int] = Field(default=None, ge=0, le=59)
    day           : Optional[int] = Field(default=None, ge=0, le=6)

class ControlCommand(BaseModel):
    action : str           = Field(..., description="'on' atau 'off'")
    mode   : Optional[str] = Field(default="manual")

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def _encode_hour(hour: int) -> tuple:
    rad = hour * 2 * np.pi / 24
    return float(np.sin(rad)), float(np.cos(rad))

def _compute_rain_score(rh: float, soil_delta: float, temp_drop: float) -> float:
    score = 0.0
    if rh >= 92:
        score += 50
    elif rh >= 85:
        score += 30
    elif rh >= 78:
        score += 15
    if soil_delta >= 8:
        score += 35
    elif soil_delta >= 3:
        score += 20
    if temp_drop >= 3:
        score += 15
    return min(score, 100.0)

def _compute_evapotranspiration(temp: float, rh: float) -> float:
    vpd = (1 - rh / 100) * 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
    return round(float(np.clip(vpd * 15, 0, 100)), 2)

def _resolve_time_wit(hour, minute, day) -> tuple:
    if hour is not None and minute is not None and day is not None:
        return hour, minute, day, "esp32"
    now  = datetime.utcnow()
    h    = (now.hour + 9) % 24
    wday = (now.weekday() + 1) % 7
    return h, now.minute, wday, "server_fallback"

def _total_minutes(hour: int, minute: int) -> int:
    return hour * 60 + minute

def _elapsed_minutes(current: int, stored) -> int:
    if stored is None:
        return 999_999
    diff = current - int(stored)
    return diff if diff >= 0 else diff + 1440

def _elapsed_seconds_real(stored_ts_str) -> float:
    if not stored_ts_str:
        return 999_999.0
    try:
        stored = datetime.fromisoformat(str(stored_ts_str))
        if stored.tzinfo is not None:
            stored = stored.replace(tzinfo=None)
        return (datetime.now() - stored).total_seconds()
    except Exception:
        return 999_999.0

def _in_watering_window(hour: int) -> tuple:
    if CFG.MORNING_WINDOW[0] <= hour <= CFG.MORNING_WINDOW[1]:
        return True, "pagi"
    if CFG.EVENING_WINDOW[0] <= hour <= CFG.EVENING_WINDOW[1]:
        return True, "sore"
    return False, ""

def _should_skip_sensor(data: SensorData, state: dict, pump_is_on: bool) -> bool:
    if data.soil_moisture <= 0.0 or data.temperature <= 0.0 or data.temperature >= 60.0:
        log.warning("ANOMALI SENSOR: Soil=%.1f%% Temp=%.1f°C", data.soil_moisture, data.temperature)
        return True
    last_soil = state.get("last_sensor_soil")
    if last_soil is not None and abs(data.soil_moisture - float(last_soil)) > 30.0 and not pump_is_on:
        log.warning("ANOMALI: Perubahan >30%% tanpa pompa")
        return True
    elapsed = _elapsed_seconds_real(state.get("last_sensor_ts"))
    if elapsed < CFG.SENSOR_DEBOUNCE_SECONDS:
        if last_soil is None:
            return False
        if abs(data.soil_moisture - float(last_soil)) <= CFG.SENSOR_TOLERANCE:
            return True
    return False

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════
def classify(
    soil: float, temp: float, rh: float, hour: int,
    soil_prev: Optional[float] = None,
    temp_prev: Optional[float] = None,
    rh_prev:   Optional[float] = None,
    rain_score_prev: Optional[float] = None,
) -> dict:
    if knn_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model KNN belum dimuat.")

    try:
        _soil_prev = soil_prev if soil_prev is not None else soil
        _temp_prev = temp_prev if temp_prev is not None else temp
        _rh_prev   = rh_prev   if rh_prev   is not None else rh

        soil_delta  = soil - _soil_prev       
        temp_drop   = _temp_prev - temp       
        soil_trend  = soil_delta

        rain_score  = _compute_rain_score(rh, soil_delta, temp_drop)
        _rs_prev    = (
            rain_score_prev
            if rain_score_prev is not None
            else _compute_rain_score(_rh_prev, 0.0, 0.0)
        )
        et          = _compute_evapotranspiration(temp, rh)
        is_hot      = 1 if temp >= CFG.HOT_TEMP_THRESHOLD else 0
        hour_sin, hour_cos = _encode_hour(hour)

        X = np.array([[
            soil, temp, rh,
            hour_sin, hour_cos,
            rain_score, _rs_prev,
            soil_trend, et, is_hot,
        ]])
        X_scaled = scaler.transform(X)

        label  = knn_model.predict(X_scaled)[0]
        proba  = knn_model.predict_proba(X_scaled)[0]
        confs  = {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(knn_model.classes_, proba)
        }
        confidence     = round(float(max(proba)) * 100, 2)
        needs_watering = label in CFG.NEEDS_WATERING_LABELS

        return {
            "label"             : label,
            "confidence"        : confidence,
            "probabilities"     : confs,
            "needs_watering"    : needs_watering,
            "description"       : model_meta.get("label_desc", {}).get(label, ""),
            "computed_features" : {
                "rain_score"        : round(rain_score, 2),
                "rain_score_prev"   : round(_rs_prev, 2),
                "soil_trend"        : round(soil_trend, 2),
                "evapotranspiration": et,
                "is_hot"            : is_hot,
                "hour_sin"          : round(hour_sin, 4),
                "hour_cos"          : round(hour_cos, 4),
            },
            "k"     : model_meta.get("best_k", knn_model.n_neighbors),
            "metric": model_meta.get("metric", "euclidean"),
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error("KNN classify error: %s", e)
        raise HTTPException(status_code=503, detail=f"Model inference error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
# FIX 3: Fetch-on-Demand State Retrieval (Menggantikan Polling)
def _get_state(force_fresh: bool = False) -> dict:
    if force_fresh:
        row = _sb_get_state_sync()
        _rt_cache["data"]      = row
        _rt_cache["timestamp"] = time.monotonic()
        return row
        
    cached = _rt_cache["data"]
    age    = time.monotonic() - _rt_cache.get("timestamp", 0)
    
    # TTL Cache 2 detik agar tidak terlalu sering tembak DB jika request berdekatan
    if cached and age < 2.0:
        return cached.copy()
        
    try:
        row = _sb_get_state_sync()
        _rt_cache["data"]      = row
        _rt_cache["timestamp"] = time.monotonic()
        return row
    except Exception as e:
        log.error("Fallback get state: %s", e)
        return cached.copy() if cached else dict(_STATE_DEFAULTS)

async def _update_state_async(**kwargs):
    if not kwargs:
        return
    loop = asyncio.get_event_loop()

    def _do():
        _sb_update_state_sync(**kwargs)
        fresh = _sb_get_state_sync()
        _rt_cache["data"]      = fresh
        _rt_cache["timestamp"] = time.monotonic()

    await loop.run_in_executor(_executor, _do)

# ══════════════════════════════════════════════════════════════════════════════
# DAILY COUNTER
# ══════════════════════════════════════════════════════════════════════════════
async def _sync_daily_counter_from_db():
    loop = asyncio.get_event_loop()
    row  = await loop.run_in_executor(_executor, _sb_get_state_sync)
    db_count    = int(row.get("session_count_today") or 0)
    db_date_raw = row.get("session_count_date")
    db_date     = None
    if db_date_raw:
        try:
            db_date = date.fromisoformat(str(db_date_raw)[:10])
        except Exception:
            pass
    today = date.today()
    async with _daily_safety_lock:
        if db_date == today:
            _daily_safety["date"]           = today
            _daily_safety["watering_count"] = db_count
        else:
            _daily_safety["date"]           = today
            _daily_safety["watering_count"] = 0

def _daily_counter_reset_if_new_day():
    today = date.today()
    if _daily_safety["date"] != today:
        _daily_safety["date"]             = today
        _daily_safety["watering_count"]   = 0
        _daily_safety["prune_done_today"] = False
        return True
    return False

def _prune_sensor_readings():
    try:
        import datetime as dt
        cutoff = (datetime.now() - dt.timedelta(days=14)).isoformat()
        _get_supabase().table("sensor_readings").delete().lt("timestamp", cutoff).execute()
        log.info("Pruned sensor readings > 14 hari.")
    except Exception as e:
        log.error("Prune error: %s", e)

async def _maybe_schedule_prune(bg_tasks: BackgroundTasks):
    async with _daily_safety_lock:
        _daily_counter_reset_if_new_day()
        if not _daily_safety["prune_done_today"]:
            _daily_safety["prune_done_today"] = True
            bg_tasks.add_task(_prune_sensor_readings)

# ══════════════════════════════════════════════════════════════════════════════
# SMART WATERING ENGINE
# ══════════════════════════════════════════════════════════════════════════════
async def _evaluate_smart_watering_async(
    knn_result, hour, minute,
    soil_moisture, air_humidity, temperature,
    state, current_total_minutes,
) -> dict:
    resp = {
        "action"           : None,
        "reason"           : "",
        "blocked_reason"   : None,
        "decision_path"    : [],
        "knn_label"        : knn_result["label"],
        "knn_confidence"   : knn_result["confidence"],
        "knn_probabilities": knn_result["probabilities"],
        "knn_computed"     : knn_result.get("computed_features", {}),
        "pending_updates"  : {},
    }

    if state.get("manual_override"):
        age = _elapsed_seconds_real(state.get("manual_override_ts"))
        if age < CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS:
            remaining = int(CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS - age)
            resp["blocked_reason"] = f"Manual override aktif ({remaining}s lagi)"
            resp["decision_path"].append("B1-manual-override")
            return resp
        else:
            resp["pending_updates"].update(manual_override=False, manual_override_ts=None)

    async def _add_pump_on_updates(upd: dict):
        async with _daily_safety_lock:
            _daily_counter_reset_if_new_day()
            _daily_safety["watering_count"] += 1
            cnt = _daily_safety["watering_count"]
        upd["session_count_today"] = cnt
        upd["session_count_date"]  = date.today().isoformat()

    if state["pump_status"]:
        elapsed_sec = _elapsed_seconds_real(state.get("pump_start_ts"))
        max_sec     = CFG.MAX_PUMP_DURATION_MINUTES * 60

        if elapsed_sec >= max_sec:
            resp["pending_updates"].update(
                pump_status=False,
                last_watered_minute=current_total_minutes,
                last_watered_ts=datetime.now().isoformat(),
                pump_start_ts=None, pump_start_minute=None, missed_session=False,
            )
            resp["action"] = "off"
            resp["reason"] = f"Auto-stop: durasi {elapsed_sec:.0f}s maks"
            resp["decision_path"].append("A1-max-duration")
            return resp

        if elapsed_sec < CFG.MIN_PUMP_DURATION_SECONDS:
            resp["reason"] = f"Warmup ({elapsed_sec:.0f}s)"
            resp["decision_path"].append("A-warmup")
            return resp

        if not knn_result["needs_watering"]:
            resp["pending_updates"].update(
                pump_status=False,
                last_watered_minute=current_total_minutes,
                last_watered_ts=datetime.now().isoformat(),
                pump_start_ts=None, pump_start_minute=None, missed_session=False,
            )
            resp["action"] = "off"
            resp["reason"] = (
                f"KNN: tanah sudah {knn_result['label']} "
                f"({knn_result['confidence']}%) — pompa dimatikan"
            )
            resp["decision_path"].append("A2-knn-off")
            return resp

        if knn_result["label"] in ("Hujan_Aktif", "Hujan_Prediksi"):
            resp["pending_updates"].update(
                pump_status=False,
                last_watered_minute=current_total_minutes,
                last_watered_ts=datetime.now().isoformat(),
                pump_start_ts=None, pump_start_minute=None, missed_session=False,
            )
            resp["action"] = "off"
            resp["reason"] = (
                f"KNN: {knn_result['label']} ({knn_result['confidence']}%) — pompa dimatikan"
            )
            resp["decision_path"].append("A3-rain-off")
            return resp

        resp["reason"] = (
            f"KNN: {knn_result['label']} ({knn_result['confidence']}%) — "
            f"pompa jalan ({elapsed_sec:.0f}s)"
        )
        resp["decision_path"].append("A4-running")
        return resp

    in_window, window_label = _in_watering_window(hour)

    night_emergency = (
        not in_window
        and soil_moisture <= CFG.CRITICAL_DRY
        and knn_result["label"] not in ("Hujan_Aktif", "Hujan_Prediksi")
    )
    if night_emergency:
        now_ts = datetime.now().isoformat()
        pump_u = dict(
            pump_status=True,
            pump_start_minute=current_total_minutes,
            pump_start_ts=now_ts,
        )
        await _add_pump_on_updates(pump_u)
        resp["pending_updates"].update(pump_u)
        resp["action"] = "on"
        resp["reason"] = (
            f"DARURAT: soil={soil_moisture:.1f}% "
            f"<= {CFG.CRITICAL_DRY}% — siram darurat malam"
        )
        resp["decision_path"].append("B2-emergency")
        return resp

    if not knn_result["needs_watering"]:
        resp["blocked_reason"] = (
            f"KNN: {knn_result['label']} ({knn_result['confidence']}%) — tidak perlu siram"
        )
        resp["decision_path"].append("B3-knn-no-water")
        return resp

    min_conf = (
        CFG.KNN_CONFIDENCE_MIN_PRIORITY
        if knn_result["label"] == "Siram_Prioritas"
        else CFG.KNN_CONFIDENCE_MIN
    )
    if knn_result["confidence"] < min_conf:
        resp["blocked_reason"] = (
            f"KNN confidence {knn_result['confidence']}% "
            f"< minimum {min_conf:.0f}%"
        )
        resp["decision_path"].append("B4-low-confidence")
        return resp

    elapsed_cd = _elapsed_minutes(current_total_minutes, state.get("last_watered_minute"))
    if elapsed_cd < CFG.COOLDOWN_MINUTES:
        resp["blocked_reason"] = (
            f"Cooldown: sisa {CFG.COOLDOWN_MINUTES - elapsed_cd} menit"
        )
        resp["decision_path"].append("B5-cooldown")
        return resp

    now_ts = datetime.now().isoformat()
    pump_u = dict(
        pump_status=True,
        pump_start_minute=current_total_minutes,
        pump_start_ts=now_ts,
    )
    await _add_pump_on_updates(pump_u)
    resp["pending_updates"].update(pump_u)
    resp["action"] = "on"
    resp["reason"] = (
        f"KNN memutuskan siram [{window_label or 'darurat'}]: "
        f"label={knn_result['label']}, "
        f"conf={knn_result['confidence']}%, "
        f"rain_score={knn_result['computed_features'].get('rain_score', 0)}, "
        f"ET={knn_result['computed_features'].get('evapotranspiration', 0)}"
    )
    resp["decision_path"].append("B6-knn-final")
    return resp

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {
        "status"      : "online",
        "version"     : APP_VERSION,
        "model_ready" : knn_model is not None,
        "model_info"  : {
            "version"   : model_meta.get("version"),
            "algorithm" : model_meta.get("algorithm"),
            "best_k"    : model_meta.get("best_k"),
            "accuracy"  : f"{float(model_meta.get('accuracy', 0)) * 100:.2f}%",
            "features"  : model_meta.get("features"),
            "n_features": model_meta.get("n_features"),
            "labels"    : model_meta.get("labels"),
        } if model_meta else None,
    }

@app.post("/sensor", dependencies=[Depends(verify_api_key)])
async def receive_sensor(data: SensorData, bg_tasks: BackgroundTasks):
    hour, minute, _day, time_source = _resolve_time_wit(data.hour, data.minute, data.day)
    timestamp         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_id            = str(uuid.uuid4())
    current_total_min = _total_minutes(hour, minute)

    state      = _get_state()
    pump_is_on = bool(state.get("pump_status", False))

    if _should_skip_sensor(data, state, pump_is_on):
        elapsed_spam = _elapsed_seconds_real(state.get("last_sensor_ts"))
        if elapsed_spam < 1.0:
            return {
                "received"      : True,
                "timestamp"     : state.get("last_updated") or timestamp,
                "device_time"   : f"{hour:02d}:{minute:02d}",
                "time_source"   : time_source,
                "debounced"     : True,
                "sensor"        : {
                    "soil_moisture": data.soil_moisture,
                    "temperature"  : data.temperature,
                    "air_humidity" : data.air_humidity,
                },
                "classification": None,
                "pump_status"   : state["pump_status"],
                "pump_action"   : None,
                "mode"          : state["mode"],
                "auto_info"     : None,
            }

    await _maybe_schedule_prune(bg_tasks)

    knn_result = classify(
        soil      = data.soil_moisture,
        temp      = data.temperature,
        rh        = data.air_humidity,
        hour      = hour,
        soil_prev = state.get("last_soil_moisture"),
        temp_prev = state.get("last_temperature"),
        rh_prev   = state.get("last_air_humidity"),
        rain_score_prev = state.get("last_rain_score"),
    )

    final_action = None
    smart_eval   = {}

    if state["mode"] == "auto":
        smart_eval = await _evaluate_smart_watering_async(
            knn_result=knn_result, hour=hour, minute=minute,
            soil_moisture=data.soil_moisture, air_humidity=data.air_humidity,
            temperature=data.temperature, state=state,
            current_total_minutes=current_total_min,
        )
        final_action = smart_eval.get("action")

    pump_status_logged = (
        (final_action == "on") if final_action is not None else state["pump_status"]
    )

    sensor_updates = dict(
        last_label          = knn_result["label"],
        last_updated        = timestamp,
        last_soil_moisture  = data.soil_moisture,
        last_temperature    = data.temperature,
        last_air_humidity   = data.air_humidity,
        last_rain_score     = knn_result["computed_features"]["rain_score"],
        last_sensor_ts      = datetime.now().isoformat(),
        last_sensor_soil    = data.soil_moisture,
    )
    pending     = smart_eval.get("pending_updates", {})
    all_updates = {**sensor_updates, **pending}

    optimistic = {**(_rt_cache["data"] or {}), **all_updates}
    _rt_cache["data"]      = _normalize_state(optimistic)
    _rt_cache["timestamp"] = time.monotonic()

    sensor_row = {
        "id"                : row_id,
        "timestamp"         : datetime.now().isoformat(),
        "soil_moisture"     : data.soil_moisture,
        "temperature"       : data.temperature,
        "air_humidity"      : data.air_humidity,
        "label"             : knn_result["label"],
        "confidence"        : knn_result["confidence"],
        "needs_watering"    : knn_result["needs_watering"],
        "description"       : knn_result.get("description", ""),
        "probabilities"     : knn_result["probabilities"],
        "computed_features" : knn_result.get("computed_features", {}),
        "pump_status"       : pump_status_logged,
        "mode"              : state["mode"],
        "hour"              : hour,
        "minute"            : minute,
    }

    loop = asyncio.get_event_loop()

    def _write_state():
        _sb_update_state_sync(**all_updates)

    def _write_sensor():
        try:
            _sb_insert_sensor_sync(sensor_row)
        except Exception as e:
            log.error("Sensor insert gagal: %s", e)

    try:
        await loop.run_in_executor(_executor, _write_state)
        asyncio.ensure_future(loop.run_in_executor(_executor, _write_sensor))
    except Exception as e:
        log.error("State write gagal: %s", e)

    new_state = _get_state()
    return {
        "received"      : True,
        "timestamp"     : timestamp,
        "device_time"   : f"{hour:02d}:{minute:02d}",
        "time_source"   : time_source,
        "debounced"     : False,
        "sensor"        : {
            "soil_moisture": data.soil_moisture,
            "temperature"  : data.temperature,
            "air_humidity" : data.air_humidity,
        },
        "classification": knn_result,
        "pump_status"   : new_state["pump_status"],
        "pump_action"   : final_action,
        "mode"          : new_state["mode"],
        "auto_info"     : {
            "reason"           : smart_eval.get("reason", ""),
            "blocked_reason"   : smart_eval.get("blocked_reason"),
            "decision_path"    : smart_eval.get("decision_path", []),
            "knn_label"        : smart_eval.get("knn_label"),
            "knn_confidence"   : smart_eval.get("knn_confidence"),
            "knn_probabilities": smart_eval.get("knn_probabilities"),
            "knn_computed"     : smart_eval.get("knn_computed"),
            "manual_override"  : new_state.get("manual_override", False),
        } if state["mode"] == "auto" else None,
    }

@app.get("/pump-status", dependencies=[Depends(verify_api_key)])
def get_pump_status():
    state = _get_state()
    return {
        "pump_status"    : state["pump_status"],
        "mode"           : state["mode"],
        "manual_override": state.get("manual_override", False),
    }

@app.post("/control", dependencies=[Depends(verify_api_key)])
async def control_pump(cmd: ControlCommand):
    action = (cmd.action or "").lower().strip()
    if action not in ("on", "off"):
        raise HTTPException(status_code=400, detail="Action harus 'on' atau 'off'.")

    mode = (cmd.mode or "manual").lower().strip()
    if mode not in ("auto", "manual"):
        mode = "manual"

    loop = asyncio.get_event_loop()

    async with _control_lock:
        state   = await loop.run_in_executor(_executor, _sb_get_state_sync)
        pump_on = action == "on"
        now_ts  = datetime.now().isoformat()

        if state["pump_status"] == pump_on and state["mode"] == mode:
            return {
                "success"        : True,
                "debounced"      : True,
                "pump_status"    : state["pump_status"],
                "mode"           : state["mode"],
                "manual_override": state.get("manual_override", False),
                "timestamp"      : now_ts,
            }

        update_kwargs: dict = {"last_control_ts": now_ts, "mode": mode}

        if state["pump_status"] != pump_on:
            update_kwargs["pump_status"] = pump_on
            if not pump_on:
                cur_min = _total_minutes(*_resolve_time_wit(None, None, None)[:2])
                update_kwargs.update(
                    pump_start_ts=None, pump_start_minute=None,
                    last_watered_ts=now_ts, last_watered_minute=cur_min,
                    manual_override=True, manual_override_ts=now_ts,
                )
            else:
                now_utc = datetime.utcnow()
                h_wit   = (now_utc.hour + 9) % 24
                update_kwargs.update(
                    pump_start_ts=now_ts,
                    pump_start_minute=_total_minutes(h_wit, now_utc.minute),
                    manual_override=False, manual_override_ts=None,
                )
                async with _daily_safety_lock:
                    _daily_counter_reset_if_new_day()
                    _daily_safety["watering_count"] += 1
                    new_count = _daily_safety["watering_count"]
                update_kwargs["session_count_today"] = new_count
                update_kwargs["session_count_date"]  = date.today().isoformat()

        try:
            await loop.run_in_executor(_executor, lambda: _sb_update_state_sync(**update_kwargs))
        except Exception as e:
            log.error("Control write gagal: %s", e)
            raise HTTPException(status_code=503, detail="Gagal menyimpan ke Supabase.")

        new_state = _normalize_state({**(_rt_cache["data"] or {}), **update_kwargs})
        _rt_cache["data"]      = new_state
        _rt_cache["timestamp"] = time.monotonic()

        return {
            "success"        : True,
            "debounced"      : False,
            "pump_status"    : new_state["pump_status"],
            "mode"           : new_state["mode"],
            "manual_override": new_state.get("manual_override", False),
            "watering_today" : _daily_safety["watering_count"],
            "timestamp"      : now_ts,
        }

@app.get("/status", dependencies=[Depends(verify_api_key)])
async def get_status():
    state = _get_state()
    loop  = asyncio.get_event_loop()

    def _get_latest():
        try:
            res = (
                _get_supabase()
                .table("sensor_readings")
                .select("*")
                .order("timestamp", desc=True)
                .limit(1)
                .execute()
            )
            return res.data[0] if res.data else None
        except Exception as e:
            log.error("latest sensor: %s", e)
        return None

    latest = await loop.run_in_executor(_executor, _get_latest)

    async with _daily_safety_lock:
        watering_today = _daily_safety["watering_count"]

    return {
        "pump_status"     : state["pump_status"],
        "mode"            : state["mode"],
        "last_label"      : state["last_label"],
        "last_updated"    : str(state["last_updated"]) if state["last_updated"] else None,
        "manual_override" : state.get("manual_override", False),
        "watering_today"  : watering_today,
        "last_watered_ts" : str(state["last_watered_ts"]) if state.get("last_watered_ts") else None,
        "watering_windows": {
            "morning": f"{CFG.MORNING_WINDOW[0]:02d}:00–{CFG.MORNING_WINDOW[1]:02d}:59 WIT",
            "evening": f"{CFG.EVENING_WINDOW[0]:02d}:00–{CFG.EVENING_WINDOW[1]:02d}:59 WIT",
        },
        "knn_config": {
            "version"               : model_meta.get("version", "3.0"),
            "n_features"            : model_meta.get("n_features", 10),
            "features"              : model_meta.get("features"),
            "confidence_min"        : CFG.KNN_CONFIDENCE_MIN,
            "confidence_min_priority": CFG.KNN_CONFIDENCE_MIN_PRIORITY,
            "critical_dry_safety"   : CFG.CRITICAL_DRY,
            "needs_watering_labels" : list(CFG.NEEDS_WATERING_LABELS),
            "note": "KNN 10 fitur adalah satu-satunya pengambil keputusan.",
        },
        "model_info": {
            "algorithm"   : model_meta.get("algorithm"),
            "version"     : model_meta.get("version"),
            "best_k"      : model_meta.get("best_k"),
            "accuracy"    : f"{float(model_meta.get('accuracy', 0)) * 100:.2f}%",
            "cv_accuracy" : f"{float(model_meta.get('cv_accuracy', 0)) * 100:.2f}%",
            "features"    : model_meta.get("features"),
            "labels"      : model_meta.get("labels"),
            "lokasi"      : model_meta.get("lokasi"),
            "jam_siram"   : model_meta.get("jam_siram"),
        } if model_meta else None,
        "latest_data": latest,
    }

@app.get("/history", dependencies=[Depends(verify_api_key)])
async def get_history(
    limit     : int  = Query(default=50, ge=1, le=500),
    pump_only : bool = Query(default=False),
):
    loop = asyncio.get_event_loop()

    def _fetch():
        try:
            query = (
                _get_supabase()
                .table("sensor_readings")
                .select("*")
                .order("timestamp", desc=True)
                .limit(limit)
            )
            if pump_only:
                query = query.eq("pump_status", True)
            res     = query.execute()
            records = sorted(res.data or [], key=lambda x: x.get("timestamp", ""))
            return records
        except Exception as e:
            log.error("History error: %s", e)
            return []

    records = await loop.run_in_executor(_executor, _fetch)
    return {"total": len(records), "pump_only": pump_only, "records": records}