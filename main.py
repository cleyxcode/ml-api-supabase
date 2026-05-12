import os
import json
import uuid
import logging
import asyncio
import concurrent.futures
import joblib
import numpy as np
from datetime import datetime, date
from typing import Optional, List
import time

from supabase import create_client, Client
from fastapi import FastAPI, HTTPException, Query, Security, Depends, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("siram-pintar")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "knn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
META_PATH   = os.path.join(BASE_DIR, "model", "model_info.json")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

VALID_API_KEY  = os.environ.get("API_KEY", "")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

APP_VERSION = "11.4.0"
# ═══════════════════════════════════════════════════════════════════════════════
# v11.4.0 — Fix rain_score false positive untuk iklim lembab tropis (Ambon)
#
# ROOT CAUSE BUG HUJAN_AKTIF FALSE POSITIVE:
#   _compute_rain_score() menggunakan threshold RH >= 92 → score +50.
#   Di Ambon, RH 85-95% adalah kondisi NORMAL sehari-hari bahkan cuaca cerah.
#   Akibatnya rain_score selalu tinggi → KNN prediksi Hujan_Aktif 100%
#   padahal tidak ada hujan sama sekali.
#
# FIX:
#   1. Naikkan threshold RH di _compute_rain_score():
#      - RH >= 97 → +40 (benar-benar hujan lebat)
#      - RH >= 94 → +20 (kemungkinan hujan)
#      - RH >= 91 → +8  (lembab tinggi, belum tentu hujan)
#   2. Tambah RAIN_SCORE_THRESHOLD = 60 di WateringConfig.
#      Rain score < 60 tidak cukup untuk trigger label hujan.
#   3. Penyiraman di window wajib pagi/sore TETAP jalan
#      walau rain_score tinggi (abaikan hujan di jam siram).
#   4. Blok A3 (pompa ON + label hujan) dan B3 (label hujan blokir pompa)
#      keduanya skip jika sedang di dalam watering window.
#
# v11.3.0 — Fix OVR loop: invalidate cache saat expired + /pump-status ikut cek expiry
# ═══════════════════════════════════════════════════════════════════════════════

_supabase: Client = None


def _get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL atau SUPABASE_KEY belum di-set!")
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
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
    CRITICAL_DRY   = 15.0

    MAX_PUMP_DURATION_MINUTES  = 5
    MIN_PUMP_DURATION_SECONDS  = 30
    COOLDOWN_MINUTES           = 60
    MIN_SESSION_GAP_MINUTES    = 10

    KNN_CONFIDENCE_MIN          = 50.0
    KNN_CONFIDENCE_MIN_PRIORITY = 35.0
    NEEDS_WATERING_LABELS       = {"Siram_Segera", "Siram_Prioritas"}

    SENSOR_DEBOUNCE_SECONDS = 1
    SENSOR_TOLERANCE        = 0.5

    MANUAL_OVERRIDE_EXPIRE_SECONDS = 180

    HOT_TEMP_THRESHOLD = 34.0

    # [FIX v11.4.0] Threshold rain_score minimum untuk dianggap hujan.
    # Di bawah nilai ini, label hujan diabaikan meskipun RH tinggi.
    # Iklim Ambon: RH tinggi adalah normal → butuh threshold lebih tinggi.
    RAIN_SCORE_THRESHOLD = 60.0


CFG = WateringConfig()

app = FastAPI(
    title="Siram Pintar API",
    description=(
        "Sistem Penyiraman IoT — KNN Super Adaptif 10 Fitur v11.4\n\n"
        "### Endpoint Pengujian (Postman)\n"
        "- `POST /test-knn` — Uji KNN dengan data sensor bebas\n"
        "- `POST /test-knn/batch` — Uji banyak skenario sekaligus\n"
        "- `GET  /test-knn/skenario` — Ambil 12 skenario preset\n"
        "- `POST /test-knn/reset` — Reset state cooldown\n"
    ),
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
    row["session_count_today"] = int(row.get("session_count_today") or 0)
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
    log.info("Siram Pintar API v%s dimulai. OVR timeout=%ds, RainThreshold=%.0f",
             APP_VERSION, CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS, CFG.RAIN_SCORE_THRESHOLD)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _sb_ensure_state_row)
    await _sync_daily_counter_from_db()

    if not os.path.exists(MODEL_PATH):
        log.warning("Model KNN belum ada! Letakkan di folder model/.")
        return
    try:
        knn_model = joblib.load(MODEL_PATH)
        scaler    = joblib.load(SCALER_PATH)
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                model_meta = json.load(f)
        log.info("Model KNN v%s dimuat. K=%s, Akurasi=%.2f%%",
                 model_meta.get("version", "?"),
                 model_meta.get("best_k"),
                 float(model_meta.get("accuracy", 0)) * 100)
    except Exception as exc:
        log.error("Gagal memuat model: %s", exc)


@app.on_event("shutdown")
async def shutdown():
    _executor.shutdown(wait=False)


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


class TestKNNRequest(BaseModel):
    soil_moisture  : float          = Field(..., ge=0,  le=100)
    temperature    : float          = Field(..., ge=0,  le=60)
    air_humidity   : float          = Field(..., ge=0,  le=100)
    hour           : int            = Field(..., ge=0,  le=23)
    soil_prev      : Optional[float]= Field(default=None, ge=0, le=100)
    temp_prev      : Optional[float]= Field(default=None, ge=0, le=60)
    rh_prev        : Optional[float]= Field(default=None, ge=0, le=100)
    label_skenario : Optional[str]  = Field(default=None)
    ekspektasi     : Optional[str]  = Field(default=None)


class TestKNNBatchRequest(BaseModel):
    skenario: List[TestKNNRequest] = Field(..., min_length=1, max_length=50)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def _encode_hour(hour: int) -> tuple:
    rad = hour * 2 * np.pi / 24
    return float(np.sin(rad)), float(np.cos(rad))


# [FIX v11.4.0] _compute_rain_score — threshold dinaikkan untuk iklim Ambon.
# RH 85-95% adalah NORMAL di Ambon (kota paling lembab di Indonesia),
# bukan indikator hujan. Sinyal utama hujan adalah kenaikan soil_delta
# yang drastis (air masuk ke tanah), bukan RH semata.
def _compute_rain_score(rh: float, soil_delta: float, temp_drop: float) -> float:
    score = 0.0

    # RH threshold dinaikkan jauh — RH tinggi di Ambon bukan berarti hujan
    if   rh >= 97: score += 40   # benar-benar sangat lembab → kemungkinan hujan lebat
    elif rh >= 94: score += 20   # lembab ekstrem → mungkin hujan
    elif rh >= 91: score +=  8   # lembab tinggi → normal untuk Ambon, skor kecil

    # soil_delta adalah indikator TERKUAT bahwa ada air masuk ke tanah
    # (pompa ON atau hujan nyata menaikkan soil moisture drastis)
    if   soil_delta >= 10: score += 45  # kenaikan sangat drastis → hampir pasti hujan/pompa
    elif soil_delta >=  5: score += 25  # kenaikan signifikan
    elif soil_delta >=  2: score +=  8  # kenaikan kecil → bisa noise sensor

    # temp_drop hanya berarti jika cukup besar (hujan biasanya turunkan suhu >5C)
    if   temp_drop >= 5: score += 15
    elif temp_drop >= 3: score +=  5

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
        return True
    last_soil = state.get("last_sensor_soil")
    if last_soil is not None and abs(data.soil_moisture - float(last_soil)) > 30.0 and not pump_is_on:
        return True
    elapsed = _elapsed_seconds_real(state.get("last_sensor_ts"))
    if elapsed < CFG.SENSOR_DEBOUNCE_SECONDS:
        if last_soil is None:
            return False
        if abs(data.soil_moisture - float(last_soil)) <= CFG.SENSOR_TOLERANCE:
            return True
    return False


# [FIX v11.4.0] Helper: apakah label termasuk kategori "hujan" dan
# rain_score-nya cukup tinggi (>= RAIN_SCORE_THRESHOLD) untuk dipercaya.
# Jika rain_score rendah, label hujan dianggap false positive dan diabaikan.
def _is_rain_label_credible(knn_result: dict) -> bool:
    label      = knn_result.get("label", "")
    rain_score = knn_result.get("computed_features", {}).get("rain_score", 0.0)
    if label not in ("Hujan_Aktif", "Hujan_Prediksi"):
        return False
    return rain_score >= CFG.RAIN_SCORE_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════
def classify(
    soil: float, temp: float, rh: float, hour: int,
    soil_prev       : Optional[float] = None,
    temp_prev       : Optional[float] = None,
    rh_prev         : Optional[float] = None,
    rain_score_prev : Optional[float] = None,
) -> dict:
    if knn_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model KNN belum dimuat.")
    try:
        _soil_prev = soil_prev if soil_prev is not None else soil
        _temp_prev = temp_prev if temp_prev is not None else temp
        _rh_prev   = rh_prev   if rh_prev   is not None else rh

        soil_delta = soil - _soil_prev
        temp_drop  = _temp_prev - temp
        soil_trend = soil_delta

        rain_score = _compute_rain_score(rh, soil_delta, temp_drop)
        _rs_prev   = (rain_score_prev if rain_score_prev is not None
                      else _compute_rain_score(_rh_prev, 0.0, 0.0))
        et         = _compute_evapotranspiration(temp, rh)
        is_hot     = 1 if temp >= CFG.HOT_TEMP_THRESHOLD else 0
        hour_sin, hour_cos = _encode_hour(hour)

        X        = np.array([[soil, temp, rh, hour_sin, hour_cos,
                               rain_score, _rs_prev, soil_trend, et, is_hot]])
        X_scaled = scaler.transform(X)

        label  = knn_model.predict(X_scaled)[0]
        proba  = knn_model.predict_proba(X_scaled)[0]
        confs  = {cls: round(float(p) * 100, 2)
                  for cls, p in zip(knn_model.classes_, proba)}
        conf   = round(float(max(proba)) * 100, 2)

        # [FIX v11.4.0] Jika KNN memprediksi label hujan TAPI rain_score
        # di bawah threshold → ini false positive (iklim lembab).
        # Override label ke "Optimal" atau "Siram_Nanti" berdasarkan soil.
        if label in ("Hujan_Aktif", "Hujan_Prediksi") and rain_score < CFG.RAIN_SCORE_THRESHOLD:
            log.info(
                "[FIX] Label hujan '%s' dibatalkan — rain_score=%.1f < threshold=%.0f. "
                "RH=%.1f%%, soil_delta=%.1f. Override ke label berbasis soil.",
                label, rain_score, CFG.RAIN_SCORE_THRESHOLD, rh, soil_delta
            )
            # Tentukan label pengganti berdasarkan kondisi tanah
            if soil <= 30.0:
                label = "Siram_Segera"
            elif soil <= 44.0:
                label = "Siram_Nanti"
            elif soil <= 70.0:
                label = "Optimal"
            else:
                label = "Basah"

            # Recalculate confidence dari distribusi probabilitas yang ada
            # (ambil probabilitas label pengganti jika ada, else set 70%)
            conf = round(confs.get(label, 70.0), 2)

        return {
            "label"             : label,
            "confidence"        : conf,
            "probabilities"     : confs,
            "needs_watering"    : label in CFG.NEEDS_WATERING_LABELS,
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
            # [FIX] Tambahkan rain_score_threshold di response untuk debugging
            "rain_score_threshold": CFG.RAIN_SCORE_THRESHOLD,
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error("KNN classify error: %s", e)
        raise HTTPException(status_code=503, detail=f"Model inference error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def _get_state(force_fresh: bool = False) -> dict:
    if force_fresh:
        row = _sb_get_state_sync()
        _rt_cache["data"]      = row
        _rt_cache["timestamp"] = time.monotonic()
        return row
    cached = _rt_cache["data"]
    age    = time.monotonic() - _rt_cache.get("timestamp", 0)
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

    # Tentukan apakah sedang di window siram wajib (pagi/sore)
    in_window, window_label = _in_watering_window(hour)

    # ══════════════════════════════════════════════════════════════════════════
    # B1. Manual Override
    # ══════════════════════════════════════════════════════════════════════════
    if state.get("manual_override"):
        age = _elapsed_seconds_real(state.get("manual_override_ts"))

        if age < CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS:
            remaining = int(CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS - age)
            resp["blocked_reason"] = (
                f"Manual override aktif ({remaining}s lagi / "
                f"{remaining // 60}m{remaining % 60:02d}s)"
            )
            resp["decision_path"].append("B1-manual-override")
            resp["pending_updates"].update(
                manual_override    = True,
                manual_override_ts = state.get("manual_override_ts"),
            )
            return resp

        else:
            log.info("[OVR] Expired setelah %ds — reset otomatis.", int(age))
            resp["pending_updates"].update(
                manual_override    = False,
                manual_override_ts = None,
            )
            _rt_cache["data"]      = None
            _rt_cache["timestamp"] = 0.0

    async def _add_pump_on_updates(upd: dict):
        async with _daily_safety_lock:
            _daily_counter_reset_if_new_day()
            _daily_safety["watering_count"] += 1
            cnt = _daily_safety["watering_count"]
        upd["session_count_today"] = cnt
        upd["session_count_date"]  = date.today().isoformat()

    # ══════════════════════════════════════════════════════════════════════════
    # Pompa sedang ON
    # ══════════════════════════════════════════════════════════════════════════
    if state["pump_status"]:
        elapsed_sec = _elapsed_seconds_real(state.get("pump_start_ts"))
        max_sec     = CFG.MAX_PUMP_DURATION_MINUTES * 60

        # A1. Durasi maksimum tercapai → matikan pompa
        if elapsed_sec >= max_sec:
            resp["pending_updates"].update(
                pump_status=False, last_watered_minute=current_total_minutes,
                last_watered_ts=datetime.now().isoformat(),
                pump_start_ts=None, pump_start_minute=None, missed_session=False,
            )
            resp["action"] = "off"
            resp["reason"] = f"Auto-stop: durasi {elapsed_sec:.0f}s"
            resp["decision_path"].append("A1-max-duration")
            return resp

        # A-warmup. Pompa baru nyala, tunggu minimum
        if elapsed_sec < CFG.MIN_PUMP_DURATION_SECONDS:
            resp["reason"] = f"Warmup ({elapsed_sec:.0f}s)"
            resp["decision_path"].append("A-warmup")
            return resp

        # A2. KNN tidak perlu siram → matikan pompa
        if not knn_result["needs_watering"]:
            resp["pending_updates"].update(
                pump_status=False, last_watered_minute=current_total_minutes,
                last_watered_ts=datetime.now().isoformat(),
                pump_start_ts=None, pump_start_minute=None, missed_session=False,
            )
            resp["action"] = "off"
            resp["reason"] = f"KNN: {knn_result['label']} ({knn_result['confidence']}%) — pompa OFF"
            resp["decision_path"].append("A2-knn-off")
            return resp

        # A3. Label hujan — matikan pompa KECUALI sedang di window wajib siram
        # [FIX v11.4.0] Di window pagi/sore, penyiraman wajib tidak bisa
        # dihentikan oleh label hujan. Hujan di Ambon terlalu sering
        # muncul sebagai false positive.
        if knn_result["label"] in ("Hujan_Aktif", "Hujan_Prediksi"):
            if in_window:
                # Window wajib siram → abaikan label hujan, biarkan pompa jalan
                resp["reason"] = (
                    f"KNN: {knn_result['label']} — DIABAIKAN karena jam siram wajib {window_label}. "
                    f"Pompa tetap ON."
                )
                resp["decision_path"].append("A3-rain-ignored-mandatory-window")
                log.info("[A3] Label hujan '%s' diabaikan — window wajib %s.",
                         knn_result['label'], window_label)
                return resp
            else:
                # Di luar window → matikan pompa
                resp["pending_updates"].update(
                    pump_status=False, last_watered_minute=current_total_minutes,
                    last_watered_ts=datetime.now().isoformat(),
                    pump_start_ts=None, pump_start_minute=None, missed_session=False,
                )
                resp["action"] = "off"
                resp["reason"] = f"KNN: {knn_result['label']} — pompa OFF (di luar jam siram)"
                resp["decision_path"].append("A3-rain-off")
                return resp

        # Pompa masih jalan normal
        resp["reason"] = f"KNN: {knn_result['label']} ({knn_result['confidence']}%) — pompa jalan"
        resp["decision_path"].append("A4-running")
        return resp

    # ══════════════════════════════════════════════════════════════════════════
    # Pompa OFF — evaluasi apakah perlu dinyalakan
    # ══════════════════════════════════════════════════════════════════════════

    # B2. Darurat malam — tanah sangat kering di luar window
    if (not in_window and soil_moisture <= CFG.CRITICAL_DRY
            and knn_result["label"] not in ("Hujan_Aktif", "Hujan_Prediksi")):
        now_ts = datetime.now().isoformat()
        pump_u = dict(pump_status=True,
                      pump_start_minute=current_total_minutes,
                      pump_start_ts=now_ts)
        await _add_pump_on_updates(pump_u)
        resp["pending_updates"].update(pump_u)
        resp["action"] = "on"
        resp["reason"] = f"DARURAT: soil={soil_moisture:.1f}% <= {CFG.CRITICAL_DRY}%"
        resp["decision_path"].append("B2-emergency")
        return resp

    # B3. KNN tidak perlu siram
    # [FIX v11.4.0] Jika sedang di window wajib siram DAN tanah kering DAN
    # label hujan adalah false positive → paksa siram tetap jalan.
    if not knn_result["needs_watering"]:
        rain_label = knn_result["label"] in ("Hujan_Aktif", "Hujan_Prediksi")
        rain_score = knn_result.get("computed_features", {}).get("rain_score", 0.0)

        if in_window and rain_label and soil_moisture < 50.0 and rain_score < CFG.RAIN_SCORE_THRESHOLD:
            # Hujan false positive di window siram + tanah masih kering → lanjut ke B6
            log.info(
                "[B3] Rain false positive diabaikan di window %s. "
                "soil=%.1f%%, rain_score=%.1f < %.0f. Lanjut siram.",
                window_label, soil_moisture, rain_score, CFG.RAIN_SCORE_THRESHOLD
            )
            resp["decision_path"].append("B3-rain-false-positive-override")
            # Jatuh ke B6 di bawah
        else:
            resp["blocked_reason"] = (
                f"KNN: {knn_result['label']} ({knn_result['confidence']}%) — tidak perlu siram"
            )
            resp["decision_path"].append("B3-knn-no-water")
            return resp

    # B4. Confidence terlalu rendah
    min_conf = (CFG.KNN_CONFIDENCE_MIN_PRIORITY
                if knn_result["label"] == "Siram_Prioritas"
                else CFG.KNN_CONFIDENCE_MIN)
    if knn_result["confidence"] < min_conf:
        resp["blocked_reason"] = (
            f"KNN confidence {knn_result['confidence']}% < minimum {min_conf:.0f}%"
        )
        resp["decision_path"].append("B4-low-confidence")
        return resp

    # B5. Cooldown belum selesai
    elapsed_cd = _elapsed_minutes(current_total_minutes, state.get("last_watered_minute"))
    if elapsed_cd < CFG.COOLDOWN_MINUTES:
        resp["blocked_reason"] = (
            f"Cooldown: sisa {CFG.COOLDOWN_MINUTES - elapsed_cd} menit"
        )
        resp["decision_path"].append("B5-cooldown")
        return resp

    # B6. POMPA ON
    now_ts = datetime.now().isoformat()
    pump_u = dict(pump_status=True,
                  pump_start_minute=current_total_minutes,
                  pump_start_ts=now_ts)
    await _add_pump_on_updates(pump_u)
    resp["pending_updates"].update(pump_u)
    resp["action"] = "on"
    resp["reason"] = (
        f"KNN [{window_label or 'darurat'}]: label={knn_result['label']}, "
        f"conf={knn_result['confidence']}%, "
        f"rain={knn_result.get('computed_features', {}).get('rain_score', 0)}"
    )
    resp["decision_path"].append("B6-knn-final")
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# HELPER TEST KNN
# ══════════════════════════════════════════════════════════════════════════════
def _run_test_classify(req: TestKNNRequest) -> dict:
    knn = classify(
        soil=req.soil_moisture, temp=req.temperature,
        rh=req.air_humidity, hour=req.hour,
        soil_prev=req.soil_prev, temp_prev=req.temp_prev, rh_prev=req.rh_prev,
    )

    in_window, wlbl = _in_watering_window(req.hour)
    decision_path   = []
    aksi_pompa      = None
    alasan          = ""
    blokir          = None

    rain_label = knn["label"] in ("Hujan_Aktif", "Hujan_Prediksi")
    rain_score = knn.get("computed_features", {}).get("rain_score", 0.0)

    if not knn["needs_watering"]:
        # [FIX v11.4.0] Jika di window siram, tanah kering, rain false positive → tetap siram
        if (in_window and rain_label
                and req.soil_moisture < 50.0
                and rain_score < CFG.RAIN_SCORE_THRESHOLD):
            aksi_pompa = "on"
            alasan = (
                f"[OVERRIDE] Rain false positive (score={rain_score:.0f} < {CFG.RAIN_SCORE_THRESHOLD:.0f}) "
                f"di window {wlbl} — tetap siram"
            )
            decision_path.append("B3-rain-false-positive-override")
            decision_path.append("B6-knn-final")
        else:
            blokir = f"KNN: {knn['label']} ({knn['confidence']}%) — tidak perlu siram"
            decision_path.append("B3-knn-no-water")
    elif rain_label:
        if in_window:
            # Di window wajib → siram tetap jalan
            aksi_pompa = "on"
            alasan = (
                f"KNN: {knn['label']} — DIABAIKAN (window wajib {wlbl}). "
                f"Pompa ON."
            )
            decision_path.append("A3-rain-ignored-mandatory-window")
            decision_path.append("B6-knn-final")
        else:
            blokir = f"KNN: {knn['label']} — siram ditunda karena hujan (luar jadwal)"
            decision_path.append("B3-hujan-block")
    elif knn["confidence"] < (CFG.KNN_CONFIDENCE_MIN_PRIORITY
                               if knn["label"] == "Siram_Prioritas"
                               else CFG.KNN_CONFIDENCE_MIN):
        blokir = f"KNN confidence {knn['confidence']}% terlalu rendah"
        decision_path.append("B4-low-confidence")
    else:
        aksi_pompa = "on"
        alasan     = f"KNN [{wlbl or 'darurat'}]: {knn['label']} ({knn['confidence']}%) → POMPA ON"
        decision_path.append("B6-knn-final")

    label_icon = {
        "Siram_Segera"   : "✅ POMPA ON  — Siram sekarang!",
        "Siram_Prioritas": "🔴 POMPA ON  — DARURAT, siram segera!",
        "Siram_Nanti"    : "⏳ TAHAN      — Kering tapi luar jadwal",
        "Optimal"        : "⏸️  TAHAN      — Tanah sudah cukup lembab",
        "Basah"          : "❌ TAHAN      — Tanah terlalu basah",
        "Hujan_Aktif"    : "🌧️  TAHAN      — Hujan sedang terjadi",
        "Hujan_Prediksi" : "🌦️  TAHAN      — Akan segera hujan",
    }.get(knn["label"], "❓ Tidak dikenal")

    cocok = (knn["label"] == req.ekspektasi) if req.ekspektasi else None

    return {
        "skenario"        : req.label_skenario or "Manual",
        "input"           : {
            "soil_moisture": req.soil_moisture, "temperature": req.temperature,
            "air_humidity" : req.air_humidity,  "hour": req.hour,
            "soil_prev"    : req.soil_prev,     "temp_prev": req.temp_prev,
            "rh_prev"      : req.rh_prev,
        },
        "knn_result"      : knn,
        "keputusan"       : {
            "label"        : knn["label"],
            "label_icon"   : label_icon,
            "aksi_pompa"   : aksi_pompa or "off/tahan",
            "pompa_nyala"  : aksi_pompa == "on",
            "alasan"       : alasan or blokir,
            "decision_path": decision_path,
        },
        "validasi"        : {
            "ekspektasi": req.ekspektasi, "hasil": knn["label"],
            "cocok"     : cocok,
            "status"    : ("✅ BENAR" if cocok else "❌ MELESET") if cocok is not None else "—",
        } if req.ekspektasi else None,
        "jam_siram_aktif" : in_window,
        "window_label"    : wlbl or "di luar jam siram",
        "computed_features": knn["computed_features"],
        # [FIX] Tambahkan info debug rain_score
        "rain_debug"      : {
            "rain_score"     : rain_score,
            "threshold"      : CFG.RAIN_SCORE_THRESHOLD,
            "is_credible"    : rain_score >= CFG.RAIN_SCORE_THRESHOLD,
            "label_is_rain"  : rain_label,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# SKENARIO PRESET
# ══════════════════════════════════════════════════════════════════════════════
SKENARIO_PRESET = [
    {"id":"S01","nama":"Pagi — tanah kering, tidak hujan","ekspektasi":"Siram_Segera",
     "keterangan":"Kondisi ideal siram pagi.",
     "data":{"soil_moisture":22,"temperature":29,"air_humidity":55,"hour":6,"soil_prev":24,"temp_prev":29,"rh_prev":54}},
    {"id":"S02","nama":"Sore — tanah kering, cuaca cerah","ekspektasi":"Siram_Segera",
     "keterangan":"Jam sore, tanah kering, siram normal.",
     "data":{"soil_moisture":25,"temperature":31,"air_humidity":58,"hour":17,"soil_prev":27,"temp_prev":32,"rh_prev":57}},
    {"id":"S03","nama":"Pagi — DARURAT sangat kering + panas ekstrem","ekspektasi":"Siram_Prioritas",
     "keterangan":"Tanah sangat kering + suhu >34°C.",
     "data":{"soil_moisture":14,"temperature":38,"air_humidity":35,"hour":6,"soil_prev":18,"temp_prev":39,"rh_prev":33}},
    {"id":"S04","nama":"Siang — kering tapi luar jam siram","ekspektasi":"Siram_Nanti",
     "keterangan":"Jam 13:00, bukan waktu siram.",
     "data":{"soil_moisture":25,"temperature":33,"air_humidity":52,"hour":13,"soil_prev":27,"temp_prev":33,"rh_prev":51}},
    {"id":"S05","nama":"Tengah malam — kering, luar jadwal","ekspektasi":"Siram_Nanti",
     "keterangan":"Jam 02:00, tidak boleh siram malam.",
     "data":{"soil_moisture":28,"temperature":26,"air_humidity":65,"hour":2,"soil_prev":29,"temp_prev":27,"rh_prev":64}},
    {"id":"S06","nama":"Tanah optimal — tidak perlu siram","ekspektasi":"Optimal",
     "keterangan":"Kelembaban 55%, zona ideal.",
     "data":{"soil_moisture":55,"temperature":27,"air_humidity":65,"hour":6,"soil_prev":55,"temp_prev":27,"rh_prev":65}},
    {"id":"S07","nama":"Tanah terlalu basah","ekspektasi":"Basah",
     "keterangan":"Tanah 83%, jenuh air.",
     "data":{"soil_moisture":83,"temperature":22,"air_humidity":88,"hour":17,"soil_prev":80,"temp_prev":22,"rh_prev":87}},
    {"id":"S08","nama":"Pagi — kering TAPI hujan deras aktif (RH 98%, tanah naik +11%)","ekspektasi":"Hujan_Aktif",
     "keterangan":"RH=98% dan tanah naik drastis — rain_score tinggi, ini hujan nyata.",
     "data":{"soil_moisture":25,"temperature":22,"air_humidity":98,"hour":6,"soil_prev":14,"temp_prev":30,"rh_prev":68}},
    {"id":"S09","nama":"Sore — RH naik, tanda akan hujan (tanah naik +6%)","ekspektasi":"Hujan_Prediksi",
     "keterangan":"RH naik dari 63% ke 88%, suhu turun 6C, tanah naik — baru dipercaya.",
     "data":{"soil_moisture":35,"temperature":24,"air_humidity":88,"hour":17,"soil_prev":29,"temp_prev":30,"rh_prev":63}},
    {"id":"S10","nama":"Sore panas — ET tinggi","ekspektasi":"Siram_Prioritas",
     "keterangan":"Suhu 36°C, tanah 22%, penguapan tinggi.",
     "data":{"soil_moisture":22,"temperature":36,"air_humidity":38,"hour":17,"soil_prev":27,"temp_prev":37,"rh_prev":36}},
    {"id":"S11","nama":"Malam — kering, luar jadwal","ekspektasi":"Siram_Nanti",
     "keterangan":"Jam 23:00, tunggu pagi.",
     "data":{"soil_moisture":18,"temperature":24,"air_humidity":60,"hour":23,"soil_prev":20,"temp_prev":25,"rh_prev":59}},
    # [FIX v11.4.0] S12 diperbarui: RH 95% saja tidak cukup tanpa soil naik drastis
    {"id":"S12","nama":"Pagi — RH tinggi TAPI soil kering (Ambon normal)","ekspektasi":"Siram_Segera",
     "keterangan":"RH 95% normal di Ambon. Soil kering 22% + soil tidak naik → bukan hujan → siram.",
     "data":{"soil_moisture":22,"temperature":27,"air_humidity":95,"hour":6,"soil_prev":23,"temp_prev":28,"rh_prev":93}},
]


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS PENGUJIAN
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/test-knn", tags=["🧪 Pengujian KNN"],
          summary="Uji KNN dengan data sensor bebas (tidak sentuh Supabase)")
async def test_knn(req: TestKNNRequest):
    return _run_test_classify(req)


@app.post("/test-knn/batch", tags=["🧪 Pengujian KNN"],
          summary="Uji banyak skenario sekaligus (maks 50)")
async def test_knn_batch(req: TestKNNBatchRequest):
    hasil_list, benar, total_exp = [], 0, 0
    for item in req.skenario:
        hasil = _run_test_classify(item)
        hasil_list.append(hasil)
        if hasil.get("validasi"):
            total_exp += 1
            if hasil["validasi"].get("cocok"):
                benar += 1
    akurasi = round(benar / total_exp * 100, 1) if total_exp else None
    return {
        "total_skenario": len(hasil_list),
        "total_dengan_ekspektasi": total_exp,
        "benar": benar, "meleset": total_exp - benar,
        "akurasi_batch": f"{akurasi}%" if akurasi is not None else "N/A",
        "ringkasan_label": {
            lbl: sum(1 for h in hasil_list if h["knn_result"]["label"] == lbl)
            for lbl in ["Siram_Segera","Siram_Prioritas","Siram_Nanti",
                        "Optimal","Basah","Hujan_Aktif","Hujan_Prediksi"]
        },
        "hasil": hasil_list,
    }


@app.get("/test-knn/skenario", tags=["🧪 Pengujian KNN"],
         summary="Ambil 12 skenario preset siap pakai")
async def get_skenario_preset():
    hasil_list = []
    for s in SKENARIO_PRESET:
        req  = TestKNNRequest(**s["data"], label_skenario=f"{s['id']} — {s['nama']}",
                               ekspektasi=s["ekspektasi"])
        hasil = _run_test_classify(req)
        hasil["keterangan"] = s["keterangan"]
        hasil_list.append(hasil)
    benar = sum(1 for h in hasil_list if h.get("validasi") and h["validasi"].get("cocok"))
    return {
        "info": "12 skenario preset untuk menguji semua kondisi KNN",
        "total": len(hasil_list), "benar": benar,
        "akurasi_preset": f"{round(benar/len(hasil_list)*100,1)}%",
        "skenario": hasil_list,
    }


@app.post("/test-knn/reset", tags=["🧪 Pengujian KNN"],
          summary="Reset state cooldown + OVR untuk pengujian ulang",
          dependencies=[Depends(verify_api_key)])
async def reset_test_state():
    loop = asyncio.get_event_loop()
    reset_fields = {
        "last_watered_minute": None, "last_watered_ts": None,
        "pump_status": False, "pump_start_ts": None, "pump_start_minute": None,
        "manual_override": False, "manual_override_ts": None,
        "session_count_today": 0, "last_sensor_ts": None, "last_sensor_soil": None,
        "last_soil_moisture": None, "last_temperature": None,
        "last_air_humidity": None, "last_rain_score": None, "missed_session": False,
    }
    try:
        await loop.run_in_executor(_executor, lambda: _sb_update_state_sync(**reset_fields))
        _rt_cache["data"] = None
        _rt_cache["timestamp"] = 0.0
        return {
            "success": True,
            "message": "State + OVR berhasil direset. Siap pengujian ulang.",
            "ovr_timeout_seconds": CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS,
            "rain_score_threshold": CFG.RAIN_SCORE_THRESHOLD,
            "reset_fields": list(reset_fields.keys()),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Gagal reset state: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS PRODUKSI
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/", tags=["Sistem"])
def root():
    return {
        "status": "online", "version": APP_VERSION,
        "model_ready": knn_model is not None,
        "ovr_timeout_seconds": CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS,
        "rain_score_threshold": CFG.RAIN_SCORE_THRESHOLD,
        "fix_notes": {
            "v11.4.0": [
                "rain_score threshold dinaikkan untuk iklim lembab tropis Ambon",
                "RH >= 97 → +40 poin (sebelumnya RH >= 92 → +50 poin)",
                "rain_score < 60 → label hujan diabaikan (override ke label berbasis soil)",
                "window wajib pagi/sore: penyiraman tetap jalan walau rain_score tinggi",
                "skenario S12 diperbarui: RH 95% tanpa kenaikan soil = SIRAM bukan hujan",
            ]
        },
        "endpoints_test": {
            "GET  /test-knn/skenario": "12 skenario preset siap uji",
            "POST /test-knn"         : "Uji 1 skenario bebas",
            "POST /test-knn/batch"   : "Uji banyak skenario sekaligus",
            "POST /test-knn/reset"   : "Reset OVR + cooldown",
        },
        "model_info": {
            "version"   : model_meta.get("version"),
            "algorithm" : model_meta.get("algorithm"),
            "best_k"    : model_meta.get("best_k"),
            "accuracy"  : f"{float(model_meta.get('accuracy', 0)) * 100:.2f}%",
            "features"  : model_meta.get("features"),
            "n_features": model_meta.get("n_features"),
            "labels"    : model_meta.get("labels"),
        } if model_meta else None,
    }


@app.post("/sensor", tags=["Produksi"], dependencies=[Depends(verify_api_key)])
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
                "received": True, "timestamp": state.get("last_updated") or timestamp,
                "device_time": f"{hour:02d}:{minute:02d}", "time_source": time_source,
                "debounced": True,
                "sensor": {"soil_moisture": data.soil_moisture,
                           "temperature": data.temperature, "air_humidity": data.air_humidity},
                "classification": None, "pump_status": state["pump_status"],
                "pump_action": None, "mode": state["mode"], "auto_info": None,
            }

    await _maybe_schedule_prune(bg_tasks)

    knn_result = classify(
        soil=data.soil_moisture, temp=data.temperature, rh=data.air_humidity, hour=hour,
        soil_prev=state.get("last_soil_moisture"), temp_prev=state.get("last_temperature"),
        rh_prev=state.get("last_air_humidity"), rain_score_prev=state.get("last_rain_score"),
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
        last_label=knn_result["label"], last_updated=timestamp,
        last_soil_moisture=data.soil_moisture, last_temperature=data.temperature,
        last_air_humidity=data.air_humidity,
        last_rain_score=knn_result["computed_features"]["rain_score"],
        last_sensor_ts=datetime.now().isoformat(), last_sensor_soil=data.soil_moisture,
    )
    pending     = smart_eval.get("pending_updates", {})
    all_updates = {**sensor_updates, **pending}

    optimistic = {**(_rt_cache["data"] or {}), **all_updates}
    _rt_cache["data"]      = _normalize_state(optimistic)
    _rt_cache["timestamp"] = time.monotonic()

    sensor_row = {
        "id": row_id, "timestamp": datetime.now().isoformat(),
        "soil_moisture": data.soil_moisture, "temperature": data.temperature,
        "air_humidity": data.air_humidity, "label": knn_result["label"],
        "confidence": knn_result["confidence"], "needs_watering": knn_result["needs_watering"],
        "description": knn_result.get("description", ""),
        "probabilities": knn_result["probabilities"],
        "computed_features": knn_result.get("computed_features", {}),
        "pump_status": pump_status_logged, "mode": state["mode"],
        "hour": hour, "minute": minute,
    }

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(_executor, lambda: _sb_update_state_sync(**all_updates))
        asyncio.ensure_future(
            loop.run_in_executor(_executor, lambda: _sb_insert_sensor_sync(sensor_row))
        )
    except Exception as e:
        log.error("State write gagal: %s", e)

    new_state = _get_state()
    return {
        "received": True, "timestamp": timestamp,
        "device_time": f"{hour:02d}:{minute:02d}", "time_source": time_source,
        "debounced": False,
        "sensor": {"soil_moisture": data.soil_moisture, "temperature": data.temperature,
                   "air_humidity": data.air_humidity},
        "classification": knn_result,
        "pump_status": new_state["pump_status"],
        "pump_action": final_action, "mode": new_state["mode"],
        "auto_info": {
            "reason"             : smart_eval.get("reason", ""),
            "blocked_reason"     : smart_eval.get("blocked_reason"),
            "decision_path"      : smart_eval.get("decision_path", []),
            "knn_label"          : smart_eval.get("knn_label"),
            "knn_confidence"     : smart_eval.get("knn_confidence"),
            "knn_probabilities"  : smart_eval.get("knn_probabilities"),
            "knn_computed"       : smart_eval.get("knn_computed"),
            "manual_override"    : new_state.get("manual_override", False),
            "ovr_expire_seconds" : CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS,
            "rain_score_threshold": CFG.RAIN_SCORE_THRESHOLD,
        } if state["mode"] == "auto" else None,
    }


@app.get("/pump-status", tags=["Produksi"], dependencies=[Depends(verify_api_key)])
def get_pump_status():
    state = _get_state()

    ovr_active = state.get("manual_override", False)
    if ovr_active:
        age = _elapsed_seconds_real(state.get("manual_override_ts"))
        if age >= CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS:
            log.info("[OVR] Expired terdeteksi di /pump-status (%ds) — reset.", int(age))
            try:
                _sb_update_state_sync(manual_override=False, manual_override_ts=None)
            except Exception:
                pass
            _rt_cache["data"]      = None
            _rt_cache["timestamp"] = 0.0
            ovr_active = False

    return {
        "pump_status"    : state["pump_status"],
        "mode"           : state["mode"],
        "manual_override": ovr_active,
    }


@app.post("/control", tags=["Produksi"], dependencies=[Depends(verify_api_key)])
async def control_pump(cmd: ControlCommand):
    action = (cmd.action or "").lower().strip()
    if action not in ("on", "off"):
        raise HTTPException(status_code=400, detail="Action harus 'on' atau 'off'.")
    mode_str = (cmd.mode or "manual").lower().strip()
    if mode_str not in ("auto", "manual"):
        mode_str = "manual"
    loop = asyncio.get_event_loop()
    async with _control_lock:
        state   = await loop.run_in_executor(_executor, _sb_get_state_sync)
        pump_on = action == "on"
        now_ts  = datetime.now().isoformat()
        if state["pump_status"] == pump_on and state["mode"] == mode_str:
            return {
                "success": True, "debounced": True,
                "pump_status": state["pump_status"], "mode": state["mode"],
                "manual_override": state.get("manual_override", False),
                "timestamp": now_ts,
            }
        update_kwargs: dict = {"last_control_ts": now_ts, "mode": mode_str}
        if state["pump_status"] != pump_on:
            update_kwargs["pump_status"] = pump_on
            if not pump_on:
                cur_min = _total_minutes(*_resolve_time_wit(None, None, None)[:2])
                update_kwargs.update(
                    pump_start_ts=None, pump_start_minute=None,
                    last_watered_ts=now_ts, last_watered_minute=cur_min,
                    manual_override=True, manual_override_ts=now_ts,
                )
                log.info("[OVR] Aktif %ds dari /control OFF.", CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS)
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
            await loop.run_in_executor(_executor,
                                       lambda: _sb_update_state_sync(**update_kwargs))
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Gagal menyimpan: {e}")
        new_state = _normalize_state({**(_rt_cache["data"] or {}), **update_kwargs})
        _rt_cache["data"]      = new_state
        _rt_cache["timestamp"] = time.monotonic()
        return {
            "success": True, "debounced": False,
            "pump_status": new_state["pump_status"], "mode": new_state["mode"],
            "manual_override": new_state.get("manual_override", False),
            "ovr_expire_seconds": CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS,
            "watering_today": _daily_safety["watering_count"],
            "timestamp": now_ts,
        }


@app.get("/status", tags=["Produksi"], dependencies=[Depends(verify_api_key)])
async def get_status():
    state = _get_state()
    loop  = asyncio.get_event_loop()
    def _get_latest():
        try:
            res = (_get_supabase().table("sensor_readings").select("*")
                   .order("timestamp", desc=True).limit(1).execute())
            return res.data[0] if res.data else None
        except Exception as e:
            log.error("latest sensor: %s", e)
        return None
    latest = await loop.run_in_executor(_executor, _get_latest)
    async with _daily_safety_lock:
        watering_today = _daily_safety["watering_count"]

    ovr_age  = _elapsed_seconds_real(state.get("manual_override_ts"))
    ovr_sisa = max(0, int(CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS - ovr_age)) if state.get("manual_override") else 0

    return {
        "pump_status": state["pump_status"], "mode": state["mode"],
        "last_label": state["last_label"],
        "last_updated": str(state["last_updated"]) if state["last_updated"] else None,
        "manual_override": state.get("manual_override", False),
        "ovr_sisa_detik": ovr_sisa,
        "watering_today": watering_today,
        "last_watered_ts": str(state["last_watered_ts"]) if state.get("last_watered_ts") else None,
        "watering_windows": {
            "morning": f"{CFG.MORNING_WINDOW[0]:02d}:00–{CFG.MORNING_WINDOW[1]:02d}:59 WIT",
            "evening": f"{CFG.EVENING_WINDOW[0]:02d}:00–{CFG.EVENING_WINDOW[1]:02d}:59 WIT",
        },
        "knn_config": {
            "confidence_min"       : CFG.KNN_CONFIDENCE_MIN,
            "critical_dry"         : CFG.CRITICAL_DRY,
            "needs_watering"       : list(CFG.NEEDS_WATERING_LABELS),
            "ovr_expire_seconds"   : CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS,
            "rain_score_threshold" : CFG.RAIN_SCORE_THRESHOLD,
        },
        "model_info": {
            "algorithm": model_meta.get("algorithm"), "version": model_meta.get("version"),
            "best_k": model_meta.get("best_k"),
            "accuracy": f"{float(model_meta.get('accuracy', 0)) * 100:.2f}%",
            "features": model_meta.get("features"), "labels": model_meta.get("labels"),
        } if model_meta else None,
        "latest_data": latest,
    }


@app.get("/history", tags=["Produksi"], dependencies=[Depends(verify_api_key)])
async def get_history(
    limit    : int  = Query(default=50, ge=1, le=500),
    pump_only: bool = Query(default=False),
):
    loop = asyncio.get_event_loop()
    def _fetch():
        try:
            q = (_get_supabase().table("sensor_readings").select("*")
                 .order("timestamp", desc=True).limit(limit))
            if pump_only:
                q = q.eq("pump_status", True)
            res = q.execute()
            return sorted(res.data or [], key=lambda x: x.get("timestamp", ""))
        except Exception as e:
            log.error("History error: %s", e)
            return []
    records = await loop.run_in_executor(_executor, _fetch)
    return {"total": len(records), "pump_only": pump_only, "records": records}