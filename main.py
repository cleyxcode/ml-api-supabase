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

from fastapi import FastAPI, HTTPException, Query, Security, Depends, BackgroundTasks
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

# ── Supabase config (dari .env) ───────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")   # anon/publishable key

# ── API Key ───────────────────────────────────────────────────────────────────
VALID_API_KEY  = os.environ.get("API_KEY", "")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── Versi ─────────────────────────────────────────────────────────────────────
APP_VERSION = "9.0.0"
# ═══════════════════════════════════════════════════════════════════════════════
# v9.0.0 — Migrasi Firebase → Supabase PostgreSQL
# ═══════════════════════════════════════════════════════════════════════════════
# Perubahan utama:
#   - Firebase Admin SDK → supabase-py (create_client)
#   - /system_state (Firebase node) → tabel `system_state` (row id=1)
#   - /sensor_readings (Firebase node) → tabel `sensor_readings`
#   - Firebase real-time listener → Supabase Realtime (postgres_changes)
#   - _rt_cache tetap dipertahankan sebagai layer caching lokal
#   - Semua logika bisnis, KNN, rain detection, safety TIDAK BERUBAH
#   - Kredensial sensitif wajib di .env (SUPABASE_URL, SUPABASE_KEY, API_KEY)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Supabase client (singleton) ───────────────────────────────────────────────
_supabase: Client = None   # diinisialisasi di startup


def _get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        raise RuntimeError("Supabase belum diinisialisasi.")
    return _supabase


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if not VALID_API_KEY:
        log.warning("API_KEY belum di-set di environment variable!")
        return "no-key-configured"
    if api_key != VALID_API_KEY:
        log.warning("Akses ditolak: API key tidak valid '%s'", api_key)
        raise HTTPException(status_code=401, detail={
            "error"  : "Unauthorized",
            "message": "API key tidak valid atau tidak ada. Sertakan header: X-API-Key: <key>",
        })
    return api_key


# ── Locks ─────────────────────────────────────────────────────────────────────
_control_lock      = asyncio.Lock()
_daily_safety_lock = asyncio.Lock()
_daily_safety = {
    "date"                 : None,
    "watering_count"       : 0,
    "locked_out"           : False,
    "last_pump_duration_sec": 0,
    "prune_done_today"     : False,
}

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6, thread_name_prefix="sb-worker")


# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI
# ══════════════════════════════════════════════════════════════════════════════
class WateringConfig:
    MORNING_WINDOW = (5, 7)
    EVENING_WINDOW = (16, 18)

    SOIL_DRY_ON   = 45.0
    SOIL_WET_OFF  = 70.0
    CRITICAL_DRY  = 20.0

    RAIN_SCORE_THRESHOLD   = 60
    RAIN_RH_HEAVY          = 92.0
    RAIN_RH_MODERATE       = 85.0
    RAIN_RH_LIGHT          = 78.0
    RAIN_SOIL_RISE_HEAVY   = 8.0
    RAIN_SOIL_RISE_LIGHT   = 3.0
    RAIN_TEMP_DROP         = 3.0
    RAIN_CLEAR_THRESHOLD   = 30
    RAIN_CONFIRM_READINGS  = 2
    RAIN_CLEAR_READINGS    = 3

    COOLDOWN_MINUTES           = 45
    POST_RAIN_COOLDOWN_MINUTES = 120
    MIN_SESSION_GAP_MINUTES    = 10

    MAX_PUMP_DURATION_MINUTES = 5
    MIN_PUMP_DURATION_SECONDS = 30

    HOT_TEMP_THRESHOLD = 34.0

    CONFIDENCE_NORMAL = 60.0
    CONFIDENCE_HOT    = 40.0
    CONFIDENCE_MISSED = 48.0

    CONTROL_DEBOUNCE_SECONDS = 1
    SENSOR_DEBOUNCE_SECONDS  = 1
    SENSOR_TOLERANCE         = 0.5

    MANUAL_OVERRIDE_EXPIRE_SECONDS = 600

    TIME_WEIGHT_IN_WINDOW   = 1.0
    TIME_WEIGHT_NEAR_WINDOW = 0.7
    TIME_WEIGHT_OUTSIDE     = 0.0


CFG = WateringConfig()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Siram Pintar API — Supabase",
    description="Sistem Penyiraman Tanaman IoT — KNN + Supabase PostgreSQL (v9.0.0)",
    version=APP_VERSION,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

knn_model  = None
scaler     = None
model_meta: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# REAL-TIME STATE CACHE
# ══════════════════════════════════════════════════════════════════════════════
_STATE_DEFAULTS = {
    "pump_status"          : False,
    "mode"                 : "auto",
    "last_label"           : None,
    "last_updated"         : None,
    "pump_start_ts"        : None,
    "pump_start_minute"    : None,
    "last_watered_minute"  : None,
    "last_watered_ts"      : None,
    "last_soil_moisture"   : None,
    "last_temperature"     : None,
    "missed_session"       : False,
    "rain_detected"        : False,
    "rain_score"           : 0,
    "rain_confirm_count"   : 0,
    "rain_clear_count"     : 0,
    "rain_started_minute"  : None,
    "last_control_ts"      : None,
    "last_sensor_ts"       : None,
    "last_sensor_soil"     : None,
    "session_count_today"  : 0,
    "session_count_date"   : None,
    "manual_override"      : False,
    "manual_override_ts"   : None,
}

_rt_cache: dict = {"data": None, "timestamp": 0.0}
_realtime_socket = None        # placeholder agar /config & /diagnostics tidak error
_polling_task    = None        # asyncio.Task background polling


def _normalize_state(raw: dict) -> dict:
    row = dict(_STATE_DEFAULTS)
    row.update(raw)
    for k in ("pump_status", "missed_session", "rain_detected", "manual_override"):
        row[k] = bool(row.get(k, False))
    for k in ("rain_score", "rain_confirm_count", "rain_clear_count", "session_count_today"):
        row[k] = int(row.get(k) or 0)
    return row


# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE DB HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _sb_get_state_sync() -> dict:
    """Baca state dari tabel system_state (row id=1)."""
    try:
        res = _get_supabase().table("system_state").select("*").eq("id", 1).single().execute()
        if res.data:
            return _normalize_state(res.data)
    except Exception as e:
        log.error("Supabase get state error: %s", e)
    return dict(_STATE_DEFAULTS)


def _sb_update_state_sync(**kwargs):
    """
    Update field di system_state row id=1.
    Kolom TIMESTAMP di schema PostgreSQL: kirim string ISO-8601 atau None.
    Kolom DATE (session_count_date): kirim string 'YYYY-MM-DD'.
    """
    if not kwargs:
        return
    payload = {**kwargs, "id": 1}
    _get_supabase().table("system_state").upsert(payload).execute()
    log.info("State updated: %s", list(kwargs.keys()))


def _sb_insert_sensor_sync(row_data: dict):
    """Insert satu baris ke tabel sensor_readings."""
    _get_supabase().table("sensor_readings").insert(row_data).execute()


def _sb_ensure_state_row():
    """Pastikan baris id=1 ada di system_state. Schema sudah handle insert via SQL."""
    try:
        res = _get_supabase().table("system_state").select("id").eq("id", 1).execute()
        if not res.data:
            # Insert minimal — kolom dengan DEFAULT akan terisi otomatis
            _get_supabase().table("system_state").insert({"id": 1}).execute()
            log.info("system_state row id=1 dibuat di Supabase.")
        else:
            log.info("system_state row id=1 sudah ada.")
    except Exception as e:
        log.error("Gagal memastikan system_state row: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND POLLING — 0.5s interval untuk real-time IoT
# Cache diupdate optimistik langsung saat write (tidak tunggu poll berikutnya).
# Polling hanya sebagai sync fallback jika ada perubahan dari luar (dashboard, dll).
# ══════════════════════════════════════════════════════════════════════════════

async def _state_polling_loop():
    """Coroutine background: poll system_state setiap 0.5 detik, update cache."""
    log.info("Background state-polling dimulai (interval 0.5s).")
    while True:
        try:
            loop = asyncio.get_event_loop()
            row  = await loop.run_in_executor(_executor, _sb_get_state_sync)
            _rt_cache["data"]      = row
            _rt_cache["timestamp"] = time.monotonic()
        except asyncio.CancelledError:
            log.info("State-polling dihentikan.")
            return
        except Exception as e:
            log.warning("Polling error (akan retry): %s", e)
        await asyncio.sleep(0.5)


def _start_supabase_listener():
    """Kompatibilitas: tidak melakukan apa-apa (polling dimulai di startup)."""
    pass


def _stop_supabase_listener():
    """Kompatibilitas: tidak melakukan apa-apa (task dibatalkan di shutdown)."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN
# ══════════════════════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup():
    global _supabase, knn_model, scaler, model_meta
    log.info("Siram Pintar API v%s (Supabase) starting...", APP_VERSION)

    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL / SUPABASE_KEY belum di-set di .env!")
        raise RuntimeError("Supabase credentials missing.")

    _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    log.info("Supabase client terhubung: %s", SUPABASE_URL)

    if VALID_API_KEY:
        log.info("API Key protection: AKTIF (key: %s***)", VALID_API_KEY[:2])

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _sb_ensure_state_row)

    # Mulai background polling (menggantikan Realtime WebSocket)
    global _polling_task
    _polling_task = asyncio.create_task(_state_polling_loop())

    # Beri waktu polling pertama selesai sebelum lanjut
    await asyncio.sleep(2.2)

    await _sync_daily_safety_from_db()

    if not os.path.exists(MODEL_PATH):
        log.warning("Model belum ada! Jalankan train_model.py terlebih dahulu.")
        return
    try:
        knn_model = joblib.load(MODEL_PATH)
        scaler    = joblib.load(SCALER_PATH)
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                model_meta = json.load(f)
        log.info("Model KNN dimuat. K=%s, Akurasi=%s%%",
                 model_meta.get("best_k"), model_meta.get("accuracy"))
    except Exception as exc:
        log.error("Gagal memuat model: %s", exc)


@app.on_event("shutdown")
async def shutdown():
    global _polling_task
    if _polling_task and not _polling_task.done():
        _polling_task.cancel()
        try:
            await _polling_task
        except asyncio.CancelledError:
            pass
    _executor.shutdown(wait=False)
    log.info("Siram Pintar API shutdown selesai.")


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
# HELPER: Waktu WIT
# ══════════════════════════════════════════════════════════════════════════════
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
        # PostgreSQL bisa kembalikan timezone-aware timestamp; normalisasi ke naive UTC
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


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: KNN time encoding
# ══════════════════════════════════════════════════════════════════════════════
def _encode_hour_cyclic(hour: int) -> tuple:
    angle = 2 * np.pi * hour / 24
    return float(np.sin(angle)), float(np.cos(angle))


def _get_time_weight(hour: int) -> float:
    in_window, _ = _in_watering_window(hour)
    if in_window:
        return CFG.TIME_WEIGHT_IN_WINDOW
    ms, me = CFG.MORNING_WINDOW[0], CFG.MORNING_WINDOW[1]
    es, ee = CFG.EVENING_WINDOW[0], CFG.EVENING_WINDOW[1]
    if hour == ms - 1 or hour == me + 1 or hour == es - 1 or hour == ee + 1:
        return CFG.TIME_WEIGHT_NEAR_WINDOW
    return CFG.TIME_WEIGHT_OUTSIDE


# ══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def _get_state(force_fresh: bool = False) -> dict:
    """
    Ambil state dari cache.
    Cache diperbarui oleh:
      1. Optimistic update langsung saat /sensor atau /control menulis
      2. Background polling 0.5s sebagai sync fallback
    TTL 0.1s hanya sebagai guard jika cache belum pernah terisi.
    """
    if force_fresh:
        row = _sb_get_state_sync()
        _rt_cache["data"]      = row
        _rt_cache["timestamp"] = time.monotonic()
        return row

    cached = _rt_cache["data"]
    age    = time.monotonic() - _rt_cache["timestamp"]

    if cached and age < 0.1:
        return cached.copy()

    # Cache kosong atau sangat stale → baca langsung
    try:
        row = _sb_get_state_sync()
        _rt_cache["data"]      = row
        _rt_cache["timestamp"] = time.monotonic()
        return row
    except Exception as e:
        log.error("Fallback get state gagal: %s", e)
        return cached.copy() if cached else dict(_STATE_DEFAULTS)


async def _update_state_async(**kwargs):
    """Update state async. Setelah write, cache di-force-refresh."""
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
# DAILY SAFETY
# ══════════════════════════════════════════════════════════════════════════════
async def _sync_daily_safety_from_db():
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
            _daily_safety["locked_out"]     = (db_count >= 10)
            log.info("_sync_daily_safety: recovered watering_count=%d", db_count)
        else:
            _daily_safety["date"]           = today
            _daily_safety["watering_count"] = 0
            _daily_safety["locked_out"]     = False
            log.info("_sync_daily_safety: hari baru, counter direset.")


def _daily_safety_reset_if_new_day():
    today = date.today()
    if _daily_safety["date"] != today:
        _daily_safety["date"]             = today
        _daily_safety["watering_count"]   = 0
        _daily_safety["locked_out"]       = False
        _daily_safety["prune_done_today"] = False
        return True
    return False


def _prune_sensor_readings():
    """Hapus sensor_readings > 14 hari. Filter pakai kolom TIMESTAMP."""
    try:
        cutoff = (datetime.now() - __import__("datetime").timedelta(days=14)).isoformat()
        _get_supabase().table("sensor_readings")\
            .delete()\
            .lt("timestamp", cutoff)\
            .execute()
        log.info("Pruned old sensor readings (>14 hari).")
    except Exception as e:
        log.error("Prune error: %s", e)


async def _maybe_schedule_prune(bg_tasks: BackgroundTasks):
    async with _daily_safety_lock:
        _daily_safety_reset_if_new_day()
        if not _daily_safety["prune_done_today"]:
            _daily_safety["prune_done_today"] = True
            bg_tasks.add_task(_prune_sensor_readings)


# ══════════════════════════════════════════════════════════════════════════════
# KNN Classify
# ══════════════════════════════════════════════════════════════════════════════
def classify(soil: float, temp: float, rh: float, hour: int = 12) -> dict:
    if knn_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model KNN belum dimuat.")
    try:
        feat  = scaler.transform(np.array([[soil, temp, rh]]))
        label = knn_model.predict(feat)[0]
        proba = knn_model.predict_proba(feat)[0]
        confs = {cls: round(float(p) * 100, 2) for cls, p in zip(knn_model.classes_, proba)}
        conf  = round(float(max(proba)) * 100, 2)
        tw    = _get_time_weight(hour)
        hs, hc = _encode_hour_cyclic(hour)
        return {
            "label"                   : label,
            "confidence"              : conf,
            "time_weight"             : tw,
            "time_adjusted_confidence": round(conf * tw, 2),
            "hour_sin"                : hs,
            "hour_cos"                : hc,
            "probabilities"           : confs,
            "needs_watering"          : label == "Kering",
            "description"             : model_meta.get("label_desc", {}).get(label, ""),
        }
    except Exception as e:
        log.error("KNN classify error: %s", e)
        raise HTTPException(status_code=503, detail="Model inference error")


# ══════════════════════════════════════════════════════════════════════════════
# RAIN DETECTION  (tidak berubah)
# ══════════════════════════════════════════════════════════════════════════════
def _compute_rain_score(air_humidity, soil_moisture, temperature,
                        last_soil, last_temp, pump_was_on):
    score, signals = 0, []
    if air_humidity >= CFG.RAIN_RH_HEAVY:
        score += 50; signals.append(f"RH={air_humidity:.0f}% (lebat)")
    elif air_humidity >= CFG.RAIN_RH_MODERATE:
        score += 30; signals.append(f"RH={air_humidity:.0f}% (sedang)")
    elif air_humidity >= CFG.RAIN_RH_LIGHT:
        score += 15; signals.append(f"RH={air_humidity:.0f}% (ringan)")

    if not pump_was_on and last_soil is not None:
        delta = soil_moisture - float(last_soil)
        if delta >= CFG.RAIN_SOIL_RISE_HEAVY:
            score += 35; signals.append(f"tanah +{delta:.1f}%")
        elif delta >= CFG.RAIN_SOIL_RISE_LIGHT:
            score += 20; signals.append(f"tanah +{delta:.1f}%")

    if last_temp is not None:
        drop = float(last_temp) - temperature
        if drop >= CFG.RAIN_TEMP_DROP:
            score += 15; signals.append(f"suhu turun -{drop:.1f}°C")

    return min(score, 100), signals


def _update_rain_state_batched(score, signals, state, current_min) -> tuple:
    currently = state["rain_detected"]
    confirm   = state["rain_confirm_count"]
    clear     = state["rain_clear_count"]
    updates   = {}

    if score >= CFG.RAIN_SCORE_THRESHOLD:
        confirm += 1; clear = 0
        if not currently and confirm >= CFG.RAIN_CONFIRM_READINGS:
            updates = dict(rain_detected=True, rain_score=score,
                           rain_confirm_count=confirm, rain_clear_count=0,
                           rain_started_minute=current_min, missed_session=True)
            return True, f"Hujan dikonfirmasi (skor={score})", updates
        elif currently:
            updates = dict(rain_score=score, rain_confirm_count=confirm, rain_clear_count=0)
            return True, f"Hujan berlanjut (skor={score})", updates
        else:
            updates = dict(rain_score=score, rain_confirm_count=confirm, rain_clear_count=0)
            return False, f"Menunggu konfirmasi ({confirm}/{CFG.RAIN_CONFIRM_READINGS})", updates

    elif score <= CFG.RAIN_CLEAR_THRESHOLD:
        clear += 1; confirm = 0
        if currently and clear >= CFG.RAIN_CLEAR_READINGS:
            updates = dict(rain_detected=False, rain_score=score,
                           rain_confirm_count=0, rain_clear_count=clear)
            return False, "", updates
        elif currently:
            updates = dict(rain_score=score, rain_confirm_count=0, rain_clear_count=clear)
            return True, "Hujan mungkin selesai, tunggu konfirmasi", updates
        else:
            updates = dict(rain_score=score, rain_confirm_count=0, rain_clear_count=clear)
            return False, "", updates
    else:
        if currently:
            updates = dict(rain_score=score)
            return True, f"Hujan ambiguos (skor={score})", updates
        return False, "", {}


def _should_skip_sensor(data: SensorData, state: dict, pump_is_on: bool) -> bool:
    if data.soil_moisture <= 0.0 or data.temperature <= 0.0 or data.temperature >= 60.0:
        log.warning("ANOMALI SENSOR: Soil=%.1f%% Temp=%.1f°C",
                    data.soil_moisture, data.temperature)
        return True

    last_soil = state.get("last_sensor_soil")
    if last_soil is not None and abs(data.soil_moisture - float(last_soil)) > 30.0:
        if not pump_is_on:
            log.warning("ANOMALI: Perubahan >30%% tanpa pompa (%.1f%% → %.1f%%)",
                        float(last_soil), data.soil_moisture)
            return True

    elapsed = _elapsed_seconds_real(state.get("last_sensor_ts"))

    if elapsed < CFG.SENSOR_DEBOUNCE_SECONDS:
        if last_soil is None:
            return False
        if abs(data.soil_moisture - float(last_soil)) <= CFG.SENSOR_TOLERANCE:
            return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# SMART WATERING ENGINE  (tidak berubah)
# ══════════════════════════════════════════════════════════════════════════════
async def _evaluate_smart_watering_async(result, hour, minute, soil_moisture,
                                         air_humidity, temperature, state,
                                         current_total_minutes) -> dict:
    resp = {
        "action"         : None,
        "reason"         : "",
        "blocked_reason" : None,
        "is_raining"     : False,
        "rain_score"     : 0,
        "hot_mode"       : temperature >= CFG.HOT_TEMP_THRESHOLD,
        "missed_session" : bool(state.get("missed_session", False)),
        "decision_path"  : [],
        "time_weight"    : result.get("time_weight", 1.0),
        "pending_updates": {},
    }

    async with _daily_safety_lock:
        _daily_safety_reset_if_new_day()
        if _daily_safety["locked_out"]:
            resp["blocked_reason"] = "Safety Lockout: Melebihi batas harian (10x)."
            resp["decision_path"].append("SAFETY_LOCKOUT")
            return resp

    if state.get("manual_override"):
        age = _elapsed_seconds_real(state.get("manual_override_ts"))
        if age < CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS:
            remaining = int(CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS - age)
            resp["blocked_reason"] = (
                f"Manual override aktif: pompa dikunci off ({remaining}s lagi)"
            )
            resp["decision_path"].append("MANUAL_OVERRIDE_BLOCK")
            return resp
        else:
            log.info("Manual override expired, reset otomatis.")
            resp["pending_updates"].update(manual_override=False, manual_override_ts=None)

    rain_score, rain_signals = _compute_rain_score(
        air_humidity=air_humidity, soil_moisture=soil_moisture,
        temperature=temperature, last_soil=state.get("last_soil_moisture"),
        last_temp=state.get("last_temperature"), pump_was_on=bool(state["pump_status"]),
    )
    is_raining, rain_reason, rain_updates = _update_rain_state_batched(
        rain_score, rain_signals, state, current_total_minutes
    )
    resp["pending_updates"].update(rain_updates)
    resp["is_raining"] = is_raining
    resp["rain_score"] = rain_score

    dynamic_dry_on  = CFG.SOIL_DRY_ON
    dynamic_wet_off = CFG.SOIL_WET_OFF

    if resp["hot_mode"]:
        dynamic_dry_on  += 5.0; dynamic_wet_off += 5.0
        resp["decision_path"].append("T-HOT_ADJUST")
    elif temperature < 25.0 and air_humidity > 80.0:
        dynamic_dry_on  -= 5.0; dynamic_wet_off -= 5.0
        resp["decision_path"].append("T-COOL_ADJUST")

    if state.get("missed_session"):
        dynamic_wet_off += 5.0
        resp["decision_path"].append("T-MISSED_ADJUST")

    dynamic_wet_off = min(95.0, dynamic_wet_off)
    dynamic_dry_on  = max(CFG.CRITICAL_DRY + 5.0, dynamic_dry_on)

    in_window, window_label = _in_watering_window(hour)
    night_emergency = not in_window and soil_moisture <= CFG.CRITICAL_DRY and not is_raining
    if night_emergency:
        window_label = "malam-darurat"

    # ── Pompa sedang ON ───────────────────────────────────────────────────────
    if state["pump_status"]:
        elapsed_sec = _elapsed_seconds_real(state.get("pump_start_ts"))
        max_sec     = 60 if night_emergency else (CFG.MAX_PUMP_DURATION_MINUTES * 60)

        if elapsed_sec >= max_sec:
            resp["pending_updates"].update(
                pump_status=False, last_watered_minute=current_total_minutes,
                last_watered_ts=datetime.now().isoformat(),
                pump_start_ts=None, pump_start_minute=None, missed_session=False,
            )
            resp["action"] = "off"
            resp["reason"] = f"Auto-stop: {elapsed_sec:.0f}s"
            resp["decision_path"].append("A1")
            return resp

        if elapsed_sec < CFG.MIN_PUMP_DURATION_SECONDS:
            resp["reason"] = f"Warmup ({elapsed_sec:.0f}s)"
            resp["decision_path"].append("A-warmup")
            return resp

        if soil_moisture >= dynamic_wet_off:
            resp["pending_updates"].update(
                pump_status=False, last_watered_minute=current_total_minutes,
                last_watered_ts=datetime.now().isoformat(),
                pump_start_ts=None, pump_start_minute=None, missed_session=False,
            )
            resp["action"] = "off"
            resp["reason"] = f"Tanah cukup ({soil_moisture:.1f}%)"
            resp["decision_path"].append("A2")
            return resp

        if is_raining:
            resp["pending_updates"].update(
                pump_status=False, last_watered_minute=current_total_minutes,
                last_watered_ts=datetime.now().isoformat(),
                pump_start_ts=None, pump_start_minute=None, missed_session=False,
            )
            resp["action"] = "off"
            resp["reason"] = "Hujan terdeteksi"
            resp["decision_path"].append("A3")
            return resp

        resp["reason"] = f"Running ({elapsed_sec:.0f}s)"
        resp["decision_path"].append("A4-running")
        return resp

    # ── Helper tambah sesi ────────────────────────────────────────────────────
    async def _add_pump_on_updates(updates: dict):
        async with _daily_safety_lock:
            _daily_safety["watering_count"] += 1
            new_count = _daily_safety["watering_count"]
            if new_count >= 10:
                _daily_safety["locked_out"] = True
        updates["session_count_today"] = new_count
        updates["session_count_date"]  = date.today().isoformat()

    # ── Darurat ───────────────────────────────────────────────────────────────
    if night_emergency or (soil_moisture <= CFG.CRITICAL_DRY and not is_raining):
        now_ts = datetime.now().isoformat()
        pump_u = dict(pump_status=True, pump_start_minute=current_total_minutes,
                      pump_start_ts=now_ts)
        await _add_pump_on_updates(pump_u)
        resp["pending_updates"].update(pump_u)
        resp["action"] = "on"
        resp["reason"] = f"SIRAM DARURAT [{window_label}]: tanah {soil_moisture:.1f}%"
        resp["decision_path"].append("B1")
        return resp

    if not in_window:
        resp["blocked_reason"] = f"Di luar jam aman ({hour:02d}:{minute:02d})"
        resp["decision_path"].append("B2")
        return resp

    if is_raining:
        resp["blocked_reason"] = f"Hujan terdeteksi (skor {rain_score})"
        resp["decision_path"].append("B3")
        return resp

    if soil_moisture >= dynamic_wet_off:
        if state.get("missed_session"):
            resp["pending_updates"]["missed_session"] = False
        resp["blocked_reason"] = f"Tanah sudah basah ({soil_moisture:.1f}%)"
        resp["decision_path"].append("B4")
        return resp

    effective_cd = (CFG.POST_RAIN_COOLDOWN_MINUTES if state.get("missed_session")
                    else CFG.COOLDOWN_MINUTES)
    elapsed_cd = _elapsed_minutes(current_total_minutes, state.get("last_watered_minute"))
    if elapsed_cd < effective_cd:
        resp["blocked_reason"] = f"Cooldown: sisa {effective_cd - elapsed_cd} mnt"
        resp["decision_path"].append("B5")
        return resp

    if not result["needs_watering"]:
        resp["blocked_reason"] = f"KNN: {result['label']} ({result['confidence']}%)"
        resp["decision_path"].append("B6")
        return resp

    base_thr = (CFG.CONFIDENCE_HOT if resp["hot_mode"]
                else (CFG.CONFIDENCE_MISSED if state.get("missed_session")
                      else CFG.CONFIDENCE_NORMAL))
    tw = result.get("time_weight", 1.0)
    eff_thr = min(base_thr * (1.0 / tw), 95.0) if 0.0 < tw < 1.0 else base_thr
    if 0.0 < tw < 1.0:
        resp["decision_path"].append(f"T-TIME_ADJ({tw:.1f})")

    if result["confidence"] < eff_thr:
        resp["blocked_reason"] = (
            f"Confidence {result['confidence']}% < threshold {eff_thr:.0f}%"
            f" (time_weight={tw:.1f})"
        )
        resp["decision_path"].append("B7")
        return resp

    if soil_moisture > dynamic_dry_on:
        resp["blocked_reason"] = f"Tanah {soil_moisture:.1f}% > batas ({dynamic_dry_on:.1f}%)"
        resp["decision_path"].append("B8")
        return resp

    now_ts = datetime.now().isoformat()
    pump_u = dict(pump_status=True, pump_start_minute=current_total_minutes,
                  pump_start_ts=now_ts)
    await _add_pump_on_updates(pump_u)
    resp["pending_updates"].update(pump_u)
    resp["action"] = "on"
    resp["reason"] = (
        f"Siram [{window_label}]: KNN {result['label']} ({result['confidence']}%), "
        f"T={temperature:.1f}°C, time_weight={tw:.1f}"
    )
    resp["decision_path"].append("B-FINAL")
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS PUBLIC
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {
        "status"      : "online",
        "message"     : "Siram Pintar API berjalan (Supabase)",
        "version"     : APP_VERSION,
        "model_ready" : knn_model is not None,
        "auth"        : "required" if VALID_API_KEY else "disabled",
        "database"    : "Supabase PostgreSQL + Realtime",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS PROTECTED
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/db-test", dependencies=[Depends(verify_api_key)])
async def db_test():
    """Cek koneksi Supabase dengan membaca system_state."""
    loop = asyncio.get_event_loop()
    try:
        state = await loop.run_in_executor(_executor, _sb_get_state_sync)
        return {
            "db_status"   : "connected",
            "supabase_url": SUPABASE_URL,
            "pump_status" : state["pump_status"],
            "mode"        : state["mode"],
        }
    except Exception as e:
        return {"db_status": "error", "detail": str(e)}


@app.get("/model-info", dependencies=[Depends(verify_api_key)])
def model_info():
    if not model_meta:
        raise HTTPException(status_code=503, detail="Model belum dimuat.")
    return model_meta


@app.post("/sensor", dependencies=[Depends(verify_api_key)])
async def receive_sensor(data: SensorData, bg_tasks: BackgroundTasks):
    hour, minute, _day, time_source = _resolve_time_wit(data.hour, data.minute, data.day)
    result    = classify(data.soil_moisture, data.temperature, data.air_humidity, hour=hour)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_id    = str(uuid.uuid4())

    state             = _get_state()
    current_total_min = _total_minutes(hour, minute)

    await _maybe_schedule_prune(bg_tasks)

    pump_is_on = bool(state.get("pump_status", False))
    skip_eval  = _should_skip_sensor(data, state, pump_is_on)

    if skip_eval:
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
                "classification": result,
                "pump_status"   : state["pump_status"],
                "pump_action"   : None,
                "mode"          : state["mode"],
                "auto_info"     : None,
            }

    final_action = None
    smart_eval   = {}

    if state["mode"] == "auto" and not skip_eval:
        smart_eval = await _evaluate_smart_watering_async(
            result=result, hour=hour, minute=minute,
            soil_moisture=data.soil_moisture, air_humidity=data.air_humidity,
            temperature=data.temperature, state=state,
            current_total_minutes=current_total_min,
        )
        final_action = smart_eval.get("action")

    pump_status_logged = (
        (final_action == "on") if final_action is not None else state["pump_status"]
    )
    sensor_updates = dict(
        last_label=result["label"], last_updated=timestamp,
        last_soil_moisture=data.soil_moisture, last_temperature=data.temperature,
        last_sensor_ts=datetime.now().isoformat(), last_sensor_soil=data.soil_moisture,
    )
    pending     = smart_eval.get("pending_updates", {})
    all_updates = {**sensor_updates, **pending}

    # ── Optimistic cache update (instan, tanpa tunggu Supabase) ──────────────
    # Cache diperbarui langsung dengan data yang akan ditulis,
    # sehingga /pump-status dan polling berikutnya sudah dapat nilai terbaru
    # bahkan sebelum round-trip ke Supabase selesai.
    optimistic = {**(_rt_cache["data"] or {}), **all_updates}
    _rt_cache["data"]      = _normalize_state(optimistic)
    _rt_cache["timestamp"] = time.monotonic()

    loop = asyncio.get_event_loop()

    sensor_row = {
        "id"            : row_id,
        "timestamp"     : datetime.now().isoformat(),
        "soil_moisture" : data.soil_moisture,
        "temperature"   : data.temperature,
        "air_humidity"  : data.air_humidity,
        "label"         : result["label"],
        "confidence"    : result["confidence"],
        "needs_watering": result["needs_watering"],
        "description"   : result.get("description", ""),
        "probabilities" : result["probabilities"],
        "pump_status"   : pump_status_logged,
        "mode"          : state["mode"],
    }

    def _write_state():
        """Tulis state update ke Supabase — prioritas tinggi."""
        _sb_update_state_sync(**all_updates)

    def _write_sensor():
        """Insert sensor reading — bisa sedikit terlambat, tidak blocking."""
        try:
            _sb_insert_sensor_sync(sensor_row)
        except Exception as e:
            log.error("Sensor insert gagal: %s", e)

    try:
        # Tulis state dulu (kritis untuk pompa), sensor insert paralel di background
        state_future   = loop.run_in_executor(_executor, _write_state)
        sensor_future  = loop.run_in_executor(_executor, _write_sensor)
        # Tunggu state selesai (pompa harus tersimpan), sensor boleh async
        await state_future
        # Sensor insert jalan di background — tidak blocking response
        asyncio.ensure_future(sensor_future)
    except Exception as e:
        log.error("State write gagal: %s", e)

    new_state = _get_state()
    return {
        "received"      : True,
        "timestamp"     : timestamp,
        "device_time"   : f"{hour:02d}:{minute:02d}",
        "time_source"   : time_source,
        "debounced"     : skip_eval,
        "sensor"        : {
            "soil_moisture": data.soil_moisture,
            "temperature"  : data.temperature,
            "air_humidity" : data.air_humidity,
        },
        "classification": result,
        "pump_status"   : new_state["pump_status"],
        "pump_action"   : final_action,
        "mode"          : new_state["mode"],
        "auto_info"     : {
            "is_raining"      : smart_eval.get("is_raining", False),
            "rain_score"      : smart_eval.get("rain_score", 0),
            "hot_mode"        : smart_eval.get("hot_mode", False),
            "missed_session"  : smart_eval.get("missed_session", False),
            "reason"          : smart_eval.get("reason", ""),
            "blocked_reason"  : smart_eval.get("blocked_reason"),
            "decision_path"   : smart_eval.get("decision_path", []),
            "time_weight"     : smart_eval.get("time_weight", 1.0),
            "manual_override" : new_state.get("manual_override", False),
        } if state["mode"] == "auto" else None,
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
            if res.data:
                return res.data[0]
        except Exception as e:
            log.error("Gagal ambil latest sensor: %s", e)
        return None

    latest = await loop.run_in_executor(_executor, _get_latest)

    async with _daily_safety_lock:
        watering_today = _daily_safety["watering_count"]
        locked_out     = _daily_safety["locked_out"]

    return {
        "pump_status"     : state["pump_status"],
        "mode"            : state["mode"],
        "last_label"      : state["last_label"],
        "last_updated"    : str(state["last_updated"]) if state["last_updated"] else None,
        "is_raining"      : state.get("rain_detected", False),
        "rain_score"      : state.get("rain_score", 0),
        "missed_session"  : state.get("missed_session", False),
        "manual_override" : state.get("manual_override", False),
        "watering_today"  : watering_today,
        "safety_locked"   : locked_out,
        "watering_windows": {
            "morning": f"{CFG.MORNING_WINDOW[0]:02d}:00–{CFG.MORNING_WINDOW[1]:02d}:59 WIT",
            "evening": f"{CFG.EVENING_WINDOW[0]:02d}:00–{CFG.EVENING_WINDOW[1]:02d}:59 WIT",
        },
        "thresholds": {
            "soil_dry_on" : CFG.SOIL_DRY_ON,
            "soil_wet_off": CFG.SOIL_WET_OFF,
            "critical_dry": CFG.CRITICAL_DRY,
        },
        "latest_data": latest,
    }


@app.get("/pump-status", dependencies=[Depends(verify_api_key)])
def get_pump_status():
    state = _get_state()
    return {
        "pump_status"    : state["pump_status"],
        "mode"           : state["mode"],
        "manual_override": state.get("manual_override", False),
        "version"        : APP_VERSION,
    }


@app.get("/history", dependencies=[Depends(verify_api_key)])
async def get_history(limit: int = Query(default=50, ge=1, le=500)):
    loop = asyncio.get_event_loop()

    def _fetch():
        try:
            res = (
                _get_supabase()
                .table("sensor_readings")
                .select("*")
                .order("timestamp", desc=True)
                .limit(limit)
                .execute()
            )
            records = res.data or []
            # Kembalikan urutan ascending (terlama → terbaru)
            return sorted(records, key=lambda x: x.get("timestamp", ""))
        except Exception as e:
            log.error("History error: %s", e)
            return []

    records = await loop.run_in_executor(_executor, _fetch)
    return {"total": len(records), "records": records}


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

        pump_changed = state["pump_status"] != pump_on
        mode_changed = state["mode"] != mode

        if not pump_changed and not mode_changed:
            return {
                "success"        : True,
                "debounced"      : True,
                "message"        : "Status tidak berubah",
                "pump_status"    : state["pump_status"],
                "mode"           : state["mode"],
                "manual_override": state.get("manual_override", False),
                "timestamp"      : state.get("last_control_ts") or now_ts,
            }

        update_kwargs: dict = {"last_control_ts": now_ts}

        if mode_changed:
            update_kwargs["mode"] = mode
            log.info("Mode berubah: %s → %s", state["mode"], mode)

        if pump_changed:
            update_kwargs["pump_status"] = pump_on

            if not pump_on:
                current_min = _total_minutes(*_resolve_time_wit(None, None, None)[:2])
                update_kwargs.update(
                    pump_start_ts=None,
                    pump_start_minute=None,
                    last_watered_ts=now_ts,
                    last_watered_minute=current_min,
                    manual_override=True,
                    manual_override_ts=now_ts,
                )
                log.info("Pompa OFF manual — manual_override diaktifkan.")
            else:
                now_utc = datetime.utcnow()
                h_wit   = (now_utc.hour + 9) % 24
                update_kwargs.update(
                    pump_start_ts=now_ts,
                    pump_start_minute=_total_minutes(h_wit, now_utc.minute),
                    manual_override=False,
                    manual_override_ts=None,
                )
                log.info("Pompa ON manual.")

        def _write_only():
            _sb_update_state_sync(**update_kwargs)

        try:
            await loop.run_in_executor(_executor, _write_only)
        except Exception as e:
            log.error("Control write gagal: %s", e)
            raise HTTPException(status_code=503, detail="Gagal menyimpan ke Supabase.")

        # Optimistic cache update — instan tanpa round-trip baca balik
        new_state = _normalize_state({**(_rt_cache["data"] or {}), **update_kwargs})
        _rt_cache["data"]      = new_state
        _rt_cache["timestamp"] = time.monotonic()

        log.info("Control OK: action=%s mode=%s → pump=%s mode=%s",
                 action, mode, new_state["pump_status"], new_state["mode"])

        return {
            "success"        : True,
            "debounced"      : False,
            "pump_status"    : new_state["pump_status"],
            "mode"           : new_state["mode"],
            "manual_override": new_state.get("manual_override", False),
            "timestamp"      : now_ts,
        }


@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(data: SensorData):
    hour, _, _, _ = _resolve_time_wit(data.hour, data.minute, data.day)
    return {
        "input" : {
            "soil_moisture": data.soil_moisture,
            "temperature"  : data.temperature,
            "air_humidity" : data.air_humidity,
            "hour"         : hour,
        },
        "result": classify(data.soil_moisture, data.temperature, data.air_humidity, hour=hour),
    }


@app.get("/config", dependencies=[Depends(verify_api_key)])
def get_config():
    return {
        "version"  : APP_VERSION,
        "database" : "Supabase PostgreSQL + Realtime",
        "supabase_url": SUPABASE_URL,
        "watering_windows": {
            "morning": f"{CFG.MORNING_WINDOW[0]:02d}:00–{CFG.MORNING_WINDOW[1]:02d}:59",
            "evening": f"{CFG.EVENING_WINDOW[0]:02d}:00–{CFG.EVENING_WINDOW[1]:02d}:59",
        },
        "soil_thresholds": {
            "dry_on_threshold"  : CFG.SOIL_DRY_ON,
            "wet_off_threshold" : CFG.SOIL_WET_OFF,
            "critical_emergency": CFG.CRITICAL_DRY,
        },
        "rain_detection": {
            "score_to_confirm": CFG.RAIN_SCORE_THRESHOLD,
            "score_to_clear"  : CFG.RAIN_CLEAR_THRESHOLD,
            "rh_heavy"        : CFG.RAIN_RH_HEAVY,
            "rh_moderate"     : CFG.RAIN_RH_MODERATE,
            "rh_light"        : CFG.RAIN_RH_LIGHT,
        },
        "pump_control": {
            "max_duration_min"      : CFG.MAX_PUMP_DURATION_MINUTES,
            "min_duration_sec"      : CFG.MIN_PUMP_DURATION_SECONDS,
            "cooldown_normal"       : CFG.COOLDOWN_MINUTES,
            "cooldown_post_rain"    : CFG.POST_RAIN_COOLDOWN_MINUTES,
            "manual_override_expire": CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS,
        },
        "knn_confidence": {
            "normal"        : CFG.CONFIDENCE_NORMAL,
            "hot_weather"   : CFG.CONFIDENCE_HOT,
            "missed_session": CFG.CONFIDENCE_MISSED,
            "hot_threshold" : CFG.HOT_TEMP_THRESHOLD,
        },
        "time_weights": {
            "in_window"  : CFG.TIME_WEIGHT_IN_WINDOW,
            "near_window": CFG.TIME_WEIGHT_NEAR_WINDOW,
            "outside"    : CFG.TIME_WEIGHT_OUTSIDE,
        },
        "realtime": {
            "listener_active"   : _polling_task is not None and not _polling_task.done(),
            "cache_ttl_fallback": "0.3s",
            "sensor_debounce"   : f"{CFG.SENSOR_DEBOUNCE_SECONDS}s",
            "sensor_tolerance"  : f"{CFG.SENSOR_TOLERANCE}%",
            "mode"              : "REST polling (2s interval)",
        },
    }


@app.post("/reset-rain", dependencies=[Depends(verify_api_key)])
async def reset_rain():
    await _update_state_async(
        rain_detected=False, rain_score=0, rain_confirm_count=0,
        rain_clear_count=0, rain_started_minute=None, missed_session=False,
    )
    return {"success": True, "message": "State hujan di-reset."}


@app.post("/reset-override", dependencies=[Depends(verify_api_key)])
async def reset_override():
    await _update_state_async(manual_override=False, manual_override_ts=None)
    return {"success": True, "message": "Manual override di-reset. Auto-watering aktif kembali."}


@app.get("/diagnostics", dependencies=[Depends(verify_api_key)])
async def get_diagnostics():
    state = _get_state(force_fresh=True)

    async with _daily_safety_lock:
        safety_snapshot = {
            k: str(v) if isinstance(v, date) else v
            for k, v in _daily_safety.items()
        }

    override_remaining = None
    if state.get("manual_override"):
        age = _elapsed_seconds_real(state.get("manual_override_ts"))
        override_remaining = max(0, int(CFG.MANUAL_OVERRIDE_EXPIRE_SECONDS - age))

    cache_age = round(time.monotonic() - _rt_cache["timestamp"], 3)

    return {
        "version"               : APP_VERSION,
        "server_time_wit"       : datetime.utcnow().strftime("%H:%M:%S") + " (UTC+9=WIT)",
        "state"                 : {k: str(v) if v is not None else None
                                   for k, v in state.items()},
        "daily_safety"          : safety_snapshot,
        "override_remaining_sec": override_remaining,
        "knn"                   : {
            "model_loaded" : knn_model is not None,
            "scaler_loaded": scaler is not None,
            "meta"         : model_meta,
        },
        "realtime_cache"        : {
            "listener_active": _polling_task is not None and not _polling_task.done(),
            "cache_age_sec"  : cache_age,
            "cache_valid"    : cache_age < 1.0,
            "mode"           : "REST polling (2s interval)",
        },
        "database": {
            "type"            : "Supabase PostgreSQL",
            "url"             : SUPABASE_URL,
            "state_table"     : "system_state",
            "readings_table"  : "sensor_readings",
        },
        "migrations_from": "v8.0.0 (Firebase RT listener) → v9.0.0 (Supabase PostgreSQL + Realtime)",
    }