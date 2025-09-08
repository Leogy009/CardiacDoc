# llm/prompt_builder.py
# ======================
"""
Builders for LLM input payload (features + history + style + confidence).
The logic follows the design doc section 5.1:
- JSON Schema fields and constraints
- History incorporation (references + summary)
- Confidence bundle (overall, sub-scores, metric-level)
- Style & personalization controls
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import os

from .schemas import INPUT_SCHEMA, validate_input

_ALLOWED = {
    "rhythm_status": {"normal","arrhythmic","unknown"},
    "vascular_status": {"normal","stiff","unknown"},
    "stress_level": {"low","moderate","high","unknown"},
    "anxiety_level": {"low","moderate","high","unknown"},
}

_ALIAS_MAP = {
    "rhythm_status": {
        "n/a": "unknown", "na": "unknown", "none": "unknown", "unknown": "unknown",
        "arrhythmia": "arrhythmic", "abnormal": "arrhythmic"
    },
    "vascular_status": {
        "poor": "stiff", "stiff": "stiff", "normal": "normal", "unknown": "unknown"
    },
    "stress_level": {
        "low": "low", "normal": "low",  # 如果你想把 normal 映射到 low，可按需调整
        "moderate": "moderate", "medium": "moderate",
        "high": "high", "severe": "high",
        "unknown": "unknown"
    },
    "anxiety_level": {
        "low": "low", "moderate": "moderate", "medium": "moderate", "high": "high",
        "unknown": "unknown"
    },
}

def _norm_cat(field: str, value) -> str:
    if value is None:
        return "unknown"
    v = str(value).strip().lower()
    v = _ALIAS_MAP.get(field, {}).get(v, v)
    if v not in _ALLOWED.get(field, {"unknown"}):
        return "unknown"
    return v

def _drop_none(obj):
    # recursively drop keys whose value is None
    if isinstance(obj, dict):
        return {k: _drop_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_none(x) for x in obj if x is not None]
    return obj

# Optional: import aggregator if present, else we degrade gracefully
try:
    from query_aggregator import create_connection, aggregate_hrv_metrics, query_hrv_rows
except Exception:  # pragma: no cover
    create_connection = None  # type: ignore
    aggregate_hrv_metrics = None  # type: ignore
    query_hrv_rows = None  # type: ignore

# ---------- Style helpers ----------
STYLE_TO_OUTPUT_LANGUAGE = {
    "concise_bilingual": "zh-CN",
    "detailed_cn": "zh-CN",
    "detailed_en": "en-US",
    "plain": "zh-CN",
    "doctor_note": "zh-CN",
}

def _style_used(top_response_style: Optional[str], controls_style: Optional[str]) -> str:
    # prefer top-level response_style if provided, else use controls.style
    if top_response_style:
        if top_response_style == "formal":
            return "detailed_cn"
        if top_response_style == "casual":
            return "plain"
        return "concise_bilingual"  # balanced
    return controls_style or "concise_bilingual"

# ---------- Confidence heuristic ----------
def compute_confidence_bundle(
    measures: Dict[str, Any],
    history_rows: Optional[List[Dict[str, Any]]] = None,
    signal_quality: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Lightweight heuristics:
    - by_signal_quality: provided SQI or derived from missing/NaN rates (fallback = 0.8)
    - by_data_sufficiency: window_length >= 60 and enough history windows -> higher
    - by_model: placeholder 0.75 (can be replaced by calibrated external model)
    - overall: mean of available sub-scores
    - metric_confidence: 0.6~0.9 based on presence/recency and plausible ranges
    """
    # signal quality
    bsq = max(0.0, min(1.0, float(signal_quality))) if signal_quality is not None else 0.8

    # data sufficiency
    wl = float(measures.get("window_length") or 0.0)
    n_hist = len(history_rows or [])
    bds = 0.6
    if wl >= 60:
        bds += 0.2
    if n_hist >= 5:
        bds += 0.1
    bds = max(0.0, min(1.0, bds))

    # model
    bm = 0.75

    # metric-level (simple presence check)
    metric_conf: List[Dict[str, Any]] = []
    for m in ("rmssd","sdnn","pnn50","sd1_sd2_ratio","lf_hf_ratio","stress_index"):
        val = measures.get(m)
        score = 0.9 if isinstance(val, (int, float)) else 0.6
        metric_conf.append({"metric": m, "score": score})

    overall = round((bsq + bds + bm) / 3.0, 3)
    band = "low" if overall < 0.5 else ("medium" if overall < 0.75 else "high")
    return {
        "overall": overall,
        "band": band,
        "by_signal_quality": round(bsq, 3),
        "by_data_sufficiency": round(bds, 3),
        "by_model": round(bm, 3),
        "confidence_source": "hybrid",
        "metric_confidence": metric_conf,
        "notes": "Heuristic confidence; replace with calibrated model when available."
    }

# ---------- History builders ----------
def build_history_refs(rows: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for r in rows[:limit]:
        refs.append({
            # id is not guaranteed; we use feature_key as main pointer in this project
            "feature_key": r.get("feature_key"),
            "window_start": _iso(r.get("window_start")),
            "window_end": _iso(r.get("window_end"))
        })
    return refs

def build_history_summary(
    rows: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    times = sorted([(r["window_start"], r["window_end"]) for r in rows], key=lambda x: x[0])
    start = _iso(times[0][0])
    end = _iso(times[-1][1])
    n = len(rows)
    # aggregates
    def _avg(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals)/len(vals), 3) if vals else None
    def _median(key: str) -> Optional[float]:
        vals = sorted([r[key] for r in rows if isinstance(r.get(key), (int, float))])
        if not vals:
            return None
        mid = len(vals)//2
        if len(vals) % 2 == 1:
            return float(vals[mid])
        return round((vals[mid-1] + vals[mid]) / 2.0, 3)

    return {
        "time_range_start": start,
        "time_range_end": end,
        "n_windows": n,
        "aggregates": {
            "rmssd_avg": _avg("rmssd"),
            "sdnn_avg": _avg("sdnn"),
            "lf_hf_ratio_median": _median("lf_hf_ratio"),
            "stress_index_avg": _avg("stress_index")
        }
    }

def _iso(t) -> str:
    # accept str/datetime; return ISO string (Z if naive)
    if isinstance(t, str):
        return t
    if isinstance(t, datetime):
        if t.tzinfo is None:
            return t.replace(tzinfo=timezone.utc).isoformat()
        return t.isoformat()
    return str(t)

# ---------- Main builder ----------
def build_llm_input_payload(
    *,
    interaction_id: str,
    patient_id: str,
    now_iso: str,
    measures: Dict[str, Any],
    symptom_description: str,
    response_style: str = "balanced",
    input_language: str = "zh-CN",
    output_language: Optional[str] = None,
    return_explanations: bool = True,
    history_rows: Optional[List[Dict[str, Any]]] = None,
    signal_quality: Optional[float] = None,
    video_semantics: Optional[Dict[str, Any]] = None,
    schema_version: str = "1.0.0",
    feature_vector_version: int = 1,
) -> Dict[str, Any]:
    """
    Assemble the input payload for LLM according to the schema.
    """
    # derive style_used (for later output)
    style_used = _style_used(response_style, None)
    out_lang = output_language or STYLE_TO_OUTPUT_LANGUAGE.get(style_used, input_language)

    current_block: Dict[str, Any] = {
        "feature_key": measures.get("feature_key"),
        "window_start": measures.get("window_start"),
        "window_end": measures.get("window_end"),
        "window_length": measures.get("window_length"),
        "env_label": measures.get("env_label"),
        "rmssd": measures.get("rmssd"),
        "sdnn": measures.get("sdnn"),
        "pnn50": measures.get("pnn50"),
        "sd1": measures.get("sd1"),
        "sd2": measures.get("sd2"),
        "sd1_sd2_ratio": measures.get("sd1_sd2_ratio"),
        "lf": measures.get("lf"),
        "hf": measures.get("hf"),
        "lf_hf_ratio": measures.get("lf_hf_ratio"),
        "hti": measures.get("hti"),
        "rhythm_status": measures.get("rhythm_status"),
        "pwv": measures.get("pwv"),
        "aix": measures.get("aix"),
        "vascular_status": measures.get("vascular_status"),
        "stress_index": measures.get("stress_index"),
        "stress_level": measures.get("stress_level"),
        "anxiety_score": measures.get("anxiety_score"),
        "anxiety_level": measures.get("anxiety_level"),
    }

    # --- normalize categorical values to match schema enums ---
    if "rhythm_status" in current_block:
        current_block["rhythm_status"] = _norm_cat("rhythm_status", current_block["rhythm_status"])
    if "vascular_status" in current_block:
        current_block["vascular_status"] = _norm_cat("vascular_status", current_block["vascular_status"])
    if "stress_level" in current_block:
        current_block["stress_level"] = _norm_cat("stress_level", current_block["stress_level"])
    if "anxiety_level" in current_block:
        current_block["anxiety_level"] = _norm_cat("anxiety_level", current_block["anxiety_level"])

    # --- drop None fields to avoid number-vs-null schema conflicts ---
    current_block = _drop_none(current_block)

    # history
    refs = build_history_refs(history_rows or [], limit=8)
    summary = build_history_summary(history_rows or [])

    if summary and "aggregates" in summary:
        summary["aggregates"] = _drop_none(summary["aggregates"])


    payload: Dict[str, Any] = {
        "interaction_id": interaction_id,
        "session_id": measures.get("session_id"),
        "schema_version": schema_version,
        "feature_vector_version": feature_vector_version,
        "patient_id": patient_id,
        "timestamp": now_iso,
        "language": input_language,
        "response_style": response_style,
        "symptom_description": symptom_description,
        "sensor_features": {
            "current": current_block,
            "recent_history_refs": refs,
        },
        "controls": {
            "output_language": out_lang,
            "return_explanations": return_explanations
        }
    }
    if summary:
        payload["sensor_features"]["history_summary"] = summary
    if video_semantics:
        payload["video_semantics"] = video_semantics

    # attach confidence (we also place it in output later; here only for prompt context)
    payload["_confidence_hint"] = compute_confidence_bundle(measures, history_rows, signal_quality)

    payload = _drop_none(payload)
    validate_input(payload)

    # validate input strictly
    validate_input(payload)
    return payload
