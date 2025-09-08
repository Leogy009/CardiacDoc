# llm/schemas.py
# =================
"""
JSON Schemas and validators for the rPPG Doctor LLM interaction.
This module mirrors the spec in the design doc (section 5.1).
"""

from __future__ import annotations
from typing import Any, Dict
import json
try:
    import jsonschema
except ImportError:  # pragma: no cover
    raise ImportError("Please install jsonschema: `pip install jsonschema`")

# ---- Input JSON Schema (rPPG → LLM) ----
INPUT_SCHEMA: Dict[str, Any] = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.org/schemas/rppg-llm-input.json",
  "title": "RPPG Doctor - LLM Input",
  "type": "object",
  "required": ["interaction_id", "patient_id", "timestamp", "sensor_features"],
  "additionalProperties": False,
  "properties": {
    "interaction_id": {"type": "string", "format": "uuid",
      "description": "Global unique id for this LLM interaction (UUIDv4)."},
    "session_id": {"type": "string", "format": "uuid",
      "description": "Session id across multiple turns (optional)."},
    "schema_version": {"type": "string", "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$",
      "description": "Semantic version for this input schema."},
    "feature_vector_version": {"type": "integer", "minimum": 1,
      "description": "Version for feature concatenation ordering."},
    "patient_id": {"type": "string", "minLength": 1,
      "description": "De-identified patient id."},
    "timestamp": {"type": "string", "format": "date-time",
      "description": "ISO 8601 timestamp for current interaction."},
    "language": {"type": "string",
      "description": "Input language, e.g., zh-CN / en-US."},
    "response_style": {"type": "string",
      "enum": ["formal", "casual", "balanced"],
      "description": "Preferred response style (top-level wins if conflicting)."},
    "symptom_description": {"type": "string", "minLength": 1,
      "description": "User complaint / symptoms (text or ASR)."},
    "sensor_features": {
      "type": "object",
      "description": "Current window features + references/summary of history.",
      "required": ["current"],
      "additionalProperties": False,
      "properties": {
        "current": {"$ref": "#/$defs/HRVFeatureBlock"},
        "recent_history_refs": {
          "type": "array",
          "items": {"$ref": "#/$defs/HRVWindowRef"},
          "description": "Pointers to recent windows for trend context."
        },
        "history_summary": {
          "type": "object",
          "additionalProperties": False,
          "description": "Optional aggregated history summary.",
          "properties": {
            "time_range_start": {"type": "string", "format": "date-time"},
            "time_range_end": {"type": "string", "format": "date-time"},
            "n_windows": {"type": "integer", "minimum": 0},
            "aggregates": {
              "type": "object",
              "additionalProperties": False,
              "properties": {
                "rmssd_avg": {"type": "number"},
                "sdnn_avg": {"type": "number"},
                "lf_hf_ratio_median": {"type": "number"},
                "stress_index_avg": {"type": "number"}
              }
            }
          }
        }
      }
    },
    "video_semantics": {
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "facial_expression": {"type": "string"},
        "eye_blink_rate": {"type": "number"}
      }
    },
    "controls": {
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "style": {"type": "string",
          "enum": ["concise_bilingual","detailed_cn","detailed_en","plain","doctor_note"],
          "description": "(deprecated) prefer top-level response_style"},
        "output_language": {"type": "string"},
        "return_explanations": {"type": "boolean", "default": True}
      }
    },
    "_confidence_hint": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "overall": {"type": "number", "minimum": 0, "maximum": 1},
            "band": {"type": "string", "enum": ["low","medium","high"]},
            "by_signal_quality": {"type": "number", "minimum": 0, "maximum": 1},
            "by_data_sufficiency": {"type": "number", "minimum": 0, "maximum": 1},
            "by_model": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence_source": {"type": "string",
            "enum": ["model_logits","external_model","rule_based","hybrid","human_review","other"]},
            "metric_confidence": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["metric","score"],
                "additionalProperties": False,
                "properties": {
                "metric": {"type": "string"},
                "score": {"type": "number", "minimum": 0, "maximum": 1}
                }
            }
            },
            "notes": {"type": "string"}
        }
    }
  },
  "$defs": {
    "HRVWindowRef": {
      "type": "object",
      "required": ["window_start", "window_end"],
      "additionalProperties": False,
      "oneOf": [{"required": ["hrv_feature_id"]}, {"required": ["feature_key"]}],
      "properties": {
        "hrv_feature_id": {"type": "integer", "minimum": 1},
        "feature_key": {"type": "string", "minLength": 16},
        "window_start": {"type": "string", "format": "date-time"},
        "window_end": {"type": "string", "format": "date-time"}
      }
    },
    "HRVFeatureBlock": {
      "type": "object",
      "required": ["window_start", "window_end"],
      "additionalProperties": False,
      "oneOf": [{"required": ["hrv_feature_id"]}, {"required": ["feature_key"]}],
      "properties": {
        "hrv_feature_id": {"type": "integer", "minimum": 1},
        "feature_key": {"type": "string", "minLength": 16},
        "window_start": {"type": "string", "format": "date-time"},
        "window_end": {"type": "string", "format": "date-time"},
        "window_length": {"type": "number", "minimum": 1, "maximum": 3600},
        "env_label": {"type": "string",
          "enum": ["rest","exercise","stress","sleep","unknown"]},
        "rmssd": {"type": ["number","null"], "minimum": 0},
        "sdnn": {"type": ["number","null"], "minimum": 0},
        "pnn50": {"type": ["number","null"], "minimum": 0, "maximum": 100},
        "sd1": {"type": ["number","null"], "minimum": 0},
        "sd2": {"type": ["number","null"], "minimum": 0},
        "sd1_sd2_ratio": {"type": ["number","null"], "minimum": 0},
        "lf": {"type": ["number","null"], "minimum": 0},
        "hf": {"type": ["number","null"], "minimum": 0},
        "lf_hf_ratio": {"type": ["number","null"], "minimum": 0},
        "hti": {"type": ["number","null"], "minimum": 0},
        "rhythm_status": {"type": "string", "enum": ["normal","arrhythmic","unknown"]},
        "pwv": {"type": ["number","null"], "minimum": 0},
        "aix": {"type": ["number","null"]},
        "vascular_status": {"type": "string", "enum": ["normal","stiff","unknown"]},
        "stress_index": {"type": ["number","null"], "minimum": 0},
        "stress_level": {"type": "string", "enum": ["low","moderate","high","unknown"]},
        "anxiety_score": {"type": ["number","null"], "minimum": 0, "maximum": 1},
        "anxiety_level": {"type": "string", "enum": ["low","moderate","high","unknown"]},
      }
    }
  }
}

# ---- Output JSON Schema (LLM → system/doctor/user) ----
OUTPUT_SCHEMA: Dict[str, Any] = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.org/schemas/rppg-llm-output.json",
  "title": "RPPG Doctor - LLM Output",
  "type": "object",
  "required": ["interaction_id", "analysis", "confidence"],
  "additionalProperties": False,
  "properties": {
    "interaction_id": {"type": "string", "format": "uuid"},
    "generated_at": {"type": "string", "format": "date-time"},
    "language": {"type": "string"},
    "schema_version": {
    "type": "string",
    "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$",
    "description": "Semantic version for this output schema (e.g., 1.0.0)."
    },
    "style_used": {
    "type": "string",
    "enum": [
        "concise_bilingual",
        "detailed_cn",
        "detailed_en",
        "plain",
        "doctor_note",
        "balanced",   # ← 新增，与你的输入 response_style 对齐
        "formal",     # ← 建议一并加入，防止模型返回 formal
        "casual"      # ← 建议一并加入，防止模型返回 casual
    ]},
    "linked_feature": {
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "hrv_feature_id": {"type": "integer", "minimum": 1},
        "feature_key": {"type": "string", "minLength": 16}
      }
    },
    "analysis": {
      "type": "object",
      "required": ["summary"],
      "additionalProperties": False,
      "properties": {
        "summary": {"type": "string", "minLength": 1},
        "key_findings": {"type": "array", "items": {"type": "string"}},
        "suspected_conditions": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "likelihood"],
            "properties": {
                "code": {"type": "string"},
                "name": {"type": "string"},
                "likelihood": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "metric": { "type": "string" },

                            # 方向型（用于数值/可比对指标，如 RMSSD/SDNN/LF_HF/stress_index 等）
                            "direction": {
                                "type": "string",
                                "enum": ["low", "high", "normal", "trend_up", "trend_down"]
                            },

                            # 状态型（用于分类指标，如 rhythm_status/vascular_status/stress_level/anxiety_level 等）
                            "state": {
                                "type": "string",
                                "enum": ["normal", "stiff", "unknown", "arrhythmic", "poor", "low", "moderate", "high"]
                            },

                            # 可选的原始值：既可能是数值，也可能是字符串（如 "stiff"）
                            "value": { "type": ["number", "string", "null"] },

                            "window_start": { "type": "string", "format": "date-time" },
                            "window_end":   { "type": "string", "format": "date-time" }
                        },

                        # 约束：metric 必须存在，且（direction 或 state）至少年选其一
                        "anyOf": [
                            { "required": ["metric", "direction"] },
                            { "required": ["metric", "state"] }
                        ]
                    }
                }
            }
          }
        },
        "triage": {
          "type": "object",
          "additionalProperties": False,
          "properties": {
            "need_medical_attention": {"type": "boolean"},
            "urgency": {"type": "string",
              "enum": ["none","routine","soon","urgent","emergent"]},
            "care_instructions": {"type": "array", "items": {"type": "string"}}
          }
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "text"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["lifestyle", "sleep", "stress_reduction", "follow_up", "test", "education"]
                    },
                    "text": { "type": "string" },
                    "reference": { "type": "string", "format": "uri" }
                }
            }
        },
        "doctor_report": {
          "type": "object",
          "additionalProperties": False,
          "properties": {
            "soap": {
              "type": "object",
              "additionalProperties": False,
              "properties": {
                "subjective": {"type": "string"},
                "objective": {"type": "string"},
                "assessment": {"type": "string"},
                "plan": {"type": "string"}
              }
            }
          }
        }
      }
    },
    "confidence": {
      "type": "object",
      "required": ["overall"],
      "additionalProperties": False,
      "properties": {
        "overall": {"type": "number", "minimum": 0, "maximum": 1},
        "band": {"type": "string", "enum": ["low","medium","high"]},
        "by_signal_quality": {"type": "number","minimum": 0,"maximum": 1},
        "by_data_sufficiency": {"type": "number","minimum": 0,"maximum": 1},
        "by_model": {"type": "number","minimum": 0,"maximum": 1},
        "confidence_source": {"type": "string",
          "enum": ["model_logits","external_model","rule_based","hybrid","human_review","other"]},
        "metric_confidence": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["metric","score"],
            "properties": {
              "metric": {"type": "string"},
              "score": {"type": "number", "minimum": 0, "maximum": 1}
            }
          }
        },
        "notes": {"type": "string"}
      }
    },
    "citations": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "source_type": {"type": "string", "enum": ["paper","guideline","internal","website"]},
          "title": {"type": "string"},
          "url": {"type": "string", "format": "uri"}
        }
      }
    },
    "warnings": {"type": "array", "items": {"type": "string"}}
  }
}

def validate_input(payload: Dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError on invalid input."""
    jsonschema.validate(instance=payload, schema=INPUT_SCHEMA)

def validate_output(payload: Dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError on invalid output."""
    jsonschema.validate(instance=payload, schema=OUTPUT_SCHEMA)

def pretty(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
