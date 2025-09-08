# llm/llm_adapter.py
# ===================
"""
Call OpenAI Responses API to get a structured answer that strictly follows
our OUTPUT_SCHEMA. It first tries response_format=json_schema (Structured Outputs).
If the model/version doesn't support it, it falls back to Tools(Function Calling)
with strict schema. Finally validates with our OUTPUT_SCHEMA.
"""

from __future__ import annotations
from typing import Any, Dict, List
import os, json
from datetime import datetime, timezone

from .schemas import OUTPUT_SCHEMA, validate_output
from .schemas import pretty  # optional pretty dump

from dotenv import load_dotenv
load_dotenv()


from openai import OpenAI
from openai import APIError
from llm.schemas import OUTPUT_SCHEMA, validate_output

_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = os.getenv("OPENAI_MODEL") 

# ---- system prompt reflecting '5.1 提示词工程' ----
_SYSTEM_PROMPT = (
    "你是一个负责心血管/心理状态联动分析的对话式医疗助手。"
    "你将收到一个JSON作为输入（rPPG时序特征、历史摘要、风格偏好与置信度提示）。"
    "你的唯一输出必须是一个**JSON对象**，且**严格符合**给定的输出JSON Schema。"
    "不得输出额外文本、注释或Markdown代码块。"
    "需要：1) 给出面向用户/医生的中文总结，2) 明确可疑状态与证据，3) 给出就医分诊建议，"
    "4) 给出生活方式/减压/随访建议，5) 在输出中填入置信度分解。"
    "此建议不替代临床诊断，如出现紧急症状需立即就医。"
)

def _build_messages(llm_input: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Compose messages for the Responses API. We include the input JSON payload
    as user content verbatim, plus an instruction reminder to output JSON only.
    """
    style = llm_input.get("response_style", "balanced")
    lang  = llm_input.get("language", "zh-CN")
    reminder = (
        "严格输出单个JSON对象；不得输出JSON外的任何字符。"
        "根据style/language与history/置信度提示生成结构化回答。"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": f"[REMINDER] {reminder}\n[INPUT_JSON]\n{json.dumps(llm_input, ensure_ascii=False)}"}
    ]

def _try_structured_outputs(messages):
    """
    Preferred path: Structured Outputs via response_format=json_schema (Responses API).
    See official docs for response_format json_schema. 
    """
    resp = _CLIENT.responses.create(
        model=_MODEL,
        input=messages,             # messages: [{"role":"system","content":"..."}, ...]
        temperature=0,
        max_output_tokens=2048,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "rppg_doctor_output",
                "strict": True,
                "schema": OUTPUT_SCHEMA
            }
        }
    )

    # 读取结构化文本（不同版本 SDK 返回路径略有不同，以下兼容）
    text = getattr(resp, "output_text", None)
    if not text:
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    text = c.text
                    break
            if text:
                break
    if not text:
        raise RuntimeError("No output_text returned by Responses API")

    data = json.loads(text)
    validate_output(data)
    return data

# ====== begin: normalization helpers (paste these once) ======
_ALLOWED_DIRECTIONS = {"low", "high", "normal", "trend_up", "trend_down"}
_ALLOWED_STATES = {"normal", "stiff", "unknown", "arrhythmic", "poor", "low", "moderate", "high"}

# 关键词映射（可按需补充）
_STATE_MAP = {
    "stiff":      ["僵硬", "硬化", "stiff", "rigid", "pulseless"],
    "arrhythmic": ["心律不齐", "房颤", "房扑", "心律失常", "arrhythm", "af", "atrial fibrillation"],
    "poor":       ["差", "较差", "poor"],
    "normal":     ["正常", "normal", "baseline normal"],
    "low":        ["偏低", "很低", "低于正常", "low"],
    "moderate":   ["中等", "中度", "moderate"],
    "high":       ["偏高", "很高", "高于正常", "高血压", "high", "elevated"],
    # "unknown" 不做关键字命中，作为兜底
}

_DIRECTION_MAP = {
    "high":      ["高", "升高", "上升", "增加", "偏高", "elevated", "increase", "higher", "↑"],
    "low":       ["低", "降低", "下降", "减少", "偏低", "decrease", "lower", "reduced", "↓"],
    "trend_up":  ["趋势上升", "逐步上升", "trend up", "↑↑"],
    "trend_down":["趋势下降", "逐步下降", "trend down", "↓↓"],
    "normal":    ["正常", "在正常范围", "within normal", "baseline"],
}

def _guess_state_from_text(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    t_lower = text.lower()
    for state, kws in _STATE_MAP.items():
        for kw in kws:
            if kw in text or kw in t_lower:
                return state
    return None

def _guess_direction_from_text(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    t_lower = text.lower()
    for direc, kws in _DIRECTION_MAP.items():
        for kw in kws:
            if kw in text or kw in t_lower:
                return direc
    return None

def _coerce_wrong_direction_to_state(data: dict):
    """若 evidence.direction 非法（如 'stiff'），迁移到 evidence.state。"""
    analysis = (data or {}).get("analysis") or {}
    conds = analysis.get("suspected_conditions") or []
    for cond in conds:
        evs = cond.get("evidence") or []
        for ev in evs:
            if not isinstance(ev, dict):
                continue
            dirv = ev.get("direction")
            if dirv and dirv not in _ALLOWED_DIRECTIONS:
                ev["state"] = dirv
                ev.pop("direction", None)

def _ensure_state_or_direction(ev: dict):
    """保证 evidence 里至少有 state 或 direction；都没有就优先 state='unknown'。"""
    if not isinstance(ev, dict):
        return
    has_dir = ev.get("direction") in _ALLOWED_DIRECTIONS
    has_state = ev.get("state") in _ALLOWED_STATES
    if not (has_dir or has_state):
        # 尝试从 value 文本再猜一次
        v = ev.get("value")
        s = _guess_state_from_text(v) if isinstance(v, str) else None
        d = _guess_direction_from_text(v) if isinstance(v, str) else None
        if s in _ALLOWED_STATES:
            ev["state"] = s
        elif d in _ALLOWED_DIRECTIONS:
            ev["direction"] = d
        else:
            ev["state"] = "unknown"

def _normalize_doctor_report_in_analysis(analysis: dict):
    """
    使 analysis.doctor_report 满足 OUTPUT_SCHEMA：
    - None -> 删除该字段
    - str  -> {'soap': {'subjective': <str>}}
    - dict -> 只保留 'soap'，其余清理；soap 的四个子键若存在则转成字符串
    - 空对象 -> 删除
    """
    if not isinstance(analysis, dict):
        return

    if "doctor_report" not in analysis:
        return

    dr = analysis.get("doctor_report")

    # 1) None -> 直接删掉（该字段非必填）
    if dr is None:
        analysis.pop("doctor_report", None)
        return

    # 2) 字符串 -> 作为 SOAP 的 subjective
    if isinstance(dr, str):
        analysis["doctor_report"] = {"soap": {"subjective": dr}}
        return

    # 3) 字典 -> 规范到只有 'soap'
    if isinstance(dr, dict):
        # 如果已有 soap
        if "soap" in dr:
            soap = dr.get("soap")
            if soap is None:
                dr["soap"] = {}
            elif isinstance(soap, dict):
                # 只接受这四个键，且值统一成字符串
                for k in list(soap.keys()):
                    if k not in ("subjective", "objective", "assessment", "plan"):
                        soap.pop(k, None)
                for k in ("subjective", "objective", "assessment", "plan"):
                    if k in soap and soap[k] is not None and not isinstance(soap[k], str):
                        soap[k] = str(soap[k])
            else:
                # soap 是个非 dict 的东西，收敛到 subjective
                dr["soap"] = {"subjective": str(soap)}
        else:
            # 没有 soap，但也许平铺了字段
            flat_fields = {}
            for k in ("subjective", "objective", "assessment", "plan"):
                if k in dr and dr[k] is not None:
                    flat_fields[k] = str(dr[k])
            if flat_fields:
                dr["soap"] = flat_fields

        # 只保留 'soap'
        for k in list(dr.keys()):
            if k != "soap":
                dr.pop(k, None)

        # 空对象则删除该字段
        if not dr or (isinstance(dr.get("soap"), dict) and not dr["soap"]):
            analysis.pop("doctor_report", None)
        return

    # 4) 其他类型 -> 转成 subjective 字符串
    analysis["doctor_report"] = {"soap": {"subjective": str(dr)}}

def _narrative_to_evidence_obj(text: str) -> dict:
    """将纯字符串证据转为 schema 合法对象，尽量补齐 state/direction。"""
    obj = {"metric": "narrative", "value": text}
    # 优先猜测状态类
    s = _guess_state_from_text(text)
    if s in _ALLOWED_STATES:
        obj["state"] = s
    else:
        # 其次猜方向
        d = _guess_direction_from_text(text)
        if d in _ALLOWED_DIRECTIONS:
            obj["direction"] = d
        else:
            obj["state"] = "unknown"
    return obj

_ALLOWED_TOP_LEVEL_KEYS = {
    "interaction_id",
    "generated_at",
    "language",
    "schema_version",
    "style_used",
    "linked_feature",
    "analysis",
    "confidence",
    "citations",
    "warnings",
}

def _normalize_output_aliases_and_evidence(data: dict):
    """
    统一/清洗 LLM 输出，使之满足 OUTPUT_SCHEMA：
    - 顶层 recommendations -> analysis.recommendations
    - 顶层 doctor_report -> analysis.doctor_report   # ★ 新增这一条
    - 证据：字符串 -> 对象；不完整对象自动补齐 state/direction（使用前面提供的辅助函数）
    - 将把 'stiff' 误写入 direction 的错误纠正到 state
    - （可选）清理未知的顶层键，避免 additionalProperties 报错
    """
    if not isinstance(data, dict):
        return

    # 0) 确保 analysis 存在
    analysis = data.setdefault("analysis", {})

    # 1) 顶层 -> analysis：recommendations
    if "recommendations" in data:
        if "recommendations" not in analysis or not analysis["recommendations"]:
            analysis["recommendations"] = data["recommendations"]
        data.pop("recommendations", None)

    # 2) 顶层 -> analysis：doctor_report   # ★ 关键修复
    if "doctor_report" in data:
        if "doctor_report" not in analysis or not analysis["doctor_report"]:
            analysis["doctor_report"] = data["doctor_report"]
        data.pop("doctor_report", None)

    _normalize_doctor_report_in_analysis(analysis)
    
    # 3) 证据清洗（字符串 -> 对象；补齐 state/direction）
    conds = analysis.get("suspected_conditions") or []
    for cond in conds:
        evs = cond.get("evidence")
        if isinstance(evs, list):
            new_evs = []
            for item in evs:
                if isinstance(item, dict):
                    _ensure_state_or_direction(item)  # 若缺少则猜测补齐
                    new_evs.append(item)
                elif isinstance(item, str):
                    new_evs.append(_narrative_to_evidence_obj(item))  # 由文本推断 state/direction
                else:
                    new_evs.append(_narrative_to_evidence_obj(str(item)))
            cond["evidence"] = new_evs

    # 4) 若 direction 写了非法值（如 "stiff"），迁移到 state
    _coerce_wrong_direction_to_state(data)

    # 5) （可选）清理未知顶层键，避免 additionalProperties 报错
    for k in list(data.keys()):
        if k not in _ALLOWED_TOP_LEVEL_KEYS:
            data.pop(k, None)
# ====== end: normalization helpers ======

def _to_function_parameters(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert our OUTPUT_SCHEMA to a 'parameters' object for tool definition.
    We keep only commonly supported keywords for function calling: type/properties/required/additionalProperties.
    """
    allowed = {k: schema[k] for k in ("type","properties","required","additionalProperties") if k in schema}
    return allowed

def _try_chat_tools(messages):
    """
    兜底路径：老兼容写法（Chat Completions + 工具函数）。
    不再使用 Responses 工具回退，从而避免 'tools[0].name' 这类不一致。
    """
    tool_def = {
        "type": "function",
        "function": {
            "name": "emit_structured_output",
            "description": "Return the final structured output JSON strictly following the schema.",
            "parameters": _to_function_parameters(OUTPUT_SCHEMA)
        }
    }

    chat_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    resp = _CLIENT.chat.completions.create(
        model=_MODEL,
        messages=chat_messages,
        tools=[tool_def],
        tool_choice={"type": "function", "function": {"name": "emit_structured_output"}},
        temperature=0,
    )

    choice = resp.choices[0]
    tool_calls = getattr(choice.message, "tool_calls", None) or []
    if tool_calls:
        args_text = tool_calls[0].function.arguments
        data = json.loads(args_text)
    else:
        # 有的模型会直接把 JSON 当文本返回
        data = json.loads(choice.message.content or "{}")

    _normalize_output_aliases_and_evidence(data)
    validate_output(data)
    return data

def _try_tools_fallback(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Fallback path: Tools(Function Calling) with strict schema.
    Works widely on models that support tools. (strict=True required)
    """
    tool_def = {
        "type": "function",
        "function": {
            "name": "emit_structured_output",
            "description": "Return the final structured output JSON strictly following the schema.",
            "parameters": _to_function_parameters(OUTPUT_SCHEMA),
            "strict": True  # ensure schema adherence
        }
    }
    resp = _CLIENT.responses.create(
        model=_MODEL,
        input=messages,
        tools=[tool_def],
        tool_choice={"type": "tool", "name": "emit_structured_output"},
        temperature=0,
        max_output_tokens=2048,
    )
    # 解析 tool_call 的 arguments
    data = None
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) == "tool_call":
                args = getattr(c, "tool_call", None)
                if args and getattr(args, "arguments", None) is not None:
                    data = args.arguments
                    break
        if data is not None:
            break

    if data is None:
        # 最后尝试直接解析文本（少数实现会把JSON当成普通文本吐出）
        text = getattr(resp, "output_text", "")
        data = json.loads(text)

    validate_output(data)
    return data

def generate_structured_response(llm_input):
    """
    Public entry: try Structured Outputs first; on failure fallback to Tools.
    Always returns an object validating against OUTPUT_SCHEMA.
    """
    messages = _build_messages(llm_input)  # 你已有的构建消息函数

    # 1) Responses + Structured Outputs（首选）
    try:
        return _try_structured_outputs(messages)
    except Exception as e1:
        # 2) Chat Completions + tools 兜底
        try:
            return _try_chat_tools(messages)
        except Exception as e2:
            raise RuntimeError(
                f"LLM call failed.\n"
                f"responses_structured_error={e1}\n"
                f"chat_tools_error={e2}"
            )

