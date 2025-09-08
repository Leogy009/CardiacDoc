# query_aggregator.py
# ====================
# 作用：
# 1) 统一创建 DB 连接（DuckDB/SQLite 自适应）
# 2) 提供按时间范围查询 hrv_features 的工具函数
# 3) 提供历史聚合指标（均值/中位数）便于给 LLM 做 history_summary

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
from datetime import datetime, timezone

_BACKEND = os.getenv("HRV_DB_BACKEND", "duckdb").lower()

def create_connection(db_path: str):
    """返回一个 DB 连接对象。DuckDB/SQLite 自动选择。"""
    if _BACKEND == "sqlite":
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    else:
        import duckdb
        conn = duckdb.connect(db_path)
        return conn

def _rows_to_dicts(rows, cursor=None) -> List[Dict[str, Any]]:
    """将查询结果转为字典列表（DuckDB/SQLite 通吃）"""
    out = []
    if rows is None:
        return out
    try:
        # DuckDB：返回的是 tuples + description
        if cursor is not None and hasattr(cursor, "description") and cursor.description:
            cols = [c[0] for c in cursor.description]
            for r in rows:
                out.append({k: v for k, v in zip(cols, r)})
            return out
    except Exception:
        pass
    # SQLite：Row 可直接 dict(r)
    for r in rows:
        try:
            out.append(dict(r))
        except Exception:
            out.append(r)
    return out

def _iso(t) -> str:
    if isinstance(t, str):
        return t
    if isinstance(t, datetime):
        if t.tzinfo is None:
            return t.replace(tzinfo=timezone.utc).isoformat()
        return t.isoformat()
    return str(t)

def query_hrv_rows(
    conn,
    patient_id: str,
    start_time: str,
    end_time: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    查询某患者在时间范围内的 hrv_features 记录，按 window_start DESC 排序。
    返回的每行包含表中的所有列（含 feature_key / 指标 / 状态等）。
    """
    q = """
    SELECT *
    FROM hrv_features
    WHERE patient_id = ?
      AND window_start >= ?
      AND window_end   <= ?
    ORDER BY window_start DESC
    """
    params = [patient_id, start_time, end_time]
    if _BACKEND == "sqlite":
        cur = conn.execute(q, params)
        rows = cur.fetchall()
        out = _rows_to_dicts(rows)
        if limit:
            out = out[:limit]
        return out
    else:
        # DuckDB
        cur = conn.execute(q, params)
        rows = cur.fetchall()
        out = _rows_to_dicts(rows, cur)
        if limit:
            out = out[:limit]
        return out

def aggregate_hrv_metrics(
    conn,
    patient_id: str,
    start_time: str,
    end_time: str
) -> Dict[str, Optional[float]]:
    """
    计算历史区间聚合（与我们在 JSON Schema 中使用的 summary 对齐）：
    - rmssd_avg
    - sdnn_avg
    - lf_hf_ratio_median
    - stress_index_avg
    """
    # 均值
    q_avg = """
    SELECT
      AVG(rmssd) AS rmssd_avg,
      AVG(sdnn)  AS sdnn_avg,
      AVG(stress_index) AS stress_index_avg
    FROM hrv_features
    WHERE patient_id = ?
      AND window_start >= ?
      AND window_end   <= ?
    """
    # 中位数：DuckDB 可直接 median()；SQLite 需要手动
    if _BACKEND == "sqlite":
        rows = query_hrv_rows(conn, patient_id, start_time, end_time)
        lf_hf_vals = sorted([r["lf_hf_ratio"] for r in rows if isinstance(r.get("lf_hf_ratio"), (int, float))])
        if lf_hf_vals:
            mid = len(lf_hf_vals) // 2
            if len(lf_hf_vals) % 2 == 1:
                lf_hf_median = float(lf_hf_vals[mid])
            else:
                lf_hf_median = float((lf_hf_vals[mid-1] + lf_hf_vals[mid]) / 2.0)
        else:
            lf_hf_median = None

        cur = conn.execute(q_avg, [patient_id, start_time, end_time])
        avg_row = cur.fetchone()
        if avg_row is None:
            return {"rmssd_avg": None, "sdnn_avg": None, "lf_hf_ratio_median": None, "stress_index_avg": None}
        avg = dict(avg_row)
        return {
            "rmssd_avg": avg.get("rmssd_avg"),
            "sdnn_avg": avg.get("sdnn_avg"),
            "lf_hf_ratio_median": lf_hf_median,
            "stress_index_avg": avg.get("stress_index_avg"),
        }
    else:
        # DuckDB：直接 median()
        q = f"""
        SELECT
          AVG(rmssd) AS rmssd_avg,
          AVG(sdnn)  AS sdnn_avg,
          MEDIAN(lf_hf_ratio) AS lf_hf_ratio_median,
          AVG(stress_index) AS stress_index_avg
        FROM hrv_features
        WHERE patient_id = ?
          AND window_start >= ?
          AND window_end   <= ?
        """
        cur = conn.execute(q, [patient_id, start_time, end_time])
        row = cur.fetchone()
        if row is None:
            return {"rmssd_avg": None, "sdnn_avg": None, "lf_hf_ratio_median": None, "stress_index_avg": None}
        return dict(row)

def fetch_recent_windows(
    conn,
    patient_id: str,
    minutes: int = 30,
    end_time_iso: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    拉取最近 minutes 内的窗口记录（用于 recent_history_refs）。
    """
    end_iso = end_time_iso or datetime.now(timezone.utc).isoformat()
    # 简化：假设窗口连续，直接 end - minutes
    # 这里不在 SQL 里做时间运算，直接给 start/end。
    from datetime import timedelta, datetime as _dt
    end_dt = _dt.fromisoformat(end_iso.replace("Z", "+00:00"))
    start_dt = end_dt - timedelta(minutes=minutes)
    return query_hrv_rows(conn, patient_id, start_dt.isoformat(), end_dt.isoformat())
