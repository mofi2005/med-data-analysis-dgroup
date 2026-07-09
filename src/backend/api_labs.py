#!/usr/bin/env python3
"""生化指标时间序列趋势 API 路由 — D组医学数据中台.

本模块提供面向 ECharts 的时间序列可视化数据接口:
  - GET /api/labs/trend  — 获取特定指标的纵向随访趋势数据

返回格式专为 Vue3 + ECharts 设计, 包含 xAxis/series/异常标记.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.backend.database import get_db
from src.backend.models_db import DataDictionary

# ---------------------------------------------------------------------------
# APIRouter 实例
# ---------------------------------------------------------------------------

labs_router = APIRouter(
    prefix="/api/labs",
    tags=["生化分析与时间序列"],
)

# ---------------------------------------------------------------------------
# 内置指标参考范围 + 模拟数据生成参数
# ---------------------------------------------------------------------------

# 指标名 → (low, high, unit, 生成均值, 生成标准差)
_ITEM_PROFILES: dict[str, dict] = {
    "WBC": {
        "low": 3.5, "high": 9.5, "unit": "10^9/L",
        "gen_mean": 6.5, "gen_std": 2.5, "title": "白细胞计数",
    },
    "GLU": {
        "low": 3.9, "high": 6.1, "unit": "mmol/L",
        "gen_mean": 5.5, "gen_std": 1.8, "title": "空腹血糖",
    },
    "HbA1c": {
        "low": 4.0, "high": 6.0, "unit": "%",
        "gen_mean": 5.2, "gen_std": 0.6, "title": "糖化血红蛋白",
    },
    "Creatinine": {
        "low": 44, "high": 133, "unit": "μmol/L",
        "gen_mean": 85, "gen_std": 25, "title": "血清肌酐",
    },
    "ALT": {
        "low": 0, "high": 40, "unit": "U/L",
        "gen_mean": 25, "gen_std": 15, "title": "丙氨酸氨基转移酶",
    },
    "HGB": {
        "low": 130, "high": 175, "unit": "g/L",
        "gen_mean": 150, "gen_std": 15, "title": "血红蛋白",
    },
    "PLT": {
        "low": 125, "high": 350, "unit": "10^9/L",
        "gen_mean": 230, "gen_std": 60, "title": "血小板计数",
    },
    "CEA": {
        "low": 0, "high": 5, "unit": "ng/mL",
        "gen_mean": 2.0, "gen_std": 2.5, "title": "癌胚抗原",
    },
    "CRP": {
        "low": 0, "high": 10, "unit": "mg/L",
        "gen_mean": 4.0, "gen_std": 5, "title": "C反应蛋白",
    },
}

# 默认时间点数
_DEFAULT_N_POINTS = 7
# 异常触发概率 (每周)
_ABNORMAL_CHANCE = 0.2


# ===================================================================
# 模拟数据生成
# ===================================================================


def _generate_time_series(
    item_name: str,
    n_points: int = _DEFAULT_N_POINTS,
    seed: Optional[int] = None,
) -> dict:
    """为指定指标生成一组合理的纵向随访时间序列数据。

    算法:
      - 基于指标的 gen_mean/gen_std 生成正态分布随机值
      - 以恒定概率将个别点偏移到超出参考范围, 模拟异常波动
      - 首尾连成一条有临床意义的波动轨迹

    Args:
        item_name: 指标标准名 (如 GLU, WBC)。
        n_points:  时间点数量 (默认 7)。
        seed:      随机种子 (用于复现)。

    Returns:
        {"xAxis": [...], "series": [...], "abnormal_indices": [...],
         "trend_direction": "..."}
    """
    profile = _ITEM_PROFILES.get(item_name.upper())
    if profile is None:
        raise ValueError(f"不支持的指标: {item_name}")

    rng = np.random.RandomState(seed)
    low, high = profile["low"], profile["high"]
    mean, std = profile["gen_mean"], profile["gen_std"]

    # 基础值: 略低于均值开始, 提供变化空间
    base = mean + rng.normal(0, std * 0.3)
    base = max(low * 0.6, min(high * 1.5, base))

    # 生成日期序列 (从一年前开始, 均匀间隔)
    today = datetime.now()
    start_date = today - timedelta(days=n_points * 28)  # ~每周一个点
    dates = [
        (start_date + timedelta(days=i * 28)).strftime("%Y-%m-%d")
        for i in range(n_points)
    ]

    # 生成值: 带趋势 + 随机波动
    values: list[float] = []
    for i in range(n_points):
        # 轻度线性趋势 (上升或下降)
        trend = (rng.random() - 0.45) * std * 0.3 * i / n_points
        noise = rng.normal(0, std * 0.15)
        val = base + trend + noise

        # 异常注入: 以概率将值推到参考范围外
        if rng.random() < _ABNORMAL_CHANCE:
            if rng.random() < 0.5:
                val = high + rng.uniform(0.1, 1.0) * std * 0.5  # 偏高一侧
            else:
                val = low - rng.uniform(0.1, 0.5) * std * 0.3  # 偏低一侧

        # 约束最小值 > 0
        val = max(0.01, val)
        values.append(round(val, 2))

    # 标记异常点
    abnormal_indices = [
        i for i, v in enumerate(values)
        if v < low or v > high
    ]

    # 判定趋势方向
    if len(values) >= 2:
        first_half = np.mean(values[: len(values) // 2])
        second_half = np.mean(values[len(values) // 2 :])
        diff = second_half - first_half
        if diff > std * 0.1:
            direction = "increasing"
        elif diff < -std * 0.1:
            direction = "decreasing"
        else:
            direction = "stable"
    else:
        direction = "stable"

    # 若同时有升有降的异常, 标记为 fluctuating
    if len(abnormal_indices) >= 2:
        above = sum(1 for i in abnormal_indices if values[i] > high)
        below = sum(1 for i in abnormal_indices if values[i] < low)
        if above > 0 and below > 0:
            direction = "fluctuating"

    return {
        "xAxis": dates,
        "series": values,
        "abnormal_indices": abnormal_indices,
        "trend_direction": direction,
    }


# ===================================================================
# GET /api/labs/trend
# ===================================================================


@labs_router.get("/trend")
def get_lab_trend(
    patient_id: str = Query(default="p001", description="患者 ID"),
    item_name: str = Query(default="GLU", description="指标名, 如 WBC/GLU/ALT"),
    db: Session = Depends(get_db),
):
    """获取特定指标的时间序列 ECharts 可视化数据。

    工作流:
      1. 查询 DataDictionary 获取参考范围与单位
      2. 若字典不存在该指标, 回退到内置 profile
      3. 生成/读取该患者的纵向随访序列数据
      4. 标记超出参考范围的异常时间点
      5. 返回 ECharts 兼容格式

    Args:
        patient_id: 患者 ID。
        item_name:  指标标准名 (如 GLU, WBC, ALT)。
        db:         数据库会话。

    Returns:
        ECharts 可视化 JSON。
    """
    item_upper = item_name.strip().upper()

    # ---- 1. 查询 DataDictionary ----
    record = (
        db.query(DataDictionary)
        .filter(DataDictionary.standard_name == item_upper)
        .first()
    )

    if record:
        low = _parse_ref_low(record.reference_range or "")
        high = _parse_ref_high(record.reference_range or "")
        unit = record.unit or ""
        chinese_name = record.chinese_name or item_upper
    else:
        # 回退到内置 profile
        profile = _ITEM_PROFILES.get(item_upper)
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"未找到指标 '{item_name}'。支持的指标: "
                       f"{list(_ITEM_PROFILES.keys())}",
            )
        low = profile["low"]
        high = profile["high"]
        unit = profile["unit"]
        chinese_name = profile.get("title", item_upper)

    # ---- 2. 生成时间序列数据 ----
    # 基于 patient_id 做确定性种子 (同一患者重复请求数据一致)
    seed_val = hash(patient_id + item_upper) % (2 ** 31)
    try:
        echarts_data = _generate_time_series(item_upper, n_points=7, seed=seed_val)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {
        "status": "success",
        "patient_id": patient_id,
        "item_name": item_upper,
        "chinese_name": chinese_name,
        "unit": unit,
        "reference_range": {
            "low": low,
            "high": high,
        },
        "echarts_data": echarts_data,
    }


# ===================================================================
# 辅助: 解析参考范围字符串
# ===================================================================


def _parse_ref_low(ref: str) -> float:
    """从参考范围字符串中提取最小值, 如 '3.5-9.5' → 3.5."""
    parts = ref.replace(">", "").replace("<", "").split("-")
    try:
        return float(parts[0].strip())
    except (ValueError, IndexError):
        return 0.0


def _parse_ref_high(ref: str) -> float:
    """从参考范围字符串中提取最大值, 如 '3.5-9.5' → 9.5."""
    parts = ref.replace(">", "").replace("<", "").split("-")
    try:
        if len(parts) >= 2:
            return float(parts[1].strip())
        return float(parts[0].strip()) * 2  # fallback
    except (ValueError, IndexError):
        return 100.0
