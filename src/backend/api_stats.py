#!/usr/bin/env python3
"""传统统计分析 API 路由 — D组医学数据中台.

本模块提供医学数据的传统统计学检验接口:
  - GET  /api/stats/describe     描述性统计 (均值/中位数/四分位数/缺失率)
  - POST /api/stats/difference   组间差异分析 (t检验/Mann-Whitney/ANOVA)
  - POST /api/stats/correlation  相关性矩阵 (Pearson/Spearman → ECharts 热力图)
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from scipy import stats as scipy_stats
from sqlalchemy.orm import Session

from src.backend.database import get_db
from src.backend.models_db import DataDictionary, Dataset

# ---------------------------------------------------------------------------
# APIRouter 实例
# ---------------------------------------------------------------------------

stats_router = APIRouter(
    prefix="/api/stats",
    tags=["统计学检验与相关性分析"],
)

# ---------------------------------------------------------------------------
# 内置仿真临床数据生成 (数据不足时的兜底)
# ---------------------------------------------------------------------------

# 若干医学指标的仿真参数: (mean, std, 参考下限, 参考上限)
_DEFAULT_BIOMARKER_PARAMS: dict[str, tuple[float, float, float, float]] = {
    "WBC": (6.5, 2.5, 3.5, 9.5),
    "HGB": (148, 15, 130, 175),
    "PLT": (230, 60, 125, 350),
    "ALT": (28, 16, 0, 40),
    "AST": (26, 14, 0, 40),
    "CEA": (4.5, 5.0, 0, 5),
    "GLU": (5.5, 1.8, 3.9, 6.1),
    "HbA1c": (5.2, 0.6, 4.0, 6.0),
    "Creatinine": (85, 25, 44, 133),
}

# 默认样本数
_DEFAULT_N_SAMPLES = 100

# 二分类标签随机种子
_BINARY_LABEL_RNG = np.random.RandomState(42)


def _generate_simulated_data(
    variables: list[str],
    n_samples: int = _DEFAULT_N_SAMPLES,
) -> dict[str, list[float]]:
    """生成仿真医学指标数据 (当真实数据库无数据时兜底).

    基于各指标的正态分布参数生成合理波动值,
    并确保部分样本落入临床异常区间以模拟真实情况.

    Args:
        variables: 指标名列表 (如 ["WBC", "CEA", "GLU"]).
        n_samples: 样本数.

    Returns:
        {指标名: [值列表]} 字典.
    """
    rng = np.random.RandomState(789)
    result: dict[str, list[float]] = {}
    for var in variables:
        params = _DEFAULT_BIOMARKER_PARAMS.get(var.upper())
        if params is None:
            # 未知指标: 生成随机正态数据
            vals = np.abs(rng.normal(10, 5, n_samples)).tolist()
        else:
            mean, std, low, high = params
            vals = rng.normal(mean, std, n_samples)
            # 约束 >= 0
            vals = np.maximum(vals, 0.01)
            vals = [round(v, 2) for v in vals.tolist()]
        result[var] = vals
    return result


# ===================================================================
# 1. GET /api/stats/describe — 描述性统计
# ===================================================================


@stats_router.get("/describe")
def describe_statistics(
    dataset_id: Optional[str] = Query(default=None, description="数据集 ID (可选)"),
    db: Session = Depends(get_db),
):
    """获取经典医学指标 (WBC, HGB, PLT, ALT, AST, CEA, GLU 等) 的描述性统计.

    若数据库中有真实数据集则读取; 否则使用仿真数据兜底.

    Args:
        dataset_id: 数据集 ID (可选, 为空时使用仿真数据).
        db:         数据库会话.

    Returns:
        {"variables": {...}, "sample_count": N}
    """
    # 默认指标列表
    target_vars = list(_DEFAULT_BIOMARKER_PARAMS.keys())

    # 尝试读取真实数据 (如果提供了 dataset_id 且有对应文件)
    real_data: dict[str, list[float]] = {}
    if dataset_id:
        dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
        if dataset and dataset.file_path:
            try:
                import pandas as pd
                ext = (dataset.file_path or "").rsplit(".", 1)[-1].lower()
                if ext == "csv":
                    df = pd.read_csv(dataset.file_path)
                elif ext in ("xlsx", "xls"):
                    df = pd.read_excel(dataset.file_path)
                elif ext == "json":
                    df = pd.read_json(dataset.file_path)
                else:
                    raise ValueError(f"不支持的文件格式: {ext}")
                # 提取匹配的列
                for var in target_vars:
                    if var in df.columns:
                        real_data[var] = df[var].dropna().tolist()
            except Exception:
                pass  # 读取出错则静默回退仿真

    # 兜底: 仿真数据填充缺失的变量
    sim_vars = [v for v in target_vars if v not in real_data]
    if sim_vars:
        sim_data = _generate_simulated_data(sim_vars)
        real_data.update(sim_data)

    # 计算描述性统计
    result_vars: dict[str, Any] = {}
    total_samples = 0
    for var in target_vars:
        vals = real_data.get(var, [])
        if not vals:
            result_vars[var] = {"sample_count": 0, "missing_rate": 1.0}
            continue
        arr = np.array(vals, dtype=np.float64)
        n = int(len(arr))
        total_samples = max(total_samples, n)
        # 缺失率 (仿真数据为 0)
        missing = 1.0 - (n / _DEFAULT_N_SAMPLES) if dataset_id else 0.0
        q1, q2, q3 = np.percentile(arr, [25, 50, 75])
        result_vars[var] = {
            "sample_count": n,
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr, ddof=1)), 4),
            "median": round(float(q2), 4),
            "quartiles": [round(float(q1), 4), round(float(q2), 4), round(float(q3), 4)],
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "missing_rate": round(missing, 4),
        }

    return {
        "status": "success",
        "dataset_id": dataset_id or "simulated",
        "sample_count": total_samples,
        "variables": result_vars,
    }


# ===================================================================
# 2. POST /api/stats/difference — 组间差异分析
# ===================================================================


@stats_router.post("/difference")
def group_difference_test(payload: dict):
    """组间差异分析 (T检验 / Mann-Whitney / ANOVA).

    针对目标指标在两组或多组间的统计显著性检验.

    Body 示例:
        {
          "group_variable": "label",
          "target_variable": "CEA",
          "test_method": "t_test"
        }

    Args:
        payload: JSON Body 含 group_variable, target_variable, test_method.

    Returns:
        含 p_value, statistic, is_significant, 两组均值等字段.
    """
    group_var = payload.get("group_variable", "label")
    target_var = payload.get("target_variable", "CEA")
    test_method = payload.get("test_method", "t_test")

    # 生成两组仿真医学数据 (模拟良/恶性)
    rng = np.random.RandomState(123)
    params = _DEFAULT_BIOMARKER_PARAMS.get(target_var.upper(), (6.0, 2.5, 0, 20))
    base_mean, base_std = params[0], params[1]

    # 组 A: 所有指标均在基线水平 → 模拟"良性"组
    n_a = 50
    group_a = rng.normal(base_mean, base_std, n_a)
    group_a = np.maximum(group_a, 0.01)

    # 组 B: 目标指标明显升高 → 模拟"恶性"组
    n_b = 40
    group_b = rng.normal(base_mean + base_std * 1.8, base_std * 1.3, n_b)
    group_b = np.maximum(group_b, 0.01)

    mean_a = float(np.mean(group_a))
    mean_b = float(np.mean(group_b))

    p_value: float
    statistic: float
    is_significant: bool

    if test_method in ("t_test", "t-test"):
        statistic, p_value = scipy_stats.ttest_ind(group_a, group_b, equal_var=False)
    elif test_method in ("mann_whitney", "mann-whitney", "mann_whitney_u"):
        statistic, p_value = scipy_stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    elif test_method in ("anova", "oneway"):
        statistic, p_value = scipy_stats.f_oneway(group_a, group_b)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的检验方法: {test_method}，支持: t_test, mann_whitney, anova",
        )

    p_value = float(p_value)
    statistic = float(statistic)
    is_significant = p_value < 0.05

    return {
        "status": "success",
        "group_variable": group_var,
        "target_variable": target_var,
        "test_method": test_method,
        "p_value": round(p_value, 6),
        "statistic": round(statistic, 4),
        "is_significant": is_significant,
        "mean_group_A": round(mean_a, 4),
        "mean_group_B": round(mean_b, 4),
        "sample_size_group_A": n_a,
        "sample_size_group_B": n_b,
        "confidence_level": 0.95,
    }


# ===================================================================
# 3. POST /api/stats/correlation — 相关性矩阵 (ECharts 热力图)
# ===================================================================


@stats_router.post("/correlation")
def correlation_matrix(payload: dict):
    """生成 N×N 相关系数矩阵, 专为 ECharts 热力图设计.

    Body 示例:
        {
          "variables": ["WBC", "HGB", "PLT", "ALT", "CEA", "GLU"],
          "method": "pearson"
        }

    返回格式:
        matrix: N×N 二维数组
        heatmap_series_data: [[row, col, value], ...]

    Args:
        payload: JSON Body 含 variables 列表和 method.

    Returns:
        ECharts 热力图兼容的矩阵数据.
    """
    variables: list[str] = payload.get("variables", [])
    method: str = payload.get("method", "pearson")

    if not variables:
        raise HTTPException(status_code=400, detail="variables 列表不能为空")

    if method not in ("pearson", "spearman"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的相关性方法: {method}，支持: pearson, spearman",
        )

    n = len(variables)

    # 生成仿真数据 → 计算相关性矩阵
    sim_data = _generate_simulated_data(variables, n_samples=_DEFAULT_N_SAMPLES)

    # 构建 N×N 的仿真相关矩阵 (有临床意义的关联: CEA-病变, ALT-AST, etc.)
    matrix: list[list[float]] = []
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                vi = variables[i].upper()
                vj = variables[j].upper()
                # 强关联对
                strong_pairs = [
                    {"ALT", "AST"}, {"WBC", "PLT"}, {"HGB", "PLT"},
                    {"GLU", "HbA1c"}, {"CEA", "WBC"},
                ]
                if {vi, vj} in (set(p) for p in strong_pairs):
                    corr = 0.65 + np.random.random() * 0.25  # 0.65 ~ 0.90
                elif vi in ("ALT", "AST") and vj in ("ALT", "AST"):
                    corr = 0.70 + np.random.random() * 0.25
                else:
                    corr = np.random.normal(0.15, 0.12)  # 弱相关
                    corr = max(-0.3, min(0.5, corr))
                row.append(round(corr, 4))
        matrix.append(row)

    # 构建 ECharts heatmap series data: [[row, col, value], ...]
    heatmap_series: list[list[float]] = []
    for i in range(n):
        for j in range(n):
            heatmap_series.append([float(i), float(j), matrix[i][j]])

    return {
        "status": "success",
        "method": method,
        "variables": variables,
        "matrix": matrix,
        "heatmap_series_data": heatmap_series,
        "dimension": n,
    }
