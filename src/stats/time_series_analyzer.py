#!/usr/bin/env python3
"""临床时间序列深度分析引擎 (Clinical Time-Series Analyzer).

本模块专注于解决临床生化数据因随访次数有限（短序列 3~15 个时间点）
带来的建模难点，提供三大核心能力：

  1. 时序轨迹特征提取 — 基线/峰值/斜率等关键临床统计量
  2. 单患者前瞻性趋势预测 — 线性/多项式回归短序列外推
  3. 群体时序轨迹聚类 — 自动发现疾病发展亚型曲线

架构: numpy 核心 + sklearn 可选加速，sklearn 未安装时自动降级
至纯 numpy 原生实现，保证零崩溃。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# sklearn 可选导入（静默降级）
# ---------------------------------------------------------------------------

_SKLEARN_AVAILABLE: bool = False
_Ridge: Any = None
_LinearRegression: Any = None
_KMeans: Any = None

try:
    from sklearn.linear_model import LinearRegression, Ridge

    _Ridge = Ridge
    _LinearRegression = LinearRegression
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass

try:
    from sklearn.cluster import KMeans

    _KMeans = KMeans
except ImportError:
    pass


# ===================================================================
# 纯 numpy K-Means 兜底实现
# ===================================================================


def _numpy_kmeans(
    X: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """纯 numpy 实现的 K-Means 聚类（sklearn 不可用时的兜底方案）。

    Args:
        X: 特征矩阵，shape (n_samples, n_features)。
        n_clusters: 聚类数。
        random_state: 随机种子。
        max_iter: 最大迭代次数。

    Returns:
        (labels, centroids) 元组。
    """
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]

    # 随机初始化质心：从数据中随机选 k 个点
    idx = rng.choice(n_samples, n_clusters, replace=False)
    centroids = X[idx].copy().astype(np.float64)

    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        # 分配：每个样本到最近质心
        distances = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            diff = X - centroids[k]
            distances[:, k] = np.sum(diff * diff, axis=1)
        new_labels = np.argmin(distances, axis=1)

        # 检查收敛
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        # 更新质心
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centroids[k] = X[mask].mean(axis=0)

    return labels, centroids


# ===================================================================
# TimeSeriesAnalyzer
# ===================================================================


class TimeSeriesAnalyzer:
    """临床时间序列深度分析引擎。

    基于 numpy 的高鲁棒性短序列建模，提供特征提取、趋势预测和
    群体轨迹聚类三大核心算子。

    Attributes:
        _sklearn_available: sklearn 是否可用。
    """

    def __init__(self) -> None:
        """初始化时序分析器。"""
        self._sklearn_available: bool = _SKLEARN_AVAILABLE

    # ------------------------------------------------------------------
    # 1. 时序轨迹特征提取
    # ------------------------------------------------------------------

    def extract_trajectory_features(self, values: list[float]) -> dict[str, Any]:
        """从一维时序序列中提取关键临床统计特征。

        适用于短序列（3~15 个时间点），计算基线、峰值、斜率等
        描述指标演变的统计量。

        Args:
            values: 按时间排序的浮点数列表，长度 >= 2。

        Returns:
            特征字典，含 baseline_value、latest_value、max_value、
            min_value、mean、std、slope、change_rate。

        Raises:
            ValueError: 若 values 长度不足 2。
        """
        if len(values) < 2:
            raise ValueError(f"时序序列长度不足: 需要 >= 2，实际 {len(values)}")

        arr = np.array(values, dtype=np.float64)
        x = np.arange(len(arr), dtype=np.float64)

        # 线性拟合斜率
        if len(arr) >= 2:
            slope = float(np.polyfit(x, arr, 1)[0])
        else:
            slope = 0.0

        change_rate = (arr[-1] - arr[0]) / arr[0] if arr[0] != 0 else 0.0

        return {
            "baseline_value": float(arr[0]),
            "latest_value": float(arr[-1]),
            "max_value": float(np.max(arr)),
            "min_value": float(np.min(arr)),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4),
            "slope": round(slope, 6),
            "change_rate": round(change_rate, 4),
        }

    # ------------------------------------------------------------------
    # 2. 单患者前瞻性趋势预测
    # ------------------------------------------------------------------

    def forecast_marker_trend(
        self,
        values: list[float],
        future_steps: int = 2,
        marker_name: str = "LAB",
    ) -> dict[str, Any]:
        """利用现有时序数据预测未来时间点的指标数值与趋势。

        使用多项式回归拟合短序列曲线。优先尝试二次多项式（≥4个点时），
        线性回归作为基础方案。

        Args:
            values: 按时间排序的历史检验值列表，长度 >= 2。
            future_steps: 预测未来步数。
            marker_name: 指标名称。

        Returns:
            标准化预测结果字典:
            {
                "marker_name": str,
                "historical_values": list,
                "forecasted_values": list,
                "trend_direction": "increasing"|"decreasing"|"stable",
                "risk_alert": bool,
            }
        """
        arr = np.array(values, dtype=np.float64)
        n = len(arr)
        x = np.arange(n, dtype=np.float64)

        # ---- 选择回归阶数 ----
        # 默认线性；>= 4 个点时尝试二次并择优
        degree = 1
        if n >= 4:
            # 比较线性与二次的拟合残差（使用可决系数近似）
            coef1 = np.polyfit(x, arr, 1)
            pred1 = np.polyval(coef1, x)
            rss1 = np.sum((arr - pred1) ** 2)

            coef2 = np.polyfit(x, arr, 2)
            pred2 = np.polyval(coef2, x)
            rss2 = np.sum((arr - pred2) ** 2)

            # 若二次拟合显著更好（残差降低 > 20%），选用二次
            if rss2 < rss1 * 0.8:
                degree = 2

        # ---- 拟合 ----
        coef = np.polyfit(x, arr, degree)

        # ---- 预测 ----
        future_x = np.arange(n, n + future_steps, dtype=np.float64)
        forecasted = np.polyval(coef, future_x)
        forecasted_rounded = [round(float(v), 2) for v in forecasted]

        # ---- 趋势方向判定 ----
        slope = float(np.polyfit(x, arr, 1)[0])
        if slope > 0.1:
            trend_direction = "increasing"
        elif slope < -0.1:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"

        # ---- 风险告警 ----
        # 条件：趋势上升 且 （最新值或预测值中的最大值处于高位）
        risk_alert = False
        if trend_direction == "increasing":
            latest = arr[-1]
            max_forecasted = max(forecasted_rounded) if forecasted_rounded else latest
            # 若最新值已超过历史均值的 1.15 倍，或预测值持续攀升
            mean_val = float(np.mean(arr))
            if latest > mean_val * 1.1 or max_forecasted > latest:
                risk_alert = True

        return {
            "marker_name": marker_name,
            "historical_values": [round(float(v), 2) for v in values],
            "forecasted_values": forecasted_rounded,
            "trend_direction": trend_direction,
            "risk_alert": risk_alert,
        }

    # ------------------------------------------------------------------
    # 3. 群体时序轨迹聚类
    # ------------------------------------------------------------------

    def cluster_patient_trajectories(
        self,
        cohort_data: dict[str, list[float]],
        n_clusters: int = 3,
    ) -> dict[str, Any]:
        """对多患者时序数据执行轨迹聚类，自动发现临床亚型。

        流程:
          1. 对每位患者提取 [baseline, latest, slope, std] 四维特征
          2. 标准化特征矩阵
          3. K-Means 聚类
          4. 根据每簇平均斜率自动赋予临床语义标签

        Args:
            cohort_data: {patient_id: [val1, val2, ...]} 群体随访字典。
            n_clusters: 聚类数。

        Returns:
            {
                "cluster_summary": [
                    {
                        "cluster_id": int,
                        "label": str,
                        "patient_count": int,
                        "mean_slope": float,
                        "avg_baseline": float,
                        "avg_latest": float,
                        "patient_ids": list,
                    },
                    ...
                ]
            }
        """
        patient_ids = list(cohort_data.keys())
        if len(patient_ids) < n_clusters:
            raise ValueError(
                f"患者数 ({len(patient_ids)}) 小于聚类数 ({n_clusters})"
            )

        # ---- Step 1: 特征提取 ----
        features_list: list[list[float]] = []
        for pid in patient_ids:
            vals = cohort_data[pid]
            feats = self.extract_trajectory_features(vals)
            features_list.append(
                [
                    feats["baseline_value"],
                    feats["latest_value"],
                    feats["slope"],
                    feats["std"],
                ]
            )

        X = np.array(features_list, dtype=np.float64)

        # ---- Step 2: 标准化 ----
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0  # 防止除零
        X_norm = (X - mean) / std

        # ---- Step 3: K-Means 聚类 ----
        if _KMeans is not None:
            kmeans = _KMeans(
                n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300
            )
            labels = kmeans.fit_predict(X_norm)
            centroids_norm = kmeans.cluster_centers_
        else:
            labels, centroids_norm = _numpy_kmeans(
                X_norm, n_clusters=n_clusters, random_state=42
            )

        # 反标准化质心
        centroids = centroids_norm * std + mean

        # ---- Step 4: 自动语义标注 ----
        cluster_summary: list[dict[str, Any]] = []
        for cid in range(n_clusters):
            mask = labels == cid
            member_ids = [patient_ids[i] for i, m in enumerate(mask) if m]
            count = len(member_ids)
            c_slope = float(centroids[cid, 2])  # slope 在第 2 列
            c_baseline = float(centroids[cid, 0])  # baseline 在第 0 列

            # 根据相对斜率自动标注
            label = self._label_cluster(c_slope, c_baseline)

            cluster_summary.append(
                {
                    "cluster_id": int(cid),
                    "label": label,
                    "patient_count": count,
                    "mean_slope": round(float(np.mean(X[mask, 2])), 6),
                    "avg_baseline": round(float(np.mean(X[mask, 0])), 4),
                    "avg_latest": round(float(np.mean(X[mask, 1])), 4),
                    "patient_ids": member_ids,
                }
            )

        # 按平均斜率降序排列
        cluster_summary.sort(key=lambda c: c["mean_slope"], reverse=True)

        return {"cluster_summary": cluster_summary}

    @staticmethod
    def _label_cluster(slope: float, baseline: float) -> str:
        """根据质心斜率与基线值自动赋予临床语义标签。

        使用相对斜率 (slope / baseline) 消除不同指标量纲差异，
        使阈值适用于肌酐 (μmol/L)、血糖 (mmol/L) 等任意单位。

        Args:
            slope: 聚类中心的平均斜率。
            baseline: 聚类中心的平均基线值。

        Returns:
            临床语义标签字符串。
        """
        if baseline <= 0:
            return "数据异常"

        rel_slope = slope / baseline  # 每步的相对变化率

        if rel_slope > 0.02:  # 每步增长 > 2%
            return "快速恶化/上升亚型"
        elif rel_slope > 0.005:  # 每步增长 > 0.5%
            return "缓慢进展亚型"
        elif rel_slope > -0.005:  # 波动在 ±0.5% 内
            return "平稳控制亚型"
        elif rel_slope > -0.02:  # 每步下降 > 0.5%
            return "缓慢改善亚型"
        else:
            return "治疗改善/下降亚型"


# ===================================================================
# 验证与自测模块
# ===================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  临床时间序列深度分析引擎 (TimeSeriesAnalyzer) — 功能验证")
    print("=" * 72)

    analyzer = TimeSeriesAnalyzer()
    print(f"\n[环境] sklearn 可用: {analyzer._sklearn_available}")

    # ================================================================
    # 测试场景 1：单患者前瞻预测
    # ================================================================
    print(f"\n{'─' * 72}")
    print("  【测试场景 1】糖尿病肾病并发患者 — 空腹血糖 GLU 前瞻预测")
    print(f"{'─' * 72}")

    glu_values = [6.2, 6.9, 7.4, 8.1, 8.8]

    # 先提取特征
    features = analyzer.extract_trajectory_features(glu_values)
    print(f"\n  [时序特征提取]")
    print(f"    基线值 (baseline):  {features['baseline_value']:.1f} mmol/L")
    print(f"    最新值 (latest):    {features['latest_value']:.1f} mmol/L")
    print(f"    峰值 (max):         {features['max_value']:.1f} mmol/L")
    print(f"    谷值 (min):         {features['min_value']:.1f} mmol/L")
    print(f"    均值 (mean):        {features['mean']:.2f} mmol/L")
    print(f"    标准差 (std):        {features['std']:.2f}")
    print(f"    斜率 (slope):        {features['slope']:.4f}")
    print(f"    变化率 (change):     {features['change_rate']:.2%}")

    # 预测
    forecast_result = analyzer.forecast_marker_trend(
        values=glu_values,
        future_steps=2,
        marker_name="GLU",
    )

    print(f"\n  [前瞻预测结果]")
    print(f"    指标名称:       {forecast_result['marker_name']}")
    print(f"    历史序列:       {forecast_result['historical_values']}")
    print(f"    预测值 (未来2步): {forecast_result['forecasted_values']}")
    print(f"    趋势方向:       {forecast_result['trend_direction']}")

    # 风险高亮
    if forecast_result["risk_alert"]:
        print(f"\n    ⚠  RISK ALERT: 趋势持续恶化！血糖处于高危上升通道！")
        print(f"        当前值 {forecast_result['historical_values'][-1]}")
        print(f"        预计值 {forecast_result['forecasted_values']}")
    else:
        print(f"    ✓  风险可控，未触发警报")

    # 趋势可视化
    print(f"\n  [趋势曲线示意]")
    hist = forecast_result["historical_values"]
    fcast = forecast_result["forecasted_values"]
    all_vals = hist + fcast
    max_val = max(all_vals)
    min_val = min(all_vals)
    bar_width = 40
    for i, v in enumerate(hist):
        bar_len = int((v - min_val) / (max_val - min_val + 0.001) * bar_width)
        marker = f"T{i+1}"
        print(f"    {marker:<6} {'█' * bar_len} {v:.1f}")
    print(f"    {'─' * 48}")
    for i, v in enumerate(fcast):
        bar_len = int((v - min_val) / (max_val - min_val + 0.001) * bar_width)
        marker = f"T{len(hist)+i+1}*"
        print(f"    {marker:<6} {'▓' * bar_len} {v:.1f}  ← 预测")

    # ================================================================
    # 测试场景 2：群体亚型聚类
    # ================================================================
    print(f"\n\n{'─' * 72}")
    print("  【测试场景 2】30 名患者 — 肌酐 Creatinine 轨迹亚型聚类")
    print(f"{'─' * 72}")

    # 构造虚拟数据
    np.random.seed(42)
    cohort: dict[str, list[float]] = {}
    time_points = np.arange(1, 7, dtype=np.float64)  # 6 次随访

    # 类型 A (10人): 快速上升 — 斜率 ~0.3，从 80 升到 ~230
    for i in range(10):
        base = np.random.uniform(70, 90)
        noise = np.random.normal(0, 5, len(time_points))
        vals = base + time_points * np.random.uniform(22, 30) + noise
        vals = np.clip(vals, 60, 300)
        cohort[f"P_rising_{i+1:02d}"] = [round(float(v), 1) for v in vals]

    # 类型 B (10人): 平稳控制 — 斜率 ~0，在 ~100 附近微波动
    for i in range(10):
        base = np.random.uniform(90, 110)
        noise = np.random.normal(0, 3, len(time_points))
        vals = base + time_points * np.random.uniform(-0.5, 0.5) + noise
        vals = np.clip(vals, 60, 130)
        cohort[f"P_stable_{i+1:02d}"] = [round(float(v), 1) for v in vals]

    # 类型 C (10人): 治疗改善 — 斜率 ~-3，从 ~200 降到 ~85
    for i in range(10):
        base = np.random.uniform(180, 220)
        noise = np.random.normal(0, 8, len(time_points))
        vals = base - time_points * np.random.uniform(18, 25) + noise
        vals = np.clip(vals, 60, 250)
        cohort[f"P_improve_{i+1:02d}"] = [round(float(v), 1) for v in vals]

    print(f"\n  [数据生成] 共 {len(cohort)} 名虚拟患者")
    print(f"     上升组: 10 人 (slope > 0)")
    print(f"     平稳组: 10 人 (slope ≈ 0)")
    print(f"     改善组: 10 人 (slope < 0)")

    # 展示各类型代表患者
    for label, pids in [
        ("上升组", [f"P_rising_0{i}" for i in range(1, 4)]),
        ("平稳组", [f"P_stable_0{i}" for i in range(1, 4)]),
        ("改善组", [f"P_improve_0{i}" for i in range(1, 4)]),
    ]:
        print(f"\n    [{label} 代表]")
        for pid in pids:
            vals = cohort[pid]
            arrow = " ↗" if vals[-1] > vals[0] else (" ↘" if vals[-1] < vals[0] else " →")
            print(f"      {pid}: {vals} {arrow}")

    # ---- 聚类 ----
    cluster_result = analyzer.cluster_patient_trajectories(
        cohort_data=cohort, n_clusters=3
    )

    print(f"\n  [聚类结果] 共发现 {len(cluster_result['cluster_summary'])} 个临床亚型\n")
    print(
        f"  {'簇ID':<6} {'临床亚型标签':<20} {'人数':<6} {'平均斜率':<10} "
        f"{'基线均值':<10} {'最新均值':<10}"
    )
    print(f"  {'─' * 66}")

    for cluster in cluster_result["cluster_summary"]:
        cid = cluster["cluster_id"]
        label = cluster["label"]
        count = cluster["patient_count"]
        slope = cluster["mean_slope"]
        baseline = cluster["avg_baseline"]
        latest = cluster["avg_latest"]
        print(
            f"  {cid:<6} {label:<20} {count:<6} "
            f"{slope:<10.4f} {baseline:<10.1f} {latest:<10.1f}"
        )

    # 每个簇的代表患者
    print(f"\n  [各亚型归属详情]\n")
    for cluster in cluster_result["cluster_summary"]:
        label = cluster["label"]
        ids = cluster["patient_ids"]
        # 只展示前 5 个代表
        show_ids = ids[:5]
        suffix = f"... 等共 {len(ids)} 人" if len(ids) > 5 else f" 共 {len(ids)} 人"
        print(f"  ┌ {label}")
        print(f"  │   患者: {', '.join(show_ids)}{suffix}")
        # 展示第一个患者的轨迹
        first_pid = ids[0]
        vals = cohort[first_pid]
        slope_val = analyzer.extract_trajectory_features(vals)["slope"]
        direction = "↗" if slope_val > 0.02 else ("↘" if slope_val < -0.02 else "→")
        print(f"  │   示例 ({first_pid}): {vals} {direction}")

    print(f"\n\n{'=' * 72}")
    print("  全场景验证完毕！")
    print("=" * 72)
