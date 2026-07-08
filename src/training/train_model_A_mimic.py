#!/usr/bin/env python3
"""XGBoost Model A 真实离线训练脚本 — MIMIC-IV Demo 数据.

基于 MIMIC-IV Demo 的 100 名真实患者血检时序数据, 提取时序特征,
训练面向"肾功能异常预测"的 XGBoost 分类器, 并将权重持久化保存。

训练流程:
  1. 加载 labevents + patients 真实表
  2. ETL 多表关联: 按 subject_id 分组提取 GLU/Creatinine 时序统计量
  3. 靶标构造: Creatinine_mean > 1.2 → 肾功能异常 (target=1)
  4. XGBoost 训练 + 评估 + joblib 持久化

依赖: pandas, numpy, xgboost, sklearn, joblib
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except ImportError:
    print("[FATAL] xgboost 未安装，请执行: pip install xgboost")
    sys.exit(1)

try:
    import joblib
except ImportError:
    print("[FATAL] joblib 未安装，请执行: pip install joblib")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 文件路径常量化
# ---------------------------------------------------------------------------

# 优先读取 .csv，不存在则回退到 .csv.gz
_RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "mimic_demo"

_PATIENTS_CSV = _RAW_DIR / "patients.csv"
_PATIENTS_GZ = _RAW_DIR / "patients.csv.gz"
_LABS_CSV = _RAW_DIR / "labevents.csv"
_LABS_GZ = _RAW_DIR / "labevents.csv.gz"

# MIMIC-IV 字典: itemid → 指标名
ITEMID_CREATININE: int = 50912   # 肌酐
ITEMID_GLUCOSE: int = 50931     # 血糖

# 临床阈值: 肌酐均值 > 1.2 mg/dL → 肾功能异常
CREATININE_THRESHOLD: float = 1.2

# 模型保存路径
_MODEL_OUTPUT = _PROJECT_ROOT / "models" / "weights" / "model_A_xgboost_v1.pkl"


# ===================================================================
# 数据加载与安全校验
# ===================================================================


def _resolve_path(csv_path: Path, gz_path: Path) -> Path:
    """解析文件路径: 优先 .csv，回退 .csv.gz。

    Args:
        csv_path: .csv 文件路径。
        gz_path: .csv.gz 文件路径。

    Returns:
        存在的文件 Path。

    Raises:
        FileNotFoundError: 两个路径均不存在。
    """
    if csv_path.is_file():
        return csv_path
    if gz_path.is_file():
        return gz_path
    raise FileNotFoundError(
        f"未找到 MIMIC Demo 数据，请先完成上传。\n"
        f"  期望: {csv_path} 或 {gz_path}"
    )


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载并校验 labevents 与 patients 表。

    Returns:
        (df_labs, df_patients) 元组。
    """
    print("=" * 68)
    print("  [INFO] 正在读取真实 MIMIC-IV Demo 数据...")
    print("=" * 68)

    # ---- 解析路径 ----
    labs_final = _resolve_path(_LABS_CSV, _LABS_GZ)
    pats_final = _resolve_path(_PATIENTS_CSV, _PATIENTS_GZ)

    compression = "gzip" if labs_final.suffix == ".gz" else None
    print(f"    labevents:  {labs_final.name} ({'gzip' if compression else 'csv'})")
    df_labs = pd.read_csv(labs_final, compression=compression)

    compression_p = "gzip" if pats_final.suffix == ".gz" else None
    print(f"    patients:   {pats_final.name} ({'gzip' if compression_p else 'csv'})")
    df_patients = pd.read_csv(pats_final, compression=compression_p)

    # ---- 基础校验 ----
    print(f"    labevents 行数: {len(df_labs):,}")
    print(f"    patients  行数: {len(df_patients):,}")
    print(f"    唯一 subject_id: labevents={df_labs['subject_id'].nunique()}, "
          f"patients={df_patients['subject_id'].nunique()}")

    # 校验关键列
    for col in ("subject_id", "itemid", "valuenum"):
        if col not in df_labs.columns:
            raise ValueError(f"labevents 缺少关键列: {col}")
    for col in ("subject_id", "anchor_age"):
        if col not in df_patients.columns:
            raise ValueError(f"patients 缺少关键列: {col}")

    # 校验 itemid
    for itemid, desc in [(ITEMID_CREATININE, "Creatinine"), (ITEMID_GLUCOSE, "Glucose")]:
        if itemid not in df_labs["itemid"].unique():
            print(f"    ⚠ WARNING: itemid={itemid} ({desc}) 在数据中不存在!")

    print(f"    ✓ 数据加载完成\n")
    return df_labs, df_patients


# ===================================================================
# ETL: 多表关联与特征提取
# ===================================================================


def etl_feature_engineering(
    df_labs: pd.DataFrame, df_patients: pd.DataFrame
) -> pd.DataFrame:
    """从 labevents 提取时序统计特征, 与 patients 表宽关联。

    Args:
        df_labs: labevents 原始表。
        df_patients: patients 原始表。

    Returns:
        患者全景宽表 DataFrame, 含 anchor_age, GLU_mean, GLU_latest,
        Creatinine_mean, target 等列。
    """
    print("─" * 68)
    print("  [INFO] ETL 特征工程: 多表关联与特征提取...")
    print("─" * 68)

    # ---- Step 1: 筛选目标指标 ----
    mask_cre = df_labs["itemid"] == ITEMID_CREATININE
    mask_glu = df_labs["itemid"] == ITEMID_GLUCOSE

    df_cre = df_labs[mask_cre][["subject_id", "valuenum"]].dropna(subset=["valuenum"])
    df_glu = df_labs[mask_glu][["subject_id", "valuenum"]].dropna(subset=["valuenum"])

    print(f"    肌酐 (itemid={ITEMID_CREATININE}) 有效记录: {len(df_cre):,}")
    print(f"    血糖 (itemid={ITEMID_GLUCOSE}) 有效记录: {len(df_glu):,}")

    # ---- Step 2: 分组聚合 ----
    # 肌酐: 均值
    cre_feat = (
        df_cre.groupby("subject_id")["valuenum"]
        .agg(Creatinine_mean="mean")
        .reset_index()
    )

    # 血糖: 均值 + 最新值 (最后一条)
    glu_agg = df_glu.groupby("subject_id")["valuenum"].agg(
        GLU_mean="mean",
        GLU_latest="last",  # pandas 按文件顺序取最后一条
    ).reset_index()

    print(f"    肌酐特征患者数: {len(cre_feat)}")
    print(f"    血糖特征患者数: {len(glu_agg)}")

    # ---- Step 3: 宽表关联 ----
    # patients → creatinine → glucose
    df_pat_features = df_patients[["subject_id", "anchor_age", "gender"]].copy()
    df_wide = df_pat_features.merge(cre_feat, on="subject_id", how="left")
    df_wide = df_wide.merge(glu_agg, on="subject_id", how="left")

    # ---- Step 4: 缺失值填补 ----
    # 使用中位数填补数值特征, 避免极端值干扰
    df_wide["Creatinine_mean"] = df_wide["Creatinine_mean"].fillna(
        df_wide["Creatinine_mean"].median()
    )
    df_wide["GLU_mean"] = df_wide["GLU_mean"].fillna(
        df_wide["GLU_mean"].median()
    )
    df_wide["GLU_latest"] = df_wide["GLU_latest"].fillna(
        df_wide["GLU_latest"].median()
    )

    print(f"    缺失值填补: [Creatinine_mean]={df_wide['Creatinine_mean'].isna().sum()} "
          f"[GLU_mean]={df_wide['GLU_mean'].isna().sum()} "
          f"[GLU_latest]={df_wide['GLU_latest'].isna().sum()}")

    print(f"    宽表行数: {len(df_wide)}, 列数: {len(df_wide.columns)}")
    print(f"    患者覆盖率: {df_wide[['Creatinine_mean','GLU_mean']].notna().all(axis=1).sum()}/{len(df_wide)}")
    print(f"    ✓ ETL 特征工程完成，患者宽表已建立。\n")
    return df_wide


# ===================================================================
# 靶标构造
# ===================================================================


def build_target(df_wide: pd.DataFrame) -> pd.DataFrame:
    """基于临床阈值生成肾功能异常预测靶标。

    Args:
        df_wide: 患者全景宽表。

    Returns:
        含 target 列的完整宽表。
    """
    print("─" * 68)
    print("  [INFO] 靶标构造: Creatinine_mean > 1.2 mg/dL → 肾功能异常")
    print("─" * 68)

    df_wide["target"] = (df_wide["Creatinine_mean"] > CREATININE_THRESHOLD).astype(int)

    n_pos = df_wide["target"].sum()
    n_neg = len(df_wide) - n_pos
    print(f"    正样本 (肾功能异常): {n_pos} ({n_pos / len(df_wide):.1%})")
    print(f"    负样本 (肾功能正常): {n_neg} ({n_neg / len(df_wide):.1%})")

    if n_pos == 0 or n_neg == 0:
        print("    ⚠ WARNING: 标签分布极端，请检查临床阈值设置!")
    print(f"    ✓ 靶标构造完成。\n")
    return df_wide


# ===================================================================
# XGBoost 训练与评估
# ===================================================================


def train_and_save(df_wide: pd.DataFrame) -> None:
    """XGBoost 模型训练、评估与持久化保存。

    Args:
        df_wide: 含特征列与 target 列的患者宽表。
    """
    print("─" * 68)
    print("  [INFO] XGBoost 模型训练与评估...")
    print("─" * 68)

    # ---- 特征集与标签 ----
    FEATURE_COLS = ["anchor_age", "GLU_mean", "GLU_latest", "Creatinine_mean"]

    # 确保所有特征列存在
    for col in FEATURE_COLS:
        if col not in df_wide.columns:
            raise ValueError(f"特征列 {col} 不存在于宽表中!")

    X = df_wide[FEATURE_COLS].values
    y = df_wide["target"].values

    print(f"    特征矩阵形状: {X.shape}")
    print(f"    特征列表: {FEATURE_COLS}")

    # ---- 划分训练/测试集 ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"    训练集: {len(X_train)}  测试集: {len(X_test)}")

    # ---- 初始化 XGBoost ----
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )

    # ---- 训练 ----
    model.fit(X_train, y_train)
    print(f"    ✓ 模型训练完成 (n_estimators=50, max_depth=4, lr=0.1)")

    # ---- 评估 ----
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n    测试集准确率 (Accuracy): {acc:.4f} ({acc:.2%})")
    print(f"\n    分类报告 (Classification Report):")
    print(f"    {'─' * 52}")
    report = classification_report(y_test, y_pred, target_names=["正常", "异常"])
    for line in report.split("\n"):
        if line.strip():
            print(f"    {line}")

    # ---- 特征重要性 ----
    print(f"\n    特征重要性 (Feature Importances):")
    print(f"    {'─' * 48}")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"    {'特征':<28} {'重要性':<10} {'占比'}")
    print(f"    {'─' * 48}")
    for i in sorted_idx:
        name = FEATURE_COLS[i]
        imp = importances[i]
        bar = "█" * int(imp * 40)
        print(f"    {name:<28} {imp:<10.4f} {imp:.2%}  {bar}")

    # ---- 持久化保存 ----
    weight_dir = os.path.dirname(str(_MODEL_OUTPUT))
    os.makedirs(weight_dir, exist_ok=True)

    joblib.dump(model, str(_MODEL_OUTPUT))
    file_size_kb = os.path.getsize(str(_MODEL_OUTPUT)) / 1024
    print(f"\n{'─' * 68}")
    print(f"  [SUCCESS] 模型 Model A 训练完成！")
    print(f"    权重已保存至: {_MODEL_OUTPUT}")
    print(f"    文件大小: {file_size_kb:.1f} KB")
    print(f"{'─' * 68}")


# ===================================================================
# 主流程
# ===================================================================

if __name__ == "__main__":
    # ---- 1. 加载 ----
    df_labs, df_patients = load_raw_data()

    # ---- 2. ETL ----
    df_wide = etl_feature_engineering(df_labs, df_patients)

    # ---- 3. 靶标 ----
    df_wide = build_target(df_wide)

    # ---- 4. 训练 & 保存 ----
    train_and_save(df_wide)

    # ---- 统计摘要 ----
    print(f"\n{'=' * 68}")
    print(f"  训练流水线统计")
    print(f"{'=' * 68}")
    print(f"  原始 labevents 记录:  {len(df_labs):,}")
    print(f"  最终宽表患者数:       {len(df_wide)}")
    print(f"  特征维度:             {4}")
    print(f"  正样本率:             {df_wide['target'].mean():.1%}")
    print(f"  模型输出:             {_MODEL_OUTPUT}")
    print(f"{'=' * 68}\n")
