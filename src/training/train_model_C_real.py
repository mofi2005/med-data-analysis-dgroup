#!/usr/bin/env python3
"""Model C 跨模态大模型真实训练脚本 — Breast Cancer Radiomics + 临床特征融合.

基于 sklearn Breast Cancer 真实影像组学数据 (569 患者 × 30 维图像特征),
模拟 D 组临床特征 (anchor_age, CEA_level) 与 E 组组学跨模态拼接,
训练 MLP 神经网络进行肿瘤良恶性预测, 并保存模型权重。

训练流程:
  1. 加载真实 E 组组学数据 (load_breast_cancer, as_frame=True)
  2. 模拟 D 组临床特征 (anchor_age, CEA_level) 并 Concat
  3. StandardScaler 归一化 → MLPClassifier 训练
  4. 评估 + joblib 持久化

依赖: numpy, pandas, sklearn, joblib
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
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ImportError:
    print("[FATAL] joblib 未安装，请执行: pip install joblib")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

_MODEL_OUTPUT = _PROJECT_ROOT / "models" / "weights" / "model_C_multimodal_v1.pkl"

# D 组模拟临床特征参数
_AGE_MIN, _AGE_MAX = 30, 80
_CEA_MEAN, _CEA_STD = 3.5, 2.0  # 肿瘤标志物 CEA 分布参数

# MLP 超参数
MLP_HIDDEN = (64, 32)
MLP_MAX_ITER = 1000
TEST_SIZE = 0.2
RANDOM_SEED = 42


# ===================================================================
# 数据加载与跨模态融合
# ===================================================================


def load_and_fuse_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """加载真实 E 组组学数据, 模拟 D 组临床特征, 完成跨模态拼接。

    Returns:
        (X, y, feature_names) 元组:
          - X: 融合后的特征矩阵 (n_samples × n_features)
          - y: 真实靶标标签
          - feature_names: 全量特征名列表
    """
    # ---- Step 1: E 组真实组学数据 ----
    print("─" * 68)
    print("  [INFO] 真实影像组学特征加载中 (Breast Cancer Dataset)...")
    print("─" * 68)

    data = load_breast_cancer(as_frame=True)
    df_radiomics = data.data  # DataFrame: 569 × 30
    y = data.target.values    # 0=恶性, 1=良性

    n_samples = len(df_radiomics)
    n_rad_features = df_radiomics.shape[1]

    print(f"    E 组组学数据: {n_samples} 患者 × {n_rad_features} 维图像提取特征")
    print(f"    正样本 (良性): {y.sum()} ({y.sum() / n_samples:.1%})")
    print(f"    负样本 (恶性): {n_samples - y.sum()} ({(1 - y.sum() / n_samples):.1%})")
    print(f"    ✓ 真实影像组学特征加载成功，维度: {df_radiomics.shape}")
    print()

    # ---- Step 2: D 组模拟临床特征 ----
    print("─" * 68)
    print("  [INFO] 临床+组学跨模态融合中...")
    print("─" * 68)

    rng = np.random.RandomState(RANDOM_SEED)
    anchor_age = rng.randint(_AGE_MIN, _AGE_MAX + 1, size=n_samples).astype(np.float64)
    cea_level = np.abs(rng.normal(_CEA_MEAN, _CEA_STD, size=n_samples))

    df_clinical = np.column_stack([anchor_age, cea_level])
    clinical_names = ["anchor_age", "CEA_level"]

    print(f"    D 组模拟临床特征: {2} 维")
    print(f"      anchor_age: [{anchor_age.min():.0f}, {anchor_age.max():.0f}], "
          f"均值 {anchor_age.mean():.1f}")
    print(f"      CEA_level:  [{cea_level.min():.2f}, {cea_level.max():.2f}], "
          f"均值 {cea_level.mean():.2f}")

    # ---- Step 3: 跨模态 Concat ----
    X = np.column_stack([df_radiomics.values, df_clinical])
    feature_names = list(df_radiomics.columns) + clinical_names

    total_dim = X.shape[1]
    print(f"    融合后特征维度: {n_rad_features} (组学) + {2} (临床) = {total_dim}")
    print(f"    ✓ 临床+组学跨模态融合完成, 全景宽表: {X.shape}")
    print()

    return X, y, feature_names


# ===================================================================
# 模型训练、评估与持久化
# ===================================================================


def train_evaluate_save(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> None:
    """训练 MLP 多模态模型, 评估并持久化保存。

    Args:
        X: 融合特征矩阵。
        y: 靶标标签。
        feature_names: 特征名列表。
    """
    print("─" * 68)
    print("  [INFO] 多模态大模型训练中 (MLP 神经网络)...")
    print("─" * 68)

    # ---- 1. 划分训练/测试集 ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )
    print(f"    训练集: {len(X_train)}  测试集: {len(X_test)}")
    print(f"    特征维度: {X_train.shape[1]} (组学 {X.shape[1] - 2} + 临床 2)")

    # ---- 2. 标准化 (异构模态强制归一化) ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"    StandardScaler 归一化完成")

    # ---- 3. MLP 训练 ----
    mlp = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN,
        max_iter=MLP_MAX_ITER,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    mlp.fit(X_train_scaled, y_train)
    print(f"    MLP 结构: 输入 {X_train.shape[1]} → 隐藏 {MLP_HIDDEN} → 输出 2")
    print(f"    实际迭代次数: {mlp.n_iter_}")
    print(f"    最终损失: {mlp.loss_:.6f}")

    # ---- 4. 评估 ----
    y_pred = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n    ────────────────────────────────────────────")
    print(f"    测试集准确率: {acc:.4f} ({acc:.2%})")
    print(f"    ────────────────────────────────────────────")
    print(f"    分类报告:")
    print()
    report = classification_report(
        y_test, y_pred,
        target_names=["恶性 (Malignant)", "良性 (Benign)"],
    )
    for line in report.split("\n"):
        if line.strip():
            print(f"    {line}")

    # ---- 5. 持久化 (scaler + model 一起打包) ----
    weight_dir = os.path.dirname(str(_MODEL_OUTPUT))
    os.makedirs(weight_dir, exist_ok=True)

    bundle = {
        "model": mlp,
        "scaler": scaler,
        "feature_names": feature_names,
        "mlp_hidden": MLP_HIDDEN,
        "n_features": X.shape[1],
    }
    joblib.dump(bundle, str(_MODEL_OUTPUT))

    file_size_kb = os.path.getsize(str(_MODEL_OUTPUT)) / 1024
    print(f"\n{'─' * 68}")
    print(f"  [SUCCESS] Model C 多模态大模型训练完毕并保存！")
    print(f"    权重路径: {_MODEL_OUTPUT}")
    print(f"    文件大小: {file_size_kb:.1f} KB")
    print(f"    包含: MLP 模型 + StandardScaler + 特征名映射")
    print(f"{'─' * 68}")


# ===================================================================
# 主流程
# ===================================================================

if __name__ == "__main__":
    print("=" * 68)
    print("  Model C 跨模态大模型真实训练")
    print("  E 组 (Breast Cancer Radiomics) + D 组 (Clinical Simulation)")
    print("=" * 68)
    print()

    # ---- 1. 加载 & 融合 ----
    X, y, feature_names = load_and_fuse_data()

    # ---- 2. 训练 & 保存 ----
    train_evaluate_save(X, y, feature_names)

    # ---- 统计摘要 ----
    print(f"\n{'=' * 68}")
    print(f"  训练流水线统计")
    print(f"{'=' * 68}")
    print(f"  数据来源:       sklearn Breast Cancer (真实影像组学)")
    print(f"  总样本数:       {X.shape[0]}")
    print(f"  总特征维度:     {X.shape[1]} (30 组学 + 2 临床)")
    print(f"  模型架构:       MLP {X.shape[1]}→{MLP_HIDDEN[0]}→{MLP_HIDDEN[1]}→2")
    print(f"  模型输出:       {_MODEL_OUTPUT}")
    print(f"{'=' * 68}\n")
