#!/usr/bin/env python3
"""Model B NLP 真实文本引擎训练脚本 — IMCS-V2 儿科问诊数据.

基于 IMCS-V2_dev.json 的 833 条真实儿科医患对话记录, 提取
self_report (患者主诉) 作为输入文本, diagnosis (诊断结论) 作为
多分类标签, 训练 TF-IDF + SVM 分类器并持久化保存。

训练流程:
  1. 解析 IMCS-V2 JSON, 提取 self_report → X, diagnosis → y
  2. Pipeline: TfidfVectorizer(char ngram 1-3) → SVC
  3. 80/20 划分, 评估, joblib 持久化

依赖: json, sklearn, joblib
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

try:
    import joblib
except ImportError:
    print("[FATAL] joblib 未安装，请执行: pip install joblib")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

_DATA_PATH = _PROJECT_ROOT / "data" / "raw" / "IMCS-V2_dev.json"
_MODEL_OUTPUT = _PROJECT_ROOT / "models" / "weights" / "model_B_nlp_v1.pkl"

# TF-IDF 参数
TFIDF_ANALYZER = "char"
TFIDF_NGRAM = (1, 3)
TFIDF_MAX_FEATURES = 3000

# SVM 参数
SVM_C = 1.0
SVM_KERNEL = "linear"
SVM_RANDOM_STATE = 42

# 数据划分
TEST_SIZE = 0.2
RANDOM_SEED = 42


# ===================================================================
# 数据加载
# ===================================================================


def load_imcs_data() -> tuple[list[str], list[str]]:
    """从 IMCS-V2_dev.json 加载 self_report 与 diagnosis。

    Returns:
        (texts, labels) 元组。
    """
    print("─" * 68)
    print("  [INFO] 正在解析 IMCS-V2 JSON 数据...")
    print("─" * 68)

    # 检查文件
    if not _DATA_PATH.is_file():
        raise FileNotFoundError(f"IMCS-V2 数据文件不存在: {_DATA_PATH}")

    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    print(f"    原始 JSON 记录数: {len(raw)}")

    texts: list[str] = []
    labels: list[str] = []
    skipped = 0

    for subject_id, record in raw.items():
        if not isinstance(record, dict):
            skipped += 1
            continue

        text = record.get("self_report", "").strip()
        label = record.get("diagnosis", "").strip()

        if not text or not label:
            skipped += 1
            continue

        texts.append(text)
        labels.append(label)

    # 类别分布
    from collections import Counter
    dist = Counter(labels)
    print(f"    有效记录数: {len(texts)} (跳过 {skipped} 条无效)")
    print(f"    类别数: {len(dist)}")
    print(f"    类别分布:")
    for cls, cnt in dist.most_common():
        bar = "█" * (cnt // 5)
        print(f"      {cls:<12} {cnt:>4}  {bar}")

    print(f"    [INFO] 已成功解析 IMCS-V2 JSON，共处理 {len(texts)} 条有效记录...")
    print()

    return texts, labels


# ===================================================================
# 模型训练、评估与持久化
# ===================================================================


def train_evaluate_save(texts: list[str], labels: list[str]) -> None:
    """构建 TF-IDF + SVM Pipeline, 训练、评估并保存。

    Args:
        texts: 主诉文本列表。
        labels: 诊断标签列表。
    """
    print("─" * 68)
    print("  [INFO] 构建 NLP Pipeline 并训练 SVM 模型...")
    print("─" * 68)

    # ---- 1. 划分训练/测试集 ----
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    print(f"    训练集: {len(X_train)}  测试集: {len(X_test)}")
    print(f"    TF-IDF: analyzer='{TFIDF_ANALYZER}', "
          f"ngram_range={TFIDF_NGRAM}, max_features={TFIDF_MAX_FEATURES}")

    # ---- 2. Pipeline ----
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                analyzer=TFIDF_ANALYZER,
                ngram_range=TFIDF_NGRAM,
                max_features=TFIDF_MAX_FEATURES,
            )),
            ("svm", SVC(
                C=SVM_C,
                kernel=SVM_KERNEL,
                random_state=SVM_RANDOM_STATE,
                probability=True,  # 保留概率输出供后续 SHAP 使用
            )),
        ]
    )

    # ---- 3. 训练 ----
    pipeline.fit(X_train, y_train)
    print(f"    SVM 训练完成 (kernel={SVM_KERNEL}, C={SVM_C})")

    # ---- 4. 评估 ----
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n    ────────────────────────────────────────────────")
    print(f"    测试集准确率: {acc:.4f} ({acc:.2%})")
    print(f"    ────────────────────────────────────────────────")
    print(f"    分类报告:")
    print()

    report = classification_report(y_test, y_pred)
    for line in report.split("\n"):
        if line.strip():
            print(f"    {line}")

    # ---- 5. 持久化 ----
    weight_dir = os.path.dirname(str(_MODEL_OUTPUT))
    os.makedirs(weight_dir, exist_ok=True)

    joblib.dump(pipeline, str(_MODEL_OUTPUT))

    file_size_kb = os.path.getsize(str(_MODEL_OUTPUT)) / 1024
    print(f"\n{'─' * 68}")
    print(f"  [SUCCESS] Model B 真实文本引擎训练完毕，已保存！")
    print(f"    权重路径: {_MODEL_OUTPUT}")
    print(f"    文件大小: {file_size_kb:.1f} KB")
    print(f"    包含: TfidfVectorizer + SVM Pipeline")
    print(f"{'─' * 68}")


# ===================================================================
# 主流程
# ===================================================================

if __name__ == "__main__":
    print("=" * 68)
    print("  Model B NLP 真实文本引擎训练")
    print("  数据: IMCS-V2_dev.json — 儿科医患对话")
    print("=" * 68)
    print()

    # ---- 1. 加载 ----
    texts, labels = load_imcs_data()

    # ---- 2. 训练 & 保存 ----
    train_evaluate_save(texts, labels)

    # ---- 统计摘要 ----
    print(f"\n{'=' * 68}")
    print(f"  训练流水线统计")
    print(f"{'=' * 68}")
    print(f"  数据来源:       IMCS-V2_dev.json")
    print(f"  有效记录:       {len(texts)} 条")
    print(f"  类别数:         {len(set(labels))} 类")
    print(f"  Pipeline:       TfidfVectorizer(char 1-3, 3000) → LinearSVC")
    print(f"  模型输出:       {_MODEL_OUTPUT}")
    print(f"{'=' * 68}\n")
