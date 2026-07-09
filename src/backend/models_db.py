#!/usr/bin/env python3
"""物理表 ORM 模型定义 (SQLAlchemy Declarative Models).

本模块基于 database.Base 定义项目核心 4 张物理表:
  1. Dataset        — 数据集表
  2. DataDictionary — 字段字典表
  3. AnalysisTask   — 分析任务表
  4. ModelResult    — 模型结果表
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.orm import declarative_base

# 复用 database.py 中已创建的 Base
from src.backend.database import Base


# ---------------------------------------------------------------------------
# 1. 数据集表
# ---------------------------------------------------------------------------

class Dataset(Base):
    """数据集表 —— 存储各组上传的原始数据文件元信息。

    字段说明:
      - dataset_id:   数据集唯一标识 (UUID4)
      - dataset_name: 数据集名称
      - data_type:    数据类型 (Excel / CSV / JSON)
      - source_type:  来源组别 (D组 / C组 / E组)
      - file_path:    文件物理存储路径
      - row_count:    数据行数
      - column_count: 数据列数
      - upload_user:  上传人标识
      - created_time: 记录创建时间
    """

    __tablename__ = "datasets"

    dataset_id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="数据集唯一标识 (UUID4)",
    )
    dataset_name = Column(
        String(255), nullable=False, default="未命名数据集", comment="数据集名称"
    )
    data_type = Column(
        String(50), nullable=False, default="CSV", comment="数据类型: Excel/CSV/JSON"
    )
    source_type = Column(
        String(50), nullable=False, default="D组", comment="来源组别: D组/C组/E组"
    )
    file_path = Column(
        String(500), nullable=True, comment="文件存储路径"
    )
    row_count = Column(
        Integer, nullable=True, default=0, comment="数据行数"
    )
    column_count = Column(
        Integer, nullable=True, default=0, comment="数据列数"
    )
    upload_user = Column(
        String(100), nullable=False, default="member1", comment="上传人标识"
    )
    created_time = Column(
        DateTime, nullable=False, default=datetime.now, comment="记录创建时间"
    )


# ---------------------------------------------------------------------------
# 5. MLOps 动态监控配置表 (自进化系统元数据)
# ---------------------------------------------------------------------------

class MLOpsConfig(Base):
    """MLOps 动态监控配置表 —— 管理模型自进化的全局元数据。

    为持续训练 (Continuous Training) 与自动评分自进化引擎
    提供持久化的配置参数存储, 支持运行时动态读写。

    字段说明:
      - key:          参数键名 (主键, 唯一索引)
      - value:        参数当前数值 (Float)
      - description:  参数中文描述
      - updated_at:   最近更新时间
    """

    __tablename__ = "mlops_config"

    key = Column(
        String(100),
        primary_key=True,
        unique=True,
        index=True,
        comment="MLOps 参数键名 (主键唯一索引)",
    )
    value = Column(
        Float, nullable=False, default=0.0, comment="参数当前数值"
    )
    description = Column(
        String(500), nullable=True, comment="参数中文描述"
    )
    updated_at = Column(
        DateTime, nullable=False, default=datetime.now, comment="最近更新时间"
    )


# ---------------------------------------------------------------------------
# MLOps 默认种子参数 (幂等: 已有则不覆盖)
# ---------------------------------------------------------------------------

DEFAULT_MLOPS_CONFIG: list[dict] = [
    {"key": "model_a_base_score", "value": 85.0,
     "description": "Model A(生化时序XGBoost)当前基础战绩P_base"},
    {"key": "model_b_base_score", "value": 80.0,
     "description": "Model B(NLP临床文本)当前基础战绩P_base"},
    {"key": "model_c_base_score", "value": 90.0,
     "description": "Model C(深度多模态大模型)当前基础战绩P_base"},
    {"key": "new_samples_since_last_ct", "value": 0.0,
     "description": "自上次自动重新训练以来新积累的病历数据条数"},
    {"key": "ct_trigger_threshold", "value": 50.0,
     "description": "触发自动持续训练(CT)的新数据积累条数阈值"},
]


def seed_mlops_config(session):
    """幂等地初始化 MLOps 自进化配置默认参数。

    若 mlops_config 表中已有记录则跳过, 不会覆盖用户手动调整后的值。

    Args:
        session: SQLAlchemy Session 实例。
    """
    from sqlalchemy import exists as _exists

    count = session.query(MLOpsConfig).count()
    if count > 0:
        return  # 已有数据，跳过

    now = datetime.now()
    for item in DEFAULT_MLOPS_CONFIG:
        session.add(MLOpsConfig(
            key=item["key"],
            value=item["value"],
            description=item["description"],
            updated_at=now,
        ))

    session.commit()
    print(f"[INFO] MLOps 自进化配置已初始化: {len(DEFAULT_MLOPS_CONFIG)} 条默认参数入库")


# ---------------------------------------------------------------------------
# 2. 字段字典表
# ---------------------------------------------------------------------------

class DataDictionary(Base):
    """字段字典表 —— 医学标准字段的元数据字典。

    为 NLP 映射引擎与前端字段选择器提供标准化的字段定义,
    涵盖血常规、肿瘤标志物、肝功能等临床类别。

    字段说明:
      - field_id:       字段唯一标识 (UUID4)
      - standard_name:  标准英文字段名 (如 WBC, CEA)
      - chinese_name:   中文名称 (如 白细胞, 癌胚抗原)
      - english_name:   英文全称
      - abbreviation:   缩写
      - alias_names:    逗号分隔的别名字符串
      - data_type:      数据类型 (numeric / category)
      - unit:           标准单位
      - reference_range: 参考范围 (如 "3.5-9.5")
      - category:       所属临床分类
      - description:    字段说明
      - created_time:   记录创建时间
    """

    __tablename__ = "data_dictionary"

    field_id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="字段唯一标识 (UUID4)",
    )
    standard_name = Column(
        String(100), nullable=False, index=True, comment="标准英文字段名, 如 WBC, CEA"
    )
    chinese_name = Column(
        String(200), nullable=False, comment="中文名称, 如 白细胞, 癌胚抗原"
    )
    english_name = Column(
        String(300), nullable=True, comment="英文全称"
    )
    abbreviation = Column(
        String(50), nullable=True, comment="缩写"
    )
    alias_names = Column(
        String(500), nullable=True, comment="别名, 逗号分隔, 如 '白细胞,WBC,white blood cell'"
    )
    data_type = Column(
        String(50), nullable=True, default="numeric", comment="字段数据类型: numeric/category"
    )
    unit = Column(
        String(50), nullable=True, comment="标准单位"
    )
    reference_range = Column(
        String(100), nullable=True, comment="参考范围, 如 '3.5-9.5'"
    )
    category = Column(
        String(100), nullable=True, comment="所属临床分类, 如 血常规/肿瘤标志物"
    )
    description = Column(
        String(500), nullable=True, comment="字段说明"
    )
    created_time = Column(
        DateTime, nullable=False, default=datetime.now, comment="记录创建时间"
    )


# ---------------------------------------------------------------------------
# 3. 分析任务表
# ---------------------------------------------------------------------------

class AnalysisTask(Base):
    """分析任务表 —— 记录每次建模/分析任务的完整配置与状态。

    字段说明:
      - task_id:           任务唯一标识 (UUID4)
      - dataset_id:        关联的数据集 ID
      - task_type:         任务类型 (classification/regression/clustering)
      - target_variable:   目标变量名
      - feature_variables: 特征变量列表 (JSON 字符串)
      - algorithm_name:    算法名称 (如 XGBoost / MLP)
      - parameter_json:    训练超参数 (JSON 字符串)
      - status:            任务状态 (Running / Completed / Failed)
      - result_path:       结果报告保存路径
      - created_time:      记录创建时间
    """

    __tablename__ = "analysis_tasks"

    task_id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="任务唯一标识 (UUID4)",
    )
    dataset_id = Column(
        String(36), nullable=False, comment="关联数据集 ID"
    )
    task_type = Column(
        String(50), nullable=False, comment="任务类型: classification/regression/clustering"
    )
    target_variable = Column(
        String(200), nullable=False, comment="目标变量名"
    )
    feature_variables = Column(
        String(2000), nullable=True, comment="特征变量列表 (JSON 字符串)"
    )
    algorithm_name = Column(
        String(100), nullable=False, comment="算法名称, 如 XGBoost/MLP"
    )
    parameter_json = Column(
        String(2000), nullable=True, comment="训练超参数 (JSON 字符串)"
    )
    status = Column(
        String(50), nullable=False, default="Pending", comment="任务状态: Pending/Running/Completed/Failed"
    )
    result_path = Column(
        String(500), nullable=True, comment="结果报告保存路径"
    )
    created_time = Column(
        DateTime, nullable=False, default=datetime.now, comment="记录创建时间"
    )


# ---------------------------------------------------------------------------
# 4. 模型结果表
# ---------------------------------------------------------------------------

class ModelResult(Base):
    """模型结果表 —— 存储训练完毕的模型评估指标与文件路径。

    字段说明:
      - model_id:               模型唯一标识 (UUID4)
      - task_id:                关联的分析任务 ID
      - model_name:             模型名称
      - task_type:              任务类型
      - train_score:            训练集得分
      - val_score:              验证集得分
      - test_score:             测试集得分 (F1 / AUC 等)
      - metric_json:            完整评估指标 (JSON 字符串)
      - feature_importance_path: SHAP 特征重要性图保存路径
      - model_path:             .pkl 模型文件物理路径
      - created_time:           记录创建时间
    """

    __tablename__ = "model_results"

    model_id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="模型唯一标识 (UUID4)",
    )
    task_id = Column(
        String(36), nullable=False, comment="关联分析任务 ID"
    )
    model_name = Column(
        String(200), nullable=False, comment="模型名称"
    )
    task_type = Column(
        String(50), nullable=False, comment="任务类型"
    )
    train_score = Column(
        Float, nullable=True, default=0.0, comment="训练集得分"
    )
    val_score = Column(
        Float, nullable=True, default=0.0, comment="验证集得分"
    )
    test_score = Column(
        Float, nullable=True, default=0.0, comment="测试集得分 (F1/AUC)"
    )
    metric_json = Column(
        String(2000), nullable=True, comment="完整评估指标 (JSON 字符串)"
    )
    feature_importance_path = Column(
        String(500), nullable=True, comment="SHAP 重要性图保存路径"
    )
    model_path = Column(
        String(500), nullable=True, comment=".pkl 模型文件物理路径"
    )
    created_time = Column(
        DateTime, nullable=False, default=datetime.now, comment="记录创建时间"
    )
