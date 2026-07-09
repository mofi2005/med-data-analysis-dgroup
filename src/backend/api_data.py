#!/usr/bin/env python3
"""数据接入与数据字典 API 路由 — D组医学数据中台.

本模块基于 FastAPI APIRouter, 提供以下核心接口:
  - POST /api/data/upload             文件上传与数据集注册
  - POST /api/data/dictionary/init_default  初始化标准字段字典
  - GET  /api/data/dictionary/match   根据原始字段名匹配标准名
  - POST /api/data/dictionary/map_fields    批量注册字段映射
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import exists
from sqlalchemy.orm import Session

# 复用现有基础设施
from src.backend.database import get_db
from src.backend.models_db import DataDictionary, Dataset

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 项目根目录 & 上传目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_UPLOAD_DIR = _PROJECT_ROOT / "data" / "uploads"

# 允许的文件扩展名
_ALLOWED_EXTENSIONS = {"csv", "xlsx", "json"}

# 默认标准字段定义 (来源于 config/standard_dictionary.json)
_DEFAULT_FIELDS_: list[dict] = [
    # 血常规
    {"standard_name": "WBC", "chinese_name": "白细胞", "english_name": "white blood cell",
     "abbreviation": "WBC", "alias_names": "白细胞,WBC,wbc,white blood cell,White Blood Cell",
     "data_type": "numeric", "unit": "10^9/L", "reference_range": "3.5-9.5",
     "category": "血常规", "description": "白细胞计数，反映机体炎症与免疫状态"},
    {"standard_name": "HGB", "chinese_name": "血红蛋白", "english_name": "hemoglobin",
     "abbreviation": "HGB", "alias_names": "血红蛋白,HGB,hgb,hemoglobin,Hb",
     "data_type": "numeric", "unit": "g/L", "reference_range": "130-175",
     "category": "血常规", "description": "血红蛋白浓度，用于贫血诊断与分级"},
    {"standard_name": "PLT", "chinese_name": "血小板", "english_name": "platelet",
     "abbreviation": "PLT", "alias_names": "血小板,PLT,plt,platelet",
     "data_type": "numeric", "unit": "10^9/L", "reference_range": "125-350",
     "category": "血常规", "description": "血小板计数，评估凝血功能与骨髓造血"},
    # 生化 / 糖尿病
    {"standard_name": "GLU", "chinese_name": "血糖", "english_name": "glucose",
     "abbreviation": "GLU", "alias_names": "血糖,GLU,glu,glucose,GLUC",
     "data_type": "numeric", "unit": "mmol/L", "reference_range": "3.9-6.1",
     "category": "糖尿病相关", "description": "空腹血糖浓度，糖尿病筛查核心指标"},
    {"standard_name": "HbA1c", "chinese_name": "糖化血红蛋白", "english_name": "glycated hemoglobin",
     "abbreviation": "HbA1c", "alias_names": "糖化血红蛋白,HbA1c,hba1c,HbA1C",
     "data_type": "numeric", "unit": "%", "reference_range": "4.0-6.0",
     "category": "糖尿病相关", "description": "糖化血红蛋白，反映近8-12周平均血糖水平"},
    # 肾功能
    {"standard_name": "Creatinine", "chinese_name": "肌酐", "english_name": "creatinine",
     "abbreviation": "CREA", "alias_names": "肌酐,Creatinine,creatinine,crea,cr,Cr,CREA",
     "data_type": "numeric", "unit": "μmol/L", "reference_range": "44-133",
     "category": "肾功能", "description": "血清肌酐，评估肾小球滤过功能"},
    {"standard_name": "Urea", "chinese_name": "尿素", "english_name": "urea",
     "abbreviation": "Urea", "alias_names": "尿素,Urea,urea,BUN",
     "data_type": "numeric", "unit": "mmol/L", "reference_range": "2.9-8.2",
     "category": "肾功能", "description": "血清尿素氮，反映肾小管重吸收与蛋白质代谢"},
    {"standard_name": "eGFR", "chinese_name": "肾小球滤过率", "english_name": "estimated glomerular filtration rate",
     "abbreviation": "eGFR", "alias_names": "eGFR,egfr,肾小球滤过率",
     "data_type": "numeric", "unit": "mL/min/1.73m²", "reference_range": ">90",
     "category": "肾功能", "description": "估算肾小球滤过率，CKD分期金标准"},
    # 肝功能
    {"standard_name": "ALT", "chinese_name": "丙氨酸氨基转移酶", "english_name": "alanine aminotransferase",
     "abbreviation": "ALT", "alias_names": "ALT,alt,GPT,丙氨酸氨基转移酶",
     "data_type": "numeric", "unit": "U/L", "reference_range": "0-40",
     "category": "肝功能", "description": "谷丙转氨酶，肝细胞损伤敏感指标"},
    {"standard_name": "AST", "chinese_name": "天门冬氨酸氨基转移酶", "english_name": "aspartate aminotransferase",
     "abbreviation": "AST", "alias_names": "AST,ast,GOT,天门冬氨酸氨基转移酶",
     "data_type": "numeric", "unit": "U/L", "reference_range": "0-40",
     "category": "肝功能", "description": "谷草转氨酶，心肌与肝细胞损伤标志"},
    # 肿瘤标志物
    {"standard_name": "CEA", "chinese_name": "癌胚抗原", "english_name": "carcinoembryonic antigen",
     "abbreviation": "CEA", "alias_names": "CEA,cea,癌胚抗原",
     "data_type": "numeric", "unit": "ng/mL", "reference_range": "0-5",
     "category": "肿瘤标志物", "description": "癌胚抗原，广谱肿瘤标志物"},
    # 炎症
    {"standard_name": "CRP", "chinese_name": "C反应蛋白", "english_name": "C-reactive protein",
     "abbreviation": "CRP", "alias_names": "CRP,crp,C反应蛋白",
     "data_type": "numeric", "unit": "mg/L", "reference_range": "0-10",
     "category": "炎症标志物", "description": "C反应蛋白，反映急性炎症与感染严重程度"},
]

# ---------------------------------------------------------------------------
# APIRouter 实例
# ---------------------------------------------------------------------------

data_router = APIRouter(
    prefix="/api/data",
    tags=["数据导入与标准化字典"],
)

# ===================================================================
# 1. 文件上传与数据集注册
# ===================================================================


@data_router.post("/upload")
async def upload_dataset_api(
    file: UploadFile = File(...),
    source_type: str = Form(default="D组"),
    db: Session = Depends(get_db),
):
    """上传医学数据文件 (.csv / .xlsx / .json) 并注册为数据集。

    工作流:
      1. 校验文件扩展名
      2. 使用 pandas 读取内容，提取行列数与列名
      3. 将文件保存至 data/uploads/
      4. 创建 Dataset ORM 记录并持久化到数据库

    Args:
        file:        上传的文件对象。
        source_type: 来源组别 (默认 "D组")。
        db:          数据库会话 (依赖注入)。

    Returns:
        {"dataset_id": "...", "row_count": N, "column_count": N,
         "columns": [...], "message": "..."}
    """
    # 0. 校验扩展名
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        detail = f"不支持的文件格式: .{ext}，仅支持 {_ALLOWED_EXTENSIONS}"
        raise HTTPException(status_code=400, detail=detail)

    # 1. 使用 pandas 读取
    try:
        content = await file.read()
        if ext == "csv":
            df = pd.read_csv(pd.io.common.BytesIO(content))  # type: ignore[arg-type]
        elif ext == "xlsx":
            df = pd.read_excel(pd.io.common.BytesIO(content))  # type: ignore[arg-type]
        else:  # json
            df = pd.read_json(pd.io.common.BytesIO(content))  # type: ignore[arg-type]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件解析失败: {str(e)}") from e

    row_count = len(df)
    column_count = len(df.columns)
    columns_list = list(df.columns)

    # 2. 确保上传目录存在
    os.makedirs(str(_UPLOAD_DIR), exist_ok=True)

    # 3. 保存文件
    dataset_id = str(uuid.uuid4())
    saved_name = f"{dataset_id}.{ext}"
    save_path = _UPLOAD_DIR / saved_name
    with open(save_path, "wb") as f:
        f.write(content)

    # 4. 构建 ORM → 入库
    dataset = Dataset(
        dataset_id=dataset_id,
        dataset_name=filename,
        data_type=ext.upper(),
        source_type=source_type,
        file_path=str(save_path),
        row_count=row_count,
        column_count=column_count,
        upload_user="member1",
        created_time=datetime.now(),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return {
        "dataset_id": dataset_id,
        "dataset_name": filename,
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns_list,
        "message": f"文件 {filename} 上传成功，已注册为数据集 {dataset_id}",
    }


# ===================================================================
# 2. 初始化标准字段字典 (幂等: 仅当表为空时插入)
# ===================================================================


@data_router.post("/dictionary/init_default")
def init_default_dictionary(db: Session = Depends(get_db)):
    """初始化 DataDictionary 表的默认医学标准字段 (幂等操作)。

    逻辑:
      - 检查 DataDictionary 表是否已有记录
      - 若为空，从内置预定义列表批量插入 (WBC, HGB, PLT, GLU, HbA1c,
        Creatinine, Urea, eGFR, ALT, AST, CEA, CRP)
      - 若已有数据，直接返回 skip 不给重复插入

    Returns:
        {"status": "created"|"skipped", "count": N, "message": "..."}
    """
    # 幂等检查: 已有数据则跳过
    existing_count = db.query(DataDictionary).count()

    if existing_count > 0:
        return {
            "status": "skipped",
            "count": existing_count,
            "message": f"DataDictionary 表已有 {existing_count} 条记录，跳过初始化。",
        }

    # 批量插入默认字段
    inserted = 0
    for item in _DEFAULT_FIELDS_:
        record = DataDictionary(
            field_id=str(uuid.uuid4()),
            standard_name=item["standard_name"],
            chinese_name=item["chinese_name"],
            english_name=item.get("english_name", ""),
            abbreviation=item.get("abbreviation", ""),
            alias_names=item.get("alias_names", ""),
            data_type=item.get("data_type", "numeric"),
            unit=item.get("unit", ""),
            reference_range=item.get("reference_range", ""),
            category=item.get("category", ""),
            description=item.get("description", ""),
            created_time=datetime.now(),
        )
        db.add(record)
        inserted += 1

    db.commit()

    return {
        "status": "created",
        "count": inserted,
        "message": f"成功初始化 {inserted} 条标准医学字段到 DataDictionary。",
    }


# ===================================================================
# 3. 字段名智能匹配
# ===================================================================


@data_router.get("/dictionary/match")
def match_field(
    raw_field_name: str = Query(..., description="原始字段名 (如 WBC, 白细胞, 肌肝)"),
    db: Session = Depends(get_db),
):
    """根据用户输入的原始字段名，在 DataDictionary 中模糊匹配标准名。

    匹配策略:
      1. 对输入去除首尾空格，统一转为小写 (英文不区分大小写)
      2. 遍历所有记录，检查 cleaned_input 是否出现在 alias_names 中
      3. 命中则返回 standard_name + 完整字典信息；未命中返回 not_found

    Args:
        raw_field_name: 待匹配的原始字段名。
        db:             数据库会话 (依赖注入)。

    Returns:
        {"matched": true|false, "standard_name": "..."|None, ...}
    """
    cleaned = raw_field_name.strip().lower()

    if not cleaned:
        raise HTTPException(status_code=400, detail="raw_field_name 不能为空")

    # 遍历字典表，在 alias_names 中查找
    all_records = db.query(DataDictionary).all()

    for record in all_records:
        aliases = record.alias_names or ""
        # 别名列表统一转小写后检查
        alias_parts = [a.strip().lower() for a in aliases.split(",") if a.strip()]
        if cleaned in alias_parts:
            return {
                "matched": True,
                "standard_name": record.standard_name,
                "field_info": {
                    "field_id": record.field_id,
                    "standard_name": record.standard_name,
                    "chinese_name": record.chinese_name,
                    "english_name": record.english_name,
                    "abbreviation": record.abbreviation,
                    "alias_names": record.alias_names,
                    "data_type": record.data_type,
                    "unit": record.unit,
                    "reference_range": record.reference_range,
                    "category": record.category,
                    "description": record.description,
                },
            }

    return {
        "matched": False,
        "standard_name": None,
        "message": f"未找到匹配 '{raw_field_name}' 的标准字段，请手动注册。",
    }


# ===================================================================
# 4. 批量字段映射注册
# ===================================================================


@data_router.post("/dictionary/map_fields")
def map_fields(
    payload: dict,
    db: Session = Depends(get_db),
):
    """接收一组字段映射关系，将原始名追加到对应标准字段的 alias_names。

    Body 格式:
        {
            "dataset_id": "uuid",
            "mappings": [
                {"raw_name": "白细胞计数", "standard_name": "WBC"},
                {"raw_name": "空腹血溏", "standard_name": "GLU"}
            ]
        }

    逻辑:
      1. 根据 standard_name 查找 DataDictionary 记录
      2. 若 raw_name 尚未在 alias_names 中出现，则去重追加
      3. 若 standard_name 不存在于字典表，记录到 skipped 列表

    Args:
        payload: JSON Body。
        db:      数据库会话。

    Returns:
        {"matched_count": N, "skipped": [...], "message": "..."}
    """
    dataset_id = payload.get("dataset_id", "")
    mappings: list[dict] = payload.get("mappings", [])

    if not mappings:
        raise HTTPException(status_code=400, detail="mappings 数组不能为空")

    matched_count = 0
    skipped: list[dict] = []

    for item in mappings:
        raw_name = item.get("raw_name", "").strip()
        standard_name = item.get("standard_name", "").strip()

        if not raw_name or not standard_name:
            skipped.append({**item, "reason": "raw_name 或 standard_name 为空"})
            continue

        # 查找字典记录
        record = (
            db.query(DataDictionary)
            .filter(DataDictionary.standard_name == standard_name)
            .first()
        )

        if record is None:
            skipped.append({**item, "reason": f"标准字段 '{standard_name}' 不存在于字典表"})
            continue

        # 去重追加别名
        existing_aliases = record.alias_names or ""
        alias_parts = [a.strip() for a in existing_aliases.split(",") if a.strip()]

        if raw_name.lower() not in [a.lower() for a in alias_parts]:
            alias_parts.append(raw_name)
            record.alias_names = ",".join(alias_parts)
            matched_count += 1
        else:
            skipped.append({**item, "reason": f"'{raw_name}' 已存在于 {standard_name} 的别名中"})

    db.commit()

    return {
        "dataset_id": dataset_id,
        "matched_count": matched_count,
        "skipped": skipped,
        "message": f"成功注册 {matched_count} 条映射，{len(skipped)} 条跳过/失败。",
    }
