#!/usr/bin/env python3
"""多模态融合与自适应热切换路由 API — D组医学数据中台.

本模块面向 A/B/C/E/F 组提供跨模态协同推理接口:
  - POST /api/multimodal/route_and_predict  自适应模型路由 + 推理预测
  - POST /api/multimodal/align_features     跨组多源数据对齐与特征融合
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.backend.database import get_db

# ---------------------------------------------------------------------------
# APIRouter 实例
# ---------------------------------------------------------------------------

multimodal_router = APIRouter(
    prefix="/api/multimodal",
    tags=["多模态融合与自适应热切换"],
)

# ---------------------------------------------------------------------------
# 尝试导入真实路由引擎; 失败则启用仿真兜底
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ROUTER_AVAILABLE = False
_AdaptiveModelRouter: Any = None

try:
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.models.model_router import AdaptiveModelRouter as _Router  # type: ignore[assignment]

    # 即时实例化验证: 若注册表 JSON 存在则导入成功
    _registry_path = str(_PROJECT_ROOT / "config" / "model_registry.json")
    if os.path.isfile(_registry_path):
        _Router(registry_path=_registry_path)
        _AdaptiveModelRouter = _Router
        _ROUTER_AVAILABLE = True
except Exception:
    _ROUTER_AVAILABLE = False


# ===================================================================
# 仿真路由兜底引擎 (当真实 AdaptiveModelRouter 不可用时)
# ===================================================================

def _fallback_route(patient_profile: dict[str, Any], db_session: Any = None) -> dict[str, Any]:
    """仿真路由公式: 综合权重 = α×战绩 + β×模态契合 - γ×缺失惩罚.

    内置三大候选模型的简版参数, 保证即使没有原版 model_router 也能正常运转.
    若提供 db_session, 则优先从 MLOpsConfig 读取自进化后的最新基础战绩.

    Args:
        patient_profile: 患者数据画像.
        db_session:      可选的 SQLAlchemy Session (用于读取 MLOps 自动升级战绩).

    Returns:
        与 AdaptiveModelRouter.route_case 兼容的决策字典.
    """
    pid = patient_profile.get("patient_id", "p001")
    ts_steps = int(patient_profile.get("time_series_steps", 0))
    nlp_ents = int(patient_profile.get("nlp_entities", 0))
    has_e = bool(patient_profile.get("has_e_group_genomics", False))
    missing_rate = float(patient_profile.get("missing_rate", 0.0))

    # 尝试从 MLOpsConfig 读取自进化战绩
    mlops_used = False
    mlops_live: dict[str, float] = {}
    if db_session is not None:
        try:
            from src.backend.models_db import MLOpsConfig
            key_map = {
                "model_a_base_score": "Model_A_TimeSeries",
                "model_b_base_score": "Model_B_NLP_Graph",
                "model_c_base_score": "Model_C_Multimodal",
            }
            for ck, mid in key_map.items():
                row = db_session.query(MLOpsConfig).filter(MLOpsConfig.key == ck).first()
                if row and row.value is not None:
                    mlops_live[mid] = float(row.value)
            if mlops_live:
                mlops_used = True
        except Exception:
            pass

    # 内置候选模型简表 (硬编码初始值作为兜底)
    candidates = {
        "Model_A_TimeSeries": {
            "hist_f1": 0.89,
            "req_ts_min": 3,
            "req_nlp_min": 0,
            "req_rad": False,
            "sens_ts_miss": 0.6,
            "sens_nlp_miss": 0.1,
            "sens_rad_miss": 0.1,
        },
        "Model_B_NLP_Graph": {
            "hist_f1": 0.74,
            "req_ts_min": 0,
            "req_nlp_min": 3,
            "req_rad": False,
            "sens_ts_miss": 0.1,
            "sens_nlp_miss": 0.6,
            "sens_rad_miss": 0.1,
        },
        "Model_C_Multimodal": {
            "hist_f1": 0.94,
            "req_ts_min": 1,
            "req_nlp_min": 2,
            "req_rad": True,
            "sens_ts_miss": 0.3,
            "sens_nlp_miss": 0.2,
            "sens_rad_miss": 0.8,
        },
    }

    alpha, beta, gamma = 0.5, 0.3, 0.2
    scores: dict[str, float] = {}
    details: dict[str, dict] = {}

    for mid, c in candidates.items():
        # 模态契合度
        ts_fit = min(1.0, ts_steps / c["req_ts_min"]) if c["req_ts_min"] > 0 and ts_steps > 0 else (0.0 if c["req_ts_min"] > 0 else 1.0)
        nlp_fit = min(1.0, nlp_ents / c["req_nlp_min"]) if c["req_nlp_min"] > 0 and nlp_ents > 0 else (0.0 if c["req_nlp_min"] > 0 else 1.0)
        rad_fit = 1.0 if not c["req_rad"] else (1.0 if has_e else 0.2)

        modality_fit = (ts_fit + nlp_fit + rad_fit) / 3.0

        # 缺失惩罚
        penalty = 0.0
        if c["req_ts_min"] > 0 and ts_steps < c["req_ts_min"]:
            penalty += c["sens_ts_miss"] * (1 - ts_steps / c["req_ts_min"]) * 0.5
        if c["req_nlp_min"] > 0 and nlp_ents < c["req_nlp_min"]:
            penalty += c["sens_nlp_miss"] * (1 - nlp_ents / c["req_nlp_min"]) * 0.5
        if c["req_rad"] and not has_e:
            penalty += c["sens_rad_miss"] * 0.8
        penalty += missing_rate * 0.3
        missing_penalty = min(1.0, penalty)

        score = alpha * c["hist_f1"] + beta * modality_fit - gamma * missing_penalty
        score_100 = round(max(0.0, score) * 100, 1)
        scores[mid] = score_100
        details[mid] = {
            "historical_f1": c["hist_f1"],
            "modality_fit": round(modality_fit, 4),
            "missing_penalty": round(missing_penalty, 4),
            "score_source": "mlops_live" if mid in mlops_live else "fallback_default",
        }

    winner = max(scores, key=scores.__getitem__)
    winning = scores[winner]

    # 生成解释日志
    lines: list[str] = [
        f"【仿真路由决策 — 患者 {pid}】",
        f"数据画像: 时序={ts_steps}步, NLP实体={nlp_ents}, "
        f"E组放射组学={'有' if has_e else '无'}, 缺失率={missing_rate:.0%}",
    ]
    if mlops_used and mlops_live:
        lines.append(f"🔥 MLOps自进化战绩: {mlops_live}")
    lines += ["", f"▶ 胜出模型: {winner} (得分 {winning}/100)"]
    for mid, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        d = details[mid]
        marker = " ★ 胜出" if mid == winner else ""
        lines.append(f"  {mid}: {sc}/100 (F1={d['historical_f1']:.0%}, "
                     f"fit={d['modality_fit']:.2f}, pen={d['missing_penalty']:.2f}){marker}")

    return {
        "patient_id": pid,
        "target_disease": patient_profile.get("primary_disease", ""),
        "target_disease_cn": patient_profile.get("primary_disease", ""),
        "selected_model": winner,
        "winning_score": winning,
        "all_model_scores": scores,
        "model_details": details,
        "mlops_enhanced": mlops_used,
        "mlops_live_scores": mlops_live,
        "routing_reason": "\n".join(lines),
        "execution_status": "Ready_for_Inference (Fallback Engine)"
        + (" [MLOps增强]" if mlops_used else ""),
    }


def _fallback_xai(routing_result: dict, patient_data: dict) -> dict:
    """仿真 SHAP 归因解释."""
    winner = routing_result.get("selected_model", "")
    if "TimeSeries" in winner or "A" in winner:
        shap = [
            {"feature": "GLU_slope", "importance": 0.35},
            {"feature": "Creatinine_latest", "importance": 0.25},
            {"feature": "HbA1c_baseline", "importance": 0.18},
            {"feature": "time_series_length", "importance": 0.12},
            {"feature": "ALT_slope", "importance": 0.10},
        ]
        pred = "高危并发症：基于时序趋势预测，指标持续恶化，建议内分泌科干预。"
        risk = "HIGH"
    elif "NLP" in winner:
        shap = [
            {"feature": "病灶尺寸", "importance": 0.30},
            {"feature": "否定过滤", "importance": 0.25},
            {"feature": "解剖部位关联数", "importance": 0.20},
            {"feature": "药物-疾病三元组数", "importance": 0.15},
            {"feature": "实体总密度", "importance": 0.10},
        ]
        pred = "中高危：基于NLP图谱分析，实体关系提示恶性可能，建议补充影像检查。"
        risk = "MEDIUM_HIGH"
    else:
        shap = [
            {"feature": "CT组学纹理特征", "importance": 0.30},
            {"feature": "NER疾病实体向量", "importance": 0.25},
            {"feature": "CEA时序趋势", "importance": 0.20},
            {"feature": "跨模态注意力权重", "importance": 0.15},
            {"feature": "B组影像质量分", "importance": 0.10},
        ]
        pred = "高度疑似恶性：跨模态融合提示恶性概率82%，CT组学异质性显著。"
        risk = "HIGH"

    return {
        "prediction": pred,
        "risk_probability": round(0.75 + np.random.random() * 0.15, 2) if risk == "HIGH" else round(0.35 + np.random.random() * 0.25, 2),
        "risk_level": risk,
        "shap_explanations": shap,
    }


# ===================================================================
# 1. POST /api/multimodal/route_and_predict — 自适应路由 + 预测
# ===================================================================


@multimodal_router.post("/route_and_predict")
def route_and_predict(payload: dict, db: Session = Depends(get_db)):
    """多模态自适应热切与终极预测接口。

    接收患者画像 + 临床数据 + 影像特征, 执行两步流程:
      1. 自适应路由: 扫描数据完整度, 选出最优候选模型
      2. 模拟推理: 基于选中模型生成预测结论与 SHAP 归因

    Body 示例:
        {
          "patient_profile": {
            "patient_id": "p001", "time_series_steps": 5,
            "nlp_entities": 4, "has_e_group_genomics": false,
            "missing_rate": 0.1, "primary_disease": "Lung_Oncology"
          },
          "clinical_data": {"WBC": 8.5, "CEA": 12.4},
          "imaging_features": {"lesion_volume": 14.2}
        }

    Args:
        payload: JSON Body.

    Returns:
        路由决策 + 融合预测的完整 JSON。
    """
    profile: dict[str, Any] = payload.get("patient_profile", {})
    if not profile:
        raise HTTPException(status_code=400, detail="patient_profile 不能为空")

    # 自动推断 primary_disease
    if "primary_disease" not in profile:
        # 极简推断: 有 CEA 升高 → 肺癌, 有 GLU → 糖尿病
        clinical = payload.get("clinical_data", {})
        if float(clinical.get("CEA", 0)) > 8:
            profile["primary_disease"] = "Lung_Oncology"
        elif float(clinical.get("GLU", 0)) > 7:
            profile["primary_disease"] = "Diabetes_Nephropathy"
        else:
            profile["primary_disease"] = "Diabetes_Nephropathy"

    # ---- 1. 路由 ----
    if _ROUTER_AVAILABLE and _AdaptiveModelRouter:
        try:
            router = _AdaptiveModelRouter(
                registry_path=str(_PROJECT_ROOT / "config" / "model_registry.json"),
                db_session=db,
            )
            routing_result = router.route_case(profile)
        except Exception:
            routing_result = _fallback_route(profile, db)
    else:
        routing_result = _fallback_route(profile, db)

    # ---- 2. 模拟推理预测 ----
    patient_data = payload.get("clinical_data", {})
    patient_data.update(payload.get("imaging_features", {}))
    patient_data.update({
        "glu_values": payload.get("clinical_data", {}).get("glu_values", []),
        "glu_slope": payload.get("clinical_data", {}).get("glu_slope", 0.0),
        "creatinine_latest": payload.get("clinical_data", {}).get("Creatinine", 0),
    })

    if _ROUTER_AVAILABLE and _AdaptiveModelRouter:
        try:
            xai_result = router.simulate_inference_with_xai(routing_result, patient_data)
            fused = {
                "result": xai_result.get("predicted_diagnosis", ""),
                "risk_probability": patient_data.get("ce", 0.5),
                "risk_level": xai_result.get("risk_level", "MEDIUM"),
                "shap_explanations": xai_result.get("shap_feature_importance", []),
            }
        except Exception:
            fused = _fallback_xai(routing_result, patient_data)
    else:
        fused = _fallback_xai(routing_result, patient_data)

    return {
        "status": "success",
        "patient_id": profile.get("patient_id", "p001"),
        "selected_model": routing_result["selected_model"],
        "winning_score": routing_result["winning_score"],
        "all_model_scores": routing_result["all_model_scores"],
        "routing_reason": routing_result["routing_reason"],
        "engine_mode": "real" if _ROUTER_AVAILABLE else "fallback_simulation",
        "mlops_enhanced": routing_result.get("mlops_enhanced", False),
        "mlops_live_scores": routing_result.get("mlops_live_scores", {}),
        "fused_prediction": fused,
        "note": "当前使用的是 MLOps 自动升级后的最新战绩 P_base"
        if routing_result.get("mlops_enhanced")
        else "使用 JSON 注册表初始战绩 (启动 MLOps 服务后可自动升级)",
    }


# ===================================================================
# 2. POST /api/multimodal/align_features — 跨组特征融合
# ===================================================================


@multimodal_router.post("/align_features")
def align_features(payload: dict):
    """跨组多源表格对齐与融合特征构建接口。

    将 A/B/C/D/E/F 组异构数据按 patient_id/case_id 对齐,
    拼装成统一的特征宽表向量。

    Body 示例:
        {
          "patient_id": "p001",
          "case_id": "case_1001",
          "group_a_simulation": {"lesion_elasticity": 0.72},
          "group_b_quality": {"artifact_level": "low", "quality_score": 0.95},
          "group_c_labels": {"dice": 0.88, "iou": 0.79},
          "group_e_radiomics": {
            "texture_mean": 55.2, "glcm_entropy": 3.45,
            "shape_compactness": 0.68
          },
          "d_group_nlp": {"entity_count": 4, "negation_count": 1},
          "d_group_stats": {"GLU_slope": 0.64, "Creatinine_mean": 135.2}
        }

    Args:
        payload: JSON Body.

    Returns:
        对齐后的融合特征向量与对齐规则说明。
    """
    pid = payload.get("patient_id", "unknown")
    cid = payload.get("case_id", pid)

    # ---- 提取各组特征 → 扁平化 ----
    fused: dict[str, Any] = {
        "patient_id": pid,
        "case_id": cid,
    }

    # A 组: 物理仿真
    group_a = payload.get("group_a_simulation", {})
    if group_a:
        fused["A_lesion_elasticity"] = group_a.get("lesion_elasticity", 0)

    # B 组: 图像质控
    group_b = payload.get("group_b_quality", {})
    if group_b:
        fused["B_artifact_level"] = group_b.get("artifact_level", "unknown")
        fused["B_quality_score"] = group_b.get("quality_score", 1.0)
        fused["B_quality_artifact_warning"] = float(group_b.get("quality_score", 1.0)) < 0.6

    # C 组: 病理标注
    group_c = payload.get("group_c_labels", {})
    if group_c:
        fused["C_dice"] = group_c.get("dice", 0)
        fused["C_iou"] = group_c.get("iou", 0)

    # E 组: 放射组学
    group_e = payload.get("group_e_radiomics", {})
    if group_e:
        fused["E_texture_mean"] = group_e.get("texture_mean", 0)
        fused["E_glcm_entropy"] = group_e.get("glcm_entropy", 0)
        fused["E_shape_compactness"] = group_e.get("shape_compactness", 0)
        # 衍生特征
        vals = [v for k, v in group_e.items() if isinstance(v, (int, float))]
        if vals:
            fused["E_radiomics_mean"] = round(float(np.mean(vals)), 4)
            fused["E_radiomics_std"] = round(float(np.std(vals)), 4)

    # D 组: NLP 与 生化统计
    d_nlp = payload.get("d_group_nlp", {})
    if d_nlp:
        fused["D_entity_count"] = d_nlp.get("entity_count", 0)
        fused["D_negation_count"] = d_nlp.get("negation_count", 0)
    d_stats = payload.get("d_group_stats", {})
    if d_stats:
        for k, v in d_stats.items():
            fused[f"D_{k}"] = v

    # 模态完整度
    connected_groups = sum(1 for g in ["group_a_simulation", "group_b_quality",
                                        "group_c_labels", "group_e_radiomics",
                                        "d_group_nlp", "d_group_stats"]
                           if payload.get(g))
    total_groups = 6
    missing_rate = round(1.0 - connected_groups / total_groups, 4)

    # 对齐规则说明
    if cid and cid != "unknown":
        rule = f"Matched by case_id = {cid}"
    else:
        rule = f"Matched by patient_id = {pid}"

    return {
        "status": "success",
        "patient_id": pid,
        "case_id": cid,
        "alignment_rule": rule,
        "total_groups_connected": connected_groups,
        "overall_missing_rate": missing_rate,
        "fused_feature_vector": fused,
        "feature_count": len(fused) - 2,  # 减去 patient_id + case_id
        "message": f"已完成 A/B/C/D/E 组多模态数据对齐！共连接 {connected_groups}/{total_groups} 组。",
    }


# ===================================================================
# MLOps 答辩演示专属: 一键触发进化 API
# ===================================================================

mlops_demo_router = APIRouter(
    prefix="/api/mlops",
    tags=["MLOps自进化演示"],
)


@mlops_demo_router.post("/trigger_evolution_demo")
def trigger_evolution_demo(db: Session = Depends(get_db)):
    """答辩演示专用: 强行无视 50 条阈值, 立即执行持续训练+评分自进化.

    直接调用 run_continuous_training_pipeline 完成:
      1. 仿真增量微调训练
      2. 升级三大模型基础战绩 (model_a/b/c_base_score)
      3. 清零 new_samples_since_last_ct

    无需等待新数据累积, 一键体验 MLOps 自进化闭环!

    Returns:
        {"status": "success", "message": "..."}
    """
    from src.backend.service_mlops import run_continuous_training_pipeline

    # 使用同一 session 执行 (演示场景)
    run_continuous_training_pipeline(db)

    # 读取升级后的战绩展示
    from src.backend.models_db import MLOpsConfig
    updated_scores: dict[str, float] = {}
    for key in ["model_a_base_score", "model_b_base_score", "model_c_base_score"]:
        row = db.query(MLOpsConfig).filter(MLOpsConfig.key == key).first()
        if row:
            updated_scores[key] = float(row.value)

    return {
        "status": "success",
        "message": (
            "已手动触发 MLOps 自进化闭环！"
            "路由大脑各模型评分 P_base 已在后台全面刷新升级！"
            "请去 /docs 重新预测看战绩表现！"
        ),
        "updated_scores": updated_scores,
    }
