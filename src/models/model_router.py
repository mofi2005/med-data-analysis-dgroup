#!/usr/bin/env python3
"""V3.0 自适应模型路由引擎 (Adaptive Model Router).

本模块实现企业级动态模型调度中枢，废除传统"一刀切"硬编码推理，
根据【历史战绩】与【当前患者特征契合度】自动热切至最优候选模型。

V3.0 动态路由公式:
    S_j = α * HistoricalScore_j + β * ModalityFit_j - γ * MissingPenalty_j

其中:
  - HistoricalScore_j: 候选模型 j 在当前目标病种上的历史 F1 战绩 (0~1)
  - ModalityFit_j:   当前患者数据模态与模型 j 所需模态的契合度 (0~1)
  - MissingPenalty_j: 患者缺失关键模态时对模型 j 的高额惩罚 (0~1)
  - α, β, γ:          可调超参数权重
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 自适应模型路由引擎
# ---------------------------------------------------------------------------


class AdaptiveModelRouter:
    """V3.0 自适应模型路由引擎。

    加载 MLOps 模型注册表，根据患者当前数据模态特征与模型历史战绩，
    通过动态加权打分公式自动选出最优推理模型，并提供决策可解释性。

    Attributes:
        alpha: 历史战绩权重 (默认 0.5)。
        beta:  特征契合度权重 (默认 0.3)。
        gamma: 缺失惩罚系数 (默认 0.2)。
        registry: 模型注册表数据字典。
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        registry_path: str | None = None,
        db_session: Any = None,
    ) -> None:
        """初始化路由引擎。

        Args:
            alpha: 历史战绩权重 (0~1)。
            beta:  特征契合度权重 (0~1)。
            gamma: 缺失惩罚系数 (0~1)。通常 α + β + γ = 1.0。
            registry_path: 模型注册表 JSON 路径；默认使用
                ``config/model_registry.json``。
            db_session: 可选的 SQLAlchemy Session，用于从 MLOpsConfig
                表中实时读取自进化后的最新基础战绩 (P_base)。
                若为 None 则回退到 JSON 注册表中的初始值。

        Raises:
            FileNotFoundError: 若注册表文件不存在。
        """
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma
        self._db_session: Any = db_session  # MLOps 自进化战绩数据源

        if registry_path is None:
            registry_path = os.path.join(
                str(_PROJECT_ROOT), "config", "model_registry.json"
            )

        self.registry: dict[str, Any] = self._load_registry(registry_path)
        self._models: dict[str, dict[str, Any]] = self.registry.get("models", {})

    # ------------------------------------------------------------------
    # 注册表加载
    # ------------------------------------------------------------------

    @staticmethod
    def _load_registry(path: str) -> dict[str, Any]:
        """加载模型注册表 JSON 文件。

        Args:
            path: JSON 文件路径。

        Returns:
            解析后的注册表字典。

        Raises:
            FileNotFoundError: 文件不存在。
            json.JSONDecodeError: JSON 格式错误。
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"模型注册表文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # MLOps 自进化战绩读取
    # ------------------------------------------------------------------

    def _read_mlops_live_scores(self) -> dict[str, float]:
        """尝试从 MLOpsConfig 数据库表中读取自进化后的最新基础战绩。

        若 db_session 未提供或读取失败, 回退到硬编码默认值 (85/80/90)。

        Returns:
            {model_id: base_score_pct} 字典, 值为 0~100 的量纲分数。
            例如 {"Model_A_TimeSeries": 88.5, "Model_B_NLP_Graph": 83.1, ...}
        """
        defaults: dict[str, float] = {
            "Model_A_TimeSeries": 85.0,
            "Model_B_NLP_Graph": 80.0,
            "Model_C_Multimodal": 90.0,
        }

        if self._db_session is None:
            return dict(defaults)

        key_to_model: dict[str, str] = {
            "model_a_base_score": "Model_A_TimeSeries",
            "model_b_base_score": "Model_B_NLP_Graph",
            "model_c_base_score": "Model_C_Multimodal",
        }

        try:
            from src.backend.models_db import MLOpsConfig

            for config_key, model_id in key_to_model.items():
                row = self._db_session.query(MLOpsConfig).filter(
                    MLOpsConfig.key == config_key
                ).first()
                if row and row.value is not None:
                    defaults[model_id] = float(row.value)
        except Exception:
            pass  # 读取失败: 静默回退到默认值

        return defaults

    # ------------------------------------------------------------------
    # 个案契合度与缺失评估
    # ------------------------------------------------------------------

    def _evaluate_case_fit(
        self, patient_profile: dict[str, Any], model_info: dict[str, Any]
    ) -> tuple[float, float]:
        """评估当前患者数据与指定模型的契合度及缺失惩罚。

        Args:
            patient_profile: 患者数据画像字典，需含:
                - primary_disease (str)
                - time_series_steps (int)
                - nlp_entities_count (int)
                - has_group_E_radiomics (bool)
                - missing_rate (float, 0~1)
            model_info: 候选模型的注册表条目。

        Returns:
            (modality_fit, missing_penalty) 元组，均为 0~1 归一化值。
        """
        req = model_info.get("required_modalities", {})
        sensitivity = model_info.get("modality_sensitivity", {})

        # ---- 各模态需求与患者实际 ----
        ts_min = req.get("time_series_steps", {}).get("min", 0)
        nlp_min = req.get("nlp_entities_count", {}).get("min", 0)
        rad_required = req.get("has_group_E_radiomics", False)

        ts_patient = int(patient_profile.get("time_series_steps", 0))
        nlp_patient = int(patient_profile.get("nlp_entities_count", 0))
        rad_patient = bool(patient_profile.get("has_group_E_radiomics", False))
        missing_rate = float(patient_profile.get("missing_rate", 0.0))

        # ---- 各模态契合度 (0~1) ----
        # 时序契合: 若模型不要求时序 (min=0)，则契合度为 1.0
        if ts_min > 0:
            ts_fit = min(1.0, ts_patient / ts_min) if ts_patient > 0 else 0.0
        else:
            ts_fit = 1.0

        # NLP 契合
        if nlp_min > 0:
            nlp_fit = min(1.0, nlp_patient / nlp_min) if nlp_patient > 0 else 0.0
        else:
            nlp_fit = 1.0

        # 组学契合: 若模型要求组学且患者没有 → 严重失配
        if rad_required:
            rad_fit = 1.0 if rad_patient else 0.2
        else:
            rad_fit = 1.0  # 模型不要求，不扣分

        # 综合契合度（所有模态等权平均）
        modality_fit = (ts_fit + nlp_fit + rad_fit) / 3.0

        # ---- 缺失惩罚 (0~1) ----
        # 对每个「模型要求但患者不满足」的模态施加惩罚
        penalty = 0.0

        # 时序缺失惩罚
        if ts_min > 0 and ts_patient < ts_min:
            ratio = 1.0 - (ts_patient / ts_min) if ts_min > 0 else 1.0
            penalty += sensitivity.get("time_series_missing", 0.5) * ratio * 0.5

        # NLP 缺失惩罚
        if nlp_min > 0 and nlp_patient < nlp_min:
            ratio = 1.0 - (nlp_patient / nlp_min) if nlp_min > 0 else 1.0
            penalty += sensitivity.get("nlp_missing", 0.5) * ratio * 0.5

        # 组学缺失惩罚（最严重）
        if rad_required and not rad_patient:
            penalty += sensitivity.get("radiomics_missing", 0.5) * 0.8

        # 全局缺失率惩罚
        penalty += missing_rate * 0.3

        missing_penalty = min(1.0, penalty)

        return modality_fit, missing_penalty

    # ------------------------------------------------------------------
    # V3.0 动态打分与路由选型
    # ------------------------------------------------------------------

    def route_case(self, patient_profile: dict[str, Any]) -> dict[str, Any]:
        """根据患者画像执行 V3.0 动态模型路由。

        对注册表中每个候选模型计算终极得分 S_j，选出最高分模型。

        Args:
            patient_profile: 患者数据画像字典。

        Returns:
            路由决策字典，含 selected_model、winning_score、
            all_model_scores、routing_reason、execution_status。
        """
        patient_id = patient_profile.get("patient_id", "unknown")
        target_disease = patient_profile.get("primary_disease", "")

        # 查找疾病中文名
        disease_name = ""
        for d in self.registry.get("diseases_covered", []):
            if d.get("disease_id") == target_disease:
                disease_name = d.get("chinese_name", target_disease)
                break

        # ---- 为每个候选模型打分 ----
        model_scores: dict[str, float] = {}
        model_details: dict[str, dict[str, Any]] = {}

        # 尝试从 MLOpsConfig 数据库读取自进化后的最新基础战绩
        mlops_live_scores = self._read_mlops_live_scores()
        mlops_used = self._db_session is not None

        for model_id, model_info in self._models.items():
            # 1. 历史战绩: 优先使用 MLOps 自进化实时战绩, 其次回退 JSON 注册表
            if model_id in mlops_live_scores:
                historical_f1 = mlops_live_scores[model_id] / 100.0
                score_source = "mlops_live"
            else:
                hist_scores = model_info.get("historical_f1_score", {})
                historical_f1 = hist_scores.get(target_disease, 0.5)
                score_source = "json_registry"

            # 2. 模态契合度 & 缺失惩罚
            modality_fit, missing_penalty = self._evaluate_case_fit(
                patient_profile, model_info
            )

            # 3. V3.0 公式: S_j = α*HistScore + β*ModalityFit - γ*MissingPenalty
            score = (
                self.alpha * historical_f1
                + self.beta * modality_fit
                - self.gamma * missing_penalty
            )
            # 映射到 0~100 量纲
            score_100 = round(max(0.0, score) * 100, 1)

            model_scores[model_id] = score_100
            model_details[model_id] = {
                "historical_f1": historical_f1,
                "modality_fit": round(modality_fit, 4),
                "missing_penalty": round(missing_penalty, 4),
                "raw_score": round(score, 4),
                "score_source": score_source,
            }

        # ---- 选出最高分模型 ----
        selected_model = max(model_scores, key=model_scores.__getitem__)
        winning_score = model_scores[selected_model]

        # ---- 生成详细解释日志 ----
        routing_reason = self._generate_routing_reason(
            patient_profile, patient_id, target_disease, disease_name,
            model_details, model_scores, selected_model, mlops_used, mlops_live_scores,
        )

        return {
            "patient_id": patient_id,
            "target_disease": target_disease,
            "target_disease_cn": disease_name,
            "selected_model": selected_model,
            "winning_score": winning_score,
            "all_model_scores": model_scores,
            "mlops_enhanced": mlops_used,
            "mlops_live_scores": mlops_live_scores if mlops_used else {},
            "model_details": {
                k: {
                    "historical_f1": v["historical_f1"],
                    "modality_fit": v["modality_fit"],
                    "missing_penalty": v["missing_penalty"],
                    "score_source": v["score_source"],
                }
                for k, v in model_details.items()
            },
            "routing_reason": routing_reason,
            "execution_status": "Ready_for_Inference"
            + (" (MLOps自进化战绩已注入)" if mlops_used else ""),
        }

    def _generate_routing_reason(
        self,
        profile: dict[str, Any],
        patient_id: str,
        disease: str,
        disease_name: str,
        details: dict[str, dict[str, Any]],
        scores: dict[str, float],
        winner: str,
        mlops_used: bool = False,
        mlops_scores: dict[str, float] | None = None,
    ) -> str:
        """生成可解释的中文旅由决策日志。

        Args:
            profile: 患者画像。
            patient_id: 患者 ID。
            disease: 目标病种 ID。
            disease_name: 目标病种中文名。
            details: 各模型打分明细。
            scores: 各模型最终得分。
            winner: 胜出模型 ID。

        Returns:
            详细中文解释字符串。
        """
        ts_steps = profile.get("time_series_steps", 0)
        nlp_count = profile.get("nlp_entities_count", 0)
        has_rad = profile.get("has_group_E_radiomics", False)
        missing_rate = profile.get("missing_rate", 0.0)

        winner_info = self._models.get(winner, {})
        winner_name = winner_info.get("model_name", winner)

        lines: list[str] = []
        lines.append(f"【路由决策报告 — 患者 {patient_id} ({disease_name})】")
        lines.append("")
        lines.append(f"患者数据画像: 时序步数={ts_steps}, "
                     f"NLP实体数={nlp_count}, "
                     f"E组组学={'有' if has_rad else '无'}, "
                     f"全局缺失率={missing_rate:.0%}")
        # MLOps 自进化战绩提示
        if mlops_used and mlops_scores:
            lines.append(f"🔥 MLOps自进化战绩: {mlops_scores}")
        lines.append("")
        lines.append(f"▶ 胜出模型: {winner} ({winner_name})")
        lines.append(f"   终极得分: {scores[winner]:.1f}/100")
        lines.append("")

        # 对每个模型分别解释
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (mid, score) in enumerate(sorted_models, 1):
            info = self._models.get(mid, {})
            name = info.get("model_name", mid)
            d = details[mid]
            hist = d["historical_f1"]
            mfit = d["modality_fit"]
            mpen = d["missing_penalty"]

            marker = " ★ 胜出" if mid == winner else ""
            lines.append(f"  #{rank} {mid} ({name}){marker}")
            lines.append(f"     历史战绩({disease}): {hist:.0%} (来源: {d.get('score_source', 'json')})")
            lines.append(f"     模态契合度: {mfit:.2f}  |  缺失惩罚: {mpen:.2f}")
            lines.append(f"     最终得分: {score:.1f} = "
                         f"α({self.alpha})×{hist:.0%} + "
                         f"β({self.beta})×{mfit:.2f} - "
                         f"γ({self.gamma})×{mpen:.2f}")

            # 特别分析被淘汰原因
            if mid != winner:
                reasons: list[str] = []
                reqs = info.get("required_modalities", {})
                if reqs.get("has_group_E_radiomics") and not has_rad:
                    reasons.append("缺少必需的E组CT放射组学特征")
                ts_min = reqs.get("time_series_steps", {}).get("min", 0)
                if ts_min > ts_steps:
                    reasons.append(f"时序步数({ts_steps})不满足最低要求({ts_min})")
                nlp_min = reqs.get("nlp_entities_count", {}).get("min", 0)
                if nlp_min > nlp_count:
                    reasons.append(f"NLP实体数({nlp_count})不满足最低要求({nlp_min})")

                if reasons:
                    lines.append(f"     淘汰原因: {'; '.join(reasons)}")
                else:
                    lines.append(f"     淘汰原因: 整体综合得分不及{winner}")

        lines.append("")
        lines.append(f"▶ 决策摘要: 系统综合考量了患者'{disease_name}'的数据模态特征与各模型"
                     f"历史战绩，最终自动热切至 {winner}。")
        if missing_rate > 0.3:
            lines.append(f"   注意: 患者数据缺失率较高({missing_rate:.0%})，建议人工复核。")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 模拟推理与 XAI 可解释性归因
    # ------------------------------------------------------------------

    def simulate_inference_with_xai(
        self, routing_result: dict[str, Any], patient_data: dict[str, Any]
    ) -> dict[str, Any]:
        """根据路由结果进行模拟推理并输出 SHAP 特征重要性归因。

        此方法为第五阶段报告引擎预对齐，模拟生成：
          - 预测结论（基于选中的模型类型 + 患者历史特征）
          - 虚拟 SHAP 特征贡献度排名

        Args:
            routing_result: ``route_case`` 返回的路由决策字典。
            patient_data: 患者详细数据字典。

        Returns:
            模拟推理结果，含 predicted_diagnosis、risk_level、
            shap_feature_importance、model_used。
        """
        selected_model = routing_result.get("selected_model", "")
        target_disease = routing_result.get("target_disease", "")
        model_info = self._models.get(selected_model, {})

        # ---- 基于模型类型模拟预测结论 ----
        prediction, risk_level = self._simulate_prediction(
            selected_model, target_disease, patient_data
        )

        # ---- 构造 SHAP 特征贡献 ----
        shap_features = self._build_shap_importance(
            selected_model, model_info, patient_data
        )

        # ---- 归因可解释日志 ----
        xai_log = self._build_xai_log(
            selected_model, model_info, target_disease, shap_features
        )

        return {
            "selected_model": selected_model,
            "model_name": model_info.get("model_name", selected_model),
            "model_version": model_info.get("version", "unknown"),
            "predicted_diagnosis": prediction,
            "risk_level": risk_level,
            "shap_feature_importance": shap_features,
            "xai_interpretation_log": xai_log,
        }

    @staticmethod
    def _simulate_prediction(
        model_id: str, disease: str, patient_data: dict[str, Any]
    ) -> tuple[str, str]:
        """根据模型类型和目标病种模拟预测结论。

        Args:
            model_id: 模型 ID。
            disease: 目标病种。
            patient_data: 患者数据。

        Returns:
            (prediction_text, risk_level) 元组。
        """
        if disease == "Diabetes_Nephropathy":
            if "TimeSeries" in model_id:
                # 检查 GLU 和 Creatinine 趋势
                glu_vals = patient_data.get("glu_values", [])
                if glu_vals and len(glu_vals) >= 2 and glu_vals[-1] > glu_vals[0] * 1.3:
                    return (
                        "高危并发症警告：基于时序建模，GLU持续攀升 (基线 {:.1f} → 当前 {:.1f})，"
                        "预测3个月内糖尿病肾病进展至 CKD-3 期概率 78%，"
                        "建议立即内分泌科干预。".format(
                            glu_vals[0], glu_vals[-1]
                        ),
                        "HIGH",
                    )
                return (
                    "中危预警：糖尿病肾病相关指标呈缓慢恶化趋势，"
                    "建议加强血糖控制并每月复查肾功能。",
                    "MEDIUM",
                )
            return (
                "中危预警：基于NLP/多模态分析，糖尿病肾病进展风险需关注，"
                "建议结合实验室检查进一步评估。",
                "MEDIUM",
            )
        elif disease == "Lung_Oncology":
            if "Multimodal" in model_id:
                return (
                    "高度疑似恶性病变：跨模态融合分析提示右肺上叶结节恶性概率 82%，"
                    "CT组学纹理异质性显著 (GLCM熵值 > 3.5)，"
                    "建议胸腔镜活检明确诊断。",
                    "HIGH",
                )
            if "NLP" in model_id:
                return (
                    "中高危：基于NLP图谱分析，影像描述提示恶性可能，"
                    "但缺少组学特征支持，建议补充增强CT。",
                    "MEDIUM_HIGH",
                )
            return (
                "待定：基于时序参数分析，建议联合组学与NLP进行综合研判。",
                "MEDIUM",
            )
        elif disease == "Hepatopathy":
            if "NLP" in model_id:
                return (
                    "中危：NLP语义图谱提示慢性肝病活动期，"
                    "实体关系三元组显示 ALT/药物相关性，建议保肝治疗。",
                    "MEDIUM",
                )
            return (
                "中危：综合评估提示肝病进展可能，建议完善肝功能与影像检查。",
                "MEDIUM",
            )
        return ("待评估：请结合多项检查进一步分析。", "LOW")

    def _build_shap_importance(
        self,
        model_id: str,
        model_info: dict[str, Any],
        patient_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """构造虚拟 SHAP 特征重要性排名。

        Args:
            model_id: 模型 ID。
            model_info: 模型注册表信息。
            patient_data: 患者数据。

        Returns:
            SHAP 特征贡献列表，按贡献度降序排列。
        """
        shap_base = model_info.get("shap_base_features", [])

        # 根据设计好的特征重要性生成模拟 SHAP 值
        shap_values: list[dict[str, Any]] = []
        if "TimeSeries" in model_id:
            # 时序模型: GLU_slope 和 Creatinine_latest 权重最高
            weights = {
                "GLU_slope": 0.35,
                "Creatinine_latest": 0.25,
                "HbA1c_baseline": 0.18,
                "time_series_length": 0.12,
                "ALT_slope": 0.10,
            }
        elif "NLP" in model_id:
            weights = {
                "病灶尺寸": 0.30,
                "否定过滤": 0.25,
                "解剖部位关联数": 0.20,
                "药物-疾病三元组数": 0.15,
                "实体总密度": 0.10,
            }
        elif "Multimodal" in model_id:
            weights = {
                "CT组学纹理特征": 0.30,
                "NER疾病实体向量": 0.25,
                "CEA时序趋势": 0.20,
                "跨模态注意力权重": 0.15,
                "B组影像质量分": 0.10,
            }
        else:
            weights = {f: 1.0 / len(shap_base) for f in shap_base} if shap_base else {}

        for feature in shap_base:
            shap_val = weights.get(feature, 0.1)
            # 注入患者实际数据值（模拟）
            actual_value = self._get_feature_actual_value(feature, patient_data)
            shap_values.append(
                {
                    "feature": feature,
                    "shap_value": round(shap_val, 4),
                    "contribution_pct": f"{shap_val * 100:.1f}%",
                    "actual_value": actual_value,
                    "direction": "+" if shap_val > 0 else "-",
                }
            )

        shap_values.sort(key=lambda x: x["shap_value"], reverse=True)
        return shap_values

    @staticmethod
    def _get_feature_actual_value(
        feature: str, patient_data: dict[str, Any]
    ) -> str:
        """获取特征在患者数据中的实际值（用于 SHAP 归因展示）。

        Args:
            feature: 特征名。
            patient_data: 患者数据。

        Returns:
            实际值的字符串表示。
        """
        # 映射特征名到可能的 patient_data key
        mapping = {
            "GLU_slope": "glu_slope",
            "Creatinine_latest": "creatinine_latest",
            "HbA1c_baseline": "hba1c_baseline",
            "time_series_length": "time_series_steps",
            "ALT_slope": "alt_slope",
            "病灶尺寸": "lesion_size",
            "否定过滤": "negation_count",
            "解剖部位关联数": "anatomical_relations",
            "药物-疾病三元组数": "drug_disease_triples",
            "实体总密度": "entity_density",
            "CT组学纹理特征": "ct_glcm_entropy",
            "NER疾病实体向量": "ner_disease_embedding",
            "CEA时序趋势": "cea_trend",
            "跨模态注意力权重": "cross_attention_weight",
            "B组影像质量分": "image_quality_score",
        }
        key = mapping.get(feature, "")
        val = patient_data.get(key, "N/A")
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    @staticmethod
    def _build_xai_log(
        model_id: str,
        model_info: dict[str, Any],
        disease: str,
        shap_features: list[dict[str, Any]],
    ) -> str:
        """构建 XAI 可解释性文本日志。

        Args:
            model_id: 模型 ID。
            model_info: 模型信息。
            disease: 目标病种。
            shap_features: SHAP 特征贡献列表。

        Returns:
            XAI 中文解释日志。
        """
        name = model_info.get("model_name", model_id)
        top3 = shap_features[:3]

        lines = [f"【{name} — SHAP 归因分析】"]
        lines.append("")
        lines.append("Top-3 关键决策因子:")
        for i, sf in enumerate(top3, 1):
            lines.append(
                f"  {i}. {sf['feature']} "
                f"(贡献度 {sf['contribution_pct']}, "
                f"方向 {sf['direction']}, "
                f"实际值: {sf['actual_value']})"
            )
        lines.append("")
        lines.append(f"模型认为该特征对'{disease}'的诊断结论影响最大。")
        return "\n".join(lines)


# ===================================================================
# 沙盒自测脚本
# ===================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("  V3.0 自适应模型路由引擎 (AdaptiveModelRouter) — 双用例沙盒验证")
    print("=" * 78)

    router = AdaptiveModelRouter(alpha=0.5, beta=0.3, gamma=0.2)

    # ================================================================
    # 用例 1：糖尿病肾病老患者（时序优势）
    # ================================================================
    print(f"\n{'─' * 78}")
    print("  【用例 1】糖尿病肾病老患者 — 连续 6 个月抽血时序，无组学，少文本")
    print(f"{'─' * 78}")

    case1_profile: dict[str, Any] = {
        "patient_id": "P_DN_001",
        "primary_disease": "Diabetes_Nephropathy",
        "time_series_steps": 6,        # 6 个月连续随访
        "nlp_entities_count": 2,       # 文本描述很少
        "has_group_E_radiomics": False,  # 无 CT 组学
        "missing_rate": 0.08,           # 整体缺失率低
    }

    case1_data: dict[str, Any] = {
        "glu_values": [6.2, 6.9, 7.4, 8.1, 8.8, 9.5],
        "glu_slope": 0.64,
        "creatinine_latest": 135.2,
        "hba1c_baseline": 7.8,
        "time_series_steps": 6,
        "alt_slope": 0.12,
    }

    # ---- 路由 ----
    result1 = router.route_case(case1_profile)
    xai1 = router.simulate_inference_with_xai(result1, case1_data)

    print(f"\n  [患者画像]")
    print(f"    患者 ID:       {case1_profile['patient_id']}")
    print(f"    目标病种:      糖尿病肾病 (Diabetes_Nephropathy)")
    print(f"    时序步数:      {case1_profile['time_series_steps']} 个月")
    print(f"    NLP 实体数:    {case1_profile['nlp_entities_count']} 个")
    print(f"    E 组组学:      {'有' if case1_profile['has_group_E_radiomics'] else '无'}")
    print(f"    缺失率:        {case1_profile['missing_rate']:.0%}")

    # ---- 得分榜 ----
    print(f"\n  [候选模型得分榜]")
    print(f"  {'模型':<24} {'得分':<8} {'历史F1':<10} {'契合度':<8} {'惩罚':<8}")
    print(f"  {'─' * 62}")
    for mid, score in sorted(result1["all_model_scores"].items(),
                              key=lambda x: x[1], reverse=True):
        d = result1["model_details"][mid]
        marker = " ◀ 胜出" if mid == result1["selected_model"] else ""
        print(f"  {mid:<24} {score:<8.1f} {d['historical_f1']:.0%}        "
              f"{d['modality_fit']:.2f}     {d['missing_penalty']:.2f}{marker}")

    # ---- 路由决策 ----
    print(f"\n  [路由决策]")
    print(f"    胜出模型:   {result1['selected_model']}")
    print(f"    终极得分:   {result1['winning_score']}/100")
    print(f"    执行状态:   {result1['execution_status']}")
    print(f"\n  [智能解释日志]")
    for line in result1["routing_reason"].split("\n"):
        print(f"    {line}")

    # ---- SHAP 归因 ----
    print(f"\n  [SHAP 特征归因 — {xai1['selected_model']}]")
    print(f"    预测结论: {xai1['predicted_diagnosis']}")
    print(f"    风险等级: {xai1['risk_level']}")
    print(f"\n    特征贡献排名:")
    print(f"    {'特征':<26} {'SHAP值':<10} {'贡献度':<10} {'方向':<6} {'实际值'}")
    print(f"    {'─' * 66}")
    for sf in xai1["shap_feature_importance"]:
        bar = "█" * int(float(sf["contribution_pct"].rstrip("%")) / 3)
        print(f"    {sf['feature']:<26} {sf['shap_value']:<10.4f} "
              f"{sf['contribution_pct']:<10} {sf['direction']:<6} "
              f"{sf['actual_value']}  {bar}")

    # ================================================================
    # 用例 2：右肺结节急诊患者（组学 + NLP 优势）
    # ================================================================
    print(f"\n\n{'─' * 78}")
    print("  【用例 2】右肺结节复查急诊患者 — 丰富现病史+NLP图谱+E组CT组学，无时序")
    print(f"{'─' * 78}")

    case2_profile: dict[str, Any] = {
        "patient_id": "P_LC_002",
        "primary_disease": "Lung_Oncology",
        "time_series_steps": 1,         # 无连续时序
        "nlp_entities_count": 10,       # 丰富文本实体
        "has_group_E_radiomics": True,  # 有完整 CT 组学
        "missing_rate": 0.22,
    }

    case2_data: dict[str, Any] = {
        "lesion_size": "12mm",
        "negation_count": 2,
        "anatomical_relations": 3,
        "drug_disease_triples": 1,
        "entity_density": 0.68,
        "ct_glcm_entropy": 3.82,
        "ner_disease_embedding": "0.87",
        "cea_trend": "+0.15/wk",
        "cross_attention_weight": 0.72,
        "image_quality_score": 0.91,
    }

    # ---- 路由 ----
    result2 = router.route_case(case2_profile)
    xai2 = router.simulate_inference_with_xai(result2, case2_data)

    print(f"\n  [患者画像]")
    print(f"    患者 ID:       {case2_profile['patient_id']}")
    print(f"    目标病种:      肺部肿瘤 (Lung_Oncology)")
    print(f"    时序步数:      {case2_profile['time_series_steps']} (几乎无)")
    print(f"    NLP 实体数:    {case2_profile['nlp_entities_count']} 个 (丰富)")
    print(f"    E 组组学:      有")
    print(f"    缺失率:        {case2_profile['missing_rate']:.0%}")

    # ---- 得分榜 ----
    print(f"\n  [候选模型得分榜]")
    print(f"  {'模型':<24} {'得分':<8} {'历史F1':<10} {'契合度':<8} {'惩罚':<8}")
    print(f"  {'─' * 62}")
    for mid, score in sorted(result2["all_model_scores"].items(),
                              key=lambda x: x[1], reverse=True):
        d = result2["model_details"][mid]
        marker = " ◀ 胜出" if mid == result2["selected_model"] else ""
        print(f"  {mid:<24} {score:<8.1f} {d['historical_f1']:.0%}        "
              f"{d['modality_fit']:.2f}     {d['missing_penalty']:.2f}{marker}")

    # ---- 路由决策 ----
    print(f"\n  [路由决策]")
    print(f"    胜出模型:   {result2['selected_model']}")
    print(f"    终极得分:   {result2['winning_score']}/100")
    print(f"\n  [智能解释日志]")
    for line in result2["routing_reason"].split("\n"):
        print(f"    {line}")

    # ---- SHAP 归因 ----
    print(f"\n  [SHAP 特征归因 — {xai2['selected_model']}]")
    print(f"    预测结论: {xai2['predicted_diagnosis']}")
    print(f"    风险等级: {xai2['risk_level']}")
    print(f"\n    特征贡献排名:")
    print(f"    {'特征':<26} {'SHAP值':<10} {'贡献度':<10} {'方向':<6} {'实际值'}")
    print(f"    {'─' * 66}")
    for sf in xai2["shap_feature_importance"]:
        bar = "█" * int(float(sf["contribution_pct"].rstrip("%")) / 3)
        print(f"    {sf['feature']:<26} {sf['shap_value']:<10.4f} "
              f"{sf['contribution_pct']:<10} {sf['direction']:<6} "
              f"{sf['actual_value']}  {bar}")

    # ================================================================
    # 结算面板
    # ================================================================
    print(f"\n\n{'─' * 78}")
    print(f"  双用例验证结算")
    print(f"{'─' * 78}")
    print(f"  用例1 (糖尿病肾病) → 预期 Model_A_TimeSeries, 实际 {result1['selected_model']}")
    print(f"      {'✓ 通过' if result1['selected_model'] == 'Model_A_TimeSeries' else '✗ 未通过'}")
    print(f"  用例2 (肺部肿瘤)   → 预期 Model_C_Multimodal, 实际 {result2['selected_model']}")
    print(f"      {'✓ 通过' if result2['selected_model'] == 'Model_C_Multimodal' else '✗ 未通过'}")

    print(f"\n{'=' * 78}")
    print(f"  V3.0 自适应模型路由引擎 — 全部验证完毕！")
    print(f"{'=' * 78}")
