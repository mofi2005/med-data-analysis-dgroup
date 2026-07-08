#!/usr/bin/env python3
"""多模态数据对齐与全景宽表拼装引擎 (Multimodal Aligner & Wide-Table Builder).

本模块是 D 组项目第五阶段（多模态融合与跨组联动）的核心工程。
在真实 MLOps 流水线中，来自不同渠道或开发小组的数据通常是离散且异构的。
本引擎以 ``case_id`` 为全局对齐主键，将各组异构碎片自动清洗、对齐并
拼接成一张规范的患者全景特征宽表，直接供大模型推理及报告系统使用。

支持的命名空间:
  - d_group_nlp          D 组自研 NLP 实体图谱（阶段二）
  - d_group_stats        D 组自研时序生化特征（阶段三）
  - group_a_simulation   A 组物理仿真病灶参数
  - group_b_quality      B 组图像伪影质量评分
  - group_c_pathology    C 组病理分割标注
  - group_e_radiomics    E 组 CT 放射组学特征
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 命名空间与默认字段映射契约
# ---------------------------------------------------------------------------

# 各组的数据路径映射：「外部碎片 JSON key → 扁平化后的 unified_feature key」
_NAMESPACE_FEATURE_MAP: dict[str, dict[str, str | None]] = {
    "d_group_nlp": {
        "primary_disease": "primary_disease",
        "negation_count": "negation_count",
        "total_entities": "nlp_total_entities",
        "has_nodule": "has_imaging_nodule",
        "disease_list": None,  # 不过滤，直接透传
    },
    "d_group_stats": {
        "GLU_latest": "GLU_latest",
        "GLU_slope": "GLU_slope",
        "Creatinine_latest": "Creatinine_latest",
        "HbA1c_latest": "HbA1c_latest",
        "ALT_latest": "ALT_latest",
        "GLU_baseline": "GLU_baseline",
        "GLU_change_rate": "GLU_change_rate",
    },
    "group_a_simulation": {
        "lesion_volume_cm3": "sim_lesion_volume",
        "vessel_density": "sim_vessel_density",
        "tissue_elasticity_kPa": "sim_tissue_elasticity",
        "thermal_diffusivity": "sim_thermal_diffusivity",
    },
    "group_b_quality": {
        "image_quality_score": "image_quality_score",
        "artifact_severity": "artifact_severity",
        "snr_db": "image_snr_db",
        "motion_blur_present": "has_motion_blur",
    },
    "group_c_pathology": {
        "segmentation_dice": "patho_seg_dice",
        "cell_density_per_mm2": "patho_cell_density",
        "mitosis_count_hpf": "patho_mitosis_count",
        "ki67_pct": "patho_ki67_pct",
    },
    "group_e_radiomics": {
        "texture_mean": "radiomics_texture_mean",
        "texture_std": "radiomics_texture_std",
        "glcm_entropy": "radiomics_glcm_entropy",
        "glcm_contrast": "radiomics_glcm_contrast",
        "shape_sphericity": "radiomics_shape_sphericity",
    },
}

# 质量告警阈值
_QUALITY_ALERT_THRESHOLD: float = 0.6

# 所有外部命名空间
_ALL_NAMESPACES: list[str] = [
    "d_group_nlp",
    "d_group_stats",
    "group_a_simulation",
    "group_b_quality",
    "group_c_pathology",
    "group_e_radiomics",
]


# ===================================================================
# MultimodalAligner
# ===================================================================


class MultimodalAligner:
    """多模态数据对齐与全景宽表拼装引擎。

    以 ``case_id`` 为全局主键，将 A/B/C/E 外组及 D 组自研的异构
    碎片数据清洗、扁平化，并拼接为统一的患者全景特征宽表。

    Attributes:
        feature_map: 命名空间 → 扁平化特征名映射字典。
        quality_threshold: 图像质量告警阈值。
    """

    def __init__(self) -> None:
        """初始化对齐引擎，加载内部映射规则与阈值。"""
        self.feature_map: dict[str, dict[str, str | None]] = _NAMESPACE_FEATURE_MAP
        self.quality_threshold: float = _QUALITY_ALERT_THRESHOLD
        self._all_namespaces: list[str] = _ALL_NAMESPACES

    # ------------------------------------------------------------------
    # 跨组碎片数据对齐与安全验证
    # ------------------------------------------------------------------

    def align_case_fragments(
        self, case_id: str, fragments: dict[str, Any]
    ) -> dict[str, Any]:
        """以 ``case_id`` 为铁律，对各组碎片数据进行清洗与扁平化对齐。

        处理流程:
          1. 校验 ``case_id`` 一致性
          2. 遍历六组命名空间，提取核心指标
          3. 生成质量告警标记
          4. 计算模态完整度

        Args:
            case_id: 病例全局唯一标识。
            fragments: 各组原始数据字典，key 为命名空间名，
                value 为该组的数据 payload。

        Returns:
            对齐后的标准化特征字典，含:
            - aligned_case_id
            - unified_features (扁平化特征)
            - modality_completeness (模态完整度)
            - quality_artifact_warning
            - raw_groups_connected
            - warnings (对齐过程中的告警列表)
        """
        # Step 1: 校验 case_id
        warnings: list[str] = []
        raw_case = fragments.get("case_id", case_id)
        if str(raw_case) != str(case_id):
            warnings.append(
                f"case_id 不一致: 期望 {case_id}, 实际 {raw_case}, "
                f"将强制使用 {case_id}"
            )

        # Step 2: 遍历各组扁平化提取
        unified_features: dict[str, Any] = {}
        connected_groups: list[str] = []

        for ns in self._all_namespaces:
            if ns not in fragments:
                continue
            group_data = fragments[ns]
            if group_data is None or (isinstance(group_data, dict) and not group_data):
                continue
            connected_groups.append(ns)

            # 提取该命名空间下的映射字段
            self._flatten_group(ns, group_data, unified_features, warnings)

        # Step 3: 质量告警检测
        quality_warning = self._check_quality_warning(fragments)

        # Step 4: 计算模态完整度
        completeness = self._compute_completeness(fragments)

        return {
            "aligned_case_id": case_id,
            "unified_features": unified_features,
            "modality_completeness": completeness,
            "quality_artifact_warning": quality_warning,
            "raw_groups_connected": connected_groups,
            "total_groups_connected": len(connected_groups),
            "warnings": warnings,
        }

    def _flatten_group(
        self,
        namespace: str,
        group_data: dict[str, Any],
        unified: dict[str, Any],
        warnings: list[str],
    ) -> None:
        """将单个命名空间的原始数据扁平化提取到统一特征字典。

        Args:
            namespace: 命名空间名。
            group_data: 该组原始数据字典。
            unified: 积累的统一特征字典（原地修改）。
            warnings: 告警列表（原地追加）。
        """
        mapping = self.feature_map.get(namespace, {})
        if not mapping:
            warnings.append(f"未注册的命名空间: {namespace}, 跳过")
            return

        for raw_key, flat_key in mapping.items():
            if raw_key not in group_data:
                continue
            val = group_data[raw_key]

            if flat_key is None:
                # None 表示透传原始键名
                unified[raw_key] = val
            else:
                unified[flat_key] = val

        # ---- 组专属衍生特征 ----
        # E 组: 若纹理特征已提取，自动计算均值
        if namespace == "group_e_radiomics":
            self._derive_radiomics_aggregates(group_data, unified)

        # D 组 NLP: 衍生否定标记
        if namespace == "d_group_nlp":
            self._derive_nlp_flags(group_data, unified)

    @staticmethod
    def _derive_radiomics_aggregates(
        group_data: dict[str, Any], unified: dict[str, Any]
    ) -> None:
        """从 E 组放射组学数据中自动计算聚合统计量。

        若原始数据中未提供 texture_mean，则从多个纹理字段计算均值。

        Args:
            group_data: E 组原始数据。
            unified: 统一特征字典。
        """
        # 若已有 texture_mean 则跳过
        if "radiomics_texture_mean" in unified:
            return

        texture_fields = [
            "texture_mean",
            "texture_std",
            "glcm_entropy",
            "glcm_contrast",
            "glcm_energy",
            "glcm_homogeneity",
            "glrlm_run_length",
            "glszm_zone_size",
        ]
        values = [
            float(group_data[k])
            for k in texture_fields
            if k in group_data and isinstance(group_data[k], (int, float))
        ]
        if values:
            unified["radiomics_texture_mean"] = round(
                sum(values) / len(values), 2
            )

    @staticmethod
    def _derive_nlp_flags(
        group_data: dict[str, Any], unified: dict[str, Any]
    ) -> None:
        """从 D 组 NLP 数据中推导否定与病灶标记。

        Args:
            group_data: D 组 NLP 原始数据。
            unified: 统一特征字典。
        """
        # 否定标记
        # 若 entities 中存在 negation_status=True，则标记 distant_metastasis
        entities = group_data.get("entities", [])
        if entities:
            negation_count = sum(
                1 for e in entities if e.get("negation_status")
            )
            if negation_count > 0:
                unified["has_negated_finding"] = True
                unified["negation_count"] = unified.get("negation_count", negation_count)

            # 检查是否有远处转移且被否定
            for e in entities:
                if e.get("standard_name") == "distant_metastasis":
                    unified["distant_metastasis"] = not e.get("negation_status", False)
                    break

        # 是否有影像结节
        if "has_imaging_nodule" not in unified:
            unified["has_imaging_nodule"] = group_data.get("has_nodule", None)

    def _check_quality_warning(self, fragments: dict[str, Any]) -> bool:
        """检测是否需要发出图像质量预警。

        触发条件:
          - B 组 image_quality_score < 0.6
          - 或 B 组 artifact_severity == "severe"
          - 或 B 组 motion_blur_present == True

        Args:
            fragments: 各组原始数据。

        Returns:
            是否需要预警。
        """
        b_data = fragments.get("group_b_quality")
        if not b_data or not isinstance(b_data, dict):
            return False

        score = b_data.get("image_quality_score", 1.0)
        severity = b_data.get("artifact_severity", "none")
        motion_blur = b_data.get("motion_blur_present", False)

        if isinstance(score, (int, float)) and float(score) < self.quality_threshold:
            return True
        if str(severity).lower() in ("severe", "moderate"):
            return True
        if motion_blur:
            return True

        return False

    def _compute_completeness(self, fragments: dict[str, Any]) -> dict[str, Any]:
        """计算多模态数据完整度。

        Args:
            fragments: 各组原始数据。

        Returns:
            模态完整度字典。
        """
        def _has_data(data: Any) -> bool:
            if data is None:
                return False
            if isinstance(data, dict) and not data:
                return False
            return True

        has_nlp = _has_data(fragments.get("d_group_nlp"))
        has_ts = _has_data(fragments.get("d_group_stats"))
        has_rad = _has_data(fragments.get("group_e_radiomics"))
        has_sim = _has_data(fragments.get("group_a_simulation"))
        has_qual = _has_data(fragments.get("group_b_quality"))
        has_path = _has_data(fragments.get("group_c_pathology"))

        present = sum([has_nlp, has_ts, has_rad, has_sim, has_qual, has_path])
        overall_missing = round(1.0 - present / len(self._all_namespaces), 2)

        return {
            "has_clinical_nlp": has_nlp,
            "has_time_series_labs": has_ts,
            "has_radiomics_E": has_rad,
            "has_simulation_A": has_sim,
            "has_quality_B": has_qual,
            "has_pathology_C": has_path,
            "total_modalities_present": present,
            "total_modalities_expected": len(self._all_namespaces),
            "overall_missing_rate": overall_missing,
        }

    # ------------------------------------------------------------------
    # 全景宽表拼装与格式标准化
    # ------------------------------------------------------------------

    def build_wide_table_payload(
        self,
        case_id: str,
        patient_id: str,
        fragments: dict[str, Any],
    ) -> dict[str, Any]:
        """执行全自动多模态数据拼装，输出标准化全景宽表。

        流程:
          1. 调用 ``align_case_fragments`` 提取对齐后的特征
          2. 包装为标准化输出结构
          3. 附加元信息 (时间戳、案例/患者 ID、完整度摘要)

        Args:
            case_id: 病例唯一标识。
            patient_id: 患者唯一标识。
            fragments: 各组原始数据碎片。

        Returns:
            标准化全景宽表字典，含:
            - case_id, patient_id, timestamp
            - modality_completeness
            - unified_features
            - quality_artifact_warning
            - raw_fragments_summary
        """
        aligned = self.align_case_fragments(case_id, fragments)

        # 注入全局质量标记
        unified_features = aligned["unified_features"]
        unified_features["quality_artifact_warning"] = aligned["quality_artifact_warning"]

        # 生成时间戳
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        payload: dict[str, Any] = {
            "case_id": case_id,
            "patient_id": patient_id,
            "timestamp": timestamp,
            "modality_completeness": aligned["modality_completeness"],
            "unified_features": unified_features,
            "quality_artifact_warning": aligned["quality_artifact_warning"],
            "raw_fragments_summary": {
                "total_groups_connected": aligned["total_groups_connected"],
                "connected_groups": aligned["raw_groups_connected"],
                "alignment_warnings": aligned["warnings"],
            },
        }

        return payload


# ===================================================================
# 沙盒自测脚本
# ===================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("  多模态数据对齐与全景宽表拼装引擎 (MultimodalAligner)")
    print("  跨组联合自测 — 糖尿病肾病 case_1001")
    print("=" * 78)

    aligner = MultimodalAligner()

    # ================================================================
    # 模拟跨组数据碎片
    # ================================================================

    # D 组 NLP 成果（阶段二 NER + 关系抽取产物）
    d_nlp_fragment: dict[str, Any] = {
        "primary_disease": "diabetes_mellitus",
        "total_entities": 8,
        "negation_count": 0,
        "has_nodule": False,
        "disease_list": ["hypertension", "diabetes_mellitus", "nephropathy"],
        "entities": [
            {
                "entity": "高血压",
                "entity_type": "disease_or_lesion",
                "standard_name": "hypertension",
                "negation_status": False,
                "history_status": True,
            },
            {
                "entity": "远处转移",
                "entity_type": "imaging_finding",
                "standard_name": "distant_metastasis",
                "negation_status": True,
            },
        ],
    }

    # D 组 统计成果（阶段三 时序分析产物）
    d_stats_fragment: dict[str, Any] = {
        "GLU_latest": 8.8,
        "GLU_slope": 0.64,
        "GLU_baseline": 6.2,
        "GLU_change_rate": 0.419,
        "Creatinine_latest": 135.2,
        "HbA1c_latest": 8.5,
        "ALT_latest": 42.0,
    }

    # E 组放射组学成果
    e_radiomics_fragment: dict[str, Any] = {
        "texture_mean": 55.2,
        "texture_std": 12.7,
        "glcm_entropy": 3.45,
        "glcm_contrast": 28.3,
        "shape_sphericity": 0.88,
    }

    # B 组图像质控成果
    b_quality_fragment: dict[str, Any] = {
        "image_quality_score": 0.95,
        "artifact_severity": "none",
        "snr_db": 42.5,
        "motion_blur_present": False,
    }

    # ---- 组装碎片字典 ----
    fragments: dict[str, Any] = {
        "case_id": "case_1001",
        "d_group_nlp": d_nlp_fragment,
        "d_group_stats": d_stats_fragment,
        "group_e_radiomics": e_radiomics_fragment,
        "group_b_quality": b_quality_fragment,
        # 以下两组未连接，模拟部分缺失
        # "group_a_simulation": ...,
        # "group_c_pathology": ...,
    }

    # ================================================================
    # 执行全自动拼装
    # ================================================================
    print(f"\n[输入碎片]")
    print(f"  case_id:  case_1001")
    print(f"  patient:  p8823")
    print(f"  已连接组: D-NLP, D-Stats, E-Radiomics, B-Quality (共 4 组)")
    print(f"  未连接组: A-Simulation, C-Pathology")

    result = aligner.build_wide_table_payload(
        case_id="case_1001",
        patient_id="p8823",
        fragments=fragments,
    )

    # ================================================================
    # 第一部分：模态完整度监控面板
    # ================================================================
    print(f"\n{'─' * 78}")
    print(f"  【第一部分】模态完整度监控 (Modality Completeness)")
    print(f"{'─' * 78}")

    mc = result["modality_completeness"]
    items = [
        ("临床 NLP 图谱 (D 组)", mc.get("has_clinical_nlp", False)),
        ("生化时序特征 (D 组)", mc.get("has_time_series_labs", False)),
        ("CT 放射组学  (E 组)", mc.get("has_radiomics_E", False)),
        ("物理仿真参数  (A 组)", mc.get("has_simulation_A", False)),
        ("图像质控评分  (B 组)", mc.get("has_quality_B", False)),
        ("病理分割标注  (C 组)", mc.get("has_pathology_C", False)),
    ]

    for label, present in items:
        icon = "✓ 已连接" if present else "✗ 缺失"
        status_color = "[ON]" if present else "[OFF]"
        bar = "████████████████" if present else "────────────────"
        print(f"  {status_color} {label:<28} {bar} {icon}")

    print(f"\n  已连接模态:  {mc.get('total_modalities_present', 0)}/{mc.get('total_modalities_expected', 6)}")
    print(f"  整体缺失率:  {mc.get('overall_missing_rate', 0):.0%}")

    # ================================================================
    # 第二部分：扁平化统一特征矩阵
    # ================================================================
    print(f"\n{'─' * 78}")
    print(f"  【第二部分】扁平化统一特征矩阵 (Unified Features)")
    print(f"{'─' * 78}")

    uf = result["unified_features"]

    # 按来源分组展示
    groups_display = {
        "D 组 NLP 语义特征": [
            ("primary_disease", "主诊断"),
            ("has_negated_finding", "含否定发现"),
            ("distant_metastasis", "远处转移"),
            ("negation_count", "否定实体数"),
            ("nlp_total_entities", "NLP 实体总数"),
            ("disease_list", "疾病列表"),
        ],
        "D 组 时序生化特征": [
            ("GLU_baseline", "GLU 基线值 (mmol/L)"),
            ("GLU_latest", "GLU 最新值 (mmol/L)"),
            ("GLU_slope", "GLU 斜率 (Δ/月)"),
            ("GLU_change_rate", "GLU 变化率"),
            ("Creatinine_latest", "肌酐最新值 (μmol/L)"),
            ("HbA1c_latest", "HbA1c 最新值 (%)"),
            ("ALT_latest", "ALT 最新值 (U/L)"),
        ],
        "E 组 放射组学特征": [
            ("radiomics_texture_mean", "纹理均值"),
            ("radiomics_texture_std", "纹理标准差"),
            ("radiomics_glcm_entropy", "GLCM 熵"),
            ("radiomics_glcm_contrast", "GLCM 对比度"),
            ("radiomics_shape_sphericity", "形状球形度"),
        ],
        "B 组 图像质控特征": [
            ("image_quality_score", "影像质量评分"),
            ("artifact_severity", "伪影严重度"),
            ("image_snr_db", "信噪比 (dB)"),
            ("has_motion_blur", "运动模糊"),
        ],
        "自动衍生特征": [
            ("quality_artifact_warning", "质量预警标记"),
        ],
    }

    for section_label, field_list in groups_display.items():
        has_any = any(uf.get(key) is not None for key, _ in field_list)
        if not has_any:
            continue
        print(f"\n  ┌ {section_label}")
        for key, chinese in field_list:
            val = uf.get(key)
            if val is None or val == []:
                continue
            # 格式化展示
            if isinstance(val, bool):
                formatted = "TRUE" if val else "FALSE"
            elif isinstance(val, float):
                formatted = f"{val:.2f}" if abs(val) >= 0.01 else f"{val:.6f}"
            elif isinstance(val, list):
                formatted = ", ".join(str(v) for v in val)
            else:
                formatted = str(val)
            print(f"  │  {chinese:<24} → {formatted}")
        print(f"  └")

    # ================================================================
    # 第三部分：质量告警与连接摘要
    # ================================================================
    print(f"\n{'─' * 78}")
    print(f"  【第三部分】质量监控与连接摘要")
    print(f"{'─' * 78}")

    qa = result.get("quality_artifact_warning", False)
    if qa:
        print(f"\n  ⚠  质量预警: 图像存在严重伪影或质量评分 < {aligner.quality_threshold}")
    else:
        print(f"\n  ✓  质量状态: 正常 (B 组评分 {uf.get('image_quality_score', 'N/A')})")

    summary = result["raw_fragments_summary"]
    print(f"\n  [连接摘要]")
    print(f"    总连接组数: {summary['total_groups_connected']} / {len(aligner._all_namespaces)}")
    print(f"    已连接:      {', '.join(summary['connected_groups'])}")
    if summary["alignment_warnings"]:
        print(f"    对齐告警:")
        for w in summary["alignment_warnings"]:
            print(f"      - {w}")
    else:
        print(f"    对齐告警:    无")

    # ================================================================
    # 第四部分：完整 JSON 输出
    # ================================================================
    print(f"\n{'─' * 78}")
    print(f"  【第四部分】标准化全景宽表 JSON")
    print(f"{'─' * 78}\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"\n{'=' * 78}")
    print(f"  多模态宽表拼装 — 验证完毕！")
    print(f"{'=' * 78}")
