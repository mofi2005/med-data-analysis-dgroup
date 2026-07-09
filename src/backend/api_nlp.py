#!/usr/bin/env python3
"""NLP 结构化与关系抽取 API 路由 — D组医学数据中台.

本模块基于复合 NLP 引擎 (正则 + 医学词库 + 规则), 提供病历文本的
全维度结构化分析, 包括:
  - 医学实体抽取 (Entities) : 疾病/症状/部位/指标/检查等
  - 关系三元组 (Triples)   : subject - relation - object
  - 否定与既往识别 (Negations & History)

API:
  POST /api/nlp/extract  — 病历文本全维度结构化分析
"""

from __future__ import annotations

import re
from typing import Any, Optional

from fastapi import APIRouter

# ---------------------------------------------------------------------------
# Pydantic 请求/响应模型 (简单字典传递, 无需额外依赖)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# APIRouter 实例
# ---------------------------------------------------------------------------

nlp_router = APIRouter(
    prefix="/api/nlp",
    tags=["文本结构化与关系抽取"],
)

# ===================================================================
# 内置医学词库
# ===================================================================

# 疾病实体 → 标准英文映射
_DISEASE_MAP: dict[str, str] = {
    "糖尿病": "diabetes_mellitus",
    "高血压": "hypertension",
    "冠心病": "coronary_heart_disease",
    "肺炎": "pneumonia",
    "肺癌": "lung_cancer",
    "慢性肾病": "chronic_kidney_disease",
    "脑卒中": "stroke",
    "肝炎": "hepatitis",
    "肝硬化": "cirrhosis",
    "肺结核": "tuberculosis",
    "甲状腺结节": "thyroid_nodule",
}

# 症状/体征实体
_SYMPTOM_MAP: dict[str, str] = {
    "咳嗽": "cough",
    "发热": "fever",
    "胸痛": "chest_pain",
    "头痛": "headache",
    "乏力": "fatigue",
    "恶心": "nausea",
}

# 解剖部位实体
_ANATOMY_MAP: dict[str, str] = {
    "右肺上叶": "right_upper_lobe",
    "左肺上叶": "left_upper_lobe",
    "右肺下叶": "right_lower_lobe",
    "左肺下叶": "left_lower_lobe",
    "肝脏": "liver",
    "肾脏": "kidney",
    "甲状腺": "thyroid",
    "胸部": "chest",
    "腹部": "abdomen",
}

# 病灶/发现实体
_LESION_MAP: dict[str, str] = {
    "结节": "nodule",
    "肿块": "mass",
    "囊肿": "cyst",
    "钙化": "calcification",
    "渗出": "exudate",
    "阴影": "shadow",
    "磨玻璃": "ground_glass_opacity",
}

# 检查/指标实体
_EXAM_MAP: dict[str, str] = {
    "CT": "ct_scan",
    "MRI": "mri_scan",
    "超声": "ultrasound",
    "X光": "x_ray",
    "血常规": "blood_routine",
    "肝功能": "liver_function",
    "肾功能": "renal_function",
}

# 化验指标
_LAB_MAP: dict[str, str] = {
    "WBC": "wbc",
    "白细胞": "wbc",
    "HGB": "hgb",
    "血红蛋白": "hgb",
    "PLT": "plt",
    "血小板": "plt",
    "GLU": "glu",
    "血糖": "glu",
    "ALT": "alt",
    "AST": "ast",
    "Creatinine": "creatinine",
    "肌酐": "creatinine",
    "CEA": "cea",
}

# 否定触发词
_NEGATION_TRIGGERS: list[str] = [
    "否认", "无", "未见", "不考虑", "未发现", "排除",
    "无明显", "未示", "没有", "不包括", "未检测到",
]

# 既往时态触发词
_HISTORY_TRIGGERS: list[str] = [
    "既往", "病史", "年前", "月前", "术后", "曾患", "平素",
    "史", "曾因", "有.*史", "长年",
]

# 关系介词映射
_RELATION_PATTERNS: list[dict[str, Any]] = [
    # 1. 解剖部位 ← 存在/发现 → 病灶  (输出时 subject=病灶, relation=位于, object=部位)
    {"sub_type": "lesion", "rel": "位于", "obj_type": "anatomy",
     "pattern": r"(右肺上叶|左肺上叶|右肺下叶|左肺下叶|甲状腺|肝脏|肾脏).{0,12}(存在|发现|可见|发现)[^\s，。；]{0,10}(结节|肿块|阴影|囊肿|钙化|磨玻璃)",
     "reverse": True},
    # 2. 病灶 → 位于/见于 → 部位
    {"sub_type": "lesion", "rel": "位于", "obj_type": "anatomy",
     "pattern": r"(结节|肿块|阴影|囊肿|钙化|磨玻璃).{0,12}(位于|见于|在).{0,6}(右肺上叶|左肺上叶|右肺下叶|左肺下叶|甲状腺|肝脏|肾脏|[^\s，。；\d]{2,4})",
     "reverse": False},
    # 3. 化验指标 → 状态 (升高/降低/正常)
    {"sub_type": "lab", "rel": "状态", "obj_type": "state",
     "pattern": r"(WBC|白细胞|GLU|血糖|ALT|AST|HGB|血红蛋白|PLT|血小板|肌酐|Creatinine|CEA).{0,8}(升高|增高|降低|下降|正常|异常)",
     "reverse": False},
    # 4. 检查 → CT 等显示某个发现 (更宽松的匹配)
    {"sub_type": "exam", "rel": "提示", "obj_type": "any",
     "pattern": r"(CT|MRI|超声|X光|血常规|肝功能).{0,4}(显示|示|可见|发现).{0,30}?(结节|肿块|阴影|囊肿|WBC|白细胞|升高|异常|钙化|磨玻璃|肺炎|炎症)",
     "reverse": False},
]

# 量化发现正则
_SIZE_PATTERN = re.compile(r"(\d+\.?\d*)\s*cm")
_MM_PATTERN = re.compile(r"(\d+\.?\d*)\s*mm")


# ===================================================================
# NLP 核心分析引擎
# ===================================================================


def _find_all_entities(text: str) -> list[dict]:
    """从文本中抽取所有医学实体 (复合规则 + 词典匹配).

    Args:
        text: 原始病历文本。

    Returns:
        实体列表, 每项含 entity, entity_type, start, end, standard_name, confidence。
    """
    entities: list[dict] = []
    occupied: set[tuple[int, int]] = set()  # 已占用的位置区间, 避免重复

    def _add(entity: str, etype: str, start: int, end: int, std_name: str,
             confidence: float = 0.95) -> None:
        if (start, end) in occupied:
            return
        occupied.add((start, end))
        entities.append({
            "entity": entity,
            "entity_type": etype,
            "start": start,
            "end": end,
            "standard_name": std_name,
            "confidence": round(confidence, 2),
        })

    # 按词长降序 (长词优先匹配, 避免碎片化)
    all_maps: list[tuple[dict[str, str], str]] = [
        (_ANATOMY_MAP, "anatomical_site"),
        (_LESION_MAP, "disease_or_lesion"),
        (_DISEASE_MAP, "disease_or_lesion"),
        (_SYMPTOM_MAP, "symptom"),
        (_EXAM_MAP, "examination"),
        (_LAB_MAP, "laboratory"),
    ]

    for mapping, etype in all_maps:
        # 按词长降序排列
        for kw in sorted(mapping, key=len, reverse=True):
            start = 0
            while True:
                idx = text.find(kw, start)
                if idx == -1:
                    break
                _add(kw, etype, idx, idx + len(kw), mapping[kw])
                start = idx + 1

    # 量化发现: 尺寸
    for m in _SIZE_PATTERN.finditer(text):
        val = m.group(1)
        _add(f"{val}cm", "imaging_finding", m.start(), m.end(), f"lesion_size_{val}cm", 0.85)
    for m in _MM_PATTERN.finditer(text):
        val = m.group(1)
        _add(f"{val}mm", "imaging_finding", m.start(), m.end(), f"lesion_size_{val}mm", 0.85)

    # 排序: 按 start 升序
    entities.sort(key=lambda x: x["start"])
    return entities


def _extract_triples(text: str, entities: list[dict]) -> list[dict]:
    """基于规则抽取关系三元组 (subject - relation - object).

    Args:
        text:     原始文本。
        entities: 已抽取的实体列表 (用于补充 subject/object 细节)。

    Returns:
        三元组列表。
    """
    triples: list[dict] = []

    # 实体按 start 排序, 构建快速查找
    entity_map: dict[str, list[dict]] = {}
    for e in entities:
        key = e["entity"]
        entity_map.setdefault(key, []).append(e)

    for rule in _RELATION_PATTERNS:
        for m in re.finditer(rule["pattern"], text):
            groups = list(m.groups())
            if len(groups) < 2:
                continue

            # 根据 reverse 标志决定 subject/object 顺序
            if rule.get("reverse") and len(groups) >= 3:
                # 反向: groups[0]=部位, groups[-1]=病灶  → 输出 病灶-位于-部位
                subj_word = groups[-1]  # 病灶
                obj_word = groups[0]    # 部位
            elif len(groups) >= 3:
                subj_word = groups[0]
                obj_word = groups[-1]
            elif len(groups) == 2:
                subj_word = groups[0]
                obj_word = groups[1]
            else:
                continue

            # 过滤数字/乱码
            if any(c.isdigit() for c in subj_word) and len(subj_word) <= 3:
                continue
            if any(c.isdigit() for c in obj_word) and len(obj_word) <= 3:
                continue

            rel_word = rule["rel"]
            conf = 0.92

            subj_std = entity_map.get(subj_word, [{}])[0].get("standard_name", subj_word)
            obj_std = entity_map.get(obj_word, [{}])[0].get("standard_name", obj_word)

            triples.append({
                "subject": subj_word,
                "subject_standard": subj_std,
                "relation": rel_word,
                "object": obj_word,
                "object_standard": obj_std,
                "confidence": round(conf, 2),
            })

    # 去重 (基于 subject+relation+object 三元组)
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for t in triples:
        key = (t["subject"], t["relation"], t["object"])
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def _detect_negations_and_history(text: str, entities: list[dict]) -> list[dict]:
    """检测否定表达与既往时态，为实体注入布尔标记。

    策略:
      - 对每个实体, 截取其所在的子句 (以 。，；! 为边界)。
      - 检查子句中是否在实体之前存在否定/既往触发词。

    Args:
        text:     原始文本。
        entities: 已抽取的实体列表。

    Returns:
        带否定/既往标记的实体列表 (仅返回有标记的)。
    """
    results: list[dict] = []

    # 子句切分
    clauses = re.split(r"[。！；\n]", text)

    for ent in entities:
        s, e = ent["start"], ent["end"]
        # 找到该实体所在的子句
        clause = ""
        clause_start = 0
        for cl in clauses:
            cl_pos = text.find(cl, clause_start)
            if cl_pos == -1:
                continue
            cl_end = cl_pos + len(cl)
            if s >= cl_pos and e <= cl_end:
                clause = cl
                break
            clause_start = cl_end
        if not clause:
            clause = text

        # 实体在子句中的相对位置
        rel_start = s - max(0, text.find(clause))

        # 否定检测: 触发词必须在实体之前
        neg_hit = False
        neg_word = ""
        for trig in _NEGATION_TRIGGERS:
            idx = clause.find(trig)
            if idx != -1 and idx < rel_start:
                # 且实体和触发词之间没有句号/分号等较强分隔
                between = clause[idx:rel_start]
                if "。" not in between and "；" not in between:
                    neg_hit = True
                    neg_word = trig
                break

        # 既往检测
        hist_hit = False
        hist_word = ""
        for trig in _HISTORY_TRIGGERS:
            idx = clause.find(trig)
            if idx != -1 and idx < rel_start + len(ent["entity"]):
                # 如果是 "有.*史" 模式则用正则
                if ".*" in trig:
                    pat = trig.replace(".*", ".*?")
                    if re.search(pat, clause):
                        hist_hit = True
                        hist_word = trig.replace(".*", "…")
                        break
                else:
                    hist_hit = True
                    hist_word = trig
                    break

        if neg_hit or hist_hit:
            entry = {
                "entity": ent["entity"],
                "entity_type": ent["entity_type"],
                "start": ent["start"],
                "end": ent["end"],
                "standard_name": ent["standard_name"],
                "negation_status": neg_hit,
                "history_status": hist_hit,
                "source_sentence": clause.strip(),
            }
            if neg_hit:
                entry["negation_trigger"] = neg_word
            if hist_hit:
                entry["history_trigger"] = hist_word
            results.append(entry)

    return results


# ===================================================================
# POST /api/nlp/extract
# ===================================================================


@nlp_router.post("/extract")
def extract_clinical_text(payload: dict):
    """病历文本全维度结构化分析。

    接收一段自由文本病历, 返回:
      - entities:  医学实体列表 (含位置/标准名/置信度)
      - triples:   关系三元组
      - negations: 否定/既往状态识别

    Body 示例:
        {
          "text": "患者既往有糖尿病史，否认高血压...",
          "patient_id": "p001"
        }

    Returns:
        {"patient_id": "...", "text_length": N,
         "entities": [...], "triples": [...], "negations": [...]}
    """
    text: str = (payload.get("text") or "").strip()
    patient_id: str = payload.get("patient_id", "p001")

    if not text:
        return {
            "status": "error",
            "message": "text 字段不能为空",
        }

    # 1. 实体抽取
    entities = _find_all_entities(text)

    # 2. 三元组抽取
    triples = _extract_triples(text, entities)

    # 3. 否定与既往识别
    negations = _detect_negations_and_history(text, entities)

    return {
        "status": "success",
        "patient_id": patient_id,
        "text_length": len(text),
        "entities": entities,
        "triples": triples,
        "negations": negations,
        "summary": {
            "entity_count": len(entities),
            "triple_count": len(triples),
            "negation_history_count": len(negations),
        },
    }
