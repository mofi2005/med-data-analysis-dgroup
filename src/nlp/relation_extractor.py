#!/usr/bin/env python3
"""临床关系与状态分析引擎 (Medical Relation Extractor).

本模块承接 MedicalNEREngine 输出的离散实体，通过 NLP 规则引擎、
介词驱动与上下文窗口分析，完成两大核心临床任务：

  1. **临床状态识别**: 精确识别否定(negation)、疑似(uncertain)、
     既往史(history) 三种关键临床状态。
  2. **实体关系抽取**: 构建 Subject-Relation-Object 三元组图谱。

架构: 纯规则引擎 + 触发词库，零外部模型依赖，保证 100% 可用性。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# 临床触发词库
# ===================================================================

# 否定触发词：其后出现的实体应标记为否定状态
NEGATION_TRIGGERS: list[str] = [
    "无明显",
    "未见明显",
    "未见",
    "未发现",
    "未示",
    "未及",
    "未触及",
    "无",
    "否认",
    "排除",
    "不考虑",
    "无法确定",
    "除外",
    "排除在外",
]

# 疑似/不确定触发词
UNCERTAINTY_TRIGGERS: list[str] = [
    "疑似",
    "可疑",
    "可能",
    "考虑",
    "倾向于",
    "似有",
    "似乎",
    "不排除",
    "怀疑",
    "建议进一步",
    "待排",
    "待查",
    "待除外",
    "不能完全排除",
]

# 既往史时态触发词
HISTORY_TRIGGERS: list[str] = [
    "既往",
    "病史",
    "年前",
    "月前",
    "术后",
    "曾患",
    "曾有",
    "平素",
    "既往史",
    "曾有史",
    "自",
    "从小",
    "自幼",
    "多年前",
    "数月前",
]


# 关系触发词映射：{关系类型: [触发词列表]}
RELATION_TRIGGERS: dict[str, list[str]] = {
    "位于": ["位于", "见于", "示", "处", "可见于", "显示", "分布于"],
    "治疗": ["服用", "应用", "治疗", "控制", "给予", "口服", "使用", "用"],
    "提示": ["示", "提示", "发现", "查见", "回报"],
    "伴有": ["伴", "合并", "继发", "伴有", "并发", "伴发"],
    "导致": ["导致", "引起", "诱发", "造成"],
}

# 前三组"主关系"对应的实体类型对（用于三元组配对）
_RELATION_TYPE_PAIRS: dict[str, list[tuple[str, str]]] = {
    "位于": [
        ("disease_or_lesion", "anatomical_site"),
        ("imaging_finding", "anatomical_site"),
    ],
    "治疗": [
        ("drug", "disease_or_lesion"),
        ("drug", "symptom"),
    ],
    "提示": [
        ("examination", "disease_or_lesion"),
        ("examination", "imaging_finding"),
    ],
    "伴有": [
        ("disease_or_lesion", "disease_or_lesion"),
        ("disease_or_lesion", "symptom"),
    ],
    "导致": [
        ("disease_or_lesion", "symptom"),
        ("disease_or_lesion", "disease_or_lesion"),
    ],
}


# ===================================================================
# 工具函数
# ===================================================================


def _split_clauses(text: str) -> list[tuple[int, int, str]]:
    """按中文标点将文本切分为子句，返回 (start, end, clause)。

    切分符: 。！？，；\\n —— 每个子句为独立语义单元（用于状态检测）。

    Args:
        text: 原始文本。

    Returns:
        [(start, end, clause_text), ...] 列表。
    """
    clauses: list[tuple[int, int, str]] = []
    pattern = r"[。！？\n，；]"
    last_end = 0
    for m in re.finditer(pattern, text):
        start = m.start()
        if start > last_end:
            clause = text[last_end:start].strip()
            if clause:
                clauses.append((last_end, start, clause))
        last_end = m.end()
    if last_end < len(text):
        clause = text[last_end:].strip()
        if clause:
            clauses.append((last_end, len(text), clause))
    return clauses


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    """按句号/问号/感叹号将文本切分为整句（用于关系三元组抽取）。

    整句范围更宽，允许跨逗号/分号的实体配对，捕获如
    "患者有高血压，服用阿司匹林" 中的药物-疾病关系。

    Args:
        text: 原始文本。

    Returns:
        [(start, end, sentence_text), ...] 列表。
    """
    sentences: list[tuple[int, int, str]] = []
    pattern = r"[。！？\n]"
    last_end = 0
    for m in re.finditer(pattern, text):
        start = m.start()
        if start > last_end:
            sentence = text[last_end:start].strip()
            if sentence:
                sentences.append((last_end, start, sentence))
        last_end = m.end()
    if last_end < len(text):
        sentence = text[last_end:].strip()
        if sentence:
            sentences.append((last_end, len(text), sentence))
    return sentences


def _find_containing_clause(
    entity_start: int, entity_end: int, clauses: list[tuple[int, int, str]]
) -> Optional[tuple[int, int, str]]:
    """查找实体所在的子句。

    Args:
        entity_start: 实体在原文中的起始位置。
        entity_end: 实体在原文中的结束位置。
        clauses: _split_clauses 的输出。

    Returns:
        包含实体的子句 (start, end, text)，或 None。
    """
    for c_start, c_end, c_text in clauses:
        if c_start <= entity_start and entity_end <= c_end:
            return (c_start, c_end, c_text)
    return None


def _extract_context_window(
    text: str,
    entity_start: int,
    entity_end: int,
    clauses: list[tuple[int, int, str]],
    before_chars: int = 10,
    after_chars: int = 6,
) -> str:
    """提取实体周围的上下文窗口。

    优先使用实体所在的子句；若子句过短，则补充前后若干字符。

    Args:
        text: 原始全文。
        entity_start: 实体起始位置。
        entity_end: 实体结束位置。
        clauses: 子句列表。
        before_chars: 向前扩展字符数。
        after_chars: 向后扩展字符数。

    Returns:
        上下文窗口字符串。
    """
    # 状态检测严格限定在实体所在子句内，避免跨句污染
    clause = _find_containing_clause(entity_start, entity_end, clauses)
    if clause:
        c_start, c_end, c_text = clause
        return c_text
    else:
        # 兜底：子句查找失败时才使用字符窗口
        window_start = max(0, entity_start - before_chars)
        window_end = min(len(text), entity_end + after_chars)
        return text[window_start:window_end]


def _contains_trigger(
    context: str, triggers: list[str], entity_name: str
) -> bool:
    """检查上下文中是否包含触发词，且触发词出现在实体之前。

    对于否定和疑似，触发词应在实体之前出现（语义上前置修饰）。
    对于既往史，触发词可以在实体前后。

    Args:
        context: 上下文窗口文本。
        triggers: 触发词列表。
        entity_name: 实体名称。

    Returns:
        是否命中触发词。
    """
    # 找到实体在上下文中的位置
    ent_idx = context.find(entity_name)
    if ent_idx == -1:
        ent_idx = len(context) // 2  # 兜底：假设实体在中间

    for trigger in triggers:
        t_idx = context.find(trigger)
        if t_idx == -1:
            continue
        # 触发词必须在实体之前（对于否定/疑似类）
        # 或至少与实体同在一个语义窗口内
        if t_idx < ent_idx:
            return True
    return False


def _contains_history_trigger(context: str, entity_name: str) -> bool:
    """检查上下文中是否包含既往史触发词（前后均可）。

    Args:
        context: 上下文窗口文本。
        triggers: 触发词列表（使用全局 HISTORY_TRIGGERS）。
        entity_name: 实体名称。

    Returns:
        是否命中既往史触发词。
    """
    # 对于既往史，触发词可能在实体前后（如 "高血压病史" → "病史" 在后）
    ent_idx = context.find(entity_name)
    if ent_idx == -1:
        ent_idx = len(context) // 2

    for trigger in HISTORY_TRIGGERS:
        if trigger in context:
            # 触发词距离实体不超过一定范围（同子句内）
            t_idx = context.find(trigger)
            distance = abs(t_idx - ent_idx)
            max_dist = max(len(context) // 2, 15)
            if distance <= max_dist:
                return True
    return False


# ===================================================================
# MedicalRelationExtractor
# ===================================================================


class MedicalRelationExtractor:
    """临床关系与状态分析引擎。

    基于触发词库 + 上下文窗口的纯规则引擎，对 NER 输出的离散实体
    进行临床状态识别和关系三元组抽取。

    Attributes:
        ner_engine: 底层 MedicalNEREngine 实例。
    """

    def __init__(self, ner_engine: Optional[Any] = None) -> None:
        """初始化关系抽取引擎。

        Args:
            ner_engine: 可选的 MedicalNEREngine 实例。若为 None，
                将自动内部实例化。
        """
        if ner_engine is None:
            from src.nlp.ner_engine import MedicalNEREngine

            ner_engine = MedicalNEREngine()

        self.ner_engine: Any = ner_engine

    # ------------------------------------------------------------------
    # 核心状态识别
    # ------------------------------------------------------------------

    def detect_entity_status(
        self, text: str, entities: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """对实体列表进行临床状态识别，动态注入否定/疑似/既往史标记。

        对每个实体：
          1. 提取上下文窗口（所在子句 + 前后扩展）
          2. 扫描否定触发词（须在实体前出现）→ negation_status
          3. 扫描疑似触发词（须在实体前出现）→ uncertain_status
          4. 扫描既往史触发词（前后均可）→ history_status

        Args:
            text: 原始临床文本全文。
            entities: NER 输出的实体列表（需含 start, end, entity 字段）。

        Returns:
            注入了 negation_status / uncertain_status / history_status
            字段的实体列表（原地修改 + 返回）。
        """
        clauses = _split_clauses(text)

        for ent in entities:
            entity_name = ent.get("entity", "")
            entity_start = ent.get("start", 0)
            entity_end = ent.get("end", len(entity_name))

            # 1. 提取上下文窗口：实体前 10 字符到后 6 字符 + 所在子句
            context = _extract_context_window(
                text, entity_start, entity_end, clauses,
                before_chars=10, after_chars=6,
            )

            # 2. 扫描三种状态触发词
            ent["negation_status"] = _contains_trigger(
                context, NEGATION_TRIGGERS, entity_name
            )
            ent["uncertain_status"] = _contains_trigger(
                context, UNCERTAINTY_TRIGGERS, entity_name
            )
            ent["history_status"] = _contains_history_trigger(context, entity_name)

        return entities

    # ------------------------------------------------------------------
    # 实体关系三元组抽取
    # ------------------------------------------------------------------

    def extract_triples(
        self, text: str, entities: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """从文本和实体列表中抽取临床关系三元组。

        策略：
          1. 按子句（逗号/句号分隔）切分文本，限制三元组搜索范围。
          2. 在同一子句内，按预定义的关系类型-实体对规则进行配对。
          3. 若配对实体之间存在关系触发词，则生成三元组。

        Args:
            text: 原始临床文本全文。
            entities: 已标注状态的实体列表。

        Returns:
            三元组列表，每项为
            ``{"subject", "relation", "object", "confidence"}``。
        """
        # 三元组抽取使用整句级别分组（。分隔），允许跨逗号/分号的语义配对
        sentences = _split_sentences(text)
        triples: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()  # 去重

        # 将实体按所属整句分组
        sentence_entities: dict[tuple[int, int], list[dict[str, Any]]] = {}
        sentence_text_map: dict[tuple[int, int], str] = {}
        for ent in entities:
            s = _find_containing_clause(
                ent.get("start", 0), ent.get("end", 0), sentences
            )
            if s:
                s_key = (s[0], s[1])
                sentence_entities.setdefault(s_key, []).append(ent)
                sentence_text_map[s_key] = s[2]

        for s_key, ents in sentence_entities.items():
            if len(ents) < 2:
                continue
            sent_text = sentence_text_map.get(s_key, "")
            for i, subj in enumerate(ents):
                for obj in ents[i + 1 :]:
                    self._try_bind_triple(
                        subj, obj, sent_text, triples, seen
                    )

        return sorted(triples, key=lambda t: t["confidence"], reverse=True)

    def _try_bind_triple(
        self,
        ent_a: dict[str, Any],
        ent_b: dict[str, Any],
        clause_text: str,
        triples: list[dict[str, Any]],
        seen: set[tuple[str, str, str]],
    ) -> None:
        """尝试在两个实体间建立关系三元组。

        检查所有预定义的关系类型-实体对规则，若配对命中则在子句内
        搜索关系触发词。双向尝试（A→B 和 B→A）。

        Args:
            ent_a: 实体 A。
            ent_b: 实体 B。
            clause_text: 所在子句文本。
            triples: 累积的三元组列表。
            seen: 去重集合。
        """
        type_a = ent_a.get("entity_type", "")
        type_b = ent_b.get("entity_type", "")
        name_a = ent_a.get("entity", "")
        name_b = ent_b.get("entity", "")

        for relation, type_pairs in _RELATION_TYPE_PAIRS.items():
            for pair_subj_type, pair_obj_type in type_pairs:
                # 正向匹配: A = subject_type, B = object_type
                if type_a == pair_subj_type and type_b == pair_obj_type:
                    self._emit_triple(
                        ent_a, ent_b, relation, clause_text, triples, seen
                    )
                # 反向匹配: B = subject_type, A = object_type
                elif type_b == pair_subj_type and type_a == pair_obj_type:
                    self._emit_triple(
                        ent_b, ent_a, relation, clause_text, triples, seen
                    )
                # 对于 "伴有" 和 "导致"，同类型配对也可尝试
                elif (
                    relation in ("伴有", "导致")
                    and type_a == pair_subj_type
                    and type_b == pair_obj_type
                ):
                    self._emit_triple(
                        ent_a, ent_b, relation, clause_text, triples, seen
                    )

    def _emit_triple(
        self,
        subject_ent: dict[str, Any],
        object_ent: dict[str, Any],
        relation: str,
        clause_text: str,
        triples: list[dict[str, Any]],
        seen: set[tuple[str, str, str]],
    ) -> None:
        """验证并生成一个三元组。

        在子句中检查是否存在关系触发词连接两个实体。
        若触发词存在，则生成三元组；若子句较短（≤10字），
        认为实体天然具有关联，confidence 略低但仍生成。

        Args:
            subject_ent: 主体实体。
            object_ent: 客体实体。
            relation: 关系名称。
            clause_text: 子句文本。
            triples: 累积列表。
            seen: 去重集合。
        """
        subj_name = subject_ent.get("entity", "")
        obj_name = object_ent.get("entity", "")
        subj_std = subject_ent.get("standard_name", subj_name)
        obj_std = object_ent.get("standard_name", obj_name)

        # 去重
        key = (subj_name, relation, obj_name)
        if key in seen:
            return

        # ---- 否定实体过滤 ----
        # "位于" 关系对被否定实体无临床意义（不能说"未见转移位于右肺"）
        # "提示" 关系保留（CT 可以提示"未见异常"）
        subj_negated = subject_ent.get("negation_status", False)
        obj_negated = object_ent.get("negation_status", False)
        if relation == "位于" and (subj_negated or obj_negated):
            return
        # 双方均为否定时，其他关系也不生成（避免 "阿司匹林-治疗-否定肺癌"）
        if subj_negated and obj_negated:
            return

        triggers = RELATION_TRIGGERS.get(relation, [])
        trigger_found = any(t in clause_text for t in triggers)
        # 短子句内邻近实体默认关联
        is_short_clause = len(clause_text.replace(" ", "")) <= 15

        if trigger_found:
            confidence = 0.88
        elif is_short_clause:
            confidence = 0.72
        else:
            return  # 既无触发词又不在短子句内，不生成

        seen.add(key)
        triples.append(
            {
                "subject": subj_std,
                "subject_entity": subj_name,
                "relation": relation,
                "object": obj_std,
                "object_entity": obj_name,
                "confidence": confidence,
            }
        )

    # ------------------------------------------------------------------
    # 全链路分析 API
    # ------------------------------------------------------------------

    def analyze_clinical_text(
        self,
        text: str,
        text_id: str = "text_001",
        patient_id: str = "p001",
    ) -> dict[str, Any]:
        """全链路临床文本智能分析。

        执行流程:
          1. NER 实体抽取 (ner_engine.extract_entities)
          2. 临床状态识别 (detect_entity_status)
          3. 实体关系抽取 (extract_triples)
          4. 组装标准化输出

        Args:
            text: 原始临床文本。
            text_id: 文本唯一标识。
            patient_id: 患者唯一标识。

        Returns:
            标准化分析结果字典:
            {
                "text_id": str,
                "patient_id": str,
                "entities": [...],   # 含 negation_status 等状态标记
                "triples": [...]     # 关系三元组
            }
        """
        # Step 1: NER 实体抽取
        ner_result = self.ner_engine.extract_entities(
            text=text, text_id=text_id, patient_id=patient_id
        )
        entities = ner_result.get("entities", [])

        # Step 2: 临床状态识别
        entities = self.detect_entity_status(text, entities)

        # Step 3: 关系三元组抽取
        triples = self.extract_triples(text, entities)

        # Step 4: 组装输出
        return {
            "text_id": text_id,
            "patient_id": patient_id,
            "entities": entities,
            "triples": triples,
        }

    # ------------------------------------------------------------------
    # 统计信息
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """返回引擎统计摘要。"""
        return {
            "ner_engine": type(self.ner_engine).__name__,
            "negation_triggers": len(NEGATION_TRIGGERS),
            "uncertainty_triggers": len(UNCERTAINTY_TRIGGERS),
            "history_triggers": len(HISTORY_TRIGGERS),
            "relation_types": len(RELATION_TRIGGERS),
        }


# ===================================================================
# 验证与自测模块
# ===================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  临床关系与状态分析引擎 (MedicalRelationExtractor)")
    print("  全链路功能验证")
    print("=" * 72)

    # ---- 初始化 ----
    extractor = MedicalRelationExtractor()
    stats = extractor.get_stats()
    print(f"\n[引擎初始化]")
    print(f"  NER 引擎:  {stats['ner_engine']}")
    print(f"  否定触发词: {stats['negation_triggers']} 个")
    print(f"  疑似触发词: {stats['uncertainty_triggers']} 个")
    print(f"  既往触发词: {stats['history_triggers']} 个")
    print(f"  关系类别:   {stats['relation_types']} 种")

    # ---- 经典测试文本 ----
    test_text = (
        "患者既往有高血压病史5年，规律服用阿司匹林。"
        "昨日复查CT示：右肺上叶可见一大小约12mm的肺结节，"
        "未见明显远处转移，不考虑肺癌，建议随访。"
    )

    print(f"\n[测试文本]")
    print(f"  {test_text}")

    # ---- 全链路分析 ----
    result = extractor.analyze_clinical_text(
        text=test_text, text_id="test_001", patient_id="p000_demo"
    )

    entities = result["entities"]
    triples = result["triples"]

    # ================================================================
    # 输出第一部分：特殊状态实体
    # ================================================================
    print(f"\n{'─' * 72}")
    print(f"  【第一部分】特殊临床状态标注")
    print(f"{'─' * 72}")

    special_entities = [
        e for e in entities
        if e.get("negation_status") or e.get("history_status") or e.get("uncertain_status")
    ]

    if special_entities:
        print(
            f"\n  {'实体':<14} {'类型':<22} {'否定':<6} {'疑似':<6} {'既往':<6} {'说明'}"
        )
        print(f"  {'─' * 68}")

        for ent in special_entities:
            name = ent["entity"]
            etype = ent.get("entity_type", "?")
            neg = "  ✓  " if ent.get("negation_status") else "  —  "
            unc = "  ✓  " if ent.get("uncertain_status") else "  —  "
            hist = "  ✓  " if ent.get("history_status") else "  —  "

            # 构造说明
            notes = []
            if ent.get("negation_status"):
                notes.append(f"被否定: 上下文中包含否定触发词")
            if ent.get("history_status"):
                notes.append(f"既往史: 上下文中包含既往时态触发词")
            if ent.get("uncertain_status"):
                notes.append(f"疑似: 上下文中包含不确定性触发词")
            note = "; ".join(notes)

            print(f"  {name:<14} {etype:<22} {neg:<6} {unc:<6} {hist:<6} {note}")

        print(f"\n  ▶ 共 {len(special_entities)} 个实体被标记为特殊临床状态")
    else:
        print(f"\n  (无特殊状态实体)")

    # ================================================================
    # 输出第二部分：关系三元组
    # ================================================================
    print(f"\n{'─' * 72}")
    print(f"  【第二部分】临床关系三元组")
    print(f"{'─' * 72}")

    if triples:
        print(
            f"\n  {'主体 (Subject)':<22} {'关系':<8} {'客体 (Object)':<22} {'置信度'}"
        )
        print(f"  {'─' * 62}")

        for t in triples:
            subj = t.get("subject_entity", t["subject"])
            obj = t.get("object_entity", t["object"])
            rel = t["relation"]
            conf = f"{t['confidence']:.2f}"

            # 构建可视化描述
            arrow = f"{subj}  —[{rel}]→  {obj}"
            print(f"  {subj:<22} {rel:<8} {obj:<22} {conf}")

        print(f"\n  ▶ 共抽取 {len(triples)} 个临床关系三元组")

        # 图状展示
        print(f"\n  [关系图谱概览]")
        for t in triples:
            subj = t.get("subject_entity", t["subject"])
            obj = t.get("object_entity", t["object"])
            rel = t["relation"]
            print(f"    {subj}  ──[{rel}]──▶  {obj}")
    else:
        print(f"\n  (未抽取到关系三元组)")

    # ================================================================
    # 统计摘要
    # ================================================================
    print(f"\n{'─' * 72}")
    print(f"  【统计摘要】")
    print(f"{'─' * 72}")
    total = len(entities)
    neg_count = sum(1 for e in entities if e.get("negation_status"))
    hist_count = sum(1 for e in entities if e.get("history_status"))
    unc_count = sum(1 for e in entities if e.get("uncertain_status"))
    print(f"  总实体数:     {total}")
    print(f"  否定状态:     {neg_count} 个")
    print(f"  既往史状态:   {hist_count} 个")
    print(f"  疑似状态:     {unc_count} 个")
    print(f"  关系三元组:   {len(triples)} 个")

    # ---- 完整 JSON 输出 ----
    print(f"\n{'─' * 72}")
    print(f"  【完整结构化 JSON 输出】")
    print(f"{'─' * 72}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 72)
    print("  全链路验证完毕！")
    print("=" * 72)
