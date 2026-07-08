#!/usr/bin/env python3
"""智能字段映射引擎 (Smart Field Mapper).

本模块提供高鲁棒性的临床字段名到标准字典主键的智能映射能力。
通过四级递进匹配策略（精确命中 → 高置信度模糊 → 人工复核 → 映射失败），
应对真实医院场景下不规范字段名（错别字、多余符号、括号、冗余描述等）。

依赖:
    - dictionary_manager: 标准字典的倒排索引
    - thefuzz (可选): 高性能模糊匹配；未安装时自动降级为 difflib
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Optional

# 确保项目根目录在 sys.path 中，支持直接脚本运行与模块导入两种场景
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 模糊匹配引擎：优先 thefuzz，静默降级 difflib
# ---------------------------------------------------------------------------

_FUZZ_ENGINE: str = "difflib"  # 实际使用的引擎名称
_fuzz_available: bool = False

try:
    from thefuzz import fuzz, process  # type: ignore[import-untyped]

    _fuzz_available = True
    _FUZZ_ENGINE = "thefuzz"
except ImportError:
    import difflib

    _fuzz_available = False
    _FUZZ_ENGINE = "difflib"


# ---------------------------------------------------------------------------
# 模糊匹配工具函数
# ---------------------------------------------------------------------------


def _compute_similarity(a: str, b: str) -> float:
    """计算两个字符串的相似度 (0.0 ~ 1.0)。

    优先使用 thefuzz.token_sort_ratio（对词序不敏感），
    未安装时降级为 difflib.SequenceMatcher。

    Args:
        a: 待比较字符串。
        b: 待比较字符串。

    Returns:
        归一化相似度分数。
    """
    if _fuzz_available:
        return fuzz.token_sort_ratio(a, b) / 100.0  # type: ignore[no-any-return]
    else:
        return difflib.SequenceMatcher(None, a, b).ratio()


def _best_match(query: str, candidates: list[str]) -> tuple[Optional[str], float]:
    """在候选列表中找出与 query 最匹配的字符串及相似度。

    Args:
        query: 查询字符串。
        candidates: 候选字符串列表。

    Returns:
        (best_candidate, score) 元组；若候选列表为空则返回 (None, 0.0)。
    """
    if not candidates:
        return None, 0.0

    best: Optional[str] = None
    best_score: float = 0.0

    if _fuzz_available:
        result = process.extractOne(query, candidates, scorer=fuzz.token_sort_ratio)
        if result is not None:
            best, best_score = result[0], result[1] / 100.0  # type: ignore[index]
    else:
        for candidate in candidates:
            score = difflib.SequenceMatcher(None, query, candidate).ratio()
            if score > best_score:
                best_score = score
                best = candidate

    return best, best_score


# ---------------------------------------------------------------------------
# SmartFieldMapper
# ---------------------------------------------------------------------------


class SmartFieldMapper:
    """智能字段映射引擎。

    负责将任意原始字段名（可能含错别字、符号、括号等）映射到标准医学字典主键。
    采用四级递进策略，从 O(1) 精确查找到模糊匹配逐级降级，

    Attributes:
        _dict_manager: 底层 DictionaryManager 实例。
        _alias_index: 倒排索引字典（别名 → standard_name）。
        _all_aliases: 所有别名的扁平列表（供模糊匹配遍历）。
    """

    def __init__(self, dict_manager: Optional[Any] = None) -> None:
        """初始化映射引擎。

        Args:
            dict_manager: 可选的外部 DictionaryManager 实例。若为 None，
                将自动实例化一个默认的 DictionaryManager。
        """
        if dict_manager is None:
            # 局部导入避免循环依赖，同时确保项目根目录可解析
            from src.ingestion.dictionary_manager import DictionaryManager

            dict_manager = DictionaryManager()

        self._dict_manager: Any = dict_manager
        self._alias_index: dict[str, str] = self._dict_manager.get_all_aliases()
        # 预提取所有别名键的列表，供模糊匹配阶段遍历
        self._all_aliases: list[str] = list(self._alias_index.keys())

    # ------------------------------------------------------------------
    # 文本预处理
    # ------------------------------------------------------------------

    def _preprocess_text(self, text: str) -> str:
        """对原始输入文本进行标准化清洗。

        清洗步骤：
          1. 去除前后空白。
          2. 将多个连续空白字符合并为单个空格。
          3. 统一转为小写。
          4. 移除方括号/尖括号/中文括号及其内容（如【血常规】、(空腹)）。
          5. 移除常见干扰符号（如多余的英文句点当非缩写时）。

        Args:
            text: 原始输入字符串。

        Returns:
            清洗后的干净字符串。
        """
        if not text:
            return ""

        # 1. 去除前后空白
        cleaned = text.strip()

        # 2. 合并连续空白
        cleaned = re.sub(r"\s+", " ", cleaned)

        # 3. 移除括号及其内容（中英文括号、方括号、中文方括号）
        #    匹配: (...)、[...]、{...}、【...】、《...》
        cleaned = re.sub(r"\([^)]*\)", "", cleaned)
        cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
        cleaned = re.sub(r"\{[^}]*\}", "", cleaned)
        cleaned = re.sub(r"【[^】]*】", "", cleaned)
        cleaned = re.sub(r"《[^》]*》", "", cleaned)

        # 4. 移除常见干扰标点符号（但保留字母数字之间的必要连接符）
        #    将中英文逗号、分号、冒号、引号等替换为空格
        cleaned = re.sub(r"[,;:，。；：""'']+", " ", cleaned)

        # 5. 移除字母/数字之间的点号分隔符（如 W.B.C → WBC）
        cleaned = re.sub(r"(?<=[a-zA-Z0-9])\.(?=[a-zA-Z0-9])", "", cleaned)
        # 6. 移除中文之间的点号（如 白.细.胞 → 白细胞）
        cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\.(?=[\u4e00-\u9fff])", "", cleaned)

        # 7. 再次合并多余空格
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # 8. 统一小写
        cleaned = cleaned.lower()

        return cleaned

    def _normalize_aggressive(self, text: str) -> str:
        """激进标准化：移除所有非字母数字和中文字符后尝试精确匹配。

        用于在第一级精确匹配失败后的兜底策略，处理极端不规范输入。

        Args:
            text: 已经过 _preprocess_text 清洗的文本。

        Returns:
            仅保留 [a-z0-9] 和中文字符的紧凑字符串。
        """
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", text)

    # ------------------------------------------------------------------
    # 核心单字段映射
    # ------------------------------------------------------------------

    def map_field(self, raw_field: str) -> dict[str, Any]:
        """对单个原始字段名执行四级递进智能映射。

        四级映射逻辑：
          1. **精确命中** (O(1)): 清洗后文本直接在倒排索引中命中 → confidence=1.0。
          1.5 **激进精确命中**: 去除所有标点空格后再次查索引。
          2. **高置信度模糊**: 相似度 ≥ 0.85 → status="high_confidence"。
          3. **人工复核区**: 相似度 0.60 ~ 0.84 → status="needs_manual_review"。
          4. **映射失败**: 相似度 < 0.60 → status="unmapped"。

        Args:
            raw_field: 原始字段名字符串。

        Returns:
            映射结果字典，包含以下键：
              - raw_field (str): 原始输入
              - standard_name (str | None): 匹配到的标准主键
              - matched_alias (str | None): 最佳匹配到的别名
              - confidence (float): 置信度 0.0 ~ 1.0
              - status (str): 匹配状态标识
        """
        # 预处理
        cleaned = self._preprocess_text(raw_field)

        # ---- 第一级：O(1) 精确命中 ----
        if cleaned in self._alias_index:
            std_name = self._alias_index[cleaned]
            return {
                "raw_field": raw_field,
                "standard_name": std_name,
                "matched_alias": cleaned,
                "confidence": 1.0,
                "status": "exact_match",
            }

        # ---- 第一级半：激进标准化后再次精确匹配 ----
        aggressive = self._normalize_aggressive(cleaned)
        if aggressive and aggressive in self._alias_index:
            std_name = self._alias_index[aggressive]
            return {
                "raw_field": raw_field,
                "standard_name": std_name,
                "matched_alias": aggressive,
                "confidence": 1.0,
                "status": "exact_match",
            }

        # ---- 第一级又半：按空格分词后分别精确匹配 ----
        # 处理 "空腹 葡萄糖" → 拆分为 ["空腹", "葡萄糖"] → "葡萄糖" 命中 GLU
        tokens = cleaned.split()
        for token in tokens:
            if token in self._alias_index:
                std_name = self._alias_index[token]
                return {
                    "raw_field": raw_field,
                    "standard_name": std_name,
                    "matched_alias": token,
                    "confidence": 1.0,
                    "status": "exact_match",
                }
            # 同时对每个 token 做激进标准化后再查
            agg_token = self._normalize_aggressive(token)
            if agg_token and agg_token in self._alias_index:
                std_name = self._alias_index[agg_token]
                return {
                    "raw_field": raw_field,
                    "standard_name": std_name,
                    "matched_alias": agg_token,
                    "confidence": 1.0,
                    "status": "exact_match",
                }

        # 如果清洗后为空字符串，直接判定为失败
        if not cleaned:
            return {
                "raw_field": raw_field,
                "standard_name": None,
                "matched_alias": None,
                "confidence": 0.0,
                "status": "unmapped",
            }

        # ---- 第二~四级：模糊匹配 ----
        best_alias, best_score = _best_match(cleaned, self._all_aliases)

        standard_name: Optional[str] = None
        matched_alias: Optional[str] = best_alias

        if best_score >= 0.85:
            # 第二级：高置信度
            status = "high_confidence"
            if best_alias:
                standard_name = self._alias_index.get(best_alias)
        elif best_score >= 0.60:
            # 第三级：需要人工复核
            status = "needs_manual_review"
            if best_alias:
                standard_name = self._alias_index.get(best_alias)
        else:
            # 第四级：映射失败
            status = "unmapped"
            standard_name = None
            matched_alias = None

        return {
            "raw_field": raw_field,
            "standard_name": standard_name,
            "matched_alias": matched_alias,
            "confidence": round(best_score, 4),
            "status": status,
        }

    # ------------------------------------------------------------------
    # 批量映射
    # ------------------------------------------------------------------

    def map_fields_batch(self, raw_fields: list[str]) -> list[dict[str, Any]]:
        """对一批原始字段名执行批量映射。

        典型场景：从 Excel 读取表头行后，一次性映射所有列名到标准字典。

        Args:
            raw_fields: 原始字段名列表。

        Returns:
            映射结果字典列表，每个元素与 ``map_field`` 返回格式一致。
        """
        return [self.map_field(f) for f in raw_fields]

    # ------------------------------------------------------------------
    # 统计信息
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """返回映射引擎的统计信息摘要。

        Returns:
            包含 fuzzy_engine、total_aliases、total_standard_fields 的字典。
        """
        return {
            "fuzzy_engine": _FUZZ_ENGINE,
            "total_aliases": len(self._all_aliases),
            "total_standard_fields": self._dict_manager.get_total_count(),
        }


# ===================================================================
# 测试验证模块
# ===================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  智能字段映射引擎 (SmartFieldMapper) — 功能验证")
    print("=" * 72)

    # ---- 0. 引擎初始化 ----
    mapper = SmartFieldMapper()
    stats = mapper.get_stats()
    print(f"\n[初始化] 模糊引擎: {stats['fuzzy_engine']}")
    print(f"         别名索引条目: {stats['total_aliases']}")
    print(f"         标准字段数:   {stats['total_standard_fields']}")

    # ---- 测试用例定义 ----
    test_cases: list[dict[str, Any]] = [
        {
            "raw": "W.B.C 计数",
            "desc": "带符号与空格的血常规缩写",
            "expect_hint": "WBC",
        },
        {
            "raw": "血清肌肝",
            "desc": "故意错别字「肝」替代「酐」",
            "expect_hint": "Creatinine",
        },
        {
            "raw": "空腹 葡萄糖(GLU)",
            "desc": "带括号与混合英文",
            "expect_hint": "GLU",
        },
        {
            "raw": "糖化血红蛋白 A1c",
            "desc": "全称 + 英文后缀",
            "expect_hint": "HbA1c",
        },
        {
            "raw": "未知外星人指标X",
            "desc": "完全无效的输入",
            "expect_hint": "unmapped",
        },
    ]

    # ---- 执行测试 ----
    print("\n" + "-" * 72)
    print(f"  {'#':<3} {'原始输入':<28} {'→ 标准名':<14} {'置信度':<8} {'状态':<20}")
    print("-" * 72)

    for i, case in enumerate(test_cases, 1):
        raw = case["raw"]
        result = mapper.map_field(raw)

        std = result["standard_name"] or "—"
        conf = f"{result['confidence']:.2f}"
        status = result["status"]

        # 状态着色标记（纯文本兼容）
        if status == "exact_match":
            flag = "[精确]"
        elif status == "high_confidence":
            flag = "[高置信]"
        elif status == "needs_manual_review":
            flag = "[待复核]"
        else:
            flag = "[失败]"

        print(f"  {i:<3} {raw:<28} → {std:<14} {conf:<8} {flag} {status}")

    print("-" * 72)

    # ---- 批量映射演示 ----
    print("\n[批量映射测试]")
    raw_headers = ["W.B.C 计数", "血清肌肝", "空腹 葡萄糖(GLU)", "糖化血红蛋白 A1c", "未知外星人指标X"]
    batch_results = mapper.map_fields_batch(raw_headers)
    for r in batch_results:
        status_symbol = "+" if r["standard_name"] else "✗"
        print(f"  {status_symbol} '{r['raw_field']}' → {r['standard_name'] or '未映射'} ({r['status']})")

    # ---- 统计摘要 ----
    mapped = sum(1 for r in batch_results if r["standard_name"] is not None)
    print(f"\n  批量结果: {mapped}/{len(batch_results)} 映射成功")

    print("\n" + "=" * 72)
    print("  所有测试用例验证完毕！")
    print("=" * 72)
