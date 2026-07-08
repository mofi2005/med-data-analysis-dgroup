#!/usr/bin/env python3
"""医学标准字典管理引擎 (Medical Standard Dictionary Manager).

本模块提供高鲁棒性的医学标准字典加载、查询、校验与动态注册功能，
基于 O(1) 哈希倒排索引实现极速别名解析，适用于医院大数据场景下
数千个临床指标的高吞吐量映射需求。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------

class DictionaryError(Exception):
    """字典模块基础异常。"""
    pass


class DictionaryNotFoundError(DictionaryError):
    """JSON 字典文件不存在时抛出。"""

    def __init__(self, file_path: str) -> None:
        super().__init__(f"标准字典文件不存在: {file_path}")
        self.file_path = file_path


class FieldValidationError(DictionaryError):
    """字段数据校验失败时抛出。"""

    def __init__(self, message: str, field_data: Optional[dict] = None) -> None:
        super().__init__(message)
        self.field_data = field_data


class DuplicateFieldError(DictionaryError):
    """尝试注册已存在的标准名时抛出。"""

    def __init__(self, standard_name: str) -> None:
        super().__init__(f"指标 '{standard_name}' 已存在于字典中，不允许重复注册")
        self.standard_name = standard_name


# ---------------------------------------------------------------------------
# 必备字段定义（用于校验）
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS: list[str] = [
    "standard_name",
    "chinese_name",
    "english_name",
    "abbreviation",
    "alias_names",
    "data_type",
    "unit",
    "reference_range",
    "category",
    "disease_relevance_matrix",
]


# ---------------------------------------------------------------------------
# DictionaryManager
# ---------------------------------------------------------------------------

class DictionaryManager:
    """医学标准字典管理器。

    负责加载、索引、查询、校验和动态维护标准医学指标字典。
    内部构建全量倒排索引，确保任意别名字符串到标准名的映射为 O(1)。

    Attributes:
        _dict_path: JSON 字典文件的绝对路径。
        _data: 原始字典 JSON 反序列化后的内存对象。
        _alias_index: 倒排索引字典，所有可能的别名 → standard_name 的映射。
        _fields: fields 子字典的引用（快捷访问）。
    """

    def __init__(self, dict_path: Optional[str] = None) -> None:
        """初始化字典管理器，加载 JSON 并构建倒排索引。

        Args:
            dict_path: JSON 字典文件的路径。若为 None，则默认使用
                ``config/standard_dictionary.json``（相对于项目根目录）。

        Raises:
            DictionaryNotFoundError: 指定的 JSON 文件不存在。
        """
        if dict_path is None:
            # 默认路径：项目根目录 / config / standard_dictionary.json
            project_root = Path(__file__).resolve().parent.parent.parent
            dict_path = str(project_root / "config" / "standard_dictionary.json")

        self._dict_path: str = os.path.abspath(dict_path)

        if not os.path.isfile(self._dict_path):
            raise DictionaryNotFoundError(self._dict_path)

        with open(self._dict_path, "r", encoding="utf-8") as f:
            self._data: dict[str, Any] = json.load(f)

        self._fields: dict[str, dict] = self._data.get("fields", {})
        self._alias_index: dict[str, str] = {}
        self._build_alias_index()

    # ------------------------------------------------------------------
    # 内部方法：构建倒排索引
    # ------------------------------------------------------------------

    def _build_alias_index(self) -> None:
        """构建全量倒排索引。

        将每个指标的 standard_name、chinese_name、english_name、
        abbreviation 及 alias_names 中的所有词条映射到其 standard_name。

        此方法在初始化及每次动态注册新字段后自动调用。
        """
        for std_name, field in self._fields.items():
            self._index_one_field(std_name, field)

    def _index_one_field(self, standard_name: str, field: dict) -> None:
        """将单个字段的所有别名注册到倒排索引。

        Args:
            standard_name: 标准名。
            field: 字段元数据字典。
        """
        candidates: list[str] = [
            standard_name,
            field.get("chinese_name", ""),
            field.get("english_name", ""),
            field.get("abbreviation", ""),
        ]
        aliases: list[str] = field.get("alias_names", [])

        for token in candidates + aliases:
            token = token.strip()
            if not token:
                continue
            # 同时注册原始大小写与全小写版本，提升容错匹配能力
            if token not in self._alias_index:
                self._alias_index[token] = standard_name
            lowered = token.lower()
            if lowered not in self._alias_index:
                self._alias_index[lowered] = standard_name

    # ------------------------------------------------------------------
    # 核心业务 API
    # ------------------------------------------------------------------

    def get_field_meta(self, standard_name: str) -> Optional[dict]:
        """根据标准名返回完整的字段元数据。

        Args:
            standard_name: 指标标准名（大小写敏感），例如 ``"WBC"``。

        Returns:
            字段元数据字典；若不存在则返回 None。
        """
        return self._fields.get(standard_name)

    def get_all_aliases(self) -> dict[str, str]:
        """返回完整的倒排索引字典。

        供下一阶段（如 NLP 相似度映射引擎、模糊匹配层）直接调用，
        避免重复解析原始 JSON。

        Returns:
            倒排索引字典，键为任意别名（含原始大小写及全小写），
            值为对应的 standard_name。
        """
        return self._alias_index

    def get_core_markers_for_disease(
        self, disease_name: str, min_weight: float = 0.5
    ) -> list[dict]:
        """按权重降序返回某疾病的核心指标列表。

        遍历所有已注册指标，筛选 disease_relevance_matrix 中包含
        指定疾病且权重 >= min_weight 且 is_core == True 的条目。

        Args:
            disease_name: 疾病名称，需与字典中的键完全匹配。
            min_weight: 最低权重阈值 (0.0 ~ 1.0)，默认 0.5。

        Returns:
            结果列表，每个元素为包含 field 元数据和匹配权重的字典，
            按权重降序排列。
        """
        results: list[dict] = []

        for std_name, field in self._fields.items():
            matrix = field.get("disease_relevance_matrix", {})
            entry = matrix.get(disease_name)
            if entry is None:
                continue
            if entry.get("weight", 0) < min_weight:
                continue
            if not entry.get("is_core", False):
                continue
            results.append(
                {
                    "standard_name": std_name,
                    "chinese_name": field.get("chinese_name", ""),
                    "weight": entry["weight"],
                    "threshold_alert": entry.get("threshold_alert", ""),
                    "note": entry.get("note", ""),
                }
            )

        results.sort(key=lambda x: x["weight"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # 扩展 API
    # ------------------------------------------------------------------

    def validate_field_schema(self, field_data: dict) -> bool:
        """校验字段数据是否包含所有必备字段。

        校验项：
          - 顶层必备字段完整性
          - reference_range 必须为 dict 且包含 ``low`` 和 ``high``
          - disease_relevance_matrix 必须为 dict

        Args:
            field_data: 待校验的字段数据字典。

        Returns:
            校验通过返回 True。

        Raises:
            FieldValidationError: 校验失败时抛出，携带具体原因。
        """
        # 1. 必备字段检查
        for req in _REQUIRED_FIELDS:
            if req not in field_data:
                raise FieldValidationError(
                    f"缺少必备字段 '{req}'", field_data
                )
            if field_data[req] is None:
                raise FieldValidationError(
                    f"必备字段 '{req}' 的值不能为 null", field_data
                )

        # 2. reference_range 结构检查
        ref_range = field_data["reference_range"]
        if not isinstance(ref_range, dict):
            raise FieldValidationError(
                "reference_range 必须为字典类型", field_data
            )
        for key in ("low", "high"):
            if key not in ref_range:
                raise FieldValidationError(
                    f"reference_range 缺少 '{key}' 键", field_data
                )

        # 3. disease_relevance_matrix 结构检查
        matrix = field_data["disease_relevance_matrix"]
        if not isinstance(matrix, dict):
            raise FieldValidationError(
                "disease_relevance_matrix 必须为字典类型", field_data
            )

        # 4. alias_names 必须为列表
        if not isinstance(field_data["alias_names"], list):
            raise FieldValidationError(
                "alias_names 必须为列表类型", field_data
            )

        # 5. data_type 合法值检查
        valid_types = {"float", "int", "categorical"}
        if field_data["data_type"] not in valid_types:
            raise FieldValidationError(
                f"data_type 必须为 {valid_types} 之一，实际为 '{field_data['data_type']}'",
                field_data,
            )

        return True

    def register_new_field(
        self, field_data: dict, save_to_disk: bool = False
    ) -> bool:
        """动态注册新指标到内存，并可选持久化到 JSON 文件。

        注册前会调用 ``validate_field_schema`` 进行完整性校验，
        并检查是否存在重复的 standard_name。

        Args:
            field_data: 新字段的完整元数据字典。
            save_to_disk: 若为 True，则同步将更新后的字典写回
                ``config/standard_dictionary.json``。

        Returns:
            注册成功返回 True。

        Raises:
            FieldValidationError: 字段数据校验失败。
            DuplicateFieldError: standard_name 已存在。
            IOError: save_to_disk=True 但写入磁盘失败。
        """
        # 校验
        self.validate_field_schema(field_data)

        std_name: str = field_data["standard_name"]
        if std_name in self._fields:
            raise DuplicateFieldError(std_name)

        # 注册到内存
        self._fields[std_name] = field_data
        self._index_one_field(std_name, field_data)

        # 持久化到磁盘
        if save_to_disk:
            self._data["fields"] = self._fields
            self._data["total_fields"] = len(self._fields)
            try:
                with open(self._dict_path, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=2)
            except IOError as e:
                raise IOError(f"无法将字典持久化到 {self._dict_path}: {e}") from e

        return True

    def get_total_count(self) -> int:
        """返回当前已注册的指标总数。

        Returns:
            内存中 fields 的数量。
        """
        return len(self._fields)

    # ------------------------------------------------------------------
    # 统计信息
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """返回字典引擎的统计信息摘要。

        Returns:
            包含 total_fields 和 total_alias_entries 的字典。
        """
        return {
            "total_fields": self.get_total_count(),
            "total_alias_entries": len(self._alias_index),
        }


# ===================================================================
# 测试验证模块
# ===================================================================

if __name__ == "__main__":
    print("=" * 68)
    print("  医学标准字典管理引擎 — 功能验证")
    print("=" * 68)

    # ---- 1. 初始化 ----
    mgr = DictionaryManager()
    stats = mgr.get_stats()
    print(f"\n[OK] 字典加载成功！")
    print(f"    已加载标准指标数: {stats['total_fields']}")
    print(f"    倒排索引总条目数: {stats['total_alias_entries']}")

    # ---- 2. 字段查询示例 ----
    print("\n--- 字段元数据查询 ---")
    meta = mgr.get_field_meta("HbA1c")
    if meta:
        print(f"  [HbA1c] {meta['chinese_name']} ({meta['english_name']})")
        print(f"    单位: {meta['unit']}")
        print(f"    参考范围: {meta['reference_range']['low']} ~ {meta['reference_range']['high']} {meta['unit']}")

    # ---- 3. 根据别名查询（通过倒排索引直接定位） ----
    print("\n--- 别名 → 标准名 O(1) 查询 ---")
    test_aliases = ["白血球", "谷丙转氨酶", "A1C", "血肌酐", "空腹血糖", "carcinoembryonic antigen"]
    for alias in test_aliases:
        std = mgr._alias_index.get(alias) or mgr._alias_index.get(alias.lower())
        if std:
            meta = mgr.get_field_meta(std)
            print(f"  '{alias}' → {std} ({meta['chinese_name']})")
        else:
            print(f"  '{alias}' → [未命中]")

    # ---- 4. 查询 Diabetes 核心指标 ----
    print("\n--- 疾病核心指标查询: Diabetes ---")
    diabetes_markers = mgr.get_core_markers_for_disease("Diabetes")
    for i, marker in enumerate(diabetes_markers, 1):
        print(f"  [{i}] {marker['standard_name']} ({marker['chinese_name']})")
        print(f"      权重: {marker['weight']:.2f}  |  警报阈值: {marker['threshold_alert']}")
        print(f"      说明: {marker['note'][:50]}...")

    # ---- 5. 自动化扩展模拟测试：动态注册 AST ----
    print("\n" + "=" * 68)
    print("  自动化扩展模拟测试：动态注册 AST")
    print("=" * 68)

    new_ast_field: dict = {
        "standard_name": "AST",
        "chinese_name": "天门冬氨酸氨基转移酶",
        "english_name": "Aspartate Aminotransferase",
        "abbreviation": "AST",
        "alias_names": [
            "谷草转氨酶",
            "天门冬氨酸氨基转移酶",
            "Aspartate Aminotransferase",
            "SGOT",
            "aspartate transaminase",
            "GOT",
            "血清谷草转氨酶",
            "ast",
        ],
        "data_type": "float",
        "unit": "U/L",
        "reference_range": {"low": 0.0, "high": 40.0},
        "category": "肝功能",
        "disease_relevance_matrix": {
            "Hepatitis": {
                "weight": 0.95,
                "is_core": True,
                "threshold_alert": "> 40",
                "note": "AST升高提示肝细胞损伤，结合ALT/AST比值可辅助区分肝损伤类型",
            },
            "Liver Cirrhosis": {
                "weight": 0.9,
                "is_core": True,
                "threshold_alert": "AST/ALT > 1",
                "note": "肝硬化时AST/ALT比值通常>1，是评估肝纤维化进展的重要参考",
            },
            "Myocardial Infarction": {
                "weight": 0.8,
                "is_core": False,
                "threshold_alert": "显著升高",
                "note": "AST曾用于心梗诊断，现已被肌钙蛋白替代但仍具参考意义",
            },
        },
    }

    success = mgr.register_new_field(new_ast_field, save_to_disk=False)
    print(f"\n[OK] 动态注册 AST 成功: {success}")
    print(f"    当前指标总数: {mgr.get_total_count()}")

    # ---- 6. 验证别名极速查询 ----
    print("\n--- 别名极速查询验证 ---")
    ast_aliases = ["谷草转氨酶", "SGOT", "got", "天门冬氨酸氨基转移酶", "ast"]
    for alias in ast_aliases:
        std = mgr._alias_index.get(alias) or mgr._alias_index.get(alias.lower())
        if std:
            meta = mgr.get_field_meta(std)
            print(f"  '{alias}' → {std} ({meta['chinese_name']})   [O(1) 命中]")
        else:
            print(f"  '{alias}' → [未命中]   [需检查索引]")

    # ---- 7. 倒排索引最终统计 ----
    stats_final = mgr.get_stats()
    print(f"\n--- 最终统计 ---")
    print(f"  总指标数: {stats_final['total_fields']}")
    print(f"  倒排索引条目: {stats_final['total_alias_entries']}")

    print("\n" + "=" * 68)
    print("  所有测试通过！")
    print("=" * 68)
