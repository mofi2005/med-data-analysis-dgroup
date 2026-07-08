import json
import re
from pathlib import Path

import pandas as pd

from app.services.dictionary_loader import load_standard_dictionary

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

FIELD_ALIASES = load_standard_dictionary() or {
    "白细胞": "WBC",
    "wbc": "WBC",
    "white blood cell": "WBC",
    "血红蛋白": "HGB",
    "hgb": "HGB",
    "hemoglobin": "HGB",
    "血小板": "PLT",
    "plt": "PLT",
    "血糖": "GLU",
    "glu": "GLU",
    "glucose": "GLU",
    "糖化血红蛋白": "HbA1c",
    "hba1c": "HbA1c",
    "肌酐": "Creatinine",
    "creatinine": "Creatinine",
    "crea": "Creatinine",
    "cr": "Creatinine",
    "尿素": "Urea",
    "urea": "Urea",
    "alt": "ALT",
    "ast": "AST",
    "cea": "CEA",
    "crp": "CRP",
}


def _load_reference_ranges() -> dict:
    with open(CONFIG_DIR / "reference_ranges.json", encoding="utf-8") as f:
        return json.load(f)


def normalize_item_name(raw_name: str) -> str:
    key = raw_name.strip().lower()
    return FIELD_ALIASES.get(raw_name.strip(), FIELD_ALIASES.get(key, raw_name.strip()))


def _column_map(df: pd.DataFrame) -> dict[str, str]:
    return {c.lower(): c for c in df.columns}


def _filter_case_rows(df: pd.DataFrame, case_id: str | None) -> pd.DataFrame:
    if not case_id:
        return df
    col_map = _column_map(df)
    if "case_id" not in col_map:
        return df
    return df[df[col_map["case_id"]].astype(str) == case_id]


def parse_lab_table(
    df: pd.DataFrame,
    patient_id: str,
    visit_id: str | None = None,
    case_id: str | None = None,
) -> dict:
    references = _load_reference_ranges()
    labs = []
    scoped_df = _filter_case_rows(df, case_id)
    col_map = _column_map(scoped_df)

    if scoped_df.empty:
        return {"case_id": case_id, "patient_id": patient_id, "visit_id": visit_id, "labs": labs}

    if {"item_name", "value"}.issubset(col_map.keys()):
        for _, row in scoped_df.iterrows():
            raw_name = str(row[col_map["item_name"]])
            standard_name = normalize_item_name(raw_name)
            value = pd.to_numeric(row[col_map["value"]], errors="coerce")
            unit = str(row[col_map["unit"]]) if "unit" in col_map and pd.notna(row[col_map["unit"]]) else None
            row_visit = str(row[col_map["visit_id"]]) if "visit_id" in col_map and pd.notna(row[col_map["visit_id"]]) else visit_id
            row_patient = str(row[col_map["patient_id"]]) if "patient_id" in col_map and pd.notna(row[col_map["patient_id"]]) else patient_id
            test_time = str(row[col_map["test_time"]]) if "test_time" in col_map and pd.notna(row[col_map["test_time"]]) else None
            item = _build_lab_item(raw_name, standard_name, value, unit, references)
            item["visit_id"] = row_visit
            item["patient_id"] = row_patient
            item["test_time"] = test_time
            labs.append(item)
        resolved_patient = str(scoped_df[col_map["patient_id"]].iloc[0]) if "patient_id" in col_map else patient_id
        resolved_visit = str(scoped_df[col_map["visit_id"]].iloc[0]) if "visit_id" in col_map else visit_id
        return {
            "case_id": case_id,
            "patient_id": resolved_patient,
            "visit_id": resolved_visit,
            "labs": labs,
        }

    for col in df.columns:
        if col.lower() in {"patient_id", "visit_id", "test_time", "id"}:
            continue
        value = pd.to_numeric(df[col].iloc[0], errors="coerce")
        standard_name = normalize_item_name(col)
        labs.append(_build_lab_item(col, standard_name, value, None, references))

    return {"case_id": case_id, "patient_id": patient_id, "visit_id": visit_id, "labs": labs}


def parse_lab_text(text: str, patient_id: str) -> dict:
    references = _load_reference_ranges()
    labs = []
    pattern = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9%\-]+)\s*[:：]?\s*([0-9.]+)\s*([A-Za-z/%μ^0-9·/.-]*)")
    for match in pattern.finditer(text):
        raw_name, value_str, unit = match.groups()
        standard_name = normalize_item_name(raw_name)
        value = float(value_str)
        labs.append(_build_lab_item(raw_name, standard_name, value, unit or None, references))
    return {"patient_id": patient_id, "labs": labs}


def _build_lab_item(raw_name: str, standard_name: str, value: float, unit: str | None, references: dict) -> dict:
    ref = references.get(standard_name, {})
    abnormal_flag = "normal"
    if pd.notna(value):
        low = ref.get("reference_low")
        high = ref.get("reference_high")
        if low is not None and value < low:
            abnormal_flag = "critical_low" if value < ref.get("critical_low", low) else "low"
        if high is not None and value > high:
            abnormal_flag = "critical_high" if value > ref.get("critical_high", high) else "high"
    return {
        "item_name": raw_name,
        "standard_item_name": standard_name,
        "value": None if pd.isna(value) else float(value),
        "unit": unit or ref.get("unit"),
        "reference_low": ref.get("reference_low"),
        "reference_high": ref.get("reference_high"),
        "abnormal_flag": abnormal_flag,
    }
