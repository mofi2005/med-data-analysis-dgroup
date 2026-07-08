import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.lab_parser import parse_lab_table, parse_lab_text
from app.services.trend_analyzer import compute_feature_slopes, latest_values_by_marker
from src.standardization.unit_engine import standardize_labs

SRC_ROOT = Path(__file__).resolve().parents[1]
MOCK_TEXT_PATH = SRC_ROOT / "fixtures" / "patient_text_features.json"
MOCK_RADIO_PATH = SRC_ROOT / "fixtures" / "patient_radiomics_features.json"


def _load_mock_map(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_dataframe(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict) and "labs" in payload:
            return pd.DataFrame(payload["labs"])
        return pd.DataFrame([payload])
    raise ValueError(f"unsupported file type: {suffix}")


def _resolve_case_id(df: pd.DataFrame, case_id: str | None, patient_id: str | None) -> str | None:
    if case_id:
        return case_id
    col_map = {c.lower(): c for c in df.columns}
    if "case_id" in col_map:
        values = df[col_map["case_id"]].dropna().astype(str).unique().tolist()
        if len(values) == 1:
            return values[0]
    return None


def _resolve_patient_id(df: pd.DataFrame, patient_id: str | None, case_id: str | None) -> str:
    if patient_id:
        return patient_id
    col_map = {c.lower(): c for c in df.columns}
    if "case_id" in col_map and case_id:
        scoped = df[df[col_map["case_id"]].astype(str) == case_id]
        if "patient_id" in col_map and not scoped.empty:
            return str(scoped[col_map["patient_id"]].iloc[0])
    if "patient_id" in col_map and not df.empty:
        return str(df[col_map["patient_id"]].iloc[0])
    return case_id or "unknown_patient"


def _extract_anchor_age(df: pd.DataFrame, case_id: str | None) -> float | None:
    col_map = {c.lower(): c for c in df.columns}
    if "anchor_age" not in col_map and "age" not in col_map:
        return None
    key = "anchor_age" if "anchor_age" in col_map else "age"
    scoped = df
    if case_id and "case_id" in col_map:
        scoped = df[df[col_map["case_id"]].astype(str) == case_id]
    if scoped.empty:
        return None
    value = pd.to_numeric(scoped[col_map[key]].iloc[0], errors="coerce")
    return None if pd.isna(value) else float(value)


def _build_clinical_features(labs: list[dict], anchor_age: float | None) -> dict[str, float]:
    latest = latest_values_by_marker(labs)
    slopes = compute_feature_slopes(labs)
    grouped_values: dict[str, list[float]] = {}
    for item in labs:
        marker = item.get("standard_item_name")
        value = item.get("value")
        if marker and value is not None:
            grouped_values.setdefault(marker, []).append(float(value))

    clinical: dict[str, float] = {}
    if anchor_age is not None:
        clinical["anchor_age"] = round(anchor_age, 2)

    if "GLU" in latest:
        clinical["GLU_latest"] = round(latest["GLU"], 4)
    if "GLU" in grouped_values:
        clinical["GLU_mean"] = round(sum(grouped_values["GLU"]) / len(grouped_values["GLU"]), 4)
    if "Creatinine" in grouped_values:
        clinical["Creatinine_mean"] = round(sum(grouped_values["Creatinine"]) / len(grouped_values["Creatinine"]), 4)
    elif "Creatinine" in latest:
        clinical["Creatinine_mean"] = round(latest["Creatinine"], 4)
    if "CEA" in latest:
        clinical["CEA_level"] = round(latest["CEA"], 4)

    for marker, slope in slopes.items():
        clinical[f"{marker}_slope"] = slope

    return clinical


def _resolve_sidecar_features(
    *,
    case_id: str | None,
    patient_id: str,
    text_features: dict | None,
    radiomics_features: dict | None,
) -> tuple[dict, dict]:
    text_map = _load_mock_map(MOCK_TEXT_PATH)
    radio_map = _load_mock_map(MOCK_RADIO_PATH)

    text = text_features or text_map.get(case_id or "") or text_map.get(patient_id) or {}
    radio = radiomics_features or radio_map.get(case_id or "") or radio_map.get(patient_id) or {}
    return text, radio


def extract_and_standardize(
    file_path: str | Path,
    *,
    patient_id: str | None = None,
    case_id: str | None = None,
    visit_id: str | None = None,
    text_features: dict | None = None,
    radiomics_features: dict | None = None,
) -> dict[str, Any]:
    """
    Parse local file, standardize units, and assemble contract 1.1 patient_payload.

    Internal fields `_case_id` and `_labs` are kept for report rendering.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    if path.suffix.lower() == ".txt":
        resolved_patient = patient_id or path.stem
        parsed = parse_lab_text(path.read_text(encoding="utf-8"), resolved_patient)
        labs = standardize_labs(parsed.get("labs", []))
        resolved_case = case_id
        anchor_age = None
    else:
        df = _read_dataframe(path)
        resolved_case = _resolve_case_id(df, case_id, patient_id)
        resolved_patient = _resolve_patient_id(df, patient_id, resolved_case)
        parsed = parse_lab_table(df, resolved_patient, visit_id=visit_id, case_id=resolved_case)
        labs = standardize_labs(parsed.get("labs", []))
        anchor_age = _extract_anchor_age(df, resolved_case)

    text, radio = _resolve_sidecar_features(
        case_id=resolved_case,
        patient_id=resolved_patient,
        text_features=text_features,
        radiomics_features=radiomics_features,
    )

    payload: dict[str, Any] = {
        "patient_id": resolved_patient,
        "clinical_features": _build_clinical_features(labs, anchor_age),
        "text_features": text,
        "radiomics_features": radio,
        "_case_id": resolved_case,
        "_labs": labs,
    }
    return payload
