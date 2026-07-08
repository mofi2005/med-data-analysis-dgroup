import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_unit_matrix() -> dict:
    with open(CONFIG_DIR / "unit_matrix.json", encoding="utf-8") as f:
        return json.load(f)


def convert_unit(marker: str, value: float, from_unit: str) -> tuple[float, str]:
    matrix = _load_unit_matrix()
    marker_cfg = matrix.get(marker)
    if not marker_cfg:
        return value, from_unit

    standard_unit = marker_cfg.get("standard_unit", from_unit)
    if from_unit == standard_unit:
        return value, standard_unit

    if from_unit not in marker_cfg:
        return value, from_unit

    base_value = value / marker_cfg[from_unit]
    return base_value * marker_cfg[standard_unit], standard_unit


def standardize_lab_units(labs: list[dict]) -> list[dict]:
    standardized = []
    for item in labs:
        marker = item.get("standard_item_name")
        value = item.get("value")
        unit = item.get("unit")
        if marker and value is not None and unit:
            new_value, new_unit = convert_unit(marker, value, unit)
            item = {**item, "value": round(new_value, 4), "unit": new_unit}
        standardized.append(item)
    return standardized
