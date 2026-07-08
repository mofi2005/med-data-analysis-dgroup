from datetime import datetime

import pandas as pd


def analyze_trend(records: list[dict], item_name: str) -> dict:
    df = pd.DataFrame(records)
    if df.empty or item_name not in df.columns:
        return {"item_name": item_name, "points": [], "trend_direction": "unknown"}

    series = pd.to_numeric(df[item_name], errors="coerce").dropna()
    if series.empty:
        return {"item_name": item_name, "points": [], "trend_direction": "unknown"}

    first_value = float(series.iloc[0])
    last_value = float(series.iloc[-1])
    change_rate = (last_value - first_value) / first_value if first_value else 0.0
    trend_direction = "up" if change_rate > 0.05 else "down" if change_rate < -0.05 else "stable"

    return {
        "item_name": item_name,
        "first_value": first_value,
        "last_value": last_value,
        "max_value": float(series.max()),
        "min_value": float(series.min()),
        "change_rate": round(change_rate, 4),
        "trend_direction": trend_direction,
        "points": series.tolist(),
    }


def _parse_test_time(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def compute_feature_slopes(labs: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[tuple[datetime | None, float]]] = {}
    for item in labs:
        marker = item.get("standard_item_name")
        value = item.get("value")
        if not marker or value is None:
            continue
        grouped.setdefault(marker, []).append((_parse_test_time(item.get("test_time")), float(value)))

    slopes: dict[str, float] = {}
    for marker, points in grouped.items():
        ordered = sorted(points, key=lambda x: (x[0] is None, x[0] or datetime.min))
        if len(ordered) < 2:
            continue
        first_time, first_value = ordered[0]
        last_time, last_value = ordered[-1]
        if first_time and last_time:
            day_span = max((last_time - first_time).days, 1)
            slopes[marker] = round((last_value - first_value) / day_span, 4)
        else:
            slopes[marker] = round(last_value - first_value, 4)
    return slopes


def latest_values_by_marker(labs: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[tuple[datetime | None, float]]] = {}
    for item in labs:
        marker = item.get("standard_item_name")
        value = item.get("value")
        if not marker or value is None:
            continue
        grouped.setdefault(marker, []).append((_parse_test_time(item.get("test_time")), float(value)))

    latest: dict[str, float] = {}
    for marker, points in grouped.items():
        ordered = sorted(points, key=lambda x: (x[0] is None, x[0] or datetime.min))
        latest[marker] = ordered[-1][1]
    return latest


def analyze_case_trend(labs: list[dict], item_name: str) -> dict:
    records = []
    for item in labs:
        if item.get("standard_item_name") != item_name:
            continue
        records.append(
            {
                "test_time": item.get("test_time"),
                item_name: item.get("value"),
            }
        )
    records = sorted(records, key=lambda x: x.get("test_time") or "")
    return analyze_trend(records, item_name)
