import json
from pathlib import Path

MOCK_PATH = Path(__file__).resolve().parents[2] / "data" / "mock" / "b_group_artifacts.json"


def load_b_group_mock(case_id: str) -> dict | None:
    if not MOCK_PATH.exists():
        return None
    with open(MOCK_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get(case_id)


def list_b_group_mock_cases() -> list[str]:
    if not MOCK_PATH.exists():
        return []
    with open(MOCK_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.keys())
