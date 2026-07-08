import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_CONFIG_DIR = Path(__file__).resolve().parent / "config"
SHARED_CONFIG_DIR = PROJECT_ROOT / "config"


def load_standard_dictionary() -> dict[str, str]:
    """Build alias -> standard_name map from shared dictionary JSON."""
    dict_path = SHARED_CONFIG_DIR / "standard_dictionary.json"
    if not dict_path.exists():
        return {}

    with open(dict_path, encoding="utf-8") as f:
        payload = json.load(f)

    aliases: dict[str, str] = {}
    for field in payload.get("fields", []):
        standard_name = field.get("standard_name")
        if not standard_name:
            continue
        aliases[standard_name] = standard_name
        aliases[standard_name.lower()] = standard_name
        for alias in field.get("alias_names", []):
            aliases[str(alias).strip()] = standard_name
            aliases[str(alias).strip().lower()] = standard_name
    return aliases
