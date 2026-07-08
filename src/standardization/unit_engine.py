from app.services.unit_converter import standardize_lab_units


def standardize_labs(labs: list[dict]) -> list[dict]:
    """Unit normalization wrapper for contract pipeline."""
    return standardize_lab_units(labs)
