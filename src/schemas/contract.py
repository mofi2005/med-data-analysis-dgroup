from typing import Any, TypedDict


class ClinicalFeatures(TypedDict, total=False):
    anchor_age: float
    GLU_mean: float
    GLU_latest: float
    Creatinine_mean: float
    CEA_level: float


class TextFeatures(TypedDict, total=False):
    self_report: str


class RadiomicsFeatures(TypedDict, total=False):
    texture_mean: float
    area_mean: float
    smoothness: float


class PatientPayload(TypedDict, total=False):
    patient_id: str
    clinical_features: ClinicalFeatures
    text_features: TextFeatures
    radiomics_features: RadiomicsFeatures
    _case_id: str
    _labs: list[dict]


class AIDecision(TypedDict, total=False):
    patient_id: str
    selected_model: str
    routing_score: float
    risk_probability: float
    prediction_label: str
    top_reason: str
    shap_values: dict[str, float]


ROUTER_PAYLOAD_KEYS = {"patient_id", "clinical_features", "text_features", "radiomics_features"}


def strip_router_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only contract 1.1 fields before sending to AdaptiveModelRouter."""
    return {key: payload[key] for key in ROUTER_PAYLOAD_KEYS if key in payload}
