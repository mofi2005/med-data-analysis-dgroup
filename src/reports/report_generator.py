from pathlib import Path
from typing import Any

from app.services.artifact_loader import load_b_group_mock
from app.services.report_exporter import export_html
from app.services.report_generator import build_report_data, generate_report
from app.settings import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _label_to_primary_disease(prediction_label: str) -> str:
    label = prediction_label or ""
    if any(key in label for key in ("糖尿病", "糖代谢", "GLU")):
        return "diabetes"
    if any(key in label for key in ("肾", "肌酐", "Creatinine")):
        return "kidney_disease"
    if any(key in label for key in ("肝", "ALT", "AST")):
        return "liver_disease"
    return "default"


def _shap_values_to_explanation(shap_values: dict[str, float] | None) -> dict[str, Any]:
    if not shap_values:
        return {}

    ranked = sorted(shap_values.items(), key=lambda item: abs(item[1]), reverse=True)
    feature_importance = [{"name": name, "importance": round(abs(value), 4)} for name, value in ranked]
    return {
        "method": "AdaptiveModelRouter.shap_values",
        "feature_importance": feature_importance,
        "chart": {
            "type": "bar",
            "title": "SHAP Feature Importance",
            "labels": [item["name"] for item in feature_importance[:10]],
            "values": [item["importance"] for item in feature_importance[:10]],
        },
        "sample_explanations": [
            {
                "row_index": 0,
                "contributions": [
                    {"name": name, "shap_value": round(value, 4)}
                    for name, value in ranked[:10]
                ],
            }
        ],
    }


def _build_prediction(ai_decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction": ai_decision.get("prediction_label"),
        "probability": ai_decision.get("risk_probability"),
        "recommended_model": ai_decision.get("selected_model"),
        "model_reason": ai_decision.get("top_reason"),
        "routing_score": ai_decision.get("routing_score"),
    }


def generate_clinical_report(
    patient_payload: dict[str, Any],
    ai_decision: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Render disease-oriented HTML report from contract 1.1 payload + 1.2 decision.
    """
    patient_id = patient_payload.get("patient_id", "unknown")
    case_id = patient_payload.get("_case_id")
    labs = patient_payload.get("_labs") or []
    primary_disease = _label_to_primary_disease(ai_decision.get("prediction_label", ""))

    artifact_metrics = load_b_group_mock(case_id) if case_id else None
    shap_explanation = _shap_values_to_explanation(ai_decision.get("shap_values"))
    prediction = _build_prediction(ai_decision)

    html = generate_report(
        patient_id=patient_id,
        case_id=case_id,
        primary_disease=primary_disease,
        labs=labs,
        prediction=prediction,
        quality_warnings=[],
        shap_explanation=shap_explanation,
        artifact_metrics=artifact_metrics or {},
        clinical_summary_override=ai_decision.get("top_reason"),
    )

    export_root = Path(output_dir or settings.export_dir)
    export_root.mkdir(parents=True, exist_ok=True)
    file_stem = case_id or patient_id
    output_path = export_root / f"{file_stem}_clinical_report.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path
