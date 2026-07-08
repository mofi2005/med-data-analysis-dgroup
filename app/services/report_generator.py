import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.config_paths import CONFIG_DIR
from app.services.report_charts import build_shap_bar_svg

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"


class DataQualityWarning(Exception):
    pass


def _load_disease_markers() -> dict:
    with open(CONFIG_DIR / "disease_markers.json", encoding="utf-8") as f:
        return json.load(f)


def _split_labs(labs: list[dict], primary_disease: str) -> tuple[list[dict], list[dict]]:
    disease_cfg = _load_disease_markers().get(primary_disease, _load_disease_markers()["default"])
    primary_markers = set(disease_cfg.get("primary", []))
    primary_labs = [lab for lab in labs if lab.get("standard_item_name") in primary_markers]
    appendix_labs = [lab for lab in labs if lab.get("standard_item_name") not in primary_markers]
    return primary_labs, appendix_labs


def build_report_data(
    *,
    patient_id: str,
    case_id: str | None,
    primary_disease: str,
    labs: list[dict],
    prediction: dict | None = None,
    quality_warnings: list[str] | None = None,
    shap_explanation: dict | None = None,
    artifact_metrics: dict | None = None,
    clinical_summary_override: str | None = None,
) -> dict[str, Any]:
    disease_cfg = _load_disease_markers().get(primary_disease, _load_disease_markers()["default"])
    primary_labs, appendix_labs = _split_labs(labs, primary_disease)
    shap_svg = ""
    if shap_explanation and shap_explanation.get("chart"):
        shap_svg = build_shap_bar_svg(shap_explanation["chart"])

    clinical_summary = clinical_summary_override or {
        "diabetes": "核心糖代谢指标升高，建议加强血糖监测并评估治疗方案。",
        "kidney_disease": "肾功能相关核心指标需重点关注，建议评估 eGFR 与肌酐变化趋势。",
        "liver_disease": "肝功能相关指标需重点监测，建议评估 ALT/AST 变化。",
    }.get(primary_disease, "请结合临床背景与检验结果综合评估。")

    return {
        "patient_id": patient_id,
        "case_id": case_id,
        "report_title": disease_cfg.get("report_title", "临床数据分析报告"),
        "primary_disease": primary_disease,
        "primary_labs": primary_labs,
        "appendix_labs": appendix_labs,
        "prediction": prediction or {},
        "quality_warnings": quality_warnings or [],
        "shap_explanation": shap_explanation or {},
        "artifact_metrics": artifact_metrics or {},
        "shap_svg": shap_svg,
        "clinical_summary": clinical_summary,
        "report_template": disease_cfg.get("report_template", "report_default.html"),
    }


def generate_report(**kwargs) -> str:
    data = build_report_data(**kwargs)
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )

    if data["quality_warnings"] and any("DataQualityWarning" in w for w in data["quality_warnings"]):
        template = env.get_template("report_data_missing.html")
        return template.render(
            patient_id=data["patient_id"],
            case_id=data["case_id"],
            warnings=data["quality_warnings"],
        )

    template = env.get_template(data["report_template"])

    return template.render(
        patient_id=data["patient_id"],
        case_id=data["case_id"],
        report_title=data["report_title"],
        primary_disease=data["primary_disease"],
        primary_labs=data["primary_labs"],
        appendix_labs=data["appendix_labs"],
        prediction=data["prediction"],
        shap_explanation=data["shap_explanation"] or None,
        artifact_metrics=data["artifact_metrics"],
        shap_svg=data["shap_svg"],
        clinical_summary=data["clinical_summary"],
    )
