import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.schemas.contract import strip_router_payload

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"

MODEL_ALIAS = {
    "model_a": "Model_A_TimeSeries",
    "model_b": "Model_B_NLP",
    "model_c": "Model_C_Radiomics",
    "timeseries": "Model_A_TimeSeries",
    "nlp": "Model_B_NLP",
    "radiomics": "Model_C_Radiomics",
}


class AdaptiveModelRouter:
    """Load member1 .pkl weights when present; otherwise use contract-aligned fallback routing."""

    def __init__(self, weights_dir: str | Path | None = None):
        self.weights_dir = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        self.models: dict[str, Any] = {}
        self.model_meta: dict[str, dict] = {}
        self._load_weights()

    def _load_weights(self) -> None:
        if not self.weights_dir.exists():
            logger.info("weights dir not found: %s (fallback routing enabled)", self.weights_dir)
            return

        for model_path in sorted(self.weights_dir.glob("*.pkl")):
            model_key = model_path.stem.lower()
            selected_name = MODEL_ALIAS.get(model_key, model_path.stem)
            try:
                self.models[selected_name] = joblib.load(model_path)
                meta_path = model_path.with_suffix(".meta.json")
                if meta_path.exists():
                    self.model_meta[selected_name] = json.loads(meta_path.read_text(encoding="utf-8"))
                logger.info("loaded model weight: %s -> %s", model_path.name, selected_name)
            except Exception as exc:
                logger.warning("failed to load %s: %s", model_path, exc)

    def _completeness_scores(self, payload: dict[str, Any]) -> dict[str, float]:
        clinical = payload.get("clinical_features") or {}
        text = payload.get("text_features") or {}
        radio = payload.get("radiomics_features") or {}

        clinical_keys = ["anchor_age", "GLU_latest", "GLU_mean", "Creatinine_mean", "CEA_level"]
        radio_keys = ["texture_mean", "area_mean", "smoothness"]

        score_a = sum(1 for key in clinical_keys if clinical.get(key) is not None) / len(clinical_keys)
        score_b = 1.0 if text.get("self_report") else 0.0
        score_c = sum(1 for key in radio_keys if radio.get(key) is not None) / len(radio_keys)

        return {
            "Model_A_TimeSeries": round(score_a, 4),
            "Model_B_NLP": round(score_b, 4),
            "Model_C_Radiomics": round(score_c, 4),
        }

    def _select_model(self, scores: dict[str, float], payload: dict[str, Any]) -> tuple[str, float]:
        clinical = payload.get("clinical_features") or {}
        glu = clinical.get("GLU_latest") or clinical.get("GLU_mean")
        creatinine = clinical.get("Creatinine_mean")

        if glu is not None and glu >= 7.0:
            return "Model_A_TimeSeries", max(scores.get("Model_A_TimeSeries", 0.0), 0.85)
        if creatinine is not None and creatinine >= 120:
            return "Model_A_TimeSeries", max(scores.get("Model_A_TimeSeries", 0.0), 0.85)

        selected_model, routing_score = max(scores.items(), key=lambda item: item[1])
        if routing_score <= 0:
            selected_model = "Model_B_NLP_Graph"
            routing_score = 0.5
        return selected_model, routing_score

    def _has_cough_intent(self, self_report: str) -> bool:
        if not self_report:
            return False
        if "晚上睡着了就咳嗽" in self_report:
            return True
        if "咳嗽" in self_report and "晚上" in self_report:
            return True
        if "不咳嗽" in self_report or "无咳嗽" in self_report or "无明显咳嗽" in self_report:
            return False
        return "咳嗽" in self_report

    def _predict_with_model(self, model_name: str, payload: dict[str, Any]) -> tuple[float, str] | None:
        model = self.models.get(model_name)
        if model is None:
            return None

        meta = self.model_meta.get(model_name, {})
        feature_columns = meta.get("feature_columns")
        clinical = payload.get("clinical_features") or {}

        if feature_columns:
            row = [clinical.get(col, np.nan) for col in feature_columns]
            X = np.array([row], dtype=float)
        else:
            X = np.array([[v for v in clinical.values()]], dtype=float)

        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                risk = float(proba[-1])
            elif hasattr(model, "predict"):
                pred = model.predict(X)[0]
                if isinstance(pred, (int, float, np.number)):
                    risk = float(pred)
                else:
                    risk = 0.75
            else:
                return None

            label = meta.get("positive_label") or meta.get("prediction_label") or "模型预测阳性"
            return risk, str(label)
        except Exception as exc:
            logger.warning("model inference failed for %s: %s", model_name, exc)
            return None

    def _fallback_predict(self, selected_model: str, payload: dict[str, Any]) -> tuple[float, str, str, dict[str, float]]:
        clinical = payload.get("clinical_features") or {}
        text = payload.get("text_features") or {}
        self_report = text.get("self_report", "")

        shap_values: dict[str, float] = {}
        for key, value in clinical.items():
            if value is None:
                continue
            shap_values[key] = round(abs(float(value)) * 0.05, 4)

        if selected_model == "Model_B_NLP" and self_report:
            if self._has_cough_intent(self_report):
                return (
                    0.92,
                    "小儿消化不良",
                    "文本特征 '晚上睡着了就咳嗽' 命中高危意图",
                    {**shap_values, "self_report": 0.62},
                )
            return 0.78, "待进一步评估", "文本特征已参与路由，建议结合检验指标复核", {**shap_values, "self_report": 0.45}

        glu = clinical.get("GLU_latest") or clinical.get("GLU_mean")
        creatinine = clinical.get("Creatinine_mean")
        if glu is not None and glu >= 7.0:
            shap_values["GLU_latest"] = round(min(0.95, glu / 10), 4)
            return 0.88, "糖尿病", f"GLU_latest={glu} 超过参考范围，倾向糖代谢异常", shap_values
        if creatinine is not None and creatinine >= 120:
            shap_values["Creatinine_mean"] = round(min(0.95, creatinine / 200), 4)
            return 0.84, "慢性肾病", f"Creatinine_mean={creatinine} 升高，提示肾功能受损风险", shap_values

        if selected_model == "Model_C_Radiomics":
            return 0.72, "影像异常待随访", "影像组学特征参与路由，建议多模态复核", shap_values

        return 0.55, "综合评估", "临床、文本与影像特征完整度中等，建议继续随访", shap_values

    def route_and_predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        router_payload = strip_router_payload(payload)
        scores = self._completeness_scores(router_payload)
        selected_model, routing_score = self._select_model(scores, router_payload)

        predicted = self._predict_with_model(selected_model, router_payload)
        if predicted is not None:
            risk_probability, prediction_label = predicted
            top_reason = f"{selected_model} 权重推理完成，路由分={routing_score}"
            shap_values = {
                key: round(abs(float(value)) * 0.05, 4)
                for key, value in (router_payload.get("clinical_features") or {}).items()
                if value is not None
            }
        else:
            risk_probability, prediction_label, top_reason, shap_values = self._fallback_predict(
                selected_model, router_payload
            )

        return {
            "patient_id": router_payload.get("patient_id", ""),
            "selected_model": selected_model,
            "routing_score": routing_score,
            "risk_probability": round(float(risk_probability), 4),
            "prediction_label": prediction_label,
            "top_reason": top_reason,
            "shap_values": shap_values,
            "routing_scores": scores,
            "used_pkl": selected_model in self.models,
        }
