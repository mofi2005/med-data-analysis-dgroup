from pathlib import Path

from src.models.model_router import AdaptiveModelRouter
from src.parsers.your_parser import extract_and_standardize
from src.reports.report_generator import generate_clinical_report

SRC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_FILE = SRC_ROOT / "fixtures" / "labs_sample.csv"


def run_local_integration(
    file_path: str | Path | None = None,
    *,
    case_id: str = "p001_c1",
    patient_id: str | None = None,
) -> dict:
    source = Path(file_path or DEFAULT_TEST_FILE)

    payload = extract_and_standardize(source, case_id=case_id, patient_id=patient_id)
    router = AdaptiveModelRouter()
    decision = router.route_case(payload)
    report_path = generate_clinical_report(payload, decision)

    return {
        "patient_payload": payload,
        "ai_decision": decision,
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    result = run_local_integration()
    print(f"[SUCCESS] 跨设备自动化链路贯通！最终报告已生成: {result['report_path']}")
    print(f"selected_model={result['ai_decision']['selected_model']}")
    print(f"target_disease_cn={result['ai_decision']['target_disease_cn']}")
