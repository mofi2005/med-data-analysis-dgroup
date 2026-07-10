import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def api_post(path: str, **kwargs):
    return requests.post(f"{API_BASE}{path}", timeout=30, **kwargs)


def api_get(path: str, **kwargs):
    return requests.get(f"{API_BASE}{path}", timeout=30, **kwargs)


def build_route_payload(
    patient_id: str,
    time_series_steps: int,
    nlp_entities: int,
    missing_rate: float,
    glu: float,
    cea: float,
) -> dict:
    return {
        "patient_profile": {
            "patient_id": patient_id,
            "time_series_steps": time_series_steps,
            "nlp_entities": nlp_entities,
            "missing_rate": missing_rate,
        },
        "clinical_data": {
            "GLU": glu,
            "CEA": cea,
        },
    }


def fetch_route_prediction(payload: dict) -> dict:
    resp = api_post("/api/multimodal/route_and_predict", json=payload)
    resp.raise_for_status()
    return resp.json()


def trigger_mlops_evolution() -> dict:
    resp = api_post("/api/mlops/trigger_evolution_demo")
    resp.raise_for_status()
    return resp.json()


def render_mlops_badge(mlops_enhanced: bool) -> None:
    if mlops_enhanced:
        st.markdown(
            """
            <div style="
                display:inline-flex; align-items:center; gap:8px;
                padding:10px 16px; border-radius:999px;
                background:linear-gradient(135deg,#fff8e1,#ffe082);
                border:2px solid #f9a825; color:#6d4c00;
                font-weight:700; font-size:15px;
            ">
                🏅 MLOps Enhanced · 当前诊断模型已自动更新至最新版本
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("ℹ️ 当前使用注册表初始战绩（可点击「触发 AI 进化演示」体验自进化闭环）")


def render_mlops_scores_chart(
    scores: dict[str, float],
    previous_scores: dict[str, float] | None = None,
) -> None:
    if not scores:
        st.warning("暂无 mlops_live_scores 数据")
        return

    labels = list(scores.keys())
    values = [float(scores[k]) for k in labels]
    fig = go.Figure()
    fig.add_bar(name="当前战绩", x=labels, y=values, marker_color="#0b5cab")
    if previous_scores:
        prev_values = [float(previous_scores.get(k, 0)) for k in labels]
        fig.add_bar(name="进化前", x=labels, y=prev_values, marker_color="#90a4ae")
    fig.update_layout(
        title="MLOps 模型战绩分 (P_base)",
        barmode="group",
        yaxis_title="Score",
        xaxis_title="Model",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_prediction_panel(data: dict) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("选中模型", data.get("selected_model", "—"))
    col2.metric("路由得分", f"{data.get('winning_score', 0):.2f}")
    col3.metric("引擎模式", data.get("engine_mode", "—"))

    render_mlops_badge(bool(data.get("mlops_enhanced")))

    previous = st.session_state.get("mlops_scores_before")
    render_mlops_scores_chart(data.get("mlops_live_scores") or {}, previous)

    st.markdown("**路由理由**")
    st.write(data.get("routing_reason") or "—")

    fused = data.get("fused_prediction") or {}
    if fused:
        st.markdown("**融合预测**")
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("预测结论", fused.get("result") or fused.get("predicted_diagnosis") or "—")
        fc2.metric("风险概率", f"{fused.get('risk_probability', 0):.2f}")
        fc3.metric("风险等级", fused.get("risk_level", "—"))

        shap_items = fused.get("shap_explanations") or fused.get("shap_feature_importance") or []
        if shap_items:
            st.markdown("**SHAP 特征贡献**")
            if isinstance(shap_items, list) and shap_items and isinstance(shap_items[0], dict):
                st.dataframe(pd.DataFrame(shap_items), use_container_width=True)
            else:
                st.json(shap_items)

    with st.expander("完整 JSON 响应"):
        st.json(data)


st.set_page_config(page_title="D组 · 医学数据中台", layout="wide")
st.title("D组 · 医学文本与生化数据分析中台")

if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = ""
if "case_id" not in st.session_state:
    st.session_state.case_id = "p001_c1"
if "mlops_prediction" not in st.session_state:
    st.session_state.mlops_prediction = None
if "mlops_scores_before" not in st.session_state:
    st.session_state.mlops_scores_before = None

tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["MLOps 演示", "数据导入", "生化分析", "统计分析", "报告生成"]
)

with tab0:
    st.subheader("多模态路由 & MLOps 自进化演示")
    st.caption("对接 POST /api/multimodal/route_and_predict 与 POST /api/mlops/trigger_evolution_demo")

    health = api_get("/api/health")
    if health.ok:
        st.success(f"后端已连接 · {health.json().get('message', 'ok')}")
    else:
        st.error("无法连接后端，请先启动：python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000")

    c1, c2, c3, c4 = st.columns(4)
    patient_id = c1.text_input("patient_id", value="p001", key="mlops_patient")
    time_series_steps = c2.number_input("时序步数", min_value=0, value=5, step=1)
    nlp_entities = c3.number_input("NLP 实体数", min_value=0, value=4, step=1)
    missing_rate = c4.slider("缺失率", 0.0, 1.0, 0.1, 0.05)

    c5, c6 = st.columns(2)
    glu = c5.number_input("GLU", min_value=0.0, value=9.8, step=0.1)
    cea = c6.number_input("CEA", min_value=0.0, value=3.5, step=0.1)

    payload = build_route_payload(
        patient_id=patient_id,
        time_series_steps=int(time_series_steps),
        nlp_entities=int(nlp_entities),
        missing_rate=float(missing_rate),
        glu=float(glu),
        cea=float(cea),
    )

    btn_predict, btn_evolve, btn_refresh = st.columns([1, 1, 1])
    with btn_predict:
        do_predict = st.button("🔮 路由预测", type="primary", use_container_width=True)
    with btn_evolve:
        do_evolve = st.button("🚀 触发 AI 进化演示", use_container_width=True)
    with btn_refresh:
        do_refresh = st.button("🔄 刷新预测结果", use_container_width=True)

    if do_predict or do_refresh:
        try:
            data = fetch_route_prediction(payload)
            st.session_state.mlops_prediction = data
            st.session_state.mlops_scores_before = None
        except requests.RequestException as exc:
            st.error(f"预测失败：{exc}")

    if do_evolve:
        try:
            current = st.session_state.get("mlops_prediction") or {}
            if current.get("mlops_live_scores"):
                st.session_state.mlops_scores_before = dict(current["mlops_live_scores"])
            evo = trigger_mlops_evolution()
            st.success(evo.get("message", "MLOps 进化完成"))
            if evo.get("updated_scores"):
                st.json({"updated_scores": evo["updated_scores"]})
            data = fetch_route_prediction(payload)
            st.session_state.mlops_prediction = data
        except requests.RequestException as exc:
            st.error(f"进化演示失败：{exc}")

    if st.session_state.mlops_prediction:
        render_prediction_panel(st.session_state.mlops_prediction)

with tab1:
    st.subheader("数据导入")
    uploaded = st.file_uploader("上传 CSV / Excel", type=["csv", "xlsx", "xls"])
    if uploaded and st.button("上传到 API"):
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "text/csv")}
        resp = api_post("/api/data/upload", files=files)
        if resp.ok:
            data = resp.json()
            st.session_state.dataset_id = data.get("dataset_id") or data.get("id") or ""
            st.success(f"上传成功")
            st.json(data)
        else:
            resp_legacy = api_post("/api/v1/ingest/upload", files=files)
            if resp_legacy.ok:
                data = resp_legacy.json()["data"]
                st.session_state.dataset_id = data["dataset_id"]
                st.success(f"上传成功，dataset_id = {data['dataset_id']}")
                st.dataframe(pd.DataFrame(data["preview"]))
            else:
                st.error(resp.text)

    st.caption("或使用样例文件")
    if st.button("上传 labs_sample.csv"):
        sample = SAMPLES / "labs_sample.csv"
        if not sample.exists():
            sample = Path(__file__).resolve().parents[1] / "src" / "fixtures" / "labs_sample.csv"
        files = {"file": ("labs_sample.csv", sample.read_bytes(), "text/csv")}
        resp = api_post("/api/data/upload", files=files)
        if not resp.ok:
            resp = api_post("/api/v1/ingest/upload", files=files)
        if resp.ok:
            body = resp.json()
            st.session_state.dataset_id = body.get("dataset_id") or body.get("data", {}).get("dataset_id", "")
            st.success("样例已上传")

with tab2:
    st.subheader("生化分析")
    st.session_state.case_id = st.text_input("case_id", value=st.session_state.case_id)
    patient_id = st.text_input("patient_id", value="p001")
    primary_disease = st.selectbox(
        "primary_disease（留空则自动调组员1，联调前可选手动指定）",
        ["", "diabetes", "kidney_disease", "liver_disease", "default"],
        index=0,
    )

    if st.button("解析 dataset") and st.session_state.dataset_id:
        body = {
            "dataset_id": st.session_state.dataset_id,
            "case_id": st.session_state.case_id,
            "patient_id": patient_id,
            "visit_id": "v001",
        }
        if primary_disease:
            body["primary_disease"] = primary_disease
        resp = api_post("/api/v1/lab/parse-dataset", json=body)
        if resp.ok:
            data = resp.json()["data"]
            st.success(f"解析完成 · primary_disease={data.get('primary_disease')}")
            st.json({"features": data.get("features"), "quality": data.get("quality")})
            st.dataframe(pd.DataFrame(data.get("labs", [])))
        else:
            st.error(resp.text)

    if st.button("读取 labs API"):
        resp = api_get(f"/api/v1/labs/{st.session_state.case_id}")
        if not resp.ok:
            resp = api_get(f"/api/labs/trend", params={"patient_id": patient_id})
        if resp.ok:
            st.json(resp.json())
        else:
            st.error(resp.text)

with tab3:
    st.subheader("统计分析")
    stats_file = st.selectbox("统计样例", ["stats_sample.csv"])
    group_col = st.text_input("分组列", value="group")
    model_id = st.text_input("模型 ID", value="demo_stats_model")

    if st.button("上传统计样例并生成报告"):
        sample = SAMPLES / stats_file
        if not sample.exists():
            st.warning("样例文件不存在，请先在「数据导入」上传 CSV")
        else:
            files = {"file": (stats_file, sample.read_bytes(), "text/csv")}
            upload = api_post("/api/data/upload", files=files)
            if not upload.ok:
                upload = api_post("/api/v1/ingest/upload", files=files)
            if not upload.ok:
                st.error(upload.text)
            else:
                body = upload.json()
                ds_id = body.get("dataset_id") or body.get("data", {}).get("dataset_id")
                report = api_get(f"/api/v1/stats/report/{ds_id}", params={"group_col": group_col})
                if not report.ok:
                    report = api_post("/api/stats/correlation", json={"dataset_id": ds_id})
                if report.ok:
                    payload = report.json()
                    data = payload.get("data", payload)
                    st.subheader("统计结果")
                    st.json(data)
                    heatmap = data.get("heatmap_series_data") or data.get("echarts_data")
                    if isinstance(heatmap, dict) and heatmap.get("values"):
                        fig = go.Figure(
                            data=go.Heatmap(
                                z=heatmap["values"],
                                x=heatmap.get("labels", heatmap.get("x")),
                                y=heatmap.get("labels", heatmap.get("y")),
                                colorscale="RdBu",
                                zmid=0,
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(report.text)

    st.divider()
    st.subheader("SHAP 模型解释")
    if st.button("训练并解释（stats_sample）"):
        sample = SAMPLES / "stats_sample.csv"
        upload = api_post("/api/v1/ingest/upload", files={"file": ("stats_sample.csv", sample.read_bytes(), "text/csv")})
        if upload.ok:
            ds_id = upload.json()["data"]["dataset_id"]
            train = api_post(
                "/api/v1/feature/train",
                json={"dataset_id": ds_id, "target_col": group_col, "model_id": model_id, "task_type": "classification"},
            )
            if train.ok:
                explain = api_post(
                    "/api/v1/feature/explain",
                    json={"model_id": model_id, "dataset_id": ds_id, "top_k": 10},
                )
                if explain.ok:
                    data = explain.json()["data"]
                    st.success(f"方法: {data.get('method')}")
                    st.dataframe(pd.DataFrame(data.get("feature_importance", [])))
                    chart = data.get("chart", {})
                    if chart.get("labels"):
                        fig = px.bar(x=chart["values"], y=chart["labels"], orientation="h", title=chart.get("title"))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(explain.text)
            else:
                st.error(train.text)

with tab4:
    st.subheader("疾病导向报告")
    case_id = st.text_input("报告 case_id", value=st.session_state.case_id, key="report_case")
    patient_id_report = st.text_input("报告 patient_id", value="p001", key="report_patient")

    if st.button("嵌入 HTML 报告（iframe）"):
        report_url = f"{API_BASE}/api/reports/download?patient_id={patient_id_report}"
        st.markdown(f"[打开报告]({report_url})")
        st.components.v1.iframe(report_url, height=700, scrolling=True)

    st.divider()
    disease_override = st.selectbox(
        "primary_disease 覆盖",
        ["auto", "diabetes", "kidney_disease", "liver_disease", "default"],
        index=0,
    )
    include_shap = st.checkbox("嵌入 SHAP 解释（需 model_id）", value=True)
    model_id_report = st.text_input("model_id（可选，用于 SHAP）", value="", key="report_model_id")

    if st.button("生成报告（旧 API）"):
        body = {
            "case_id": case_id,
            "patient_id": patient_id_report,
            "include_prediction": True,
            "include_shap": include_shap,
            "format": "html",
        }
        if disease_override != "auto":
            body["primary_disease"] = disease_override
        if model_id_report.strip():
            body["model_id"] = model_id_report.strip()
        resp = api_post("/api/v1/report/generate", json=body)
        if resp.ok:
            html = resp.json()["data"]["html"]
            st.components.v1.html(html, height=700, scrolling=True)
        else:
            st.error(resp.text)

st.caption("请先启动 API：python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000")
