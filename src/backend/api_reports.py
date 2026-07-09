#!/usr/bin/env python3
"""临床自动分析报告导出 API — D组医学数据中台.

提供:
  - GET /api/reports/download  下载/生成临床诊断决策报告 (HTML/JSON)
"""

from __future__ import annotations

import json as _json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# ---------------------------------------------------------------------------
# APIRouter 实例
# ---------------------------------------------------------------------------

reports_router = APIRouter(
    prefix="/api/reports",
    tags=["自动化决策报告导出"],
)

# ---------------------------------------------------------------------------
# 导出目录
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_EXPORT_DIR = _PROJECT_ROOT / "data" / "exports"
os.makedirs(str(_EXPORT_DIR), exist_ok=True)


# ===================================================================
# HTML 报告生成器
# ===================================================================

def _render_triple_rows(triples: list[dict]) -> str:
    rows = ""
    for t in triples:
        rows += f"<tr><td>{t.get('subject','')}</td><td>{t.get('relation','')}</td>"
        rows += f"<td>{t.get('object','')}</td><td>{t.get('confidence',0)}</td></tr>"
    return rows


def _render_neg_rows(negations: list[dict]) -> str:
    rows = ""
    for n in negations:
        flag = "否定" if n.get("negation_status") else "既往"
        rows += f"<tr><td>{n.get('entity','')}</td><td>{flag}</td>"
        rows += f"<td>{n.get('source_sentence','')}</td></tr>"
    return rows


def _render_lab_table(lab: dict) -> str:
    dates = lab.get("dates", [])
    values = lab.get("values", [])
    if not dates:
        return ""
    hdr = "<tr><th>时间点</th>" + "".join(f"<td>{d}</td>" for d in dates) + "</tr>"
    row = "<tr><th>检测值</th>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>"
    return f"<table>{hdr}{row}</table>"


def _render_shap_rows(shap: list[dict]) -> str:
    rows = ""
    for sf in shap:
        pct = sf.get("importance", 0) * 100
        rows += (
            f"<tr><td>{sf.get('feature','')}</td>"
            f"<td><div style='background:#3498db;height:18px;width:{pct:.0f}%;"
            f"border-radius:3px;display:inline-block;'></div>"
            f" <span>{pct:.1f}%</span></td></tr>"
        )
    return rows


def _build_html(pid, name, age, gender, disease, nlp, lab, mm) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rid = f"RPT-{pid}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    risk = mm.get("risk_level", "MEDIUM")
    rcol = {"HIGH": "#e74c3c", "MEDIUM_HIGH": "#e67e22", "MEDIUM": "#f39c12", "LOW": "#27ae60"}.get(risk, "#333")

    triples_html = _render_triple_rows(nlp.get("triples", []))
    neg_html = _render_neg_rows(nlp.get("negations", []))
    lab_table = _render_lab_table(lab)
    shap_html = _render_shap_rows(mm.get("shap_features", []))

    ec = nlp.get("entity_count", 0)
    tc = nlp.get("triple_count", len(nlp.get("triples", [])))
    nc = nlp.get("negation_count", len(nlp.get("negations", [])))

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>临床诊断决策报告 — {pid}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Microsoft YaHei",sans-serif;background:#f0f2f5;color:#333}}
.c{{max-width:960px;margin:20px auto;background:#fff;border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,.08)}}
.h{{background:linear-gradient(135deg,#2c3e50,#3498db);color:#fff;padding:28px 32px;border-radius:8px 8px 0 0}}
.h h1{{font-size:22px;margin-bottom:6px}}
.h .m{{font-size:13px;opacity:.85}}
.s{{padding:20px 32px;border-bottom:1px solid #eee}}
.s:last-child{{border-bottom:none}}
.s h2{{font-size:17px;color:#2c3e50;margin-bottom:14px;padding-left:10px;border-left:3px solid #3498db}}
table{{width:100%;border-collapse:collapse;margin-top:8px}}
th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid #e8e8e8;font-size:13px}}
th{{background:#f7f8fa;color:#555;font-weight:600}}
.badge{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;color:#fff}}
.g{{display:grid;grid-template-columns:1fr 1fr;gap:10px 24px}}
.gi{{font-size:13px}}
.gi l{{color:#888;margin-right:6px}}
.rb{{padding:16px;border-radius:6px;margin:12px 0;text-align:center;font-size:16px;font-weight:bold;color:#fff}}
.f{{text-align:center;padding:18px;color:#999;font-size:12px}}
.f .st{{color:#3498db;font-weight:bold;font-size:14px}}
</style>
</head>
<body>
<div class="c">
<div class="h"><h1>临床诊断决策报告</h1><div class="m">编号: {rid} | 生成: {now} | D组医学数据中台</div></div>

<div class="s"><h2>一、患者基本信息</h2>
<div class="g">
<div class="gi"><l>患者ID:</l>{pid}</div><div class="gi"><l>姓名:</l>{name}</div>
<div class="gi"><l>年龄:</l>{age} 岁</div><div class="gi"><l>性别:</l>{gender}</div>
<div class="gi"><l>主要诊断:</l><b>{disease}</b></div>
<div class="gi"><l>状态:</l><span class="badge" style="background:#27ae60">已完成</span></div>
</div></div>

<div class="s"><h2>二、NLP 文本结构化分析</h2>
<p style="font-size:13px;color:#666;margin-bottom:10px">实体 {ec} 个 | 三元组 {tc} 组 | 否定/既往 {nc} 条</p>
{('<h3 style="font-size:14px;color:#555;margin-top:12px">关系三元组</h3>'
  '<table><tr><th>主体</th><th>关系</th><th>客体</th><th>置信度</th></tr>'
  + triples_html + '</table>') if triples_html else ''}
{('<h3 style="font-size:14px;color:#555;margin-top:16px">否定与既往识别</h3>'
  '<table><tr><th>实体</th><th>状态</th><th>来源</th></tr>'
  + neg_html + '</table>') if neg_html else ''}
</div>

<div class="s"><h2>三、生化指标时序趋势</h2>
<p style="font-size:13px;color:#666;margin-bottom:10px">
指标: {lab.get('item_name','N/A')} | 参考范围: {lab.get('ref_range','N/A')} | 趋势: {lab.get('trend_direction','N/A')}
</p>
{lab_table}
</div>

<div class="s"><h2>四、多模态 AI 决策</h2>
<p style="font-size:13px;color:#666;margin-bottom:10px">
模型: <b>{mm.get('selected_model','N/A')}</b> | 评分: {mm.get('winning_score','N/A')}/100
</p>
<div class="rb" style="background:{rcol}">
风险等级: {risk} — {mm.get('prediction','待评估')}
</div>
{('<pre style="background:#f7f8fa;padding:12px;border-radius:4px;font-size:12px;'
  'white-space:pre-wrap">' + mm.get('routing_reason','') + '</pre>') if mm.get('routing_reason') else ''}
{('<h3 style="font-size:14px;color:#555;margin-top:16px">SHAP 特征重要性</h3>'
  '<table><tr><th>特征</th><th>贡献度</th></tr>'
  + shap_html + '</table>') if shap_html else ''}
</div>

<div class="s"><h2>五、综合建议</h2>
<p style="font-size:13px;color:#555;line-height:1.8">
1. 结合 NLP 分析与生化趋势，建议重点关注异常指标。<br>
2. 多模态 AI 已给出风险评估，可作为辅助决策参考。<br>
3. 如需二次意见，可提交 MDT 会诊。<br>
4. AI 鉴别理由日志展示了模型决策路径，便于临床追溯。
</p></div>

<div class="f">
<p>由 <span class="st">D组：医学文本与生化数据分析中台</span> 自动生成</p>
<p>仅供临床参考，不作为最终诊断依据 | {now}</p>
</div>
</div>
</body></html>"""


# ===================================================================
# GET /api/reports/download
# ===================================================================


@reports_router.get("/download")
def download_report(
    patient_id: str = Query(default="p001", description="患者 ID"),
    report_type: str = Query(default="html", description="报告格式: html / json"),
):
    """下载或自动生成临床诊断决策报告。

    Args:
        patient_id:  患者 ID。
        report_type: 输出格式 (html / json)。

    Returns:
        HTML / JSON 临床报告。
    """
    rt = report_type.lower().strip()
    if rt not in ("html", "json"):
        raise HTTPException(status_code=400, detail="report_type 仅支持 html 或 json")

    html_path = _EXPORT_DIR / f"{patient_id}_clinical_report.html"
    json_path = _EXPORT_DIR / f"{patient_id}_clinical_report.json"

    # ---- 已有文件直接返回 ----
    if rt == "html" and html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    if rt == "json" and json_path.exists():
        return FileResponse(
            path=str(json_path), media_type="application/json",
            filename=f"{patient_id}_report.json",
        )

    # ---- 生成仿真数据 ----
    seed = abs(hash(patient_id)) % (2 ** 31)
    rng = np.random.RandomState(seed)

    name = f"患者{patient_id[-4:]}"
    age = int(rng.randint(35, 78))
    gender = "男" if rng.random() > 0.4 else "女"
    disease_pool = ["糖尿病肾病", "肺部肿瘤", "肝脏病变", "血常规异常"]
    disease = disease_pool[seed % len(disease_pool)]

    # 推断模型
    if "糖尿" in disease:
        sel_model = "Model_A_TimeSeries"
        pred = "高危并发症：GLU 持续攀升，预测 CKD-3 期风险 78%，建议内分泌科干预。"
        risk = "HIGH"
    elif "肺" in disease:
        sel_model = "Model_C_Multimodal"
        pred = "高度疑似恶性：CT 组学异质性显著 (GLCM熵值 > 3.5)，恶性概率 82%，建议活检。"
        risk = "HIGH"
    else:
        sel_model = "Model_B_NLP_Graph"
        pred = "中危：NLP 图谱提示慢性病变活动期，建议完善检查。"
        risk = "MEDIUM"

    nlp_summary = {
        "entity_count": int(rng.randint(3, 10)),
        "triple_count": int(rng.randint(1, 5)),
        "negation_count": int(rng.randint(0, 3)),
        "triples": [
            {"subject": "结节", "relation": "位于", "object": "右肺上叶", "confidence": 0.92},
            {"subject": "WBC", "relation": "状态", "object": "升高", "confidence": 0.89},
        ],
        "negations": [
            {"entity": "远处转移", "negation_status": True,
             "source_sentence": "未见明显远处转移"},
            {"entity": "高血压", "negation_status": False, "history_status": True,
             "source_sentence": "既往有高血压病史"},
        ],
    }
    labs_summary = {
        "item_name": "GLU",
        "ref_range": "3.9-6.1 mmol/L",
        "trend_direction": "increasing" if rng.random() > 0.4 else "fluctuating",
        "abnormal_indices": [2, 5] if rng.random() > 0.3 else [2],
        "dates": [
            "2026-01-01", "2026-01-08", "2026-01-15", "2026-01-22",
            "2026-01-29", "2026-02-05", "2026-02-12",
        ],
        "values": [round(float(rng.normal(5.5, 1.8)), 1) for _ in range(7)],
    }
    mm_summary = {
        "selected_model": sel_model,
        "winning_score": round(float(rng.uniform(65, 95)), 1),
        "risk_level": risk,
        "prediction": pred,
        "routing_reason": (
            f"【AI 路由决策日志】\n"
            f"数据画像: 时序=6步, NLP=4实体, 组学={'有' if '肺' in disease else '无'}\n"
            f"胜出模型: {sel_model} (得分 {round(float(rng.uniform(75, 90)), 1)}/100)\n"
            f"决策依据: 综合考量模态契合度与历史战绩, 自动热切至此模型。"
        ),
        "shap_features": [
            {"feature": "病灶尺寸", "importance": 0.35},
            {"feature": "CT 组学纹理", "importance": 0.28},
            {"feature": "CEA 趋势", "importance": 0.20},
            {"feature": "NLP 实体密度", "importance": 0.12},
            {"feature": "时序步数", "importance": 0.05},
        ],
    }

    if rt == "html":
        html = _build_html(patient_id, name, age, gender, disease,
                           nlp_summary, labs_summary, mm_summary)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        return HTMLResponse(content=html)

    # JSON
    data = {
        "report_id": f"RPT-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "patient_id": patient_id,
        "patient_name": name,
        "age": age,
        "gender": gender,
        "primary_disease": disease,
        "generated_at": datetime.now().isoformat(),
        "nlp_analysis": nlp_summary,
        "labs_trend": labs_summary,
        "multimodal_decision": mm_summary,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False, indent=2)
    return JSONResponse(content=data)
