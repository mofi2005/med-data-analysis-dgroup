#!/usr/bin/env python3
"""D组医学数据中台 FastAPI 核心入口。

本模块负责:
  - 应用启动时自动创建 SQLite 物理表 (Base.metadata.create_all)
  - 配置跨域 CORS 中间件 (供 Vue 前端对接)
  - 提供基础健康检查接口 GET /api/health

启动命令:
    uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

# 必须在导入 ORM 模型之前导入 engine + Base, 以确保 create_all 能发现所有表
from src.backend.database import Base, SessionLocal, engine
from src.backend.models_db import (  # noqa: F401  # 保证模型类在 create_all 之前被注册
    AnalysisTask,
    DataDictionary,
    Dataset,
    MLOpsConfig,
    ModelResult,
    seed_mlops_config,
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.api_data import data_router
from src.backend.api_labs import labs_router
from src.backend.api_models import models_router
from src.backend.api_multimodal import mlops_demo_router, multimodal_router
from src.backend.api_nlp import nlp_router
from src.backend.api_reports import reports_router
from src.backend.api_stats import stats_router

# ---------------------------------------------------------------------------
# 应用启动: 自动创建所有物理表
# ---------------------------------------------------------------------------

Base.metadata.create_all(bind=engine)

# ---- 幂等初始化 MLOps 自进化配置默认参数 ----
with SessionLocal() as session:
    seed_mlops_config(session)

# ---------------------------------------------------------------------------
# FastAPI 应用实例
# ---------------------------------------------------------------------------

app = FastAPI(
    title="D组：医学文本与生化数据分析中台 API",
    description=(
        "D 组项目后端数据中台，提供数据集管理、医学标准字典查询、"
        "NLP 实体抽取、统计分析、模型路由与多模态融合等核心 RESTful API。"
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS 跨域配置 (允许组员 2 的 Vue 前端无条件访问)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # 生产环境应替换为前端实际域名
    allow_credentials=True,
    allow_methods=["*"],         # GET, POST, PUT, DELETE, OPTIONS 等全部放行
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 注册路由
# ---------------------------------------------------------------------------

app.include_router(data_router)
app.include_router(nlp_router)
app.include_router(labs_router)
app.include_router(stats_router)
app.include_router(models_router)
app.include_router(multimodal_router)
app.include_router(mlops_demo_router)
app.include_router(reports_router)

# ---------------------------------------------------------------------------
# 基础健康检查接口
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health_check():
    """基础健康检查: 验证后端服务是否正常运行且物理表已就绪。

    Returns:
        JSON 状态对象。
    """
    return {
        "status": "ok",
        "message": "D组后端数据中台已就绪！物理表已成功加载！",
        "version": "1.0.0",
    }


# ---------------------------------------------------------------------------
# 直接运行入口 (开发调试)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
