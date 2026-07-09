#!/usr/bin/env python3
"""数据库连接配置文件 (SQLAlchemy + SQLite).

本模块负责:
  - 创建 SQLite 数据库文件的存储目录
  - 配置 SQLAlchemy engine 与 session
  - 提供 FastAPI 依赖注入函数 get_db()

数据库文件: data/medical_dgroup.db (项目根目录下)
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# 项目根目录 (src/backend/database.py → 上溯两级到项目根)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# SQLite 数据库文件路径
_DB_DIR = _PROJECT_ROOT / "data"
_DB_FILE = _DB_DIR / "medical_dgroup.db"

# 确保 data 目录存在
os.makedirs(str(_DB_DIR), exist_ok=True)

# SQLite 连接 URL (check_same_thread=False 适配 FastAPI 异步环境)
DATABASE_URL: str = f"sqlite:///{_DB_FILE}"

# 创建 SQLAlchemy 引擎
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite 多线程安全
    echo=False,  # 生产环境关闭 SQL 回显；调试时可设为 True
)

# 创建 Session 工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ORM 基类 —— 所有物理表模型继承自此类
Base = declarative_base()


def get_db() -> Session:
    """FastAPI 依赖注入: 为每个请求创建独立的数据库会话。

    用法 (在 FastAPI 路由函数中):
        @app.get("/api/xxx")
        def read_xxx(db: Session = Depends(get_db)):
            ...

    Yields:
        SQLAlchemy Session 实例; 请求结束后自动关闭。
    """
    db: Session = SessionLocal()  # type: ignore[assignment]
    try:
        yield db
    finally:
        db.close()
