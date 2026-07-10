# D组医学数据中台 · Vue 3 前端

按《前后端交接文档》对接 **16 个 RESTful API**，使用 **Vue 3 + Vite + ECharts**。

## 功能模块

| 导航 | API | 图表 |
|------|-----|------|
| 数据与字典 | upload / dictionary ×3 | — |
| NLP | extract | — |
| 生化时序 | GET /api/labs/trend | **折线图**（异常点标红） |
| 统计 | describe / difference / correlation | **热力图** |
| 模型 | train / list / importance | — |
| 多模态 & MLOps | route_and_predict / align / trigger_evolution_demo | **战绩柱状图** + 勋章 |
| 报告 | GET /api/reports/download | **iframe 嵌入** |

## 启动

**终端 1 — 后端**

```powershell
cd D:\medical-data-analysis-d
.\.venv\Scripts\activate
python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
```

**终端 2 — Vue 前端**

```powershell
cd D:\medical-data-analysis-d\frontend-vue
npm install
npm run dev
```

浏览器打开：**http://localhost:5173**

> Vite 已将 `/api` 代理到 `http://127.0.0.1:8000`，无需额外配置 CORS。

## 答辩演示顺序

1. **多模态 & MLOps** → 路由预测 → 点亮 MLOps 勋章 → 触发进化 → 战绩图变化  
2. **生化时序** → GLU 折线图（异常点）  
3. **统计** → 相关性热力图  
4. **报告** → iframe 展示 HTML 临床报告  

## 生产构建

```powershell
npm run build
npm run preview
```
