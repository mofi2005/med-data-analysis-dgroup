# 组员2 契约流水线（src）

按《D组架构联调与跨设备数据契约说明书》实现：

| 模块 | 说明 |
|------|------|
| `parsers/your_parser.py` | `extract_and_standardize()` → 协议 1.1 `patient_payload` |
| `models/model_router.py` | `AdaptiveModelRouter.route_and_predict()` → 协议 1.2 `ai_decision` |
| `reports/report_generator.py` | `generate_clinical_report()` → HTML 报告 |
| `api/main_pipeline.py` | 本地一键联调入口 |

## 运行（组员1 合并后）

```bash
# 仓库根目录执行，确保 PYTHONPATH 包含项目根
python -m src.api.main_pipeline
```

## 组员1 权重目录

将 `.pkl` 放在仓库根目录 `models/weights/`（与 `src/` 同级），`AdaptiveModelRouter` 会自动加载。

## 合并依赖说明

`src/` 中部分模块会 `import app.*`（解析、报告渲染、单位换算）。  
组员1 在服务器集中集成时，需保证同仓库内存在对应的 `app/` 与 `templates/`（或由组员1 替换为等价实现）。

## 测试样例

`src/fixtures/` 内含联调用 CSV 与 text/radiomics mock JSON。
