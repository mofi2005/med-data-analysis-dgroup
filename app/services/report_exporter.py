import os
import time

def export_html(html_content, output_path=None, filename=None, **kwargs):
    """应急修补函数：保存 HTML 报告到本地"""
    if not output_path:
        os.makedirs("output_reports", exist_ok=True)
        fname = filename or f"clinical_report_{int(time.time())}.html"
        output_path = os.path.join("output_reports", fname)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(html_content))
    print(f"[EXPORTER SUCCESS] 临床报告已成功导出至: {output_path}")
    return output_path
