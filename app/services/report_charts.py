def build_shap_bar_svg(chart: dict, width: int = 520, height: int = 240) -> str:
    labels = chart.get("labels") or []
    values = chart.get("values") or []
    if not labels or not values:
        return ""

    max_val = max(values) or 1.0
    bar_height = 22
    gap = 10
    left_pad = 120
    top_pad = 30

    svg_height = top_pad + len(labels) * (bar_height + gap) + 20
    bars = []
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = top_pad + idx * (bar_height + gap)
        bar_width = int((value / max_val) * (width - left_pad - 40))
        bars.append(
            f'<rect x="{left_pad}" y="{y}" width="{bar_width}" height="{bar_height}" fill="#0b5cab" rx="4"/>'
        )
        bars.append(f'<text x="10" y="{y + 16}" font-size="13" fill="#333">{label}</text>')
        bars.append(f'<text x="{left_pad + bar_width + 8}" y="{y + 16}" font-size="12" fill="#666">{value}</text>')

    title = chart.get("title", "SHAP Feature Importance")
    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{svg_height}" viewBox="0 0 {width} {svg_height}">
  <text x="10" y="20" font-size="15" font-weight="bold" fill="#0b5cab">{title}</text>
  {''.join(bars)}
</svg>
""".strip()
