<script setup>
import { computed } from "vue";
import VChart from "vue-echarts";
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { HeatmapChart } from "echarts/charts";
import { GridComponent, TooltipComponent, VisualMapComponent, TitleComponent } from "echarts/components";

use([CanvasRenderer, HeatmapChart, GridComponent, TooltipComponent, VisualMapComponent, TitleComponent]);

const props = defineProps({
  result: { type: Object, default: null },
});

const option = computed(() => {
  if (!props.result?.variables?.length) return null;
  const vars = props.result.variables;
  const data = props.result.heatmap_series_data || [];

  return {
    title: { text: `相关性热力图 (${props.result.method || "pearson"})`, left: "center" },
    tooltip: {
      position: "top",
      formatter: (p) => `${vars[p.data[1]]} × ${vars[p.data[0]]}: ${p.data[2]}`,
    },
    grid: { height: "62%", top: "12%" },
    xAxis: { type: "category", data: vars, splitArea: { show: true } },
    yAxis: { type: "category", data: vars, splitArea: { show: true } },
    visualMap: {
      min: -1,
      max: 1,
      calculable: true,
      orient: "horizontal",
      left: "center",
      bottom: "2%",
      inRange: { color: ["#313695", "#4575b4", "#ffffbf", "#f46d43", "#a50026"] },
    },
    series: [
      {
        name: "Correlation",
        type: "heatmap",
        data,
        label: { show: true, formatter: (p) => p.data[2].toFixed(2), fontSize: 10 },
        emphasis: { itemStyle: { shadowBlur: 8, shadowColor: "rgba(0,0,0,0.4)" } },
      },
    ],
  };
});
</script>

<template>
  <VChart v-if="option" class="chart-box" :option="option" autoresize />
  <p v-else class="badge-muted">暂无热力图数据，请点击「生成热力图」</p>
</template>
