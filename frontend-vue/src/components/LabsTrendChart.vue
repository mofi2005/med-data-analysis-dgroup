<script setup>
import { computed, ref } from "vue";
import VChart from "vue-echarts";
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { LineChart, HeatmapChart, BarChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  VisualMapComponent,
  TitleComponent,
} from "echarts/components";

use([
  CanvasRenderer,
  LineChart,
  HeatmapChart,
  BarChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  VisualMapComponent,
  TitleComponent,
]);

const props = defineProps({
  trend: { type: Object, default: null },
});

const option = computed(() => {
  if (!props.trend?.echarts_data) return null;
  const { xAxis, series, abnormal_indices = [] } = props.trend.echarts_data;
  const refLow = props.trend.reference_range?.low;
  const refHigh = props.trend.reference_range?.high;

  const markPoint = abnormal_indices.map((idx) => ({
    coord: [xAxis[idx], series[idx]],
    value: "异常",
    itemStyle: { color: "#dc2626" },
  }));

  return {
    title: {
      text: `${props.trend.chinese_name || props.trend.item_name} 时序趋势`,
      subtext: `参考范围: ${refLow} - ${refHigh} ${props.trend.unit || ""}`,
      left: "center",
    },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: xAxis, boundaryGap: false },
    yAxis: { type: "value", name: props.trend.unit || "" },
    series: [
      {
        name: props.trend.item_name,
        type: "line",
        smooth: true,
        data: series,
        lineStyle: { width: 3, color: "#0b5cab" },
        areaStyle: { color: "rgba(11, 92, 171, 0.08)" },
        markPoint: { data: markPoint, symbolSize: 48 },
        markLine:
          refLow != null && refHigh != null
            ? {
                silent: true,
                data: [
                  { yAxis: refLow, lineStyle: { color: "#16a34a", type: "dashed" } },
                  { yAxis: refHigh, lineStyle: { color: "#dc2626", type: "dashed" } },
                ],
              }
            : undefined,
      },
    ],
  };
});
</script>

<template>
  <VChart v-if="option" class="chart-box" :option="option" autoresize />
  <p v-else class="badge-muted">暂无折线图数据，请点击「加载趋势」</p>
</template>
