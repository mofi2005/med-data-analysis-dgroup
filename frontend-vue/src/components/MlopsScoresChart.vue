<script setup>
import { computed } from "vue";
import VChart from "vue-echarts";
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { BarChart } from "echarts/charts";
import { GridComponent, TooltipComponent, LegendComponent, TitleComponent } from "echarts/components";

use([CanvasRenderer, BarChart, GridComponent, TooltipComponent, LegendComponent, TitleComponent]);

const props = defineProps({
  scores: { type: Object, default: () => ({}) },
  previousScores: { type: Object, default: null },
});

const option = computed(() => {
  const keys = Object.keys(props.scores || {});
  if (!keys.length) return null;

  const series = [
    {
      name: "当前战绩",
      type: "bar",
      data: keys.map((k) => props.scores[k]),
      itemStyle: { color: "#0b5cab" },
    },
  ];

  if (props.previousScores) {
    series.push({
      name: "进化前",
      type: "bar",
      data: keys.map((k) => props.previousScores[k] ?? 0),
      itemStyle: { color: "#94a3b8" },
    });
  }

  return {
    title: { text: "MLOps 模型战绩 (P_base)", left: "center" },
    tooltip: { trigger: "axis" },
    legend: { top: 28 },
    xAxis: { type: "category", data: keys, axisLabel: { rotate: 15, fontSize: 11 } },
    yAxis: { type: "value", name: "Score" },
    series,
  };
});
</script>

<template>
  <VChart v-if="option" class="chart-box" :option="option" autoresize />
</template>
