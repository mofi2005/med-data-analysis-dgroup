<script setup>
import { computed, ref } from 'vue'
import { getLabTrend } from '@/api'

const patientId = ref('p001')
const itemName = ref('GLU')
const trend = ref(null)
const error = ref('')

const chartOption = computed(() => {
  if (!trend.value?.echarts_data) return null
  const d = trend.value.echarts_data
  const abnormal = new Set(d.abnormal_indices || [])
  const data = (d.series || []).map((v, i) => ({
    value: v,
    itemStyle: abnormal.has(i) ? { color: '#e74c3c' } : { color: '#0b5cab' },
  }))
  const ref = trend.value.reference_range || {}
  return {
    title: {
      text: `${trend.value.chinese_name || itemName.value} 随访趋势`,
      subtext: `参考范围 ${ref.low}-${ref.high} ${trend.value.unit || ''} · 趋势 ${d.trend_direction}`,
    },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: d.xAxis || [] },
    yAxis: { type: 'value', name: trend.value.unit || '' },
    series: [{ type: 'line', smooth: true, data, markLine: ref.low != null ? {
      silent: true,
      lineStyle: { type: 'dashed', color: '#94a3b8' },
      data: [{ yAxis: ref.low }, { yAxis: ref.high }],
    } : undefined }],
  }
})

async function loadTrend() {
  error.value = ''
  try {
    trend.value = await getLabTrend(patientId.value, itemName.value)
  } catch (e) {
    error.value = e.message
  }
}
</script>

<template>
  <p class="page-desc">GET /api/labs/trend — ECharts 折线图（异常点标红）</p>
  <div v-if="error" class="alert alert-error">{{ error }}</div>

  <div class="card">
    <div class="grid-3">
      <div class="form-row">
        <label>patient_id</label>
        <input v-model="patientId" />
      </div>
      <div class="form-row">
        <label>item_name</label>
        <select v-model="itemName">
          <option>GLU</option>
          <option>WBC</option>
          <option>HbA1c</option>
          <option>Creatinine</option>
          <option>CEA</option>
          <option>ALT</option>
        </select>
      </div>
      <div class="form-row" style="justify-content:flex-end;">
        <label>&nbsp;</label>
        <button class="btn btn-primary" @click="loadTrend">加载折线图</button>
      </div>
    </div>
  </div>

  <div class="card" v-if="chartOption">
    <VChart class="chart-box" :option="chartOption" autoresize />
  </div>
</template>
