<script setup>
import { computed, ref } from 'vue'
import { describeStats, differenceTest, correlationMatrix } from '@/api'

const variables = ref('WBC,HGB,PLT,ALT,CEA,GLU')
const describe = ref(null)
const diff = ref(null)
const corr = ref(null)
const error = ref('')

const heatmapOption = computed(() => {
  if (!corr.value) return null
  const vars = corr.value.variables || []
  const data = corr.value.heatmap_series_data || []
  const vals = data.map((d) => d[2])
  const min = Math.min(...vals, -1)
  const max = Math.max(...vals, 1)
  return {
    title: { text: `相关性热力图 (${corr.value.method})` },
    tooltip: { position: 'top' },
    grid: { top: 60, bottom: 80 },
    xAxis: { type: 'category', data: vars, splitArea: { show: true } },
    yAxis: { type: 'category', data: vars, splitArea: { show: true } },
    visualMap: {
      min,
      max,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 10,
      inRange: { color: ['#313695', '#ffffbf', '#a50026'] },
    },
    series: [{
      type: 'heatmap',
      data,
      label: { show: true, formatter: (p) => p.data[2].toFixed(2) },
      emphasis: { itemStyle: { shadowBlur: 10 } },
    }],
  }
})

async function loadDescribe() {
  error.value = ''
  try {
    describe.value = await describeStats()
  } catch (e) {
    error.value = e.message
  }
}

async function loadDiff() {
  error.value = ''
  try {
    diff.value = await differenceTest({
      group_variable: 'label',
      target_variable: 'CEA',
      test_method: 't_test',
    })
  } catch (e) {
    error.value = e.message
  }
}

async function loadCorr() {
  error.value = ''
  try {
    const vars = variables.value.split(',').map((s) => s.trim()).filter(Boolean)
    corr.value = await correlationMatrix({ variables: vars, method: 'pearson' })
  } catch (e) {
    error.value = e.message
  }
}
</script>

<template>
  <p class="page-desc">GET describe · POST difference · POST correlation 热力图</p>
  <div v-if="error" class="alert alert-error">{{ error }}</div>

  <div class="card">
    <button class="btn btn-primary" @click="loadDescribe">描述统计</button>
    <button class="btn" @click="loadDiff">组间差异 T 检验</button>
    <button class="btn btn-primary" @click="loadCorr">生成热力图</button>
    <div class="form-row" style="margin-top:12px;">
      <label>热力图变量（逗号分隔）</label>
      <input v-model="variables" />
    </div>
  </div>

  <div class="grid-2" v-if="describe || diff">
    <div class="card" v-if="describe">
      <h3>描述统计 (n={{ describe.sample_count }})</h3>
      <pre class="json">{{ JSON.stringify(describe.variables, null, 2) }}</pre>
    </div>
    <div class="card" v-if="diff">
      <h3>组间差异 · {{ diff.target_variable }}</h3>
      <div class="metrics">
        <div class="metric"><div class="label">p-value</div><div class="value">{{ diff.p_value }}</div></div>
        <div class="metric"><div class="label">显著</div><div class="value">{{ diff.is_significant ? '是' : '否' }}</div></div>
      </div>
      <pre class="json">{{ JSON.stringify(diff, null, 2) }}</pre>
    </div>
  </div>

  <div class="card" v-if="heatmapOption">
    <VChart class="chart-box" :option="heatmapOption" autoresize />
  </div>
</template>
