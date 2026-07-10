<script setup>
import { computed, ref } from 'vue'
import { routeAndPredict, alignFeatures, triggerEvolutionDemo } from '@/api'
import MlopsBadge from '@/components/MlopsBadge.vue'

const patientId = ref('p001')
const glu = ref(9.8)
const cea = ref(3.5)
const prediction = ref(null)
const alignResult = ref(null)
const scoresBefore = ref(null)
const error = ref('')
const loading = ref(false)

const payload = computed(() => ({
  patient_profile: {
    patient_id: patientId.value,
    time_series_steps: 5,
    nlp_entities: 4,
    missing_rate: 0.1,
  },
  clinical_data: { GLU: glu.value, CEA: cea.value },
}))

const scoresOption = computed(() => {
  const current = prediction.value?.mlops_live_scores
  if (!current || !Object.keys(current).length) return null
  const labels = Object.keys(current)
  const series = [{ name: '当前', type: 'bar', data: labels.map((k) => current[k]), itemStyle: { color: '#0b5cab' } }]
  if (scoresBefore.value) {
    series.push({
      name: '进化前',
      type: 'bar',
      data: labels.map((k) => scoresBefore.value[k] ?? 0),
      itemStyle: { color: '#90a4ae' },
    })
  }
  return {
    title: { text: 'MLOps 模型战绩 P_base' },
    tooltip: { trigger: 'axis' },
    legend: { top: 28 },
    xAxis: { type: 'category', data: labels },
    yAxis: { type: 'value', max: 100 },
    series,
  }
})

async function predict() {
  error.value = ''
  loading.value = true
  try {
    prediction.value = await routeAndPredict(payload.value)
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

async function evolve() {
  error.value = ''
  loading.value = true
  try {
    if (prediction.value?.mlops_live_scores) {
      scoresBefore.value = { ...prediction.value.mlops_live_scores }
    }
    const evo = await triggerEvolutionDemo()
    alert(evo.message || '进化完成')
    prediction.value = await routeAndPredict(payload.value)
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

async function align() {
  error.value = ''
  try {
    alignResult.value = await alignFeatures({
      patient_id: patientId.value,
      case_id: `${patientId.value}_c1`,
      group_b_quality: { artifact_level: 'low', quality_score: 0.95 },
      group_e_radiomics: { texture_mean: 15.3, glcm_entropy: 3.45 },
      d_group_stats: { GLU_slope: 0.64, Creatinine_mean: 88 },
      d_group_nlp: { entity_count: 4, negation_count: 1 },
    })
  } catch (e) {
    error.value = e.message
  }
}
</script>

<template>
  <p class="page-desc">POST route_and_predict · align_features · trigger_evolution_demo</p>
  <div v-if="error" class="alert alert-error">{{ error }}</div>

  <div class="card">
    <div class="grid-3">
      <div class="form-row"><label>patient_id</label><input v-model="patientId" /></div>
      <div class="form-row"><label>GLU</label><input v-model.number="glu" type="number" step="0.1" /></div>
      <div class="form-row"><label>CEA</label><input v-model.number="cea" type="number" step="0.1" /></div>
    </div>
    <button class="btn btn-primary" :disabled="loading" @click="predict">🔮 路由预测</button>
    <button class="btn btn-warning" :disabled="loading" @click="evolve">🚀 触发 AI 进化演示</button>
    <button class="btn" @click="align">多模态对齐</button>
  </div>

  <div v-if="prediction" class="card">
    <MlopsBadge :enhanced="!!prediction.mlops_enhanced" />
    <div class="metrics">
      <div class="metric"><div class="label">选中模型</div><div class="value" style="font-size:16px;">{{ prediction.selected_model }}</div></div>
      <div class="metric"><div class="label">路由得分</div><div class="value">{{ prediction.winning_score }}</div></div>
      <div class="metric"><div class="label">引擎</div><div class="value" style="font-size:14px;">{{ prediction.engine_mode }}</div></div>
    </div>
    <p><strong>路由理由</strong></p>
    <pre class="json" style="white-space:pre-wrap;">{{ prediction.routing_reason }}</pre>
    <VChart v-if="scoresOption" class="chart-box" :option="scoresOption" autoresize />
    <pre class="json">{{ JSON.stringify(prediction, null, 2) }}</pre>
  </div>

  <div class="card" v-if="alignResult">
    <h3>多模态对齐结果</h3>
    <pre class="json">{{ JSON.stringify(alignResult, null, 2) }}</pre>
  </div>
</template>
