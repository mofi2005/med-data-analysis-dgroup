<script setup>
import { computed, ref } from 'vue'
import { trainModel, listModels, getModelImportance } from '@/api'

const models = ref(null)
const trainResult = ref(null)
const importance = ref(null)
const selectedModelId = ref('')
const error = ref('')

const importanceOption = computed(() => {
  if (!importance.value?.feature_importance) return null
  const items = importance.value.feature_importance
  return {
    title: { text: '特征重要性' },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'value' },
    yAxis: { type: 'category', data: items.map((i) => i.feature || i.name) },
    series: [{ type: 'bar', data: items.map((i) => i.importance), itemStyle: { color: '#0b5cab' } }],
  }
})

async function loadList() {
  error.value = ''
  try {
    models.value = await listModels()
    const first = models.value?.models?.[0]
    if (first) selectedModelId.value = first.model_id
  } catch (e) {
    error.value = e.message
  }
}

async function doTrain() {
  error.value = ''
  try {
    trainResult.value = await trainModel({
      task_name: '病灶恶性风险评估',
      task_type: 'classification',
      target_variable: 'is_malignant',
      features: ['CEA', 'WBC', 'ALT', 'GLU'],
      algorithm: 'RandomForest',
    })
    selectedModelId.value = trainResult.value.model_id
    await loadList()
  } catch (e) {
    error.value = e.message
  }
}

async function loadImportance() {
  if (!selectedModelId.value) return
  error.value = ''
  try {
    importance.value = await getModelImportance(selectedModelId.value)
  } catch (e) {
    error.value = e.message
  }
}
</script>

<template>
  <p class="page-desc">POST train · GET list · GET importance</p>
  <div v-if="error" class="alert alert-error">{{ error }}</div>

  <div class="card">
    <button class="btn btn-primary" @click="doTrain">触发训练</button>
    <button class="btn" @click="loadList">刷新模型列表</button>
    <button class="btn btn-primary" @click="loadImportance">查看特征重要性</button>
  </div>

  <div class="grid-2">
    <div class="card" v-if="trainResult">
      <h3>训练结果</h3>
      <pre class="json">{{ JSON.stringify(trainResult, null, 2) }}</pre>
    </div>
    <div class="card" v-if="models">
      <h3>模型列表 ({{ models.total || models.models?.length || 0 }})</h3>
      <div class="form-row">
        <label>model_id</label>
        <select v-model="selectedModelId">
          <option v-for="m in models.models || []" :key="m.model_id" :value="m.model_id">
            {{ m.task_name || m.model_id }}
          </option>
        </select>
      </div>
      <pre class="json">{{ JSON.stringify(models, null, 2) }}</pre>
    </div>
  </div>

  <div class="card" v-if="importanceOption">
    <VChart class="chart-box" :option="importanceOption" autoresize />
  </div>
</template>
