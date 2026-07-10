<script setup>
import { ref } from 'vue'
import { extractNlp } from '@/api'

const patientId = ref('p001')
const text = ref('患者既往有糖尿病史，否认高血压。结节位于右肺上叶，WBC 升高。')
const result = ref(null)
const error = ref('')

async function runExtract() {
  error.value = ''
  try {
    result.value = await extractNlp({ patient_id: patientId.value, text: text.value })
  } catch (e) {
    error.value = e.message
  }
}
</script>

<template>
  <p class="page-desc">POST /api/nlp/extract — 实体、三元组、否定/既往史</p>
  <div v-if="error" class="alert alert-error">{{ error }}</div>

  <div class="card">
    <div class="grid-2">
      <div class="form-row">
        <label>patient_id</label>
        <input v-model="patientId" />
      </div>
    </div>
    <div class="form-row">
      <label>病历文本</label>
      <textarea v-model="text" />
    </div>
    <button class="btn btn-primary" @click="runExtract">抽取结构化 NLP</button>
  </div>

  <div v-if="result" class="grid-3">
    <div class="card">
      <h3>实体 ({{ result.entities?.length || 0 }})</h3>
      <pre class="json">{{ JSON.stringify(result.entities, null, 2) }}</pre>
    </div>
    <div class="card">
      <h3>三元组 ({{ result.triples?.length || 0 }})</h3>
      <pre class="json">{{ JSON.stringify(result.triples, null, 2) }}</pre>
    </div>
    <div class="card">
      <h3>否定/既往 ({{ result.negations?.length || 0 }})</h3>
      <pre class="json">{{ JSON.stringify(result.negations, null, 2) }}</pre>
    </div>
  </div>
</template>
